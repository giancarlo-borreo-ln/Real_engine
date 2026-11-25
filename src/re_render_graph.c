/**
 * @file re_render_graph.c
 * @brief Declarative render graph: barrier synthesis, transient aliasing
 *        and read-before-write validation.
 *
 * Contract (the one the Frama-C/CBMC harnesses prove):
 *   - compiles only when every read/write index is inside the resource
 *     array and no pass reads a resource whose contents are undefined;
 *   - emits, before each access, exactly one barrier when the tracked
 *     layout differs from the layout the access requires;
 *   - emits, after the last pass, one barrier per resource whose tracked
 *     layout differs from its final layout;
 *   - aliases transients with disjoint lifetimes onto earlier storage.
 *
 * Layout rule (no per-access layout fields exist in ReGraphPass, so the
 * required layout is derived from the access kind and the resource):
 *   - READ  -> DEPTH_READ_ONLY when final_layout is DEPTH_READ_ONLY,
 *             SHADER_READ otherwise;
 *   - WRITE -> DEPTH_ATTACHMENT when final_layout is DEPTH_ATTACHMENT or
 *             DEPTH_READ_ONLY, COLOR_ATTACHMENT otherwise.
 *
 * Access order inside one pass is reads first, then writes. That matches
 * the header rule "a read is legal when written by some *earlier* pass":
 * a pass that both reads and writes a resource is still rejected when
 * the resource had no defined contents before the pass.
 *
 * barrier_start layout: barrier_start[k] is the number of barriers
 * emitted before pass k (k in [0, pass_count)), and barrier_start
 * [pass_count] is barrier_count after the end-of-frame segment. Hence
 * the barriers that precede pass k are [start[k], start[k + 1]) for
 * every k < pass_count - 1. The slice [start[pass_count - 1],
 * start[pass_count]) holds the last pass's barriers followed by the
 * end-of-frame transitions; the two groups cannot be told apart from
 * the array alone (see the header's wording, which claims the final
 * segment holds only end-of-frame transitions).
 *
 * Heap use: only the per-resource scratch arrays below are allocated
 * with malloc; every output array lives in the caller arena.
 */
#include "re_render_graph.h"

#include "re_mem.h"

#include <stdlib.h>

/** Sentinel pass index meaning "no pass accessed this resource". */
#define RE_GRAPH_NO_PASS UINT32_MAX

/**
 * @brief Per-resource scratch state used while compiling.
 *
 * layout[r] is the tracked layout, initialized[r] the contents validity,
 * [first_write, last_read] the lifetime interval used by the aliasing
 * pass, and order[] the transients sorted by first_write_pass.
 */
typedef struct GraphState {
    ReGraphLayout *layout;   /**< [resource_count] tracked layouts. */
    bool *initialized;       /**< [resource_count] contents validity. */
    uint32_t *first_write;   /**< [resource_count] first write pass. */
    uint32_t *last_read;     /**< [resource_count] last read pass. */
    uint32_t *order;         /**< [resource_count] transient sort order. */
} GraphState;

/**
 * @brief Overflow-checked size_t multiplication.
 * @return False when a * b does not fit in size_t.
 */
static bool checked_mul(size_t a, size_t b, size_t *out)
{
    if (a != 0u && b > SIZE_MAX / a) {
        return false;
    }
    *out = a * b;
    return true;
}

/**
 * @brief Overflow-checked size_t addition.
 * @return False when a + b does not fit in size_t.
 */
static bool checked_add(size_t a, size_t b, size_t *out)
{
    if (b > SIZE_MAX - a) {
        return false;
    }
    *out = a + b;
    return true;
}

/**
 * @brief Layout a read of resource must observe.
 * @param resource Resource being read.
 * @return DEPTH_READ_ONLY for depth resources, SHADER_READ otherwise.
 */
static ReGraphLayout required_layout_for(const ReGraphResource *resource, bool write)
{
    if (write) {
        return (resource->final_layout == RE_LAYOUT_DEPTH_ATTACHMENT ||
                resource->final_layout == RE_LAYOUT_DEPTH_READ_ONLY)
                   ? RE_LAYOUT_DEPTH_ATTACHMENT
                   : RE_LAYOUT_COLOR_ATTACHMENT;
    }
    return resource->final_layout == RE_LAYOUT_DEPTH_READ_ONLY ? RE_LAYOUT_DEPTH_READ_ONLY
                                                               : RE_LAYOUT_SHADER_READ;
}

/**
 * @brief Appends one layout transition; a no-op when src == dst.
 *
 * The caller guarantees that the barriers array has room: its capacity is
 * the worst case computed by barrier_capacity.
 */
static void emit_barrier(ReGraphCompiled *out, uint32_t res, ReGraphLayout src, ReGraphLayout dst)
{
    if (src == dst) {
        return;
    }
    out->barriers[out->barrier_count].res = res;
    out->barriers[out->barrier_count].src = src;
    out->barriers[out->barrier_count].dst = dst;
    out->barrier_count++;
}

/**
 * @brief Worst-case barrier count: one per access plus one per resource.
 * @return False when the total does not fit in size_t.
 */
static bool barrier_capacity(const ReGraphPass *passes, uint32_t pass_count,
                             uint32_t resource_count, size_t *out)
{
    size_t total = resource_count;

    for (uint32_t k = 0; k < pass_count; k++) {
        if (!checked_add(total, passes[k].read_count, &total) ||
            !checked_add(total, passes[k].write_count, &total)) {
            return false;
        }
    }
    *out = total;
    return true;
}

/**
 * @brief Rejects edges that point outside the resource array.
 * @return True when every read/write index is in [0, resource_count).
 */
static bool check_edges(const ReGraphPass *passes, uint32_t pass_count, uint32_t resource_count)
{
    for (uint32_t k = 0; k < pass_count; k++) {
        const ReGraphPass *pass = &passes[k];

        if ((pass->read_count != 0u && pass->reads == NULL) ||
            (pass->write_count != 0u && pass->writes == NULL)) {
            return false;
        }
        for (uint32_t i = 0; i < pass->read_count; i++) {
            if (pass->reads[i] >= resource_count) {
                return false;
            }
        }
        for (uint32_t i = 0; i < pass->write_count; i++) {
            if (pass->writes[i] >= resource_count) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Allocates the four output arrays from the arena.
 * @return False on arena exhaustion or size overflow.
 */
static bool alloc_outputs(ReGraphCompiled *out, ReArena *arena, uint32_t resource_count,
                          uint32_t pass_count, size_t barrier_cap)
{
    size_t pass_order_bytes;
    size_t barrier_start_bytes;
    size_t alias_bytes;
    size_t barrier_bytes;
    size_t start_count;

    if (!checked_add(pass_count, 1u, &start_count) ||
        !checked_mul(pass_count, sizeof(uint32_t), &pass_order_bytes) ||
        !checked_mul(start_count, sizeof(uint32_t), &barrier_start_bytes) ||
        !checked_mul(resource_count, sizeof(uint32_t), &alias_bytes) ||
        !checked_mul(barrier_cap, sizeof(ReGraphBarrier), &barrier_bytes)) {
        return false;
    }
    out->pass_order = re_arena_alloc(arena, pass_order_bytes);
    out->barrier_start = re_arena_alloc(arena, barrier_start_bytes);
    out->alias_of = re_arena_alloc(arena, alias_bytes);
    out->barriers = re_arena_alloc(arena, barrier_bytes);
    return out->pass_order != NULL && out->barrier_start != NULL &&
           out->alias_of != NULL && out->barriers != NULL;
}

/**
 * @brief Allocates and initializes the heap-side scratch arrays.
 * @return False on OOM (nothing is leaked).
 */
static bool state_init(GraphState *state, const ReGraphResource *resources, uint32_t resource_count)
{
    size_t n = resource_count;

    state->layout = malloc(n * sizeof *state->layout);
    state->initialized = malloc(n * sizeof *state->initialized);
    state->first_write = malloc(n * sizeof *state->first_write);
    state->last_read = malloc(n * sizeof *state->last_read);
    state->order = malloc(n * sizeof *state->order);
    if (state->layout == NULL || state->initialized == NULL || state->first_write == NULL ||
        state->last_read == NULL || state->order == NULL) {
        free(state->layout);
        free(state->initialized);
        free(state->first_write);
        free(state->last_read);
        free(state->order);
        state->layout = NULL;
        state->initialized = NULL;
        state->first_write = NULL;
        state->last_read = NULL;
        state->order = NULL;
        return false;
    }
    for (uint32_t r = 0; r < resource_count; r++) {
        state->layout[r] = resources[r].initial_layout;
        state->initialized[r] = resources[r].initial_layout != RE_LAYOUT_UNDEFINED;
        state->first_write[r] = RE_GRAPH_NO_PASS;
        state->last_read[r] = RE_GRAPH_NO_PASS;
    }
    return true;
}

/**
 * @brief Releases the scratch arrays (all pointers become NULL).
 */
static void state_free(GraphState *state)
{
    free(state->layout);
    free(state->initialized);
    free(state->first_write);
    free(state->last_read);
    free(state->order);
    state->layout = NULL;
    state->initialized = NULL;
    state->first_write = NULL;
    state->last_read = NULL;
    state->order = NULL;
}

/**
 * @brief Processes one pass: read-before-write checks, then barriers.
 * @return False on a read-before-write violation.
 */
static bool process_pass(const ReGraphPass *pass, const ReGraphResource *resources,
                         GraphState *state, ReGraphCompiled *out)
{
    for (uint32_t i = 0; i < pass->read_count; i++) {
        uint32_t r = pass->reads[i];
        ReGraphLayout want = required_layout_for(&resources[r], false);

        if (!state->initialized[r]) {
            return false; /* Read of undefined contents. */
        }
        emit_barrier(out, r, state->layout[r], want);
        state->layout[r] = want;
    }
    for (uint32_t i = 0; i < pass->write_count; i++) {
        uint32_t r = pass->writes[i];
        ReGraphLayout want = required_layout_for(&resources[r], true);

        emit_barrier(out, r, state->layout[r], want);
        state->layout[r] = want;
        state->initialized[r] = true;
    }
    return true;
}

/**
 * @brief Walks the passes in submission order, recording barrier starts.
 * @return False on a read-before-write violation.
 */
static bool walk_passes(const ReGraphPass *passes, uint32_t pass_count,
                        const ReGraphResource *resources, GraphState *state, ReGraphCompiled *out)
{
    for (uint32_t k = 0; k < pass_count; k++) {
        out->pass_order[k] = k;
        out->barrier_start[k] = out->barrier_count;
        if (!process_pass(&passes[k], resources, state, out)) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Records each resource's [first write, last read] pass interval.
 */
static void record_intervals(const ReGraphPass *passes, uint32_t pass_count, GraphState *state)
{
    for (uint32_t k = 0; k < pass_count; k++) {
        const ReGraphPass *pass = &passes[k];

        for (uint32_t i = 0; i < pass->write_count; i++) {
            uint32_t r = pass->writes[i];

            if (state->first_write[r] == RE_GRAPH_NO_PASS) {
                state->first_write[r] = k;
            }
        }
        for (uint32_t i = 0; i < pass->read_count; i++) {
            state->last_read[pass->reads[i]] = k;
        }
    }
}

/**
 * @brief Fills order[] with written transients sorted by first_write_pass.
 *
 * Stable insertion sort: ties keep ascending resource index because the
 * shift condition is strict. O(resource_count^2) worst case, which is
 * irrelevant at graph sizes; returns the number of entries in *count.
 */
static void build_alias_order(const ReGraphResource *resources, uint32_t resource_count,
                              const GraphState *state, uint32_t *order, uint32_t *count)
{
    uint32_t used = 0;

    for (uint32_t r = 0; r < resource_count; r++) {
        if (!resources[r].transient || state->first_write[r] == RE_GRAPH_NO_PASS) {
            continue; /* Non-transients and never-written transients: own storage. */
        }
        uint32_t j = used;

        while (j > 0 && state->first_write[order[j - 1u]] > state->first_write[r]) {
            order[j] = order[j - 1u];
            j--;
        }
        order[j] = r;
        used++;
    }
    *count = used;
}

/**
 * @brief First-fit owner whose storage lifetime ends before res starts.
 * @return Storage owner index, or res itself when nothing fits.
 */
static uint32_t pick_alias_slot(const GraphState *state, const uint32_t *alias_of,
                                const uint32_t *order, uint32_t placed, uint32_t res)
{
    for (uint32_t i = 0; i < placed; i++) {
        uint32_t cand = order[i];

        if (alias_of[cand] == cand && state->last_read[cand] < state->first_write[res]) {
            return cand;
        }
    }
    return res;
}

/**
 * @brief Greedy transient aliasing over the sorted lifetime intervals.
 *
 * alias_of[r] == r means "keeps its own storage". When r aliases a slot
 * owner, that owner's lifetime is extended to r's last read, so a later
 * transient can never be placed on a slot that is still live.
 */
static void assign_aliases(const ReGraphResource *resources, uint32_t resource_count,
                           GraphState *state, ReGraphCompiled *out)
{
    uint32_t count = 0;

    for (uint32_t r = 0; r < resource_count; r++) {
        out->alias_of[r] = r;
    }
    build_alias_order(resources, resource_count, state, state->order, &count);
    for (uint32_t i = 0; i < count; i++) {
        uint32_t r = state->order[i];
        uint32_t slot = pick_alias_slot(state, out->alias_of, state->order, i, r);

        out->alias_of[r] = slot;
        if (slot != r && state->last_read[slot] < state->last_read[r]) {
            state->last_read[slot] = state->last_read[r];
        }
    }
}

/**
 * @brief Emits the end-of-frame transitions to each final layout.
 *
 * A final layout of UNDEFINED leaves the resource where it is: the graph
 * makes no promise about its contents.
 */
static void emit_final_barriers(const ReGraphResource *resources, uint32_t resource_count,
                                GraphState *state, ReGraphCompiled *out)
{
    for (uint32_t r = 0; r < resource_count; r++) {
        ReGraphLayout want = resources[r].final_layout;

        if (want == RE_LAYOUT_UNDEFINED) {
            continue;
        }
        emit_barrier(out, r, state->layout[r], want);
        state->layout[r] = want;
    }
}

bool re_graph_compile(const ReGraphResource *resources, uint32_t resource_count,
                      const ReGraphPass *passes, uint32_t pass_count,
                      ReGraphCompiled *out, void *mem, size_t mem_size)
{
    ReArena arena;
    GraphState state;
    size_t barrier_cap;

    if (resources == NULL || passes == NULL || out == NULL) {
        return false;
    }
    if (resource_count == 0u || pass_count == 0u) {
        return false;
    }
    if (!check_edges(passes, pass_count, resource_count) ||
        !barrier_capacity(passes, pass_count, resource_count, &barrier_cap)) {
        return false;
    }
    if (!re_arena_init(&arena, mem, mem_size)) {
        return false;
    }
    out->barrier_count = 0u;
    out->pass_count = pass_count;
    if (!alloc_outputs(out, &arena, resource_count, pass_count, barrier_cap)) {
        return false;
    }
    if (!state_init(&state, resources, resource_count)) {
        return false;
    }
    bool ok = walk_passes(passes, pass_count, resources, &state, out);

    if (ok) {
        record_intervals(passes, pass_count, &state);
        assign_aliases(resources, resource_count, &state, out);
        emit_final_barriers(resources, resource_count, &state, out);
        out->barrier_start[pass_count] = out->barrier_count;
    }
    state_free(&state);
    return ok;
}

bool re_graph_validate(const ReGraphResource *resources, uint32_t resource_count,
                       const ReGraphCompiled *c)
{
    if (resources == NULL || c == NULL || resource_count == 0u) {
        return false;
    }
    if (c->pass_order == NULL || c->barriers == NULL || c->barrier_start == NULL ||
        c->alias_of == NULL || c->pass_count == 0u) {
        return false;
    }
    if (c->barrier_start[0] != 0u || c->barrier_start[c->pass_count] != c->barrier_count) {
        return false;
    }
    for (uint32_t k = 0; k < c->pass_count; k++) {
        if (c->barrier_start[k] > c->barrier_start[k + 1u] ||
            c->barrier_start[k + 1u] > c->barrier_count ||
            c->pass_order[k] >= c->pass_count) {
            return false;
        }
    }
    for (uint32_t i = 0; i < c->barrier_count; i++) {
        uint32_t src = (uint32_t)c->barriers[i].src;
        uint32_t dst = (uint32_t)c->barriers[i].dst;

        if (c->barriers[i].res >= resource_count || src >= (uint32_t)RE_LAYOUT_COUNT ||
            dst >= (uint32_t)RE_LAYOUT_COUNT || src == dst) {
            return false;
        }
    }
    for (uint32_t r = 0; r < resource_count; r++) {
        uint32_t alias = c->alias_of[r];

        if (alias >= resource_count) {
            return false;
        }
        if (alias != r && !(resources[r].transient && alias < r)) {
            return false;
        }
    }
    return true;
}
