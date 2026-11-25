/**
 * @file re_ecs.c
 * @brief Implementation of the generational sparse-set archetype ECS.
 *
 * Layout invariants:
 *  - An entity slot is live when entity_gen[slot] equals the handle
 *    generation (and that generation is non-zero). Every despawn and every
 *    respawn bumps the stored generation, so a handle issued for an older
 *    life of the slot can never match again (ABA protection).
 *  - entity_arch/chunk/row are valid only for live slots.
 *  - An archetype owns a stack of chunks; only the last chunk can be
 *    partially filled. Chunks are never released individually: the arena
 *    only reclaims memory through re_ecs_clear.
 *  - Nothing is ever freed. Every table growth allocates a fresh block,
 *    copies the old contents and abandons the old block in the arena.
 *    That is the documented trade-off of the arena-backed design.
 *  - This file is single-threaded, like the arena it allocates from.
 */
#include "re_ecs.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/** Initial slot count of every growable table. */
#define RE_ECS_INITIAL_CAP 8u

#ifdef __CPROVER__
/** Fixed table ceiling for the CBMC harnesses: tables never grow. */
#define RE_ECS_CAP_MAX 8u
#endif

/** "No archetype" sentinel used by the internal helpers. */
#define RE_ARCH_NONE UINT32_MAX

/** Column index meaning "component absent from this archetype". */
#define RE_COLUMN_ABSENT 0xFFu

/**
 * @brief Header stored in front of every SoA column allocation.
 *
 * The arena hands out raw bytes, so each column block remembers its own
 * element stride and capacity. data sits at offset 16, which keeps the
 * column 16-byte aligned when the arena block is (it always is).
 */
typedef struct ColumnBlock {
    size_t stride;   /**< Element stride in bytes. */
    size_t capacity; /**< Number of elements the column can hold. */
    unsigned char data[]; /**< Column storage. */
} ColumnBlock;

/* ------------------------------------------------------------------------- */
/* Arena-backed table growth                                                  */
/* ------------------------------------------------------------------------- */

/**
 * @brief Capacity a table needs to reach after growing from cap.
 * @return Target capacity, or 0 when the growth is impossible.
 */
static uint32_t table_grow_target(uint32_t cap, uint32_t needed) {
#ifdef __CPROVER__
    (void)cap;
    return needed <= RE_ECS_CAP_MAX ? RE_ECS_CAP_MAX : 0u;
#else
    uint32_t target = cap < RE_ECS_INITIAL_CAP ? RE_ECS_INITIAL_CAP : cap;
    while (target < needed) {
        if (target > UINT32_MAX / 2u) {
            target = needed;
            break;
        }
        target *= 2u;
    }
    return target;
#endif
}

/**
 * @brief Ensures *array holds at least needed elements.
 *
 * The fresh block is zero-filled so never-initialized slots behave like a
 * zeroed struct, then the old contents are copied over. The old block is
 * intentionally leaked to the arena.
 *
 * @param array Array pointer to update.
 * @param cap Capacity of *array, updated on success.
 * @param needed Required element count.
 * @param elem_size Size of one element in bytes.
 * @param arena Arena to allocate from.
 * @return True when the table holds at least needed elements.
 */
static bool table_reserve(void **array, uint32_t *cap, uint32_t needed,
                          size_t elem_size, ReArena *arena) {
    if (needed <= *cap) {
        return true;
    }
    uint32_t target = table_grow_target(*cap, needed);
    if (target == 0u || (size_t)target > SIZE_MAX / elem_size) {
        return false;
    }
    size_t bytes = (size_t)target * elem_size;
    void *fresh = re_arena_alloc(arena, bytes);
    if (fresh == NULL) {
        return false;
    }
    memset(fresh, 0, bytes);
    if (*array != NULL) {
        memcpy(fresh, *array, (size_t)*cap * elem_size);
    }
    *array = fresh;
    *cap = target;
    return true;
}

/**
 * @brief Grows the four per-slot entity tables to at least needed slots.
 *
 * All four tables see identical inputs, so they all land on the same
 * capacity and stay index-compatible.
 *
 * @param ecs World.
 * @param needed Required slot count.
 * @return True on success; on failure the world is left unchanged.
 */
static bool entity_tables_grow(ReEcs *ecs, uint32_t needed) {
    uint32_t c0 = ecs->entity_cap;
    uint32_t c1 = c0;
    uint32_t c2 = c0;
    uint32_t c3 = c0;
    void *a0 = ecs->entity_gen;
    void *a1 = ecs->entity_arch;
    void *a2 = ecs->entity_chunk;
    void *a3 = ecs->entity_row;
    const size_t elem = sizeof(uint32_t);
    if (!table_reserve(&a0, &c0, needed, elem, &ecs->arena) ||
        !table_reserve(&a1, &c1, needed, elem, &ecs->arena) ||
        !table_reserve(&a2, &c2, needed, elem, &ecs->arena) ||
        !table_reserve(&a3, &c3, needed, elem, &ecs->arena)) {
        return false;
    }
    ecs->entity_gen = a0;
    ecs->entity_arch = a1;
    ecs->entity_chunk = a2;
    ecs->entity_row = a3;
    ecs->entity_cap = c0;
    return true;
}

/* ------------------------------------------------------------------------- */
/* Generation scheme                                                          */
/* ------------------------------------------------------------------------- */

/**
 * @brief Next generation of a slot: never 0, strictly increasing.
 * @param current Stored generation (0 for a never-used slot).
 */
static uint32_t generation_next(uint32_t current) {
    uint32_t next = current + 1u;
    return next == 0u ? 1u : next;
}

/* ------------------------------------------------------------------------- */
/* Chunks                                                                     */
/* ------------------------------------------------------------------------- */

/**
 * @brief Entities one chunk of the archetype can hold.
 *
 * Derived from RE_CHUNK_BYTES minus the entity column; a single oversized
 * component forces one entity per chunk instead of dividing by zero.
 *
 * @param ecs World (component sizes).
 * @param arch Archetype.
 * @return Capacity in [1, RE_CHUNK_BYTES / sizeof(ReEntity)].
 */
static uint32_t chunk_capacity(const ReEcs *ecs, const ReArchetype *arch) {
    size_t bytes = sizeof(ReEntity);
    for (uint32_t i = 0; i < arch->column_count; i++) {
        size_t size = ecs->component_sizes[arch->columns[i]];
        if (size > RE_CHUNK_BYTES || bytes > RE_CHUNK_BYTES - size) {
            return 1u;
        }
        bytes += size;
    }
    size_t capacity = (size_t)RE_CHUNK_BYTES / bytes;
    return capacity == 0u ? 1u : (uint32_t)capacity;
}

/**
 * @brief Allocates one zeroed SoA column per archetype component.
 *
 * @param ecs World.
 * @param arch Archetype owning the columns.
 * @param capacity Elements per column.
 * @param columns Output array with arch->column_count entries.
 * @return True on success; on failure *columns is left partially filled and
 *         must be discarded by the caller.
 */
static bool chunk_alloc_columns(ReEcs *ecs, const ReArchetype *arch,
                                uint32_t capacity, void **columns) {
    for (uint32_t i = 0; i < arch->column_count; i++) {
        size_t stride = ecs->component_sizes[arch->columns[i]];
        if (stride == 0u ||
            (size_t)capacity > (SIZE_MAX - sizeof(ColumnBlock)) / stride) {
            return false;
        }
        void *block = re_arena_alloc(&ecs->arena,
                                     sizeof(ColumnBlock) + (size_t)capacity * stride);
        if (block == NULL) {
            return false;
        }
        ColumnBlock *column = block;
        column->stride = stride;
        column->capacity = capacity;
        memset(column->data, 0, (size_t)capacity * stride);
        columns[i] = column->data;
    }
    return true;
}

/**
 * @brief Allocates and zeroes the storage of one chunk.
 *
 * Nothing is published into *chunk until every allocation succeeded, so a
 * failed attempt leaves a zeroed slot that a retry can rebuild.
 *
 * @param ecs World.
 * @param arch Archetype owning the chunk.
 * @param index Chunk index already reserved in arch->chunks.
 * @param capacity Entities the chunk must hold.
 * @return True on success.
 */
static bool chunk_create(ReEcs *ecs, ReArchetype *arch, uint32_t index,
                         uint32_t capacity) {
    ReChunk *chunk = &arch->chunks[index];
    ReEntity *entities = re_arena_alloc(&ecs->arena,
                                        (size_t)capacity * sizeof(ReEntity));
    if (entities == NULL) {
        return false;
    }
    memset(entities, 0, (size_t)capacity * sizeof(ReEntity));
    void **columns = NULL;
    if (arch->column_count > 0u) {
        columns = re_arena_alloc(&ecs->arena,
                                 (size_t)arch->column_count * sizeof(void *));
        if (columns == NULL || !chunk_alloc_columns(ecs, arch, capacity, columns)) {
            return false;
        }
    }
    chunk->count = 0u;
    chunk->capacity = capacity;
    chunk->entities = entities;
    chunk->columns = columns;
    return true;
}

/**
 * @brief Ensures the archetype's last chunk exists and has a free row.
 * @param ecs World.
 * @param arch Archetype.
 * @param index_out Receives the chunk index to append into.
 * @return True on success.
 */
static bool chunk_prepare(ReEcs *ecs, ReArchetype *arch, uint32_t *index_out) {
    if (arch->chunk_count > 0u) {
        const ReChunk *tail = &arch->chunks[arch->chunk_count - 1u];
        if (tail->count < tail->capacity) {
            *index_out = arch->chunk_count - 1u;
            return true;
        }
    }
    uint32_t index = arch->chunk_count;
    if (!table_reserve((void **)&arch->chunks, &arch->chunk_cap, index + 1u,
                       sizeof(ReChunk), &ecs->arena)) {
        return false;
    }
    ReChunk *chunk = &arch->chunks[index];
    if (chunk->entities == NULL &&
        !chunk_create(ecs, arch, index, chunk_capacity(ecs, arch))) {
        return false;
    }
    /* A chunk popped by chunk_remove_at keeps its storage; reuse it. */
    chunk->count = 0u;
    arch->chunk_count = index + 1u;
    *index_out = index;
    return true;
}

/**
 * @brief Swap-removes row from a chunk, patching the moved entity's slot.
 * @param ecs World.
 * @param arch Archetype owning the chunk.
 * @param chunk_index Chunk index.
 * @param row Row to remove.
 */
static void chunk_remove_at(ReEcs *ecs, ReArchetype *arch, uint32_t chunk_index,
                            uint32_t row) {
    ReChunk *chunk = &arch->chunks[chunk_index];
    uint32_t last = chunk->count - 1u;
    if (row < last) {
        ReEntity moved = chunk->entities[last];
        chunk->entities[row] = moved;
        for (uint32_t i = 0; i < arch->column_count; i++) {
            size_t stride = ecs->component_sizes[arch->columns[i]];
            unsigned char *column = chunk->columns[i];
            memcpy(column + (size_t)row * stride, column + (size_t)last * stride, stride);
        }
        uint32_t slot = re_entity_index(moved);
        ecs->entity_chunk[slot] = chunk_index;
        ecs->entity_row[slot] = row;
    }
    chunk->count = last;
    if (last == 0u && chunk_index + 1u == arch->chunk_count) {
        arch->chunk_count = chunk_index;
    }
}

/**
 * @brief Appends an entity with zero-initialized component data.
 * @param ecs World.
 * @param arch_id Destination archetype.
 * @param e Entity handle.
 * @param chunk_out Receives the chunk index.
 * @param row_out Receives the row.
 * @return True on success.
 */
static bool arch_append(ReEcs *ecs, uint32_t arch_id, ReEntity e,
                        uint32_t *chunk_out, uint32_t *row_out) {
    ReArchetype *arch = &ecs->archetypes[arch_id];
    uint32_t index = 0u;
    if (!chunk_prepare(ecs, arch, &index)) {
        return false;
    }
    ReChunk *chunk = &arch->chunks[index];
    uint32_t row = chunk->count;
    /* A recycled row may still hold the previous occupant's bytes. */
    for (uint32_t i = 0; i < arch->column_count; i++) {
        size_t stride = ecs->component_sizes[arch->columns[i]];
        memset((unsigned char *)chunk->columns[i] + (size_t)row * stride, 0, stride);
    }
    chunk->entities[row] = e;
    chunk->count = row + 1u;
    *chunk_out = index;
    *row_out = row;
    return true;
}

/* ------------------------------------------------------------------------- */
/* Archetypes                                                                 */
/* ------------------------------------------------------------------------- */

/**
 * @brief Finds an archetype whose component set equals type.
 * @return Archetype id, or UINT32_MAX when absent.
 */
static uint32_t arch_find(const ReEcs *ecs, ReMask type) {
    for (uint32_t i = 0; i < ecs->archetype_count; i++) {
        if (re_mask_equal(ecs->archetypes[i].type, type)) {
            return i;
        }
    }
    return RE_ARCH_NONE;
}

/**
 * @brief Zero-initializes a fresh archetype slot: no edges, no columns.
 * @param arch Archetype slot.
 * @param id Archetype id.
 * @param type Component set.
 */
static void arch_reset(ReArchetype *arch, uint32_t id, ReMask type) {
    memset(arch, 0, sizeof *arch);
    arch->type = type;
    arch->id = id;
    memset(arch->column_pos, RE_COLUMN_ABSENT, sizeof arch->column_pos);
    /* C23 requires two's complement: all-bits-one is exactly -1. */
    memset(arch->edge_add, 0xFF, sizeof arch->edge_add);
    memset(arch->edge_rem, 0xFF, sizeof arch->edge_rem);
}

/**
 * @brief Derives the column table of dst from src, adding or dropping c.
 *
 * Removing a component also removes its column, so the remaining columns
 * keep their relative order and memcpy can copy them position by position.
 *
 * @param dst Archetype being built.
 * @param src Archetype the entity comes from.
 * @param c Component added or removed.
 * @param adding True for an add transition, false for a removal.
 */
static void arch_copy_columns(ReArchetype *dst, const ReArchetype *src,
                              ReComponentId c, bool adding) {
    uint32_t count = 0u;
    for (uint32_t i = 0; i < src->column_count; i++) {
        ReComponentId column = src->columns[i];
        if (!adding && column == c) {
            continue;
        }
        dst->columns[count] = column;
        dst->column_pos[column] = (uint8_t)count;
        count++;
    }
    if (adding) {
        dst->columns[count] = c;
        dst->column_pos[c] = (uint8_t)count;
        count++;
    }
    dst->column_count = count;
}

/**
 * @brief Creates an archetype by adding or removing one component.
 *
 * @param ecs World.
 * @param type Component set of the new archetype.
 * @param src_id Source archetype id (RE_ARCH_NONE for the empty archetype).
 * @param c Component being added or removed.
 * @param adding True for an add transition, false for a removal.
 * @return New archetype id, or UINT32_MAX on failure.
 */
static uint32_t arch_create(ReEcs *ecs, ReMask type, uint32_t src_id,
                            ReComponentId c, bool adding) {
    if (ecs->archetype_count > (uint32_t)INT32_MAX) {
        return RE_ARCH_NONE; /* ids must fit the int32_t edge cache */
    }
    uint32_t id = ecs->archetype_count;
    if (!table_reserve((void **)&ecs->archetypes, &ecs->archetype_cap, id + 1u,
                       sizeof(ReArchetype), &ecs->arena)) {
        return RE_ARCH_NONE;
    }
    ReArchetype *arch = &ecs->archetypes[id];
    arch_reset(arch, id, type);
    if (src_id < ecs->archetype_count) {
        arch_copy_columns(arch, &ecs->archetypes[src_id], c, adding);
    }
    ecs->archetype_count = id + 1u;
    return id;
}

/**
 * @brief Resolves (creating and caching on miss) one add/remove edge.
 *
 * The edge cache turns a repeated transition into O(1); the miss path scans
 * the archetype table so every component set exists at most once.
 *
 * @param ecs World.
 * @param arch_id Source archetype.
 * @param c Component to add or remove.
 * @param adding True for add, false for remove.
 * @return Destination archetype id, or UINT32_MAX on failure.
 */
static uint32_t arch_transition(ReEcs *ecs, uint32_t arch_id, ReComponentId c,
                                bool adding) {
    ReMask bit = re_mask_bit(c);
    ReMask cur = ecs->archetypes[arch_id].type;
    ReMask type = adding ? re_mask_or(cur, bit) : re_mask_and(cur, re_mask_not(bit));
    uint32_t dst = arch_find(ecs, type);
    if (dst == RE_ARCH_NONE) {
        dst = arch_create(ecs, type, arch_id, c, adding);
        if (dst == RE_ARCH_NONE) {
            return RE_ARCH_NONE;
        }
    }
    if (adding) {
        ecs->archetypes[arch_id].edge_add[c] = (int32_t)dst;
    } else {
        ecs->archetypes[arch_id].edge_rem[c] = (int32_t)dst;
    }
    return dst;
}

/**
 * @brief Moves an entity to another archetype, copying/dropping columns.
 * @param ecs World.
 * @param e Live entity handle.
 * @param dst_id Destination archetype (different mask from the source).
 * @return True on success; on failure the entity is untouched.
 */
static bool move_entity(ReEcs *ecs, ReEntity e, uint32_t dst_id) {
    uint32_t slot = re_entity_index(e);
    ReArchetype *src = &ecs->archetypes[ecs->entity_arch[slot]];
    ReArchetype *dst = &ecs->archetypes[dst_id];
    uint32_t src_chunk = ecs->entity_chunk[slot];
    uint32_t src_row = ecs->entity_row[slot];
    uint32_t dst_chunk = 0u;
    if (!chunk_prepare(ecs, dst, &dst_chunk)) {
        return false;
    }
    ReChunk *from = &src->chunks[src_chunk];
    ReChunk *to = &dst->chunks[dst_chunk];
    uint32_t row = to->count;
    for (uint32_t i = 0; i < dst->column_count; i++) {
        ReComponentId c = dst->columns[i];
        size_t stride = ecs->component_sizes[c];
        unsigned char *out = (unsigned char *)to->columns[i] + (size_t)row * stride;
        uint8_t pos = src->column_pos[c];
        if (!re_mask_test(&src->type, (unsigned)c)) {
            memset(out, 0, stride); /* new component starts zeroed */
        } else {
            const unsigned char *in = (const unsigned char *)from->columns[pos];
            memcpy(out, in + (size_t)src_row * stride, stride);
        }
    }
    to->entities[row] = e;
    to->count = row + 1u;
    ecs->entity_arch[slot] = dst_id;
    ecs->entity_chunk[slot] = dst_chunk;
    ecs->entity_row[slot] = row;
    chunk_remove_at(ecs, src, src_chunk, src_row);
    return true;
}

/* ------------------------------------------------------------------------- */
/* Bootstrap                                                                  */
/* ------------------------------------------------------------------------- */

/**
 * @brief Resets every field except the arena and creates the empty archetype.
 * @param ecs World whose arena is already initialized.
 * @return True on success.
 */
static bool re_ecs_bootstrap(ReEcs *ecs) {
    ReArena arena = ecs->arena;
    memset(ecs, 0, sizeof *ecs);
    ecs->arena = arena;
    uint32_t id = arch_create(ecs, re_mask_zero(), RE_ARCH_NONE, 0u, false);
    if (id == RE_ARCH_NONE) {
        return false;
    }
    ecs->empty_arch = id;
    return true;
}

/* ------------------------------------------------------------------------- */
/* Public API                                                                 */
/* ------------------------------------------------------------------------- */

bool re_ecs_init(ReEcs *ecs, void *mem, size_t mem_size) {
    if (ecs == NULL || mem == NULL || mem_size == 0u) {
        return false;
    }
    memset(ecs, 0, sizeof *ecs);
    if (!re_arena_init(&ecs->arena, mem, mem_size)) {
        return false;
    }
    return re_ecs_bootstrap(ecs);
}

ReComponentId re_ecs_register(ReEcs *ecs, size_t component_size) {
    if (ecs == NULL || component_size == 0u) {
        return (ReComponentId)RE_COMPONENT_NONE;
    }
    int bit = re_mask_first_bit(re_mask_not(ecs->component_used));
    if (bit < 0) {
        return (ReComponentId)RE_COMPONENT_NONE;
    }
    ReComponentId id = (ReComponentId)bit;
    ecs->component_sizes[id] = component_size;
    re_mask_set(&ecs->component_used, (unsigned)id);
    ecs->component_count++;
    return id;
}

size_t re_ecs_component_size(const ReEcs *ecs, ReComponentId id) {
    if (ecs == NULL || id >= (ReComponentId)RE_COMPONENT_MAX ||
        !re_mask_test(&ecs->component_used, (unsigned)id)) {
        return 0u;
    }
    return ecs->component_sizes[id];
}

ReEntity re_ecs_spawn(ReEcs *ecs) {
    if (ecs == NULL || ecs->empty_arch >= ecs->archetype_count) {
        return RE_ENTITY_NULL; /* NULL world or a failed init */
    }
    bool recycled = ecs->free_count > 0u;
    uint32_t slot;
    if (recycled) {
        slot = ecs->free_slots[ecs->free_count - 1u];
    } else {
        if (ecs->next_slot == UINT32_MAX) {
            return RE_ENTITY_NULL;
        }
        if (ecs->next_slot == ecs->entity_cap &&
            !entity_tables_grow(ecs, ecs->next_slot + 1u)) {
            return RE_ENTITY_NULL;
        }
        slot = ecs->next_slot;
    }
    uint32_t chunk = 0u;
    uint32_t row = 0u;
    ReEntity e = re_entity_make(slot, generation_next(ecs->entity_gen[slot]));
    if (!arch_append(ecs, ecs->empty_arch, e, &chunk, &row)) {
        return RE_ENTITY_NULL;
    }
    ecs->entity_gen[slot] = re_entity_gen(e);
    ecs->entity_arch[slot] = ecs->empty_arch;
    ecs->entity_chunk[slot] = chunk;
    ecs->entity_row[slot] = row;
    if (recycled) {
        ecs->free_count--;
    } else {
        ecs->next_slot++;
    }
    return e;
}

bool re_ecs_alive(const ReEcs *ecs, ReEntity e) {
    if (ecs == NULL) {
        return false;
    }
    uint32_t slot = re_entity_index(e);
    uint32_t gen = re_entity_gen(e);
    return gen != 0u && slot < ecs->entity_cap && ecs->entity_gen[slot] == gen;
}

void re_ecs_despawn(ReEcs *ecs, ReEntity e) {
    if (!re_ecs_alive(ecs, e)) {
        return;
    }
    uint32_t slot = re_entity_index(e);
    ReArchetype *arch = &ecs->archetypes[ecs->entity_arch[slot]];
    chunk_remove_at(ecs, arch, ecs->entity_chunk[slot], ecs->entity_row[slot]);
    ecs->entity_gen[slot] = generation_next(ecs->entity_gen[slot]);
    if (!table_reserve((void **)&ecs->free_slots, &ecs->free_cap,
                       ecs->free_count + 1u, sizeof(uint32_t), &ecs->arena)) {
        /* The slot stays dead but is not recycled: the world loses one slot. */
        return;
    }
    ecs->free_slots[ecs->free_count] = slot;
    ecs->free_count++;
}

bool re_ecs_add(ReEcs *ecs, ReEntity e, ReComponentId c) {
    if (ecs == NULL || c >= (ReComponentId)RE_COMPONENT_MAX ||
        !re_mask_test(&ecs->component_used, (unsigned)c)) {
        return false;
    }
    if (!re_ecs_alive(ecs, e)) {
        return false;
    }
    uint32_t slot = re_entity_index(e);
    uint32_t arch_id = ecs->entity_arch[slot];
    if (re_mask_test(&ecs->archetypes[arch_id].type, (unsigned)c)) {
        return true;
    }
    int32_t edge = ecs->archetypes[arch_id].edge_add[c];
    uint32_t dst = edge >= 0 ? (uint32_t)edge : arch_transition(ecs, arch_id, c, true);
    if (dst == RE_ARCH_NONE) {
        return false;
    }
    return move_entity(ecs, e, dst);
}

bool re_ecs_remove(ReEcs *ecs, ReEntity e, ReComponentId c) {
    if (ecs == NULL) {
        return false;
    }
    if (!re_ecs_alive(ecs, e) || c >= (ReComponentId)RE_COMPONENT_MAX) {
        return true; /* nothing to remove: the component is absent */
    }
    uint32_t slot = re_entity_index(e);
    uint32_t arch_id = ecs->entity_arch[slot];
    if (!re_mask_test(&ecs->archetypes[arch_id].type, (unsigned)c)) {
        return true;
    }
    int32_t edge = ecs->archetypes[arch_id].edge_rem[c];
    uint32_t dst = edge >= 0 ? (uint32_t)edge : arch_transition(ecs, arch_id, c, false);
    if (dst == RE_ARCH_NONE) {
        return false;
    }
    return move_entity(ecs, e, dst);
}

void *re_ecs_get(ReEcs *ecs, ReEntity e, ReComponentId c) {
    if (!re_ecs_alive(ecs, e) || c >= (ReComponentId)RE_COMPONENT_MAX) {
        return NULL;
    }
    uint32_t slot = re_entity_index(e);
    const ReArchetype *arch = &ecs->archetypes[ecs->entity_arch[slot]];
    if (!re_mask_test(&arch->type, (unsigned)c)) {
        return NULL;
    }
    uint32_t pos = arch->column_pos[c];
    if (pos >= arch->column_count) {
        return NULL;
    }
    const ReChunk *chunk = &arch->chunks[ecs->entity_chunk[slot]];
    size_t stride = ecs->component_sizes[c];
    return (unsigned char *)chunk->columns[pos] + (size_t)ecs->entity_row[slot] * stride;
}

bool re_ecs_has(const ReEcs *ecs, ReEntity e, ReComponentId c) {
    if (!re_ecs_alive(ecs, e) || c >= (ReComponentId)RE_COMPONENT_MAX) {
        return false;
    }
    uint32_t slot = re_entity_index(e);
    return re_mask_test(&ecs->archetypes[ecs->entity_arch[slot]].type, (unsigned)c);
}

ReMask re_ecs_mask_of(const ReEcs *ecs, ReEntity e) {
    if (!re_ecs_alive(ecs, e)) {
        return re_mask_zero();
    }
    uint32_t slot = re_entity_index(e);
    return ecs->archetypes[ecs->entity_arch[slot]].type;
}

void *re_ecs_arch_column(const ReArchetype *arch, ReComponentId c,
                         size_t *stride_out) {
    if (stride_out != NULL) {
        *stride_out = 0u;
    }
    if (arch == NULL || c >= (ReComponentId)RE_COMPONENT_MAX) {
        return NULL;
    }
    if (!re_mask_test(&arch->type, (unsigned)c)) {
        return NULL;
    }
    uint32_t pos = arch->column_pos[c];
    if (pos >= arch->column_count) {
        return NULL;
    }
    for (uint32_t i = 0; i < arch->chunk_count; i++) {
        const ReChunk *chunk = &arch->chunks[i];
        if (chunk->count == 0u) {
            continue;
        }
        const unsigned char *data = chunk->columns[pos];
        /* container_of: the header lives immediately before the column. */
        const ColumnBlock *block =
            (const ColumnBlock *)(data - offsetof(ColumnBlock, data));
        if (stride_out != NULL) {
            *stride_out = block->stride;
        }
        return chunk->columns[pos];
    }
    return NULL;
}

bool re_ecs_iter_next(ReEcsIter *it, const ReArchetype **arch_out,
                      const ReChunk **chunk_out, const ReEntity **entities_out,
                      uint32_t *count_out) {
    if (it == NULL || it->ecs == NULL) {
        return false;
    }
    const ReEcs *ecs = it->ecs;
    while (it->arch < ecs->archetype_count) {
        const ReArchetype *arch = &ecs->archetypes[it->arch];
        if (re_mask_subset(it->with, arch->type) &&
            !re_mask_overlaps(it->without, arch->type)) {
            while (it->chunk < arch->chunk_count) {
                const ReChunk *chunk = &arch->chunks[it->chunk];
                it->chunk++;
                if (chunk->count == 0u) {
                    continue;
                }
                *arch_out = arch;
                *chunk_out = chunk;
                *entities_out = chunk->entities;
                *count_out = chunk->count;
                return true;
            }
        }
        it->arch++;
        it->chunk = 0u;
    }
    return false;
}

void *re_ecs_chunk_column(const ReChunk *chunk, const ReArchetype *arch,
                          ReComponentId c, size_t *stride_out) {
    if (stride_out != NULL) {
        *stride_out = 0u;
    }
    if (chunk == NULL || arch == NULL || c >= (ReComponentId)RE_COMPONENT_MAX) {
        return NULL;
    }
    if (!re_mask_test(&arch->type, (unsigned)c)) {
        return NULL;
    }
    uint32_t pos = arch->column_pos[c];
    if (pos >= arch->column_count) {
        return NULL;
    }
    const unsigned char *data = chunk->columns[pos];
    if (data == NULL) {
        return NULL;
    }
    /* container_of: the header lives immediately before the column. */
    const ColumnBlock *block =
        (const ColumnBlock *)(data - offsetof(ColumnBlock, data));
    if (stride_out != NULL) {
        *stride_out = block->stride;
    }
    return chunk->columns[pos];
}

const ReArchetype *re_ecs_archetype(const ReEcs *ecs, uint32_t archetype_id) {
    if (ecs == NULL || archetype_id >= ecs->archetype_count) {
        return NULL;
    }
    return &ecs->archetypes[archetype_id];
}

uint32_t re_ecs_entity_count(const ReEcs *ecs) {
    if (ecs == NULL) {
        return 0u;
    }
    uint32_t total = 0u;
    for (uint32_t i = 0; i < ecs->archetype_count; i++) {
        const ReArchetype *arch = &ecs->archetypes[i];
        for (uint32_t j = 0; j < arch->chunk_count; j++) {
            total += arch->chunks[j].count;
        }
    }
    return total;
}

uint32_t re_ecs_archetype_count(const ReEcs *ecs) {
    return ecs == NULL ? 0u : ecs->archetype_count;
}

void re_ecs_clear(ReEcs *ecs) {
    if (ecs == NULL) {
        return;
    }
    re_arena_reset(&ecs->arena);
    (void)re_ecs_bootstrap(ecs);
}
