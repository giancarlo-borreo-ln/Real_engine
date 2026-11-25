/**
 * @file re_render_graph.h
 * @brief Declarative render graph: barrier synthesis, transient
 *        aliasing and read-before-write validation.
 *
 * Passes are declared in submission order with per-resource read/write
 * lists. Compilation walks the passes tracking each resource's layout
 * and initialization, and:
 *   - fails (returns false) when a pass reads a resource that was
 *     never written and has no defined initial contents (the
 *     read-before-write hazard the harnesses prove absent);
 *   - emits a layout-transition barrier before every access whose
 *     required layout differs from the tracked one;
 *   - emits the transition to each resource's final layout after the
 *     last pass;
 *   - aliases transient resources with disjoint lifetime intervals
 *     onto the same storage slot (greedy first-fit on sorted
 *     intervals).
 *
 * The graph is GPU-agnostic: ReGraphLayout is an abstract layout
 * domain and the Vulkan backend maps layouts to VkImageLayout plus
 * access/stage pairs. All output arrays live in the caller arena.
 */
#ifndef RE_RENDER_GRAPH_H
#define RE_RENDER_GRAPH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/** Abstract resource layouts (small domain for the proofs). */
typedef enum ReGraphLayout {
    RE_LAYOUT_UNDEFINED = 0,        /**< Contents undefined. */
    RE_LAYOUT_COLOR_ATTACHMENT,     /**< Written as a color target. */
    RE_LAYOUT_DEPTH_ATTACHMENT,     /**< Written as a depth target. */
    RE_LAYOUT_DEPTH_READ_ONLY,      /**< Read as a depth attachment. */
    RE_LAYOUT_SHADER_READ,          /**< Sampled / storage read. */
    RE_LAYOUT_TRANSFER_DST,         /**< Copy destination. */
    RE_LAYOUT_PRESENT,              /**< Presentable image. */
    RE_LAYOUT_COUNT
} ReGraphLayout;

/**
 * @brief One graph resource.
 */
typedef struct ReGraphResource {
    const char *name;            /**< Debug name (may be NULL). */
    bool transient;              /**< Aliasing candidate (no defined
                                      initial contents, not read after
                                      its last access). */
    ReGraphLayout initial_layout; /**< Layout before the first pass.
                                      UNDEFINED means the contents are
                                      undefined until first written. */
    ReGraphLayout final_layout;   /**< Layout after the last pass. */
} ReGraphResource;

/**
 * @brief One graph pass.
 */
typedef struct ReGraphPass {
    const char *name;          /**< Debug name (may be NULL). */
    const uint32_t *reads;     /**< Resource indices read (read_count). */
    uint32_t read_count;       /**< Reads array length. */
    const uint32_t *writes;    /**< Resource indices written (write_count). */
    uint32_t write_count;      /**< Writes array length. */
    void (*execute)(void *ctx); /**< Backend body (called in compiled order). */
    void *ctx;                 /**< Argument passed to execute. */
} ReGraphPass;

/**
 * @brief One synthesized layout transition (before a pass or at the end).
 */
typedef struct ReGraphBarrier {
    uint32_t res;          /**< Resource index. */
    ReGraphLayout src;     /**< Layout before the barrier. */
    ReGraphLayout dst;     /**< Layout after the barrier. */
} ReGraphBarrier;

/**
 * @brief Compiled graph: pass order, barrier list, aliases.
 */
typedef struct ReGraphCompiled {
    uint32_t *pass_order;    /**< Pass indices in submission order. */
    ReGraphBarrier *barriers;/**< Packed barrier array. */
    uint32_t *barrier_start; /**< [pass_count + 1]: barriers before pass k
                                  are [start[k], start[k+1]); the final
                                  segment holds the end-of-frame
                                  transitions to final layouts. */
    uint32_t barrier_count;  /**< Packed barriers. */
    uint32_t *alias_of;      /**< [resource_count]: alias_of[r] = r when the
                                  resource keeps its own storage, or the
                                  index of an earlier resource whose
                                  storage it shares. */
    uint32_t pass_count;     /**< Number of passes. */
} ReGraphCompiled;

/**
 * @brief Compiles a render graph into caller-arena memory.
 *
 * Read-before-write rule: resource r may be read by pass p only when r
 * was written by some earlier pass, or initial_layout != UNDEFINED and
 * no pass wrote it yet. A violation makes compilation fail with false.
 *
 * @param resources Resource array.
 * @param resource_count Number of resources.
 * @param passes Pass array (already in submission order).
 * @param pass_count Number of passes.
 * @param out Compiled output.
 * @param mem Arena memory for the output arrays.
 * @param mem_size Size of mem in bytes.
 * @return True on success (false = invalid graph or OOM).
 */
bool re_graph_compile(const ReGraphResource *resources, uint32_t resource_count,
                      const ReGraphPass *passes, uint32_t pass_count,
                      ReGraphCompiled *out, void *mem, size_t mem_size);

/**
 * @brief Validates a compiled graph's structure (used by proofs and
 *        debug builds): barrier indices in range, layouts in domain,
 *        alias indices earlier than the aliasing resource.
 * @param resources Resource array.
 * @param resource_count Number of resources.
 * @param c Compiled graph.
 * @return True when structurally valid.
 */
bool re_graph_validate(const ReGraphResource *resources, uint32_t resource_count,
                       const ReGraphCompiled *c);

#endif /* RE_RENDER_GRAPH_H */
