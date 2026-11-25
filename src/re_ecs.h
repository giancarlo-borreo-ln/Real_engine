/**
 * @file re_ecs.h
 * @brief Generational sparse-set archetype ECS with chunked SoA storage.
 *
 * Entities are 64-bit handles: 32-bit slot index plus 32-bit generation.
 * Reusing a slot bumps the generation, so a stale handle can never alias
 * a new entity (the ABA guarantee the proof harnesses verify).
 *
 * Entities live in archetypes (unique component-type masks). Each
 * archetype stores its entities in chunks of at most RE_CHUNK_BYTES
 * bytes, one contiguous column per component (structure of arrays).
 * Iteration visits whole chunks, which keeps cache lines hot.
 */
#ifndef RE_ECS_H
#define RE_ECS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "re_core.h"
#include "re_mem.h"

/** 64-bit generational entity handle. */
typedef uint64_t ReEntity;

/** Handle that never refers to a live entity. */
#define RE_ENTITY_NULL UINT64_MAX

/** @return Slot index of the entity. */
static inline uint32_t re_entity_index(ReEntity e) {
    return (uint32_t)(e & UINT64_C(0xFFFFFFFF));
}

/** @return Generation of the entity. */
static inline uint32_t re_entity_gen(ReEntity e) {
    return (uint32_t)(e >> 32u);
}

/** @return Handle built from index and generation. */
static inline ReEntity re_entity_make(uint32_t index, uint32_t generation) {
    return ((uint64_t)generation << 32u) | (uint64_t)index;
}

/** Maximum number of distinct component types. */
#define RE_COMPONENT_MAX 256u

/** Component type identifier assigned by re_ecs_register. */
typedef uint32_t ReComponentId;

/** Sentinel for "no component". */
#define RE_COMPONENT_NONE RE_COMPONENT_MAX

/** Budget in bytes for one chunk of an archetype. */
#define RE_CHUNK_BYTES 16384u

/**
 * @brief One chunk: entities plus one SoA column per component.
 */
typedef struct ReChunk {
    uint32_t count;    /**< Number of live entities in this chunk. */
    uint32_t capacity; /**< Maximum entities the columns can hold. */
    ReEntity *entities; /**< Entity handle column (count entries). */
    void **columns;     /**< One pointer per archetype column (count entries). */
} ReChunk;

/**
 * @brief Archetype: every entity in it has exactly the same components.
 */
typedef struct ReArchetype {
    ReMask type;                              /**< Component set of this archetype. */
    uint32_t id;                              /**< Archetype index in the world. */
    uint32_t column_count;                    /**< Number of component columns. */
    ReComponentId columns[RE_COMPONENT_MAX];  /**< Column component ids (0..column_count-1 valid). */
    uint8_t column_pos[RE_COMPONENT_MAX];     /**< Inverse map: component -> column index (0xFF = absent). */
    uint32_t chunk_count;                     /**< Number of chunks in use. */
    uint32_t chunk_cap;                       /**< Allocated chunk slots. */
    ReChunk *chunks;                          /**< Chunk array. */
    int32_t edge_add[RE_COMPONENT_MAX];       /**< Archetype reached by adding a component (lazy cache). */
    int32_t edge_rem[RE_COMPONENT_MAX];       /**< Archetype reached by removing a component (lazy cache). */
} ReArchetype;

/**
 * @brief The ECS world: entity table, archetypes and component registry.
 *
 * All dynamic storage comes from the caller-provided arena (re_ecs_init
 * wraps it). Growing a table allocates a new block and copies the old
 * one; the old block stays in the arena until re_ecs_clear.
 */
typedef struct ReEcs {
    ReArena arena;                 /**< Storage arena for every table below. */
    uint32_t *entity_gen;          /**< Generation per slot. */
    uint32_t *entity_arch;         /**< Archetype id per slot (live slots only). */
    uint32_t *entity_chunk;        /**< Chunk index per slot. */
    uint32_t *entity_row;          /**< Row inside the chunk per slot. */
    uint32_t entity_cap;           /**< Slots allocated in the tables. */
    uint32_t next_slot;            /**< First never-used slot. */
    uint32_t *free_slots;          /**< Stack of recycled slots. */
    uint32_t free_count;           /**< Entries in free_slots. */
    uint32_t free_cap;             /**< Capacity of free_slots. */
    ReArchetype *archetypes;       /**< Archetype table. */
    uint32_t archetype_count;      /**< Live archetypes. */
    uint32_t archetype_cap;        /**< Allocated archetype slots. */
    size_t component_sizes[RE_COMPONENT_MAX]; /**< Byte size per component id. */
    ReMask component_used;         /**< Registered component ids. */
    uint32_t component_count;      /**< Number of registered components. */
    uint32_t empty_arch;           /**< Archetype id with no components. */
} ReEcs;

/**
 * @brief Initializes an empty ECS world over caller-provided memory.
 * @param ecs World to initialize.
 * @param mem Backing memory block.
 * @param mem_size Size of mem in bytes.
 * @return True on success.
 */
bool re_ecs_init(ReEcs *ecs, void *mem, size_t mem_size);

/**
 * @brief Registers a component type.
 * @param ecs World.
 * @param component_size Byte size of the component (> 0).
 * @return Component id, or RE_COMPONENT_NONE when the registry is full
 *         or the size is zero.
 */
ReComponentId re_ecs_register(ReEcs *ecs, size_t component_size);

/**
 * @return Byte size of a registered component, or 0 when unregistered.
 */
size_t re_ecs_component_size(const ReEcs *ecs, ReComponentId id);

/**
 * @brief Creates an entity with no components.
 * @param ecs World.
 * @return Entity handle, or RE_ENTITY_NULL on out-of-memory.
 */
ReEntity re_ecs_spawn(ReEcs *ecs);

/**
 * @brief True while the handle refers to a live entity.
 *
 * Checks the slot bounds and the generation, which is what defeats
 * ABA: a handle whose slot was recycled has a stale generation.
 * @param ecs World.
 * @param e Entity handle.
 */
bool re_ecs_alive(const ReEcs *ecs, ReEntity e);

/**
 * @brief Destroys the entity (no-op when already dead).
 * @param ecs World.
 * @param e Entity handle.
 */
void re_ecs_despawn(ReEcs *ecs, ReEntity e);

/**
 * @brief Adds a component to the entity (no-op when already present).
 *
 * Moves the entity to the archetype with the extended type, copying all
 * existing component data. The new component starts zero-initialized.
 * @param ecs World.
 * @param e Entity handle.
 * @param c Component id.
 * @return True when the component is present afterwards.
 */
bool re_ecs_add(ReEcs *ecs, ReEntity e, ReComponentId c);

/**
 * @brief Removes a component from the entity (no-op when absent).
 * @param ecs World.
 * @param e Entity handle.
 * @param c Component id.
 * @return True when the component is absent afterwards.
 */
bool re_ecs_remove(ReEcs *ecs, ReEntity e, ReComponentId c);

/**
 * @return Mutable component pointer, or NULL when absent or dead.
 */
void *re_ecs_get(ReEcs *ecs, ReEntity e, ReComponentId c);

/**
 * @return True when the entity has the component.
 */
bool re_ecs_has(const ReEcs *ecs, ReEntity e, ReComponentId c);

/**
 * @return Component mask of the entity's archetype (zero when dead).
 */
ReMask re_ecs_mask_of(const ReEcs *ecs, ReEntity e);

/**
 * @return Pointer to the whole SoA column of a chunk for component c,
 *         with the element stride in stride_out. NULL when the
 *         archetype lacks the component.
 */
void *re_ecs_arch_column(const ReArchetype *arch, ReComponentId c, size_t *stride_out);

/**
 * @brief Chunk iterator over archetypes matching a mask.
 */
typedef struct ReEcsIter {
    const ReEcs *ecs;  /**< World being iterated. */
    ReMask with;       /**< Required components. */
    ReMask without;    /**< Forbidden components. */
    uint32_t arch;     /**< Next archetype id to examine. */
    uint32_t chunk;    /**< Next chunk index in the current archetype. */
} ReEcsIter;

/**
 * @brief Initializes an iterator.
 * @param ecs World.
 * @param with Required components.
 * @param without Forbidden components.
 * @param it Iterator to initialize.
 */
static inline void re_ecs_iter_init(const ReEcs *ecs, ReMask with, ReMask without, ReEcsIter *it) {
    it->ecs = ecs;
    it->with = with;
    it->without = without;
    it->arch = 0u;
    it->chunk = 0u;
}

/**
 * @brief Advances to the next non-empty chunk.
 * @param it Iterator.
 * @param arch_out Receives the archetype.
 * @param chunk_out Receives the chunk (its columns are reachable through
 *                  re_ecs_chunk_column).
 * @param entities_out Receives the entity column.
 * @param count_out Receives the number of entities in the chunk.
 * @return False when iteration is exhausted.
 */
bool re_ecs_iter_next(ReEcsIter *it, const ReArchetype **arch_out,
                      const ReChunk **chunk_out, const ReEntity **entities_out,
                      uint32_t *count_out);

/**
 * @brief Pointer to the whole SoA column of one specific chunk for
 *        component c, with the element stride in stride_out.
 *
 * Unlike re_ecs_arch_column (which exposes the first non-empty chunk of
 * an archetype), this works on any chunk returned by re_ecs_iter_next.
 * @param chunk Chunk to read from.
 * @param arch Archetype the chunk belongs to.
 * @param c Component id.
 * @param stride_out Receives the element stride (may be NULL).
 * @return Column pointer, or NULL when the archetype lacks the component.
 */
void *re_ecs_chunk_column(const ReChunk *chunk, const ReArchetype *arch,
                          ReComponentId c, size_t *stride_out);

/**
 * @return Archetype by id, or NULL when the id is out of range.
 */
const ReArchetype *re_ecs_archetype(const ReEcs *ecs, uint32_t archetype_id);

/**
 * @return Number of live entities.
 */
uint32_t re_ecs_entity_count(const ReEcs *ecs);

/**
 * @return Number of archetypes.
 */
uint32_t re_ecs_archetype_count(const ReEcs *ecs);

/**
 * @brief Destroys all entities and archetypes, resetting the arena.
 * @param ecs World.
 */
void re_ecs_clear(ReEcs *ecs);

#endif /* RE_ECS_H */
