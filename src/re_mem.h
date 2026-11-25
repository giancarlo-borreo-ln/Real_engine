/**
 * @file re_mem.h
 * @brief Memory subsystem: linear frame arena, fixed-slot free-list pool,
 *        and lock-free MPMC bounded ring buffer.
 *
 * All three allocators are cache-line padded so hot fields never share a
 * line with unrelated data. The arena and pool are single-threaded; the
 * ring buffer is safe for concurrent single-producer/multi-consumer use
 * with any number of producers and consumers.
 */
#ifndef RE_MEM_H
#define RE_MEM_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "re_core.h"

/**
 * @brief Linear frame arena: bump-pointer allocation with O(1) reset.
 *
 * The arena never frees individual blocks; re_arena_reset reclaims
 * everything at once. All allocations are 16-byte aligned at minimum.
 * The struct is padded to one cache line (64 bytes).
 */
typedef struct ReArena {
    unsigned char *begin;  /**< Base of the committed region (16-aligned). */
    unsigned char *end;    /**< One past the last committed byte. */
    size_t offset;         /**< Current bump offset from begin. */
    size_t capacity;       /**< Committed byte count (end - begin). */
    size_t peak;           /**< High-water mark of offset. */
    unsigned char _pad[RE_CACHE_LINE - 5u * sizeof(size_t)]; /**< Cache line fill. */
} ReArena;

/** Size of the arena struct must equal one cache line. */
_Static_assert(sizeof(ReArena) == RE_CACHE_LINE, "ReArena must occupy one cache line");

/**
 * @brief Initializes an arena over a caller-provided memory block.
 *
 * The usable region starts at the first 16-byte-aligned address inside
 * mem; if mem is too small to hold even one aligned byte, init fails.
 *
 * @param a Arena to initialize.
 * @param mem Backing memory.
 * @param size Size of mem in bytes.
 * @return True on success.
 */
bool re_arena_init(ReArena *a, void *mem, size_t size);

/**
 * @brief Bump-allocates size bytes aligned to align.
 *
 * align must be a power of two >= 16. Returns NULL on out-of-memory.
 * Allocations never overlap each other and never touch the arena metadata.
 *
 * @param a Arena.
 * @param size Bytes to allocate.
 * @param align Alignment (power of two >= 16).
 * @return Aligned block pointer, or NULL on OOM.
 */
void *re_arena_alloc_align(ReArena *a, size_t size, size_t align);

/**
 * @brief Bump-allocates size bytes with 16-byte alignment.
 * @param a Arena.
 * @param size Bytes to allocate.
 * @return Aligned block pointer, or NULL on OOM.
 */
void *re_arena_alloc(ReArena *a, size_t size);

/**
 * @brief Resets the arena to empty, reclaiming all allocations.
 * @param a Arena.
 */
void re_arena_reset(ReArena *a);

/**
 * @return Number of bytes currently allocated from the arena.
 */
size_t re_arena_used(const ReArena *a);

/**
 * @return Total usable capacity of the arena in bytes.
 */
size_t re_arena_capacity(const ReArena *a);

/**
 * @return Base address of the usable region.
 */
void *re_arena_base(const ReArena *a);

/**
 * @brief Fixed-slot free-list pool allocator.
 *
 * Allocation and release are both O(1). The free list is threaded through
 * the free slots themselves (the first sizeof(uint32_t) bytes of a free
 * slot store the next free index), so slots must be at least 4 bytes.
 * The struct is padded to one cache line (64 bytes).
 */
typedef struct RePool {
    void *mem;            /**< Slot storage base. */
    size_t slot_size;     /**< Stride between slots (>= 4, caller alignment). */
    uint32_t capacity;    /**< Number of slots. */
    uint32_t free_head;   /**< Index of first free slot or RE_POOL_NIL. */
    uint32_t live;        /**< Number of outstanding allocations. */
    unsigned char _pad[RE_CACHE_LINE - 2u * 8u - 3u * 4u]; /**< Cache line fill. */
} RePool;

/** Sentinel index meaning "no free slot". */
#define RE_POOL_NIL UINT32_MAX

/** Size of the pool struct must equal one cache line. */
_Static_assert(sizeof(RePool) == RE_CACHE_LINE, "RePool must occupy one cache line");

/** Smallest slot stride able to hold the free-list link (bytes). */
#define RE_POOL_MIN_STRIDE ((size_t)4)

/**
 * @brief Effective slot stride for a requested slot size.
 *
 * The free list threads a 4-byte link through free slots, so strides
 * below 4 are raised to 4. Storage for a pool must be sized as
 * capacity * re_pool_stride(slot_size) bytes.
 * @param slot_size Requested slot size in bytes.
 * @return Effective stride (max(slot_size, 4)).
 */
static inline size_t re_pool_stride(size_t slot_size) {
    return slot_size < RE_POOL_MIN_STRIDE ? RE_POOL_MIN_STRIDE : slot_size;
}

/**
 * @brief Initializes a pool over caller-provided slot storage.
 *
 * The storage must hold capacity * re_pool_stride(slot_size) bytes.
 * Use re_pool_slot_size after init to learn the effective stride.
 *
 * @param p Pool to initialize.
 * @param mem Slot storage (capacity * re_pool_stride(slot_size) bytes).
 * @param slot_size Requested slot size in bytes.
 * @param capacity Number of slots.
 * @return True on success.
 */
bool re_pool_init(RePool *p, void *mem, size_t slot_size, uint32_t capacity);

/**
 * @return Effective slot stride in bytes (>= requested slot size).
 */
size_t re_pool_slot_size(const RePool *p);

/**
 * @brief Allocates one slot.
 * @param p Pool.
 * @return Slot pointer, or NULL when the pool is exhausted.
 */
void *re_pool_alloc(RePool *p);

/**
 * @brief Releases a slot previously returned by re_pool_alloc.
 *
 * Validates that ptr lies on a slot boundary inside the storage and that
 * the slot is currently live; invalid pointers are rejected.
 *
 * @param p Pool.
 * @param ptr Slot pointer.
 * @return True when the slot was released.
 */
bool re_pool_free(RePool *p, void *ptr);

/**
 * @return Number of slots in the pool.
 */
uint32_t re_pool_capacity(const RePool *p);

/**
 * @return Number of currently allocated slots.
 */
uint32_t re_pool_live(const RePool *p);

/** Byte size of one MPMC ring slot (padded to a cache line). */
#define RE_MPMC_SLOT_SIZE 64u

/**
 * @brief One ring buffer slot: per-cell ticket word and a payload
 *        pointer, padded to a cache line so producers and consumers on
 *        different cores never contend on the same line.
 *
 * state holds the low 32 bits of the ticket this cell is serving
 * (Vyukov bounded-MPMC protocol): a producer for ticket t spins until
 * state == t, stores the item and publishes state = t + 1; a consumer
 * for ticket t spins until state == t + 1, reads the item and publishes
 * state = t + capacity. Initial state of cell i is i.
 */
typedef struct ReMpmcSlot {
    _Atomic uint32_t state; /**< Low 32 bits of the ticket in service. */
    uint32_t _pad0;
    void *item;             /**< Payload pointer. */
    unsigned char _pad1[RE_MPMC_SLOT_SIZE - 2u * 4u - 8u]; /**< Cache line fill. */
} ReMpmcSlot;

/** Slot must occupy exactly one cache line. */
_Static_assert(sizeof(ReMpmcSlot) == RE_MPMC_SLOT_SIZE, "ReMpmcSlot must be one cache line");

/**
 * @brief Lock-free MPMC bounded ring buffer of pointers (Vyukov design).
 *
 * Producers and consumers reserve monotonically increasing tickets with
 * relaxed fetch-adds; each cell carries the ticket it is serving, which
 * guarantees exactly-once delivery and FIFO order under any number of
 * concurrent producers and consumers (weak-memory proven by the GenMC
 * harness). Capacity must be a power of two and at most 2^31 (the ticket
 * word is 32 bits). The struct is cache-line padded.
 */
typedef struct ReMpmcRing {
    _Atomic uint64_t head; /**< Next ticket a producer will reserve. */
    _Atomic uint64_t tail; /**< Next ticket a consumer will reserve. */
    uint32_t capacity;     /**< Number of slots (power of two, <= 2^31). */
    uint32_t mask;         /**< capacity - 1, ticket-to-slot mapping. */
    ReMpmcSlot *slots;     /**< Slot storage (capacity * RE_MPMC_SLOT_SIZE bytes). */
    unsigned char _pad[RE_CACHE_LINE - 2u * 8u - 2u * 4u - 8u]; /**< Cache line fill. */
} ReMpmcRing;

/** Size of the ring struct must equal one cache line. */
_Static_assert(sizeof(ReMpmcRing) == RE_CACHE_LINE, "ReMpmcRing must occupy one cache line");

/**
 * @brief Initializes a ring over caller-provided slot storage.
 * @param r Ring to initialize.
 * @param slots_mem Storage for capacity * RE_MPMC_SLOT_SIZE bytes.
 * @param capacity Number of slots, power of two, at most 2^31.
 * @return True on success.
 */
bool re_mpmc_init(ReMpmcRing *r, void *slots_mem, uint32_t capacity);

/**
 * @brief Publishes an item into the ring.
 *
 * Blocks briefly (yielding) only when the ring is momentarily full.
 * @param r Ring.
 * @param item Non-NULL payload pointer.
 * @return False when the ring is full.
 */
bool re_mpmc_push(ReMpmcRing *r, void *item);

/**
 * @brief Consumes the oldest item from the ring.
 * @param r Ring.
 * @param out Receives the payload pointer.
 * @return False when the ring is empty.
 */
bool re_mpmc_pop(ReMpmcRing *r, void **out);

/**
 * @return Ring capacity.
 */
uint32_t re_mpmc_capacity(const ReMpmcRing *r);

#endif /* RE_MEM_H */
