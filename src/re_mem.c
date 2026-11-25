/**
 * @file re_mem.c
 * @brief Memory subsystem: frame arena, fixed-slot pool, lock-free MPMC ring.
 *
 * Contract notes that are not visible from re_mem.h:
 *  - Arena and pool are single-threaded and never call into the heap.
 *  - Runtime failures (null storage, zero size, out of memory, invalid
 *    pointer) are reported through the return value. Only a NULL allocator
 *    handle is a programming error and trips RE_ASSERT.
 *  - re_arena_alloc_align clamps an invalid alignment (not a power of two,
 *    or below 16) up to 16 instead of rejecting the request.
 *  - A zero-size arena allocation returns the current aligned cursor without
 *    advancing it.
 */

#include "re_mem.h"

#include <string.h>

#include "re_thread.h"

/** Minimum arena alignment and default allocation alignment (bytes). */
#define RE_ARENA_ALIGN ((size_t)16)

/** Smallest pool stride able to hold the 4-byte free-list link. */
#define RE_POOL_MIN_STRIDE ((size_t)4)

/* ------------------------------------------------------------------------- */
/* ReArena                                                                    */
/* ------------------------------------------------------------------------- */

bool re_arena_init(ReArena *a, void *mem, size_t size) {
    RE_ASSERT(a != NULL);
    a->begin = NULL;
    a->end = NULL;
    a->offset = 0u;
    a->capacity = 0u;
    a->peak = 0u;
    if (mem == NULL || size == 0u) {
        return false;
    }
    uintptr_t base = (uintptr_t)mem;
    size_t aligned = re_align_up((size_t)base, RE_ARENA_ALIGN);
    /* re_align_up saturates on overflow, which can move the rounded address
       below the block; subtracting then would wrap, so reject explicitly. */
    if (aligned < (size_t)base) {
        return false;
    }
    size_t skip = aligned - (size_t)base;
    if (skip >= size) {
        return false; /* alignment alone consumes the whole block */
    }
    a->begin = (unsigned char *)mem + skip;
    a->capacity = size - skip;
    a->end = a->begin + a->capacity;
    return true;
}

void *re_arena_alloc_align(ReArena *a, size_t size, size_t align) {
    RE_ASSERT(a != NULL);
    if (!re_is_pow2_size(align) || align < RE_ARENA_ALIGN) {
        align = RE_ARENA_ALIGN;
    }
    if (a->begin == NULL) {
        return NULL;
    }
    size_t offset = a->offset;
    /* Alignment is a property of the returned address, not of the offset:
       begin is only 16-aligned, so a 64-byte request may need extra padding. */
    size_t misalign = (size_t)(((uintptr_t)a->begin + offset) % align);
    if (misalign != 0u) {
        size_t pad = align - misalign;
        if (pad > a->capacity - offset) {
            return NULL;
        }
        offset += pad;
    }
    if (offset > a->capacity || size > a->capacity - offset) {
        return NULL;
    }
    a->offset = offset + size;
    if (a->offset > a->peak) {
        a->peak = a->offset;
    }
    return a->begin + offset;
}

void *re_arena_alloc(ReArena *a, size_t size) {
    return re_arena_alloc_align(a, size, RE_ARENA_ALIGN);
}

void re_arena_reset(ReArena *a) {
    RE_ASSERT(a != NULL);
    a->offset = 0u;
}

size_t re_arena_used(const ReArena *a) {
    RE_ASSERT(a != NULL);
    return a->offset;
}

size_t re_arena_capacity(const ReArena *a) {
    RE_ASSERT(a != NULL);
    return a->capacity;
}

void *re_arena_base(const ReArena *a) {
    RE_ASSERT(a != NULL);
    return a->begin;
}

/* ------------------------------------------------------------------------- */
/* RePool                                                                     */
/* ------------------------------------------------------------------------- */

/** Link value written into a live slot by re_pool_alloc. Distinct from every
 *  free-list link (indices < capacity and RE_POOL_NIL), so re_pool_free can
 *  detect double frees and frees of never-allocated slots in O(1). */
#define RE_POOL_LIVE UINT32_C(0xFFFFFFFE)

/** @brief Address of slot `index` (no bounds check; caller validates). */
static unsigned char *pool_slot_at(const RePool *p, uint32_t index) {
    return (unsigned char *)p->mem + (size_t)index * p->slot_size;
}

/**
 * @brief Reads the next-free link stored in a slot.
 *
 * memcpy avoids both alignment and strict-aliasing requirements: the caller
 * supplies raw bytes, so a slot is not guaranteed to be 4-byte aligned.
 */
static uint32_t pool_next(const unsigned char *slot) {
    uint32_t next;
    memcpy(&next, slot, sizeof next);
    return next;
}

/** @brief Writes the next-free link into a slot (see pool_next). */
static void pool_set_next(unsigned char *slot, uint32_t next) {
    memcpy(slot, &next, sizeof next);
}

bool re_pool_init(RePool *p, void *mem, size_t slot_size, uint32_t capacity) {
    RE_ASSERT(p != NULL);
    p->mem = NULL;
    p->slot_size = 0u;
    p->capacity = 0u;
    p->free_head = RE_POOL_NIL;
    p->live = 0u;
    if (mem == NULL || slot_size == 0u || capacity == 0u) {
        return false;
    }
    size_t stride = slot_size < RE_POOL_MIN_STRIDE ? RE_POOL_MIN_STRIDE : slot_size;
    if (stride > SIZE_MAX / (size_t)capacity) {
        return false; /* storage size is not representable */
    }
    p->mem = mem;
    p->slot_size = stride;
    p->capacity = capacity;
    p->free_head = 0u;
    for (uint32_t i = 0u; i < capacity; i++) {
        uint32_t next = (i + 1u == capacity) ? RE_POOL_NIL : i + 1u;
        pool_set_next(pool_slot_at(p, i), next);
    }
    return true;
}

size_t re_pool_slot_size(const RePool *p) {
    RE_ASSERT(p != NULL);
    return p->slot_size;
}

void *re_pool_alloc(RePool *p) {
    RE_ASSERT(p != NULL);
    uint32_t index = p->free_head;
    if (index == RE_POOL_NIL || p->mem == NULL) {
        return NULL;
    }
    RE_ASSERT(index < p->capacity);
    unsigned char *slot = pool_slot_at(p, index);
    p->free_head = pool_next(slot);
    /* Sentinel: a live slot's link holds RE_POOL_LIVE, so a later double
       free (or a free of a never-allocated slot, whose init link is part
       of the free chain) is detectable in O(1). */
    pool_set_next(slot, RE_POOL_LIVE);
    p->live++;
    return slot;
}

bool re_pool_free(RePool *p, void *ptr) {
    RE_ASSERT(p != NULL);
    /* live == 0 means no slot can legitimately be released. */
    if (ptr == NULL || p->mem == NULL || p->live == 0u) {
        return false;
    }
    uintptr_t base = (uintptr_t)p->mem;
    uintptr_t addr = (uintptr_t)ptr;
    if (addr < base) {
        return false;
    }
    size_t offset = (size_t)(addr - base);
    if (offset >= p->slot_size * (size_t)p->capacity) {
        return false;
    }
    if (offset % p->slot_size != 0u) {
        return false; /* interior of a slot */
    }
    uint32_t index = (uint32_t)(offset / p->slot_size);
    RE_ASSERT(index < p->capacity);
    unsigned char *slot = pool_slot_at(p, index);
    /* Liveness sentinel (see re_pool_alloc): a slot whose link is not the
       sentinel is already free (double free) or was never allocated. */
    if (pool_next(slot) != RE_POOL_LIVE) {
        return false;
    }
    pool_set_next(slot, p->free_head);
    p->free_head = index;
    p->live--;
    return true;
}

uint32_t re_pool_capacity(const RePool *p) {
    RE_ASSERT(p != NULL);
    return p->capacity;
}

uint32_t re_pool_live(const RePool *p) {
    RE_ASSERT(p != NULL);
    return p->live;
}

/* ------------------------------------------------------------------------- */
/* ReMpmcRing                                                                 */
/* ------------------------------------------------------------------------- */

bool re_mpmc_init(ReMpmcRing *r, void *slots_mem, uint32_t capacity) {
    RE_ASSERT(r != NULL);
    atomic_init(&r->head, 0u);
    atomic_init(&r->tail, 0u);
    r->capacity = 0u;
    r->mask = 0u;
    r->slots = NULL;
    if (slots_mem == NULL || capacity == 0u || !re_is_pow2_u32(capacity)) {
        return false;
    }
    /* The ticket word is 32 bits: in-flight tickets must stay distinct. */
    if (capacity > 0x80000000u) {
        return false;
    }
    /* Slot storage must satisfy the slot alignment: reject rather than
       format a misaligned pointer and reach undefined behavior later. */
    if ((uintptr_t)slots_mem % (uintptr_t)_Alignof(ReMpmcSlot) != 0u) {
        return false;
    }
    r->capacity = capacity;
    r->mask = capacity - 1u;
    r->slots = (ReMpmcSlot *)slots_mem;
    for (uint32_t i = 0u; i < capacity; i++) {
        atomic_init(&r->slots[i].state, i);
        r->slots[i].item = NULL;
    }
    return true;
}

bool re_mpmc_push(ReMpmcRing *r, void *item) {
    RE_ASSERT(r != NULL);
    RE_ASSERT(r->slots != NULL);
    /* Claim a ticket only after observing room, so a failed claim never
       has to be rolled back while a consumer may already wait on the cell. */
    uint64_t seq = atomic_load_explicit(&r->head, memory_order_relaxed);
    for (;;) {
        uint64_t tail = atomic_load_explicit(&r->tail, memory_order_relaxed);
        if (seq >= tail && seq - tail >= (uint64_t)r->capacity) {
            return false; /* full at this linearization point */
        }
        if (atomic_compare_exchange_weak_explicit(&r->head, &seq, seq + 1u,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) {
            break; /* a failed exchange refreshes seq to the current head */
        }
    }
    ReMpmcSlot *slot = &r->slots[(size_t)(seq & (uint64_t)r->mask)];
    /* Vyukov protocol: state == seq means the consumer of ticket
       seq - capacity has finished reading this cell. */
    while (atomic_load_explicit(&slot->state, memory_order_acquire) != (uint32_t)seq) {
        re_thread_yield();
    }
    slot->item = item;
    atomic_store_explicit(&slot->state, (uint32_t)(seq + 1u), memory_order_release);
    return true;
}

bool re_mpmc_pop(ReMpmcRing *r, void **out) {
    RE_ASSERT(r != NULL);
    RE_ASSERT(out != NULL);
    RE_ASSERT(r->slots != NULL);
    uint64_t cell = atomic_load_explicit(&r->tail, memory_order_relaxed);
    for (;;) {
        uint64_t head = atomic_load_explicit(&r->head, memory_order_relaxed);
        if (cell >= head) {
            return false; /* empty at this linearization point */
        }
        if (atomic_compare_exchange_weak_explicit(&r->tail, &cell, cell + 1u,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) {
            break; /* cell claimed exclusively; tail may not overshoot head */
        }
    }
    ReMpmcSlot *slot = &r->slots[(size_t)(cell & (uint64_t)r->mask)];
    /* state == cell + 1 means the producer published the payload. */
    while (atomic_load_explicit(&slot->state, memory_order_acquire) !=
           (uint32_t)(cell + 1u)) {
        re_thread_yield();
    }
    *out = slot->item;
    /* Release the cell for the producer of ticket cell + capacity. */
    atomic_store_explicit(&slot->state, (uint32_t)(cell + (uint64_t)r->capacity),
                          memory_order_release);
    return true;
}

uint32_t re_mpmc_capacity(const ReMpmcRing *r) {
    RE_ASSERT(r != NULL);
    return r->capacity;
}
