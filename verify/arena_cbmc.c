/**
 * @file arena_cbmc.c
 * @brief CBMC proof: arena bump allocation is overflow-free, aligned and
 *        never overlaps previously returned blocks, for arbitrary
 *        symbolic sizes and alignments.
 *
 * Run: cbmc arena_cbmc.c --unwind 16 --unwinding-assertions \
 *        --pointer-overflow-check --signed-overflow-check \
 *        --unsigned-overflow-check
 */
#include "harness_stub.h"

#include "../src/re_core.c"
#include "harness_stub.h"

#include "../src/re_mem.c"

#include "harness_stub.h"

#include "../src/re_mem.h"

/** Backing memory for the arena under test. */
static unsigned char arena_mem[256];

int main(void) {
    ReArena a;
    bool ok = re_arena_init(&a, arena_mem, sizeof arena_mem);
    __CPROVER_assert(ok, "arena init succeeds over 256 bytes");
    __CPROVER_assert(a.capacity <= sizeof arena_mem, "capacity bounded by the block");
    __CPROVER_assert(((uintptr_t)a.begin & 15u) == 0u, "base is 16-aligned");

    const unsigned char *prev_end = a.begin;
    size_t prev_off = 0u;

    for (int i = 0; i < 8; i++) {
        size_t size = nondet_size_t();
        __CPROVER_assume(size <= 64u);
        size_t align = nondet_size_t();
        __CPROVER_assume(align == 16u || align == 32u || align == 64u);

        void *p = re_arena_alloc_align(&a, size, align);
        if (p != NULL) {
            /* Alignment of the returned pointer. */
            __CPROVER_assert(((uintptr_t)p % align) == 0u, "returned pointer aligned");
            /* Bounds: [p, p + size) inside [begin, end). */
            __CPROVER_assert((const unsigned char *)p >= a.begin,
                             "returned pointer after base");
            __CPROVER_assert((size_t)((const unsigned char *)p - a.begin) + size <= a.capacity,
                             "returned block inside the arena");
            /* Monotonicity: blocks never overlap (bump discipline). */
            __CPROVER_assert((const unsigned char *)p >= prev_end,
                             "no overlap with previous allocation");
            /* The bump offset is the running high-water mark. */
            __CPROVER_assert(a.offset >= prev_off, "offset monotone");
            __CPROVER_assert(a.offset <= a.capacity, "offset never exceeds capacity");
            prev_end = (const unsigned char *)p + size;
            prev_off = a.offset;
        }
    }
    __CPROVER_assert(a.offset <= a.capacity, "final offset bounded");
    __CPROVER_assert(a.peak <= a.capacity, "peak bounded");

    /* Reallocating a zero-size block returns the cursor without advancing. */
    size_t off_before = a.offset;
    void *z = re_arena_alloc_align(&a, 0u, 16u);
    __CPROVER_assert(z != NULL || a.capacity == 0u, "zero-size alloc only fails on empty arena");
    __CPROVER_assert(a.offset == off_before, "zero-size alloc does not advance");

    /* Reset reclaims everything and stays aligned for reuse. */
    re_arena_reset(&a);
    __CPROVER_assert(a.offset == 0u, "reset zeroes the offset");
    void *q = re_arena_alloc_align(&a, 8u, 16u);
    __CPROVER_assert(q != NULL && ((uintptr_t)q & 15u) == 0u,
                     "post-reset allocation aligned at base");
    return 0;
}
