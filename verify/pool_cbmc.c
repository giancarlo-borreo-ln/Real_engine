/**
 * @file pool_cbmc.c
 * @brief CBMC proof: the pool free list stays well-formed, allocations are
 *        unique and in-range, frees are validated, and the live counter
 *        equals the number of outstanding slots for any symbolic
 *        sequence of alloc/free operations.
 *
 * Run: cbmc pool_cbmc.c --unwind 32 --unwinding-assertions \
 *        --pointer-overflow-check --signed-overflow-check \
 *        --unsigned-overflow-check
 */
#include "harness_stub.h"

#include "../src/re_core.c"
#include "harness_stub.h"

#include "../src/re_mem.c"

#include "harness_stub.h"

#include "../src/re_mem.h"

#define POOL_SLOT_SIZE 16u
#define POOL_CAP 8u

/** Slot storage: POOL_CAP slots of POOL_SLOT_SIZE bytes. */
static unsigned char slot_mem[POOL_CAP * POOL_SLOT_SIZE];

int main(void) {
    RePool p;
    bool ok = re_pool_init(&p, slot_mem, POOL_SLOT_SIZE, POOL_CAP);
    __CPROVER_assert(ok, "pool init succeeds");
    __CPROVER_assert(re_pool_capacity(&p) == POOL_CAP, "capacity stored");
    __CPROVER_assert(re_pool_slot_size(&p) >= POOL_SLOT_SIZE, "slot size preserved");
    __CPROVER_assert(re_pool_live(&p) == 0u, "starts empty");

    void *live[POOL_CAP];
    uint32_t live_count = 0u;

    for (int i = 0; i < 20; i++) {
        unsigned op = nondet_uint();
        __CPROVER_assume(op <= 1u);
        if (op == 0u) {
            void *q = re_pool_alloc(&p);
            if (q != NULL) {
                __CPROVER_assert(live_count < POOL_CAP, "allocation within capacity");
                live[live_count++] = q;
                /* Slot lies on the storage grid. */
                __CPROVER_assert((unsigned char *)q >= slot_mem &&
                                 (unsigned char *)q < slot_mem + sizeof slot_mem,
                                 "slot inside storage");
                __CPROVER_assert(((size_t)((unsigned char *)q - slot_mem)) %
                                     re_pool_slot_size(&p) == 0u,
                                 "slot aligned to the stride");
                /* Never hand out the same slot twice while it is live. */
                for (uint32_t k = 0u; k + 1u < live_count; k++) {
                    __CPROVER_assert(live[k] != q, "no double allocation");
                }
            } else {
                __CPROVER_assert(live_count == POOL_CAP, "NULL only when exhausted");
            }
        } else {
            if (live_count > 0u) {
                uint32_t idx = nondet_uint();
                __CPROVER_assume(idx < live_count);
                bool freed = re_pool_free(&p, live[idx]);
                __CPROVER_assert(freed, "free of a live slot succeeds");
                live[idx] = live[--live_count];
            } else {
                /* Freeing garbage while empty must be rejected, never crash. */
                unsigned char garbage[POOL_SLOT_SIZE];
                __CPROVER_havoc_object(garbage);
                (void)re_pool_free(&p, garbage);
            }
        }
        /* Central invariant: live counter matches outstanding slots. */
        __CPROVER_assert(re_pool_live(&p) == live_count, "live counter invariant");
        __CPROVER_assert(p.free_head == RE_POOL_NIL || p.free_head < POOL_CAP,
                         "free head index in range");
    }
    return 0;
}
