/**
 * @file ring_fifo_genmc.c
 * @brief GenMC proof: the ring preserves FIFO order. One producer pushes
 *        items 1..N in increasing order; each consumer must observe its
 *        received sequence strictly increasing (its own subsequence of
 *        the global FIFO order).
 *
 * Run: genmc -unroll 16 ring_fifo_genmc.c (or via verify.sh)
 */
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

#include "../src/re_core.h"
#include "../src/re_mem.h"

/* See ring_genmc.c: the model checker provides no sched_yield; the
   harness supplies a no-op re_thread_yield and a minimal re_panic. */
void re_thread_yield(void) {
}

void re_panic(const char *file, int line, const char *message) {
    (void)file;
    (void)line;
    (void)message;
    assert(0 && "re_panic called");
}

#include "../src/re_mem.c"

#define RING_CAP 2u
#define NCONS 2u
#define NITEMS 8u

static ReMpmcRing g_ring;
static unsigned char g_slots[RING_CAP * RE_MPMC_SLOT_SIZE];
static _Atomic int g_received = 0;

static void *consumer(void *arg) {
    (void)arg;
    uint32_t last = 0u;
    for (;;) {
        void *out = NULL;
        if (re_mpmc_pop(&g_ring, &out)) {
            uint32_t k = (uint32_t)(uintptr_t)out;
            assert(k >= 1u && k <= NITEMS);
            assert(k > last); /* per-consumer FIFO order */
            last = k;
            atomic_fetch_add_explicit(&g_received, 1, memory_order_relaxed);
        } else if (atomic_load_explicit(&g_received, memory_order_acquire) == NITEMS) {
            break;
        }
    }
    return NULL;
}

int main(void) {
    pthread_t ct[NCONS];
    bool ok = re_mpmc_init(&g_ring, g_slots, RING_CAP);
    assert(ok);
    for (uintptr_t i = 0; i < NCONS; i++) {
        assert(pthread_create(&ct[i], NULL, consumer, NULL) == 0);
    }
    for (uint32_t k = 1u; k <= NITEMS; k++) {
        while (!re_mpmc_push(&g_ring, (void *)(uintptr_t)k)) {
        }
    }
    for (uintptr_t i = 0; i < NCONS; i++) {
        assert(pthread_join(ct[i], NULL) == 0);
    }
    assert(atomic_load_explicit(&g_received, memory_order_relaxed) == NITEMS);
    return 0;
}
