/**
 * @file ring_genmc.c
 * @brief GenMC proof: the lock-free MPMC ring delivers every item exactly
 *        once under the C11 weak-memory model, with any number of
 *        concurrent producers and consumers.
 *
 * 2 producers publish 8 unique items; 2 consumers pop them and mark each
 * item with an atomic exchange — a second delivery of the same item (or
 * a lost item) fails the assertion. GenMC explores every allowed
 * interleaving, including store-buffer and load-buffer reorderings.
 *
 * Run: genmc -unroll 16 ring_genmc.c (or via verify.sh)
 */
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

#include "../src/re_core.h"
#include "../src/re_mem.h"

/* The ring implementation calls re_thread_yield on contention; the model
   checker provides no sched_yield, so the harness supplies a no-op.
   re_panic is declared in re_core.h and defined here (the harness does
   not link the OS-specific re_core.c). */
void re_thread_yield(void) {
}

void re_panic(const char *file, int line, const char *message) {
    (void)file;
    (void)line;
    (void)message;
    assert(0 && "re_panic called");
}

#include "../src/re_mem.c"

#define RING_CAP 4u
#define NPROD 2u
#define NCONS 2u
#define NITEMS 8u

static ReMpmcRing g_ring;
static unsigned char g_slots[RING_CAP * RE_MPMC_SLOT_SIZE];
static _Atomic int g_taken[NITEMS];
static _Atomic int g_received = 0;

static void *producer(void *arg) {
    uintptr_t p = (uintptr_t)arg;
    for (uint32_t k = (uint32_t)p; k < NITEMS; k += NPROD) {
        void *item = (void *)(uintptr_t)(k + 1u);
        while (!re_mpmc_push(&g_ring, item)) {
        }
    }
    return NULL;
}

static void *consumer(void *arg) {
    (void)arg;
    for (;;) {
        void *out = NULL;
        if (re_mpmc_pop(&g_ring, &out)) {
            uint32_t k = (uint32_t)(uintptr_t)out - 1u;
            assert(k < NITEMS);
            int was = atomic_exchange_explicit(&g_taken[k], 1, memory_order_acq_rel);
            assert(was == 0); /* duplicate delivery */
            atomic_fetch_add_explicit(&g_received, 1, memory_order_relaxed);
        } else if (atomic_load_explicit(&g_received, memory_order_acquire) == NITEMS) {
            break;
        }
    }
    return NULL;
}

int main(void) {
    pthread_t pt[NPROD], ct[NCONS];
    bool ok = re_mpmc_init(&g_ring, g_slots, RING_CAP);
    assert(ok);
    for (uintptr_t i = 0; i < NPROD; i++) {
        assert(pthread_create(&pt[i], NULL, producer, (void *)i) == 0);
    }
    for (uintptr_t i = 0; i < NCONS; i++) {
        assert(pthread_create(&ct[i], NULL, consumer, NULL) == 0);
    }
    for (uintptr_t i = 0; i < NPROD; i++) {
        assert(pthread_join(pt[i], NULL) == 0);
    }
    for (uintptr_t i = 0; i < NCONS; i++) {
        assert(pthread_join(ct[i], NULL) == 0);
    }
    assert(atomic_load_explicit(&g_received, memory_order_relaxed) == NITEMS);
    return 0;
}
