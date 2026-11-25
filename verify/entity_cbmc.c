/**
 * @file entity_cbmc.c
 * @brief CBMC proof: generational handles defeat ABA. For any symbolic
 *        sequence of spawn/despawn/add/remove operations, a handle that
 *        was ever observed dead stays dead forever — even after its slot
 *        is recycled by a later spawn.
 *
 * Under __CPROVER__ the ECS caps every table at 8 entries (no growth),
 * which keeps the state space tractable. Run:
 *   cbmc entity_cbmc.c --unwind 32 --unwinding-assertions \
 *        --pointer-overflow-check --signed-overflow-check \
 *        --unsigned-overflow-check
 */
#include "harness_stub.h"

#include "../src/re_core.c"
#include "harness_stub.h"

#include "../src/re_mem.c"
#include "harness_stub.h"

#include "../src/re_ecs.c"

#include "harness_stub.h"

#include "../src/re_ecs.h"

/** ECS backing memory (small world for the proof). */
static unsigned char world_mem[8192];

/** Component slots registered by the harness. */
static ReComponentId comp_a, comp_b;

/** Handle log: every handle ever issued by spawn. */
#define MAX_HANDLES 16u
static ReEntity handles[MAX_HANDLES];
static uint32_t handle_count = 0u;
/* dead_forever[i]: alive(handles[i]) returned false at least once. */
static bool dead_forever[MAX_HANDLES];
/* despawned[i]: despawn was called on handles[i] while alive. */
static bool despawned[MAX_HANDLES];

static void log_spawn(ReEcs *ecs) {
    ReEntity e = re_ecs_spawn(ecs);
    if (e != RE_ENTITY_NULL && handle_count < MAX_HANDLES) {
        handles[handle_count] = e;
        dead_forever[handle_count] = false;
        despawned[handle_count] = false;
        handle_count++;
        __CPROVER_assert(re_ecs_alive(ecs, e), "fresh handle is alive");
    }
}

static void log_despawn(ReEcs *ecs, uint32_t idx) {
    ReEntity e = handles[idx];
    if (re_ecs_alive(ecs, e)) {
        re_ecs_despawn(ecs, e);
        despawned[idx] = true;
        __CPROVER_assert(!re_ecs_alive(ecs, e), "handle dead after despawn");
    }
}

int main(void) {
    ReEcs ecs;
    bool ok = re_ecs_init(&ecs, world_mem, sizeof world_mem);
    __CPROVER_assert(ok, "ecs init succeeds");
    comp_a = re_ecs_register(&ecs, 8u);
    comp_b = re_ecs_register(&ecs, 8u);
    __CPROVER_assert(comp_a != RE_COMPONENT_NONE && comp_b != RE_COMPONENT_NONE,
                     "components registered");

    /* Symbolic operation sequence: spawn / despawn / mutate / probe. */
    for (int step = 0; step < 24; step++) {
        unsigned op = nondet_uint();
        __CPROVER_assume(op <= 3u);
        if (op == 0u) {
            log_spawn(&ecs);
        } else if (handle_count > 0u) {
            uint32_t idx = nondet_uint();
            __CPROVER_assume(idx < handle_count);
            if (op == 1u) {
                log_despawn(&ecs, idx);
            } else if (op == 2u && re_ecs_alive(&ecs, handles[idx])) {
                unsigned c = nondet_uint();
                __CPROVER_assume(c <= 1u);
                bool added = re_ecs_add(&ecs, handles[idx], c == 0u ? comp_a : comp_b);
                __CPROVER_assert(added, "add on a live entity succeeds");
                void *p = re_ecs_get(&ecs, handles[idx], c == 0u ? comp_a : comp_b);
                __CPROVER_assert(p != NULL, "get returns the component");
            } else {
                bool alive_now = re_ecs_alive(&ecs, handles[idx]);
                if (!alive_now) {
                    /* Record the tombstone: this handle must stay dead. */
                    dead_forever[idx] = true;
                }
                if (alive_now && despawned[idx]) {
                    /* Impossible: despawned handles never revive. */
                    __CPROVER_assert(0, "a despawned handle revived (ABA!)");
                }
            }
        }
        /* Slot recycling must never resurrect an old handle. */
        for (uint32_t i = 0u; i < handle_count; i++) {
            if (dead_forever[i] || despawned[i]) {
                __CPROVER_assert(!re_ecs_alive(&ecs, handles[i]),
                                 "stale handle stays dead after slot reuse");
            }
        }
    }
    return 0;
}
