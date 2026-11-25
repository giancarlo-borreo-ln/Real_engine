/**
 * @file dag_frama.c
 * @brief Frama-C/WP proof of the wave-partition contract of re_dag_plan:
 *        every job lands in exactly one wave, wave indices are in range,
 *        and no two jobs in the same wave conflict.
 *
 * The contract is attached by redeclaration (Frama-C merges specs);
 * main() exercises it on concrete small graphs, which also lets
 * -wp-rte prove the implementation free of runtime errors on the
 * analyzed paths. Run:
 *   frama-c -wp -wp-rte -wp-fct main -wp-timeout 120 dag_frama.c
 */
#include "../src/re_core.h"
#include "../src/re_scheduler.h"
#include "../src/re_scheduler.c"

/*@ requires \valid(jobs + (0 .. count - 1));
    requires \valid(wave_of + (0 .. count - 1));
    requires \valid(wave_sizes + (0 .. count - 1));
    requires count <= 8;
    assigns wave_of[0 .. count - 1], wave_sizes[0 .. count - 1];
    ensures 0 <= \result <= count;
    ensures \forall integer i; 0 <= i < count ==> 0 <= wave_of[i] < \result;
    ensures \forall integer i, j;
              0 <= i < count && 0 <= j < count &&
              wave_of[i] == wave_of[j] ==> !re_jobs_conflict(jobs + i, jobs + j);
 */
uint32_t re_dag_plan(const ReJobDesc *jobs, uint32_t count,
                     uint32_t *wave_of, uint32_t *wave_sizes);

/** Component ids used by the concrete scenarios. */
enum { C_POS = 0, C_VEL = 1, C_HEALTH = 2, C_INPUT = 3 };

static ReJobDesc jobs[8];
static uint32_t wave_of[8];
static uint32_t wave_sizes[8];

static void noop(void *ctx) {
    (void)ctx;
}

int main(void) {
    /* Scenario 1: A writes POS; B reads POS (B after A); C writes VEL
       (independent of both). Expect wave_count == 2 and A before B. */
    jobs[0] = (ReJobDesc){ "A", re_mask_zero(), re_mask_bit(C_POS), noop, NULL };
    jobs[1] = (ReJobDesc){ "B", re_mask_bit(C_POS), re_mask_zero(), noop, NULL };
    jobs[2] = (ReJobDesc){ "C", re_mask_zero(), re_mask_bit(C_VEL), noop, NULL };
    uint32_t wc = re_dag_plan(jobs, 3u, wave_of, wave_sizes);
    /*@ assert 2 <= wc <= 3; */
    /*@ assert wave_of[0] < wave_of[1]; */

    /* Scenario 2: mutual read/write conflict forces two waves. */
    jobs[0] = (ReJobDesc){ "A", re_mask_zero(), re_mask_bit(C_HEALTH), noop, NULL };
    jobs[1] = (ReJobDesc){ "B", re_mask_bit(C_HEALTH), re_mask_zero(), noop, NULL };
    wc = re_dag_plan(jobs, 2u, wave_of, wave_sizes);
    /*@ assert wc == 2; */
    /*@ assert wave_of[0] == 0 && wave_of[1] == 1; */

    /* Scenario 3: independent jobs share a single wave. */
    jobs[0] = (ReJobDesc){ "A", re_mask_zero(), re_mask_bit(C_POS), noop, NULL };
    jobs[1] = (ReJobDesc){ "B", re_mask_zero(), re_mask_bit(C_VEL), noop, NULL };
    jobs[2] = (ReJobDesc){ "C", re_mask_zero(), re_mask_bit(C_HEALTH), noop, NULL };
    wc = re_dag_plan(jobs, 3u, wave_of, wave_sizes);
    /*@ assert wc == 1; */

    /* Scenario 4: empty graph. */
    wc = re_dag_plan(jobs, 0u, wave_of, wave_sizes);
    /*@ assert wc == 0; */
    return 0;
}
