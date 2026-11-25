/**
 * @file re_scheduler.h
 * @brief DAG job scheduler: conflict resolution into parallel waves plus
 *        a work-stealing thread pool.
 *
 * Jobs declare read and write component masks. Two jobs conflict when
 * one writes what the other reads or writes; conflicting jobs must run
 * in different waves. re_dag_plan computes a wave partition with Kahn's
 * algorithm: every job appears exactly once and no wave contains two
 * conflicting jobs (this property is what the Frama-C harness proves).
 *
 * The thread pool executes waves with per-worker Chase-Lev deques and
 * work stealing. Task execution order inside a wave is unspecified;
 * callers needing deterministic results must make tasks commutative or
 * sort their output afterwards.
 */
#ifndef RE_SCHEDULER_H
#define RE_SCHEDULER_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "re_core.h"
#include "re_thread.h"

/**
 * @brief One job in a job graph.
 */
typedef struct ReJobDesc {
    const char *name;   /**< Human-readable name (may be NULL). */
    ReMask read;        /**< Components the job reads. */
    ReMask write;       /**< Components the job writes. */
    void (*fn)(void *ctx); /**< Job body. */
    void *ctx;          /**< Argument passed to fn. */
} ReJobDesc;

/**
 * @brief True when the two jobs must not run in the same wave:
 *        either writes the same component, or one writes a component
 *        the other reads.
 * @param a First job.
 * @param b Second job.
 */
static inline bool re_jobs_conflict(const ReJobDesc *a, const ReJobDesc *b) {
    ReMask a_w = a->write, b_w = b->write;
    return re_mask_overlaps(a_w, b_w) ||
           re_mask_overlaps(a_w, b->read) ||
           re_mask_overlaps(b_w, a->read);
}

/**
 * @brief Computes the conflict-free wave partition of a job graph.
 *
 * Kahn's algorithm over dependency edges oriented by declaration order:
 * edge (i, j) exists for i < j exactly when re_jobs_conflict(i, j).
 * Jobs sharing a wave therefore never conflict, and a job that must run
 * after another (write-after-read, read-after-write, write-after-write)
 * must be declared after it. The orientation makes the edge graph a DAG,
 * so every job is placed in exactly one wave.
 *
 * @param jobs Job array (declaration order = dependency order).
 * @param count Number of jobs.
 * @param wave_of Receives the wave index of each job (caller array of count).
 * @param wave_sizes Receives the job count of each wave (caller array of count).
 * @return Number of waves (0 when count is 0).
 */
uint32_t re_dag_plan(const ReJobDesc *jobs, uint32_t count,
                     uint32_t *wave_of, uint32_t *wave_sizes);

/**
 * @brief Chase-Lev work-stealing deque of task pointers.
 *
 * bottom is atomic because thieves read it (acquire) while the owner
 * writes it: C11 data-race freedom requires an atomic object.
 */
typedef struct ReDeque {
    _Atomic size_t top; /**< Shared by thieves. */
    _Atomic size_t bottom; /**< Owner-written, thief-read. */
    void **tasks;       /**< Task buffer (cap entries, cap power of two). */
    size_t cap;         /**< Buffer capacity. */
} ReDeque;

/**
 * @brief Thread pool with one Chase-Lev deque per worker plus one for
 *        the submitting thread (deques[0]).
 */
typedef struct ReScheduler {
    uint32_t worker_count;  /**< Number of worker threads. */
    ReThread *threads;      /**< Worker handles. */
    ReDeque *deques;        /**< worker_count + 1 deques. */
    void ***deque_bufs;     /**< Per-deque task buffers (for growth). */
    ReMutex mutex;          /**< Guards the idle wait. */
    ReCond cond;            /**< Signaled when work arrives or finishes. */
    _Atomic uint32_t outstanding; /**< Tasks not yet completed. */
    _Atomic uint32_t shutdown_flag; /**< Set by re_scheduler_destroy. */
} ReScheduler;

/**
 * @brief Initializes the pool with the given worker count.
 *
 * worker_count 0 selects the number of logical CPUs minus one
 * (at least 1). All thread resources are released by
 * re_scheduler_destroy.
 *
 * @param s Scheduler to initialize.
 * @param worker_count Worker threads (0 = automatic).
 * @return True on success.
 */
bool re_scheduler_init(ReScheduler *s, uint32_t worker_count);

/**
 * @brief Stops and joins every worker and frees the pool resources.
 * @param s Scheduler.
 */
void re_scheduler_destroy(ReScheduler *s);

/**
 * @return Number of worker threads.
 */
uint32_t re_scheduler_workers(const ReScheduler *s);

/**
 * @brief Range task body: processes indices [begin, end).
 */
typedef void (*ReTaskFn)(void *ctx, uint32_t begin, uint32_t end);

/**
 * @brief Splits [0, count) into ranges and runs them on the pool.
 *
 * Blocks until every range is complete. The calling thread also helps
 * execute tasks. Task order is unspecified.
 *
 * @param s Scheduler.
 * @param fn Range body.
 * @param ctx Shared context.
 * @param count Number of items.
 * @param min_chunk Smallest range size handed to one task (> 0).
 * @return True on success.
 */
bool re_scheduler_parallel_for(ReScheduler *s, ReTaskFn fn, void *ctx,
                               uint32_t count, uint32_t min_chunk);

/**
 * @brief Plans a job graph into waves and executes them wave by wave.
 *
 * All jobs of a wave are enqueued before any of them runs, so
 * conflicting jobs can never overlap in time.
 *
 * @param s Scheduler.
 * @param jobs Job array.
 * @param count Number of jobs.
 * @return True on success.
 */
bool re_scheduler_run_jobs(ReScheduler *s, const ReJobDesc *jobs, uint32_t count);

#endif /* RE_SCHEDULER_H */
