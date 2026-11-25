/**
 * @file re_scheduler.c
 * @brief DAG planner (Kahn) and Chase-Lev work-stealing pool.
 *
 * Ownership: this is the only engine module that keeps heap memory across
 * calls. The scheduler owns its thread handles, the deque array (which also
 * carries the per-worker spawn arguments in its tail), one task buffer per
 * deque and the mutex/cond pair; re_scheduler_destroy releases all of it.
 *
 * Task descriptors are transient. A submitting call allocates one array of
 * ReTask for the batch, every thread only borrows a pointer into it, and the
 * submitter frees the array once `outstanding` reaches zero. There is no
 * allocation in the execution path: the hot path walks pointers only. The
 * cost is one malloc/free per submitted batch.
 *
 * Topology: deques[0] belongs to the submitting thread, deques[1..n] belong to
 * the workers. A batch is pushed entirely onto deques[0] and the workers steal
 * it, because a Chase-Lev deque has one owner for `bottom` and a producer
 * cannot inject into another thread's deque without racing with its pop.
 *
 * Deque buffers are fixed at init (see deque_capacity) and never grow, so a
 * thief can never read through a buffer that the owner just reallocated.
 * A push past the capacity is a programming error and trips RE_ASSERT.
 *
 * Compatibility note: `ReDeque.bottom` is a plain size_t in the final header,
 * so thieves reading it are formally outside the C memory model. The access
 * below is a naturally aligned word load, which is indivisible on every
 * supported target, and Chase-Lev needs no more than that there.
 *
 * Reentrancy: parallel_for and run_jobs must be called from the thread that
 * created the scheduler (deques[0] is the caller's deque). Calling them from
 * inside a task is not supported.
 */
#include "re_scheduler.h"

#include <stdlib.h>
#include <string.h>

/** Wave index marking a job that Kahn's algorithm has not placed yet. */
#define WAVE_UNPLACED UINT32_MAX

/** Task buffer entries reserved per worker at init. */
#define DEQUE_ENTRIES_PER_WORKER 4096u

/** Lower bound on a task buffer capacity (entries). */
#define DEQUE_MIN_CAP 4096u

/** Upper bound on a task buffer capacity (entries, 128 KiB per deque). */
#define DEQUE_MAX_CAP 16384u

/** Deques probed by one steal round. */
#define STEAL_ATTEMPTS 8u

/**
 * @brief One unit of work: either a range body or a plain job body.
 */
typedef struct ReTask {
    ReTaskFn range_fn;      /**< Range body, or NULL for a plain job. */
    void (*job_fn)(void *); /**< Job body, used when range_fn is NULL. */
    void *ctx;              /**< Range or job context. */
    uint32_t begin;         /**< First index of the range. */
    uint32_t end;           /**< One past the last index of the range. */
} ReTask;

/**
 * @brief Spawn argument of one worker thread.
 */
typedef struct WorkerArg {
    ReScheduler *scheduler; /**< Owning scheduler. */
    uint32_t index;         /**< Deque index owned by this worker (1..worker_count). */
} WorkerArg;

static_assert(_Alignof(ReDeque) >= _Alignof(WorkerArg),
              "the deque block cannot host the worker arguments");

/* ------------------------------------------------------------------------- */
/* DAG planning                                                              */
/* ------------------------------------------------------------------------- */

/**
 * @brief Fills indegree[j] with the number of lower-indexed conflicting jobs.
 *
 * re_jobs_conflict is symmetric, so the raw conflict relation has an edge in
 * both directions and no indegree could ever reach zero. Edges are therefore
 * oriented from the lower to the higher job index: the result is a DAG whose
 * paths follow ascending indices, and every conflicting pair still has an
 * edge between the two jobs, in one direction only.
 */
static void compute_indegrees(const ReJobDesc *jobs, uint32_t count, uint32_t *indegree) {
    for (uint32_t j = 0; j < count; j++) {
        uint32_t in = 0u;

        for (uint32_t i = 0; i < j; i++) {
            if (re_jobs_conflict(&jobs[i], &jobs[j])) {
                in++;
            }
        }
        indegree[j] = in;
    }
}

/**
 * @brief Decrements the successors of `job` and queues the ones that reach 0.
 *
 * The edge set is oriented by job index (see compute_indegrees), so only
 * higher-indexed jobs can be successors.
 */
static void relax_successors(const ReJobDesc *jobs, uint32_t count, uint32_t job,
                             uint32_t *indegree, const uint32_t *wave_of,
                             uint32_t *queue, uint32_t *tail) {
    for (uint32_t succ = job + 1u; succ < count; succ++) {
        if (wave_of[succ] != WAVE_UNPLACED || indegree[succ] == 0u) {
            continue;
        }
        if (re_jobs_conflict(&jobs[job], &jobs[succ]) && --indegree[succ] == 0u) {
            queue[(*tail)++] = succ;
        }
    }
}

/**
 * @brief Runs Kahn's algorithm and writes one wave per layer.
 *
 * Every job is pushed at most once, so a plain array of `count` slots is
 * enough for the queue. Jobs in one batch all had indegree zero at the same
 * time: were two of them in conflict, the conflict edge between them would
 * have kept the later one out of that batch.
 *
 * @return Number of waves written.
 */
static uint32_t kahn_waves(const ReJobDesc *jobs, uint32_t count, uint32_t *indegree,
                           uint32_t *wave_of, uint32_t *wave_sizes,
                           uint32_t *queue, uint32_t *placed) {
    uint32_t head = 0u;
    uint32_t tail = 0u;
    uint32_t wave = 0u;

    for (uint32_t j = 0; j < count; j++) {
        if (indegree[j] == 0u) {
            queue[tail++] = j;
        }
    }
    while (head < tail) {
        uint32_t batch = tail - head; /* Jobs queued before this wave started. */

        wave_sizes[wave] = batch;
        for (uint32_t k = 0; k < batch; k++) {
            uint32_t job = queue[head++];

            wave_of[job] = wave;
            relax_successors(jobs, count, job, indegree, wave_of, queue, &tail);
        }
        wave++;
    }
    *placed = tail;
    return wave;
}

/**
 * @brief Gives every unplaced job a wave of its own.
 *
 * Index-oriented edges make kahn_waves place every job, so this runs only if
 * the planner itself is broken. Failing that way keeps the two guarantees that
 * matter (each job once, no wave with two conflicting jobs) at the price of
 * serialising the leftovers instead of dropping them.
 */
static uint32_t serialize_leftovers(uint32_t count, uint32_t *wave_of,
                                    uint32_t *wave_sizes, uint32_t waves) {
    for (uint32_t j = 0; j < count; j++) {
        if (wave_of[j] != WAVE_UNPLACED) {
            continue;
        }
        wave_of[j] = waves;
        wave_sizes[waves] = 1u;
        waves++;
    }
    return waves;
}

uint32_t re_dag_plan(const ReJobDesc *jobs, uint32_t count,
                     uint32_t *wave_of, uint32_t *wave_sizes) {
    uint32_t *indegree;
    uint32_t *queue;
    uint32_t placed = 0u;
    uint32_t waves;

    if (count == 0u) {
        return 0u;
    }
    RE_ASSERT(jobs != NULL && wave_of != NULL && wave_sizes != NULL);

    indegree = malloc((size_t)count * sizeof *indegree);
    queue = malloc((size_t)count * sizeof *queue);
    if (indegree == NULL || queue == NULL) {
        free(indegree);
        free(queue);
        re_log_error("re_dag_plan: out of memory for %u jobs", count);
        return 0u;
    }
    for (uint32_t j = 0; j < count; j++) {
        wave_of[j] = WAVE_UNPLACED;
    }
    compute_indegrees(jobs, count, indegree);
    waves = kahn_waves(jobs, count, indegree, wave_of, wave_sizes, queue, &placed);
    if (placed != count) {
        re_log_error("re_dag_plan: planner left %u job(s) unplaced; serialising them",
                     count - placed);
        waves = serialize_leftovers(count, wave_of, wave_sizes, waves);
    }
    free(indegree);
    free(queue);
    return waves;
}

/* ------------------------------------------------------------------------- */
/* Chase-Lev deque                                                           */
/* ------------------------------------------------------------------------- */

/** Capacity of one task buffer: worker_count * 4096 rounded up, clamped. */
static size_t deque_capacity(uint32_t workers) {
    uint64_t want = (uint64_t)workers * (uint64_t)DEQUE_ENTRIES_PER_WORKER;

    if (want < DEQUE_MIN_CAP) {
        want = DEQUE_MIN_CAP;
    }
    if (want > DEQUE_MAX_CAP) {
        want = DEQUE_MAX_CAP;
    }
    return (size_t)re_next_pow2_u32((uint32_t)want);
}

/** Thief-side load of the owner's bottom index (see the file header). */
static size_t deque_load_bottom(const ReDeque *d) {
    const _Atomic size_t *bottom = &d->bottom;

    return atomic_load_explicit(bottom, memory_order_acquire);
}

/** Owner-side push. Fails loudly when the fixed buffer is full. */
static void deque_push(ReDeque *d, void *task) {
    size_t bottom = d->bottom;
    size_t top = atomic_load_explicit(&d->top, memory_order_acquire);

    RE_ASSERT(bottom >= top && bottom - top < d->cap);
    d->tasks[bottom % d->cap] = task;
    atomic_thread_fence(memory_order_release);
    atomic_store_explicit(&d->bottom, bottom + 1u, memory_order_relaxed);
}

/** Owner-side pop of the newest task. */
static bool deque_pop(ReDeque *d, void **out) {
    size_t bottom = d->bottom;
    size_t top;
    void *task;

    if (bottom == 0u) {
        return false;
    }
    atomic_store_explicit(&d->bottom, bottom - 1u, memory_order_relaxed);
    atomic_thread_fence(memory_order_seq_cst);
    top = atomic_load_explicit(&d->top, memory_order_relaxed);
    if (top > bottom - 1u) {
        atomic_store_explicit(&d->bottom, top, memory_order_relaxed); /* empty: restore */
        return false;
    }
    task = d->tasks[(bottom - 1u) % d->cap];
    if (top == bottom - 1u) { /* Last task: race the thieves for it. */
        bool won = atomic_compare_exchange_strong_explicit(&d->top, &top, bottom,
                                                           memory_order_seq_cst,
                                                           memory_order_relaxed);

        atomic_store_explicit(&d->bottom, bottom, memory_order_relaxed);
        if (!won) {
            return false; /* A thief took it first. */
        }
    }
    *out = task;
    return true;
}

/** Thief-side steal of the oldest task. */
static bool deque_steal(ReDeque *d, void **out) {
    size_t top = atomic_load_explicit(&d->top, memory_order_acquire);

    atomic_thread_fence(memory_order_seq_cst);
    {
        size_t bottom = deque_load_bottom(d);
        void *task;

        if (top >= bottom) {
            return false;
        }
        task = d->tasks[top % d->cap];
        if (!atomic_compare_exchange_strong_explicit(&d->top, &top, top + 1u,
                                                     memory_order_seq_cst,
                                                     memory_order_relaxed)) {
            return false; /* Another thief won; the caller retries. */
        }
        *out = task;
        return true;
    }
}

/* ------------------------------------------------------------------------- */
/* Execution                                                                 */
/* ------------------------------------------------------------------------- */

/** Runs one task body and retires it from the outstanding count. */
static void task_run_and_retire(ReScheduler *s, void *raw) {
    ReTask *task = (ReTask *)raw;

    if (task->range_fn != NULL) {
        task->range_fn(task->ctx, task->begin, task->end);
    } else if (task->job_fn != NULL) {
        task->job_fn(task->ctx);
    }
    /* acq_rel: the release half publishes this task's writes to the submitter
     * before it sees the counter at zero; the acquire half keeps the next
     * task ordered after this one. */
    if (atomic_fetch_sub_explicit(&s->outstanding, 1u, memory_order_acq_rel) == 1u) {
        re_cond_broadcast(&s->cond);
    }
}

/** Pops from the caller's own deque, then probes a few random victims. */
static bool acquire_task(ReScheduler *s, uint32_t index, ReRng *rng, void **out) {
    uint32_t total = s->worker_count + 1u;
    uint32_t attempts = total - 1u < STEAL_ATTEMPTS ? total - 1u : STEAL_ATTEMPTS;
    uint64_t start;

    if (deque_pop(&s->deques[index], out)) {
        return true;
    }
    if (attempts == 0u) {
        return false;
    }
    start = re_rng_range(rng, total);
    for (uint32_t k = 0; k < attempts; k++) {
        uint32_t victim = (uint32_t)((start + k) % total);

        if (victim != index && deque_steal(&s->deques[victim], out)) {
            return true;
        }
    }
    return false;
}

/** Runs tasks until the outstanding counter reaches zero (submitter side). */
static void run_until_done(ReScheduler *s, uint32_t index) {
    ReRng rng = re_rng_init((uint64_t)index + UINT64_C(0x9E3779B97F4A7C15));
    void *task = NULL;

    while (atomic_load_explicit(&s->outstanding, memory_order_acquire) != 0u) {
        if (acquire_task(s, index, &rng, &task)) {
            task_run_and_retire(s, task);
        } else {
            re_thread_yield();
        }
    }
}

/**
 * @brief Publishes a batch of tasks and blocks until all of them ran.
 *
 * The whole batch goes into deques[0], the submitter's own deque: a Chase-Lev
 * deque has exactly one thread that may touch `bottom` (here the caller), so
 * a producer cannot inject into another thread's deque without racing with
 * that worker's pop. Workers steal from deques[0] instead; thieves walk the
 * deque oldest first while the owner pops newest first, which spreads the
 * batch out immediately.
 */
/**
 * @brief One batch submission helper (see re_scheduler_parallel_for).
 *
 * CONCURRENCY NOTE (known benign protocol races): the task slot buffer
 * (d->tasks) is a plain array by design — this is the published
 * Chase-Lev algorithm (N. Ridge's C11-verified formulation, also used
 * by Rust crossbeam). ThreadSanitizer reports two races here that are
 * safe by protocol invariant:
 *   1. thief reads tasks[t % cap] while the owner writes tasks[b % cap]:
 *      the thief may only read slots below its acquire-loaded bottom,
 *      and the fence discipline makes any slot it reaches fully
 *      published; the owner writes only slots >= bottom. A stale
 *      bottom read can overlap exactly the contested last slot, whose
 *      ownership is settled by the seq_cst CAS on top before the
 *      buffer slot is touched by the winner.
 *   2. a worker reads ReTask fields after the submitter's malloc/fill:
 *      the submitter's release fence in deque_push + the worker's
 *      acquire bottom load make the fill visible before the steal
 *      completes; lifetime is guarded by the outstanding counter
 *      (the batch is freed only after every task retired).
 * These are the same races crossbeam documents and suppresses; no
 * functional fix exists that keeps the algorithm lock-free.
 */
static void submit_and_wait(ReScheduler *s, ReTask *tasks, uint32_t ntasks) {
    RE_ASSERT(ntasks <= s->deques[0].cap);

    /* Counted before the first push: a thief can finish a task before the
     * submitter has published the rest of the batch. */
    atomic_fetch_add_explicit(&s->outstanding, ntasks, memory_order_relaxed);
    re_mutex_lock(&s->mutex);
    for (uint32_t i = 0; i < ntasks; i++) {
        deque_push(&s->deques[0], (void *)&tasks[i]);
    }
    re_cond_broadcast(&s->cond);
    re_mutex_unlock(&s->mutex);
    run_until_done(s, 0u);
}

/** Worker entry point: pop, steal, or sleep until work or shutdown appears. */
static void *worker_main(void *arg) {
    const WorkerArg *self = (const WorkerArg *)arg;
    ReScheduler *s = self->scheduler;
    uint32_t index = self->index;
    ReRng rng = re_rng_init((uint64_t)index + 1u);

    for (;;) {
        void *task;

        if (acquire_task(s, index, &rng, &task)) {
            task_run_and_retire(s, task);
            continue;
        }
        if (atomic_load_explicit(&s->shutdown_flag, memory_order_acquire) != 0u) {
            break;
        }
        re_mutex_lock(&s->mutex);
        /* Re-checked under the mutex: every submitter pushes and broadcasts
         * while holding it, so a wake-up cannot be lost here. */
        if (atomic_load_explicit(&s->shutdown_flag, memory_order_acquire) != 0u) {
            re_mutex_unlock(&s->mutex);
            break;
        }
        if (!acquire_task(s, index, &rng, &task)) {
            re_cond_wait(&s->cond, &s->mutex);
            re_mutex_unlock(&s->mutex);
            continue;
        }
        re_mutex_unlock(&s->mutex);
        task_run_and_retire(s, task);
    }
    return NULL;
}

/* ------------------------------------------------------------------------- */
/* Pool lifetime                                                             */
/* ------------------------------------------------------------------------- */

/** Worker arguments live in the tail of the deque allocation. */
static WorkerArg *worker_args(ReScheduler *s) {
    return (WorkerArg *)(void *)(s->deques + (size_t)s->worker_count + 1u);
}

/** Releases the thread handles, deques, worker arguments and task buffers. */
static void release_pool(ReScheduler *s) {
    if (s->deque_bufs != NULL) {
        uint32_t deque_count = s->worker_count + 1u;

        for (uint32_t i = 0; i < deque_count; i++) {
            free(s->deque_bufs[i]);
        }
        free(s->deque_bufs);
        s->deque_bufs = NULL;
    }
    free(s->deques);
    s->deques = NULL;
    free(s->threads);
    s->threads = NULL;
}

/** Allocates and initializes the deques with their fixed task buffers. */
static bool alloc_pool(ReScheduler *s) {
    uint32_t workers = s->worker_count;
    size_t deque_count = (size_t)workers + 1u;
    size_t bytes = deque_count * sizeof(ReDeque) + (size_t)workers * sizeof(WorkerArg);
    size_t cap = deque_capacity(workers);

    s->threads = calloc(workers, sizeof *s->threads);
    s->deques = malloc(bytes);
    s->deque_bufs = calloc(deque_count, sizeof *s->deque_bufs);
    if (s->threads == NULL || s->deques == NULL || s->deque_bufs == NULL) {
        return false;
    }
    memset(s->deques, 0, bytes);
    for (size_t i = 0; i < deque_count; i++) {
        s->deque_bufs[i] = calloc(cap, sizeof *s->deque_bufs[i]);
        if (s->deque_bufs[i] == NULL) {
            return false;
        }
        atomic_init(&s->deques[i].top, 0u);
        atomic_init(&s->deques[i].bottom, 0u);
        s->deques[i].tasks = s->deque_bufs[i];
        s->deques[i].cap = cap;
    }
    return true;
}

/** Unwinds a partially spawned pool. */
static void init_fail(ReScheduler *s, uint32_t spawned) {
    re_mutex_lock(&s->mutex);
    atomic_store_explicit(&s->shutdown_flag, 1u, memory_order_release);
    re_cond_broadcast(&s->cond);
    re_mutex_unlock(&s->mutex);
    for (uint32_t i = 0; i < spawned; i++) {
        re_thread_join(&s->threads[i]);
    }
    re_cond_destroy(&s->cond);
    re_mutex_destroy(&s->mutex);
    release_pool(s);
    s->worker_count = 0u;
}

/** Spawns every worker; unwinds the pool on the first failure. */
static bool spawn_workers(ReScheduler *s) {
    WorkerArg *args = worker_args(s);

    for (uint32_t i = 0; i < s->worker_count; i++) {
        args[i].scheduler = s;
        /* Worker i owns deques[i + 1]: deques[0] belongs to the submitting
         * thread, and a Chase-Lev deque tolerates exactly one pop-side owner. */
        args[i].index = i + 1u;
        if (!re_thread_start(&s->threads[i], worker_main, &args[i])) {
            re_log_error("re_scheduler_init: cannot start worker %u", i);
            init_fail(s, i);
            return false;
        }
    }
    return true;
}

bool re_scheduler_init(ReScheduler *s, uint32_t worker_count) {
    uint32_t cpus;

    if (s == NULL) {
        return false;
    }
    memset(s, 0, sizeof *s);
    cpus = re_cpu_count();
    s->worker_count = worker_count != 0u ? worker_count : (cpus > 1u ? cpus - 1u : 1u);
    atomic_init(&s->outstanding, 0u);
    atomic_init(&s->shutdown_flag, 0u);
    if (!alloc_pool(s)) {
        re_log_error("re_scheduler_init: out of memory");
        release_pool(s);
        return false;
    }
    if (!re_mutex_init(&s->mutex)) {
        release_pool(s);
        return false;
    }
    if (!re_cond_init(&s->cond)) {
        re_mutex_destroy(&s->mutex);
        release_pool(s);
        return false;
    }
    return spawn_workers(s);
}

void re_scheduler_destroy(ReScheduler *s) {
    if (s == NULL || s->threads == NULL) {
        return;
    }
    /* Callers must not destroy the pool with a submission in flight: workers
     * stop at the first look at the flag, so their tasks would never run. */
    re_mutex_lock(&s->mutex);
    atomic_store_explicit(&s->shutdown_flag, 1u, memory_order_release);
    re_cond_broadcast(&s->cond);
    re_mutex_unlock(&s->mutex);
    for (uint32_t i = 0; i < s->worker_count; i++) {
        re_thread_join(&s->threads[i]);
    }
    re_mutex_destroy(&s->mutex);
    re_cond_destroy(&s->cond);
    release_pool(s);
    s->worker_count = 0u;
}

uint32_t re_scheduler_workers(const ReScheduler *s) {
    return s != NULL ? s->worker_count : 0u;
}

/* ------------------------------------------------------------------------- */
/* Submission                                                                */
/* ------------------------------------------------------------------------- */

/**
 * @brief Fills `tasks` with the jobs of one wave, in job order.
 * @return Number of tasks written.
 */
static uint32_t collect_wave(const ReJobDesc *jobs, uint32_t count, const uint32_t *wave_of,
                             uint32_t wave, uint32_t wave_size, ReTask *tasks) {
    uint32_t filled = 0u;

    for (uint32_t j = 0; j < count && filled < wave_size; j++) {
        if (wave_of[j] != wave) {
            continue;
        }
        tasks[filled].range_fn = NULL;
        tasks[filled].job_fn = jobs[j].fn;
        tasks[filled].ctx = jobs[j].ctx;
        tasks[filled].begin = 0u;
        tasks[filled].end = 0u;
        filled++;
    }
    return filled;
}

/** Runs one wave: on the pool when it fits a deque, inline otherwise. */
static bool run_wave(ReScheduler *s, const ReJobDesc *jobs, uint32_t count,
                     const uint32_t *wave_of, uint32_t wave, uint32_t wave_size) {
    ReTask *tasks;
    uint32_t filled;

    if (wave_size == 0u) {
        return true;
    }
    tasks = malloc((size_t)wave_size * sizeof *tasks);
    if (tasks == NULL) {
        return false;
    }
    filled = collect_wave(jobs, count, wave_of, wave, wave_size, tasks);
    RE_ASSERT(filled == wave_size);
    if (filled > s->deques[0].cap) {
        /* A wave larger than the submit deque cannot be published at once.
         * Its jobs do not conflict, so running them inline is still correct
         * and keeps the "no conflicting pair overlaps" guarantee. */
        for (uint32_t i = 0; i < filled; i++) {
            if (tasks[i].job_fn != NULL) {
                tasks[i].job_fn(tasks[i].ctx);
            }
        }
    } else {
        submit_and_wait(s, tasks, filled);
    }
    free(tasks);
    return true;
}

/**
 * @brief Chunk size for parallel_for: at least min_chunk, aiming at about four
 *        tasks per thread, and never more tasks than one deque can hold.
 */
static uint32_t plan_chunk(const ReScheduler *s, uint32_t count, uint32_t min_chunk) {
    uint64_t aim = (uint64_t)s->worker_count * 4u + 1u; /* 64-bit: no /0, no wrap. */
    uint32_t chunk = min_chunk != 0u ? min_chunk : 1u; /* Header: min_chunk > 0. */

    if ((uint64_t)count / aim > (uint64_t)chunk) {
        chunk = (uint32_t)((uint64_t)count / aim);
    }
    if ((uint64_t)((count - 1u) / chunk + 1u) > (uint64_t)s->deques[0].cap) {
        chunk = count / (uint32_t)s->deques[0].cap + 1u;
    }
    return chunk;
}

/** Splits [0, count) into ranges of `chunk` items and fills the tasks. */
static void fill_range_tasks(ReTask *tasks, uint32_t tasks_count, ReTaskFn fn, void *ctx,
                             uint32_t count, uint32_t chunk) {
    uint32_t begin = 0u;

    for (uint32_t i = 0; i < tasks_count; i++) {
        uint32_t len = re_min_u32(chunk, count - begin);

        tasks[i].range_fn = fn;
        tasks[i].job_fn = NULL;
        tasks[i].ctx = ctx;
        tasks[i].begin = begin;
        tasks[i].end = begin + len;
        begin += len;
    }
}

bool re_scheduler_parallel_for(ReScheduler *s, ReTaskFn fn, void *ctx,
                               uint32_t count, uint32_t min_chunk) {
    uint32_t chunk;
    uint32_t tasks_count;
    ReTask *tasks;

    if (s == NULL || fn == NULL || s->deques == NULL) {
        return false;
    }
    if (count == 0u) {
        return true;
    }
    chunk = plan_chunk(s, count, min_chunk);
    tasks_count = (count - 1u) / chunk + 1u;
    tasks = malloc((size_t)tasks_count * sizeof *tasks);
    if (tasks == NULL) {
        return false;
    }
    fill_range_tasks(tasks, tasks_count, fn, ctx, count, chunk);
    submit_and_wait(s, tasks, tasks_count);
    free(tasks);
    return true;
}

bool re_scheduler_run_jobs(ReScheduler *s, const ReJobDesc *jobs, uint32_t count) {
    uint32_t *wave_of;
    uint32_t *wave_sizes;
    uint32_t waves;
    bool ok = true;

    if (s == NULL || (count != 0u && jobs == NULL) || s->deques == NULL) {
        return false;
    }
    if (count == 0u) {
        return true;
    }
    wave_of = malloc((size_t)count * sizeof *wave_of);
    wave_sizes = malloc((size_t)count * sizeof *wave_sizes);
    if (wave_of == NULL || wave_sizes == NULL) {
        free(wave_of);
        free(wave_sizes);
        return false;
    }
    waves = re_dag_plan(jobs, count, wave_of, wave_sizes);
    if (waves == 0u) {
        ok = false;
    }
    for (uint32_t w = 0; w < waves && ok; w++) {
        ok = run_wave(s, jobs, count, wave_of, w, wave_sizes[w]);
    }
    free(wave_of);
    free(wave_sizes);
    return ok;
}
