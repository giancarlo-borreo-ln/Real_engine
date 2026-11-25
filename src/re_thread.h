/**
 * @file re_thread.h
 * @brief Minimal portable threading primitives (POSIX pthreads and Win32).
 *
 * The types carry inline storage sized for the largest implementation so
 * they can be embedded by value in engine structures without heap traffic.
 * Initializers and thread creation return false when the OS call fails;
 * the rest are void no-ops on invalid arguments.
 */
#ifndef RE_THREAD_H
#define RE_THREAD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/** Inline storage size for a mutex (bytes). */
#define RE_MUTEX_STORAGE 64u

/** Inline storage size for a condition variable (bytes). */
#define RE_COND_STORAGE 80u

/** Inline storage size for a thread handle (bytes). */
#define RE_THREAD_STORAGE 16u

/**
 * @brief Mutual exclusion lock with inline storage.
 */
typedef struct ReMutex {
    _Alignas(8) unsigned char storage[RE_MUTEX_STORAGE]; /**< OS handle lives here. */
} ReMutex;

/**
 * @brief Condition variable with inline storage.
 */
typedef struct ReCond {
    _Alignas(8) unsigned char storage[RE_COND_STORAGE]; /**< OS handle lives here. */
} ReCond;

/**
 * @brief Thread handle with inline storage.
 */
typedef struct ReThread {
    _Alignas(8) unsigned char storage[RE_THREAD_STORAGE]; /**< OS handle lives here. */
} ReThread;

/** Thread entry point signature. */
typedef void *(*ReThreadFn)(void *arg);

/**
 * @brief Initializes a mutex.
 * @param m Mutex to initialize.
 * @return True on success.
 */
bool re_mutex_init(ReMutex *m);

/**
 * @brief Destroys a mutex.
 * @param m Mutex to destroy.
 */
void re_mutex_destroy(ReMutex *m);

/**
 * @brief Locks a mutex.
 * @param m Mutex to lock.
 */
void re_mutex_lock(ReMutex *m);

/**
 * @brief Unlocks a mutex.
 * @param m Mutex to unlock.
 */
void re_mutex_unlock(ReMutex *m);

/**
 * @brief Initializes a condition variable.
 * @param c Condition variable to initialize.
 * @return True on success.
 */
bool re_cond_init(ReCond *c);

/**
 * @brief Destroys a condition variable.
 * @param c Condition variable to destroy.
 */
void re_cond_destroy(ReCond *c);

/**
 * @brief Waits on the condition variable. The mutex must be locked; it is
 *        atomically released while waiting and re-acquired on wake-up.
 * @param c Condition variable.
 * @param m Locked mutex.
 */
void re_cond_wait(ReCond *c, ReMutex *m);

/**
 * @brief Wakes one waiter on the condition variable.
 * @param c Condition variable.
 */
void re_cond_signal(ReCond *c);

/**
 * @brief Wakes all waiters on the condition variable.
 * @param c Condition variable.
 */
void re_cond_broadcast(ReCond *c);

/**
 * @brief Spawns a thread.
 * @param t Thread handle to fill.
 * @param fn Entry point.
 * @param arg Argument passed to fn.
 * @return True on success.
 */
bool re_thread_start(ReThread *t, ReThreadFn fn, void *arg);

/**
 * @brief Waits for a thread to finish and reaps it.
 * @param t Thread handle.
 */
void re_thread_join(ReThread *t);

/**
 * @brief Number of logical CPUs visible to the OS (at least 1).
 */
uint32_t re_cpu_count(void);

/**
 * @brief Yields the processor to other runnable threads.
 */
void re_thread_yield(void);

/**
 * @brief Monotonic sleep in milliseconds (best effort, never negative).
 * @param ms Milliseconds to sleep.
 */
void re_thread_sleep_ms(uint32_t ms);

#endif /* RE_THREAD_H */
