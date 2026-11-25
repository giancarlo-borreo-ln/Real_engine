/**
 * @file re_thread.c
 * @brief POSIX and Win32 implementation of the inline-storage threading layer.
 *
 * Every OS handle is constructed inside the caller-provided ReMutex, ReCond
 * or ReThread storage: no handle is ever heap allocated. Alignment and size
 * of the embedded types are checked at compile time against the header
 * constants, so a layout change is a build error, not a runtime bug.
 *
 * The Win32 path is the only place that touches the heap, and only for the
 * transient ThreadStart bundle that carries fn and arg through the
 * DWORD-returning CreateThread entry point; the trampoline frees it before
 * the user function runs.
 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "re_thread.h"

#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <stdlib.h>
#else
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>
#endif

/* Nanoseconds in one millisecond. */
#define RE_NS_PER_MS 1000000L

#ifdef _WIN32

/* Windows CRITICAL_SECTION/CONDITION_VARIABLE/HANDLE placement checks. */
static_assert(sizeof(CRITICAL_SECTION) <= RE_MUTEX_STORAGE, "RE_MUTEX_STORAGE too small for CRITICAL_SECTION");
static_assert(sizeof(CONDITION_VARIABLE) <= RE_COND_STORAGE, "RE_COND_STORAGE too small for CONDITION_VARIABLE");
static_assert(sizeof(HANDLE) <= RE_THREAD_STORAGE, "RE_THREAD_STORAGE too small for HANDLE");
static_assert(_Alignof(CRITICAL_SECTION) <= 8u, "ReMutex storage alignment too weak for CRITICAL_SECTION");
static_assert(_Alignof(CONDITION_VARIABLE) <= 8u, "ReCond storage alignment too weak for CONDITION_VARIABLE");
static_assert(_Alignof(HANDLE) <= 8u, "ReThread storage alignment too weak for HANDLE");

/* Argument bundle handed to the Win32 thread trampoline. */
typedef struct ThreadStart {
    ReThreadFn fn;
    void *arg;
} ThreadStart;

/* Views the mutex storage as its CRITICAL_SECTION. */
static CRITICAL_SECTION *win_mutex(ReMutex *m) {
    return (CRITICAL_SECTION *)(void *)m->storage;
}

/* Views the condition storage as its CONDITION_VARIABLE. */
static CONDITION_VARIABLE *win_cond(ReCond *c) {
    return (CONDITION_VARIABLE *)(void *)c->storage;
}

/* Views the thread storage as its HANDLE slot. */
static HANDLE *win_thread(ReThread *t) {
    return (HANDLE *)(void *)t->storage;
}

/* Adapts a ReThreadFn to the DWORD-returning Win32 start routine. */
static DWORD WINAPI thread_trampoline(LPVOID raw) {
    ThreadStart *start = (ThreadStart *)raw;
    ReThreadFn fn = start->fn;
    void *arg = start->arg;

    free(start); /* Owned by the thread once CreateThread succeeded. */
    (void)fn(arg);
    return 0u;
}

#else

/* POSIX placement checks against the header storage sizes. */
static_assert(sizeof(pthread_mutex_t) <= RE_MUTEX_STORAGE, "RE_MUTEX_STORAGE too small for pthread_mutex_t");
static_assert(sizeof(pthread_cond_t) <= RE_COND_STORAGE, "RE_COND_STORAGE too small for pthread_cond_t");
static_assert(sizeof(pthread_t) <= RE_THREAD_STORAGE, "RE_THREAD_STORAGE too small for pthread_t");
static_assert(_Alignof(pthread_mutex_t) <= 8u, "ReMutex storage alignment too weak for pthread_mutex_t");
static_assert(_Alignof(pthread_cond_t) <= 8u, "ReCond storage alignment too weak for pthread_cond_t");
static_assert(_Alignof(pthread_t) <= 8u, "ReThread storage alignment too weak for pthread_t");

/* Views the mutex storage as its pthread_mutex_t. */
static pthread_mutex_t *posix_mutex(ReMutex *m) {
    return (pthread_mutex_t *)(void *)m->storage;
}

/* Views the condition storage as its pthread_cond_t. */
static pthread_cond_t *posix_cond(ReCond *c) {
    return (pthread_cond_t *)(void *)c->storage;
}

/* Views the thread storage as its pthread_t slot. */
static pthread_t *posix_thread(ReThread *t) {
    return (pthread_t *)(void *)t->storage;
}

#endif

bool re_mutex_init(ReMutex *m) {
    if (m == NULL) {
        return false;
    }
    memset(m->storage, 0, sizeof m->storage);
#ifdef _WIN32
    InitializeCriticalSection(win_mutex(m));
    return true;
#else
    return pthread_mutex_init(posix_mutex(m), NULL) == 0;
#endif
}

void re_mutex_destroy(ReMutex *m) {
    if (m == NULL) {
        return;
    }
#ifdef _WIN32
    DeleteCriticalSection(win_mutex(m));
#else
    (void)pthread_mutex_destroy(posix_mutex(m));
#endif
    memset(m->storage, 0, sizeof m->storage);
}

void re_mutex_lock(ReMutex *m) {
    if (m == NULL) {
        return;
    }
#ifdef _WIN32
    EnterCriticalSection(win_mutex(m));
#else
    (void)pthread_mutex_lock(posix_mutex(m));
#endif
}

void re_mutex_unlock(ReMutex *m) {
    if (m == NULL) {
        return;
    }
#ifdef _WIN32
    LeaveCriticalSection(win_mutex(m));
#else
    (void)pthread_mutex_unlock(posix_mutex(m));
#endif
}

bool re_cond_init(ReCond *c) {
    if (c == NULL) {
        return false;
    }
    memset(c->storage, 0, sizeof c->storage);
#ifdef _WIN32
    InitializeConditionVariable(win_cond(c));
    return true;
#else
    return pthread_cond_init(posix_cond(c), NULL) == 0;
#endif
}

void re_cond_destroy(ReCond *c) {
    if (c == NULL) {
        return;
    }
#ifdef _WIN32
    /* CONDITION_VARIABLE has no destructor on Win32. */
#else
    (void)pthread_cond_destroy(posix_cond(c));
#endif
    memset(c->storage, 0, sizeof c->storage);
}

void re_cond_wait(ReCond *c, ReMutex *m) {
    if (c == NULL || m == NULL) {
        return;
    }
#ifdef _WIN32
    (void)SleepConditionVariableCS(win_cond(c), win_mutex(m), INFINITE);
#else
    (void)pthread_cond_wait(posix_cond(c), posix_mutex(m));
#endif
}

void re_cond_signal(ReCond *c) {
    if (c == NULL) {
        return;
    }
#ifdef _WIN32
    WakeConditionVariable(win_cond(c));
#else
    (void)pthread_cond_signal(posix_cond(c));
#endif
}

void re_cond_broadcast(ReCond *c) {
    if (c == NULL) {
        return;
    }
#ifdef _WIN32
    WakeAllConditionVariable(win_cond(c));
#else
    (void)pthread_cond_broadcast(posix_cond(c));
#endif
}

bool re_thread_start(ReThread *t, ReThreadFn fn, void *arg) {
    if (t == NULL || fn == NULL) {
        return false;
    }
    memset(t->storage, 0, sizeof t->storage);
#ifdef _WIN32
    {
        ThreadStart *start = (ThreadStart *)malloc(sizeof *start);
        HANDLE handle;

        if (start == NULL) {
            return false;
        }
        start->fn = fn;
        start->arg = arg;
        handle = CreateThread(NULL, 0u, thread_trampoline, start, 0u, NULL);
        if (handle == NULL) {
            free(start);
            return false;
        }
        *win_thread(t) = handle;
        return true;
    }
#else
    return pthread_create(posix_thread(t), NULL, fn, arg) == 0;
#endif
}

void re_thread_join(ReThread *t) {
    if (t == NULL) {
        return;
    }
#ifdef _WIN32
    if (*win_thread(t) != NULL) {
        (void)WaitForSingleObject(*win_thread(t), INFINITE);
        (void)CloseHandle(*win_thread(t));
    }
#else
    (void)pthread_join(*posix_thread(t), NULL);
#endif
    memset(t->storage, 0, sizeof t->storage);
}

uint32_t re_cpu_count(void) {
#ifdef _WIN32
    SYSTEM_INFO info;

    GetSystemInfo(&info);
    return info.dwNumberOfProcessors > 0u ? (uint32_t)info.dwNumberOfProcessors : 1u;
#else
    long count = sysconf(_SC_NPROCESSORS_ONLN);

    if (count < 1) {
        return 1u;
    }
    return (uint64_t)count > UINT32_MAX ? UINT32_MAX : (uint32_t)count;
#endif
}

void re_thread_yield(void) {
#ifdef _WIN32
    (void)SwitchToThread();
#else
    (void)sched_yield();
#endif
}

void re_thread_sleep_ms(uint32_t ms) {
    if (ms == 0u) {
        return;
    }
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    {
        struct timespec request;
        struct timespec remainder;

        request.tv_sec = (time_t)(ms / 1000u);
        request.tv_nsec = (long)(ms % 1000u) * RE_NS_PER_MS;
        /* Retry on EINTR so the caller really sleeps for at least ms. */
        while (nanosleep(&request, &remainder) != 0) {
            if (errno != EINTR) {
                break;
            }
            request = remainder;
        }
    }
#endif
}
