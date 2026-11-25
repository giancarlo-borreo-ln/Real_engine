/**
 * @file re_core.c
 * @brief Engine-wide primitives: panic, logging, monotonic time and sleeping.
 *
 * This is the only translation unit that touches the operating system for
 * the facilities declared in re_core.h; everything else stays freestanding.
 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "re_core.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <errno.h>
#include <time.h>
#endif

/* Stack buffer size for a single formatted log line (bytes). */
#define RE_LOG_BUFFER 1024u

/* Stack buffer size for the debugger copy of a panic message (bytes). */
#define RE_PANIC_BUFFER 512u

/* Formats one line with a level prefix and emits it as a single stdio call,
 * so concurrent loggers cannot interleave inside a line. */
static void log_line(const char *prefix, const char *fmt, va_list args) {
    char text[RE_LOG_BUFFER];

    if (vsnprintf(text, sizeof text, fmt, args) < 0) {
        return; /* Formatting failed: drop the line instead of printing junk. */
    }
    (void)fprintf(stderr, "%s %s\n", prefix, text);
    (void)fflush(stderr);
}

_Noreturn void re_panic(const char *file, int line, const char *message) {
    const char *where = file != NULL ? file : "<unknown>";
    const char *what = message != NULL ? message : "(no message)";

    (void)fprintf(stderr, "REAL ENGINE PANIC at %s:%d: %s\n", where, line, what);
    (void)fflush(stderr);
#ifdef _WIN32
    {
        char text[RE_PANIC_BUFFER];

        if (snprintf(text, sizeof text, "REAL ENGINE PANIC at %s:%d: %s", where, line, what) > 0) {
            OutputDebugStringA(text);
        }
    }
#endif
    abort();
}

void re_log_info(const char *fmt, ...) {
    va_list args;

    va_start(args, fmt);
    log_line("[info]", fmt, args);
    va_end(args);
}

void re_log_warn(const char *fmt, ...) {
    va_list args;

    va_start(args, fmt);
    log_line("[warn]", fmt, args);
    va_end(args);
}

void re_log_error(const char *fmt, ...) {
    va_list args;

    va_start(args, fmt);
    log_line("[error]", fmt, args);
    va_end(args);
}

uint64_t re_time_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;

    if (QueryPerformanceFrequency(&frequency) == 0 || frequency.QuadPart <= 0) {
        return 0u;
    }
    if (QueryPerformanceCounter(&counter) == 0 || counter.QuadPart < 0) {
        return 0u;
    }
    {
        uint64_t ticks = (uint64_t)counter.QuadPart;
        uint64_t rate = (uint64_t)frequency.QuadPart;
        uint64_t seconds = ticks / rate;
        uint64_t rest = ticks % rate;

        return seconds * UINT64_C(1000000000) + rest * UINT64_C(1000000000) / rate;
    }
#else
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0u;
    }
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
#endif
}

void re_sleep_ms(uint32_t ms) {
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
        request.tv_nsec = (long)(ms % 1000u) * 1000000L;
        /* Restart after a signal so the call really sleeps for >= ms. */
        while (nanosleep(&request, &remainder) != 0) {
            if (errno != EINTR) {
                break;
            }
            request = remainder;
        }
    }
#endif
}
