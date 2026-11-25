/**
 * @file re_null.c
 * @brief Null renderer: keeps frame pacing without a Vulkan device.
 *
 * re_renderer_create (in re_renderer.c) falls back to
 * re_null_renderer_create when Vulkan is compiled out, no loader is
 * installed or no physical device can be opened, so the simulation
 * always runs. The renderer sleeps for one 60 Hz frame per tick,
 * reports once per second and rasterizes nothing.
 *
 * Symbol ownership: the public re_renderer_* entry points live in
 * re_renderer.c (both build configurations); this file exports only the
 * null backend internals that re_renderer.c dispatches to.
 */
#include "re_core.h"
#include "re_renderer.h"
#include "re_thread.h"

#include <stdlib.h>
#include <string.h>

/** Frame budget in milliseconds (60 Hz pacing). */
#define RE_NULL_FRAME_MS 16u

/** Report period in nanoseconds (one line per second). */
#define RE_NULL_REPORT_NS UINT64_C(1000000000)

/** Default window title when the configuration omits one. */
#define RE_NULL_TITLE "real engine"

/**
 * @brief Private state of the null renderer.
 *
 * re_renderer.h keeps ReRenderer opaque, so each backend owns its own
 * definition; nothing outside this file can inspect these fields.
 * The first field (is_null) is shared with the Vulkan backend's struct
 * so the dispatcher in re_renderer.c can tell the backends apart.
 */
struct ReRenderer {
    bool is_null;            /**< Backend tag (always true here). */
    uint64_t start_ns;       /**< Creation timestamp. */
    uint64_t frame_count;    /**< Frames paced so far. */
    uint64_t last_report_ns; /**< Timestamp of the last report line. */
    uint32_t width;          /**< Requested width (informational). */
    uint32_t height;         /**< Requested height (informational). */
    const char *title;       /**< Requested title (not owned). */
    bool windowed;           /**< Requested windowing (ignored). */
    bool should_close;       /**< Close request; never set by this backend. */
};

/**
 * @brief Creates the null renderer.
 *
 * Never fails for a NULL configuration: the defaults are width 0,
 * height 0 and a headless 60 Hz loop.
 *
 * @param cfg Configuration, may be NULL.
 * @return The renderer, or NULL when the allocation itself fails.
 */
ReRenderer *re_null_renderer_create(const ReRenderConfig *cfg) {
    ReRenderer *r = calloc(1u, sizeof *r);

    if (r == NULL) {
        re_log_error("null renderer: out of memory");
        return NULL;
    }
    r->is_null = true;
    r->width = cfg != NULL ? cfg->width : 0u;
    r->height = cfg != NULL ? cfg->height : 0u;
    r->title = cfg != NULL && cfg->title != NULL ? cfg->title : RE_NULL_TITLE;
    r->windowed = cfg != NULL ? cfg->windowed : false;
    r->start_ns = re_time_ns();
    r->last_report_ns = r->start_ns;
    re_log_info("null renderer active (no Vulkan device): frames are paced but not rasterized");
    return r;
}

/**
 * @brief Paces one frame and reports progress once per second.
 *
 * A NULL snapshot is a no-op: the frame is not counted, nothing is
 * logged and the call still succeeds, because the null backend has no
 * error states.
 *
 * @param r Renderer.
 * @param scene Snapshot that would have been drawn.
 * @return True unless r is NULL.
 */
bool re_null_render(ReRenderer *r, const ReSceneSnapshot *scene) {
    uint64_t now;

    if (r == NULL) {
        return false;
    }
    if (scene == NULL) {
        return true;
    }
    r->frame_count++;
    now = re_time_ns();
    if (now - r->last_report_ns >= RE_NULL_REPORT_NS) {
        r->last_report_ns = now;
        re_log_info("null frame %llu | instances %u | point lights %u",
                    (unsigned long long)r->frame_count, scene->instance_count,
                    scene->point_light_count);
    }
    re_thread_sleep_ms(RE_NULL_FRAME_MS);
    return true;
}

/**
 * @brief True when a close was requested.
 *
 * The null renderer never sets the flag: a headless run is bounded by
 * the engine's own duration, not by a window. The field exists so the
 * API contract is honoured exactly.
 * @param r Renderer.
 */
bool re_null_should_close(const ReRenderer *r) {
    return r != NULL && r->should_close;
}

/**
 * @brief Reports every key as released (there is no window or device).
 * @param r Renderer, may be NULL.
 * @param out Receives the zeroed snapshot, may be NULL.
 */
void re_null_poll_input(ReRenderer *r, ReInput *out) {
    (void)r;
    if (out != NULL) {
        memset(out, 0, sizeof *out);
    }
}

/**
 * @brief Releases the null renderer.
 * @param r Renderer, may be NULL.
 */
void re_null_destroy(ReRenderer *r) {
    free(r);
}
