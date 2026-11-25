/**
 * @file re_renderer.h
 * @brief Renderer front end: scene snapshot API plus the Vulkan
 *        implementation (bindless clustered forward+, CSM, GPU culling)
 *        and the null fallback used when Vulkan is unavailable.
 *
 * The engine only ever sees the opaque ReRenderer and the
 * ReSceneSnapshot it fills each render tick; re_renderer_create
 * transparently falls back to the null renderer when no Vulkan device
 * can be created, so the simulation always runs.
 *
 * Descriptor set contract (set 0, shared by every pipeline):
 *   binding 0: camera UBO (std140, see ReRenderCameraUbo)
 *   binding 1: light UBO (std140, see ReRenderLightUbo)
 *   binding 2: cluster grid SSBO (RE_CLUSTER_COUNT uvec2: offset, count)
 *   binding 3: shadow cascade sampler array (sampler2DArray,
 *              RE_SHADOW_CASCADES layers)
 *   binding 4: bindless textures (sampler2D array, RE_BINDLESS_COUNT,
 *              update-after-bind, partially bound, variable count)
 *   binding 5: instance SSBO (ReRenderInstance records)
 *   binding 6: visible list SSBO (uint32 instance indices)
 *   binding 7: draw args SSBO (indirect args + atomic drawCount)
 *   binding 8: light list SSBO (ReRenderPointLightGpu records)
 */
#ifndef RE_RENDERER_H
#define RE_RENDERER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "re_vec.h"

/** Descriptor set index shared by every pipeline. */
#define RE_BIND_GLOBAL 0u

/** Camera UBO binding. */
#define RE_BIND_CAMERA_UBO 0u
/** Light UBO binding. */
#define RE_BIND_LIGHT_UBO 1u
/** Cluster grid SSBO binding. */
#define RE_BIND_CLUSTER_SSBO 2u
/** Shadow cascade samplers binding. */
#define RE_BIND_SHADOW_SAMPLERS 3u
/** Bindless texture array binding. */
#define RE_BIND_BINDLESS_TEXTURES 4u
/** Instance SSBO binding. */
#define RE_BIND_INSTANCE_SSBO 5u
/** Visible list SSBO binding. */
#define RE_BIND_VISIBLE_SSBO 6u
/** Draw args SSBO binding. */
#define RE_BIND_DRAW_ARGS_SSBO 7u
/** Light list SSBO binding. */
#define RE_BIND_LIGHT_LIST_SSBO 8u

/** Maximum instances the renderer accepts per frame. */
#define RE_MAX_INSTANCES 16384u

/** Bindless texture descriptor count. */
#define RE_BINDLESS_COUNT 1024u

/** Shadow cascade count. */
#define RE_SHADOW_CASCADES 2u

/** Shadow map resolution (square). */
#define RE_SHADOW_DIM 2048u

/** Cluster grid dimensions. */
#define RE_CLUSTER_X 16u
#define RE_CLUSTER_Y 8u
#define RE_CLUSTER_Z 16u

/** Cluster grid size (16 * 8 * 16). */
#define RE_CLUSTER_COUNT (RE_CLUSTER_X * RE_CLUSTER_Y * RE_CLUSTER_Z)

/** Maximum point lights. */
#define RE_MAX_POINT_LIGHTS 128u

/**
 * @brief Reinterprets a 32-bit word as a float without changing its bits.
 *
 * The GPU instance record stores the uint32 material id and flags words
 * in the w lanes of two vec4 fields (std430 has no padding between a
 * vec3 and the following scalar). This helper makes that packing
 * explicit on the host side.
 * @param bits Raw 32-bit pattern.
 * @return The same pattern viewed as a float.
 */
static inline float re_bits_to_f32(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = bits;
    return v.f;
}

/**
 * @brief One renderable instance (96 bytes, std430-compatible).
 *
 * The two vec4 fields are deliberate: ReVec3 is 16-byte aligned, so
 * spelling aabb_min as a vec3 followed by a uint32 would insert 12
 * bytes of padding before aabb_max. The GPU shaders read a tightly
 * packed array with a 96-byte stride, which is why the layout is
 * expressed exactly as the std430 record:
 *
 *   mat4 model | vec4 (aabb_min.xyz, material) | vec4 (aabb_max.xyz, flags)
 *
 * The material and flags words are uint32 bit patterns stored in the
 * float lanes; use re_bits_to_f32 when writing them.
 */
typedef struct ReSceneInstance {
    ReMat4 model;              /**< Model matrix. */
    ReVec4 aabb_min_material;  /**< xyz AABB min, w material bits. */
    ReVec4 aabb_max_flags;     /**< xyz AABB max, w flags bits. */
} ReSceneInstance;

_Static_assert(sizeof(ReSceneInstance) == 96u,
               "ReSceneInstance must match the 96-byte std430 GPU record");
_Static_assert(offsetof(ReSceneInstance, aabb_min_material) == 64u,
               "instance AABB min/material must start at byte 64");
_Static_assert(offsetof(ReSceneInstance, aabb_max_flags) == 80u,
               "instance AABB max/flags must start at byte 80");

/** Instance flag: cast shadows. */
#define RE_INSTANCE_SHADOW_CASTER 1u

/**
 * @brief Camera for the frame.
 */
typedef struct ReSceneCamera {
    ReMat4 view;        /**< View matrix. */
    ReMat4 proj;        /**< Projection matrix (reverse Z). */
    ReVec3 position;    /**< Camera position. */
    float far_plane;    /**< Far plane distance. */
} ReSceneCamera;

/**
 * @brief Directional sun light.
 */
typedef struct ReSceneLight {
    ReVec3 direction; /**< Direction the light travels (will be normalized). */
    ReVec3 color;     /**< Radiance. */
    float intensity;  /**< Multiplier. */
    float ambient;    /**< Ambient term added everywhere. */
} ReSceneLight;

/**
 * @brief Point light.
 */
typedef struct RePointLight {
    ReVec3 position; /**< Center. */
    float radius;    /**< Influence radius. */
    ReVec3 color;    /**< Radiance. */
    float intensity; /**< Multiplier. */
} RePointLight;

/**
 * @brief Everything the renderer needs for one frame.
 */
typedef struct ReSceneSnapshot {
    const ReSceneInstance *instances; /**< Instance array. */
    uint32_t instance_count;          /**< Instances (<= RE_MAX_INSTANCES). */
    ReSceneCamera camera;             /**< Frame camera. */
    ReSceneLight sun;                 /**< Sun light. */
    const RePointLight *point_lights; /**< Point lights. */
    uint32_t point_light_count;       /**< Point lights (<= RE_MAX_POINT_LIGHTS). */
    ReVec3 clear_color;               /**< Sky color. */
} ReSceneSnapshot;

/**
 * @brief Keyboard keys exposed to the game layer.
 *
 * The renderer owns the window, so it is the natural place to collect
 * keyboard state. Games poll a snapshot once per frame; the engine does
 * not interpret the keys.
 */
typedef enum ReKey {
    RE_KEY_W = 0,   /**< W. */
    RE_KEY_A,       /**< A. */
    RE_KEY_S,       /**< S. */
    RE_KEY_D,       /**< D. */
    RE_KEY_UP,      /**< Up arrow. */
    RE_KEY_DOWN,    /**< Down arrow. */
    RE_KEY_LEFT,    /**< Left arrow. */
    RE_KEY_RIGHT,   /**< Right arrow. */
    RE_KEY_SPACE,   /**< Space. */
    RE_KEY_LSHIFT,  /**< Left shift. */
    RE_KEY_RSHIFT,  /**< Right shift. */
    RE_KEY_ENTER,   /**< Return / Enter. */
    RE_KEY_R,       /**< R. */
    RE_KEY_COUNT    /**< Number of tracked keys. */
} ReKey;

/**
 * @brief Keyboard snapshot for one frame.
 *
 * Keys are reported as held (down) rather than as edge events, which is
 * what continuous movement needs. Escape is handled by
 * re_renderer_should_close instead of appearing here.
 */
typedef struct ReInput {
    bool down[RE_KEY_COUNT]; /**< True while the key is held. */
} ReInput;

/**
 * @brief Renderer configuration.
 */
typedef struct ReRenderConfig {
    bool windowed;    /**< True = create a window + swapchain. */
    uint32_t width;   /**< Window / render width. */
    uint32_t height;  /**< Window / render height. */
    const char *title; /**< Window title. */
    int32_t window_x; /**< Initial window x, or -1 for the default. */
    int32_t window_y; /**< Initial window y, or -1 for the default. */
} ReRenderConfig;

/**
 * @brief Opaque renderer handle.
 */
typedef struct ReRenderer ReRenderer;

/**
 * @brief Creates the best available renderer.
 *
 * Tries the Vulkan backend (when compiled in and a device exists),
 * otherwise falls back to the null renderer, which keeps frame pacing
 * but performs no rasterization. Returns NULL only on catastrophic
 * allocation failure.
 * @param cfg Configuration.
 */
ReRenderer *re_renderer_create(const ReRenderConfig *cfg);

/**
 * @brief Renders one frame of the snapshot.
 * @param r Renderer.
 * @param scene Scene to draw.
 * @return False on a device-loss-class error.
 */
bool re_renderer_render(ReRenderer *r, const ReSceneSnapshot *scene);

/**
 * @brief True when the user closed the window (always false headless).
 *
 * Also pumps pending window events, so calling it once per frame keeps
 * the input snapshot fresh.
 * @param r Renderer.
 */
bool re_renderer_should_close(const ReRenderer *r);

/**
 * @brief Reports the presentation extent of the current frame.
 *
 * This can differ from ReRenderConfig::width/height: XWayland and
 * fractional desktop scaling make the swapchain larger than the logical
 * window. Games building their projection from the window size should
 * use this extent instead.
 * @param r Renderer.
 * @param width Receives the width, or 0 when unavailable (null backend).
 * @param height Receives the height, or 0 when unavailable.
 */
void re_renderer_extent(const ReRenderer *r, uint32_t *width, uint32_t *height);

/**
 * @brief True when the renderer is the no-window null fallback.
 *
 * Games use this to switch to an autopilot/demo mode on machines where
 * Vulkan or a display is unavailable.
 * @param r Renderer (NULL counts as null).
 */
bool re_renderer_is_null(const ReRenderer *r);

/**
 * @brief Copies the current keyboard snapshot.
 *
 * Pumps pending window events first. Null and headless backends report
 * every key as up. Unknown renderer pointers are treated as headless.
 * @param r Renderer.
 * @param out Receives the snapshot (zeroed first).
 */
void re_renderer_poll_input(ReRenderer *r, ReInput *out);

/**
 * @brief Tears the renderer down.
 * @param r Renderer.
 */
void re_renderer_destroy(ReRenderer *r);

/**
 * @brief Creates the null renderer directly (no Vulkan at all).
 * @param cfg Configuration (windowed is ignored).
 */
ReRenderer *re_null_renderer_create(const ReRenderConfig *cfg);

#endif /* RE_RENDERER_H */
