/**
 * @file re_vk_internal.h
 * @brief Private ABI shared by the Vulkan renderer backend files.
 *
 * re_renderer.h exposes only the opaque ReRenderer pointer, so the
 * backend files (re_renderer.c, re_renderer_frame.c, re_null.c) each
 * need the concrete state. This header is the single definition every
 * backend file includes: struct ReRenderer, the GPU resource groups and
 * the cross-file helper declarations. Do not duplicate these layouts in
 * the .c files — a drift between two copies is a memory-corruption bug
 * (the review caught exactly that: two independent mirrors).
 *
 * All contents are compiled only with RE_HAS_VULKAN; re_null.c has its
 * own unrelated struct and never includes this header.
 */
#ifndef RE_VK_INTERNAL_H
#define RE_VK_INTERNAL_H

#if defined(RE_HAS_VULKAN)

#include "re_renderer.h"
#include "re_vk.h"

#if defined(RE_HAS_X11)
#include <X11/Xlib.h> /* Display, Window, Atom */
#endif

/**
 * @brief GPU images owned by the renderer, all inside one device
 *        memory block.
 */
typedef struct ReVkGpuImages {
    VkImage color;            /**< HDR colour target (R16G16B16A16). */
    VkImageView color_view;   /**< HDR colour view. */
    VkImage depth;            /**< Depth target (D32, reverse-Z). */
    VkImageView depth_view;   /**< Depth view. */
    VkImage shadow;           /**< Shadow array, RE_SHADOW_CASCADES layers. */
    VkImageView shadow_view;  /**< Array view of the shadow map. */
    VkDeviceMemory img_mem;   /**< The single image memory block. */
    VkSampler shadow_sampler; /**< Sampler bound at binding 3. */
    VkSampler tex_sampler;    /**< Sampler used by the bindless array. */
} ReVkGpuImages;

/**
 * @brief GPU buffers owned by the renderer, all inside one device
 *        memory block (persistently mapped).
 */
typedef struct ReVkGpuBuffers {
    VkBuffer camera_ubo;      /**< Binding 0. */
    VkBuffer light_ubo;       /**< Binding 1. */
    VkBuffer cluster_ssbo;    /**< Binding 2. */
    VkBuffer light_list_ssbo; /**< Binding 8. */
    VkBuffer instance_ssbo;   /**< Binding 5. */
    VkBuffer visible_ssbo;    /**< Binding 6. */
    VkBuffer draw_args_ssbo;  /**< Binding 7. */
    VkBuffer staging;         /**< Upload staging region. */
    VkDeviceMemory buf_mem;   /**< The single buffer memory block. */
    void *mapped;             /**< Persistent map of buf_mem. */
    VkDeviceSize staging_size;/**< Staging byte size. */
    uint32_t bindless_count;  /**< Bound bindless descriptors. */
    VkImageView *bindless_views; /**< Bindless image views (index 0 = checker). */
} ReVkGpuBuffers;

/**
 * @brief Bookkeeping for one buffer carved out of the buffer block.
 */
typedef struct ReVkBufferRecord {
    VkBuffer buffer;      /**< Buffer handle. */
    VkDeviceSize offset;  /**< Offset inside the block. */
    VkDeviceSize size;    /**< Allocated size. */
} ReVkBufferRecord;

/**
 * @brief Pipelines and shader modules owned by the frame layer.
 *
 * Module order: 0 cull, 1 forward vert, 2 forward frag, 3 shadow vert,
 * 4 tonemap frag, 5 unused.
 */
typedef struct ReVkPipelines {
    VkPipeline shadow_pipe;   /**< Depth-only cascade pipeline. */
    VkPipeline forward_pipe;  /**< Clustered forward pipeline. */
    VkPipeline tonemap_pipe;  /**< Fullscreen tonemap pipeline. */
    VkPipeline cull_pipe;     /**< Frustum cull compute pipeline. */
    VkShaderModule modules[6];/**< Shader modules in the order above. */
} ReVkPipelines;

/** Forward declaration of the frame layer's per-frame state. */
typedef struct ReRendererFrame ReRendererFrame;

/**
 * @brief Concrete renderer state (opaque to the engine).
 *
 * Single canonical layout: re_renderer.c fills the lifecycle fields,
 * re_renderer_frame.c reads the resource groups. Both include this
 * header, so they cannot drift.
 */
struct ReRenderer {
    bool is_null;                   /**< True for the null backend (re_null.c). */
    ReVkApi api;                    /**< Runtime-resolved entry points. */
    VkInstance instance;            /**< VkInstance. */
    VkPhysicalDevice phys;          /**< Selected physical device. */
    VkDevice device;                /**< Logical device. */
    uint32_t gfx_family;            /**< Graphics queue family. */
    uint32_t present_family;        /**< Present queue family. */
    VkQueue gfx;                    /**< Graphics queue. */
    VkQueue present;                /**< Present queue. */
    VkSurfaceKHR surface;           /**< Window surface (VK_NULL_HANDLE headless). */
    VkSwapchainKHR swapchain;       /**< Swapchain (VK_NULL_HANDLE headless). */
    uint32_t sw_count;              /**< Swapchain image count. */
    VkImage *sw_images;             /**< Swapchain images. */
    VkImageView *sw_views;          /**< Swapchain views. */
    VkFormat sw_format;             /**< Swapchain format. */
    VkExtent2D sw_extent;           /**< Swapchain extent. */
    ReVkGpuImages gpu;              /**< Image resources. */
    ReVkGpuBuffers buf;             /**< Buffer resources. */
    ReVkPipelines pipes;            /**< Pipelines and modules. */
    VkPipelineLayout pipe_layout;   /**< Shared pipeline layout. */
    VkDescriptorSetLayout dsl;      /**< Set 0 layout. */
    VkDescriptorPool dpool;         /**< Descriptor pool. */
    VkDescriptorSet dset;           /**< Set 0. */
    VkCommandPool cpool;            /**< Command pool. */
    VkCommandBuffer *cmds;          /**< One command buffer per frame slot. */
    VkSemaphore *acquire;           /**< Acquire semaphores. */
    VkSemaphore *release;           /**< Release semaphores. */
    VkFence *fence;                 /**< Frame fences. */
    uint32_t frame_index;           /**< Current frame slot. */
    uint32_t image_index;           /**< Acquired swapchain image index. */
    uint32_t frames_in_flight;      /**< Frame slot count. */
    uint32_t width;                 /**< Render width. */
    uint32_t height;                /**< Render height. */
    int32_t window_x;               /**< Requested window x (-1 = default). */
    int32_t window_y;               /**< Requested window y (-1 = default). */
    const char *title;              /**< Window title (not owned). */
    VkDebugUtilsMessengerEXT messenger; /**< Debug messenger (or VK_NULL_HANDLE). */
    ReRendererFrame *frame;         /**< Frame layer state. */
    VkDeviceSize img_capacity;      /**< Image block size. */
    VkDeviceSize img_used;          /**< Image block high-water mark. */
    VkDeviceSize buf_capacity;      /**< Buffer block size. */
    VkDeviceSize buf_used;          /**< Buffer block high-water mark. */
    uint32_t img_memory_type;       /**< Image memory type index. */
    uint32_t buf_memory_type;       /**< Buffer memory type index. */
    ReVkBufferRecord *buf_records;  /**< Buffer block bookkeeping. */
    uint32_t buf_record_count;      /**< Entries used. */
    uint32_t buf_record_capacity;   /**< Entries allocated. */
    bool windowed;                  /**< Windowed configuration. */
    bool device_lost;               /**< Latched fatal device error. */
    bool debug_enabled;             /**< Debug messenger active. */
    bool keys[RE_KEY_COUNT];        /**< Held keyboard keys (X11 focus). */
    bool quit_requested;            /**< Escape or WM_DELETE_WINDOW seen. */
#if defined(RE_HAS_X11)
    Display *display;               /**< X11 display. */
    Window window;                  /**< X11 window. */
    Atom wm_delete;                 /**< WM_DELETE_WINDOW atom. */
#endif
};

/* Frame layer entry points (defined in re_renderer_frame.c). */
bool re_frame_create(ReRenderer *r); /* sets r->frame on success */
void re_frame_destroy(ReRenderer *r);
bool re_frame_render(ReRenderer *r, const ReSceneSnapshot *scene);

/* Allocation helpers the frame layer calls (defined in re_renderer.c). */
VkResult re_rr_create_image(ReRenderer *r, VkFormat fmt,
                            VkImageUsageFlags usage, uint32_t w,
                            uint32_t h, uint32_t layers, VkImage *img,
                            VkImageView *view, VkImageAspectFlags aspect);
VkResult re_rr_create_buffer(ReRenderer *r, VkDeviceSize size,
                             VkBufferUsageFlags usage, VkBuffer *buf);
VkResult re_rr_upload(ReRenderer *r, const void *data,
                      VkDeviceSize size, VkDeviceSize dst_offset);
void re_rr_copy_to_buffer(ReRenderer *r, VkBuffer dst, VkDeviceSize size,
                          VkDeviceSize src_offset);
void re_rr_copy_to_image(ReRenderer *r, VkImage dst, uint32_t w,
                         uint32_t h, uint32_t layers,
                         VkDeviceSize src_offset);
VkDeviceMemory re_rr_image_memory(ReRenderer *r);
VkDeviceMemory re_rr_buffer_memory(ReRenderer *r);
void *re_rr_mapped(ReRenderer *r);
void re_rr_cmd(ReRenderer *r, VkCommandBuffer *cb);

#endif /* RE_HAS_VULKAN */

#endif /* RE_VK_INTERNAL_H */
