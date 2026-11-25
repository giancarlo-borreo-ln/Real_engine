/* SHARED-STATE CONTRACT with re_renderer_frame.c:
 *   struct ReRenderer (defined in this file, opaque to the world):
 *     - bool is_null;                  // true only in re_null.c
 *     - ReVkApi api;                   // function table
 *     - VkInstance instance; VkPhysicalDevice phys; VkDevice device;
 *     - uint32_t gfx_family, present_family; VkQueue gfx, present;
 *     - VkSurfaceKHR surface;          // VK_NULL_HANDLE when headless
 *     - VkSwapchainKHR swapchain; uint32_t sw_count; VkImage *sw_images;
 *       VkImageView *sw_views; VkFormat sw_format; VkExtent2D sw_extent;
 *     - ReVkGpuImages gpu; ReVkGpuBuffers buf;   // see below
 *     - ReVkPipelines pipes; VkPipelineLayout pipe_layout; VkDescriptorSetLayout dsl;
 *       VkDescriptorPool dpool; VkDescriptorSet dset;
 *     - VkCommandPool cpool; VkCommandBuffer *cmds;    // FRAMES_IN_FLIGHT entries
 *     - VkSemaphore *acquire, *release; VkFence *fence; // FRAMES_IN_FLIGHT entries
 *     - uint32_t frame_index, image_index;             // per-frame state
 *     - uint32_t frames_in_flight;
 *     - uint32_t width, height; const char *title;     // config echo
 *     - VkDebugUtilsMessengerEXT messenger;            // or VK_NULL_HANDLE
 *     - struct ReRendererFrame *frame;                 // opaque to you; frame agent's per-frame data (created/destroyed via function pointers below)
 * The frame agent provides (declared extern in your file header comment, defined in re_renderer_frame.c):
 *     extern struct ReRendererFrame *re_frame_create(struct ReRenderer *r);   // builds pipelines+descriptors
 *     extern void re_frame_destroy(struct ReRenderer *r);
 *     extern bool re_frame_render(struct ReRenderer *r, const ReSceneSnapshot *scene);
 *   Your code CALLS those three (forward-declared). All GPU images/buffers must live in plain arrays of Vulkan handles grouped as:
 *     typedef struct { VkImage color; VkImageView color_view; VkImage depth; VkImageView depth_view;
 *                      VkImage shadow; VkImageView shadow_view; VkDeviceMemory img_mem;
 *                      VkSampler shadow_sampler, tex_sampler; } ReVkGpuImages;
 *     typedef struct { VkBuffer camera_ubo, light_ubo, cluster_ssbo, light_list_ssbo,
 *                      instance_ssbo, visible_ssbo, draw_args_ssbo, staging;
 *                      VkDeviceMemory buf_mem; void *mapped; VkDeviceSize staging_size;
 *                      uint32_t bindless_count; VkImageView *bindless_views; } ReVkGpuBuffers;
 *     typedef struct { VkPipeline shadow_pipe, forward_pipe, tonemap_pipe, cull_pipe;
 *                      VkShaderModule shadow_vs, fwd_vs, fwd_fs, tonemap_vs, tonemap_fs, cull_cs; } ReVkPipelines;
 *   Memory strategy: TWO device-memory allocations total (one for all images, one for all buffers).
 *   Render targets: color R16G16B16A16_SFLOAT, depth D32_SFLOAT (reverse-Z), shadow D16_UNORM array of RE_SHADOW_CASCADES layers at RE_SHADOW_DIM.
 *   FRAMES_IN_FLIGHT = 2. The frame agent records commands; you own acquire/submit/present, swapchain rebuild, and the X11 window.
 */
/**
 * @file re_renderer.c
 * @brief Vulkan lifecycle: loader, instance, physical device, device,
 *        memory blocks, frame sync, swapchain, X11 window and the
 *        render/present loop. The frame layer (re_renderer_frame.c)
 *        owns pipelines and per-frame recording through the re_rr_*
 *        helpers exported at the bottom of the shared block below.
 *
 * Design decisions that the frame layer depends on:
 *   - Two VkDeviceMemory objects: a device-local image block and a
 *     host-visible, persistently mapped buffer block. Every VkImage and
 *     VkBuffer the frame layer creates is a bump allocation inside one
 *     of those two blocks (re_rr_create_image / re_rr_create_buffer).
 *   - The buffer block is mapped once; r->buf.mapped points at its base.
 *     Buffers created through re_rr_create_buffer are recorded with
 *     their block offset, so re_rr_upload can translate an offset inside
 *     r->buf.staging into a mapped address.
 *   - re_rr_upload must be called before re_rr_copy_to_image /
 *     re_rr_copy_to_buffer, and the copy helpers record into the current
 *     frame command buffer (r->cmds[r->frame_index]): the frame layer is
 *     responsible for starting that command buffer and for the layout
 *     barriers around the copies.
 *   - re_rr_copy_to_buffer copies from the given staging offset;
 *     re_rr_copy_to_image copies from the given staging offset and always
 *     uses the color aspect.
 *   - re_frame_destroy must destroy every VkImage/VkBuffer/VkImageView the
 *     frame layer created; this file frees the two memory blocks after it
 *     returns and never destroys frame-layer objects itself.
 *
 * The backend is compiled in only with RE_HAS_VULKAN; without it every
 * entry point degrades to the null renderer and the file stays portable.
 * The windowed path needs RE_HAS_X11: without it, windowed configurations
 * log a warning and run headless.
 */
#include "re_renderer.h"

#include <stdlib.h>
#include <string.h>

#include "re_core.h"
#include "re_render_graph.h"

/* Null-backend internals (defined in re_null.c): the public entry points
 * in this file dispatch to them when the renderer is the null backend or
 * when Vulkan is compiled out. */
extern bool re_null_render(ReRenderer *r, const ReSceneSnapshot *scene);
extern bool re_null_should_close(const ReRenderer *r);
extern void re_null_poll_input(ReRenderer *r, ReInput *out);
extern void re_null_destroy(ReRenderer *r);

#if defined(RE_HAS_VULKAN)

#if defined(RE_HAS_X11)
#include <X11/Xatom.h>
#include <X11/XKBlib.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>
#endif

#include "re_vk.h"

#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

/** Command buffers, semaphores and fences kept in flight at once. */
#define RE_FRAMES_IN_FLIGHT 2u

/** Alignment of each device-memory block (bytes). */
#define RE_ALLOC_GRANULARITY 65536u

/** Slack reserved in the image block for frame-layer textures (bytes). */
#define RE_IMAGE_RESERVE_BYTES ((VkDeviceSize)64u * 1024u * 1024u)

/** Size of the single buffer block (bytes). */
#define RE_BUFFER_BLOCK_BYTES ((VkDeviceSize)64u * 1024u * 1024u)

/* --------------------------------------------------------------------- */
/* Dynamic loader                                                        */
/* --------------------------------------------------------------------- */

/** OS handle of the opened Vulkan loader library. */
static void *g_loader = NULL;

/** Reference count: every live ReRenderer holds one loader reference. */
static uint32_t g_loader_refs = 0u;

/** Loader trampoline for instance-level functions. */
static PFN_vkGetInstanceProcAddr g_get_instance_proc = NULL;

/** Loader trampoline for device-level functions. */
static PFN_vkGetDeviceProcAddr g_get_device_proc = NULL;

#if !defined(_WIN32)
_Static_assert(sizeof(PFN_vkGetInstanceProcAddr) == sizeof(void *),
               "the loader needs object and function pointers of equal size");
#endif

/**
 * @brief Closes the loader library and clears the trampolines.
 */
static void close_loader(void) {
    if (g_loader_refs > 0u) {
        g_loader_refs--;
    }
    if (g_loader_refs > 0u) {
        return; /* Another live renderer still owns the loader. */
    }
    g_get_instance_proc = NULL;
    g_get_device_proc = NULL;
#if defined(_WIN32)
    if (g_loader != NULL) {
        (void)FreeLibrary((HMODULE)g_loader);
        g_loader = NULL;
    }
#else
    if (g_loader != NULL) {
        (void)dlclose(g_loader);
        g_loader = NULL;
    }
#endif
}

/**
 * @brief Opens the system loader and copies the two trampolines out of it.
 *
 * memcpy is used instead of an object-to-function-pointer cast, which ISO C
 * does not allow.
 *
 * @return True when vkGetInstanceProcAddr resolved.
 */
static bool load_trampolines(void) {
    g_loader_refs = 1u; /* Released by close_loader, also on failure. */
#if defined(_WIN32)
    FARPROC gipa;
    FARPROC gdpa;

    g_loader = (void *)LoadLibraryA("vulkan-1.dll");
    if (g_loader == NULL) {
        return false;
    }
    gipa = GetProcAddress((HMODULE)g_loader, "vkGetInstanceProcAddr");
    gdpa = GetProcAddress((HMODULE)g_loader, "vkGetDeviceProcAddr");
    if (gipa == NULL) {
        close_loader();
        return false;
    }
    memcpy(&g_get_instance_proc, &gipa, sizeof g_get_instance_proc);
    if (gdpa != NULL) {
        memcpy(&g_get_device_proc, &gdpa, sizeof g_get_device_proc);
    }
#else
    void *gipa;
    void *gdpa;

    g_loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (g_loader == NULL) {
        return false;
    }
    gipa = dlsym(g_loader, "vkGetInstanceProcAddr");
    gdpa = dlsym(g_loader, "vkGetDeviceProcAddr");
    if (gipa == NULL) {
        close_loader();
        return false;
    }
    memcpy(&g_get_instance_proc, &gipa, sizeof g_get_instance_proc);
    if (gdpa != NULL) {
        memcpy(&g_get_device_proc, &gdpa, sizeof g_get_device_proc);
    }
#endif
    return g_get_instance_proc != NULL;
}

/** Resolves a required device-level entry and warns when it is missing. */
#define RE_LOAD_DEVICE_ENTRY(api, device, name)                             \
    do {                                                                    \
        (api)->name = (PFN_##name)g_get_device_proc((device), #name);       \
        if ((api)->name == NULL) {                                          \
            re_log_warn("vulkan: device entry %s unavailable", #name);      \
        }                                                                   \
    } while (0)

bool re_vk_load_api(ReVkApi *api) {
    if (api == NULL) {
        return false;
    }
    memset(api, 0, sizeof *api);
    if (g_loader == NULL) {
        if (!load_trampolines()) {
            re_log_warn("vulkan: cannot open the system loader (libvulkan.so.1)");
            return false;
        }
    } else {
        g_loader_refs++; /* A second renderer shares the open loader. */
    }
    /* Only global commands are dispatchable without an instance. */
    api->vkCreateInstance = (PFN_vkCreateInstance)g_get_instance_proc(
        VK_NULL_HANDLE, "vkCreateInstance");
    api->vkEnumerateInstanceExtensionProperties =
        (PFN_vkEnumerateInstanceExtensionProperties)g_get_instance_proc(
            VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties");
    if (api->vkCreateInstance == NULL ||
        api->vkEnumerateInstanceExtensionProperties == NULL) {
        re_log_warn("vulkan: loader is missing the global entry points");
        close_loader();
        return false;
    }
    re_log_info("vulkan: system loader opened, global entry points resolved");
    return true;
}

/** Resolves one device-level entry, warning when it is missing. */
#define RE_LOAD_DEV(api, device, name) RE_LOAD_DEVICE_ENTRY(api, device, name)

void re_vk_load_device(ReVkApi *api, VkInstance instance, VkDevice device) {
    if (api == NULL || device == VK_NULL_HANDLE) {
        re_log_warn("vulkan: cannot resolve device entries without a device");
        return;
    }
    if (g_get_instance_proc == NULL) {
        re_log_warn("vulkan: loader trampolines are gone");
        return;
    }
    if (g_get_device_proc == NULL) {
        g_get_device_proc = (PFN_vkGetDeviceProcAddr)g_get_instance_proc(
            instance, "vkGetDeviceProcAddr");
    }
    if (g_get_device_proc == NULL) {
        re_log_warn("vulkan: vkGetDeviceProcAddr unavailable");
        return;
    }
    RE_LOAD_DEV(api, device, vkSetDebugUtilsObjectNameEXT);
    RE_LOAD_DEV(api, device, vkDestroyDevice);
    RE_LOAD_DEV(api, device, vkGetDeviceQueue);
    RE_LOAD_DEV(api, device, vkCreateSwapchainKHR);
    RE_LOAD_DEV(api, device, vkDestroySwapchainKHR);
    RE_LOAD_DEV(api, device, vkGetSwapchainImagesKHR);
    RE_LOAD_DEV(api, device, vkAcquireNextImageKHR);
    RE_LOAD_DEV(api, device, vkQueuePresentKHR);
    RE_LOAD_DEV(api, device, vkQueueSubmit);
    RE_LOAD_DEV(api, device, vkQueueWaitIdle);
    RE_LOAD_DEV(api, device, vkDeviceWaitIdle);
    RE_LOAD_DEV(api, device, vkCreateImage);
    RE_LOAD_DEV(api, device, vkGetImageMemoryRequirements);
    RE_LOAD_DEV(api, device, vkAllocateMemory);
    RE_LOAD_DEV(api, device, vkFreeMemory);
    RE_LOAD_DEV(api, device, vkBindImageMemory);
    RE_LOAD_DEV(api, device, vkCreateImageView);
    RE_LOAD_DEV(api, device, vkDestroyImageView);
    RE_LOAD_DEV(api, device, vkDestroyImage);
    RE_LOAD_DEV(api, device, vkCreateBuffer);
    RE_LOAD_DEV(api, device, vkGetBufferMemoryRequirements);
    RE_LOAD_DEV(api, device, vkBindBufferMemory);
    RE_LOAD_DEV(api, device, vkMapMemory);
    RE_LOAD_DEV(api, device, vkUnmapMemory);
    RE_LOAD_DEV(api, device, vkFlushMappedMemoryRanges);
    RE_LOAD_DEV(api, device, vkDestroyBuffer);
    RE_LOAD_DEV(api, device, vkCreateSampler);
    RE_LOAD_DEV(api, device, vkDestroySampler);
    RE_LOAD_DEV(api, device, vkCreateCommandPool);
    RE_LOAD_DEV(api, device, vkDestroyCommandPool);
    RE_LOAD_DEV(api, device, vkAllocateCommandBuffers);
    RE_LOAD_DEV(api, device, vkFreeCommandBuffers);
    RE_LOAD_DEV(api, device, vkBeginCommandBuffer);
    RE_LOAD_DEV(api, device, vkEndCommandBuffer);
    RE_LOAD_DEV(api, device, vkResetCommandBuffer);
    RE_LOAD_DEV(api, device, vkCmdPipelineBarrier2);
    RE_LOAD_DEV(api, device, vkCmdPipelineBarrier);
    RE_LOAD_DEV(api, device, vkCmdBeginRendering);
    RE_LOAD_DEV(api, device, vkCmdEndRendering);
    RE_LOAD_DEV(api, device, vkCmdBindPipeline);
    RE_LOAD_DEV(api, device, vkCmdSetViewport);
    RE_LOAD_DEV(api, device, vkCmdSetScissor);
    RE_LOAD_DEV(api, device, vkCmdDraw);
    RE_LOAD_DEV(api, device, vkCmdDrawIndexed);
    RE_LOAD_DEV(api, device, vkCmdDrawIndexedIndirect);
    RE_LOAD_DEV(api, device, vkCmdBindVertexBuffers);
    RE_LOAD_DEV(api, device, vkCmdBindIndexBuffer);
    RE_LOAD_DEV(api, device, vkCmdBindDescriptorSets);
    RE_LOAD_DEV(api, device, vkCmdPushConstants);
    RE_LOAD_DEV(api, device, vkCmdDispatch);
    RE_LOAD_DEV(api, device, vkCmdCopyBuffer);
    RE_LOAD_DEV(api, device, vkCmdCopyBufferToImage);
    RE_LOAD_DEV(api, device, vkCreateShaderModule);
    RE_LOAD_DEV(api, device, vkDestroyShaderModule);
    RE_LOAD_DEV(api, device, vkCreatePipelineLayout);
    RE_LOAD_DEV(api, device, vkDestroyPipelineLayout);
    RE_LOAD_DEV(api, device, vkCreateDescriptorSetLayout);
    RE_LOAD_DEV(api, device, vkDestroyDescriptorSetLayout);
    RE_LOAD_DEV(api, device, vkCreateDescriptorPool);
    RE_LOAD_DEV(api, device, vkDestroyDescriptorPool);
    RE_LOAD_DEV(api, device, vkAllocateDescriptorSets);
    RE_LOAD_DEV(api, device, vkUpdateDescriptorSets);
    RE_LOAD_DEV(api, device, vkCreateGraphicsPipelines);
    RE_LOAD_DEV(api, device, vkCreateComputePipelines);
    RE_LOAD_DEV(api, device, vkDestroyPipeline);
    RE_LOAD_DEV(api, device, vkCreateSemaphore);
    RE_LOAD_DEV(api, device, vkDestroySemaphore);
    RE_LOAD_DEV(api, device, vkCreateFence);
    RE_LOAD_DEV(api, device, vkDestroyFence);
    RE_LOAD_DEV(api, device, vkWaitForFences);
    RE_LOAD_DEV(api, device, vkResetFences);
    re_log_info("vulkan: device entry points resolved");
}

#undef RE_LOAD_DEV

/* --------------------------------------------------------------------- */
/* Shared device state, struct ReRenderer and the cross-file helpers
   live in re_vk_internal.h (single canonical definition). */
#include "re_vk_internal.h"


/* --------------------------------------------------------------------- */
/* Memory blocks                                                         */
/* --------------------------------------------------------------------- */

/**
 * @brief Finds a memory type matching the bits and the property request.
 *
 * @param mp Physical device memory properties.
 * @param type_bits Allowed type bits (UINT32_MAX means any type).
 * @param required Properties the type must have.
 * @param preferred Properties the type should also have (0 = don't care).
 * @param[out] index Receives the chosen memory type index.
 * @return True when a type was found.
 */
static bool find_memory_type(const VkPhysicalDeviceMemoryProperties *mp,
                             uint32_t type_bits, VkMemoryPropertyFlags required,
                             VkMemoryPropertyFlags preferred, uint32_t *index) {
    for (uint32_t pass = 0u; pass < 2u; pass++) {
        VkMemoryPropertyFlags wanted = pass == 0u ? required | preferred : required;
        for (uint32_t i = 0u; i < mp->memoryTypeCount; i++) {
            VkMemoryPropertyFlags flags = mp->memoryTypes[i].propertyFlags;
            if ((type_bits & (UINT32_C(1) << i)) == 0u) {
                continue;
            }
            if ((flags & wanted) == wanted) {
                *index = i;
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief Bump-allocates an aligned range inside a block.
 *
 * @param[in,out] used Bytes already handed out.
 * @param capacity Block capacity in bytes.
 * @param size Requested size.
 * @param alignment Required alignment (non-zero).
 * @param[out] offset Receives the aligned offset.
 * @return False when the block cannot hold the request.
 */
static bool bump_allocate(VkDeviceSize *used, VkDeviceSize capacity,
                          VkDeviceSize size, VkDeviceSize alignment,
                          VkDeviceSize *offset) {
    VkDeviceSize remainder;
    VkDeviceSize padding;
    VkDeviceSize start;

    if (size == 0u || alignment == 0u || *used > capacity) {
        return false;
    }
    remainder = *used % alignment;
    padding = remainder == 0u ? 0u : alignment - remainder;
    if (padding > capacity - *used) {
        return false;
    }
    start = *used + padding;
    if (size > capacity - start) {
        return false;
    }
    *offset = start;
    *used = start + size;
    return true;
}

/**
 * @brief Allocates one device-memory block.
 *
 * @param r Renderer (device must exist).
 * @param size Block size in bytes.
 * @param type Memory type index.
 * @param[out] out Receives the memory handle.
 * @param[out] capacity Receives the block size.
 * @return True on success.
 */
static bool allocate_memory_block(struct ReRenderer *r, VkDeviceSize size,
                                  uint32_t type, VkDeviceMemory *out,
                                  VkDeviceSize *capacity) {
    VkMemoryAllocateInfo info;
    VkResult res;

    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.allocationSize = size;
    info.memoryTypeIndex = type;
    res = r->api.vkAllocateMemory(r->device, &info, NULL, out);
    if (res != VK_SUCCESS) {
        re_log_warn("vulkan: %llu-byte allocation failed (VkResult %d)",
                    (unsigned long long)size, (int)res);
        *out = VK_NULL_HANDLE;
        return false;
    }
    *capacity = size;
    return true;
}

/**
 * @brief Estimated image block size for a render extent.
 *
 * Color (RGBA16F) + depth (D32F) + the shadow array (D16 x cascades) plus
 * a reserve for frame-layer textures, rounded to the block granularity.
 *
 * @param w Render width.
 * @param h Render height.
 * @return Capacity in bytes.
 */
static VkDeviceSize image_block_bytes(uint32_t w, uint32_t h) {
    VkDeviceSize pixels = (VkDeviceSize)w * (VkDeviceSize)h;
    VkDeviceSize shadow = (VkDeviceSize)RE_SHADOW_DIM *
                          (VkDeviceSize)RE_SHADOW_DIM *
                          (VkDeviceSize)RE_SHADOW_CASCADES * 2u;
    VkDeviceSize bytes = pixels * 8u + pixels * 4u + shadow +
                         RE_IMAGE_RESERVE_BYTES;

    return (VkDeviceSize)re_align_up((size_t)bytes,
                                     (size_t)RE_ALLOC_GRANULARITY);
}

/**
 * @brief Records a buffer and its block offset.
 *
 * @param r Renderer.
 * @param buffer Buffer handle.
 * @param offset Block offset.
 * @param size Buffer size.
 * @return True on success (false only on host allocation failure).
 */
static bool record_buffer(struct ReRenderer *r, VkBuffer buffer,
                          VkDeviceSize offset, VkDeviceSize size) {
    if (r->buf_record_count == r->buf_record_capacity) {
        uint32_t next = r->buf_record_capacity == 0u ? 8u
                                                     : r->buf_record_capacity * 2u;
        ReVkBufferRecord *grown =
            realloc(r->buf_records, (size_t)next * sizeof *grown);
        if (grown == NULL) {
            return false;
        }
        r->buf_records = grown;
        r->buf_record_capacity = next;
    }
    r->buf_records[r->buf_record_count].buffer = buffer;
    r->buf_records[r->buf_record_count].offset = offset;
    r->buf_records[r->buf_record_count].size = size;
    r->buf_record_count++;
    return true;
}

/**
 * @brief Looks up the block placement of a buffer.
 *
 * @param r Renderer.
 * @param buffer Buffer handle.
 * @return Record pointer, or NULL when the buffer is not registered.
 */
static const ReVkBufferRecord *find_buffer_record(const struct ReRenderer *r,
                                                  VkBuffer buffer) {
    for (uint32_t i = 0u; i < r->buf_record_count; i++) {
        if (r->buf_records[i].buffer == buffer) {
            return &r->buf_records[i];
        }
    }
    return NULL;
}

/* --------------------------------------------------------------------- */
/* Exporting helpers used by re_renderer_frame.c                         */
/* --------------------------------------------------------------------- */

/** Creates the view matching an image. */
static VkResult create_image_view(struct ReRenderer *r, VkImage image,
                                  VkFormat fmt, uint32_t layers,
                                  VkImageAspectFlags aspect,
                                  VkImageView *view) {
    VkImageViewCreateInfo info;

    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image = image;
    info.viewType = layers > 1u ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                : VK_IMAGE_VIEW_TYPE_2D;
    info.format = fmt;
    info.subresourceRange.aspectMask = aspect;
    info.subresourceRange.baseMipLevel = 0u;
    info.subresourceRange.levelCount = 1u;
    info.subresourceRange.baseArrayLayer = 0u;
    info.subresourceRange.layerCount = layers;
    return r->api.vkCreateImageView(r->device, &info, NULL, view);
}

/**
 * @brief Binds an image at a fresh offset of the single image block.
 *
 * @param r Renderer with an allocated image block.
 * @param image Image to bind.
 * @return VkResult of the binding (block exhaustion or a disallowed
 *         memory type fail without leaking block space).
 */
static VkResult bind_image_memory(struct ReRenderer *r, VkImage image) {
    VkMemoryRequirements req;
    VkDeviceSize used = r->img_used;
    VkDeviceSize offset = 0u;
    VkResult res;

    r->api.vkGetImageMemoryRequirements(r->device, image, &req);
    if (r->img_memory_type >= 32u ||
        (req.memoryTypeBits & (UINT32_C(1) << r->img_memory_type)) == 0u) {
        re_log_warn("vulkan: image memory type %u is not allowed",
                    r->img_memory_type);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (!bump_allocate(&r->img_used, r->img_capacity, req.size, req.alignment,
                       &offset)) {
        re_log_warn("vulkan: image block exhausted (%llu of %llu bytes used)",
                    (unsigned long long)r->img_used,
                    (unsigned long long)r->img_capacity);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    res = r->api.vkBindImageMemory(r->device, image, r->gpu.img_mem, offset);
    if (res != VK_SUCCESS) {
        re_log_warn("vulkan: vkBindImageMemory failed (VkResult %d)", (int)res);
        r->img_used = used;
    }
    return res;
}

VkResult re_rr_create_image(struct ReRenderer *r, VkFormat fmt,
                            VkImageUsageFlags usage, uint32_t w, uint32_t h,
                            uint32_t layers, VkImage *img, VkImageView *view,
                            VkImageAspectFlags aspect) {
    VkImageCreateInfo info;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceSize used;
    VkResult res;

    if (r == NULL || r->is_null || img == NULL || view == NULL || w == 0u ||
        h == 0u || layers == 0u || r->device == VK_NULL_HANDLE) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    *img = VK_NULL_HANDLE;
    *view = VK_NULL_HANDLE;
    used = r->img_used;
    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = fmt;
    info.extent.width = w;
    info.extent.height = h;
    info.extent.depth = 1u;
    info.mipLevels = 1u;
    info.arrayLayers = layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    res = r->api.vkCreateImage(r->device, &info, NULL, &image);
    if (res != VK_SUCCESS) {
        re_log_warn("vulkan: vkCreateImage failed (VkResult %d)", (int)res);
        return res;
    }
    res = bind_image_memory(r, image);
    if (res != VK_SUCCESS) {
        r->api.vkDestroyImage(r->device, image, NULL);
        return res;
    }
    res = create_image_view(r, image, fmt, layers, aspect, view);
    if (res != VK_SUCCESS) {
        re_log_warn("vulkan: vkCreateImageView failed (VkResult %d)", (int)res);
        r->img_used = used;
        r->api.vkDestroyImage(r->device, image, NULL);
        return res;
    }
    *img = image;
    return VK_SUCCESS;
}

/**
 * @brief Binds a buffer at a fresh offset of the single buffer block.
 *
 * @param r Renderer with an allocated buffer block.
 * @param buffer Buffer to bind.
 * @param[out] offset Receives the block offset.
 * @return VkResult of the binding.
 */
static VkResult bind_buffer_memory(struct ReRenderer *r, VkBuffer buffer,
                                   VkDeviceSize *offset) {
    VkMemoryRequirements req;
    VkDeviceSize used = r->buf_used;
    VkResult res;

    r->api.vkGetBufferMemoryRequirements(r->device, buffer, &req);
    if (r->buf_memory_type >= 32u ||
        (req.memoryTypeBits & (UINT32_C(1) << r->buf_memory_type)) == 0u) {
        re_log_warn("vulkan: buffer memory type %u is not allowed",
                    r->buf_memory_type);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (!bump_allocate(&r->buf_used, r->buf_capacity, req.size, req.alignment,
                       offset)) {
        re_log_warn("vulkan: buffer block exhausted (%llu of %llu bytes used)",
                    (unsigned long long)r->buf_used,
                    (unsigned long long)r->buf_capacity);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    res = r->api.vkBindBufferMemory(r->device, buffer, r->buf.buf_mem, *offset);
    if (res != VK_SUCCESS) {
        re_log_warn("vulkan: vkBindBufferMemory failed (VkResult %d)", (int)res);
        r->buf_used = used;
    }
    return res;
}

VkResult re_rr_create_buffer(struct ReRenderer *r, VkDeviceSize size,
                             VkBufferUsageFlags usage, VkBuffer *buf) {
    VkBufferCreateInfo info;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize used;
    VkDeviceSize offset = 0u;
    VkResult res;

    if (r == NULL || r->is_null || buf == NULL || size == 0u ||
        r->device == VK_NULL_HANDLE) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    *buf = VK_NULL_HANDLE;
    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    res = r->api.vkCreateBuffer(r->device, &info, NULL, &buffer);
    if (res != VK_SUCCESS) {
        re_log_warn("vulkan: vkCreateBuffer failed (VkResult %d)", (int)res);
        return res;
    }
    used = r->buf_used;
    res = bind_buffer_memory(r, buffer, &offset);
    if (res != VK_SUCCESS) {
        r->api.vkDestroyBuffer(r->device, buffer, NULL);
        return res;
    }
    if (!record_buffer(r, buffer, offset, size)) {
        r->buf_used = used;
        r->api.vkDestroyBuffer(r->device, buffer, NULL);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    *buf = buffer;
    return VK_SUCCESS;
}

VkResult re_rr_upload(struct ReRenderer *r, const void *data, VkDeviceSize size,
                      VkDeviceSize dst_offset) {
    const ReVkBufferRecord *staging;
    VkMappedMemoryRange range;
    unsigned char *base;
    VkResult res;

    if (r == NULL || r->is_null || data == NULL || size == 0u ||
        r->device == VK_NULL_HANDLE) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (r->buf.mapped == NULL) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    staging = find_buffer_record(r, r->buf.staging);
    if (staging == NULL) {
        re_log_warn("vulkan: upload before the staging buffer was created");
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (dst_offset > staging->size || size > staging->size - dst_offset) {
        re_log_warn("vulkan: upload of %llu bytes at %llu exceeds staging (%llu)",
                    (unsigned long long)size, (unsigned long long)dst_offset,
                    (unsigned long long)staging->size);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    base = (unsigned char *)r->buf.mapped;
    memcpy(base + (size_t)(staging->offset + dst_offset), data, (size_t)size);
    /* The block is host-coherent in the common case; flushing the whole
     * block is the only offset that is always aligned. */
    memset(&range, 0, sizeof range);
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = r->buf.buf_mem;
    range.offset = 0u;
    range.size = VK_WHOLE_SIZE;
    res = r->api.vkFlushMappedMemoryRanges(r->device, 1u, &range);
    return res;
}

void re_rr_copy_to_buffer(struct ReRenderer *r, VkBuffer dst, VkDeviceSize size,
                          VkDeviceSize src_offset) {
    VkBufferCopy region;
    VkCommandBuffer cb;

    if (r == NULL || r->is_null || dst == VK_NULL_HANDLE || size == 0u ||
        r->cmds == NULL || r->frame_index >= r->frames_in_flight) {
        return;
    }
    cb = r->cmds[r->frame_index];
    memset(&region, 0, sizeof region);
    region.srcOffset = src_offset;
    region.dstOffset = 0u;
    region.size = size;
    r->api.vkCmdCopyBuffer(cb, r->buf.staging, dst, 1u, &region);
}

void re_rr_copy_to_image(struct ReRenderer *r, VkImage dst, uint32_t w,
                         uint32_t h, uint32_t layers, VkDeviceSize src_offset) {
    VkBufferImageCopy region;
    VkCommandBuffer cb;

    if (r == NULL || r->is_null || dst == VK_NULL_HANDLE || w == 0u || h == 0u ||
        layers == 0u || r->cmds == NULL ||
        r->frame_index >= r->frames_in_flight) {
        return;
    }
    cb = r->cmds[r->frame_index];
    memset(&region, 0, sizeof region);
    region.bufferOffset = src_offset;
    region.bufferRowLength = 0u;
    region.bufferImageHeight = 0u;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0u;
    region.imageSubresource.baseArrayLayer = 0u;
    region.imageSubresource.layerCount = layers;
    region.imageExtent.width = w;
    region.imageExtent.height = h;
    region.imageExtent.depth = 1u;
    r->api.vkCmdCopyBufferToImage(cb, r->buf.staging, dst,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1u,
                                  &region);
}

VkDeviceMemory re_rr_image_memory(struct ReRenderer *r) {
    return r == NULL || r->is_null ? VK_NULL_HANDLE : r->gpu.img_mem;
}

VkDeviceMemory re_rr_buffer_memory(struct ReRenderer *r) {
    return r == NULL || r->is_null ? VK_NULL_HANDLE : r->buf.buf_mem;
}

void *re_rr_mapped(struct ReRenderer *r) {
    return r == NULL || r->is_null ? NULL : r->buf.mapped;
}

void re_rr_cmd(struct ReRenderer *r, VkCommandBuffer *cb) {
    if (cb == NULL) {
        return;
    }
    *cb = VK_NULL_HANDLE;
    if (r == NULL || r->is_null || r->cmds == NULL ||
        r->frame_index >= r->frames_in_flight) {
        return;
    }
    *cb = r->cmds[r->frame_index];
}

/* --------------------------------------------------------------------- */
/* Instance and device creation                                          */
/* --------------------------------------------------------------------- */

/** Debug callback: every validation message becomes a warning line. */
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT types,
    const VkDebugUtilsMessengerCallbackDataEXT *data, void *user) {
    (void)severity;
    (void)types;
    (void)user;
    re_log_warn("[vk] %s", data != NULL && data->pMessage != NULL
                               ? data->pMessage
                               : "(message unavailable)");
    return VK_FALSE;
}

/** Checks one instance extension by name. */
static bool has_instance_extension(const ReVkApi *api, const char *name) {
    VkExtensionProperties *props;
    uint32_t count = 0u;
    bool found = false;

    if (api->vkEnumerateInstanceExtensionProperties(NULL, &count, NULL) !=
        VK_SUCCESS) {
        return false;
    }
    if (count == 0u) {
        return false;
    }
    props = malloc((size_t)count * sizeof *props);
    if (props == NULL) {
        return false;
    }
    if (api->vkEnumerateInstanceExtensionProperties(NULL, &count, props) ==
        VK_SUCCESS) {
        for (uint32_t i = 0u; i < count; i++) {
            if (strcmp(props[i].extensionName, name) == 0) {
                found = true;
                break;
            }
        }
    }
    free(props);
    return found;
}

/**
 * @brief Creates the VkInstance.
 *
 * @param r Renderer (title is echoed into the application info).
 * @param want_surface True when a surface will be created.
 * @param want_debug True when VK_EXT_debug_utils should be enabled.
 * @return True on success; on failure the instance stays VK_NULL_HANDLE.
 */
static bool create_instance(struct ReRenderer *r, bool want_surface,
                            bool want_debug) {
    VkApplicationInfo app;
    VkInstanceCreateInfo info;
    const char *extensions[3];
    uint32_t count = 0u;
    VkResult res;

    memset(&app, 0, sizeof app);
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = r->title;
    app.applicationVersion = VK_MAKE_API_VERSION(0u, 1u, 0u, 0u);
    app.pEngineName = "Real Engine";
    app.engineVersion = VK_MAKE_API_VERSION(0u, RE_VERSION_MAJOR,
                                            RE_VERSION_MINOR, RE_VERSION_PATCH);
    app.apiVersion = VK_API_VERSION_1_3;
    if (want_surface) {
        extensions[count++] = VK_KHR_SURFACE_EXTENSION_NAME;
#if defined(RE_HAS_X11)
        extensions[count++] = VK_KHR_XLIB_SURFACE_EXTENSION_NAME;
#endif
    }
    if (want_debug) {
        extensions[count++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    }
    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &app;
    info.enabledExtensionCount = count;
    info.ppEnabledExtensionNames = count > 0u ? extensions : NULL;
    res = r->api.vkCreateInstance(&info, NULL, &r->instance);
    if (res != VK_SUCCESS && want_debug) {
        re_log_warn("vulkan: instance with debug utils failed (%d), retrying",
                    (int)res);
        info.enabledExtensionCount = count - 1u;
        res = r->api.vkCreateInstance(&info, NULL, &r->instance);
        want_debug = false;
    }
    if (res != VK_SUCCESS) {
        re_log_warn("vulkan: vkCreateInstance failed (VkResult %d)", (int)res);
        r->instance = VK_NULL_HANDLE;
        return false;
    }
    r->debug_enabled = want_debug;
    re_log_info("vulkan: instance created (Vulkan 1.3, %u extensions)",
                info.enabledExtensionCount);
    return true;
}

/** Resolves the instance-level entries that are always needed. */
static bool load_instance_required(struct ReRenderer *r) {
    VkInstance instance = r->instance;
    bool ok = true;

#define RE_INSTANCE_REQUIRED(name)                                          \
    do {                                                                    \
        r->api.name = (PFN_##name)g_get_instance_proc(instance, #name);     \
        if (r->api.name == NULL) {                                          \
            re_log_warn("vulkan: instance entry %s unavailable", #name);    \
            ok = false;                                                     \
        }                                                                   \
    } while (0)

    RE_INSTANCE_REQUIRED(vkDestroyInstance);
    RE_INSTANCE_REQUIRED(vkEnumeratePhysicalDevices);
    RE_INSTANCE_REQUIRED(vkGetPhysicalDeviceProperties);
    RE_INSTANCE_REQUIRED(vkGetPhysicalDeviceProperties2);
    RE_INSTANCE_REQUIRED(vkGetPhysicalDeviceFeatures2);
    RE_INSTANCE_REQUIRED(vkGetPhysicalDeviceQueueFamilyProperties);
    RE_INSTANCE_REQUIRED(vkGetPhysicalDeviceMemoryProperties);
    RE_INSTANCE_REQUIRED(vkCreateDevice);

#undef RE_INSTANCE_REQUIRED
    return ok;
}

/** Resolves the surface entries needed by the windowed path. */
static bool load_instance_surface(struct ReRenderer *r) {
    VkInstance instance = r->instance;
    bool ok = true;

#define RE_INSTANCE_SURFACE(name)                                           \
    do {                                                                    \
        r->api.name = (PFN_##name)g_get_instance_proc(instance, #name);     \
        if (r->api.name == NULL) {                                          \
            re_log_warn("vulkan: surface entry %s unavailable", #name);     \
            ok = false;                                                     \
        }                                                                   \
    } while (0)

    RE_INSTANCE_SURFACE(vkDestroySurfaceKHR);
    RE_INSTANCE_SURFACE(vkGetPhysicalDeviceSurfaceSupportKHR);
    RE_INSTANCE_SURFACE(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    RE_INSTANCE_SURFACE(vkGetPhysicalDeviceSurfaceFormatsKHR);
    RE_INSTANCE_SURFACE(vkGetPhysicalDeviceSurfacePresentModesKHR);
#if defined(RE_HAS_X11)
    RE_INSTANCE_SURFACE(vkCreateXlibSurfaceKHR);
#endif

#undef RE_INSTANCE_SURFACE
    return ok;
}

/** Resolves the optional debug-utils entries (missing is not fatal). */
static void load_instance_optional(struct ReRenderer *r) {
    VkInstance instance = r->instance;

    r->api.vkCreateDebugUtilsMessengerEXT =
        (PFN_vkCreateDebugUtilsMessengerEXT)g_get_instance_proc(
            instance, "vkCreateDebugUtilsMessengerEXT");
    r->api.vkDestroyDebugUtilsMessengerEXT =
        (PFN_vkDestroyDebugUtilsMessengerEXT)g_get_instance_proc(
            instance, "vkDestroyDebugUtilsMessengerEXT");
}

/** Installs the validation-message callback when the extension is loaded. */
static void create_messenger(struct ReRenderer *r) {
    VkDebugUtilsMessengerCreateInfoEXT info;
    VkResult res;

    if (!r->debug_enabled || r->api.vkCreateDebugUtilsMessengerEXT == NULL) {
        return;
    }
    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debug_callback;
    res = r->api.vkCreateDebugUtilsMessengerEXT(r->instance, &info, NULL,
                                                &r->messenger);
    if (res != VK_SUCCESS) {
        re_log_warn("vulkan: debug messenger not created (VkResult %d)",
                    (int)res);
        r->messenger = VK_NULL_HANDLE;
        return;
    }
    re_log_info("vulkan: validation message callback installed");
}

/**
 * @brief Picks the graphics family and a present-capable family.
 *
 * @param api Resolved function table.
 * @param phys Physical device.
 * @param surface Surface, or VK_NULL_HANDLE when headless.
 * @param[out] gfx_family Graphics family index.
 * @param[out] present_family Present family index.
 * @return True when both families exist.
 */
static bool find_queue_families(const ReVkApi *api, VkPhysicalDevice phys,
                                VkSurfaceKHR surface, uint32_t *gfx_family,
                                uint32_t *present_family) {
    VkQueueFamilyProperties *props;
    uint32_t count = 0u;
    uint32_t gfx = 0u;
    uint32_t present = 0u;
    bool have_gfx = false;
    bool have_present = false;

    api->vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, NULL);
    if (count == 0u) {
        return false;
    }
    props = malloc((size_t)count * sizeof *props);
    if (props == NULL) {
        return false;
    }
    api->vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, props);
    for (uint32_t i = 0u; i < count; i++) {
        if (!have_gfx && (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0u) {
            have_gfx = true;
            gfx = i;
        }
        if (surface == VK_NULL_HANDLE || have_present) {
            continue;
        }
        {
            VkBool32 supported = VK_FALSE;
            VkResult res = api->vkGetPhysicalDeviceSurfaceSupportKHR(
                phys, i, surface, &supported);
            if (res == VK_SUCCESS && supported != VK_FALSE) {
                have_present = true;
                present = i;
            }
        }
    }
    free(props);
    if (surface == VK_NULL_HANDLE) {
        present = gfx;
        have_present = have_gfx;
    }
    if (!have_gfx || !have_present) {
        return false;
    }
    *gfx_family = gfx;
    *present_family = present;
    return true;
}

/**
 * @brief Queries the feature chain and verifies the hard requirements.
 *
 * @param api Function table.
 * @param phys Physical device.
 * @param[out] q12 Queried Vulkan 1.2 features.
 * @param[out] q13 Queried Vulkan 1.3 features.
 * @return True when synchronization2, dynamic rendering, descriptor
 *         indexing and non-uniform sampled arrays are all supported.
 */
static bool query_device_features(const ReVkApi *api, VkPhysicalDevice phys,
                                  VkPhysicalDeviceVulkan12Features *q12,
                                  VkPhysicalDeviceVulkan13Features *q13) {
    VkPhysicalDeviceFeatures2 core;

    memset(q13, 0, sizeof *q13);
    q13->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    memset(q12, 0, sizeof *q12);
    q12->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    q12->pNext = q13;
    memset(&core, 0, sizeof core);
    core.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    core.pNext = q12;
    api->vkGetPhysicalDeviceFeatures2(phys, &core);
    core.pNext = q12;
    q12->pNext = q13;
    q13->pNext = NULL;
    if (q13->synchronization2 == VK_FALSE || q13->dynamicRendering == VK_FALSE) {
        re_log_warn("vulkan: device lacks synchronization2 or dynamic rendering");
        return false;
    }
    if (q12->descriptorIndexing == VK_FALSE ||
        q12->shaderSampledImageArrayNonUniformIndexing == VK_FALSE) {
        re_log_warn("vulkan: device lacks the bindless descriptor features");
        return false;
    }
    return true;
}

/**
 * @brief Builds the pNext chain enabling exactly the requested features.
 *
 * Optional descriptor features are enabled only when the device reported
 * them; the four hard requirements are already verified by the query.
 */
static void fill_enable_chain(const VkPhysicalDeviceVulkan12Features *q12,
                              const VkPhysicalDeviceVulkan13Features *q13,
                              VkPhysicalDeviceFeatures2 *core,
                              VkPhysicalDeviceVulkan12Features *e12,
                              VkPhysicalDeviceVulkan13Features *e13) {
    VkBool32 partial = q12->descriptorBindingPartiallyBound;
    VkBool32 sampled_after = q12->descriptorBindingSampledImageUpdateAfterBind;
    VkBool32 variable = q12->descriptorBindingVariableDescriptorCount;
    VkBool32 runtime = q12->runtimeDescriptorArray;
    VkBool32 maintenance4 = q13->maintenance4;

    memset(e13, 0, sizeof *e13);
    e13->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    e13->synchronization2 = VK_TRUE;
    e13->dynamicRendering = VK_TRUE;
    e13->maintenance4 = maintenance4;
    memset(e12, 0, sizeof *e12);
    e12->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    e12->pNext = e13;
    e12->descriptorIndexing = VK_TRUE;
    e12->shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    e12->descriptorBindingPartiallyBound = partial;
    e12->descriptorBindingSampledImageUpdateAfterBind = sampled_after;
    e12->descriptorBindingVariableDescriptorCount = variable;
    e12->runtimeDescriptorArray = runtime;
    memset(core, 0, sizeof *core);
    core->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    core->pNext = e12;
}

/**
 * @brief Selects the first physical device that satisfies the contract.
 *
 * @param r Renderer with a live instance and (optionally) a surface.
 * @return True when r->phys, r->gfx_family and r->present_family are set.
 */
static bool pick_physical_device(struct ReRenderer *r) {
    VkPhysicalDevice *devices;
    uint32_t count = 0u;
    bool found = false;

    if (r->api.vkEnumeratePhysicalDevices(r->instance, &count, NULL) !=
            VK_SUCCESS ||
        count == 0u) {
        re_log_warn("vulkan: no physical device reported");
        return false;
    }
    devices = malloc((size_t)count * sizeof *devices);
    if (devices == NULL) {
        return false;
    }
    if (r->api.vkEnumeratePhysicalDevices(r->instance, &count, devices) !=
        VK_SUCCESS) {
        free(devices);
        return false;
    }
    for (uint32_t i = 0u; i < count && !found; i++) {
        VkPhysicalDeviceProperties props;
        VkPhysicalDeviceVulkan12Features q12;
        VkPhysicalDeviceVulkan13Features q13;
        uint32_t gfx = 0u;
        uint32_t present = 0u;

        if (!find_queue_families(&r->api, devices[i], r->surface, &gfx,
                                 &present)) {
            continue;
        }
        if (!query_device_features(&r->api, devices[i], &q12, &q13)) {
            continue;
        }
        r->phys = devices[i];
        r->gfx_family = gfx;
        r->present_family = present;
        r->api.vkGetPhysicalDeviceProperties(devices[i], &props);
        re_log_info("vulkan: device %u: %s (API %u.%u.%u)", i,
                    props.deviceName,
                    (unsigned)(VK_API_VERSION_MAJOR(props.apiVersion)),
                    (unsigned)(VK_API_VERSION_MINOR(props.apiVersion)),
                    (unsigned)(VK_API_VERSION_PATCH(props.apiVersion)));
        found = true;
    }
    free(devices);
    if (!found) {
        re_log_warn("vulkan: no device with graphics, present and the "
                    "required 1.3 features");
    }
    return found;
}

/** Creates the device, loads its entries and fetches the queues. */
static bool create_device(struct ReRenderer *r) {
    VkDeviceQueueCreateInfo queues[2];
    VkPhysicalDeviceVulkan12Features q12;
    VkPhysicalDeviceVulkan13Features q13;
    VkPhysicalDeviceVulkan12Features e12;
    VkPhysicalDeviceVulkan13Features e13;
    VkPhysicalDeviceFeatures2 core;
    VkDeviceCreateInfo info;
    const char *extensions[1];
    float priority = 1.0f;
    uint32_t queue_count = 1u;
    VkResult res;

    if (!query_device_features(&r->api, r->phys, &q12, &q13)) {
        return false;
    }
    fill_enable_chain(&q12, &q13, &core, &e12, &e13);
    memset(queues, 0, sizeof queues);
    queues[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queues[0].queueFamilyIndex = r->gfx_family;
    queues[0].queueCount = 1u;
    queues[0].pQueuePriorities = &priority;
    if (r->present_family != r->gfx_family) {
        queues[1] = queues[0];
        queues[1].queueFamilyIndex = r->present_family;
        queue_count = 2u;
    }
    extensions[0] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.pNext = &core;
    info.queueCreateInfoCount = queue_count;
    info.pQueueCreateInfos = queues;
    info.enabledExtensionCount = r->surface != VK_NULL_HANDLE ? 1u : 0u;
    info.ppEnabledExtensionNames = info.enabledExtensionCount > 0u ? extensions
                                                                   : NULL;
    res = r->api.vkCreateDevice(r->phys, &info, NULL, &r->device);
    if (res != VK_SUCCESS) {
        re_log_warn("vulkan: vkCreateDevice failed (VkResult %d)", (int)res);
        r->device = VK_NULL_HANDLE;
        return false;
    }
    re_vk_load_device(&r->api, r->instance, r->device);
    if (r->api.vkGetDeviceQueue == NULL || r->api.vkDestroyDevice == NULL) {
        re_log_warn("vulkan: device queue entry points unavailable");
        return false;
    }
    r->api.vkGetDeviceQueue(r->device, r->gfx_family, 0u, &r->gfx);
    if (r->present_family == r->gfx_family) {
        r->present = r->gfx;
    } else {
        r->api.vkGetDeviceQueue(r->device, r->present_family, 0u, &r->present);
    }
    re_log_info("vulkan: device created (graphics family %u, present family %u)",
                r->gfx_family, r->present_family);
    return true;
}

/**
 * @brief Returns true when every entry this file calls is resolved.
 *
 * A loader that misses a core entry would otherwise turn into a crash on
 * the first frame; fail creation instead.
 */
static bool device_entries_ok(const ReVkApi *api) {
    return api->vkDestroyDevice != NULL && api->vkGetDeviceQueue != NULL &&
           api->vkDeviceWaitIdle != NULL && api->vkQueueSubmit != NULL &&
           api->vkQueuePresentKHR != NULL &&
           api->vkCreateSwapchainKHR != NULL &&
           api->vkDestroySwapchainKHR != NULL &&
           api->vkGetSwapchainImagesKHR != NULL &&
           api->vkAcquireNextImageKHR != NULL && api->vkCreateImage != NULL &&
           api->vkCreateImageView != NULL && api->vkDestroyImageView != NULL &&
           api->vkAllocateMemory != NULL && api->vkFreeMemory != NULL &&
           api->vkCreateBuffer != NULL && api->vkDestroyBuffer != NULL &&
           api->vkMapMemory != NULL && api->vkUnmapMemory != NULL &&
           api->vkCreateCommandPool != NULL &&
           api->vkDestroyCommandPool != NULL &&
           api->vkAllocateCommandBuffers != NULL &&
           api->vkCreateSemaphore != NULL && api->vkCreateFence != NULL &&
           api->vkWaitForFences != NULL && api->vkResetFences != NULL;
}

/* --------------------------------------------------------------------- */
/* Frame synchronisation and memory blocks                               */
/* --------------------------------------------------------------------- */

/** Creates the command pool and the per-frame command buffers. */
static bool create_command_buffers(struct ReRenderer *r) {
    VkCommandPoolCreateInfo pool_info;
    VkCommandBufferAllocateInfo cmd_info;

    memset(&pool_info, 0, sizeof pool_info);
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = r->gfx_family;
    if (r->api.vkCreateCommandPool(r->device, &pool_info, NULL, &r->cpool) !=
        VK_SUCCESS) {
        re_log_warn("vulkan: vkCreateCommandPool failed");
        return false;
    }
    r->cmds = calloc((size_t)RE_FRAMES_IN_FLIGHT, sizeof *r->cmds);
    if (r->cmds == NULL) {
        return false;
    }
    memset(&cmd_info, 0, sizeof cmd_info);
    cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_info.commandPool = r->cpool;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = RE_FRAMES_IN_FLIGHT;
    if (r->api.vkAllocateCommandBuffers(r->device, &cmd_info, r->cmds) !=
        VK_SUCCESS) {
        re_log_warn("vulkan: vkAllocateCommandBuffers failed");
        return false;
    }
    return true;
}

/** Creates the per-frame semaphores and signaled fences. */
static bool create_frame_sync(struct ReRenderer *r) {
    VkSemaphoreCreateInfo sem_info;
    VkFenceCreateInfo fence_info;

    r->acquire = calloc((size_t)RE_FRAMES_IN_FLIGHT, sizeof *r->acquire);
    r->release = calloc((size_t)RE_FRAMES_IN_FLIGHT, sizeof *r->release);
    r->fence = calloc((size_t)RE_FRAMES_IN_FLIGHT, sizeof *r->fence);
    if (r->acquire == NULL || r->release == NULL || r->fence == NULL) {
        return false;
    }
    memset(&sem_info, 0, sizeof sem_info);
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    memset(&fence_info, 0, sizeof fence_info);
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    /* Signaled so the first wait of every slot returns immediately. */
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (uint32_t i = 0u; i < RE_FRAMES_IN_FLIGHT; i++) {
        if (r->api.vkCreateSemaphore(r->device, &sem_info, NULL,
                                     &r->acquire[i]) != VK_SUCCESS ||
            r->api.vkCreateSemaphore(r->device, &sem_info, NULL,
                                     &r->release[i]) != VK_SUCCESS ||
            r->api.vkCreateFence(r->device, &fence_info, NULL, &r->fence[i]) !=
                VK_SUCCESS) {
            re_log_warn("vulkan: frame sync object creation failed");
            return false;
        }
    }
    return true;
}

/** Creates the command buffers, semaphores and fences of the frame loop. */
static bool create_sync(struct ReRenderer *r) {
    if (!create_command_buffers(r)) {
        return false;
    }
    if (!create_frame_sync(r)) {
        return false;
    }
    r->frames_in_flight = RE_FRAMES_IN_FLIGHT;
    return true;
}

/** Allocates the two device-memory blocks and maps the buffer block. */
static bool allocate_blocks(struct ReRenderer *r) {
    VkPhysicalDeviceMemoryProperties props;
    VkDeviceSize image_bytes = image_block_bytes(r->width, r->height);
    void *mapped = NULL;

    r->api.vkGetPhysicalDeviceMemoryProperties(r->phys, &props);
    if (!find_memory_type(&props, UINT32_MAX, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          0u, &r->img_memory_type)) {
        re_log_warn("vulkan: no device-local memory type for images");
        return false;
    }
    if (!allocate_memory_block(r, image_bytes, r->img_memory_type,
                               &r->gpu.img_mem, &r->img_capacity)) {
        return false;
    }
    if (!find_memory_type(&props, UINT32_MAX,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          &r->buf_memory_type)) {
        re_log_warn("vulkan: no host-visible memory type for buffers");
        return false;
    }
    if (!allocate_memory_block(r, RE_BUFFER_BLOCK_BYTES, r->buf_memory_type,
                               &r->buf.buf_mem, &r->buf_capacity)) {
        return false;
    }
    if (r->api.vkMapMemory(r->device, r->buf.buf_mem, 0u, VK_WHOLE_SIZE, 0u,
                           &mapped) != VK_SUCCESS) {
        re_log_warn("vulkan: vkMapMemory on the buffer block failed");
        return false;
    }
    r->buf.mapped = mapped;
    re_log_info("vulkan: memory blocks: images %llu MiB, buffers %llu MiB",
                (unsigned long long)(r->img_capacity / (1024u * 1024u)),
                (unsigned long long)(r->buf_capacity / (1024u * 1024u)));
    return true;
}

/**
 * @brief Creates the HDR colour, depth and shadow render targets.
 *
 * The frame layer records into these images but does not own them:
 * re_renderer.c allocates them in the same image block as the textures
 * and destroys them before the block is freed.
 *
 * @param r Renderer with an allocated image block.
 * @return True on success.
 */
static bool create_render_targets(struct ReRenderer *r) {
    if (re_rr_create_image(r, VK_FORMAT_R16G16B16A16_SFLOAT,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT,
                           r->width, r->height, 1u, &r->gpu.color,
                           &r->gpu.color_view,
                           VK_IMAGE_ASPECT_COLOR_BIT) != VK_SUCCESS) {
        re_log_error("renderer: HDR colour target creation failed");
        return false;
    }
    if (re_rr_create_image(r, VK_FORMAT_D32_SFLOAT,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT,
                           r->width, r->height, 1u, &r->gpu.depth,
                           &r->gpu.depth_view,
                           VK_IMAGE_ASPECT_DEPTH_BIT) != VK_SUCCESS) {
        re_log_error("renderer: depth target creation failed");
        return false;
    }
    if (re_rr_create_image(r, VK_FORMAT_D16_UNORM,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT,
                           RE_SHADOW_DIM, RE_SHADOW_DIM, RE_SHADOW_CASCADES,
                           &r->gpu.shadow, &r->gpu.shadow_view,
                           VK_IMAGE_ASPECT_DEPTH_BIT) != VK_SUCCESS) {
        re_log_error("renderer: shadow target creation failed");
        return false;
    }
    return true;
}

/** @brief Destroys the render targets created by create_render_targets. */
static void destroy_render_targets(struct ReRenderer *r) {
    if (r->api.vkDestroyImageView != NULL) {
        if (r->gpu.color_view != VK_NULL_HANDLE) {
            r->api.vkDestroyImageView(r->device, r->gpu.color_view, NULL);
        }
        if (r->gpu.depth_view != VK_NULL_HANDLE) {
            r->api.vkDestroyImageView(r->device, r->gpu.depth_view, NULL);
        }
        if (r->gpu.shadow_view != VK_NULL_HANDLE) {
            r->api.vkDestroyImageView(r->device, r->gpu.shadow_view, NULL);
        }
    }
    if (r->api.vkDestroyImage != NULL) {
        if (r->gpu.color != VK_NULL_HANDLE) {
            r->api.vkDestroyImage(r->device, r->gpu.color, NULL);
        }
        if (r->gpu.depth != VK_NULL_HANDLE) {
            r->api.vkDestroyImage(r->device, r->gpu.depth, NULL);
        }
        if (r->gpu.shadow != VK_NULL_HANDLE) {
            r->api.vkDestroyImage(r->device, r->gpu.shadow, NULL);
        }
    }
    r->gpu.color = VK_NULL_HANDLE;
    r->gpu.color_view = VK_NULL_HANDLE;
    r->gpu.depth = VK_NULL_HANDLE;
    r->gpu.depth_view = VK_NULL_HANDLE;
    r->gpu.shadow = VK_NULL_HANDLE;
    r->gpu.shadow_view = VK_NULL_HANDLE;
}

/* --------------------------------------------------------------------- */
/* X11 window and surface                                                */
/* --------------------------------------------------------------------- */

#if defined(RE_HAS_X11)
/**
 * @brief Opens the display and creates the window.
 *
 * Failure is not fatal: the renderer continues headless.
 *
 * @param r Renderer with width, height and title set.
 * @return True when the window exists.
 */
static bool create_window(struct ReRenderer *r) {
    Display *display = XOpenDisplay(NULL);
    int screen;
    Window root;
    Window window;

    if (display == NULL) {
        re_log_warn("x11: cannot open the display, running headless");
        return false;
    }
    screen = DefaultScreen(display);
    root = RootWindow(display, screen);
    window = XCreateSimpleWindow(display, root,
                                 r->window_x >= 0 ? r->window_x : 0,
                                 r->window_y >= 0 ? r->window_y : 0,
                                 (unsigned)r->width, (unsigned)r->height, 0,
                                 BlackPixel(display, screen),
                                 BlackPixel(display, screen));
    if (window == 0u) {
        re_log_warn("x11: XCreateSimpleWindow failed, running headless");
        XCloseDisplay(display);
        return false;
    }
    XSelectInput(display, window,
                 KeyPressMask | KeyReleaseMask | FocusChangeMask |
                     StructureNotifyMask);
    {
        Bool supported = False;
        (void)XkbSetDetectableAutoRepeat(display, True, &supported);
    }
    r->wm_delete = XInternAtom(display, "WM_DELETE_WINDOW", False);
    if (r->wm_delete != 0u) {
        (void)XSetWMProtocols(display, window, &r->wm_delete, 1);
    }
    XStoreName(display, window, r->title);
    XMapRaised(display, window);
    XFlush(display);
    r->display = display;
    r->window = window;
    re_log_info("x11: window %ux%u mapped", r->width, r->height);
    return true;
}

/**
 * @brief Maps an X11 keysym onto the engine's ReKey table.
 * @param sym Keysym from the event.
 * @return Index into ReInput::down, or -1 when the key is not tracked.
 */
static int key_index(KeySym sym) {
    switch (sym) {
    case XK_w: case XK_W: return RE_KEY_W;
    case XK_a: case XK_A: return RE_KEY_A;
    case XK_s: case XK_S: return RE_KEY_S;
    case XK_d: case XK_D: return RE_KEY_D;
    case XK_Up: return RE_KEY_UP;
    case XK_Down: return RE_KEY_DOWN;
    case XK_Left: return RE_KEY_LEFT;
    case XK_Right: return RE_KEY_RIGHT;
    case XK_space: return RE_KEY_SPACE;
    case XK_Shift_L: return RE_KEY_LSHIFT;
    case XK_Shift_R: return RE_KEY_RSHIFT;
    case XK_Return: case XK_KP_Enter: return RE_KEY_ENTER;
    case XK_r: case XK_R: return RE_KEY_R;
    default: return -1;
    }
}

/**
 * @brief Drains pending X11 events into the key/quit state.
 *
 * Auto-repeat is enabled in create_window, so a held key produces only
 * repeated KeyPress events and a single KeyRelease when actually let go.
 * @param r Renderer; no-op without a window.
 */
static void pump_window_events(struct ReRenderer *r) {
    if (r->display == NULL) {
        return;
    }
    while (XPending(r->display) > 0) {
        XEvent event;

        XNextEvent(r->display, &event);
        switch (event.type) {
        case KeyPress:
        case KeyRelease: {
            const KeySym sym = XLookupKeysym(&event.xkey, 0);
            const int key = key_index(sym);
            if (key >= 0) {
                r->keys[key] = event.type == KeyPress;
            } else if (sym == XK_Escape && event.type == KeyPress) {
                r->quit_requested = true;
            }
            break;
        }
        case FocusOut:
            memset(r->keys, 0, sizeof r->keys);
            break;
        case ClientMessage:
            if ((Atom)event.xclient.data.l[0] == r->wm_delete) {
                r->quit_requested = true;
            }
            break;
        case DestroyNotify:
            r->quit_requested = true;
            break;
        default:
            break;
        }
    }
}
#endif /* RE_HAS_X11 */

/** Creates the presentation surface (no-op headless). */
static bool create_surface(struct ReRenderer *r) {
#if defined(RE_HAS_X11)
    VkXlibSurfaceCreateInfoKHR info;
    VkResult res;

    if (r->display == NULL || r->window == 0u ||
        r->api.vkCreateXlibSurfaceKHR == NULL) {
        return false;
    }
    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
    info.dpy = r->display;
    info.window = r->window;
    res = r->api.vkCreateXlibSurfaceKHR(r->instance, &info, NULL, &r->surface);
    if (res != VK_SUCCESS) {
        re_log_warn("vulkan: vkCreateXlibSurfaceKHR failed (VkResult %d)",
                    (int)res);
        r->surface = VK_NULL_HANDLE;
        return false;
    }
    re_log_info("vulkan: presentation surface created");
    return true;
#else
    (void)r;
    return false;
#endif
}

/* --------------------------------------------------------------------- */
/* Swapchain                                                             */
/* --------------------------------------------------------------------- */

/** Chooses the swap extent (surface-defined or clamped to the config). */
static VkExtent2D choose_extent(const VkSurfaceCapabilitiesKHR *caps,
                                uint32_t width, uint32_t height) {
    VkExtent2D extent;

    if (caps->currentExtent.width != UINT32_MAX) {
        return caps->currentExtent;
    }
    extent.width = width;
    extent.height = height;
    if (extent.width < caps->minImageExtent.width) {
        extent.width = caps->minImageExtent.width;
    }
    if (extent.width > caps->maxImageExtent.width) {
        extent.width = caps->maxImageExtent.width;
    }
    if (extent.height < caps->minImageExtent.height) {
        extent.height = caps->minImageExtent.height;
    }
    if (extent.height > caps->maxImageExtent.height) {
        extent.height = caps->maxImageExtent.height;
    }
    return extent;
}

/** Picks B8G8R8A8_UNORM + SRGB_NONLINEAR, else the first reported format. */
static bool choose_format(struct ReRenderer *r, VkSurfaceFormatKHR *out) {
    VkSurfaceFormatKHR *formats;
    uint32_t count = 0u;

    if (r->api.vkGetPhysicalDeviceSurfaceFormatsKHR(r->phys, r->surface, &count,
                                                    NULL) != VK_SUCCESS ||
        count == 0u) {
        return false;
    }
    formats = malloc((size_t)count * sizeof *formats);
    if (formats == NULL) {
        return false;
    }
    if (r->api.vkGetPhysicalDeviceSurfaceFormatsKHR(r->phys, r->surface, &count,
                                                    formats) != VK_SUCCESS) {
        free(formats);
        return false;
    }
    *out = formats[0];
    if (out->format == VK_FORMAT_UNDEFINED) {
        /* Per the WSI spec a single UNDEFINED entry means "any format". */
        out->format = VK_FORMAT_B8G8R8A8_UNORM;
        out->colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    }
    for (uint32_t i = 0u; i < count; i++) {
        if (formats[i].format == VK_FORMAT_B8G8R8A8_UNORM &&
            formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            *out = formats[i];
            break;
        }
    }
    free(formats);
    return true;
}

/** Picks a composite alpha mode the surface supports. */
static VkCompositeAlphaFlagBitsKHR
choose_composite_alpha(const VkSurfaceCapabilitiesKHR *caps) {
    if ((caps->supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) !=
        0u) {
        return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    }
    if ((caps->supportedCompositeAlpha &
         VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) != 0u) {
        return VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    }
    return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
}

/** Creates the swapchain images views and stores them in the renderer. */
static bool create_swapchain_views(struct ReRenderer *r) {
    uint32_t count = 0u;

    if (r->api.vkGetSwapchainImagesKHR(r->device, r->swapchain, &count, NULL) !=
            VK_SUCCESS ||
        count == 0u) {
        re_log_warn("vulkan: swapchain reports no images");
        return false;
    }
    r->sw_images = malloc((size_t)count * sizeof *r->sw_images);
    r->sw_views = calloc((size_t)count, sizeof *r->sw_views);
    if (r->sw_images == NULL || r->sw_views == NULL) {
        return false;
    }
    if (r->api.vkGetSwapchainImagesKHR(r->device, r->swapchain, &count,
                                       r->sw_images) != VK_SUCCESS) {
        re_log_warn("vulkan: vkGetSwapchainImagesKHR failed");
        return false;
    }
    r->sw_count = count;
    for (uint32_t i = 0u; i < count; i++) {
        if (create_image_view(r, r->sw_images[i], r->sw_format, 1u,
                              VK_IMAGE_ASPECT_COLOR_BIT,
                              &r->sw_views[i]) != VK_SUCCESS) {
            re_log_warn("vulkan: swapchain view %u failed", i);
            return false;
        }
    }
    return true;
}

/** Destroys the swapchain and everything hanging off it. */
static void destroy_swapchain(struct ReRenderer *r) {
    if (r->sw_views != NULL && r->api.vkDestroyImageView != NULL) {
        for (uint32_t i = 0u; i < r->sw_count; i++) {
            if (r->sw_views[i] != VK_NULL_HANDLE) {
                r->api.vkDestroyImageView(r->device, r->sw_views[i], NULL);
            }
        }
    }
    free(r->sw_views);
    free(r->sw_images);
    r->sw_views = NULL;
    r->sw_images = NULL;
    r->sw_count = 0u;
    if (r->swapchain != VK_NULL_HANDLE &&
        r->api.vkDestroySwapchainKHR != NULL) {
        r->api.vkDestroySwapchainKHR(r->device, r->swapchain, NULL);
    }
    r->swapchain = VK_NULL_HANDLE;
}

/**
 * @brief Fills a swapchain create info for the current surface.
 *
 * @param r Renderer with family indices and the config extent.
 * @param caps Surface capabilities.
 * @param format Chosen surface format.
 * @param families Scratch storage the info points at (2 entries).
 * @param[out] info Receives the filled create info.
 */
static void fill_swapchain_info(struct ReRenderer *r,
                                const VkSurfaceCapabilitiesKHR *caps,
                                const VkSurfaceFormatKHR *format,
                                uint32_t families[2],
                                VkSwapchainCreateInfoKHR *info) {
    uint32_t requested = caps->minImageCount + 1u;

    if (caps->maxImageCount > 0u && requested > caps->maxImageCount) {
        requested = caps->maxImageCount;
    }
    families[0] = r->gfx_family;
    families[1] = r->present_family;
    memset(info, 0, sizeof *info);
    info->sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info->surface = r->surface;
    info->minImageCount = requested;
    info->imageFormat = format->format;
    info->imageColorSpace = format->colorSpace;
    info->imageExtent = choose_extent(caps, r->width, r->height);
    info->imageArrayLayers = 1u;
    info->imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (r->gfx_family == r->present_family) {
        info->imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    } else {
        info->imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        info->queueFamilyIndexCount = 2u;
        info->pQueueFamilyIndices = families;
    }
    info->preTransform = caps->currentTransform;
    info->compositeAlpha = choose_composite_alpha(caps);
    info->presentMode = VK_PRESENT_MODE_FIFO_KHR;
    info->clipped = VK_TRUE;
    info->oldSwapchain = VK_NULL_HANDLE;
}

/**
 * @brief Creates the swapchain for the current surface.
 *
 * @param r Renderer with a surface and a device.
 * @return True on success.
 */
static bool build_swapchain(struct ReRenderer *r) {
    VkSurfaceCapabilitiesKHR caps;
    VkSwapchainCreateInfoKHR info;
    VkSurfaceFormatKHR format;
    uint32_t families[2];

    if (r->api.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(r->phys, r->surface,
                                                         &caps) != VK_SUCCESS) {
        re_log_warn("vulkan: surface capabilities unavailable");
        return false;
    }
    if (!choose_format(r, &format)) {
        re_log_warn("vulkan: surface has no usable format");
        return false;
    }
    fill_swapchain_info(r, &caps, &format, families, &info);
    if (r->api.vkCreateSwapchainKHR(r->device, &info, NULL, &r->swapchain) !=
        VK_SUCCESS) {
        re_log_warn("vulkan: vkCreateSwapchainKHR failed");
        r->swapchain = VK_NULL_HANDLE;
        return false;
    }
    r->sw_format = format.format;
    r->sw_extent = info.imageExtent;
    if (!create_swapchain_views(r)) {
        destroy_swapchain(r);
        return false;
    }
    re_log_info("vulkan: swapchain %ux%u, %u images", r->sw_extent.width,
                r->sw_extent.height, r->sw_count);
    return true;
}

/** Recreates the swapchain after an out-of-date result. */
static bool rebuild_swapchain(struct ReRenderer *r) {
    if (r->surface == VK_NULL_HANDLE || r->device == VK_NULL_HANDLE) {
        return false;
    }
    (void)r->api.vkDeviceWaitIdle(r->device);
    destroy_swapchain(r);
    re_log_info("vulkan: rebuilding the swapchain");
    return build_swapchain(r);
}

/* --------------------------------------------------------------------- */
/* Frame submission                                                      */
/* --------------------------------------------------------------------- */

/** Waits for the fence of the current slot. */
static bool wait_fence(struct ReRenderer *r) {
    VkResult res = r->api.vkWaitForFences(r->device, 1u,
                                          &r->fence[r->frame_index], VK_TRUE,
                                          UINT64_MAX);

    if (res != VK_SUCCESS) {
        re_log_warn("vulkan: vkWaitForFences failed (VkResult %d)", (int)res);
        r->device_lost = true;
        return false;
    }
    return true;
}

/** Acquires the next swapchain image, rebuilding the chain once if stale. */
static bool acquire_image(struct ReRenderer *r) {
    for (uint32_t attempt = 0u; attempt < 2u; attempt++) {
        uint32_t index = 0u;
        VkResult res = r->api.vkAcquireNextImageKHR(
            r->device, r->swapchain, UINT64_MAX, r->acquire[r->frame_index],
            VK_NULL_HANDLE, &index);

        if (res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR) {
            r->image_index = index;
            return true;
        }
        if (res != VK_ERROR_OUT_OF_DATE_KHR) {
            re_log_warn("vulkan: vkAcquireNextImageKHR failed (VkResult %d)",
                        (int)res);
            r->device_lost = true;
            return false;
        }
        if (!rebuild_swapchain(r)) {
            r->device_lost = true;
            return false;
        }
    }
    re_log_warn("vulkan: swapchain still out of date after a rebuild");
    return false;
}

/** Submits the frame command buffer of the current slot. */
static bool submit_frame(struct ReRenderer *r) {
    VkSubmitInfo info;
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkResult res;

    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    if (r->surface != VK_NULL_HANDLE) {
        info.waitSemaphoreCount = 1u;
        info.pWaitSemaphores = &r->acquire[r->frame_index];
        info.pWaitDstStageMask = &wait_stage;
        info.signalSemaphoreCount = 1u;
        info.pSignalSemaphores = &r->release[r->frame_index];
    }
    info.commandBufferCount = 1u;
    info.pCommandBuffers = &r->cmds[r->frame_index];
    res = r->api.vkResetFences(r->device, 1u, &r->fence[r->frame_index]);
    if (res != VK_SUCCESS) {
        re_log_warn("vulkan: vkResetFences failed (VkResult %d)", (int)res);
        r->device_lost = true;
        return false;
    }
    res = r->api.vkQueueSubmit(r->gfx, 1u, &info, r->fence[r->frame_index]);
    if (res != VK_SUCCESS) {
        re_log_warn("vulkan: vkQueueSubmit failed (VkResult %d)", (int)res);
        r->device_lost = true;
        return false;
    }
    return true;
}

/** Presents the current image, rebuilding the chain when it went stale. */
static bool present_frame(struct ReRenderer *r) {
    VkPresentInfoKHR info;
    VkResult res;

    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1u;
    info.pWaitSemaphores = &r->release[r->frame_index];
    info.swapchainCount = 1u;
    info.pSwapchains = &r->swapchain;
    info.pImageIndices = &r->image_index;
    res = r->api.vkQueuePresentKHR(r->present, &info);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) {
        return rebuild_swapchain(r);
    }
    if (res != VK_SUCCESS) {
        re_log_warn("vulkan: vkQueuePresentKHR failed (VkResult %d)", (int)res);
        r->device_lost = true;
        return false;
    }
    return true;
}

/* --------------------------------------------------------------------- */
/* Public renderer lifecycle                                             */
/* --------------------------------------------------------------------- */

/** Destroys the per-frame fences, semaphores, command pool and buffers. */
static void destroy_sync(struct ReRenderer *r) {
    for (uint32_t i = 0u; i < RE_FRAMES_IN_FLIGHT; i++) {
        if (r->fence != NULL && r->fence[i] != VK_NULL_HANDLE &&
            r->api.vkDestroyFence != NULL) {
            r->api.vkDestroyFence(r->device, r->fence[i], NULL);
        }
        if (r->acquire != NULL && r->acquire[i] != VK_NULL_HANDLE &&
            r->api.vkDestroySemaphore != NULL) {
            r->api.vkDestroySemaphore(r->device, r->acquire[i], NULL);
        }
        if (r->release != NULL && r->release[i] != VK_NULL_HANDLE &&
            r->api.vkDestroySemaphore != NULL) {
            r->api.vkDestroySemaphore(r->device, r->release[i], NULL);
        }
    }
    free(r->fence);
    free(r->acquire);
    free(r->release);
    r->fence = NULL;
    r->acquire = NULL;
    r->release = NULL;
    if (r->cpool != VK_NULL_HANDLE && r->api.vkDestroyCommandPool != NULL) {
        r->api.vkDestroyCommandPool(r->device, r->cpool, NULL);
    }
    free(r->cmds);
    r->cmds = NULL;
    r->cpool = VK_NULL_HANDLE;
}

/** Unmaps and frees the two device-memory blocks. */
static void destroy_memory_blocks(struct ReRenderer *r) {
    if (r->buf.mapped != NULL && r->api.vkUnmapMemory != NULL) {
        r->api.vkUnmapMemory(r->device, r->buf.buf_mem);
        r->buf.mapped = NULL;
    }
    if (r->gpu.img_mem != VK_NULL_HANDLE && r->api.vkFreeMemory != NULL) {
        r->api.vkFreeMemory(r->device, r->gpu.img_mem, NULL);
        r->gpu.img_mem = VK_NULL_HANDLE;
    }
    if (r->buf.buf_mem != VK_NULL_HANDLE && r->api.vkFreeMemory != NULL) {
        r->api.vkFreeMemory(r->device, r->buf.buf_mem, NULL);
        r->buf.buf_mem = VK_NULL_HANDLE;
    }
    free(r->buf_records);
    r->buf_records = NULL;
    r->buf_record_count = 0u;
    r->buf_record_capacity = 0u;
}

/** Destroys the debug messenger and the instance. */
static void destroy_instance_chain(struct ReRenderer *r) {
    if (r->messenger != VK_NULL_HANDLE &&
        r->api.vkDestroyDebugUtilsMessengerEXT != NULL) {
        r->api.vkDestroyDebugUtilsMessengerEXT(r->instance, r->messenger, NULL);
    }
    r->messenger = VK_NULL_HANDLE;
    if (r->instance != VK_NULL_HANDLE && r->api.vkDestroyInstance != NULL) {
        r->api.vkDestroyInstance(r->instance, NULL);
    }
    r->instance = VK_NULL_HANDLE;
}

#if defined(RE_HAS_X11)
/** Destroys the window and closes the display. */
static void destroy_window(struct ReRenderer *r) {
    if (r->display == NULL) {
        return;
    }
    if (r->window != 0u) {
        XDestroyWindow(r->display, r->window);
    }
    XCloseDisplay(r->display);
    r->display = NULL;
    r->window = 0u;
}
#endif /* RE_HAS_X11 */

/** Frees every object reachable from a partially built or live renderer. */
static void teardown(struct ReRenderer *r) {
    if (r->device != VK_NULL_HANDLE) {
        if (r->api.vkDeviceWaitIdle != NULL) {
            (void)r->api.vkDeviceWaitIdle(r->device);
        }
        if (r->frame != NULL) {
            re_frame_destroy(r);
            r->frame = NULL;
        }
        destroy_sync(r);
        destroy_swapchain(r);
        destroy_render_targets(r);
        destroy_memory_blocks(r);
        if (r->api.vkDestroyDevice != NULL) {
            r->api.vkDestroyDevice(r->device, NULL);
        }
        r->device = VK_NULL_HANDLE;
    }
    if (r->surface != VK_NULL_HANDLE && r->api.vkDestroySurfaceKHR != NULL) {
        r->api.vkDestroySurfaceKHR(r->instance, r->surface, NULL);
    }
    r->surface = VK_NULL_HANDLE;
#if defined(RE_HAS_X11)
    destroy_window(r);
#endif
    destroy_instance_chain(r);
}

/**
 * @brief Opens the loader, creates the instance and resolves its entries.
 *
 * @param r Renderer with windowed, title and debug configuration set.
 * @return True when the instance and every required entry are ready.
 */
static bool init_instance(struct ReRenderer *r) {
    bool debug;

    if (!re_vk_load_api(&r->api)) {
        return false;
    }
    debug = has_instance_extension(&r->api, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (!create_instance(r, r->windowed, debug)) {
        return false;
    }
    if (!load_instance_required(r)) {
        return false;
    }
    load_instance_optional(r);
    create_messenger(r);
    if (r->windowed && !load_instance_surface(r)) {
        re_log_warn("renderer: surface entries missing, running headless");
        r->windowed = false;
    }
    return true;
}

/**
 * @brief Creates the window and the presentation surface.
 *
 * Every failure degrades to headless instead of aborting creation.
 *
 * @param r Renderer with a live instance.
 */
static void init_presentation(struct ReRenderer *r) {
    bool have_window = false;

#if defined(RE_HAS_X11)
    if (r->windowed && !create_window(r)) {
        r->windowed = false;
    }
    have_window = r->display != NULL;
#else
    if (r->windowed) {
        re_log_warn("renderer: no X11 support compiled in, running headless");
        r->windowed = false;
    }
#endif
    if (r->windowed && have_window && !create_surface(r)) {
        re_log_warn("renderer: surface creation failed, running headless");
        r->windowed = false;
    }
}

/**
 * @brief Creates the device, the sync objects, the memory blocks and the
 *        frame layer.
 *
 * @param r Renderer with a live instance, surface and physical device.
 * @return True when the frame layer is live.
 */
static bool init_gpu(struct ReRenderer *r) {
    if (!pick_physical_device(r)) {
        return false;
    }
    if (!create_device(r)) {
        return false;
    }
    if (!device_entries_ok(&r->api)) {
        re_log_warn("renderer: essential device entry points missing");
        return false;
    }
    if (!create_sync(r)) {
        return false;
    }
    if (!allocate_blocks(r)) {
        return false;
    }
    if (!create_render_targets(r)) {
        return false;
    }
    if (!re_frame_create(r)) {
        re_log_warn("renderer: frame layer creation failed");
        return false;
    }
    return true;
}

ReRenderer *re_renderer_create(const ReRenderConfig *cfg) {
    ReRenderConfig defaults;
    ReRenderer *r;

    if (cfg == NULL) {
        defaults.windowed = false;
        defaults.width = 1280u;
        defaults.height = 720u;
        defaults.title = "Real Engine";
        cfg = &defaults;
    }
    r = calloc(1u, sizeof *r);
    if (r == NULL) {
        return NULL;
    }
    r->width = cfg->width > 0u ? cfg->width : 1280u;
    r->height = cfg->height > 0u ? cfg->height : 720u;
    r->window_x = cfg->window_x;
    r->window_y = cfg->window_y;
    r->title = cfg->title != NULL ? cfg->title : "Real Engine";
    r->frames_in_flight = RE_FRAMES_IN_FLIGHT;
    r->windowed = cfg->windowed;
    if (!init_instance(r)) {
        goto fail;
    }
    init_presentation(r);
    if (!init_gpu(r)) {
        goto fail;
    }
    if (r->surface != VK_NULL_HANDLE && !build_swapchain(r)) {
        goto fail;
    }
    re_log_info("renderer: ready (%s, %ux%u)", r->windowed ? "windowed"
                                                           : "headless",
                r->width, r->height);
    return r;
fail:
    teardown(r);
    close_loader();
    free(r);
    re_log_warn("renderer: Vulkan setup failed, using the null renderer");
    return re_null_renderer_create(cfg);
}

bool re_renderer_render(ReRenderer *r, const ReSceneSnapshot *scene) {
    if (r == NULL) {
        return false;
    }
    if (r->is_null) {
        return re_null_render(r, scene); /* paced null frame + report */
    }
    if (r->device_lost || r->frame == NULL || r->cmds == NULL) {
        return false;
    }
    if (!wait_fence(r)) {
        return false;
    }
    if (r->surface != VK_NULL_HANDLE) {
        if (!acquire_image(r)) {
            return false;
        }
    } else {
        r->image_index = 0u;
    }
    if (r->api.vkResetCommandBuffer != NULL &&
        r->api.vkResetCommandBuffer(r->cmds[r->frame_index], 0u) != VK_SUCCESS) {
        re_log_warn("renderer: command buffer reset failed");
        return false;
    }
    if (!re_frame_render(r, scene)) {
        re_log_warn("renderer: frame recording failed");
        return false;
    }
    if (!submit_frame(r)) {
        return false;
    }
    if (r->surface != VK_NULL_HANDLE && !present_frame(r)) {
        return false;
    }
    r->frame_index = (r->frame_index + 1u) % r->frames_in_flight;
    return true;
}

bool re_renderer_is_null(const ReRenderer *r) {
    return r == NULL || r->is_null;
}

void re_renderer_extent(const ReRenderer *r, uint32_t *width, uint32_t *height) {
    uint32_t w = 0u;
    uint32_t h = 0u;

    if (width == NULL || height == NULL) {
        return;
    }
#if defined(RE_HAS_VULKAN)
    if (r != NULL && !r->is_null && r->sw_extent.width > 0u &&
        r->sw_extent.height > 0u) {
        w = r->sw_extent.width;
        h = r->sw_extent.height;
    }
#else
    (void)r;
#endif
    *width = w;
    *height = h;
}

void re_renderer_poll_input(ReRenderer *r, ReInput *out) {
    if (out == NULL) {
        return;
    }
    memset(out, 0, sizeof *out);
    if (r == NULL) {
        return;
    }
    if (r->is_null) {
        re_null_poll_input(r, out);
        return;
    }
#if defined(RE_HAS_X11)
    if (r->windowed && r->display != NULL) {
        pump_window_events(r);
        memcpy(out->down, r->keys, sizeof out->down);
    }
#else
    (void)r;
#endif
}

bool re_renderer_should_close(const ReRenderer *r) {
    if (r == NULL) {
        return false;
    }
    if (r->is_null) {
        return re_null_should_close(r);
    }
    if (!r->windowed) {
        return false;
    }
#if defined(RE_HAS_X11)
    if (r->display == NULL) {
        return false;
    }
    /* The event pump only mutates key/quit state, never the opaque
     * renderer identity, so the const signature stays honest. */
    pump_window_events((struct ReRenderer *)(void *)r);
    return r->quit_requested;
#else
    return false;
#endif
}

void re_renderer_destroy(ReRenderer *r) {
    if (r == NULL) {
        return;
    }
    if (r->is_null) {
        re_null_destroy(r);
        return;
    }
    teardown(r);
    close_loader();
    free(r);
}

#else /* !RE_HAS_VULKAN */

/*
 * Without Vulkan there is no device to query: every call degrades to the
 * null renderer, which the engine drives exactly like a real one.
 */

ReRenderer *re_renderer_create(const ReRenderConfig *cfg) {
    ReRenderConfig defaults;

    if (cfg == NULL) {
        defaults.windowed = false;
        defaults.width = 1280u;
        defaults.height = 720u;
        defaults.title = "Real Engine";
        cfg = &defaults;
    }
    re_log_info("renderer: built without Vulkan, using the null renderer");
    return re_null_renderer_create(cfg);
}

bool re_renderer_render(ReRenderer *r, const ReSceneSnapshot *scene) {
    return re_null_render(r, scene);
}

bool re_renderer_should_close(const ReRenderer *r) {
    return re_null_should_close(r);
}

bool re_renderer_is_null(const ReRenderer *r) {
    return r == NULL || r->is_null;
}

void re_renderer_extent(const ReRenderer *r, uint32_t *width, uint32_t *height) {
    uint32_t w = 0u;
    uint32_t h = 0u;

    if (width == NULL || height == NULL) {
        return;
    }
#if defined(RE_HAS_VULKAN)
    if (r != NULL && !r->is_null && r->sw_extent.width > 0u &&
        r->sw_extent.height > 0u) {
        w = r->sw_extent.width;
        h = r->sw_extent.height;
    }
#else
    (void)r;
#endif
    *width = w;
    *height = h;
}

void re_renderer_poll_input(ReRenderer *r, ReInput *out) {
    re_null_poll_input(r, out);
}

void re_renderer_destroy(ReRenderer *r) {
    re_null_destroy(r);
}

#endif /* RE_HAS_VULKAN */
