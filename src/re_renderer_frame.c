/**
 * @file re_renderer_frame.c
 * @brief Per-frame Vulkan work: descriptor set, pipelines, meshes, the
 *        camera/light/cluster uploads and the command buffer that
 *        records GPU culling, cascaded shadow maps, the clustered
 *        forward pass and the tonemap pass.
 *
 * The file is compiled only when RE_HAS_VULKAN is defined; without the
 * macro it is an empty translation unit, because nothing outside the
 * Vulkan backend ever calls re_frame_create/re_frame_render/
 * re_frame_destroy.
 *
 * Ownership: re_frame_create owns the descriptor set layout, the
 * descriptor pool, set 0 (ReRenderer::dset), the pipeline layout
 * (ReRenderer::pipe_layout), the four pipelines and the shader modules
 * (ReRenderer::pipes), the two samplers
 * (ReRenderer::gpu.shadow_sampler / tex_sampler), the meshes, the
 * checker texture and every host-side frame buffer. re_frame_destroy
 * releases all of it; re_renderer.c must not create those objects.
 *
 * Synchronization uses vkCmdPipelineBarrier2 exclusively. Layouts are
 * re-derived from VK_IMAGE_LAYOUT_UNDEFINED at the start of every frame
 * (all attachments are cleared anyway), which keeps the layout tracker
 * trivial at the price of one discard per image per frame.
 *
 * Depth convention: every projection in re_vec.h is reverse-Z (near
 * maps to 1.0, far to 0.0) and the attachments are cleared to 0.0, so
 * depth tests use VK_COMPARE_OP_GREATER_OR_EQUAL. LESS_OR_EQUAL would
 * reject every fragment; the depth bias is applied in shadow.vert
 * because ReVkApi has no vkCmdSetDepthBias entry point.
 *
 * Bindless ABI (binding 4): slot 0 is a generated 64x64 checker texture,
 * slot 1 is the HDR colour image that tonemap.frag samples. Slots 2..1023
 * are unbound and the layout marks the array partially bound.
 *
 * Binding 8 ABI: unlike a pure index list, the buffer starts with the
 * ReRenderPointLightGpu records (what re_renderer.h documents) and
 * stores the packed cluster light indices after them, starting at byte
 * RE_LIGHT_INDEX_OFFSET:
 *
 *   byte 0 .. 4095                      ReRenderPointLightGpu[128]
 *   byte 4096 .. 4096 + 4 * capacity    uint32 cluster light indices
 *
 * The cluster grid's offset field is relative to the index region. This
 * is the only layout that satisfies both the header's binding table and
 * the packed-list contract; see the report for the conflict.
 */
#if defined(RE_HAS_VULKAN)

#include "re_core.h"
#include "re_renderer.h"
#include "re_vec.h"
#include "re_vk.h"
#include "re_vk_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** Cube vertices (six faces of four corners). */
#define RE_CUBE_VERTEX_COUNT 24u

/** Cube indices (six faces of two triangles). */
#define RE_CUBE_INDEX_COUNT 36u

/** Sphere latitude bands. */
#define RE_SPHERE_RINGS 16u

/** Sphere longitude bands. */
#define RE_SPHERE_SECTORS 32u

/** Sphere vertices ((rings + 1) * (sectors + 1), seam duplicated). */
#define RE_SPHERE_VERTEX_COUNT ((RE_SPHERE_RINGS + 1u) * (RE_SPHERE_SECTORS + 1u))

/** Sphere indices (two triangles per quad). */
#define RE_SPHERE_INDEX_COUNT (RE_SPHERE_RINGS * RE_SPHERE_SECTORS * 6u)

/**
 * Packed light-index capacity of the cluster light list.
 *
 * Truncation policy: a light whose sphere covers the whole view alone
 * needs RE_CLUSTER_COUNT = 2048 slots, so 64 slots per light on average
 * is a deliberate budget, not a correctness bound. The build drops the
 * lights that do not fit (lowest light index wins) and the affected
 * cells report only the lights that were really stored, so the shader
 * never reads outside the buffer.
 */
#define RE_LIGHT_LIST_CAPACITY (RE_MAX_POINT_LIGHTS * 64u)

/** Byte offset of the packed light-index region inside binding 8. */
#define RE_LIGHT_INDEX_OFFSET (RE_MAX_POINT_LIGHTS * sizeof(ReRenderPointLightGpu))

/** Total host image uploaded into binding 8 every frame. */
#define RE_LIGHT_BLOCK_BYTES (RE_LIGHT_INDEX_OFFSET + (RE_LIGHT_LIST_CAPACITY * sizeof(uint32_t)))

/** Checker texture edge length (texels). */
#define RE_CHECKER_DIM 64u

/** Workgroup size of cull.comp (must match its layout(local_size_x = 64)). */
#define RE_CULL_GROUP_SIZE 64u

/** Staging buffer size in bytes (one upload at a time, reused). */
#define RE_FRAME_STAGING_BYTES ((VkDeviceSize)4u * 1024u * 1024u)

/** Push-constant range size; covers cull (4 bytes) and shadow (64 bytes). */
#define RE_PUSH_RANGE_SIZE 64u

/** Sanity bound for the acquired swapchain image index. */
#define RE_SW_MAX_IMAGES 8u

/* ------------------------------------------------------------------ */
/* GPU-side layouts.                                                  */
/* ------------------------------------------------------------------ */

/**
 * @brief Camera uniform block (std140, 320 bytes).
 *
 * @code
 * layout(std140, set = 0, binding = 0) uniform Camera {
 *     mat4 view; mat4 proj; mat4 view_proj;
 *     vec4 frustum_planes[6]; vec4 cam_pos_far;
 *     uint instance_count; uint point_light_count; vec2 _pad;
 * } camera;
 * @endcode
 */
typedef struct ReRenderCameraUbo {
    ReMat4 view;                /**< World -> view. */
    ReMat4 proj;                /**< View -> clip (reverse Z). */
    ReMat4 view_proj;           /**< proj * view. */
    ReVec4 frustum_planes[6];   /**< left, right, bottom, top, near, far. */
    ReVec4 cam_pos_far;         /**< xyz world eye, w far plane. */
    uint32_t instance_count;    /**< Instances submitted this frame. */
    uint32_t point_light_count; /**< Point lights in the cluster build. */
    float _pad[2];              /**< std140 tail padding. */
} ReRenderCameraUbo;

static_assert(sizeof(ReRenderCameraUbo) == 320, "camera UBO layout must stay std140");

/**
 * @brief Sun and cascade uniform block (std140, 176 bytes).
 *
 * @code
 * layout(std140, set = 0, binding = 1) uniform Lights {
 *     vec4 sun_dir_intensity; vec4 sun_color_ambient;
 *     vec4 cascade_splits;    mat4 light_vp[RE_SHADOW_CASCADES];
 * } lights;
 * @endcode
 */
typedef struct ReRenderLightUbo {
    ReVec4 sun_dir_intensity;            /**< xyz normalized, w intensity. */
    ReVec4 sun_color_ambient;            /**< xyz colour, w ambient. */
    ReVec4 cascade_splits;               /**< x = cascade 0 end, y = far. */
    ReMat4 light_vp[RE_SHADOW_CASCADES]; /**< Per-cascade light view-projection. */
} ReRenderLightUbo;

static_assert(sizeof(ReRenderLightUbo) == 176, "light UBO layout must stay std140");

/** @brief Point light record (std430, 32 bytes). */
typedef struct ReRenderPointLightGpu {
    ReVec4 pos_radius;      /**< xyz world centre, w influence radius. */
    ReVec4 color_intensity; /**< xyz radiance, w intensity. */
} ReRenderPointLightGpu;

static_assert(sizeof(ReRenderPointLightGpu) == 32, "point light record must stay 32 bytes");

/** @brief One cluster cell (std430, 8 bytes). */
typedef struct ReRenderClusterGpu {
    uint32_t offset; /**< First index into the light-list region. */
    uint32_t count;  /**< Number of lights stored in the cell. */
} ReRenderClusterGpu;

static_assert(sizeof(ReRenderClusterGpu) == 8, "cluster cell must stay 8 bytes");

/**
 * @brief Indirect draw arguments plus the atomic draw counter (std430).
 *
 * cull.comp increments draw_count (byte 20) and pbr_clustered.vert
 * degenerates every instance index at or above it. The renderer does not
 * issue vkCmdDrawIndexedIndirect: with one shared argument record there
 * would be a single draw for both meshes, so the forward pass issues two
 * plain indexed draws over RE_MAX_INSTANCES instances and lets the
 * vertex shader discard the invisible ones. The counter is still kept
 * authoritative because the vertex shader reads it.
 */
typedef struct ReRenderDrawArgs {
    uint32_t index_count;    /**< Index count of the submitted draw. */
    uint32_t instance_count; /**< Always RE_MAX_INSTANCES. */
    int32_t first_index;     /**< First index. */
    int32_t vertex_offset;   /**< Vertex offset. */
    uint32_t first_instance; /**< First instance. */
    uint32_t draw_count;     /**< Atomic: visible instance count. */
} ReRenderDrawArgs;

static_assert(sizeof(ReRenderDrawArgs) == 24, "draw args must stay 24 bytes");
static_assert(offsetof(ReRenderDrawArgs, draw_count) == 20, "draw_count must live at byte 20");

/* ------------------------------------------------------------------ */
/* struct ReRenderer and the cross-file helpers live in re_vk_internal.h
   (single canonical definition, shared with re_renderer.c). */


/* ------------------------------------------------------------------ */
/* Per-frame host data.                                               */
/* ------------------------------------------------------------------ */

/** @brief One mesh vertex: position + normal (vertex locations 0 and 1). */
typedef struct ReFrameVertex {
    float position[3]; /**< Object-space position. */
    float normal[3];   /**< Object-space normal. */
} ReFrameVertex;

static_assert(sizeof(ReFrameVertex) == 24, "vertex stride must stay 24 bytes");

/** @brief A CPU mesh plus its device buffers. */
typedef struct ReFrameMesh {
    ReFrameVertex *vertices; /**< Host vertex copy. */
    uint32_t vertex_count;   /**< Vertices. */
    uint32_t *indices;       /**< Host index copy. */
    uint32_t index_count;    /**< Indices. */
    VkBuffer vertex_buffer;  /**< Device vertex buffer. */
    VkBuffer index_buffer;   /**< Device index buffer. */
} ReFrameMesh;

/** @brief Inclusive cluster range overlapped by one light sphere. */
typedef struct ReClusterRange {
    uint32_t x0; /**< First column. */
    uint32_t x1; /**< Last column. */
    uint32_t y0; /**< First row. */
    uint32_t y1; /**< Last row. */
    uint32_t z0; /**< First slice. */
    uint32_t z1; /**< Last slice. */
    bool empty;  /**< True when the sphere misses the cluster volume. */
} ReClusterRange;

/** @brief Camera data the cluster build needs. */
typedef struct ReClusterCtx {
    ReMat4 view;  /**< World -> view. */
    float tan_x;  /**< Half-extent factor along x (1 / proj[0][0]). */
    float tan_y;  /**< Half-extent factor along y (1 / proj[1][1]). */
    float near_d; /**< Near plane distance. */
    float far_d;  /**< Far plane distance. */
} ReClusterCtx;

/**
 * @brief Per-renderer frame state allocated by re_frame_create.
 *
 * Every array is sized once at creation: the engine contract fixes
 * RE_MAX_INSTANCES, RE_CLUSTER_COUNT and RE_MAX_POINT_LIGHTS at compile
 * time, so no frame does heap traffic.
 */
typedef struct ReRendererFrame {
    ReFrameMesh cube;                              /**< Unit cube, [-0.5, 0.5]. */
    ReFrameMesh sphere;                            /**< UV sphere, radius 0.5. */
    VkDescriptorSetLayout set_layout;              /**< Set 0 layout. */
    VkDescriptorPool pool;                         /**< Pool that owns set 0. */
    VkImageView shadow_layers[RE_SHADOW_CASCADES]; /**< One view per cascade layer. */
    VkImage checker;                               /**< Checker texture (bindless 0). */
    VkImageView checker_view;                      /**< Checker view. */
    ReRenderClusterGpu *clusters;                  /**< RE_CLUSTER_COUNT cells. */
    uint32_t *cluster_cursor;                      /**< Prefix-sum write cursors. */
    unsigned char *light_block;                    /**< Binding 8 host image. */
    ReRenderPointLightGpu *lights;                 /**< Records inside light_block. */
    uint32_t *light_list;                          /**< Indices inside light_block. */
    ReRenderCameraUbo camera;                      /**< Camera block of this frame. */
    ReRenderLightUbo light;                        /**< Light block of this frame. */
    VkDeviceSize staging_cursor;                   /**< Next free staging byte. */
    uint64_t frames_rendered;                      /**< Frames recorded so far. */
} ReRendererFrame;

/* ------------------------------------------------------------------ */
/* File and staging helpers.                                          */
/* ------------------------------------------------------------------ */

/**
 * @brief Reads a whole file into a malloc'd buffer.
 * @param path File to read.
 * @param size_out Receives the byte count.
 * @param data_out Receives the allocation (caller frees).
 * @return True on success.
 */
static bool frame_read_file(const char *path, uint32_t *size_out, void **data_out) {
    FILE *file = fopen(path, "rb");
    long length;
    void *data;

    if (file == NULL) {
        return false;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        (void)fclose(file);
        return false;
    }
    length = ftell(file);
    /* SPIR-V blobs are whole 32-bit words; reject empty and non-word sizes
       so a truncated .spv is never handed to the driver (which may round
       the size up and read past the buffer). */
    if (length <= 0 || length > 0x7FFFFFFFL || (length % 4L) != 0L ||
        fseek(file, 0, SEEK_SET) != 0) {
        (void)fclose(file);
        return false;
    }
    data = malloc((size_t)length);
    if (data == NULL) {
        (void)fclose(file);
        return false;
    }
    if (fread(data, 1u, (size_t)length, file) != (size_t)length) {
        free(data);
        (void)fclose(file);
        return false;
    }
    (void)fclose(file);
    *size_out = (uint32_t)length;
    *data_out = data;
    return true;
}

/**
 * @brief Loads a SPIR-V blob from bin/shaders or shaders (relative to CWD).
 * @param name File name, for example "cull.comp.spv".
 * @param size_out Receives the byte count.
 * @param data_out Receives the allocation (caller frees).
 * @return True when one of the candidate paths opened.
 */
static bool load_spv(const char *name, uint32_t *size_out, const void **data_out) {
    static const char *const PREFIXES[2] = {"bin/shaders/", "shaders/"};
    char path[256];

    for (size_t i = 0; i < 2u; i++) {
        int written = snprintf(path, sizeof path, "%s%s", PREFIXES[i], name);
        if (written > 0 && (size_t)written < sizeof path) {
            void *data = NULL;
            uint32_t size = 0;
            if (frame_read_file(path, &size, &data)) {
                *size_out = size;
                *data_out = data;
                return true;
            }
        }
    }
    return false;
}

/** @brief Creates one shader module from a SPIR-V file. */
static bool frame_create_module(ReRenderer *r, const char *name, VkShaderModule *out) {
    uint32_t size = 0;
    const void *data = NULL;
    VkShaderModuleCreateInfo info;
    VkResult res;

    if (!load_spv(name, &size, &data)) {
        re_log_error("renderer: cannot open shader %s (looked in bin/shaders and shaders)", name);
        return false;
    }
    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = (size_t)size;
    info.pCode = (const uint32_t *)data;
    res = r->api.vkCreateShaderModule(r->device, &info, NULL, out);
    free((void *)data);
    if (res != VK_SUCCESS) {
        re_log_error("renderer: vkCreateShaderModule(%s) failed with %d", name, (int)res);
        *out = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

/** @brief Copies host bytes into a given staging offset, logging failures. */
static bool frame_stage_at(ReRenderer *r, const void *data, VkDeviceSize size,
                           VkDeviceSize offset) {
    VkResult res = re_rr_upload(r, data, size, offset);

    if (res != VK_SUCCESS) {
        re_log_error("renderer: staging upload of %llu bytes failed (%d)",
                     (unsigned long long)size, (int)res);
        return false;
    }
    return true;
}

/** @brief Copies host bytes into the start of the staging buffer. */
static bool frame_stage(ReRenderer *r, const void *data, VkDeviceSize size) {
    return frame_stage_at(r, data, size, 0u);
}

/**
 * @brief Stages, then copies bytes into a device buffer.
 *
 * Every per-frame upload gets its own staging span. All copies are
 * recorded into one command buffer and execute after frame_prepare
 * returns, so reusing offset 0 for each upload would make every buffer
 * receive the last uploaded bytes (usually the 24-byte draw arguments),
 * leaving the camera and instance data zeroed.
 */
static bool frame_upload_buffer(ReRenderer *r, VkBuffer dst, const void *data, VkDeviceSize size) {
    ReRendererFrame *f = r->frame;
    VkDeviceSize offset = re_align_up(f->staging_cursor, 16u);
    VkDeviceSize end = offset + re_align_up(size, 16u);

    if (end > r->buf.staging_size) {
        re_log_error("renderer: per-frame staging overflow (%llu > %llu)",
                     (unsigned long long)end,
                     (unsigned long long)r->buf.staging_size);
        return false;
    }
    if (!frame_stage_at(r, data, size, offset)) {
        return false;
    }
    re_rr_copy_to_buffer(r, dst, size, offset);
    f->staging_cursor = end;
    return true;
}

/* ------------------------------------------------------------------ */
/* Command buffer helpers.                                            */
/* ------------------------------------------------------------------ */

/** @brief Records one image layout transition through synchronization2. */
static void frame_transition(VkCommandBuffer cb, const ReVkApi *api, VkImage image,
                             VkImageAspectFlags aspect, uint32_t base_layer, uint32_t layer_count,
                             VkImageLayout from, VkImageLayout to,
                             VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                             VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkImageMemoryBarrier2 barrier;
    VkDependencyInfo dep;

    memset(&barrier, 0, sizeof barrier);
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = src_stage;
    barrier.srcAccessMask = src_access;
    barrier.dstStageMask = dst_stage;
    barrier.dstAccessMask = dst_access;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspect;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = base_layer;
    barrier.subresourceRange.layerCount = layer_count;

    memset(&dep, 0, sizeof dep);
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &barrier;
    api->vkCmdPipelineBarrier2(cb, &dep);
}

/** @brief Discards an image and makes it a colour attachment. */
static void frame_to_color_attachment(VkCommandBuffer cb, const ReVkApi *api, VkImage image) {
    frame_transition(cb, api, image, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
}

/** @brief Discards a depth image (or array) and makes it a depth attachment. */
static void frame_to_depth_attachment(VkCommandBuffer cb, const ReVkApi *api, VkImage image,
                                      uint32_t layers) {
    frame_transition(cb, api, image, VK_IMAGE_ASPECT_DEPTH_BIT, 0, layers,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                         VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
}

/** @brief Makes a written colour attachment readable by a later pass. */
static void frame_color_to_read(VkCommandBuffer cb, const ReVkApi *api, VkImage image,
                                VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    frame_transition(cb, api, image, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, dst_stage, dst_access);
}

/** @brief Makes the shadow array readable by the forward fragment shader. */
static void frame_shadow_to_read(VkCommandBuffer cb, const ReVkApi *api, VkImage image) {
    frame_transition(cb, api, image, VK_IMAGE_ASPECT_DEPTH_BIT, 0, RE_SHADOW_CASCADES,
                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
}

/** @brief Ends the frame with the swapchain image ready for presentation. */
static void frame_color_to_present(VkCommandBuffer cb, const ReVkApi *api, VkImage image) {
    frame_transition(cb, api, image, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
}

/** @brief Records one buffer dependency covering the whole buffer. */
static void frame_buffer_dependency(VkCommandBuffer cb, const ReVkApi *api, VkBuffer buffer,
                                   VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                                   VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkBufferMemoryBarrier2 barrier;
    VkDependencyInfo dep;

    memset(&barrier, 0, sizeof barrier);
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask = src_stage;
    barrier.srcAccessMask = src_access;
    barrier.dstStageMask = dst_stage;
    barrier.dstAccessMask = dst_access;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;

    memset(&dep, 0, sizeof dep);
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers = &barrier;
    api->vkCmdPipelineBarrier2(cb, &dep);
}

/** @brief Sets a full-target viewport and scissor rectangle. */
static void frame_set_viewport(VkCommandBuffer cb, const ReVkApi *api, uint32_t width,
                               uint32_t height) {
    VkViewport viewport;
    VkRect2D scissor;

    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)width;
    viewport.height = (float)height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = width;
    scissor.extent.height = height;
    api->vkCmdSetViewport(cb, 0, 1, &viewport);
    api->vkCmdSetScissor(cb, 0, 1, &scissor);
}

/**
 * @brief Sets a Y-flipped full-target viewport.
 *
 * Vulkan's framebuffer origin is the top-left while the engine's
 * projections follow the OpenGL convention (Y up in NDC). A negative
 * viewport height flips the image so the world is not rendered upside
 * down. Only the camera-space forward pass needs this; the depth-only
 * cascade pass and the fullscreen tonemap keep the default orientation.
 */
static void frame_set_viewport_flipped(VkCommandBuffer cb, const ReVkApi *api, uint32_t width,
                                       uint32_t height) {
    VkViewport viewport;
    VkRect2D scissor;

    viewport.x = 0.0f;
    viewport.y = (float)height;
    viewport.width = (float)width;
    viewport.height = -(float)height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = width;
    scissor.extent.height = height;
    api->vkCmdSetViewport(cb, 0, 1, &viewport);
    api->vkCmdSetScissor(cb, 0, 1, &scissor);
}

/* ------------------------------------------------------------------ */
/* Mesh generation.                                                   */
/* ------------------------------------------------------------------ */

/** @brief Emits one outward-facing quad with a shared normal. */
static void frame_emit_quad(ReFrameVertex *verts, uint32_t *indices, uint32_t *vcount,
                            uint32_t *icount, const float corner[4][3], const float normal[3]) {
    static const uint32_t OFFSETS[6] = {0u, 1u, 2u, 0u, 2u, 3u};
    uint32_t base = *vcount;

    for (uint32_t i = 0; i < 4u; i++) {
        ReFrameVertex *vert = &verts[base + i];
        for (uint32_t a = 0; a < 3u; a++) {
            vert->position[a] = corner[i][a];
            vert->normal[a] = normal[a];
        }
    }
    for (uint32_t i = 0; i < 6u; i++) {
        indices[*icount + i] = base + OFFSETS[i];
    }
    *vcount += 4u;
    *icount += 6u;
}

/**
 * @brief Builds a unit cube centred on the origin.
 *
 * Every face carries an orthonormal (u, v) pair with cross(u, v) = n, so
 * the corners (c-u-v, c+u-v, c+u+v, c-u+v) are already counter-clockwise
 * when seen from outside, which is what the forward pipeline expects
 * (front face counter-clockwise, back faces culled).
 */
static void frame_build_cube(ReFrameVertex *verts, uint32_t *indices) {
    static const float FACES[6][3][3] = {
        {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
        {{0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
        {{0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
        {{0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
    };
    uint32_t vcount = 0u;
    uint32_t icount = 0u;

    for (uint32_t f = 0; f < 6u; f++) {
        const float *normal = FACES[f][0];
        const float *axis_u = FACES[f][1];
        const float *axis_v = FACES[f][2];
        float corner[4][3];

        for (uint32_t i = 0; i < 4u; i++) {
            const float su = (i == 0u || i == 3u) ? -0.5f : 0.5f;
            const float sv = (i < 2u) ? -0.5f : 0.5f;
            for (uint32_t a = 0; a < 3u; a++) {
                /* 0.525, not 0.5: the forward pass also draws the
                 * inscribed unit sphere for every instance, and matching
                 * extents make the two surfaces z-fight. */
                corner[i][a] = normal[a] * 0.525f + axis_u[a] * su +
                               axis_v[a] * sv;
            }
        }
        frame_emit_quad(verts, indices, &vcount, &icount, corner, normal);
    }
}

/** @brief Builds a UV sphere of radius 0.5 centred on the origin. */
static void frame_build_sphere(ReFrameVertex *verts, uint32_t *indices) {
    const float pi = 3.14159265358979323846f;

    for (uint32_t ring = 0; ring <= RE_SPHERE_RINGS; ring++) {
        float v = (float)ring / (float)RE_SPHERE_RINGS;
        float theta = v * pi;
        float sin_theta = sinf(theta);
        float cos_theta = cosf(theta);

        for (uint32_t sector = 0; sector <= RE_SPHERE_SECTORS; sector++) {
            float u = (float)sector / (float)RE_SPHERE_SECTORS;
            float phi = u * 2.0f * pi;
            float nx = sin_theta * cosf(phi);
            float ny = cos_theta;
            float nz = sin_theta * sinf(phi);
            ReFrameVertex *vert = &verts[ring * (RE_SPHERE_SECTORS + 1u) + sector];

            vert->position[0] = nx * 0.5f;
            vert->position[1] = ny * 0.5f;
            vert->position[2] = nz * 0.5f;
            vert->normal[0] = nx;
            vert->normal[1] = ny;
            vert->normal[2] = nz;
        }
    }
    for (uint32_t ring = 0; ring < RE_SPHERE_RINGS; ring++) {
        for (uint32_t sector = 0; sector < RE_SPHERE_SECTORS; sector++) {
            uint32_t a = ring * (RE_SPHERE_SECTORS + 1u) + sector;
            uint32_t b = a + (RE_SPHERE_SECTORS + 1u);
            uint32_t out = (ring * RE_SPHERE_SECTORS + sector) * 6u;

            /* cross(a+1 - a, b - a) = cross(dphi, dtheta) points outwards. */
            indices[out + 0u] = a;
            indices[out + 1u] = a + 1u;
            indices[out + 2u] = b;
            indices[out + 3u] = a + 1u;
            indices[out + 4u] = b + 1u;
            indices[out + 5u] = b;
        }
    }
}

/**
 * @brief Uploads host bytes through staging on a self-contained
 *        one-shot command buffer.
 *
 * Creation-time uploads cannot use re_rr_copy_to_buffer, because that
 * helper records into the live per-frame command buffer, which is not
 * recording yet during re_frame_create. Recording a copy outside a
 * begun command buffer corrupts driver state.
 */
static bool frame_upload_buffer_sync(ReRenderer *r, VkBuffer dst,
                                     const void *data, VkDeviceSize size);

/** @brief Creates a mesh's device buffers and uploads the host data. */
static bool frame_mesh_upload(ReRenderer *r, ReFrameMesh *mesh) {
    VkDeviceSize vertex_bytes = (VkDeviceSize)mesh->vertex_count * sizeof(ReFrameVertex);
    VkDeviceSize index_bytes = (VkDeviceSize)mesh->index_count * sizeof(uint32_t);
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    if (re_rr_create_buffer(r, vertex_bytes, usage | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            &mesh->vertex_buffer) != VK_SUCCESS) {
        re_log_error("renderer: mesh vertex buffer allocation failed");
        return false;
    }
    if (re_rr_create_buffer(r, index_bytes, usage | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                            &mesh->index_buffer) != VK_SUCCESS) {
        re_log_error("renderer: mesh index buffer allocation failed");
        return false;
    }
    if (!frame_upload_buffer_sync(r, mesh->vertex_buffer, mesh->vertices,
                                  vertex_bytes)) {
        return false;
    }
    return frame_upload_buffer_sync(r, mesh->index_buffer, mesh->indices,
                                    index_bytes);
}

/** @brief Allocates a mesh's host arrays, fills them and uploads. */
static bool frame_mesh_build(ReRenderer *r, ReFrameMesh *mesh, uint32_t vertex_count,
                             uint32_t index_count,
                             void (*build)(ReFrameVertex *, uint32_t *)) {
    mesh->vertices = calloc(vertex_count, sizeof *mesh->vertices);
    mesh->indices = calloc(index_count, sizeof *mesh->indices);
    if (mesh->vertices == NULL || mesh->indices == NULL) {
        re_log_error("renderer: out of host memory for the primitive meshes");
        return false;
    }
    mesh->vertex_count = vertex_count;
    mesh->index_count = index_count;
    build(mesh->vertices, mesh->indices);
    if (mesh->indices[0] >= vertex_count) {
        re_log_error("renderer: generated mesh index out of range");
        return false;
    }
    return frame_mesh_upload(r, mesh);
}

/* ------------------------------------------------------------------ */
/* Descriptor set and pipeline layout.                                */
/* ------------------------------------------------------------------ */

/**
 * @brief Creates the set 0 layout with the nine contract bindings.
 *
 * The bindless array is declared with the fixed RE_BINDLESS_COUNT
 * descriptors plus UPDATE_AFTER_BIND and PARTIALLY_BOUND, and only the
 * first few slots are actually written. VARIABLE_DESCRIPTOR_COUNT is not
 * usable here: VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-
 * pBindingFlags-03004 requires that flag on the highest binding number,
 * but re_renderer.h pins the bindless array at binding 4 and puts four
 * storage buffers after it.
 */
static bool frame_create_set_layout(ReRenderer *r) {
    VkDescriptorSetLayoutBinding bindings[9];
    VkDescriptorBindingFlags flags[9];
    VkDescriptorSetLayoutBindingFlagsCreateInfo flag_info;
    VkDescriptorSetLayoutCreateInfo info;
    VkResult res;

    for (uint32_t i = 0; i < 9u; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_ALL;
        bindings[i].pImmutableSamplers = NULL;
    }
    bindings[RE_BIND_CAMERA_UBO].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[RE_BIND_LIGHT_UBO].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[RE_BIND_SHADOW_SAMPLERS].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[RE_BIND_BINDLESS_TEXTURES].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[RE_BIND_BINDLESS_TEXTURES].descriptorCount = RE_BINDLESS_COUNT;
    memset(flags, 0, sizeof flags);
    flags[RE_BIND_BINDLESS_TEXTURES] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                       VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

    memset(&flag_info, 0, sizeof flag_info);
    flag_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flag_info.bindingCount = 9;
    flag_info.pBindingFlags = flags;
    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.pNext = &flag_info;
    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    info.bindingCount = 9;
    info.pBindings = bindings;
    res = r->api.vkCreateDescriptorSetLayout(r->device, &info, NULL, &r->frame->set_layout);
    if (res != VK_SUCCESS) {
        re_log_error("renderer: vkCreateDescriptorSetLayout failed (%d)", (int)res);
        return false;
    }
    return true;
}

/** @brief Creates the update-after-bind descriptor pool. */
static bool frame_create_pool(ReRenderer *r) {
    VkDescriptorPoolSize sizes[3];
    VkDescriptorPoolCreateInfo info;
    VkResult res;

    sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[0].descriptorCount = 2;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[1].descriptorCount = 8;
    sizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[2].descriptorCount = RE_BINDLESS_COUNT + 1u;
    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    info.maxSets = 1;
    info.poolSizeCount = 3;
    info.pPoolSizes = sizes;
    res = r->api.vkCreateDescriptorPool(r->device, &info, NULL, &r->frame->pool);
    if (res != VK_SUCCESS) {
        re_log_error("renderer: vkCreateDescriptorPool failed (%d)", (int)res);
        return false;
    }
    return true;
}

/** @brief Allocates set 0 with the fixed bindless descriptor count. */
static bool frame_allocate_set(ReRenderer *r) {
    VkDescriptorSetAllocateInfo info;
    VkResult res;

    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.descriptorPool = r->frame->pool;
    info.descriptorSetCount = 1;
    info.pSetLayouts = &r->frame->set_layout;
    res = r->api.vkAllocateDescriptorSets(r->device, &info, &r->dset);
    if (res != VK_SUCCESS) {
        re_log_error("renderer: vkAllocateDescriptorSets failed (%d)", (int)res);
        return false;
    }
    return true;
}

/** @brief Creates the shared pipeline layout with one 64-byte push range. */
static bool frame_create_pipeline_layout(ReRenderer *r) {
    VkPushConstantRange range;
    VkPipelineLayoutCreateInfo info;
    VkResult res;

    range.stageFlags = VK_SHADER_STAGE_ALL;
    range.offset = 0;
    range.size = RE_PUSH_RANGE_SIZE;
    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.setLayoutCount = 1;
    info.pSetLayouts = &r->frame->set_layout;
    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges = &range;
    res = r->api.vkCreatePipelineLayout(r->device, &info, NULL, &r->pipe_layout);
    if (res != VK_SUCCESS) {
        re_log_error("renderer: vkCreatePipelineLayout failed (%d)", (int)res);
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* Samplers, one-shot submissions and the checker texture.            */
/* ------------------------------------------------------------------ */

/**
 * @brief Creates the shadow and bindless samplers.
 *
 * The shadow sampler is a plain (non-comparison) sampler because
 * re_renderer.h declares binding 3 as sampler2DArray, so PCF is the
 * fragment shader's job. If the shader ever switches to
 * sampler2DArrayShadow, enable the comparison with
 * VK_COMPARE_OP_GREATER_OR_EQUAL to match the reverse-Z depth buffer.
 */
static bool frame_create_samplers(ReRenderer *r) {
    VkSamplerCreateInfo info;
    VkResult res;

    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = VK_FILTER_LINEAR;
    info.minFilter = VK_FILTER_LINEAR;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.maxLod = 0.0f;
    res = r->api.vkCreateSampler(r->device, &info, NULL, &r->gpu.shadow_sampler);
    if (res != VK_SUCCESS) {
        re_log_error("renderer: shadow sampler creation failed (%d)", (int)res);
        return false;
    }
    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    res = r->api.vkCreateSampler(r->device, &info, NULL, &r->gpu.tex_sampler);
    if (res != VK_SUCCESS) {
        re_log_error("renderer: texture sampler creation failed (%d)", (int)res);
        return false;
    }
    return true;
}

/** @brief One-shot command buffer used by the creation-time uploads. */
typedef struct ReFrameOneShot {
    VkCommandPool pool;  /**< Transient pool. */
    VkCommandBuffer cmd; /**< Single primary buffer. */
    VkFence fence;       /**< Completion fence. */
} ReFrameOneShot;

/** @brief Opens a one-shot command buffer on the graphics queue. */
static bool frame_oneshot_begin(ReRenderer *r, ReFrameOneShot *os) {
    VkCommandPoolCreateInfo pool_info;
    VkCommandBufferAllocateInfo alloc_info;
    VkCommandBufferBeginInfo begin_info;
    VkFenceCreateInfo fence_info;

    memset(os, 0, sizeof *os);
    memset(&pool_info, 0, sizeof pool_info);
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = r->gfx_family;
    if (r->api.vkCreateCommandPool(r->device, &pool_info, NULL, &os->pool) != VK_SUCCESS) {
        return false;
    }
    memset(&alloc_info, 0, sizeof alloc_info);
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = os->pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    if (r->api.vkAllocateCommandBuffers(r->device, &alloc_info, &os->cmd) != VK_SUCCESS) {
        return false;
    }
    memset(&fence_info, 0, sizeof fence_info);
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (r->api.vkCreateFence(r->device, &fence_info, NULL, &os->fence) != VK_SUCCESS) {
        return false;
    }
    memset(&begin_info, 0, sizeof begin_info);
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    return r->api.vkBeginCommandBuffer(os->cmd, &begin_info) == VK_SUCCESS;
}

/** @brief Submits, waits for and tears down a one-shot command buffer. */
static bool frame_oneshot_end(ReRenderer *r, ReFrameOneShot *os) {
    VkSubmitInfo submit;
    VkResult res = VK_ERROR_UNKNOWN;

    if (r->api.vkEndCommandBuffer(os->cmd) == VK_SUCCESS) {
        memset(&submit, 0, sizeof submit);
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &os->cmd;
        res = r->api.vkQueueSubmit(r->gfx, 1, &submit, os->fence);
        if (res == VK_SUCCESS) {
            res = r->api.vkWaitForFences(r->device, 1, &os->fence, VK_TRUE, UINT64_MAX);
        }
    }
    if (os->fence != VK_NULL_HANDLE) {
        r->api.vkDestroyFence(r->device, os->fence, NULL);
    }
    if (os->cmd != VK_NULL_HANDLE) {
        r->api.vkFreeCommandBuffers(r->device, os->pool, 1, &os->cmd);
    }
    if (os->pool != VK_NULL_HANDLE) {
        r->api.vkDestroyCommandPool(r->device, os->pool, NULL);
    }
    return res == VK_SUCCESS;
}

/** @brief Stores host bytes in staging and copies them synchronously. */
static bool frame_upload_buffer_sync(ReRenderer *r, VkBuffer dst,
                                     const void *data, VkDeviceSize size) {
    ReFrameOneShot os;
    VkBufferCopy region;
    bool ok;

    if (!frame_stage(r, data, size) || !frame_oneshot_begin(r, &os)) {
        return false;
    }
    memset(&region, 0, sizeof region);
    region.srcOffset = 0u;
    region.dstOffset = 0u;
    region.size = size;
    r->api.vkCmdCopyBuffer(os.cmd, r->buf.staging, dst, 1u, &region);
    ok = frame_oneshot_end(r, &os);
    return ok;
}

/** @brief Uploads RGBA8 pixels into an image and leaves it SHADER_READ. */
static bool frame_upload_image(ReRenderer *r, VkImage image, uint32_t width, uint32_t height,
                               const void *pixels, VkDeviceSize size) {
    ReFrameOneShot os;
    VkBufferImageCopy region;
    bool ok;

    if (!frame_stage(r, pixels, size) || !frame_oneshot_begin(r, &os)) {
        return false;
    }
    frame_transition(os.cmd, &r->api, image, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                     VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    memset(&region, 0, sizeof region);
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width = width;
    region.imageExtent.height = height;
    region.imageExtent.depth = 1;
    r->api.vkCmdCopyBufferToImage(os.cmd, r->buf.staging, image,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    frame_transition(os.cmd, &r->api, image, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    ok = frame_oneshot_end(r, &os);
    return ok;
}

/** @brief Generates the 64x64 checker texture bound at bindless slot 0. */
static bool frame_create_checker(ReRenderer *r) {
    ReRendererFrame *f = r->frame;
    const uint32_t dim = RE_CHECKER_DIM;
    const VkDeviceSize bytes = (VkDeviceSize)dim * dim * 4u;
    unsigned char *pixels = malloc((size_t)bytes);
    bool ok;

    if (pixels == NULL) {
        return false;
    }
    for (uint32_t y = 0; y < dim; y++) {
        for (uint32_t x = 0; x < dim; x++) {
            const bool bright = ((x / 8u) + (y / 8u)) % 2u == 0u;
            unsigned char *pixel = &pixels[(y * dim + x) * 4u];
            pixel[0] = bright ? 225u : 45u;
            pixel[1] = bright ? 225u : 45u;
            pixel[2] = bright ? 235u : 60u;
            pixel[3] = 255u;
        }
    }
    ok = re_rr_create_image(r, VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, dim, dim,
                            1u, &f->checker, &f->checker_view, VK_IMAGE_ASPECT_COLOR_BIT) ==
        VK_SUCCESS;
    if (ok) {
        ok = frame_upload_image(r, f->checker, dim, dim, pixels, bytes);
    }
    free(pixels);
    if (!ok) {
        re_log_error("renderer: checker texture creation failed");
    }
    return ok;
}

/**
 * @brief Creates the per-frame uniform/SSBO buffers and the staging area.
 *
 * Every descriptor binding the frame layer writes points into this set;
 * the buffers live in the renderer's single host-visible block and are
 * destroyed by re_frame_destroy.
 *
 * @param r Renderer.
 * @return True when every buffer was created.
 */
static bool frame_create_buffers(ReRenderer *r) {
    const VkBufferUsageFlags ubo = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    if (re_rr_create_buffer(r, sizeof(ReRenderCameraUbo), ubo,
                            &r->buf.camera_ubo) != VK_SUCCESS ||
        re_rr_create_buffer(r, sizeof(ReRenderLightUbo), ubo,
                            &r->buf.light_ubo) != VK_SUCCESS ||
        re_rr_create_buffer(r,
                            (VkDeviceSize)RE_CLUSTER_COUNT *
                                sizeof(ReRenderClusterGpu),
                            ssbo, &r->buf.cluster_ssbo) != VK_SUCCESS ||
        re_rr_create_buffer(r, RE_LIGHT_BLOCK_BYTES, ssbo,
                            &r->buf.light_list_ssbo) != VK_SUCCESS ||
        re_rr_create_buffer(r,
                            (VkDeviceSize)RE_MAX_INSTANCES *
                                sizeof(ReSceneInstance),
                            ssbo, &r->buf.instance_ssbo) != VK_SUCCESS ||
        re_rr_create_buffer(r,
                            (VkDeviceSize)RE_MAX_INSTANCES * sizeof(uint32_t),
                            ssbo, &r->buf.visible_ssbo) != VK_SUCCESS ||
        re_rr_create_buffer(r, sizeof(ReRenderDrawArgs),
                            ssbo | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                            &r->buf.draw_args_ssbo) != VK_SUCCESS ||
        re_rr_create_buffer(r, RE_FRAME_STAGING_BYTES,
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            &r->buf.staging) != VK_SUCCESS) {
        re_log_error("renderer: frame buffer creation failed");
        return false;
    }
    r->buf.staging_size = RE_FRAME_STAGING_BYTES;
    return true;
}

/** @brief Creates one single-layer view per cascade of the shadow array. */
static bool frame_create_shadow_layers(ReRenderer *r) {
    ReRendererFrame *f = r->frame;

    for (uint32_t layer = 0; layer < RE_SHADOW_CASCADES; layer++) {
        VkImageViewCreateInfo info;
        VkResult res;

        memset(&info, 0, sizeof info);
        info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        info.image = r->gpu.shadow;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = VK_FORMAT_D16_UNORM;
        info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.baseArrayLayer = layer;
        info.subresourceRange.layerCount = 1;
        res = r->api.vkCreateImageView(r->device, &info, NULL, &f->shadow_layers[layer]);
        if (res != VK_SUCCESS) {
            re_log_error("renderer: shadow layer view %u creation failed (%d)", layer, (int)res);
            return false;
        }
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* Descriptor writes.                                                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Fills set 0 once; every handle it points at is stable.
 *
 * Bindless slot 0 is the checker texture and slot 1 is the HDR colour
 * image, which is what tonemap.frag samples. Slots 2..1023 stay unbound
 * and the layout declares the array partially bound.
 */
static void frame_write_descriptors(ReRenderer *r) {
    ReRendererFrame *f = r->frame;
    VkDescriptorBufferInfo camera;
    VkDescriptorBufferInfo lights;
    VkDescriptorBufferInfo clusters;
    VkDescriptorBufferInfo instances;
    VkDescriptorBufferInfo visible;
    VkDescriptorBufferInfo draw_args;
    VkDescriptorBufferInfo light_list;
    VkDescriptorImageInfo shadow;
    VkDescriptorImageInfo bindless[2];
    VkWriteDescriptorSet writes[9];

    camera.buffer = r->buf.camera_ubo;
    camera.offset = 0;
    camera.range = VK_WHOLE_SIZE;
    lights = camera;
    lights.buffer = r->buf.light_ubo;
    clusters = camera;
    clusters.buffer = r->buf.cluster_ssbo;
    instances = camera;
    instances.buffer = r->buf.instance_ssbo;
    visible = camera;
    visible.buffer = r->buf.visible_ssbo;
    draw_args = camera;
    draw_args.buffer = r->buf.draw_args_ssbo;
    light_list = camera;
    light_list.buffer = r->buf.light_list_ssbo;
    memset(&shadow, 0, sizeof shadow);
    shadow.sampler = r->gpu.shadow_sampler;
    shadow.imageView = r->gpu.shadow_view;
    shadow.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    memset(bindless, 0, sizeof bindless);
    bindless[0].sampler = r->gpu.tex_sampler;
    bindless[0].imageView = f->checker_view;
    bindless[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bindless[1].sampler = r->gpu.tex_sampler;
    bindless[1].imageView = r->gpu.color_view;
    bindless[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    memset(writes, 0, sizeof writes);
    for (uint32_t i = 0; i < 9u; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = r->dset;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    writes[RE_BIND_CAMERA_UBO].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[RE_BIND_CAMERA_UBO].pBufferInfo = &camera;
    writes[RE_BIND_LIGHT_UBO].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[RE_BIND_LIGHT_UBO].pBufferInfo = &lights;
    writes[RE_BIND_CLUSTER_SSBO].pBufferInfo = &clusters;
    writes[RE_BIND_INSTANCE_SSBO].pBufferInfo = &instances;
    writes[RE_BIND_VISIBLE_SSBO].pBufferInfo = &visible;
    writes[RE_BIND_DRAW_ARGS_SSBO].pBufferInfo = &draw_args;
    writes[RE_BIND_LIGHT_LIST_SSBO].pBufferInfo = &light_list;
    writes[RE_BIND_SHADOW_SAMPLERS].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[RE_BIND_SHADOW_SAMPLERS].pImageInfo = &shadow;
    writes[RE_BIND_BINDLESS_TEXTURES].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[RE_BIND_BINDLESS_TEXTURES].descriptorCount = 2;
    writes[RE_BIND_BINDLESS_TEXTURES].pImageInfo = bindless;
    r->api.vkUpdateDescriptorSets(r->device, 9, writes, 0, NULL);
}

/* ------------------------------------------------------------------ */
/* Pipelines.                                                         */
/* ------------------------------------------------------------------ */

/** @brief Everything that differs between the graphics pipelines. */
typedef struct ReFramePipeDesc {
    VkPipelineShaderStageCreateInfo stages[2];           /**< Stages. */
    uint32_t stage_count;                                /**< Used stages. */
    const VkVertexInputBindingDescription *bindings;     /**< Vertex bindings. */
    const VkVertexInputAttributeDescription *attributes; /**< Vertex attributes. */
    uint32_t binding_count;    /**< Bindings used. */
    uint32_t attribute_count;  /**< Attributes used. */
    VkFormat color_format;     /**< Undefined for depth-only passes. */
    VkFormat depth_format;     /**< Undefined for depth-less passes. */
    VkBool32 depth_write;      /**< Depth write enable. */
    VkCompareOp depth_compare; /**< Depth compare op. */
    VkCullModeFlags cull_mode; /**< Face culling. */
    VkPipeline *out;           /**< Receives the pipeline. */
} ReFramePipeDesc;

/** @brief Creates one graphics pipeline with dynamic rendering state. */
static bool frame_create_graphics_pipeline(ReRenderer *r, const ReFramePipeDesc *desc) {
    VkPipelineVertexInputStateCreateInfo vertex_input;
    VkPipelineInputAssemblyStateCreateInfo assembly;
    VkPipelineViewportStateCreateInfo viewport;
    VkPipelineRasterizationStateCreateInfo raster;
    VkPipelineMultisampleStateCreateInfo multisample;
    VkPipelineDepthStencilStateCreateInfo depth;
    VkPipelineColorBlendAttachmentState attachment;
    VkPipelineColorBlendStateCreateInfo blend;
    VkDynamicState dynamics[2];
    VkPipelineDynamicStateCreateInfo dynamic;
    VkPipelineRenderingCreateInfo rendering;
    VkGraphicsPipelineCreateInfo info;
    VkResult res;

    memset(&vertex_input, 0, sizeof vertex_input);
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount = desc->binding_count;
    vertex_input.pVertexBindingDescriptions = desc->bindings;
    vertex_input.vertexAttributeDescriptionCount = desc->attribute_count;
    vertex_input.pVertexAttributeDescriptions = desc->attributes;
    memset(&assembly, 0, sizeof assembly);
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    memset(&viewport, 0, sizeof viewport);
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    memset(&raster, 0, sizeof raster);
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = desc->cull_mode;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    memset(&multisample, 0, sizeof multisample);
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    memset(&depth, 0, sizeof depth);
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = desc->depth_format != VK_FORMAT_UNDEFINED ? VK_TRUE : VK_FALSE;
    depth.depthWriteEnable = desc->depth_write;
    depth.depthCompareOp = desc->depth_compare;
    memset(&attachment, 0, sizeof attachment);
    attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    memset(&blend, 0, sizeof blend);
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = desc->color_format != VK_FORMAT_UNDEFINED ? 1u : 0u;
    blend.pAttachments = &attachment;
    dynamics[0] = VK_DYNAMIC_STATE_VIEWPORT;
    dynamics[1] = VK_DYNAMIC_STATE_SCISSOR;
    memset(&dynamic, 0, sizeof dynamic);
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamics;
    memset(&rendering, 0, sizeof rendering);
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.colorAttachmentCount = desc->color_format != VK_FORMAT_UNDEFINED ? 1u : 0u;
    rendering.pColorAttachmentFormats =
        desc->color_format != VK_FORMAT_UNDEFINED ? &desc->color_format : NULL;
    rendering.depthAttachmentFormat = desc->depth_format;
    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.pNext = &rendering;
    info.stageCount = desc->stage_count;
    info.pStages = desc->stages;
    info.pVertexInputState = &vertex_input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = desc->depth_format != VK_FORMAT_UNDEFINED ? &depth : NULL;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = r->pipe_layout;
    res = r->api.vkCreateGraphicsPipelines(r->device, VK_NULL_HANDLE, 1, &info, NULL, desc->out);
    if (res != VK_SUCCESS) {
        re_log_error("renderer: graphics pipeline creation failed (%d)", (int)res);
        return false;
    }
    return true;
}

/** @brief Fills the vertex input shared by both primitive meshes. */
static void frame_vertex_layout(VkVertexInputBindingDescription *binding,
                                VkVertexInputAttributeDescription attributes[2]) {
    binding->binding = 0;
    binding->stride = (uint32_t)sizeof(ReFrameVertex);
    binding->inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    attributes[0].location = 0;
    attributes[0].binding = 0;
    attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset = 0;
    attributes[1].location = 1;
    attributes[1].binding = 0;
    attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].offset = (uint32_t)offsetof(ReFrameVertex, normal);
}

/** @brief Creates the compute pipeline used for frustum culling. */
static bool frame_create_cull_pipeline(ReRenderer *r) {
    VkComputePipelineCreateInfo info;
    VkResult res;

    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = r->pipes.modules[0];
    info.stage.pName = "main";
    info.layout = r->pipe_layout;
    res = r->api.vkCreateComputePipelines(r->device, VK_NULL_HANDLE, 1, &info, NULL,
                                          &r->pipes.cull_pipe);
    if (res != VK_SUCCESS) {
        re_log_error("renderer: cull pipeline creation failed (%d)", (int)res);
        return false;
    }
    return true;
}

/** @brief Fills a graphics pipeline descriptor for the shadow pass. */
static void frame_desc_shadow(ReRenderer *r, const VkVertexInputBindingDescription *binding,
                              const VkVertexInputAttributeDescription *attributes,
                              ReFramePipeDesc *desc) {
    memset(desc, 0, sizeof *desc);
    desc->stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    desc->stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    desc->stages[0].module = r->pipes.modules[3];
    desc->stages[0].pName = "main";
    desc->stage_count = 1;
    desc->bindings = binding;
    desc->attributes = attributes;
    desc->binding_count = 1;
    desc->attribute_count = 2;
    desc->depth_format = VK_FORMAT_D32_SFLOAT;
    desc->depth_write = VK_TRUE;
    desc->depth_compare = VK_COMPARE_OP_GREATER_OR_EQUAL;
    desc->cull_mode = VK_CULL_MODE_NONE;
    desc->out = &r->pipes.shadow_pipe;
}

/** @brief Fills a graphics pipeline descriptor for the forward pass. */
static void frame_desc_forward(ReRenderer *r, const VkVertexInputBindingDescription *binding,
                               const VkVertexInputAttributeDescription *attributes,
                               ReFramePipeDesc *desc) {
    frame_desc_shadow(r, binding, attributes, desc);
    /* frame_desc_shadow points the vertex stage at shadow.vert; the forward
     * pass needs the camera-space vertex shader instead. Without this the
     * forward draw consumes an unset light-view-projection push constant
     * and every vertex collapses, leaving only the clear colour visible. */
    desc->stages[0].module = r->pipes.modules[1];
    desc->stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    desc->stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    desc->stages[1].module = r->pipes.modules[2];
    desc->stages[1].pName = "main";
    desc->stage_count = 2;
    desc->depth_format = VK_FORMAT_D32_SFLOAT;
    desc->color_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    desc->cull_mode = VK_CULL_MODE_BACK_BIT;
    desc->out = &r->pipes.forward_pipe;
}

/**
 * @brief Creates the shadow, forward and tonemap pipelines.
 *
 * Tonemap vertex stage: modules[5] is the dedicated fullscreen shader
 * (tonemap.vert, gl_VertexIndex triangle, no vertex attributes), so the
 * tonemap draw needs no bound vertex buffer.
 */
static bool frame_create_graphics_pipelines(ReRenderer *r) {
    VkVertexInputBindingDescription binding;
    VkVertexInputAttributeDescription attributes[2];
    ReFramePipeDesc desc;
    VkFormat tonemap_format = r->sw_format != VK_FORMAT_UNDEFINED ? r->sw_format
                                                                 : VK_FORMAT_R16G16B16A16_SFLOAT;

    frame_vertex_layout(&binding, attributes);
    frame_desc_shadow(r, &binding, attributes, &desc);
    if (!frame_create_graphics_pipeline(r, &desc)) {
        return false;
    }
    frame_desc_forward(r, &binding, attributes, &desc);
    if (!frame_create_graphics_pipeline(r, &desc)) {
        return false;
    }
    memset(&desc, 0, sizeof desc);
    desc.stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    desc.stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    desc.stages[0].module = r->pipes.modules[5]; /* tonemap.vert: no attributes */
    desc.stages[0].pName = "main";
    desc.stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    desc.stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    desc.stages[1].module = r->pipes.modules[4];
    desc.stages[1].pName = "main";
    desc.stage_count = 2;
    desc.color_format = tonemap_format;
    desc.depth_format = VK_FORMAT_UNDEFINED;
    desc.depth_compare = VK_COMPARE_OP_ALWAYS;
    desc.cull_mode = VK_CULL_MODE_NONE;
    desc.out = &r->pipes.tonemap_pipe;
    return frame_create_graphics_pipeline(r, &desc);
}

/* ------------------------------------------------------------------ */
/* Camera, light and cluster data.                                    */
/* ------------------------------------------------------------------ */

/** @return The number of instances a snapshot actually submits. */
static uint32_t frame_instance_count(const ReSceneSnapshot *scene) {
    if (scene->instances == NULL) {
        return 0u;
    }
    return re_min_u32(scene->instance_count, RE_MAX_INSTANCES);
}

/** @return The far plane clamped to a finite, positive value. */
static float frame_safe_far(float far_plane) {
    if (!isfinite(far_plane) || far_plane <= 0.0f) {
        return 1000.0f;
    }
    return re_clamp_f32(far_plane, 0.01f, 100000.0f);
}

/** @return The near plane encoded in a reverse-Z perspective projection. */
static float frame_near_plane(ReMat4 proj, float far_plane) {
    float m22 = proj.col[2].z;
    float near_plane;

    if (!(m22 > 0.0f) || !isfinite(m22)) {
        return far_plane * 0.005f;
    }
    near_plane = far_plane * m22 / (1.0f + m22);
    if (!(near_plane > 0.0f) || near_plane >= far_plane) {
        return far_plane * 0.005f;
    }
    return near_plane;
}

/** @brief Extracts the orthonormal camera basis from a view matrix. */
static void frame_camera_basis(ReMat4 view, ReVec3 *right, ReVec3 *up, ReVec3 *forward) {
    *right = re_vec3_normalize(re_vec3_make(view.col[0].x, view.col[1].x, view.col[2].x));
    *up = re_vec3_normalize(re_vec3_make(view.col[0].y, view.col[1].y, view.col[2].y));
    *forward =
        re_vec3_normalize(re_vec3_neg(re_vec3_make(view.col[0].z, view.col[1].z, view.col[2].z)));
}

/** @brief Extracts the half-extent factors of a perspective projection. */
static void frame_camera_tans(ReMat4 proj, float *tan_x, float *tan_y) {
    float cx = proj.col[0].x;
    float cy = proj.col[1].y;

    *tan_x = cx > 0.0f && isfinite(cx) ? 1.0f / cx : 1.0f;
    *tan_y = cy > 0.0f && isfinite(cy) ? 1.0f / cy : 1.0f;
}

/** @brief Writes the eight world-space corners of one depth slab. */
static void frame_split_corners(ReVec3 eye, ReVec3 right, ReVec3 up, ReVec3 forward, float tan_x,
                                float tan_y, float d0, float d1, ReVec3 corners[8]) {
    const float depths[2] = {d0, d1};

    for (uint32_t i = 0; i < 2u; i++) {
        ReVec3 base = re_vec3_add(eye, re_vec3_scale(forward, depths[i]));
        ReVec3 rx = re_vec3_scale(right, depths[i] * tan_x);
        ReVec3 uy = re_vec3_scale(up, depths[i] * tan_y);
        corners[i * 4u + 0u] = re_vec3_sub(re_vec3_sub(base, rx), uy);
        corners[i * 4u + 1u] = re_vec3_sub(re_vec3_add(base, rx), uy);
        corners[i * 4u + 2u] = re_vec3_add(re_vec3_add(base, rx), uy);
        corners[i * 4u + 3u] = re_vec3_add(re_vec3_sub(base, rx), uy);
    }
}

/**
 * @brief Builds the light view-projection of one cascade.
 *
 * The slab is bounded by a sphere so the orthographic box is stable
 * under rotation; the eye sits 1.1 radii behind the centre along the
 * light direction, which keeps every corner in front of the near plane.
 */
static ReMat4 frame_cascade_matrix(ReSceneCamera camera, ReVec3 sun_dir, float d0, float d1) {
    ReVec3 right;
    ReVec3 up;
    ReVec3 forward;
    ReVec3 corners[8];
    ReVec3 center = re_vec3_zero();
    float tan_x;
    float tan_y;
    float radius = 0.0f;

    frame_camera_basis(camera.view, &right, &up, &forward);
    frame_camera_tans(camera.proj, &tan_x, &tan_y);
    frame_split_corners(camera.position, right, up, forward, tan_x, tan_y, d0, d1, corners);
    for (uint32_t i = 0; i < 8u; i++) {
        center = re_vec3_add(center, corners[i]);
    }
    center = re_vec3_scale(center, 1.0f / 8.0f);
    for (uint32_t i = 0; i < 8u; i++) {
        radius = re_max_f32(radius, re_vec3_length(re_vec3_sub(corners[i], center)));
    }
    radius = re_max_f32(radius, 0.01f);
    {
        ReVec3 eye = re_vec3_sub(center, re_vec3_scale(sun_dir, radius * 1.1f));
        ReVec3 up_ref = fabsf(sun_dir.y) > 0.99f ? re_vec3_make(0.0f, 0.0f, 1.0f)
                                                 : re_vec3_make(0.0f, 1.0f, 0.0f);
        ReMat4 view = re_mat4_look_at(eye, center, up_ref);
        ReMat4 proj = re_mat4_ortho(-radius, radius, -radius, radius, 0.0f, radius * 2.4f);
        return re_mat4_mul(proj, view);
    }
}

/** @brief Fills the camera block of the frame. */
static void frame_fill_camera(const ReSceneSnapshot *scene, uint32_t instance_count,
                              uint32_t light_count, ReRenderCameraUbo *ubo) {
    ReMat4 view_proj = re_mat4_mul(scene->camera.proj, scene->camera.view);
    ReFrustum frustum = re_frustum_from_mat4(view_proj);

    memset(ubo, 0, sizeof *ubo);
    ubo->view = scene->camera.view;
    ubo->proj = scene->camera.proj;
    ubo->view_proj = view_proj;
    for (uint32_t i = 0; i < 6u; i++) {
        ubo->frustum_planes[i] = frustum.planes[i];
    }
    ubo->cam_pos_far =
        re_vec4_make(scene->camera.position.x, scene->camera.position.y, scene->camera.position.z,
                     frame_safe_far(scene->camera.far_plane));
    ubo->instance_count = instance_count;
    ubo->point_light_count = light_count;
}

/**
 * @brief Fills the sun and cascade block, including both light matrices.
 *
 * Cascade ends use the usual blend of a linear and a logarithmic split
 * (0.5 each), which keeps the near cascade dense without collapsing the
 * far one. cascade_splits.x is the end of cascade 0, .y the far plane.
 */
static void frame_fill_light(const ReSceneSnapshot *scene, ReRenderLightUbo *ubo) {
    ReVec3 dir = re_vec3_normalize(scene->sun.direction);
    float far_plane = frame_safe_far(scene->camera.far_plane);
    float near_plane = frame_near_plane(scene->camera.proj, far_plane);
    float splits[RE_SHADOW_CASCADES];
    float prev = near_plane;

    if (re_vec3_length_sq(dir) < 1e-12f) {
        dir = re_vec3_make(0.0f, -1.0f, 0.0f);
    }
    for (uint32_t i = 1; i <= RE_SHADOW_CASCADES; i++) {
        float t = (float)i / (float)RE_SHADOW_CASCADES;
        float linear = near_plane + (far_plane - near_plane) * t;
        float logarithmic = near_plane * powf(far_plane / near_plane, t);
        splits[i - 1u] = 0.5f * linear + 0.5f * logarithmic;
    }
    memset(ubo, 0, sizeof *ubo);
    ubo->sun_dir_intensity = re_vec4_make(dir.x, dir.y, dir.z, scene->sun.intensity);
    ubo->sun_color_ambient = re_vec4_make(scene->sun.color.x, scene->sun.color.y,
                                          scene->sun.color.z, scene->sun.ambient);
    ubo->cascade_splits = re_vec4_make(splits[0], splits[RE_SHADOW_CASCADES - 1u], 0.0f, 0.0f);
    for (uint32_t i = 0; i < RE_SHADOW_CASCADES; i++) {
        ubo->light_vp[i] = frame_cascade_matrix(scene->camera, dir, prev, splits[i]);
        prev = splits[i];
    }
}

/** @brief Converts the scene point lights into the std430 record layout. */
static uint32_t frame_fill_point_lights(const ReSceneSnapshot *scene,
                                        ReRenderPointLightGpu *out) {
    uint32_t count = re_min_u32(scene->point_light_count, RE_MAX_POINT_LIGHTS);

    /* The whole record array travels to the GPU every frame, so clear the
     * unused tail instead of shipping last frame's lights. */
    memset(out, 0, (size_t)RE_MAX_POINT_LIGHTS * sizeof *out);
    if (scene->point_lights == NULL) {
        return 0u;
    }
    for (uint32_t i = 0; i < count; i++) {
        const RePointLight *light = &scene->point_lights[i];
        out[i].pos_radius = re_vec4_make(light->position.x, light->position.y, light->position.z,
                                         re_max_f32(light->radius, 0.0f));
        out[i].color_intensity = re_vec4_make(light->color.x, light->color.y, light->color.z,
                                              light->intensity);
    }
    return count;
}

/* ------------------------------------------------------------------ */
/* Cluster grid.                                                      */
/* ------------------------------------------------------------------ */

/** @return Linear index of a cell (z-major, then y, then x). */
static uint32_t frame_cluster_index(uint32_t x, uint32_t y, uint32_t z) {
    return (z * RE_CLUSTER_Y + y) * RE_CLUSTER_X + x;
}

/** @return The depth slice of a view-space depth (exponential distribution). */
static uint32_t frame_cluster_slice(float depth, float near_d, float far_d) {
    float ratio = logf(depth / near_d) / logf(far_d / near_d);
    float slice = ratio * (float)RE_CLUSTER_Z;

    /* Guard BEFORE the cast: float -> uint32_t is UB outside the range. */
    if (!(slice >= 0.0f) || !(slice < (float)RE_CLUSTER_Z)) {
        return !(slice >= 0.0f) ? 0u : RE_CLUSTER_Z - 1u;
    }
    return (uint32_t)slice;
}

/** @return The column or row of a normalized device coordinate. */
static uint32_t frame_cluster_axis(float ndc, uint32_t cells) {
    float unit = ndc * 0.5f + 0.5f;
    float scaled = unit * (float)cells;

    /* Guard BEFORE the cast: float -> uint32_t is UB outside the range. */
    if (!(scaled >= 0.0f) || !(scaled < (float)cells)) {
        return !(scaled >= 0.0f) ? 0u : cells - 1u;
    }
    return (uint32_t)scaled;
}

/**
 * @brief Computes the inclusive cluster range a light sphere overlaps.
 *
 * This is the CPU mirror of the fragment shader's lookup: x/y come from
 * the normalized device rectangle of the sphere, z from the exponential
 * depth distribution between near and far (slice index =
 * floor(log(depth / near) / log(far / near) * RE_CLUSTER_Z)).
 */
static ReClusterRange frame_cluster_range(const ReClusterCtx *ctx, ReVec3 world_pos,
                                          float radius) {
    ReClusterRange range;
    ReVec3 view_pos = re_mat4_transform_point(ctx->view, world_pos);
    float depth = -view_pos.z;
    float d0 = depth - radius;
    float d1 = depth + radius;
    float width = depth > ctx->near_d ? depth : ctx->near_d;
    float sx;
    float sy;

    memset(&range, 0, sizeof range);
    if (d1 <= ctx->near_d || d0 >= ctx->far_d) {
        range.empty = true;
        return range;
    }
    range.z0 = frame_cluster_slice(re_max_f32(d0, ctx->near_d), ctx->near_d, ctx->far_d);
    range.z1 = frame_cluster_slice(re_min_f32(d1, ctx->far_d), ctx->near_d, ctx->far_d);
    if (d0 <= ctx->near_d) {
        range.x0 = 0u;
        range.x1 = RE_CLUSTER_X - 1u;
        range.y0 = 0u;
        range.y1 = RE_CLUSTER_Y - 1u;
        return range;
    }
    sx = view_pos.x / (width * ctx->tan_x);
    sy = view_pos.y / (width * ctx->tan_y);
    range.x0 = frame_cluster_axis(sx - radius / (width * ctx->tan_x), RE_CLUSTER_X);
    range.x1 = frame_cluster_axis(sx + radius / (width * ctx->tan_x), RE_CLUSTER_X);
    range.y0 = frame_cluster_axis(sy - radius / (width * ctx->tan_y), RE_CLUSTER_Y);
    range.y1 = frame_cluster_axis(sy + radius / (width * ctx->tan_y), RE_CLUSTER_Y);
    return range;
}

/** @brief First pass: counts how many lights touch each cell. */
static void frame_cluster_count(ReRenderClusterGpu *grid, const ReClusterCtx *ctx,
                                const ReRenderPointLightGpu *lights, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        ReVec3 pos = re_vec3_make(lights[i].pos_radius.x, lights[i].pos_radius.y,
                                  lights[i].pos_radius.z);
        ReClusterRange r = frame_cluster_range(ctx, pos, lights[i].pos_radius.w);

        if (r.empty) {
            continue;
        }
        for (uint32_t z = r.z0; z <= r.z1; z++) {
            for (uint32_t y = r.y0; y <= r.y1; y++) {
                for (uint32_t x = r.x0; x <= r.x1; x++) {
                    grid[frame_cluster_index(x, y, z)].offset++;
                }
            }
        }
    }
}

/** @brief Turns the per-cell counters into offsets; returns the total. */
static uint32_t frame_cluster_prefix(ReRenderClusterGpu *grid, uint32_t *cursor) {
    uint32_t total = 0u;

    for (uint32_t i = 0; i < RE_CLUSTER_COUNT; i++) {
        uint32_t count = grid[i].offset;
        grid[i].offset = total;
        grid[i].count = 0u;
        cursor[i] = total;
        total += count;
    }
    return total;
}

/** @brief Second pass: writes the light indices into the packed list. */
static void frame_cluster_fill(uint32_t *cursor, const ReClusterCtx *ctx,
                               const ReRenderPointLightGpu *lights, uint32_t count,
                               uint32_t *list) {
    for (uint32_t i = 0; i < count; i++) {
        ReVec3 pos = re_vec3_make(lights[i].pos_radius.x, lights[i].pos_radius.y,
                                  lights[i].pos_radius.z);
        ReClusterRange r = frame_cluster_range(ctx, pos, lights[i].pos_radius.w);

        if (r.empty) {
            continue;
        }
        for (uint32_t z = r.z0; z <= r.z1; z++) {
            for (uint32_t y = r.y0; y <= r.y1; y++) {
                for (uint32_t x = r.x0; x <= r.x1; x++) {
                    uint32_t cell = frame_cluster_index(x, y, z);
                    if (cursor[cell] < RE_LIGHT_LIST_CAPACITY) {
                        list[cursor[cell]] = i;
                        cursor[cell]++;
                    }
                }
            }
        }
    }
}

/** @brief Stores the per-cell counts after truncation to the list capacity. */
static void frame_cluster_finalize(ReRenderClusterGpu *grid, const uint32_t *cursor) {
    for (uint32_t i = 0; i < RE_CLUSTER_COUNT; i++) {
        const uint32_t offset = grid[i].offset;
        grid[i].count = offset < RE_LIGHT_LIST_CAPACITY ? cursor[i] - offset : 0u;
    }
}

/**
 * @brief Rebuilds the cluster grid and the packed light list.
 *
 * Deterministic: lights are visited in scene order, cells in x/y/z order,
 * and the list is truncated (never wrapped) when the capacity is hit.
 */
static void frame_build_clusters(ReRendererFrame *f, const ReSceneSnapshot *scene,
                                 uint32_t light_count) {
    ReClusterCtx ctx;
    float far_d = frame_safe_far(scene->camera.far_plane);

    ctx.view = scene->camera.view;
    frame_camera_tans(scene->camera.proj, &ctx.tan_x, &ctx.tan_y);
    ctx.near_d = frame_near_plane(scene->camera.proj, far_d);
    ctx.far_d = far_d;
    memset(f->light_list, 0, (size_t)RE_LIGHT_LIST_CAPACITY * sizeof(uint32_t));
    for (uint32_t i = 0; i < RE_CLUSTER_COUNT; i++) {
        f->clusters[i].offset = 0u;
        f->clusters[i].count = 0u;
    }
    frame_cluster_count(f->clusters, &ctx, f->lights, light_count);
    (void)frame_cluster_prefix(f->clusters, f->cluster_cursor);
    frame_cluster_fill(f->cluster_cursor, &ctx, f->lights, light_count, f->light_list);
    frame_cluster_finalize(f->clusters, f->cluster_cursor);
}

/* ------------------------------------------------------------------ */
/* Per-frame uploads.                                                 */
/* ------------------------------------------------------------------ */

/** @brief Uploads the two UBOs, the instance array and the light block. */
static bool frame_upload_scene(ReRenderer *r, const ReSceneSnapshot *scene,
                               uint32_t instance_count) {
    ReRendererFrame *f = r->frame;
    VkDeviceSize instance_bytes = (VkDeviceSize)instance_count * sizeof(ReSceneInstance);

    if (!frame_upload_buffer(r, r->buf.camera_ubo, &f->camera, sizeof f->camera) ||
        !frame_upload_buffer(r, r->buf.light_ubo, &f->light, sizeof f->light) ||
        !frame_upload_buffer(r, r->buf.cluster_ssbo, f->clusters,
                             (VkDeviceSize)RE_CLUSTER_COUNT * sizeof(ReRenderClusterGpu)) ||
        !frame_upload_buffer(r, r->buf.light_list_ssbo, f->light_block, RE_LIGHT_BLOCK_BYTES)) {
        return false;
    }
    if (instance_bytes > 0u &&
        !frame_upload_buffer(r, r->buf.instance_ssbo, scene->instances, instance_bytes)) {
        return false;
    }
    return true;
}

/**
 * @brief Clears the atomic draw counter before the cull dispatch.
 *
 * ReVkApi has no vkCmdFillBuffer, so the 24-byte argument record travels
 * through staging like every other upload. instance_count stays at
 * RE_MAX_INSTANCES: the vertex shader gates on draw_count instead.
 */
static bool frame_reset_draw_args(ReRenderer *r) {
    ReRenderDrawArgs args;

    memset(&args, 0, sizeof args);
    args.instance_count = RE_MAX_INSTANCES;
    return frame_upload_buffer(r, r->buf.draw_args_ssbo, &args, sizeof args);
}

/** @brief Runs the CPU-side frame preparation and every upload. */
static bool frame_prepare(ReRenderer *r, const ReSceneSnapshot *scene, uint32_t instance_count) {
    ReRendererFrame *f = r->frame;
    uint32_t light_count;

    /* Every upload below goes through re_rr_upload, which writes the
     * persistent staging mapping; without it nothing can reach the GPU. */
    if (re_rr_mapped(r) == NULL) {
        re_log_error("renderer: staging buffer is not mapped");
        return false;
    }
    f->staging_cursor = 0u;
    light_count = frame_fill_point_lights(scene, f->lights);
    frame_fill_camera(scene, instance_count, light_count, &f->camera);
    frame_fill_light(scene, &f->light);
    frame_build_clusters(f, scene, light_count);
    if (!frame_upload_scene(r, scene, instance_count)) {
        return false;
    }
    return frame_reset_draw_args(r);
}

/* ------------------------------------------------------------------ */
/* Recording.                                                         */
/* ------------------------------------------------------------------ */

/** @brief Orders the uploads before the dispatch and the two draw passes. */
static void frame_record_upload_barrier(ReRenderer *r, VkCommandBuffer cb) {
    const VkBuffer buffers[10] = {
        r->buf.camera_ubo,        r->buf.light_ubo,
        r->buf.cluster_ssbo,      r->buf.light_list_ssbo,
        r->buf.instance_ssbo,     r->buf.draw_args_ssbo,
        r->frame->cube.vertex_buffer,   r->frame->cube.index_buffer,
        r->frame->sphere.vertex_buffer, r->frame->sphere.index_buffer};
    VkBufferMemoryBarrier2 barriers[10];
    VkDependencyInfo dep;

    memset(barriers, 0, sizeof barriers);
    for (uint32_t i = 0; i < 10u; i++) {
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barriers[i].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
        barriers[i].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT;
        barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                   VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                   VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT |
                                    VK_ACCESS_2_SHADER_WRITE_BIT |
                                    VK_ACCESS_2_UNIFORM_READ_BIT |
                                    VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[i].buffer = buffers[i];
        barriers[i].offset = 0;
        barriers[i].size = VK_WHOLE_SIZE;
    }
    memset(&dep, 0, sizeof dep);
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.bufferMemoryBarrierCount = 10;
    dep.pBufferMemoryBarriers = barriers;
    r->api.vkCmdPipelineBarrier2(cb, &dep);
}

/** @brief Dispatches the frustum cull over every submitted instance. */
static void frame_record_cull(ReRenderer *r, VkCommandBuffer cb, uint32_t instance_count) {
    uint32_t groups = (instance_count + RE_CULL_GROUP_SIZE - 1u) / RE_CULL_GROUP_SIZE;

    r->api.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, r->pipes.cull_pipe);
    r->api.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, r->pipe_layout, 0, 1,
                                   &r->dset, 0, NULL);
    r->api.vkCmdPushConstants(cb, r->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                              (uint32_t)sizeof instance_count, &instance_count);
    if (groups > 0u) {
        r->api.vkCmdDispatch(cb, groups, 1, 1);
    }
}

/** @brief Orders the cull writes before the vertex shader reads them. */
static void frame_record_cull_barrier(ReRenderer *r, VkCommandBuffer cb) {
    const VkPipelineStageFlags2 DST = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                      VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    const VkAccessFlags2 DST_ACCESS =
        VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;

    frame_buffer_dependency(cb, &r->api, r->buf.visible_ssbo,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                            DST, DST_ACCESS);
    frame_buffer_dependency(cb, &r->api, r->buf.draw_args_ssbo,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                            DST, DST_ACCESS);
}

/** @brief Binds a mesh and issues one instanced indexed draw. */
static void frame_draw_mesh(ReRenderer *r, VkCommandBuffer cb, const ReFrameMesh *mesh,
                            uint32_t instance_count) {
    VkDeviceSize offset = 0;

    if (instance_count == 0u || mesh->index_buffer == VK_NULL_HANDLE) {
        return;
    }
    r->api.vkCmdBindVertexBuffers(cb, 0, 1, &mesh->vertex_buffer, &offset);
    r->api.vkCmdBindIndexBuffer(cb, mesh->index_buffer, 0, VK_INDEX_TYPE_UINT32);
    r->api.vkCmdDrawIndexed(cb, mesh->index_count, instance_count, 0, 0, 0);
}

/** @brief Records the depth-only render of one cascade layer. */
static void frame_record_shadow_layer(ReRenderer *r, VkCommandBuffer cb, uint32_t instance_count,
                                      uint32_t layer) {
    VkRenderingAttachmentInfo depth;
    VkRenderingInfo info;
    VkRect2D area;
    ReMat4 light_vp = r->frame->light.light_vp[layer];

    memset(&depth, 0, sizeof depth);
    depth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth.imageView = r->frame->shadow_layers[layer];
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth.clearValue.depthStencil.depth = 0.0f;
    area.offset.x = 0;
    area.offset.y = 0;
    area.extent.width = RE_SHADOW_DIM;
    area.extent.height = RE_SHADOW_DIM;
    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    info.renderArea = area;
    info.layerCount = 1;
    info.pDepthAttachment = &depth;

    r->api.vkCmdBeginRendering(cb, &info);
    frame_set_viewport(cb, &r->api, RE_SHADOW_DIM, RE_SHADOW_DIM);
    r->api.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipes.shadow_pipe);
    r->api.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipe_layout, 0, 1,
                                   &r->dset, 0, NULL);
    r->api.vkCmdPushConstants(cb, r->pipe_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                              (uint32_t)sizeof light_vp, &light_vp);
    frame_draw_mesh(r, cb, &r->frame->cube, instance_count);
    frame_draw_mesh(r, cb, &r->frame->sphere, instance_count);
    r->api.vkCmdEndRendering(cb);
}

/** @brief Records the clustered forward pass into the HDR colour target. */
static void frame_record_forward(ReRenderer *r, VkCommandBuffer cb, const ReSceneSnapshot *scene) {
    VkRenderingAttachmentInfo color;
    VkRenderingAttachmentInfo depth;
    VkRenderingInfo info;
    VkRect2D area;

    memset(&color, 0, sizeof color);
    color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color.imageView = r->gpu.color_view;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color.float32[0] = re_max_f32(scene->clear_color.x, 0.0f);
    color.clearValue.color.float32[1] = re_max_f32(scene->clear_color.y, 0.0f);
    color.clearValue.color.float32[2] = re_max_f32(scene->clear_color.z, 0.0f);
    color.clearValue.color.float32[3] = 1.0f;
    memset(&depth, 0, sizeof depth);
    depth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth.imageView = r->gpu.depth_view;
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.clearValue.depthStencil.depth = 0.0f;
    area.offset.x = 0;
    area.offset.y = 0;
    area.extent.width = r->width;
    area.extent.height = r->height;
    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    info.renderArea = area;
    info.layerCount = 1;
    info.colorAttachmentCount = 1;
    info.pColorAttachments = &color;
    info.pDepthAttachment = &depth;

    r->api.vkCmdBeginRendering(cb, &info);
    frame_set_viewport_flipped(cb, &r->api, r->width, r->height);
    r->api.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipes.forward_pipe);
    r->api.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipe_layout, 0, 1,
                                   &r->dset, 0, NULL);
    /* Both meshes are instanced up to RE_MAX_INSTANCES; the vertex shader
     * degenerates every instance index above draw_count. */
    frame_draw_mesh(r, cb, &r->frame->cube, RE_MAX_INSTANCES);
    frame_draw_mesh(r, cb, &r->frame->sphere, RE_MAX_INSTANCES);
    r->api.vkCmdEndRendering(cb);
}

/** @brief Records the fullscreen tonemap pass into the swapchain image. */
static void frame_record_tonemap(ReRenderer *r, VkCommandBuffer cb) {
    VkRenderingAttachmentInfo color;
    VkRenderingInfo info;
    VkRect2D area;
    uint32_t index = r->image_index < RE_SW_MAX_IMAGES ? r->image_index : 0u;

    if (r->sw_views == NULL) {
        return;
    }
    memset(&color, 0, sizeof color);
    color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color.imageView = r->sw_views[index];
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    area.offset.x = 0;
    area.offset.y = 0;
    area.extent.width = r->sw_extent.width;
    area.extent.height = r->sw_extent.height;
    memset(&info, 0, sizeof info);
    info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    info.renderArea = area;
    info.layerCount = 1;
    info.colorAttachmentCount = 1;
    info.pColorAttachments = &color;

    r->api.vkCmdBeginRendering(cb, &info);
    frame_set_viewport(cb, &r->api, r->sw_extent.width, r->sw_extent.height);
    r->api.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipes.tonemap_pipe);
    r->api.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipe_layout, 0, 1,
                                   &r->dset, 0, NULL);
    {
        float target_size[2] = {(float)r->sw_extent.width, (float)r->sw_extent.height};
        r->api.vkCmdPushConstants(cb, r->pipe_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                  (uint32_t)sizeof target_size, target_size);
    }
    r->api.vkCmdDraw(cb, 3, 1, 0, 0);
    r->api.vkCmdEndRendering(cb);
}

/**
 * @brief Brings the attachments into their first-use layouts.
 *
 * UNDEFINED is used as the old layout for every attachment because all of
 * them are cleared before first use; the swapchain image is transitioned
 * separately, right before the tonemap pass.
 */
static void frame_record_setup(ReRenderer *r, VkCommandBuffer cb) {
    frame_to_color_attachment(cb, &r->api, r->gpu.color);
    frame_to_depth_attachment(cb, &r->api, r->gpu.depth, 1u);
    frame_to_depth_attachment(cb, &r->api, r->gpu.shadow, RE_SHADOW_CASCADES);
}

/** @brief Records the tail of the frame: tonemap or the headless hand-off. */
static void frame_record_resolve(ReRenderer *r, VkCommandBuffer cb) {
    bool windowed = r->sw_images != NULL && r->sw_views != NULL;
    uint32_t index = r->image_index < RE_SW_MAX_IMAGES ? r->image_index : 0u;

    if (!windowed) {
        /* Headless: the HDR image is the output; the caller reads it. */
        frame_color_to_read(cb, &r->api, r->gpu.color, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                            VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
        return;
    }
    frame_color_to_read(cb, &r->api, r->gpu.color, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                        VK_ACCESS_2_SHADER_READ_BIT);
    frame_to_color_attachment(cb, &r->api, r->sw_images[index]);
    frame_record_tonemap(r, cb);
    frame_color_to_present(cb, &r->api, r->sw_images[index]);
}

/* ------------------------------------------------------------------ */
/* Public entry points.                                               */
/* ------------------------------------------------------------------ */

/**
 * @brief Allocates the per-frame state and creates every pipeline.
 * @param r Renderer with device, queue, images and buffers already up.
 * @return False when a Vulkan object could not be created; the caller
 *         should fall back to the null renderer and call
 *         re_frame_destroy.
 */
bool re_frame_create(ReRenderer *r) {
    ReRendererFrame *f;

    if (r == NULL || r->device == VK_NULL_HANDLE) {
        return false;
    }
    f = calloc(1, sizeof *f);
    if (f == NULL) {
        return false;
    }
    r->frame = f;
    f->clusters = calloc(RE_CLUSTER_COUNT, sizeof *f->clusters);
    f->cluster_cursor = calloc(RE_CLUSTER_COUNT, sizeof *f->cluster_cursor);
    f->light_block = calloc(1u, RE_LIGHT_BLOCK_BYTES);
    if (f->clusters == NULL || f->cluster_cursor == NULL || f->light_block == NULL) {
        re_log_error("renderer: out of host memory for the frame state");
        re_frame_destroy(r);
        return false;
    }
    f->lights = (ReRenderPointLightGpu *)(void *)f->light_block;
    f->light_list = (uint32_t *)(void *)(f->light_block + RE_LIGHT_INDEX_OFFSET);
    if (!frame_create_buffers(r)) {
        re_frame_destroy(r);
        return false;
    }
    if (!frame_create_set_layout(r) || !frame_create_pool(r) || !frame_allocate_set(r) ||
        !frame_create_pipeline_layout(r)) {
        re_frame_destroy(r);
        return false;
    }
    if (!frame_create_module(r, "cull.comp.spv", &r->pipes.modules[0]) ||
        !frame_create_module(r, "pbr_clustered.vert.spv", &r->pipes.modules[1]) ||
        !frame_create_module(r, "pbr_clustered.frag.spv", &r->pipes.modules[2]) ||
        !frame_create_module(r, "shadow.vert.spv", &r->pipes.modules[3]) ||
        !frame_create_module(r, "tonemap.frag.spv", &r->pipes.modules[4]) ||
        !frame_create_module(r, "tonemap.vert.spv", &r->pipes.modules[5])) {
        re_frame_destroy(r);
        return false;
    }
    if (!frame_create_samplers(r) || !frame_create_shadow_layers(r) ||
        !frame_create_checker(r)) {
        re_frame_destroy(r);
        return false;
    }
    if (!frame_mesh_build(r, &f->cube, RE_CUBE_VERTEX_COUNT, RE_CUBE_INDEX_COUNT,
                          frame_build_cube) ||
        !frame_mesh_build(r, &f->sphere, RE_SPHERE_VERTEX_COUNT, RE_SPHERE_INDEX_COUNT,
                          frame_build_sphere)) {
        re_frame_destroy(r);
        return false;
    }
    if (!frame_create_cull_pipeline(r) || !frame_create_graphics_pipelines(r)) {
        re_frame_destroy(r);
        return false;
    }
    frame_write_descriptors(r);
    re_log_info("renderer: frame state ready (%u x %u, %u cascades, %u clusters)",
                r->width, r->height, RE_SHADOW_CASCADES, RE_CLUSTER_COUNT);
    return true;
}

/**
 * @brief Records one frame into the current slot's command buffer.
 *
 * The command buffer is begun first, then the per-frame host uploads are
 * staged and their copy commands are recorded into that same buffer.
 * Creation-time uploads use a dedicated one-shot command buffer instead
 * (frame_upload_buffer_sync/frame_upload_image), because no live frame
 * command buffer exists during re_frame_create. Submission and
 * presentation belong to re_renderer.c, which must not call this
 * function while the slot's buffer is still pending.
 *
 * @param r Renderer.
 * @param scene Snapshot to draw.
 * @return False when an upload or the recording itself failed.
 */
bool re_frame_render(ReRenderer *r, const ReSceneSnapshot *scene) {
    VkCommandBufferBeginInfo begin;
    VkCommandBuffer cb;
    uint32_t instances;

    if (r == NULL || r->frame == NULL || scene == NULL) {
        return false;
    }
    cb = VK_NULL_HANDLE;
    re_rr_cmd(r, &cb);
    if (cb == VK_NULL_HANDLE) {
        re_log_error("renderer: no command buffer for the current frame slot");
        return false;
    }
    instances = frame_instance_count(scene);
    memset(&begin, 0, sizeof begin);
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (r->api.vkBeginCommandBuffer(cb, &begin) != VK_SUCCESS) {
        re_log_error("renderer: vkBeginCommandBuffer failed");
        return false;
    }
    /* Uploads must be recorded INSIDE the begun command buffer:
     * re_rr_copy_to_buffer writes copy commands into this slot. */
    if (!frame_prepare(r, scene, instances)) {
        (void)r->api.vkEndCommandBuffer(cb);
        return false;
    }
    frame_record_upload_barrier(r, cb);
    frame_record_setup(r, cb);
    frame_record_cull(r, cb, instances);
    frame_record_cull_barrier(r, cb);
    for (uint32_t layer = 0; layer < RE_SHADOW_CASCADES; layer++) {
        frame_record_shadow_layer(r, cb, instances, layer);
    }
    frame_shadow_to_read(cb, &r->api, r->gpu.shadow);
    frame_record_forward(r, cb, scene);
    frame_record_resolve(r, cb);
    if (r->api.vkEndCommandBuffer(cb) != VK_SUCCESS) {
        re_log_error("renderer: vkEndCommandBuffer failed");
        return false;
    }
    r->frame->frames_rendered++;
    return true;
}

/**
 * @brief Releases everything re_frame_create created.
 *
 * Safe on a partially initialized renderer: every handle is checked
 * against VK_NULL_HANDLE before it is destroyed, and the per-frame
 * pointer is cleared so a second call is a no-op.
 * @param r Renderer.
 */
void re_frame_destroy(ReRenderer *r) {
    if (r == NULL) {
        return;
    }
    if (r->device != VK_NULL_HANDLE) {
        VkPipeline *pipelines[4];
        pipelines[0] = &r->pipes.shadow_pipe;
        pipelines[1] = &r->pipes.forward_pipe;
        pipelines[2] = &r->pipes.tonemap_pipe;
        pipelines[3] = &r->pipes.cull_pipe;
        for (uint32_t i = 0; i < 4u; i++) {
            if (*pipelines[i] != VK_NULL_HANDLE) {
                r->api.vkDestroyPipeline(r->device, *pipelines[i], NULL);
                *pipelines[i] = VK_NULL_HANDLE;
            }
        }
        for (uint32_t i = 0; i < 6u; i++) {
            if (r->pipes.modules[i] != VK_NULL_HANDLE) {
                r->api.vkDestroyShaderModule(r->device, r->pipes.modules[i], NULL);
                r->pipes.modules[i] = VK_NULL_HANDLE;
            }
        }
        if (r->pipe_layout != VK_NULL_HANDLE) {
            r->api.vkDestroyPipelineLayout(r->device, r->pipe_layout, NULL);
            r->pipe_layout = VK_NULL_HANDLE;
        }
        if (r->gpu.shadow_sampler != VK_NULL_HANDLE) {
            r->api.vkDestroySampler(r->device, r->gpu.shadow_sampler, NULL);
            r->gpu.shadow_sampler = VK_NULL_HANDLE;
        }
        if (r->gpu.tex_sampler != VK_NULL_HANDLE) {
            r->api.vkDestroySampler(r->device, r->gpu.tex_sampler, NULL);
            r->gpu.tex_sampler = VK_NULL_HANDLE;
        }
        {
            VkBuffer *const buffers[] = {
                &r->buf.camera_ubo,   &r->buf.light_ubo,
                &r->buf.cluster_ssbo, &r->buf.light_list_ssbo,
                &r->buf.instance_ssbo, &r->buf.visible_ssbo,
                &r->buf.draw_args_ssbo, &r->buf.staging};
            for (size_t i = 0u; i < RE_ARRAY_LEN(buffers); i++) {
                if (*buffers[i] != VK_NULL_HANDLE) {
                    r->api.vkDestroyBuffer(r->device, *buffers[i], NULL);
                    *buffers[i] = VK_NULL_HANDLE;
                }
            }
        }
    }
    if (r->frame == NULL) {
        return;
    }
    if (r->device != VK_NULL_HANDLE) {
        ReRendererFrame *f = r->frame;

        if (f->cube.vertex_buffer != VK_NULL_HANDLE) {
            r->api.vkDestroyBuffer(r->device, f->cube.vertex_buffer, NULL);
        }
        if (f->cube.index_buffer != VK_NULL_HANDLE) {
            r->api.vkDestroyBuffer(r->device, f->cube.index_buffer, NULL);
        }
        if (f->sphere.vertex_buffer != VK_NULL_HANDLE) {
            r->api.vkDestroyBuffer(r->device, f->sphere.vertex_buffer, NULL);
        }
        if (f->sphere.index_buffer != VK_NULL_HANDLE) {
            r->api.vkDestroyBuffer(r->device, f->sphere.index_buffer, NULL);
        }
        if (f->checker_view != VK_NULL_HANDLE) {
            r->api.vkDestroyImageView(r->device, f->checker_view, NULL);
        }
        if (f->checker != VK_NULL_HANDLE) {
            r->api.vkDestroyImage(r->device, f->checker, NULL);
        }
        for (uint32_t i = 0; i < RE_SHADOW_CASCADES; i++) {
            if (f->shadow_layers[i] != VK_NULL_HANDLE) {
                r->api.vkDestroyImageView(r->device, f->shadow_layers[i], NULL);
            }
        }
        if (f->pool != VK_NULL_HANDLE) {
            r->api.vkDestroyDescriptorPool(r->device, f->pool, NULL);
            r->dset = VK_NULL_HANDLE;
        }
        if (f->set_layout != VK_NULL_HANDLE) {
            r->api.vkDestroyDescriptorSetLayout(r->device, f->set_layout, NULL);
        }
    }
    {
        ReRendererFrame *f = r->frame;
        free(f->cube.vertices);
        free(f->cube.indices);
        free(f->sphere.vertices);
        free(f->sphere.indices);
        free(f->clusters);
        free(f->cluster_cursor);
        free(f->light_block);
        free(f);
    }
    r->frame = NULL;
}

#else /* !RE_HAS_VULKAN */

/* Without Vulkan headers this backend does not exist; the null renderer
   (re_null.c) provides the implementation. A typedef keeps the
   translation unit non-empty under -Wpedantic. */
typedef int re_frame_requires_vulkan;

#endif /* RE_HAS_VULKAN */
