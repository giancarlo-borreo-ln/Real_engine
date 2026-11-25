/**
 * @file pbr_clustered.vert
 * @brief Clustered forward PBR vertex stage, also reused as the tonemap
 *        fullscreen vertex stage.
 *
 * Forward path: fetch the model matrix through the visible list produced
 * by cull.comp, transform position and normal, forward the flat material
 * id and the clip position. Instances at or above draw_count degenerate.
 *
 * Tonemap path: the engine's tonemap pipeline reuses this shader and draws
 * three vertices with no vertex buffer bound (see
 * frame_create_graphics_pipelines in src/re_renderer_frame.c). An unbound
 * vertex attribute reads as zero under Vulkan robustness, while every cube
 * and sphere vertex has a non-zero position and normal, so all-zero
 * attributes at gl_VertexIndex 0..2 identify the fullscreen draw. The
 * shader then emits the standard oversized screen triangle.
 *
 * Descriptor set 0 bindings (see src/re_renderer.h):
 *   0 camera UBO (std140)
 *   5 instance SSBO (std430)
 *   6 visible list SSBO (std430)
 *   7 draw args SSBO (std430, draw_count at byte 20)
 *
 * No push constants.
 */
#version 460

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

layout(location = 0) out vec3 v_world_position;
layout(location = 1) out vec3 v_world_normal;
layout(location = 2) flat out uint v_material_id;

/* Binding 0: camera block. view_proj is the only member used here. */
layout(std140, set = 0, binding = 0) uniform Camera {
    mat4 view;
    mat4 proj;
    mat4 view_proj;
    vec4 frustum_planes[6];
    vec4 cam_pos_far;
    uint instance_count;
    uint point_light_count;
    vec2 padding;
} camera;

/* Binding 5: 96-byte instance record. aabb_min_material.w carries the
 * material id as a raw uint32 bit pattern, hence floatBitsToUint. */
struct Instance {
    mat4 model;
    vec4 aabb_min_material;
    vec4 aabb_max_flags;
};

layout(std430, set = 0, binding = 5) readonly buffer Instances {
    Instance instances[];
} instance_buffer;

/* Binding 6: visible instance indices, in cull append order. */
layout(std430, set = 0, binding = 6) readonly buffer Visible {
    uint visible_list[];
} visible_buffer;

/* Binding 7: draw arguments; only the visible count is needed here. */
layout(std430, set = 0, binding = 7) readonly buffer DrawArgs {
    uint index_count;
    uint instance_count;
    int first_index;
    int vertex_offset;
    uint first_instance;
    uint draw_count;
} draw_args;

/** @brief Emits the oversized fullscreen triangle for the tonemap draw. */
void emit_fullscreen_triangle() {
    vec2 corner = vec2(float((gl_VertexIndex << 1u) & 2u),
                       float(gl_VertexIndex & 2u));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);

    /* The tonemap fragment stage consumes none of these. */
    v_world_position = vec3(0.0);
    v_world_normal = vec3(0.0, 0.0, 1.0);
    v_material_id = 0u;
}

void main() {
    /* No vertex buffer bound: both attributes are zero. See the file
     * comment for why this is the tonemap draw and not a mesh vertex. */
    if (gl_VertexIndex < 3u &&
        all(equal(a_position, vec3(0.0))) &&
        all(equal(a_normal, vec3(0.0)))) {
        emit_fullscreen_triangle();
        return;
    }

    if (gl_InstanceIndex >= draw_args.draw_count) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    uint visible_index = visible_buffer.visible_list[gl_InstanceIndex];
    Instance instance = instance_buffer.instances[visible_index];

    vec4 world_position = instance.model * vec4(a_position, 1.0);

    v_world_position = world_position.xyz;
    /* mat3(model) is correct only for rigid or uniformly scaled models;
     * non-uniform scale would require the inverse transpose. */
    v_world_normal = mat3(instance.model) * a_normal;
    v_material_id = floatBitsToUint(instance.aabb_min_material.w);

    gl_Position = camera.view_proj * world_position;
}
