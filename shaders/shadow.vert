/**
 * @file shadow.vert
 * @brief Depth-only cascaded shadow map vertex stage.
 *
 * The renderer issues one draw per cascade layer with push constant
 * light_vp = cascade projection * light view. The instance model matrix is
 * fetched through the visible list produced by cull.comp; instances at or
 * above draw_count collapse to a degenerate triangle and rasterize nothing.
 *
 * Reverse-Z: the cascade projection is a reverse-Z orthographic matrix, so
 * near maps to depth 1 and far to depth 0, and the engine clears the shadow
 * layer to 0 with VK_COMPARE_OP_GREATER_OR_EQUAL. gl_Position.z is written
 * unchanged: no flip and no vertex-stage bias. The fragment shader applies
 * the comparison bias while sampling.
 *
 * Descriptor set 0 bindings (see src/re_renderer.h):
 *   5 instance SSBO (std430)
 *   6 visible list SSBO (std430)
 *   7 draw args SSBO (std430, draw_count at byte 20)
 *
 * Push constant: mat4 light_vp (offset 0, 64 bytes).
 */
#version 460

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

/* Binding 5: 96-byte instance record. */
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

/* Push constant: light view-projection of the cascade being rendered. */
layout(push_constant) uniform PushShadow {
    mat4 light_vp;
} push_data;

/** @brief Collapses an invisible instance to a degenerate clip position. */
vec4 degenerate_position() {
    return vec4(0.0, 0.0, 2.0, 1.0);
}

void main() {
    /* The draw is instanced over the whole submitted range; everything at
     * or above the atomic counter was culled. */
    if (gl_InstanceIndex >= draw_args.draw_count) {
        gl_Position = degenerate_position();
        return;
    }

    uint visible_index = visible_buffer.visible_list[gl_InstanceIndex];
    mat4 model = instance_buffer.instances[visible_index].model;
    vec4 world_position = model * vec4(a_position, 1.0);

    gl_Position = push_data.light_vp * world_position;
}
