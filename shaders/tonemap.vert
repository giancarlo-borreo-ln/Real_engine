/**
 * @file tonemap.vert
 * @brief Fullscreen-triangle vertex shader for the tonemap pass.
 *
 * No vertex buffer, no instance data: the three vertices come from
 * gl_VertexIndex. Emits the oversized triangle that covers the whole
 * clip-space square, so tonemap.frag runs once per pixel. This replaces
 * reusing the mesh vertex shader for the post pass, whose vertex
 * attributes would be undefined without a bound vertex buffer.
 */
#version 460

/** Clip-space corners: x = -1..3, y = -1..3 covers [0,1]^2 fully. */
const vec2 corners[3] = vec2[3](
    vec2(-1.0, -1.0),
    vec2(3.0, -1.0),
    vec2(-1.0, 3.0)
);

void main() {
    gl_Position = vec4(corners[gl_VertexIndex], 0.0, 1.0);
}
