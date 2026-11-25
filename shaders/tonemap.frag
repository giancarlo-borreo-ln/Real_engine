/**
 * @file tonemap.frag
 * @brief Fullscreen tonemap: HDR colour -> ACES fitted -> gamma.
 *
 * The vertex stage is pbr_clustered.vert reused with no vertex buffer and
 * exactly three vertices (see frame_create_graphics_pipelines in
 * src/re_renderer_frame.c), so this stage needs no vertex inputs.
 *
 * The HDR colour lives in bindless slot 1. The UV is derived from the
 * presentation target size, passed as a push constant: the HDR image can
 * be smaller than the swapchain (for example under fractional desktop
 * scaling), and dividing by the HDR size would tile or clip the image.
 *
 * Descriptor set 0 binding 4: bindless textures (sampler2D[]).
 *
 * Push constant: vec2 target_size (presentation extent in pixels).
 */
#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) out vec4 out_color;

/* Binding 4: bindless array; slot 0 checker, slot 1 HDR colour. */
layout(set = 0, binding = 4) uniform sampler2D textures[];

layout(push_constant) uniform TonemapPush {
    vec2 target_size;
} push_data;

/** @brief ACES fitted curve (Narkowicz) applied per channel. */
vec3 aces_fitted(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    return clamp((color * (a * color + b)) / (color * (c * color + d) + e),
                 0.0, 1.0);
}

void main() {
    vec2 uv = gl_FragCoord.xy / push_data.target_size;
    vec3 hdr = texture(textures[nonuniformEXT(1)], uv).rgb;
    vec3 mapped = aces_fitted(max(hdr, vec3(0.0)));

    out_color = vec4(pow(mapped, vec3(1.0 / 2.2)), 1.0);
}
