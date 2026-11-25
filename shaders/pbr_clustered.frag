/**
 * @file pbr_clustered.frag
 * @brief Clustered forward+ PBR fragment stage (linear HDR output).
 *
 * Shading model: one directional sun (GGX specular, Lambert diffuse,
 * Schlick fresnel) with a two-cascade shadow lookup, plus point lights read
 * from the per-cluster light list. The output is linear HDR radiance; the
 * tonemap pass is responsible for ACES and gamma.
 *
 * Texturing: the vertex layout has no UV attribute, so the albedo lookup
 * uses a planar world-space projection (world_pos.xz * 0.5). Roughness and
 * metallic are constants (0.6 / 0.0) because ReSceneInstance carries no
 * material parameters beyond the bindless texture index.
 *
 * Reverse-Z: the shadow cascades store reverse-Z depth (1 near the light,
 * 0 far) and are cleared to 0. A fragment is lit when its light-space depth
 * plus a small bias is greater than or equal to the stored depth; PCF is a
 * manual 3x3 comparison because binding 3 is a plain sampler2DArray, not a
 * comparison sampler.
 *
 * Descriptor set 0 bindings (see src/re_renderer.h):
 *   0 camera UBO (std140)
 *   1 light UBO (std140)
 *   2 cluster grid SSBO (std430, uvec2 cells)
 *   3 shadow sampler (sampler2DArray, RE_SHADOW_CASCADES layers)
 *   4 bindless textures (sampler2D[], nonuniformEXT indices)
 *   8 light list SSBO (std430: 128 x 32-byte records, then uint indices)
 *
 * No push constants.
 */
#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 v_world_position;
layout(location = 1) in vec3 v_world_normal;
layout(location = 2) flat in uint v_material_id;

layout(location = 0) out vec4 out_color;

/* Binding 0: camera block (320 bytes). */
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

/* Binding 1: sun and cascade block (192 bytes). */
layout(std140, set = 0, binding = 1) uniform Lights {
    vec4 sun_dir_intensity;
    vec4 sun_color_ambient;
    vec4 cascade_splits;
    mat4 light_vp[2];
} light_data;

/* Binding 2: 16 x 8 x 16 cluster grid; .x = first index into the light
 * index region, .y = number of lights stored for the cell. */
layout(std430, set = 0, binding = 2) readonly buffer ClusterGrid {
    uvec2 clusters[2048];
} cluster_grid;

/* Binding 3: two-layer depth array, plain (non-comparison) sampler. */
layout(set = 0, binding = 3) uniform sampler2DArray shadow_sampler;

/* Binding 4: bindless albedo array; slot 0 is the checker texture. */
layout(set = 0, binding = 4) uniform sampler2D textures[];

/**
 * @brief Flat material palette for game characters and level props.
 *
 * Ids must match prototype/src/palette.h. Material 0 keeps the checkered
 * bindless texture; anything else is a stylised flat colour.
 */
vec3 material_color(uint id) {
    if (id == 1u) return vec3(0.89, 0.87, 0.81); /* robe */
    if (id == 2u) return vec3(0.73, 0.65, 0.50); /* cream */
    if (id == 3u) return vec3(0.34, 0.22, 0.13); /* leather */
    if (id == 4u) return vec3(0.62, 0.10, 0.10); /* red sash */
    if (id == 5u) return vec3(0.72, 0.52, 0.36); /* skin */
    if (id == 6u) return vec3(0.11, 0.11, 0.13); /* dark cloth */
    if (id == 7u) return vec3(0.88, 0.66, 0.16); /* gold */
    if (id == 8u) return vec3(0.12, 0.72, 0.28); /* goal */
    if (id == 9u) return vec3(0.56, 0.51, 0.44); /* stone */
    if (id == 10u) return vec3(0.37, 0.33, 0.29); /* dark stone */
    return vec3(0.8, 0.8, 0.8);
}

/* Binding 8: 128 point-light records (32 bytes each, 4096 bytes), followed
 * by the packed cluster light indices. std430 places the runtime array at
 * byte 4096, which is exactly ReRenderFrame's RE_LIGHT_INDEX_OFFSET. */
struct PointLightGpu {
    vec4 pos_radius;
    vec4 color_intensity;
};

layout(std430, set = 0, binding = 8) readonly buffer LightList {
    PointLightGpu records[128];
    uint light_indices[];
} light_list;

const float PI = 3.14159265359;
const float SHADOW_TEXEL = 1.0 / 2048.0;
const float SHADOW_BIAS = 0.0010;
const float SHADOW_SLOPE_BIAS = 0.0015;
const float POINT_LIGHT_GAIN = 10.0;
const uint CLUSTER_X = 16u;
const uint CLUSTER_Y = 8u;
const uint CLUSTER_Z = 16u;
const uint MAX_POINT_LIGHTS = 128u;

/** @brief GGX/Trowbridge-Reitz normal distribution term. */
float distribution_ggx(float n_dot_h, float roughness) {
    float alpha = roughness * roughness;
    float alpha_sq = alpha * alpha;
    float denominator = n_dot_h * n_dot_h * (alpha_sq - 1.0) + 1.0;

    return alpha_sq / (PI * denominator * denominator);
}

/** @brief Smith Schlick-GGX geometry term for direct lighting. */
float geometry_smith(float n_dot_v, float n_dot_l, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float g_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    float g_l = n_dot_l / (n_dot_l * (1.0 - k) + k);

    return g_v * g_l;
}

/** @brief Schlick fresnel approximation. */
vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (vec3(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

/** @brief Recovers the near plane from a reverse-Z perspective matrix. */
float view_near_plane(float far_distance) {
    float m22 = camera.proj[2][2];

    if (m22 > 0.0 && !isinf(m22)) {
        float near_distance = far_distance * m22 / (1.0 + m22);

        if (near_distance > 0.0 && near_distance < far_distance) {
            return near_distance;
        }
    }
    return far_distance * 0.005;
}

/** @brief Maps a normalized device coordinate to a cluster column/row. */
uint cluster_axis(float ndc, uint cells) {
    float scaled = (ndc * 0.5 + 0.5) * float(cells);

    /* Guard before the float-to-uint conversion: out-of-range is UB. */
    if (!(scaled >= 0.0)) {
        return 0u;
    }
    if (scaled >= float(cells)) {
        return cells - 1u;
    }
    return uint(scaled);
}

/** @brief Exponential depth slice, mirroring the CPU cluster builder. */
uint cluster_slice(float depth, float near_distance, float far_distance) {
    float scaled = log(depth / near_distance) / log(far_distance / near_distance) *
                   float(CLUSTER_Z);

    if (!(scaled >= 0.0)) {
        return 0u;
    }
    if (scaled >= float(CLUSTER_Z)) {
        return CLUSTER_Z - 1u;
    }
    return uint(scaled);
}

/** @brief Cluster cell of a world position, using the camera projection.
 *
 * The CPU builder derives the cell from the view-space position and the
 * projection tangents; projecting the position and using the resulting NDC
 * is the same mapping, and avoids needing the viewport size here.
 */
uint cluster_index(vec3 world_position, float view_depth) {
    vec4 clip = camera.proj * (camera.view * vec4(world_position, 1.0));
    float far_distance = camera.cam_pos_far.w;
    uint x;
    uint y;
    uint z;

    if (!(clip.w > 0.0)) {
        return 0u;
    }
    x = cluster_axis(clip.x / clip.w, CLUSTER_X);
    y = cluster_axis(clip.y / clip.w, CLUSTER_Y);
    z = cluster_slice(view_depth, view_near_plane(far_distance), far_distance);

    return (z * CLUSTER_Y + y) * CLUSTER_X + x;
}

/** @brief 3x3 PCF lookup against one reverse-Z shadow cascade. */
float sun_shadow(vec3 world_position, float n_dot_l) {
    float view_depth = -(camera.view * vec4(world_position, 1.0)).z;
    uint cascade;
    vec4 light_clip;
    vec3 ndc;
    vec2 uv;
    float bias;
    float lit = 0.0;

    /* Outside the cascaded range: nothing occludes the sun. */
    if (view_depth > light_data.cascade_splits.y) {
        return 1.0;
    }
    cascade = (view_depth <= light_data.cascade_splits.x) ? 0u : 1u;

    light_clip = light_data.light_vp[cascade] * vec4(world_position, 1.0);
    ndc = light_clip.xyz / light_clip.w;
    uv = ndc.xy * 0.5 + 0.5;

    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) ||
        ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }

    /* Bias grows at grazing angles where depth slope is largest. */
    bias = SHADOW_BIAS + SHADOW_SLOPE_BIAS * (1.0 - clamp(n_dot_l, 0.0, 1.0));

    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 offset = vec2(float(x), float(y)) * SHADOW_TEXEL;
            float stored = textureLod(shadow_sampler,
                                      vec3(uv + offset, float(cascade)), 0.0).r;

            /* Reverse-Z: larger means closer to the light. */
            lit += step(stored, ndc.z + bias);
        }
    }
    return lit / 9.0;
}

/** @brief Adds every point light stored in the fragment's cluster cell. */
void add_point_lights(vec3 world_position, vec3 normal, vec3 albedo,
                      inout vec3 radiance) {
    float view_depth = -(camera.view * vec4(world_position, 1.0)).z;
    uint cell_index = cluster_index(world_position, view_depth);
    uvec2 cell = cluster_grid.clusters[cell_index];

    for (uint i = 0u; i < cell.y; ++i) {
        uint light_index = light_list.light_indices[cell.x + i];
        PointLightGpu light;
        vec3 to_light;
        float distance;
        float radius;
        float falloff;
        float n_dot_l;

        if (light_index >= MAX_POINT_LIGHTS) {
            continue;
        }
        light = light_list.records[light_index];
        to_light = light.pos_radius.xyz - world_position;
        distance = length(to_light);
        radius = max(light.pos_radius.w, 1e-4);

        if (distance >= radius) {
            continue;
        }
        n_dot_l = max(dot(normal, to_light / max(distance, 1e-4)), 0.0);
        falloff = clamp(1.0 - distance / radius, 0.0, 1.0);
        falloff *= falloff;

        /* Deliberately simple material response: Lambert albedo scaled by
         * an arbitrary 10x gain (see the shader contract). */
        radiance += albedo * light.color_intensity.rgb *
                    (light.color_intensity.w * falloff * n_dot_l *
                     POINT_LIGHT_GAIN);
    }
}

void main() {
    vec3 world_position = v_world_position;
    vec3 normal = normalize(v_world_normal);
    vec3 view_dir = normalize(camera.cam_pos_far.xyz - world_position);
    vec2 uv = world_position.xz * 0.5;
    vec3 albedo = v_material_id == 0u
                      ? texture(textures[nonuniformEXT(v_material_id)], uv).rgb
                      : material_color(v_material_id);
    float roughness = 0.6;
    float metallic = 0.0;
    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    vec3 light_dir = normalize(-light_data.sun_dir_intensity.xyz);
    vec3 half_dir = normalize(view_dir + light_dir);
    float n_dot_l = max(dot(normal, light_dir), 0.0);
    float n_dot_v = max(dot(normal, view_dir), 1e-4);
    float n_dot_h = max(dot(normal, half_dir), 0.0);
    float v_dot_h = max(dot(view_dir, half_dir), 0.0);
    vec3 sun_color = light_data.sun_color_ambient.rgb;
    float sun_intensity = light_data.sun_dir_intensity.w;
    vec3 radiance = vec3(0.0);

    if (n_dot_l > 0.0) {
        float shadow = sun_shadow(world_position, n_dot_l);
        float d = distribution_ggx(n_dot_h, roughness);
        float g = geometry_smith(n_dot_v, n_dot_l, roughness);
        vec3 f = fresnel_schlick(v_dot_h, f0);
        vec3 specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
        vec3 diffuse = (vec3(1.0) - f) * (1.0 - metallic) * albedo / PI;

        radiance += (diffuse + specular) * sun_color * sun_intensity *
                    n_dot_l * shadow;
    }

    /* Sun colour doubles as the ambient tint; .w is the ambient strength. */
    radiance += albedo * sun_color * light_data.sun_color_ambient.w;

    add_point_lights(world_position, normal, albedo, radiance);

    out_color = vec4(radiance, 1.0);
}
