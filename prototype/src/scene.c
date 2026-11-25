/**
 * @file scene.c
 * @brief Instance and camera construction for the platformer prototype.
 *
 * Every visible object is an axis-aligned box: the engine's forward pass
 * draws the unit cube through the instance SSBO, so a model matrix of
 * translate * rotate * scale is all a platform, a coin or the player
 * needs. Coin spin is visual-only; gameplay uses the coin centre.
 */
#include "scene.h"

#include <math.h>
#include <string.h>

#include "character.h"
#include "palette.h"
#include "re_core.h"

/** Coin visual edge length (metres). */
#define COIN_VISUAL_SIZE 0.45f

/** Finish pillar visual size (metres). */
#define GOAL_VISUAL_X 0.8f
#define GOAL_VISUAL_Y 2.4f
#define GOAL_VISUAL_Z 0.8f

/** Camera distance behind the z = 0 gameplay plane. */
#define CAMERA_DISTANCE 12.0f

/** Vertical field of view of the side camera (radians). */
#define CAMERA_FOV_Y 0.7853981634f

/**
 * @brief Appends one box instance.
 * @param scene Scene being built.
 * @param center World centre.
 * @param visual_half Half extents of the drawn box.
 * @param aabb_half Half extents used for culling (>= visual_half).
 * @param rotation Rotation applied before scaling.
 */
static void add_box(GameScene *scene, ReVec3 center, ReVec3 visual_half,
                    ReVec3 aabb_half, ReQuat rotation, uint32_t material)
{
    ReSceneInstance *instance;
    ReMat4 model;
    ReVec3 size;

    if (scene->instance_count >= GAME_SCENE_MAX_INSTANCES) {
        return;
    }
    instance = &scene->instances[scene->instance_count++];
    size = re_vec3_make(visual_half.x * 2.0f, visual_half.y * 2.0f,
                        visual_half.z * 2.0f);
    model = re_mat4_translate(center);
    model = re_mat4_mul(model, re_quat_to_mat4(rotation));
    model = re_mat4_mul(model, re_mat4_scale(size));

    instance->model = model;
    instance->aabb_min_material =
        re_vec4_make(center.x - aabb_half.x, center.y - aabb_half.y,
                     center.z - aabb_half.z, re_bits_to_f32(material));
    instance->aabb_max_flags =
        re_vec4_make(center.x + aabb_half.x, center.y + aabb_half.y,
                     center.z + aabb_half.z,
                     re_bits_to_f32(RE_INSTANCE_SHADOW_CASTER));
}

/** @brief Appends an axis-aligned box instance. */
static void add_aabb_box(GameScene *scene, ReVec3 center, ReVec3 half, uint32_t material)
{
    add_box(scene, center, half, half, re_quat_identity(), material);
}

/** @brief Submits the static platforms. */
static void add_platforms(const Game *g, GameScene *scene)
{
    uint32_t i;

    for (i = 0u; i < g->platform_count; i++) {
        const GamePlatform *p = &g->platforms[i];
        add_aabb_box(scene, re_vec3_make(p->cx, p->cy, p->cz),
                     re_vec3_make(p->hx, p->hy, p->hz),
                     (i % 2u == 0u) ? GAME_MAT_STONE : GAME_MAT_STONE_DARK);
    }
}

/** @brief Submits the procedural hooded-assassin rig. */
static void add_character(const Game *g, GameScene *scene)
{
    CharacterPose pose;
    uint32_t i;

    character_build_pose(g, &pose);
    for (i = 0u; i < pose.count; i++) {
        const CharacterPart *part = &pose.parts[i];
        float radius = re_vec3_length(part->half);
        add_box(scene, part->center, part->half,
                re_vec3_make(radius, radius, radius), part->rotation,
                part->material);
    }
}

/** @brief Submits uncollected coins with a bob and spin, plus their glows. */
static void add_coins(const Game *g, GameScene *scene)
{
    uint32_t i;

    for (i = 0u; i < g->coin_count; i++) {
        const GameCoin *coin = &g->coins[i];
        ReQuat spin;
        ReVec3 center;
        RePointLight *light;

        if (coin->collected) {
            continue;
        }
        center = re_vec3_make(coin->x,
                              coin->y + sinf(g->time * 2.5f + (float)i) * 0.08f,
                              coin->z);
        spin = re_quat_from_axis_angle(re_vec3_make(0.0f, 1.0f, 0.0f),
                                       g->time * 3.0f + (float)i);
        add_box(scene, center,
                re_vec3_make(COIN_VISUAL_SIZE, COIN_VISUAL_SIZE, COIN_VISUAL_SIZE),
                re_vec3_make(COIN_VISUAL_SIZE * 1.5f, COIN_VISUAL_SIZE * 1.5f,
                             COIN_VISUAL_SIZE * 1.5f),
                spin, GAME_MAT_GOLD);
        if (scene->light_count < GAME_SCENE_MAX_LIGHTS) {
            light = &scene->lights[scene->light_count++];
            light->position = center;
            light->radius = 2.6f;
            light->color = re_vec3_make(1.0f, 0.55f, 0.12f);
            light->intensity = 1.8f;
        }
    }
}

/** @brief Submits the finish pillar and its green glow. */
static void add_finish(const Game *g, GameScene *scene)
{
    ReVec3 center = re_vec3_make(g->goal_x, g->goal_y, 0.0f);
    RePointLight *light;

    add_aabb_box(scene, center,
                 re_vec3_make(GOAL_VISUAL_X * 0.5f, GOAL_VISUAL_Y * 0.5f,
                              GOAL_VISUAL_Z * 0.5f), GAME_MAT_GOAL);
    if (scene->light_count < GAME_SCENE_MAX_LIGHTS) {
        light = &scene->lights[scene->light_count++];
        light->position = center;
        light->radius = 5.0f;
        light->color = re_vec3_make(0.15f, 1.0f, 0.35f);
        light->intensity = 3.0f;
    }
}

/** @brief Fills the side-view camera from the smoothed focus point. */
static void fill_camera(const Game *g, uint32_t width, uint32_t height, GameScene *scene)
{
    float aspect = (height > 0u) ? (float)width / (float)height : 1.0f;
    ReVec3 eye = re_vec3_make(g->cam_x + 1.5f, g->cam_y + 2.2f, CAMERA_DISTANCE);
    ReVec3 target = re_vec3_make(g->cam_x + 0.9f, g->cam_y + 0.6f, 0.0f);

    scene->camera.view = re_mat4_look_at(eye, target, re_vec3_make(0.0f, 1.0f, 0.0f));
    scene->camera.proj = re_mat4_perspective(CAMERA_FOV_Y, aspect, 0.1f, 100.0f);
    scene->camera.position = eye;
    scene->camera.far_plane = 100.0f;
}

void game_build_scene(const Game *g, uint32_t width, uint32_t height, GameScene *out)
{
    if (g == NULL || out == NULL) {
        return;
    }
    memset(out, 0, sizeof *out);

    add_platforms(g, out);
    add_coins(g, out);
    add_finish(g, out);
    add_character(g, out);

    out->sun.direction = re_vec3_make(0.35f, -1.0f, -0.45f);
    out->sun.color = re_vec3_make(1.0f, 0.94f, 0.82f);
    out->sun.intensity = 3.2f;
    out->sun.ambient = 0.20f;
    out->clear_color = re_vec3_make(0.10f, 0.14f, 0.22f);

    fill_camera(g, width, height, out);
}
