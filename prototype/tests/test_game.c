/**
 * @file test_game.c
 * @brief Deterministic, headless tests for the platformer simulation.
 *
 * The suite is silent on success and exits non-zero on the first failed
 * check. Every test runs on the real ReWorld, so the physics edge cases
 * (stepping, landing, falling out of the level) are the actual engine
 * paths, not mocks.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "character.h"
#include "game.h"
#include "scene.h"

/** Backing memory for each test world (bytes). */
#define TEST_WORLD_MEMORY (8u * 1024u * 1024u)

/** Fixed timestep used by every test. */
#define TEST_DT (1.0f / (float)GAME_TICK_HZ)

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

/**
 * @brief Runs a scripted input for a number of ticks.
 * @param g Game.
 * @param move Horizontal input.
 * @param jump_pressed Jump edge on the first tick only.
 * @param jump_held Jump held for every tick.
 * @param ticks Tick count.
 */
static void run_ticks(Game *g, float move, bool jump_pressed, bool jump_held,
                      uint32_t ticks)
{
    uint32_t i;

    for (i = 0u; i < ticks; i++) {
        GameInput in;
        in.move = move;
        in.jump_pressed = jump_pressed && i == 0u;
        in.jump_held = jump_held;
        game_step(g, &in, TEST_DT);
    }
}

/** @return Player body of a game, never NULL for an initialised game. */
static ReBody *player(Game *g)
{
    ReBody *p = re_world_body(&g->world, g->player);
    CHECK(p != NULL);
    return p;
}

/** @brief Spawn, settle, movement, jumping and ground detection. */
static void test_movement(Game *g)
{
    ReBody *p;
    float start_x;

    p = player(g);
    start_x = p->position.x;
    run_ticks(g, 0.0f, false, false, 120u);
    CHECK(g->grounded);
    CHECK(p->position.y > 0.70f && p->position.y < 0.95f);

    run_ticks(g, 1.0f, false, false, 84u);
    CHECK(p->position.x > start_x + 3.0f);
    CHECK(g->run_distance > 1.0f);

    game_reset(g);
    run_ticks(g, 0.0f, false, false, 60u);
    CHECK(g->grounded);
    start_x = p->position.x;
    run_ticks(g, 0.0f, true, true, 30u);
    CHECK(p->position.y > 1.30f);
    CHECK(!g->grounded);
    CHECK(g->air_time > 0.1f);
    run_ticks(g, 0.0f, false, true, 120u);
    CHECK(g->grounded);
    CHECK(p->position.x > start_x - 0.75f);
}

/** @brief Coins are collected by proximity exactly once. */
static void test_coins(Game *g)
{
    ReBody *p;

    game_reset(g);
    p = player(g);
    p->position = re_vec3_make(g->coins[1].x, g->coins[1].y - 0.2f, 0.0f);
    p->lin_vel = re_vec3_zero();
    game_step(g, &(GameInput){0.0f, false, false}, TEST_DT);
    CHECK(g->coins[1].collected);
    CHECK(g->coins_collected == 1u);

    game_step(g, &(GameInput){0.0f, false, false}, TEST_DT);
    CHECK(g->coins_collected == 1u);
}

/** @brief Falling below the kill plane respawns without clearing coins. */
static void test_fall_respawn(Game *g)
{
    ReBody *p;
    uint32_t collected;

    game_reset(g);
    p = player(g);
    p->position = re_vec3_make(10.0f, GAME_KILL_Y - 1.0f, 0.0f);
    collected = g->coins_collected;
    game_step(g, &(GameInput){0.0f, false, false}, TEST_DT);
    CHECK(g->deaths == 1u);
    CHECK(fabsf(p->position.x - g->spawn_x) < 0.01f);
    CHECK(fabsf(p->position.y - g->spawn_y) < 0.01f);
    CHECK(g->coins_collected == collected);
}

/** @brief Reset rewinds coins, counters, flags and the camera. */
static void test_reset(Game *g)
{
    game_reset(g);
    g->coins[0].collected = true;
    g->coins_collected = 1u;
    g->won = true;
    g->deaths = 3u;
    g->time = 4.0f;
    game_reset(g);
    CHECK(!g->coins[0].collected);
    CHECK(g->coins_collected == 0u);
    CHECK(!g->won);
    CHECK(g->deaths == 0u);
    CHECK(g->time == 0.0f);
    CHECK(fabsf(player(g)->position.x - g->spawn_x) < 0.01f);
}

/** @brief The bot can finish the level, which exercises the whole route. */
static void test_autopilot_finishes(Game *g)
{
    uint32_t tick;
    uint32_t max_ticks = 45u * GAME_TICK_HZ;

    game_reset(g);
    for (tick = 0u; tick < max_ticks && !g->won; tick++) {
        GameInput in;
        game_autopilot(g, &in);
        game_step(g, &in, TEST_DT);
        CHECK(player(g)->position.y > GAME_KILL_Y);
    }
    CHECK(g->won);
    CHECK(g->coins_collected > 0u);
    CHECK(g->deaths == 0u);
}

/** @brief The rig emits finite, bounded parts in every animation state. */
static void test_character_pose(Game *g)
{
    CharacterPose pose;
    uint32_t i;

    game_reset(g);
    character_build_pose(g, &pose);
    CHECK(pose.count > 8u);
    for (i = 0u; i < pose.count; i++) {
        CHECK(isfinite(pose.parts[i].center.x));
        CHECK(isfinite(pose.parts[i].center.y));
        CHECK(isfinite(pose.parts[i].half.x));
        CHECK(pose.parts[i].half.x > 0.0f);
    }

    run_ticks(g, 0.0f, false, false, 60u);
    CHECK(g->grounded);
    run_ticks(g, 1.0f, true, true, 30u);
    character_build_pose(g, &pose);
    CHECK(pose.count > 8u);
    CHECK(!g->grounded);
}

/** @brief Scene construction counts every object and keeps AABBs valid. */
static void test_scene(Game *g)
{
    GameScene scene;
    ReMat4 view_proj;
    ReVec4 clip;
    ReVec3 center;
    uint32_t i;

    CharacterPose pose;

    game_reset(g);
    character_build_pose(g, &pose);
    CHECK(pose.count > 8u);
    CHECK(pose.count <= CHARACTER_MAX_PARTS);
    game_build_scene(g, 1280u, 720u, &scene);
    CHECK(scene.instance_count ==
          g->platform_count + g->coin_count + 1u + pose.count);
    CHECK(scene.light_count == g->coin_count + 1u);
    for (i = 0u; i < scene.instance_count; i++) {
        CHECK(scene.instances[i].aabb_min_material.x <=
              scene.instances[i].aabb_max_flags.x);
        CHECK(scene.instances[i].aabb_min_material.y <=
              scene.instances[i].aabb_max_flags.y);
        CHECK(scene.instances[i].aabb_min_material.z <=
              scene.instances[i].aabb_max_flags.z);
    }

    /* The player must project inside the viewport, not behind the camera. */
    center = player(g)->position;
    view_proj = re_mat4_mul(scene.camera.proj, scene.camera.view);
    clip = re_mat4_transform_vec4(view_proj, (ReVec4){center.x, center.y, center.z, 1.0f});
    CHECK(clip.w > 0.1f);
    CHECK(fabsf(clip.x / clip.w) < 1.0f);
    CHECK(fabsf(clip.y / clip.w) < 1.0f);
}

int main(void)
{
    void *mem = malloc(TEST_WORLD_MEMORY);
    Game *g = malloc(sizeof *g);

    CHECK(mem != NULL && g != NULL);
    CHECK(game_init(g, mem, TEST_WORLD_MEMORY));
    CHECK(g->platform_count > 0u);
    CHECK(g->coin_count > 0u);
    CHECK(g->player != RE_BODY_NULL);

    test_movement(g);
    test_character_pose(g);
    test_coins(g);
    test_fall_respawn(g);
    test_reset(g);
    test_scene(g);
    test_autopilot_finishes(g);

    free(g);
    free(mem);
    return 0;
}
