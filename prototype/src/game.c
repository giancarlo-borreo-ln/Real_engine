/**
 * @file game.c
 * @brief Single-level platformer simulation over the engine's rigid world.
 *
 * The player is a vertical capsule whose rotation is pinned: contacts
 * still generate linear impulses, but inv_inertia is zero and the
 * orientation is restored every tick, so the character never tips over.
 * Movement is velocity-driven (accelerate towards the input direction)
 * while gravity and collision response stay owned by ReWorld.
 *
 * Ground detection reads the contact list of the tick that just ran:
 * a contact supports the player when its normal points up towards the
 * player's centre of mass. This is cheaper and more reliable than a
 * downward ray because it uses the exact contacts the solver used.
 */
#include "game.h"

#include <math.h>
#include <string.h>

#include "re_core.h"

/** Half turn used to stand the capsule up (local +X becomes world +Y). */
#define GAME_HALF_PI 1.5707963268f

/** Static level geometry. */
static const GamePlatform LEVEL_PLATFORMS[] = {
    {-6.0f, -1.0f, 0.0f, 6.0f, 1.0f, 2.0f},  /* Starting ground, top y = 0. */
    {3.0f, 0.5f, 0.0f, 3.0f, 0.5f, 2.0f},    /* First step, top y = 1.0. */
    {10.0f, 1.8f, 0.0f, 2.5f, 0.4f, 2.0f},   /* Floating platform, top 2.2. */
    {16.0f, 3.2f, 0.0f, 2.2f, 0.4f, 2.0f},   /* High platform, top 3.6. */
    {23.0f, 1.5f, 0.0f, 3.0f, 0.4f, 2.0f},   /* Descending platform, top 1.9. */
    {33.0f, -1.0f, 0.0f, 9.5f, 1.0f, 2.0f},  /* Final ground, top y = 0. */
    {38.0f, 1.2f, 0.0f, 2.0f, 0.5f, 2.0f},   /* Finish platform, top 1.7. */
    {-12.5f, 2.0f, 0.0f, 0.5f, 3.0f, 2.0f},  /* Left boundary wall. */
    {43.0f, 2.0f, 0.0f, 0.5f, 3.0f, 2.0f},   /* Right boundary wall. */
};

/** Coins placed along the intended route. */
static const GameCoin LEVEL_COINS[] = {
    {-2.5f, 1.6f, 0.0f, false},
    {3.0f, 2.2f, 0.0f, false},
    {10.0f, 3.4f, 0.0f, false},
    {16.0f, 4.8f, 0.0f, false},
    {23.0f, 3.1f, 0.0f, false},
    {33.0f, 1.2f, 0.0f, false},
    {38.0f, 2.6f, 0.0f, false},
};

/** Spawn point (centre of the capsule). */
#define GAME_SPAWN_X (-6.0f)
#define GAME_SPAWN_Y 1.0f

/** Finish pillar position. */
#define GAME_FINISH_X 38.0f
#define GAME_FINISH_Y 2.9f

/**
 * @brief Builds the default world tuning for a platformer.
 * @return Sanitized world configuration.
 */
static ReWorldConfig game_world_config(void)
{
    ReWorldConfig cfg = RE_WORLD_CONFIG_DEFAULT;

    cfg.gravity_y = -24.0f;
    cfg.max_bodies = 64u;
    cfg.max_contacts = 1024u;
    cfg.max_pairs = 2048u;
    cfg.bvh_capacity = 512u;
    cfg.solver_iterations = 8u;
    cfg.broadphase_margin = 0.02f;
    cfg.sleep_lin_vel = 0.0f;
    cfg.sleep_ang_vel = 0.0f;
    cfg.sleep_time = 100000u;
    cfg.damping = 0.0f;
    return cfg;
}

/**
 * @brief Adds one static box platform to the world.
 * @param g Game whose world receives the body.
 * @param p Platform description.
 * @return Body id, or RE_BODY_NULL on failure.
 */
static uint32_t add_platform(Game *g, const GamePlatform *p)
{
    ReBody body;

    memset(&body, 0, sizeof body);
    body.position = re_vec3_make(p->cx, p->cy, p->cz);
    body.orientation = re_quat_identity();
    body.inv_mass = 0.0f;
    body.inv_inertia = re_vec3_zero();
    body.restitution = 0.0f;
    body.friction = 0.85f;
    body.kind = RE_SHAPE_BOX;
    body.half = re_vec3_make(p->hx, p->hy, p->hz);
    body.active = false;
    return re_world_add_body(&g->world, &body);
}

/**
 * @brief Adds the player capsule body.
 * @return Player body id, or RE_BODY_NULL on failure.
 */
static uint32_t add_player(Game *g)
{
    ReBody body;

    memset(&body, 0, sizeof body);
    body.position = re_vec3_make(GAME_SPAWN_X, GAME_SPAWN_Y, 0.0f);
    body.orientation = re_quat_from_axis_angle(re_vec3_make(0.0f, 0.0f, 1.0f),
                                               GAME_HALF_PI);
    body.inv_mass = 1.0f;
    body.inv_inertia = re_vec3_zero(); /* Rotation is locked. */
    body.restitution = 0.0f;
    body.friction = 0.12f;
    body.kind = RE_SHAPE_CAPSULE;
    body.radius = GAME_PLAYER_RADIUS;
    body.half = re_vec3_make(GAME_PLAYER_HALF, 0.0f, 0.0f);
    body.active = true;
    return re_world_add_body(&g->world, &body);
}

/** @brief Copies the static level tables into the game state. */
static void load_level_data(Game *g)
{
    g->platform_count = (uint32_t)RE_ARRAY_LEN(LEVEL_PLATFORMS);
    memcpy(g->platforms, LEVEL_PLATFORMS, sizeof LEVEL_PLATFORMS);

    g->coin_count = (uint32_t)RE_ARRAY_LEN(LEVEL_COINS);
    memcpy(g->coins, LEVEL_COINS, sizeof LEVEL_COINS);

    g->spawn_x = GAME_SPAWN_X;
    g->spawn_y = GAME_SPAWN_Y;
    g->goal_x = GAME_FINISH_X;
    g->goal_y = GAME_FINISH_Y;
}

/** @brief Puts the player back at the spawn point without touching score. */
static void respawn_player(Game *g)
{
    ReBody *p = re_world_body(&g->world, g->player);

    if (p == NULL) {
        return;
    }
    p->position = re_vec3_make(g->spawn_x, g->spawn_y, 0.0f);
    p->orientation = re_quat_from_axis_angle(re_vec3_make(0.0f, 0.0f, 1.0f),
                                             GAME_HALF_PI);
    p->lin_vel = re_vec3_zero();
    p->ang_vel = re_vec3_zero();
    p->force = re_vec3_zero();
    p->torque = re_vec3_zero();
    p->sleep_timer = 0u;
    p->active = true;
    g->grounded = false;
    g->coyote = 0.0f;
    g->jump_buffer = 0.0f;
    g->cam_x = g->spawn_x;
    g->cam_y = g->spawn_y;
    g->run_distance = 0.0f;
    g->air_time = 0.0f;
    g->land_timer = 0.0f;
    g->roll_timer = 0.0f;
    g->pre_step_vy = 0.0f;
    g->facing = 1.0f;
    g->was_grounded = false;
}

bool game_init(Game *g, void *mem, size_t mem_size)
{
    ReWorldConfig cfg;
    uint32_t i;

    if (g == NULL || mem == NULL) {
        return false;
    }
    memset(g, 0, sizeof *g);
    load_level_data(g);
    cfg = game_world_config();
    if (!re_world_init(&g->world, &cfg, mem, mem_size)) {
        return false;
    }
    for (i = 0u; i < g->platform_count; i++) {
        if (add_platform(g, &g->platforms[i]) == RE_BODY_NULL) {
            return false;
        }
    }
    g->player = add_player(g);
    if (g->player == RE_BODY_NULL) {
        return false;
    }
    game_reset(g);
    return true;
}

void game_reset(Game *g)
{
    uint32_t i;

    if (g == NULL) {
        return;
    }
    for (i = 0u; i < g->coin_count; i++) {
        g->coins[i].collected = false;
    }
    g->won = false;
    g->time = 0.0f;
    g->deaths = 0u;
    g->coins_collected = 0u;
    respawn_player(g);
}

/**
 * @brief True when a contact from this tick holds the player from below.
 * @param g Game state after a world step.
 * @return True when at least one contact supports the player.
 */
static bool player_is_grounded(const Game *g)
{
    const ReWorld *w = &g->world;
    const ReBody *p;
    uint32_t i;

    if (g->player >= w->body_count) {
        return false;
    }
    p = &w->bodies[g->player];
    for (i = 0u; i < w->contact_count; i++) {
        const ReContact *c = &w->contacts[i];
        ReVec3 up;
        if (c->a != g->player && c->b != g->player) {
            continue;
        }
        /* Contact normals point from B to A. */
        up = (c->a == g->player) ? c->normal : re_vec3_neg(c->normal);
        if (up.y > 0.55f && c->point.y < p->position.y + 0.05f) {
            return true;
        }
    }
    return false;
}

/** @brief Accelerates the player's horizontal velocity toward the target. */
static void apply_horizontal(ReBody *p, float move, float dt)
{
    float target = re_clamp_f32(move, -1.0f, 1.0f) * GAME_MOVE_SPEED;
    float accel = (p->lin_vel.y > -0.5f && p->lin_vel.y < 0.5f) ? GAME_GROUND_ACCEL
                                                               : GAME_AIR_ACCEL;
    float max_dv = accel * dt;

    p->lin_vel.x += re_clamp_f32(target - p->lin_vel.x, -max_dv, max_dv);
}

/**
 * @brief Applies one tick of player control before the physics step.
 * @param g Game state (grounded/coyote from the previous tick).
 * @param p Player body.
 * @param in Input.
 * @param dt Timestep.
 */
static void apply_control(Game *g, ReBody *p, const GameInput *in, float dt)
{
    if (in->jump_pressed) {
        g->jump_buffer = GAME_JUMP_BUFFER;
    }
    apply_horizontal(p, in->move, dt);
    p->lin_vel.x = re_clamp_f32(p->lin_vel.x, -GAME_MOVE_SPEED, GAME_MOVE_SPEED);

    if (g->jump_buffer > 0.0f && (g->grounded || g->coyote > 0.0f)) {
        p->lin_vel.y = GAME_JUMP_SPEED;
        g->jump_buffer = 0.0f;
        g->coyote = 0.0f;
        g->grounded = false;
    }
    if (p->lin_vel.y > 0.0f && !in->jump_held) {
        p->lin_vel.y -= GAME_JUMP_CUT * dt;
    }
}

/** @brief Collects coins close enough to the player's centre. */
static void collect_coins(Game *g, const ReBody *p)
{
    uint32_t i;

    for (i = 0u; i < g->coin_count; i++) {
        GameCoin *coin = &g->coins[i];
        float dx;
        float dy;
        float dz;
        if (coin->collected) {
            continue;
        }
        dx = coin->x - p->position.x;
        dy = coin->y - p->position.y;
        dz = coin->z - p->position.z;
        if (dx * dx + dy * dy + dz * dz <= GAME_COIN_RADIUS * GAME_COIN_RADIUS) {
            coin->collected = true;
            g->coins_collected++;
        }
    }
}

/** @brief Eases the camera focus toward the player. */
static void follow_camera(Game *g, const ReBody *p, float dt)
{
    float k = re_min_f32(1.0f, dt * 6.0f);
    float target_x = p->position.x + 1.2f;
    float target_y = p->position.y + 1.0f;

    g->cam_x += (target_x - g->cam_x) * k;
    g->cam_y += (target_y - g->cam_y) * k;
}

void game_step(Game *g, const GameInput *in, float dt)
{
    ReBody *p;
    float goal_dx;
    float goal_dy;

    if (g == NULL || in == NULL || !(dt > 0.0f)) {
        return;
    }
    p = re_world_body(&g->world, g->player);
    if (p == NULL) {
        return;
    }
    if (p->position.y < GAME_KILL_Y) {
        g->deaths++;
        respawn_player(g);
        return;
    }

    /* Keep the capsule upright and awake while the character is played. */
    p->orientation = re_quat_from_axis_angle(re_vec3_make(0.0f, 0.0f, 1.0f),
                                             GAME_HALF_PI);
    p->ang_vel = re_vec3_zero();
    p->position.z = 0.0f;
    p->lin_vel.z = 0.0f;
    p->active = true;
    p->sleep_timer = 0u;

    apply_control(g, p, in, dt);
    g->pre_step_vy = p->lin_vel.y;
    re_world_step(&g->world, dt);

    p = re_world_body(&g->world, g->player);
    if (p == NULL) {
        return;
    }
    p->position.z = 0.0f;
    p->lin_vel.z = 0.0f;
    p->lin_vel.y = re_max_f32(p->lin_vel.y, -GAME_MAX_FALL);

    g->grounded = player_is_grounded(g);
    if (g->grounded) {
        g->coyote = GAME_COYOTE_TIME;
    } else {
        g->coyote = re_max_f32(0.0f, g->coyote - dt);
    }
    g->jump_buffer = re_max_f32(0.0f, g->jump_buffer - dt);

    /* Animation state: landing impact, air time and facing direction. */
    if (g->grounded && !g->was_grounded) {
        float impact = -g->pre_step_vy;
        g->land_timer = impact > 8.5f ? 0.30f : 0.12f;
        if (impact > 8.5f && fabsf(p->lin_vel.x) > 2.0f) {
            g->roll_timer = 0.36f;
        }
    }
    if (g->grounded) {
        g->air_time = 0.0f;
        g->run_distance += fabsf(p->lin_vel.x) * dt;
    } else {
        g->air_time += dt;
    }
    if (fabsf(p->lin_vel.x) > 0.25f) {
        g->facing = p->lin_vel.x >= 0.0f ? 1.0f : -1.0f;
    }
    g->land_timer = re_max_f32(0.0f, g->land_timer - dt);
    g->roll_timer = re_max_f32(0.0f, g->roll_timer - dt);
    g->was_grounded = g->grounded;

    collect_coins(g, p);
    goal_dx = p->position.x - g->goal_x;
    goal_dy = p->position.y - g->goal_y;
    if (!g->won &&
        goal_dx * goal_dx + goal_dy * goal_dy <= GAME_GOAL_RADIUS * GAME_GOAL_RADIUS) {
        g->won = true;
    }
    follow_camera(g, p, dt);
    g->time += dt;
}

float game_time(const Game *g)
{
    return g != NULL ? g->time : 0.0f;
}

/**
 * @brief Probes the level surface a little ahead of the player.
 * @param g Game state.
 * @param x World X to probe.
 * @param feet_y Player's feet height.
 * @param step_up Receives true when a higher platform blocks the way.
 * @return True when a platform supports the player at x.
 */
static bool probe_ground(const Game *g, float x, float feet_y, bool *step_up)
{
    bool support = false;
    uint32_t i;

    *step_up = false;
    for (i = 0u; i < g->platform_count; i++) {
        const GamePlatform *p = &g->platforms[i];
        float top;

        if (x < p->cx - p->hx || x > p->cx + p->hx) {
            continue;
        }
        top = p->cy + p->hy;
        if (top <= feet_y + 0.20f && top >= feet_y - 2.0f) {
            support = true;
        } else if (top > feet_y + 0.20f && top <= feet_y + 1.9f) {
            *step_up = true;
        }
    }
    return support;
}

void game_autopilot(const Game *g, GameInput *out)
{
    const ReBody *p;
    float feet_y;
    bool step_up = false;
    bool support;

    if (g == NULL || out == NULL || g->player >= g->world.body_count) {
        return;
    }
    p = &g->world.bodies[g->player];
    out->jump_pressed = false;
    out->jump_held = true;
    if (g->won) {
        out->move = 0.0f;
        return;
    }
    out->move = 1.0f;

    feet_y = p->position.y - GAME_PLAYER_HALF_HEIGHT;
    support = probe_ground(g, p->position.x + 2.2f, feet_y, &step_up);
    if (g->grounded && (!support || step_up)) {
        out->jump_pressed = true;
    }
}
