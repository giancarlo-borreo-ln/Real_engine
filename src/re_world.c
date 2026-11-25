/**
 * @file re_world.c
 * @brief Deterministic rigid-body world: force integration, broadphase
 *        refit, narrowphase and solver orchestration.
 *
 * Storage model: every column the world needs is carved out of one arena
 * in re_world_init. ReWorld has no scratch field, so two auxiliary
 * buffers live in an invisible tail of the pair allocation and are
 * reached by byte offset from w->pairs:
 *   - radix sort scratch: one uint64_t per pair slot;
 *   - solver scratch: four floats per contact slot.
 * The two do not overlap in time (sorting happens before narrowphase,
 * the solver runs after it), so a single tail of
 * max(pair_cap * 8, contact_cap * 16) bytes serves both. re_solver.c
 * uses the same helper; the offsets must stay in sync.
 *
 * Determinism: pair keys are sorted, contacts are emitted in key order
 * and the solver iterates in contact order, so the tick never depends on
 * tree shape, hash order or scheduling.
 */
#include "re_world.h"

#include <math.h>
#include <string.h>

#include "re_solver.h"

/** Damping ceiling (lin/ang velocity is scaled by 1 - damping per tick). */
#define RE_WORLD_MAX_DAMPING 0.999f

/** Narrowphase/contact capacities below this are raised to 1. */
#define RE_WORLD_MIN_CAP 1u

/** BVH needs at least a leaf and a root-candidate slot. */
#define RE_WORLD_MIN_BVH_CAP 2u

/* One-time log latches. Process-wide, log-only: they never affect the
 * simulation state, only how many times a truncation warning is printed. */
static bool g_warned_pairs;
static bool g_warned_contacts;

/* ------------------------------ helpers ----------------------------- */

/** @return v when it is finite, fallback otherwise (NaN-safe sanitizer). */
static float sanitize(float v, float fallback)
{
    return isfinite(v) ? v : fallback;
}

/** @return v clamped to [0, +inf), mapping NaN to 0. */
static float sanitize_nonneg(float v)
{
    if (!(v >= 0.0f)) {
        return 0.0f;
    }
    return v;
}

/**
 * @brief Start of the auxiliary scratch tail of the pair allocation.
 *
 * @param w World.
 * @return Pointer to max(pair_cap * 8, contact_cap * 16) usable bytes.
 */
static void *scratch_base(const ReWorld *w)
{
    unsigned char *base = (unsigned char *)(void *)w->pairs;
    return (void *)(base + (size_t)w->pair_cap * sizeof(uint64_t));
}

/** @return World transform of a body (rotation + translation, no scale). */
static ReMat4 body_transform(const ReBody *b)
{
    ReMat4 m = re_mat4_from_quat(b->orientation);
    m.col[3].x = b->position.x;
    m.col[3].y = b->position.y;
    m.col[3].z = b->position.z;
    m.col[3].w = 1.0f;
    return m;
}

/** @return Tight world-space AABB of the body's shape (no broadphase margin). */
static ReAabb body_aabb(const ReBody *b)
{
    switch (b->kind) {
    case RE_SHAPE_SPHERE:
        return re_aabb_expand(re_aabb_from_point(b->position), b->radius);
    case RE_SHAPE_BOX: {
        ReAabb local = {re_vec3_neg(b->half), b->half};
        return re_aabb_transform(local, body_transform(b));
    }
    case RE_SHAPE_CAPSULE: {
        ReVec3 axis = re_quat_rotate_vec3(b->orientation,
                                          re_vec3_make(b->half.x, 0.0f, 0.0f));
        ReVec3 p0 = re_vec3_sub(b->position, axis);
        ReVec3 p1 = re_vec3_add(b->position, axis);
        return re_aabb_expand(re_aabb_from_two_points(p0, p1), b->radius);
    }
    case RE_SHAPE_HULL: {
        if (b->hull_points == NULL || b->hull_count == 0u) {
            return re_aabb_from_point(b->position);
        }
        ReAabb local = re_aabb_from_point(b->hull_points[0]);
        for (uint32_t i = 1u; i < b->hull_count; i++) {
            local = re_aabb_merge(local, re_aabb_from_point(b->hull_points[i]));
        }
        return re_aabb_transform(local, body_transform(b));
    }
    }
    return re_aabb_from_point(b->position);
}

/** @return Stack shape view of a body (no ownership). */
static ReConvexShape shape_of(const ReBody *b)
{
    ReConvexShape s;
    s.kind = b->kind;
    s.p = b->position;
    s.q = b->orientation;
    s.radius = b->radius;
    s.half = b->half;
    s.points = b->hull_points;
    s.point_count = b->hull_count;
    return s;
}

/**
 * @brief Stable LSD radix sort over 64-bit keys, 8 passes of 8 bits.
 *
 * @param v Keys (sorted in place).
 * @param tmp Scratch with room for n keys.
 * @param n Key count.
 */
static void radix_sort_u64(uint64_t *v, uint64_t *tmp, uint32_t n)
{
    if (n < 2u) {
        return;
    }
    uint64_t *src = v;
    uint64_t *dst = tmp;
    for (unsigned shift = 0u; shift < 64u; shift += 8u) {
        uint32_t count[256] = {0};
        uint32_t offset[256];
        for (uint32_t i = 0u; i < n; i++) {
            count[(size_t)((src[i] >> shift) & UINT64_C(0xFF))]++;
        }
        uint32_t sum = 0u;
        for (size_t b = 0u; b < 256u; b++) {
            offset[b] = sum;
            sum += count[b];
        }
        for (uint32_t i = 0u; i < n; i++) {
            size_t b = (size_t)((src[i] >> shift) & UINT64_C(0xFF));
            dst[offset[b]++] = src[i];
        }
        uint64_t *swap = src;
        src = dst;
        dst = swap;
    }
    /* Eight (even) passes leave the sorted keys back in v. */
    RE_ASSERT(src == v);
}

/* ------------------------------- init ------------------------------- */

/**
 * @brief Clamps a configuration into the supported ranges.
 * @param cfg Caller configuration (NULL selects RE_WORLD_CONFIG_DEFAULT).
 * @return Sanitized copy.
 */
static ReWorldConfig sanitize_config(const ReWorldConfig *cfg)
{
    ReWorldConfig c = (cfg != NULL) ? *cfg : (ReWorldConfig)RE_WORLD_CONFIG_DEFAULT;
    c.max_bodies = re_max_u32(c.max_bodies, RE_WORLD_MIN_CAP);
    c.max_contacts = re_max_u32(c.max_contacts, RE_WORLD_MIN_CAP);
    c.max_pairs = re_max_u32(c.max_pairs, RE_WORLD_MIN_CAP);
    c.bvh_capacity = re_max_u32(c.bvh_capacity, RE_WORLD_MIN_BVH_CAP);
    c.solver_iterations = re_max_u32(c.solver_iterations, 1u);
    c.sleep_time = re_max_u32(c.sleep_time, 1u);
    c.broadphase_margin = sanitize_nonneg(c.broadphase_margin);
    c.sleep_lin_vel = sanitize_nonneg(c.sleep_lin_vel);
    c.sleep_ang_vel = sanitize_nonneg(c.sleep_ang_vel);
    c.damping = re_clamp_f32(sanitize_nonneg(c.damping), 0.0f, RE_WORLD_MAX_DAMPING);
    c.gravity_y = sanitize(c.gravity_y, 0.0f);
    return c;
}

/** @brief Carves the body/tree/contact/pair/warm columns out of the arena. */
static bool allocate_storage(ReWorld *w)
{
    const ReWorldConfig *c = &w->cfg;
    size_t pair_bytes = (size_t)c->max_pairs * sizeof(uint64_t);
    size_t solver_bytes = (size_t)c->max_contacts * 4u * sizeof(float);
    size_t scratch_bytes = pair_bytes > solver_bytes ? pair_bytes : solver_bytes;

    w->bodies = re_arena_alloc(&w->arena, (size_t)c->max_bodies * sizeof(ReBody));
    w->bvh_nodes = re_arena_alloc(&w->arena, (size_t)c->bvh_capacity * sizeof(ReBvhNode));
    w->bvh_free = re_arena_alloc(&w->arena, (size_t)c->bvh_capacity * sizeof(uint32_t));
    w->contacts = re_arena_alloc(&w->arena, (size_t)c->max_contacts * sizeof(ReContact));
    w->pairs = re_arena_alloc(&w->arena, pair_bytes + re_align_up(scratch_bytes, 16u));
    w->warm = re_arena_alloc(&w->arena, (size_t)c->max_pairs * sizeof(ReWarmEntry));
    return w->bodies != NULL && w->bvh_nodes != NULL && w->bvh_free != NULL &&
           w->contacts != NULL && w->pairs != NULL && w->warm != NULL;
}

/** @brief Initializes a world over caller-provided memory. */
bool re_world_init(ReWorld *w, const ReWorldConfig *cfg, void *mem, size_t mem_size)
{
    if (w == NULL) {
        return false;
    }
    ReWorldConfig c = sanitize_config(cfg);
    memset(w, 0, sizeof *w);
    w->cfg = c;
    w->body_cap = c.max_bodies;
    w->contact_cap = c.max_contacts;
    w->pair_cap = c.max_pairs;
    w->warm_cap = c.max_pairs;
    if (!re_arena_init(&w->arena, mem, mem_size)) {
        return false;
    }
    if (!allocate_storage(w)) {
        return false;
    }
    re_bvh_init(&w->bvh, w->bvh_nodes, w->bvh_free, c.bvh_capacity, c.broadphase_margin);
    return true;
}

/* ------------------------------ bodies ------------------------------ */

/** @brief Adds a body (copies init into the array). */
uint32_t re_world_add_body(ReWorld *w, const ReBody *init)
{
    if (w == NULL || init == NULL || w->body_count >= w->body_cap) {
        return RE_BODY_NULL;
    }
    uint32_t id = w->body_count;
    ReBody *b = &w->bodies[id];
    *b = *init;
    b->id = id;
    b->bvh_node = RE_BVH_NULL;
    b->aabb = re_aabb_empty();
    int32_t leaf = re_bvh_insert(&w->bvh, body_aabb(b), id);
    if (leaf == RE_BVH_NULL) {
        return RE_BODY_NULL;
    }
    b->bvh_node = leaf;
    b->aabb = w->bvh.nodes[leaf].aabb;
    w->body_count = id + 1u;
    return id;
}

/** @brief Removes a body; the last body swaps into the hole. */
void re_world_remove_body(ReWorld *w, uint32_t id)
{
    if (w == NULL || id >= w->body_count) {
        return;
    }
    ReBody *victim = &w->bodies[id];
    if (victim->bvh_node != RE_BVH_NULL) {
        re_bvh_remove(&w->bvh, victim->bvh_node);
        victim->bvh_node = RE_BVH_NULL;
    }
    uint32_t last = w->body_count - 1u;
    if (id != last) {
        w->bodies[id] = w->bodies[last];
        w->bodies[id].id = id;
        int32_t node = w->bodies[id].bvh_node;
        if (node != RE_BVH_NULL) {
            w->bvh.nodes[node].proxy = id;
        }
    }
    w->body_count = last;
}

/** @return Body pointer, or NULL for an invalid id. */
ReBody *re_world_body(ReWorld *w, uint32_t id)
{
    if (w == NULL || id >= w->body_count) {
        return NULL;
    }
    return &w->bodies[id];
}

/* ------------------------------ pipeline ---------------------------- */

/** @brief Integrates forces, torque and per-tick damping; clears accumulators. */
static void integrate_forces(ReWorld *w, float dt)
{
    ReVec3 gravity = re_vec3_make(0.0f, w->cfg.gravity_y, 0.0f);
    float damp = 1.0f - w->cfg.damping;
    for (uint32_t i = 0u; i < w->body_count; i++) {
        ReBody *b = &w->bodies[i];
        if (b->inv_mass > 0.0f && b->active) {
            ReVec3 accel = re_vec3_add(re_vec3_scale(b->force, b->inv_mass), gravity);
            b->lin_vel = re_vec3_add(b->lin_vel, re_vec3_scale(accel, dt));
            ReVec3 local = re_quat_rotate_vec3(re_quat_conjugate(b->orientation), b->torque);
            local = re_vec3_mul(local, b->inv_inertia);
            ReVec3 world_torque = re_quat_rotate_vec3(b->orientation, local);
            b->ang_vel = re_vec3_add(b->ang_vel, re_vec3_scale(world_torque, dt));
            b->lin_vel = re_vec3_scale(b->lin_vel, damp);
            b->ang_vel = re_vec3_scale(b->ang_vel, damp);
        }
        b->force = re_vec3_zero();
        b->torque = re_vec3_zero();
    }
}

/** @brief Refits every awake leaf; the cached AABB mirrors the tree's fat box. */
static void refit_broadphase(ReWorld *w)
{
    for (uint32_t i = 0u; i < w->body_count; i++) {
        ReBody *b = &w->bodies[i];
        if (!b->active || b->bvh_node == RE_BVH_NULL) {
            continue;
        }
        re_bvh_update(&w->bvh, b->bvh_node, body_aabb(b));
        b->aabb = w->bvh.nodes[b->bvh_node].aabb;
    }
}

/** @brief Collects and sorts this tick's broadphase pairs (truncating on overflow). */
static void collect_pairs(ReWorld *w)
{
    uint32_t total = re_bvh_query_pairs(&w->bvh, w->pairs, w->pair_cap);
    if (total > w->pair_cap) {
        if (!g_warned_pairs) {
            re_log_warn("re_world: broadphase pairs truncated (%u > %u cap)", total, w->pair_cap);
            g_warned_pairs = true;
        }
        total = w->pair_cap;
    }
    w->pair_count = total;
    radix_sort_u64(w->pairs, (uint64_t *)scratch_base(w), total);
}

/**
 * @brief Appends one contact, truncating (with a warning) when full.
 *
 * @param w World.
 * @param a First body.
 * @param b Second body.
 * @param point Anchor (world).
 * @param normal Unit normal from B to A.
 * @param depth Penetration depth.
 */
static void push_contact(ReWorld *w, const ReBody *a, const ReBody *b, ReVec3 point,
                         ReVec3 normal, float depth)
{
    if (w->contact_count >= w->contact_cap) {
        if (!g_warned_contacts) {
            re_log_warn("re_world: contacts truncated at capacity %u", w->contact_cap);
            g_warned_contacts = true;
        }
        return;
    }
    ReContact *c = &w->contacts[w->contact_count];
    c->point = point;
    c->normal = normal;
    c->r_a = re_vec3_sub(point, a->position);
    c->r_b = re_vec3_sub(point, b->position);
    c->depth = sanitize_nonneg(depth);
    c->friction = sqrtf(sanitize_nonneg(a->friction) * sanitize_nonneg(b->friction));
    c->restitution = re_clamp_f32(sanitize_nonneg(a->restitution), 0.0f, 1.0f);
    c->restitution = re_max_f32(c->restitution,
                                re_clamp_f32(sanitize_nonneg(b->restitution), 0.0f, 1.0f));
    c->a = a->id;
    c->b = b->id;
    w->contact_count++;
}

/**
 * @brief Analytic sphere/sphere overlap test.
 * @return True on overlap; fills point, unit normal (B to A) and depth.
 */
static bool sphere_sphere(const ReBody *a, const ReBody *b, ReVec3 *point, ReVec3 *normal,
                          float *depth)
{
    ReVec3 delta = re_vec3_sub(a->position, b->position);
    float dist2 = re_vec3_length_sq(delta);
    float radius_sum = a->radius + b->radius;
    if (!(dist2 < radius_sum * radius_sum)) {
        return false;
    }
    float dist = sqrtf(dist2);
    *normal = dist > 1e-6f ? re_vec3_scale(delta, 1.0f / dist) : re_vec3_make(1.0f, 0.0f, 0.0f);
    *point = re_vec3_scale(re_vec3_add(a->position, b->position), 0.5f);
    *depth = radius_sum - dist;
    return true;
}

/** @brief Narrowphases one pair and emits at most one contact. */
static void narrowphase_pair(ReWorld *w, uint32_t ia, uint32_t ib)
{
    const ReBody *a = &w->bodies[ia];
    const ReBody *b = &w->bodies[ib];
    if (!a->active && !b->active) {
        return;
    }
    if (a->kind == RE_SHAPE_SPHERE && b->kind == RE_SHAPE_SPHERE) {
        ReVec3 point;
        ReVec3 normal;
        float depth = 0.0f;
        if (sphere_sphere(a, b, &point, &normal, &depth)) {
            push_contact(w, a, b, point, normal, depth);
        }
        return;
    }
    ReConvexShape sa = shape_of(a);
    ReConvexShape sb = shape_of(b);
    ReEpaResult epa = re_gjk_penetration(&sa, &sb);
    if (!epa.ok) {
        return;
    }
    ReVec3 normal = re_vec3_normalize(epa.normal);
    if (re_vec3_length_sq(normal) < 0.25f) {
        normal = re_vec3_make(1.0f, 0.0f, 0.0f);
    }
    ReVec3 point = re_vec3_scale(re_vec3_add(epa.point_a, epa.point_b), 0.5f);
    push_contact(w, a, b, point, normal, epa.depth);
}

/** @brief Turns sorted broadphase pairs into contacts. */
static void narrowphase(ReWorld *w)
{
    for (uint32_t i = 0u; i < w->pair_count; i++) {
        uint64_t pair = w->pairs[i];
        uint32_t lo = (uint32_t)(pair & UINT64_C(0xFFFFFFFF));
        uint32_t hi = (uint32_t)(pair >> 32u);
        if (lo == hi || hi >= w->body_count) {
            continue;
        }
        narrowphase_pair(w, lo, hi);
    }
}

/** @brief Integrates velocities into poses (semi-implicit Euler + quat derivative). */
static void integrate_velocities(ReWorld *w, float dt)
{
    float half_dt = 0.5f * dt;
    for (uint32_t i = 0u; i < w->body_count; i++) {
        ReBody *b = &w->bodies[i];
        if (b->inv_mass <= 0.0f || !b->active) {
            continue;
        }
        b->position = re_vec3_add_scaled(b->position, b->lin_vel, dt);
        ReQuat spin = {b->ang_vel.x, b->ang_vel.y, b->ang_vel.z, 0.0f};
        ReQuat delta = re_quat_mul(spin, b->orientation);
        ReQuat q = b->orientation;
        q.x += half_dt * delta.x;
        q.y += half_dt * delta.y;
        q.z += half_dt * delta.z;
        q.w += half_dt * delta.w;
        b->orientation = re_quat_normalize(q);
    }
}

/** @brief Sleep management: counts slow ticks, clears active when saturated. */
static void update_sleep(ReWorld *w)
{
    float lin2 = w->cfg.sleep_lin_vel * w->cfg.sleep_lin_vel;
    float ang2 = w->cfg.sleep_ang_vel * w->cfg.sleep_ang_vel;
    for (uint32_t i = 0u; i < w->body_count; i++) {
        ReBody *b = &w->bodies[i];
        bool slow = re_vec3_length_sq(b->lin_vel) <= lin2 &&
                    re_vec3_length_sq(b->ang_vel) <= ang2;
        if (slow) {
            if (b->sleep_timer < UINT32_MAX) {
                b->sleep_timer++;
            }
        } else {
            b->sleep_timer = 0u;
        }
        b->active = b->inv_mass > 0.0f && b->sleep_timer < w->cfg.sleep_time;
    }
}

/** @brief Advances the simulation by one fixed timestep. */
void re_world_step(ReWorld *w, float dt)
{
    if (w == NULL || !(dt > 0.0f) || !isfinite(dt)) {
        return;
    }
    w->contact_count = 0u;
    integrate_forces(w, dt);
    refit_broadphase(w);
    collect_pairs(w);
    narrowphase(w);
    re_solver_solve(w, w->contacts, w->contact_count, w->cfg.solver_iterations);
    /* Wake before the sleep manager: waking resets sleep_timer, and the
     * sleep manager must still be able to put a settled body back to
     * sleep in the same tick (otherwise nothing ever sleeps). */
    re_solver_wake_bodies(w, w->contacts, w->contact_count);
    integrate_velocities(w, dt);
    update_sleep(w);
    w->time += dt;
}

/* ------------------------------- query ------------------------------ */

/** @brief Ray-cast traversal state. */
typedef struct RayContext {
    const ReWorld *world; /**< World being queried. */
    ReRay ray;            /**< Ray. */
    float t_max;          /**< Maximum distance. */
    ReRayHit *out;        /**< Hit buffer. */
    uint32_t cap;         /**< Hit buffer capacity. */
    uint32_t count;       /**< Hits written. */
} RayContext;

/** @brief Broadphase leaf callback: shape-tests the body and records the hit. */
static bool ray_leaf_cb(uint32_t proxy, float t, void *ud)
{
    (void)t;
    RayContext *ctx = (RayContext *)ud;
    if (proxy >= ctx->world->body_count) {
        return false;
    }
    if (ctx->count >= ctx->cap) {
        return true;
    }
    ReConvexShape s = shape_of(&ctx->world->bodies[proxy]);
    float t_hit = 0.0f;
    ReVec3 normal = re_vec3_zero();
    if (!re_shape_ray(&s, ctx->ray, ctx->t_max, &t_hit, &normal)) {
        return false;
    }
    ReRayHit *hit = &ctx->out[ctx->count];
    hit->body = proxy;
    hit->t = t_hit;
    hit->point = re_ray_point_at(ctx->ray, t_hit);
    hit->normal = normal;
    ctx->count++;
    return false;
}

/** @brief Insertion sort of hits by entry distance. */
static void sort_hits(ReRayHit *hits, uint32_t n)
{
    for (uint32_t i = 1u; i < n; i++) {
        ReRayHit key = hits[i];
        uint32_t j = i;
        while (j > 0u && hits[j - 1u].t > key.t) {
            hits[j] = hits[j - 1u];
            j--;
        }
        hits[j] = key;
    }
}

/** @brief Ray-casts the world; hits are sorted by distance. */
uint32_t re_world_raycast(const ReWorld *w, ReRay ray, float t_max, ReRayHit *out, uint32_t cap)
{
    if (w == NULL || out == NULL || cap == 0u) {
        return 0u;
    }
    RayContext ctx = {w, ray, t_max, out, cap, 0u};
    (void)re_bvh_raycast(&w->bvh, ray, t_max, ray_leaf_cb, &ctx);
    sort_hits(out, ctx.count);
    return ctx.count;
}

/** @brief Fills stats with the last tick's counters. */
void re_world_stats(const ReWorld *w, ReWorldStats *stats)
{
    if (w == NULL || stats == NULL) {
        return;
    }
    stats->broadphase_pairs = w->pair_count;
    stats->contacts = w->contact_count;
    uint32_t awake = 0u;
    for (uint32_t i = 0u; i < w->body_count; i++) {
        if (w->bodies[i].active) {
            awake++;
        }
    }
    stats->awake_bodies = awake;
}

/** @return Number of live bodies. */
uint32_t re_world_body_count(const ReWorld *w)
{
    return w == NULL ? 0u : w->body_count;
}

/** @brief True when two bodies overlap right now (GJK distance == 0). */
bool re_world_overlap(const ReWorld *w, uint32_t a, uint32_t b)
{
    if (w == NULL || a >= w->body_count || b >= w->body_count) {
        return false;
    }
    ReConvexShape sa = shape_of(&w->bodies[a]);
    ReConvexShape sb = shape_of(&w->bodies[b]);
    return re_gjk_distance(&sa, &sb).hit;
}
