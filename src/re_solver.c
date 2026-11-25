/**
 * @file re_solver.c
 * @brief Sequential impulse solver with warm starting, Coulomb friction
 *        and a direct positional (NGS-lite) penetration push.
 *
 * Two deliberate design choices, both forced by the fixed public API:
 *
 * 1. No timestep reaches re_solver_solve, so Baumgarte velocity bias is
 *    impossible to compute here. Instead the solver runs a pure
 *    velocity-level PGS pass and then one direct position-correction
 *    pass (correction = 0.2 * (depth - slop) split by inverse mass).
 *    This is a Gauss-Seidel-free positional push: deterministic, stable
 *    for resting stacks, and it converges to a residual depth of slop.
 *
 * 2. The friction basis is derived from the contact normal alone rather
 *    than from the current relative velocity. A normal-derived basis is
 *    identical every tick, so the cached warm-start impulses stay valid;
 *    a velocity-derived basis would silently rotate under the stored
 *    scalars. The clamped per-axis formulation is the usual box
 *    approximation of the friction cone.
 *
 * Per-contact scratch (four floats each) is carved from the tail of the
 * world's pair allocation; see the note in re_world.c. The broadphase
 * pair array is dead by the time the solver runs.
 */
#include "re_solver.h"

#include <math.h>

/** Penetration tolerated before positional correction kicks in (metres). */
#define RE_SOLVER_SLOP 0.005f

/** Fraction of the excess penetration removed per tick. */
#define RE_SOLVER_CORRECTION 0.2f

/** Hard cap on the positional correction applied to one contact per tick. */
#define RE_SOLVER_MAX_CORRECTION 0.25f

/** Relative normal speed below which restitution is ignored (m/s). */
#define RE_SOLVER_RESTITUTION_VELOCITY 1.0f

/** Correction magnitude that counts as "the body was pushed" (metres). */
#define RE_SOLVER_WAKE_PUSH 0.001f

/** Contacts shallower than this never wake a sleeping body (metres). */
#define RE_SOLVER_WAKE_DEPTH 0.001f

/** @brief Per-contact accumulators carved from the pair allocation tail. */
typedef struct SolverScratch {
    float *j_n;  /**< Accumulated normal impulse. */
    float *j_t1; /**< Accumulated tangent-1 impulse. */
    float *j_t2; /**< Accumulated tangent-2 impulse. */
    float *target; /**< Restitution target normal velocity. */
} SolverScratch;

/* ------------------------------ helpers ----------------------------- */

/** @return Packed cache key for the ordered pair (a, b); ids must be sorted. */
static uint64_t pair_key(uint32_t a, uint32_t b)
{
    return ((uint64_t)a << 32u) | (uint64_t)b;
}

/**
 * @brief Start of the auxiliary scratch tail of the pair allocation.
 *
 * Must match the layout produced by re_world_init (re_world.c).
 * @param w World.
 * @return Scratch base.
 */
static void *scratch_base(const ReWorld *w)
{
    unsigned char *base = (unsigned char *)(void *)w->pairs;
    return (void *)(base + (size_t)w->pair_cap * sizeof(uint64_t));
}

/** @return Scratch view for a world. */
static SolverScratch solver_scratch(const ReWorld *w)
{
    float *base = (float *)scratch_base(w);
    size_t cap = (size_t)w->contact_cap;
    SolverScratch s;
    s.j_n = base;
    s.j_t1 = base + cap;
    s.j_t2 = base + 2u * cap;
    s.target = base + 3u * cap;
    return s;
}

/** @brief Binary search of the warm cache for key; fills the index on a hit. */
static bool warm_find(const ReWorld *w, uint64_t key, uint32_t *out_index)
{
    uint32_t lo = 0u;
    uint32_t hi = w->warm_count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2u;
        if (w->warm[mid].pair < key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo < w->warm_count && w->warm[lo].pair == key) {
        *out_index = lo;
        return true;
    }
    return false;
}

/** @brief Deterministic tangent basis for a unit normal. */
static void tangent_basis(ReVec3 n, ReVec3 *t1, ReVec3 *t2)
{
    ReVec3 axis = fabsf(n.x) < 0.9f ? re_vec3_make(1.0f, 0.0f, 0.0f)
                                    : re_vec3_make(0.0f, 1.0f, 0.0f);
    ReVec3 a = re_vec3_cross(n, axis);
    if (re_vec3_length_sq(a) < 1e-8f) {
        a = re_vec3_cross(n, re_vec3_make(0.0f, 0.0f, 1.0f));
    }
    *t1 = re_vec3_normalize(a);
    *t2 = re_vec3_cross(n, *t1);
}

/** @return R * diag(inv_inertia) * R^T * v (world-space inverse inertia). */
static ReVec3 world_inv_inertia(const ReBody *b, ReVec3 v)
{
    ReVec3 local = re_quat_rotate_vec3(re_quat_conjugate(b->orientation), v);
    local = re_vec3_mul(local, b->inv_inertia);
    return re_quat_rotate_vec3(b->orientation, local);
}

/** @return Relative velocity of the anchors of A and B at the contact. */
static ReVec3 relative_velocity(const ReBody *a, const ReBody *b, ReVec3 r_a, ReVec3 r_b)
{
    ReVec3 va = re_vec3_add(a->lin_vel, re_vec3_cross(a->ang_vel, r_a));
    ReVec3 vb = re_vec3_add(b->lin_vel, re_vec3_cross(b->ang_vel, r_b));
    return re_vec3_sub(va, vb);
}

/** @brief Applies impulse J (world) to both bodies, respecting infinite mass. */
static void apply_impulse(ReBody *a, ReBody *b, ReVec3 r_a, ReVec3 r_b, ReVec3 j)
{
    if (a->inv_mass > 0.0f) {
        a->lin_vel = re_vec3_add(a->lin_vel, re_vec3_scale(j, a->inv_mass));
        a->ang_vel = re_vec3_add(a->ang_vel, world_inv_inertia(a, re_vec3_cross(r_a, j)));
    }
    if (b->inv_mass > 0.0f) {
        b->lin_vel = re_vec3_sub(b->lin_vel, re_vec3_scale(j, b->inv_mass));
        b->ang_vel = re_vec3_sub(b->ang_vel, world_inv_inertia(b, re_vec3_cross(r_b, j)));
    }
}

/** @return Effective mass along dir: 1 / (J . k^-1 J). */
static float effective_mass(const ReBody *a, const ReBody *b, ReVec3 r_a, ReVec3 r_b, ReVec3 dir)
{
    float k = a->inv_mass + b->inv_mass;
    if (a->inv_mass > 0.0f) {
        ReVec3 ra = re_vec3_cross(r_a, dir);
        k += re_vec3_dot(ra, world_inv_inertia(a, ra));
    }
    if (b->inv_mass > 0.0f) {
        ReVec3 rb = re_vec3_cross(r_b, dir);
        k += re_vec3_dot(rb, world_inv_inertia(b, rb));
    }
    return k;
}

/** @return True when at least one body of the contact is dynamic. */
static bool pair_is_dynamic(const ReBody *a, const ReBody *b)
{
    return a->inv_mass > 0.0f || b->inv_mass > 0.0f;
}

/* ---------------------------- warm start ---------------------------- */

/** @brief Applies the previous tick's impulses recovered from the cache. */
static void warm_start(ReWorld *w, const ReContact *contacts, uint32_t count,
                       const SolverScratch *s)
{
    for (uint32_t i = 0u; i < count; i++) {
        const ReContact *c = &contacts[i];
        /* Direct callers may hand us arbitrary contact records: validate
           the body ids like every other pass before dereferencing. */
        if (c->a >= w->body_count || c->b >= w->body_count) {
            continue;
        }
        uint32_t index = 0u;
        uint64_t key = pair_key(c->a, c->b);
        if (!warm_find(w, key, &index)) {
            continue;
        }
        ReBody *a = &w->bodies[c->a];
        ReBody *b = &w->bodies[c->b];
        if (!pair_is_dynamic(a, b)) {
            continue;
        }
        const ReWarmEntry *e = &w->warm[index];
        ReVec3 n = re_vec3_normalize(c->normal);
        ReVec3 t1;
        ReVec3 t2;
        tangent_basis(n, &t1, &t2);
        float limit = c->friction * e->n_impulse;
        s->j_n[i] = e->n_impulse;
        s->j_t1[i] = re_clamp_f32(e->t1_impulse, -limit, limit);
        s->j_t2[i] = re_clamp_f32(e->t2_impulse, -limit, limit);
        ReVec3 j = re_vec3_scale(n, s->j_n[i]);
        j = re_vec3_add(j, re_vec3_scale(t1, s->j_t1[i]));
        j = re_vec3_add(j, re_vec3_scale(t2, s->j_t2[i]));
        apply_impulse(a, b, c->r_a, c->r_b, j);
    }
}

/** @brief Computes the restitution target velocity from the pre-solve state. */
static void compute_targets(const ReWorld *w, const ReContact *contacts, uint32_t count,
                            const SolverScratch *s)
{
    for (uint32_t i = 0u; i < count; i++) {
        const ReContact *c = &contacts[i];
        s->target[i] = 0.0f;
        if (c->restitution <= 0.0f || c->a >= w->body_count || c->b >= w->body_count) {
            continue;
        }
        ReVec3 n = re_vec3_normalize(c->normal);
        ReVec3 rv = relative_velocity(&w->bodies[c->a], &w->bodies[c->b], c->r_a, c->r_b);
        float vn = re_vec3_dot(rv, n);
        if (vn < -RE_SOLVER_RESTITUTION_VELOCITY) {
            s->target[i] = c->restitution * vn;
        }
    }
}

/* ------------------------------ solving ----------------------------- */

/** @brief One clamped normal impulse step. */
static void solve_normal(ReBody *a, ReBody *b, const ReContact *c, ReVec3 n, float target,
                         float *j_n)
{
    float k = effective_mass(a, b, c->r_a, c->r_b, n);
    if (!(k > 0.0f)) {
        return;
    }
    ReVec3 rv = relative_velocity(a, b, c->r_a, c->r_b);
    float vn = re_vec3_dot(rv, n);
    float next = *j_n + (target - vn) / k;
    if (next < 0.0f) {
        next = 0.0f;
    }
    float applied = next - *j_n;
    *j_n = next;
    if (applied != 0.0f) {
        apply_impulse(a, b, c->r_a, c->r_b, re_vec3_scale(n, applied));
    }
}

/** @brief Clamped friction impulse step along one tangent direction. */
static void solve_tangent(ReBody *a, ReBody *b, const ReContact *c, ReVec3 t, float limit,
                          float *j_t)
{
    float k = effective_mass(a, b, c->r_a, c->r_b, t);
    if (!(k > 0.0f)) {
        return;
    }
    ReVec3 rv = relative_velocity(a, b, c->r_a, c->r_b);
    float vt = re_vec3_dot(rv, t);
    float next = re_clamp_f32(*j_t - vt / k, -limit, limit);
    float applied = next - *j_t;
    *j_t = next;
    if (applied != 0.0f) {
        apply_impulse(a, b, c->r_a, c->r_b, re_vec3_scale(t, applied));
    }
}

/** @brief Solves one contact: normal then both friction tangents. */
static void solve_contact(ReWorld *w, const ReContact *c, const SolverScratch *s, uint32_t i)
{
    if (c->a >= w->body_count || c->b >= w->body_count) {
        return;
    }
    ReBody *a = &w->bodies[c->a];
    ReBody *b = &w->bodies[c->b];
    if (!pair_is_dynamic(a, b)) {
        return;
    }
    ReVec3 n = re_vec3_normalize(c->normal);
    solve_normal(a, b, c, n, s->target[i], &s->j_n[i]);
    if (c->friction <= 0.0f) {
        return;
    }
    ReVec3 t1;
    ReVec3 t2;
    tangent_basis(n, &t1, &t2);
    float limit = c->friction * s->j_n[i];
    solve_tangent(a, b, c, t1, limit, &s->j_t1[i]);
    solve_tangent(a, b, c, t2, limit, &s->j_t2[i]);
}

/** @brief One PGS sweep over every contact in deterministic order. */
static void solve_iteration(ReWorld *w, const ReContact *contacts, uint32_t count,
                            const SolverScratch *s)
{
    for (uint32_t i = 0u; i < count; i++) {
        solve_contact(w, &contacts[i], s, i);
    }
}

/** @brief Wakes a body that was visibly pushed by the position correction. */
static void wake_if_pushed(ReBody *b, float push)
{
    if (b->inv_mass > 0.0f && push > RE_SOLVER_WAKE_PUSH) {
        b->active = true;
        b->sleep_timer = 0u;
    }
}

/** @brief Pushes overlapping bodies apart (one pass, split by inverse mass). */
static void positional_correction(ReWorld *w, const ReContact *contacts, uint32_t count)
{
    for (uint32_t i = 0u; i < count; i++) {
        const ReContact *c = &contacts[i];
        if (c->a >= w->body_count || c->b >= w->body_count || c->depth <= RE_SOLVER_SLOP) {
            continue;
        }
        ReBody *a = &w->bodies[c->a];
        ReBody *b = &w->bodies[c->b];
        float k = a->inv_mass + b->inv_mass;
        if (!(k > 0.0f)) {
            continue;
        }
        ReVec3 n = re_vec3_normalize(c->normal);
        float corr = (c->depth - RE_SOLVER_SLOP) * RE_SOLVER_CORRECTION / k;
        corr = re_min_f32(corr, RE_SOLVER_MAX_CORRECTION);
        if (corr <= 0.0f) {
            continue;
        }
        a->position = re_vec3_add_scaled(a->position, n, corr * a->inv_mass);
        b->position = re_vec3_add_scaled(b->position, n, -corr * b->inv_mass);
        wake_if_pushed(a, corr * a->inv_mass);
        wake_if_pushed(b, corr * b->inv_mass);
    }
}

/* --------------------------- warm rebuild --------------------------- */

/** @brief Sift-down step of the warm-cache heapsort. */
static void warm_sift(ReWarmEntry *v, uint32_t root, uint32_t n)
{
    for (;;) {
        uint32_t child = 2u * root + 1u;
        if (child >= n) {
            return;
        }
        if (child + 1u < n && v[child].pair < v[child + 1u].pair) {
            child++;
        }
        if (v[root].pair >= v[child].pair) {
            return;
        }
        ReWarmEntry tmp = v[root];
        v[root] = v[child];
        v[child] = tmp;
        root = child;
    }
}

/** @brief In-place heapsort of warm entries by pair key (unsorted fallback). */
static void warm_sort(ReWarmEntry *v, uint32_t n)
{
    if (n < 2u) {
        return;
    }
    for (uint32_t i = n / 2u; i-- > 0u;) {
        warm_sift(v, i, n);
    }
    for (uint32_t end = n - 1u; end > 0u; end--) {
        ReWarmEntry tmp = v[0];
        v[0] = v[end];
        v[end] = tmp;
        warm_sift(v, 0u, end);
    }
}

/** @brief Merges adjacent equal keys, summing their impulses. @return New count. */
static uint32_t warm_merge(ReWarmEntry *v, uint32_t n)
{
    if (n < 2u) {
        return n;
    }
    uint32_t out = 1u;
    for (uint32_t i = 1u; i < n; i++) {
        if (v[i].pair == v[out - 1u].pair) {
            v[out - 1u].n_impulse += v[i].n_impulse;
            v[out - 1u].t1_impulse += v[i].t1_impulse;
            v[out - 1u].t2_impulse += v[i].t2_impulse;
        } else {
            v[out] = v[i];
            out++;
        }
    }
    return out;
}

/** @return True when the contact keys are strictly increasing. */
static bool contacts_ordered(const ReContact *contacts, uint32_t count)
{
    for (uint32_t i = 1u; i < count; i++) {
        uint64_t prev = pair_key(contacts[i - 1u].a, contacts[i - 1u].b);
        uint64_t cur = pair_key(contacts[i].a, contacts[i].b);
        if (cur <= prev) {
            return false;
        }
    }
    return true;
}

/** @brief Appends one warm entry, dropping it silently when the cache is full. */
static void warm_append(ReWorld *w, uint32_t *out, uint64_t key, const SolverScratch *s,
                        uint32_t i)
{
    if (*out >= w->warm_cap) {
        return;
    }
    ReWarmEntry *e = &w->warm[*out];
    e->pair = key;
    e->n_impulse = s->j_n[i];
    e->t1_impulse = s->j_t1[i];
    e->t2_impulse = s->j_t2[i];
    (*out)++;
}

/**
 * @brief Rebuilds the warm cache from this tick's contacts (pair key order).
 *
 * The world emits contacts in ascending key order, so the fast path is a
 * single append sweep. An unsorted contact array (a direct caller) falls
 * back to heapsort plus duplicate merging.
 */
static void warm_rebuild(ReWorld *w, const ReContact *contacts, uint32_t count,
                         const SolverScratch *s)
{
    uint32_t out = 0u;
    if (contacts_ordered(contacts, count)) {
        for (uint32_t i = 0u; i < count; i++) {
            warm_append(w, &out, pair_key(contacts[i].a, contacts[i].b), s, i);
        }
    } else {
        for (uint32_t i = 0u; i < count; i++) {
            warm_append(w, &out, pair_key(contacts[i].a, contacts[i].b), s, i);
        }
        warm_sort(w->warm, out);
        out = warm_merge(w->warm, out);
    }
    w->warm_count = out;
}

/* ------------------------------ public ------------------------------ */

/** @brief Solves one tick's contacts and maintains the warm-start cache. */
void re_solver_solve(ReWorld *w, ReContact *contacts, uint32_t contact_count, uint32_t iterations)
{
    if (w == NULL || contacts == NULL || w->contact_cap == 0u) {
        return;
    }
    /* The world guarantees contact_count <= contact_cap; a direct caller
     * may exceed it, in which case the surplus is ignored (documented). */
    uint32_t count = contact_count < w->contact_cap ? contact_count : w->contact_cap;
    if (count == 0u) {
        w->warm_count = 0u;
        return;
    }
    SolverScratch s = solver_scratch(w);
    for (uint32_t i = 0u; i < count; i++) {
        s.j_n[i] = 0.0f;
        s.j_t1[i] = 0.0f;
        s.j_t2[i] = 0.0f;
        s.target[i] = 0.0f;
    }
    compute_targets(w, contacts, count, &s);
    warm_start(w, contacts, count, &s);
    for (uint32_t it = 0u; it < iterations; it++) {
        solve_iteration(w, contacts, count, &s);
    }
    positional_correction(w, contacts, count);
    warm_rebuild(w, contacts, count, &s);
}

/** @return True when the body's speeds exceed the world's sleep thresholds. */
static bool body_is_moving(const ReBody *b, float lin2, float ang2)
{
    return re_vec3_length_sq(b->lin_vel) > lin2 || re_vec3_length_sq(b->ang_vel) > ang2;
}

/**
 * @brief Wakes bodies involved in contacts with enough impulse.
 *
 * Approximation: the normal impulse of the last tick is not part of the
 * contact record, so "enough impulse" is expressed as a deep contact
 * (depth > 1 mm) whose counterpart body is still moving above the sleep
 * thresholds, i.e. a genuinely moving intruder. Resting neighbours do not
 * wake each other, which keeps settled stacks asleep. The sleep manager
 * also wakes any body whose velocity rose above the thresholds, which
 * covers the slow-pusher case through the impulses written above.
 */
void re_solver_wake_bodies(ReWorld *w, const ReContact *contacts, uint32_t contact_count)
{
    if (w == NULL || contacts == NULL) {
        return;
    }
    float lin2 = w->cfg.sleep_lin_vel * w->cfg.sleep_lin_vel;
    float ang2 = w->cfg.sleep_ang_vel * w->cfg.sleep_ang_vel;
    for (uint32_t i = 0u; i < contact_count; i++) {
        const ReContact *c = &contacts[i];
        if (c->depth <= RE_SOLVER_WAKE_DEPTH || c->a >= w->body_count || c->b >= w->body_count) {
            continue;
        }
        ReBody *a = &w->bodies[c->a];
        ReBody *b = &w->bodies[c->b];
        bool moving_b = body_is_moving(b, lin2, ang2);
        bool moving_a = body_is_moving(a, lin2, ang2);
        if (!a->active && a->inv_mass > 0.0f && moving_b) {
            a->active = true;
            a->sleep_timer = 0u;
        }
        if (!b->active && b->inv_mass > 0.0f && moving_a) {
            b->active = true;
            b->sleep_timer = 0u;
        }
    }
}
