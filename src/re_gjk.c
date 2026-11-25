/**
 * @file re_gjk.c
 * @brief Narrowphase primitives: support mappings, GJK distance, EPA
 *        penetration, conservative advancement and ray casts.
 *
 * Termination contract: GJK runs at most RE_GJK_MAX_ITER iterations,
 * EPA at most RE_EPA_MAX_ITER, conservative advancement at most
 * RE_CCD_MAX_ITER and the hull ray cast at most RE_GJK_MAX_ITER. Every
 * pool write is guarded against its capacity, so no input can index
 * past the fixed arrays, and every working struct is zeroed before use
 * (deterministic, no uninitialised reads).
 *
 * Sign convention: ReEpaResult::normal points from B to A, i.e. the
 * direction A must move to separate. Witnesses satisfy
 * |point_a - point_b| == distance when separated and
 * point_a - point_b == -normal * depth when penetrating (the contact
 * points are crossed by the overlap).
 */
#include "re_gjk.h"

#include <math.h>
#include <string.h>

/** Squared distance below which the origin counts as inside the simplex. */
/* Distance GJK converges to a residual bounded by float precision on the
 * support functions (~1e-4 at game scale): declaring a hit below
 * dist^2 < 1e-6 (dist < 1e-3) is the standard engine-scale contact
 * threshold. Genuine separations >= 1e-3 still report hit == false. */
#define GJK_CONTACT_EPS2 1.0e-6f

/** Relative progress threshold used to stop the GJK loop. */
#define GJK_PROGRESS_EPS 1.0e-6f

/** EPA: growth below this means the closest face is converged. */
#define EPA_GROWTH_EPS 1.0e-6f

/** EPA: depths below this are reported as ok == false (mere touching). */
#define EPA_MIN_DEPTH 1.0e-6f

/** EPA: relative squared distance under which a support point is a duplicate. */
#define EPA_VERTEX_EPS2 1.0e-10f

/** Maximum number of horizon edges produced by one EPA expansion. */
#define EPA_HORIZON_MAX (3u * RE_EPA_MAX_FACES)

/** Distance below which a ray is considered to touch a hull. */
#define RAY_CONTACT_EPS 1.0e-5f

/** Number of bisection refinements for the hull ray cast. */
#define RAY_BISECT_STEPS 8

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

/** @return d when non-zero, else the deterministic fallback (1, 0, 0). */
static ReVec3 vec3_dir_or_axis(ReVec3 d)
{
    if (!(re_vec3_length_sq(d) > 0.0f)) {
        return re_vec3_make(1.0f, 0.0f, 0.0f);
    }
    return d;
}

/** @return -1 for negative input, +1 otherwise (sign(0) == +1, deterministic). */
static float sign_nonzero(float v)
{
    return (v < 0.0f) ? -1.0f : 1.0f;
}

/** @return v with a guaranteed non-negative length-like magnitude. */
static float magnitude(float v)
{
    return fabsf(v);
}

/** @return Unit-length quaternion of q (zero maps to identity). */
static ReQuat shape_quat(const ReConvexShape *s)
{
    return re_quat_normalize(s->q);
}

/** @return The conjugate q* = (-x, -y, -z, w), written out explicitly. */
static ReQuat quat_conjugate(ReQuat q)
{
    ReQuat c;
    c.x = -q.x;
    c.y = -q.y;
    c.z = -q.z;
    c.w = q.w;
    return c;
}

/** @return v rotated from world space into the shape's local frame. */
static ReVec3 shape_to_local(const ReConvexShape *s, ReVec3 v)
{
    return re_quat_rotate_vec3(quat_conjugate(shape_quat(s)), v);
}

/** @return v rotated from the shape's local frame into world space. */
static ReVec3 shape_to_world(const ReConvexShape *s, ReVec3 v)
{
    return re_quat_rotate_vec3(shape_quat(s), v);
}

/** @return Component i (0..2) of v; out-of-range indices read z. */
static float vec3_comp(ReVec3 v, int i)
{
    if (i == 0) {
        return v.x;
    }
    if (i == 1) {
        return v.y;
    }
    return v.z;
}

/** @return The unit axis i (0..2) scaled by s. */
static ReVec3 vec3_axis(int i, float s)
{
    return re_vec3_make((i == 0) ? s : 0.0f, (i == 1) ? s : 0.0f, (i == 2) ? s : 0.0f);
}

/** @return A copy of s translated by delta. */
static ReConvexShape shape_translated(ReConvexShape s, ReVec3 delta)
{
    s.p = re_vec3_add(s.p, delta);
    return s;
}

/** @return A point shape (zero-radius sphere) at p. */
static ReConvexShape shape_point(ReVec3 p)
{
    ReConvexShape s;
    memset(&s, 0, sizeof s);
    s.kind = RE_SHAPE_SPHERE;
    s.p = p;
    s.q = re_quat_identity();
    s.radius = 0.0f;
    return s;
}

/* ------------------------------------------------------------------ */
/* Support mappings                                                    */
/* ------------------------------------------------------------------ */

/** @return World-space support point of a box in local direction ld. */
static ReVec3 support_box(const ReConvexShape *s, ReVec3 ld)
{
    ReVec3 h = re_vec3_abs(s->half);
    ReVec3 local = re_vec3_make(sign_nonzero(ld.x) * h.x,
                                sign_nonzero(ld.y) * h.y,
                                sign_nonzero(ld.z) * h.z);
    return re_vec3_add(s->p, shape_to_world(s, local));
}

/** @return World-space support point of a capsule in local direction ld. */
static ReVec3 support_capsule(const ReConvexShape *s, ReVec3 ld)
{
    ReVec3 n = re_vec3_normalize(vec3_dir_or_axis(ld));
    ReVec3 base = re_vec3_make(sign_nonzero(ld.x) * magnitude(s->half.x), 0.0f, 0.0f);
    ReVec3 local = re_vec3_add_scaled(base, n, magnitude(s->radius));
    return re_vec3_add(s->p, shape_to_world(s, local));
}

/** @return World-space support point of a hull; false when it is empty. */
static bool support_hull(const ReConvexShape *s, ReVec3 ld, ReVec3 *out)
{
    if (s->points == NULL || s->point_count == 0u) {
        return false;
    }
    uint32_t best = 0u;
    float best_dot = re_vec3_dot(ld, s->points[0]);
    for (uint32_t i = 1u; i < s->point_count; ++i) {
        float d = re_vec3_dot(ld, s->points[i]);
        if (d > best_dot) {
            best_dot = d;
            best = i;
        }
    }
    *out = re_vec3_add(s->p, shape_to_world(s, s->points[best]));
    return true;
}

bool re_shape_support(const ReConvexShape *s, ReVec3 d, ReVec3 *out)
{
    if (s == NULL || out == NULL) {
        return false;
    }
    ReVec3 dir = vec3_dir_or_axis(d);
    ReVec3 ld = shape_to_local(s, dir);
    if (!(re_vec3_length_sq(ld) > 0.0f)) {
        ld = re_vec3_make(1.0f, 0.0f, 0.0f);
    }
    if (s->kind == RE_SHAPE_SPHERE) {
        ReVec3 n = re_vec3_normalize(dir);
        *out = re_vec3_add_scaled(s->p, n, magnitude(s->radius));
        return true;
    }
    if (s->kind == RE_SHAPE_BOX) {
        *out = support_box(s, ld);
        return true;
    }
    if (s->kind == RE_SHAPE_CAPSULE) {
        *out = support_capsule(s, ld);
        return true;
    }
    if (s->kind == RE_SHAPE_HULL) {
        return support_hull(s, ld, out);
    }
    return false;
}

/**
 * @brief Minkowski support: A along dir, B along -dir.
 * @return False when either shape has no support mapping.
 */
static bool support_pair(const ReConvexShape *a, const ReConvexShape *b, ReVec3 dir,
                         ReVec3 *wa, ReVec3 *wb)
{
    ReVec3 sa;
    ReVec3 sb;
    if (!re_shape_support(a, dir, &sa)) {
        return false;
    }
    if (!re_shape_support(b, re_vec3_neg(dir), &sb)) {
        return false;
    }
    *wa = sa;
    *wb = sb;
    return true;
}

/* ------------------------------------------------------------------ */
/* Simplex reduction                                                   */
/* ------------------------------------------------------------------ */

/**
 * @brief Minkowski simplex plus the matching surface witnesses.
 */
typedef struct Simplex {
    ReVec3 w[4];  /**< Minkowski difference points (pa - pb). */
    ReVec3 pa[4]; /**< Support points on A. */
    ReVec3 pb[4]; /**< Support points on B. */
    int n;        /**< Vertex count in [0, 4]. */
} Simplex;

/**
 * @brief Best candidate feature found while reducing a simplex.
 */
typedef struct SimplexCandidate {
    float dist_sq; /**< Squared distance of `closest` from the origin. */
    int n;         /**< Supporting vertex count in [1, 3]. */
    int idx[3];    /**< Indices into the source simplex. */
    float bary[3]; /**< Barycentric weights, parallel to idx. */
    ReVec3 closest;/**< Closest point on the candidate feature. */
} SimplexCandidate;

/**
 * @brief Result of reducing a simplex to its closest feature.
 */
typedef struct SimplexClosest {
    Simplex reduced;      /**< Reduced simplex (1..3 vertices). */
    ReVec3 closest;       /**< Closest point of the Minkowski set to origin. */
    ReVec3 point_a;       /**< Witness on A (barycentric blend of pa). */
    ReVec3 point_b;       /**< Witness on B (barycentric blend of pb). */
    float dist_sq;        /**< |closest|^2. */
    bool contains_origin; /**< True when a tetrahedron encloses the origin. */
} SimplexClosest;

/** @brief Resets a candidate so the first proposal always wins. */
static void candidate_seed(SimplexCandidate *c)
{
    memset(c, 0, sizeof *c);
    c->dist_sq = INFINITY;
}

/**
 * @brief Offers a feature to the candidate search, keeping the nearest.
 *
 * Ties are resolved in favour of the earlier proposal, which keeps the
 * reduction deterministic (vertices before edges before faces).
 */
static void candidate_consider(SimplexCandidate *c, const int *idx, const float *bary, int n,
                               ReVec3 closest)
{
    float d2 = re_vec3_length_sq(closest);
    if (!(d2 < c->dist_sq)) {
        return;
    }
    c->dist_sq = d2;
    c->n = n;
    c->closest = closest;
    for (int i = 0; i < n; ++i) {
        c->idx[i] = idx[i];
        c->bary[i] = bary[i];
    }
}

/** @brief Offers a single vertex to the candidate search. */
static void candidate_add_vertex(SimplexCandidate *c, const Simplex *s, int i0)
{
    int idx[3] = {i0, i0, i0};
    float bary[3] = {1.0f, 0.0f, 0.0f};
    candidate_consider(c, idx, bary, 1, s->w[i0]);
}

/** @brief Offers the closest point of an edge (and its endpoints). */
static void candidate_add_segment(SimplexCandidate *c, const Simplex *s, int i0, int i1)
{
    ReVec3 a = s->w[i0];
    ReVec3 ab = re_vec3_sub(s->w[i1], a);
    float len2 = re_vec3_length_sq(ab);
    float t = 0.0f;
    if (len2 > 0.0f) {
        t = -re_vec3_dot(a, ab) / len2;
    }
    if (t < 0.0f) {
        t = 0.0f;
    }
    if (t > 1.0f) {
        t = 1.0f;
    }
    if (t <= 0.0f) {
        candidate_add_vertex(c, s, i0);
        return;
    }
    if (t >= 1.0f) {
        candidate_add_vertex(c, s, i1);
        return;
    }
    int idx[3] = {i0, i1, i1};
    float bary[3] = {1.0f - t, t, 0.0f};
    candidate_consider(c, idx, bary, 2, re_vec3_add_scaled(a, ab, t));
}

/**
 * @brief Barycentric coordinates of p in the triangle (a, b, c).
 * @return False when the triangle is degenerate (relative tolerance).
 */
static bool triangle_bary(ReVec3 a, ReVec3 b, ReVec3 c, ReVec3 p, float *u, float *v, float *w)
{
    ReVec3 v0 = re_vec3_sub(b, a);
    ReVec3 v1 = re_vec3_sub(c, a);
    ReVec3 v2 = re_vec3_sub(p, a);
    float d00 = re_vec3_dot(v0, v0);
    float d01 = re_vec3_dot(v0, v1);
    float d11 = re_vec3_dot(v1, v1);
    float d20 = re_vec3_dot(v2, v0);
    float d21 = re_vec3_dot(v2, v1);
    float den = d00 * d11 - d01 * d01;
    if (!(den > 1.0e-7f * d00 * d11)) {
        return false;
    }
    *v = (d11 * d20 - d01 * d21) / den;
    *w = (d00 * d21 - d01 * d20) / den;
    *u = 1.0f - *v - *w;
    return true;
}

/** @brief Offers the interior of a triangle when the origin projects inside it. */
static void candidate_add_face(SimplexCandidate *c, const Simplex *s, int i0, int i1, int i2)
{
    ReVec3 a = s->w[i0];
    ReVec3 b = s->w[i1];
    ReVec3 cc = s->w[i2];
    ReVec3 nrm = re_vec3_cross(re_vec3_sub(b, a), re_vec3_sub(cc, a));
    float len2 = re_vec3_length_sq(nrm);
    if (!(len2 > 0.0f)) {
        return;
    }
    ReVec3 n = re_vec3_scale(nrm, 1.0f / sqrtf(len2));
    ReVec3 proj = re_vec3_scale(n, re_vec3_dot(n, a));
    float u = 0.0f;
    float v = 0.0f;
    float w = 0.0f;
    if (!triangle_bary(a, b, cc, proj, &u, &v, &w)) {
        return;
    }
    if (u < 0.0f || v < 0.0f || w < 0.0f) {
        return;
    }
    int idx[3] = {i0, i1, i2};
    float bary[3] = {u, v, w};
    candidate_consider(c, idx, bary, 3, proj);
}

/** @brief Offers every feature (vertices, edges, interior) of one face. */
static void triangle_candidates(SimplexCandidate *c, const Simplex *s, int i0, int i1, int i2)
{
    candidate_add_vertex(c, s, i0);
    candidate_add_vertex(c, s, i1);
    candidate_add_vertex(c, s, i2);
    candidate_add_segment(c, s, i0, i1);
    candidate_add_segment(c, s, i1, i2);
    candidate_add_segment(c, s, i2, i0);
    candidate_add_face(c, s, i0, i1, i2);
}

/**
 * @brief True when the origin and `opposite` lie on the same side of (a, b, c).
 */
static bool origin_same_side(ReVec3 a, ReVec3 b, ReVec3 c, ReVec3 opposite)
{
    ReVec3 nrm = re_vec3_cross(re_vec3_sub(b, a), re_vec3_sub(c, a));
    float side_o = re_vec3_dot(nrm, re_vec3_neg(a));
    float side_p = re_vec3_dot(nrm, re_vec3_sub(opposite, a));
    if (side_p > 0.0f) {
        return side_o >= 0.0f;
    }
    if (side_p < 0.0f) {
        return side_o <= 0.0f;
    }
    return true;
}

/** @return True when the tetrahedron encloses (or touches) the origin. */
static bool tetra_contains_origin(const Simplex *s)
{
    return origin_same_side(s->w[0], s->w[1], s->w[2], s->w[3])
        && origin_same_side(s->w[0], s->w[1], s->w[3], s->w[2])
        && origin_same_side(s->w[0], s->w[2], s->w[3], s->w[1])
        && origin_same_side(s->w[1], s->w[2], s->w[3], s->w[0]);
}

/** @return True when the tetrahedron has non-negligible volume. */
static bool tetra_non_degenerate(const Simplex *s)
{
    ReVec3 a = s->w[0];
    ReVec3 e1 = re_vec3_sub(s->w[1], a);
    ReVec3 e2 = re_vec3_sub(s->w[2], a);
    ReVec3 e3 = re_vec3_sub(s->w[3], a);
    float six_vol = fabsf(re_vec3_dot(re_vec3_cross(e1, e2), e3));
    float longest = re_vec3_length_sq(e1);
    if (re_vec3_length_sq(e2) > longest) {
        longest = re_vec3_length_sq(e2);
    }
    if (re_vec3_length_sq(e3) > longest) {
        longest = re_vec3_length_sq(e3);
    }
    if (!(longest > 0.0f)) {
        return false;
    }
    return six_vol > 1.0e-6f * longest * sqrtf(longest);
}

/** @brief Fills the reduced simplex and the blended witnesses. */
static void simplex_finalize(const Simplex *s, const SimplexCandidate *c, SimplexClosest *out)
{
    out->reduced.n = (c->n > 0) ? c->n : 1;
    out->closest = re_vec3_zero();
    out->point_a = re_vec3_zero();
    out->point_b = re_vec3_zero();
    for (int i = 0; i < out->reduced.n; ++i) {
        int src = (c->n > 0) ? c->idx[i] : 0;
        out->reduced.w[i] = s->w[src];
        out->reduced.pa[i] = s->pa[src];
        out->reduced.pb[i] = s->pb[src];
        out->closest = re_vec3_add_scaled(out->closest, s->w[src], c->bary[i]);
        out->point_a = re_vec3_add_scaled(out->point_a, s->pa[src], c->bary[i]);
        out->point_b = re_vec3_add_scaled(out->point_b, s->pb[src], c->bary[i]);
    }
    out->dist_sq = re_vec3_length_sq(out->closest);
    out->contains_origin = false;
}

/** @brief Reduction for a tetrahedron that encloses the origin. */
static void simplex_inside_tetra(const Simplex *s, SimplexClosest *out)
{
    memset(out, 0, sizeof *out);
    out->reduced.n = 4;
    out->point_a = re_vec3_zero();
    out->point_b = re_vec3_zero();
    for (int i = 0; i < 4; ++i) {
        out->reduced.w[i] = s->w[i];
        out->reduced.pa[i] = s->pa[i];
        out->reduced.pb[i] = s->pb[i];
        out->point_a = re_vec3_add_scaled(out->point_a, s->pa[i], 0.25f);
        out->point_b = re_vec3_add_scaled(out->point_b, s->pb[i], 0.25f);
    }
    out->closest = re_vec3_zero();
    out->dist_sq = 0.0f;
    out->contains_origin = true;
}

/** @brief Closes the origin on the closest feature of a 1..4 vertex simplex. */
static SimplexClosest simplex_reduce(const Simplex *s)
{
    SimplexClosest out;
    memset(&out, 0, sizeof out);
    if (s->n <= 1) {
        SimplexCandidate c;
        candidate_seed(&c);
        candidate_add_vertex(&c, s, 0);
        simplex_finalize(s, &c, &out);
    } else if (s->n == 2) {
        SimplexCandidate c;
        candidate_seed(&c);
        candidate_add_segment(&c, s, 0, 1);
        simplex_finalize(s, &c, &out);
    } else if (s->n == 3) {
        SimplexCandidate c;
        candidate_seed(&c);
        triangle_candidates(&c, s, 0, 1, 2);
        simplex_finalize(s, &c, &out);
    } else if (tetra_contains_origin(s) && tetra_non_degenerate(s)) {
        simplex_inside_tetra(s, &out);
    } else {
        SimplexCandidate c;
        candidate_seed(&c);
        triangle_candidates(&c, s, 0, 1, 2);
        triangle_candidates(&c, s, 0, 1, 3);
        triangle_candidates(&c, s, 0, 2, 3);
        triangle_candidates(&c, s, 1, 2, 3);
        simplex_finalize(s, &c, &out);
    }
    return out;
}

/* ------------------------------------------------------------------ */
/* GJK distance                                                        */
/* ------------------------------------------------------------------ */

/**
 * @brief GJK working state.
 */
typedef struct GjkState {
    Simplex simplex; /**< Current (reduced) simplex. */
    ReVec3 closest;  /**< Closest Minkowski point to the origin. */
    ReVec3 point_a;  /**< Witness on A. */
    ReVec3 point_b;  /**< Witness on B. */
    float dist_sq;   /**< |closest|^2. */
    bool hit;        /**< Shapes overlap. */
    bool inside;     /**< Origin strictly inside the simplex. */
} GjkState;

/** @brief Conservative result built from the shape centres. */
static GjkState gjk_fallback(const ReConvexShape *a, const ReConvexShape *b)
{
    GjkState st;
    memset(&st, 0, sizeof st);
    st.closest = re_vec3_sub(a->p, b->p);
    st.dist_sq = re_vec3_length_sq(st.closest);
    st.point_a = a->p;
    st.point_b = b->p;
    return st;
}

/** @brief Stores a reduction as the current best state. */
static void gjk_accept(GjkState *st, const SimplexClosest *sc, bool hit)
{
    st->simplex = sc->reduced;
    st->closest = sc->closest;
    st->point_a = sc->point_a;
    st->point_b = sc->point_b;
    st->dist_sq = sc->dist_sq;
    st->hit = hit;
    st->inside = sc->contains_origin;
}

/** @brief Outcome of one GJK iteration. */
typedef enum GjkStep {
    GJK_STEP_CONTINUE, /**< Simplex improved; keep iterating. */
    GJK_STEP_DONE,     /**< Converged (or hit) with the current state. */
    GJK_STEP_FAIL      /**< A support mapping is missing: use the fallback. */
} GjkStep;

/** @brief Working set threaded through the GJK iterations. */
typedef struct GjkWork {
    Simplex simplex;  /**< Current simplex. */
    ReVec3 closest;   /**< Closest Minkowski point so far. */
    float closest_sq; /**< |closest|^2, INFINITY before the first vertex. */
    ReVec3 dir;       /**< Next search direction. */
    bool flipped;     /**< The degenerate first-direction flip was used. */
} GjkWork;

/** @brief Standard GJK stopping rule for the current simplex. */
static bool gjk_no_progress(const GjkWork *work, ReVec3 w)
{
    if (work->simplex.n == 0) {
        return !(re_vec3_dot(w, work->dir) >= 0.0f);
    }
    float progress = re_vec3_dot(w, work->dir) - re_vec3_dot(work->closest, work->dir);
    return progress < GJK_PROGRESS_EPS * (1.0f + work->closest_sq);
}

/**
 * @brief Reduces the extended simplex and adopts it as the new state.
 * @return GJK_STEP_DONE when converged or overlapping.
 */
static GjkStep gjk_reduce_step(GjkWork *work, GjkState *st)
{
    SimplexClosest sc = simplex_reduce(&work->simplex);
    if (sc.dist_sq > work->closest_sq) {
        return GJK_STEP_DONE; /* rounding regression: keep the previous best */
    }
    gjk_accept(st, &sc, false);
    if (sc.contains_origin || sc.dist_sq < GJK_CONTACT_EPS2) {
        st->hit = true;
        return GJK_STEP_DONE;
    }
    work->simplex = sc.reduced;
    work->closest = sc.closest;
    work->closest_sq = sc.dist_sq;
    work->dir = re_vec3_neg(sc.closest);
    return GJK_STEP_CONTINUE;
}

/**
 * @brief One GJK iteration: support query, progress test, reduce, accept.
 */
static GjkStep gjk_iterate(const ReConvexShape *a, const ReConvexShape *b, GjkWork *work,
                           GjkState *st)
{
    ReVec3 wa = re_vec3_zero();
    ReVec3 wb = re_vec3_zero();
    if (!support_pair(a, b, work->dir, &wa, &wb)) {
        return GJK_STEP_FAIL;
    }
    ReVec3 w = re_vec3_sub(wa, wb);
    if (gjk_no_progress(work, w)) {
        if (work->simplex.n > 0) {
            return GJK_STEP_DONE;
        }
        if (work->flipped) {
            return GJK_STEP_FAIL;
        }
        work->flipped = true;
        work->dir = re_vec3_neg(work->dir);
        return GJK_STEP_CONTINUE;
    }
    if (work->simplex.n >= 4) {
        return GJK_STEP_DONE;
    }
    int n = work->simplex.n;
    work->simplex.w[n] = w;
    work->simplex.pa[n] = wa;
    work->simplex.pb[n] = wb;
    work->simplex.n = n + 1;
    return gjk_reduce_step(work, st);
}

/** @brief Distance GJK. At most RE_GJK_MAX_ITER support queries. */
static GjkState gjk_run(const ReConvexShape *a, const ReConvexShape *b)
{
    GjkState st = gjk_fallback(a, b);
    GjkWork work;
    memset(&work, 0, sizeof work);
    work.closest = st.closest;
    work.closest_sq = INFINITY;
    work.dir = re_vec3_sub(b->p, a->p);
    if (!(re_vec3_length_sq(work.dir) > 0.0f)) {
        work.dir = re_vec3_make(1.0f, 0.0f, 0.0f);
    }
    for (uint32_t iter = 0; iter < RE_GJK_MAX_ITER; ++iter) {
        GjkStep step = gjk_iterate(a, b, &work, &st);
        if (step == GJK_STEP_FAIL) {
            return gjk_fallback(a, b);
        }
        if (step == GJK_STEP_DONE) {
            break;
        }
    }
    return st;
}

ReGjkResult re_gjk_distance(const ReConvexShape *a, const ReConvexShape *b)
{
    ReGjkResult out;
    memset(&out, 0, sizeof out);
    if (a == NULL || b == NULL) {
        return out;
    }
    GjkState st = gjk_run(a, b);
    out.hit = st.hit;
    out.distance = st.hit ? 0.0f : sqrtf(fmaxf(st.dist_sq, 0.0f));
    out.point_a = st.point_a;
    out.point_b = st.point_b;
    return out;
}

/* ------------------------------------------------------------------ */
/* EPA                                                                 */
/* ------------------------------------------------------------------ */

/**
 * @brief EPA polytope vertices with their generating witnesses.
 */
typedef struct EpaPolytope {
    ReVec3 p[RE_EPA_MAX_VERTICES];  /**< Minkowski points. */
    ReVec3 pa[RE_EPA_MAX_VERTICES]; /**< Witness on A. */
    ReVec3 pb[RE_EPA_MAX_VERTICES]; /**< Witness on B. */
    uint32_t vcount;                /**< Used vertices (<= RE_EPA_MAX_VERTICES). */
} EpaPolytope;

_Static_assert(RE_EPA_MAX_VERTICES >= 4u, "EPA needs pool room for a tetrahedron");
_Static_assert(RE_EPA_MAX_FACES >= 4u, "EPA needs pool room for a tetrahedron");

/**
 * @brief One polytope face, wound consistently and outward facing.
 */
typedef struct EpaFace {
    uint32_t idx[3]; /**< Vertex indices, counter-clockwise seen from outside. */
    ReVec3 normal;   /**< Unit outward normal. */
    float dist;      /**< Signed plane distance: >= 0 for faces in the mesh. */
} EpaFace;

/** @brief True when w coincides (relatively) with an existing vertex. */
static bool epa_is_duplicate(const EpaPolytope *poly, ReVec3 w)
{
    float w_sq = re_vec3_length_sq(w);
    for (uint32_t i = 0; i < poly->vcount; ++i) {
        float scale = 1.0f + w_sq + re_vec3_length_sq(poly->p[i]);
        if (re_vec3_length_sq(re_vec3_sub(poly->p[i], w)) <= EPA_VERTEX_EPS2 * scale) {
            return true;
        }
    }
    return false;
}

/** @brief Appends a vertex when the pool has room. */
static bool epa_push_vertex(EpaPolytope *poly, ReVec3 w, ReVec3 wa, ReVec3 wb, uint32_t *index)
{
    if (poly->vcount >= RE_EPA_MAX_VERTICES) {
        return false;
    }
    uint32_t i = poly->vcount;
    poly->p[i] = w;
    poly->pa[i] = wa;
    poly->pb[i] = wb;
    poly->vcount = i + 1u;
    if (index != NULL) {
        *index = i;
    }
    return true;
}

/**
 * @brief Builds a face from three vertices using the given winding.
 *
 * The plane distance is signed: it is positive when the face points away
 * from the origin, which is the invariant the expansion relies on.
 * @return False when the triangle is degenerate.
 */
static bool epa_face_init(const EpaPolytope *poly, uint32_t i0, uint32_t i1, uint32_t i2,
                          EpaFace *f)
{
    ReVec3 a = poly->p[i0];
    ReVec3 b = poly->p[i1];
    ReVec3 c = poly->p[i2];
    ReVec3 nrm = re_vec3_cross(re_vec3_sub(b, a), re_vec3_sub(c, a));
    float len2 = re_vec3_length_sq(nrm);
    if (!(len2 > 0.0f)) {
        return false;
    }
    f->idx[0] = i0;
    f->idx[1] = i1;
    f->idx[2] = i2;
    f->normal = re_vec3_scale(nrm, 1.0f / sqrtf(len2));
    f->dist = re_vec3_dot(f->normal, a);
    return true;
}

/** @brief Appends a support point off the simplex' affine span. */
static bool epa_expand_one(const ReConvexShape *a, const ReConvexShape *b, EpaPolytope *poly)
{
    static const ReVec3 AXES[6] = {
        {1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
        {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}};
    for (uint32_t k = 0; k < 6u; ++k) {
        ReVec3 wa = re_vec3_zero();
        ReVec3 wb = re_vec3_zero();
        if (!support_pair(a, b, AXES[k], &wa, &wb)) {
            return false;
        }
        ReVec3 w = re_vec3_sub(wa, wb);
        if (re_vec3_length_sq(re_vec3_sub(w, poly->p[0])) > 1.0e-12f) {
            return epa_push_vertex(poly, w, wa, wb, NULL);
        }
    }
    return false;
}

/** @brief Appends a support point that is not collinear with the segment. */
static bool epa_expand_two(const ReConvexShape *a, const ReConvexShape *b, EpaPolytope *poly)
{
    ReVec3 base = poly->p[0];
    ReVec3 ab = re_vec3_sub(poly->p[1], base);
    float len2 = re_vec3_length_sq(ab);
    if (!(len2 > 0.0f)) {
        return false;
    }
    for (int k = 0; k < 3; ++k) {
        ReVec3 perp = re_vec3_cross(ab, vec3_axis(k, 1.0f));
        if (!(re_vec3_length_sq(perp) > 1.0e-12f * len2)) {
            continue;
        }
        for (int sgn = 0; sgn < 2; ++sgn) {
            ReVec3 dir = (sgn == 0) ? perp : re_vec3_neg(perp);
            ReVec3 wa = re_vec3_zero();
            ReVec3 wb = re_vec3_zero();
            if (!support_pair(a, b, dir, &wa, &wb)) {
                return false;
            }
            ReVec3 off = re_vec3_sub(re_vec3_sub(wa, wb), base);
            ReVec3 along = re_vec3_scale(ab, re_vec3_dot(off, ab) / len2);
            if (re_vec3_length_sq(re_vec3_sub(off, along)) > 1.0e-10f * len2) {
                return epa_push_vertex(poly, re_vec3_sub(wa, wb), wa, wb, NULL);
            }
        }
    }
    return false;
}

/** @brief Appends a support point off the triangle's plane. */
static bool epa_expand_three(const ReConvexShape *a, const ReConvexShape *b, EpaPolytope *poly)
{
    ReVec3 base = poly->p[0];
    ReVec3 nrm = re_vec3_cross(re_vec3_sub(poly->p[1], base), re_vec3_sub(poly->p[2], base));
    float len2 = re_vec3_length_sq(nrm);
    if (!(len2 > 0.0f)) {
        if (poly->vcount > 2u) {
            poly->vcount = 2u;
        }
        return epa_expand_two(a, b, poly);
    }
    float nlen = sqrtf(len2);
    ReVec3 scale_ref = re_vec3_sub(poly->p[1], base);
    float edge = sqrtf(re_vec3_length_sq(scale_ref));
    if (!(edge > 0.0f)) {
        edge = 1.0f;
    }
    for (int sgn = 0; sgn < 2; ++sgn) {
        ReVec3 dir = (sgn == 0) ? nrm : re_vec3_neg(nrm);
        ReVec3 wa = re_vec3_zero();
        ReVec3 wb = re_vec3_zero();
        if (!support_pair(a, b, dir, &wa, &wb)) {
            return false;
        }
        ReVec3 w = re_vec3_sub(wa, wb);
        float plane_dist = fabsf(re_vec3_dot(nrm, re_vec3_sub(w, base))) / nlen;
        if (plane_dist > 1.0e-6f * edge) {
            return epa_push_vertex(poly, w, wa, wb, NULL);
        }
    }
    return false;
}

/** @brief Grows a 1..3 vertex simplex into a tetrahedron (<= 3 steps). */
static bool epa_expand(const ReConvexShape *a, const ReConvexShape *b, EpaPolytope *poly)
{
    while (poly->vcount < 4u) {
        uint32_t before = poly->vcount;
        bool ok = false;
        if (before == 1u) {
            ok = epa_expand_one(a, b, poly);
        } else if (before == 2u) {
            ok = epa_expand_two(a, b, poly);
        } else if (before == 3u) {
            ok = epa_expand_three(a, b, poly);
        } else {
            return false;
        }
        if (!ok || poly->vcount <= before) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Copies the GJK simplex into an EPA polytope and closes it to a
 *        tetrahedron that contains the origin.
 */
static bool epa_build(const ReConvexShape *a, const ReConvexShape *b, const GjkState *st,
                      EpaPolytope *poly)
{
    memset(poly, 0, sizeof *poly);
    for (int i = 0; i < st->simplex.n && i < 4; ++i) {
        if (!epa_push_vertex(poly, st->simplex.w[i], st->simplex.pa[i], st->simplex.pb[i],
                             NULL)) {
            return false;
        }
    }
    if (!epa_expand(a, b, poly)) {
        return false;
    }
    Simplex tetra;
    memset(&tetra, 0, sizeof tetra);
    tetra.n = 4;
    for (int i = 0; i < 4; ++i) {
        tetra.w[i] = poly->p[i];
    }
    return tetra_non_degenerate(&tetra) && tetra_contains_origin(&tetra);
}

/** @brief Index of the face closest to the origin. */
static uint32_t epa_best_face(const EpaFace *faces, uint32_t fcount)
{
    uint32_t best = 0u;
    float best_dist = INFINITY;
    for (uint32_t i = 0; i < fcount; ++i) {
        if (faces[i].dist < best_dist) {
            best_dist = faces[i].dist;
            best = i;
        }
    }
    return best;
}

/** @brief Builds the four outward faces of the initial tetrahedron. */
static bool epa_init_faces(const EpaPolytope *poly, EpaFace *faces, uint32_t *fcount)
{
    static const uint32_t FACES[4][3] = {{0u, 1u, 2u}, {0u, 2u, 3u}, {0u, 3u, 1u}, {1u, 3u, 2u}};
    static const uint32_t OPPOSITE[4] = {3u, 1u, 2u, 0u};
    for (uint32_t i = 0; i < 4u; ++i) {
        if (!epa_face_init(poly, FACES[i][0], FACES[i][1], FACES[i][2], &faces[i])) {
            return false;
        }
        ReVec3 inward = re_vec3_sub(poly->p[OPPOSITE[i]], poly->p[FACES[i][0]]);
        if (re_vec3_dot(faces[i].normal, inward) > 0.0f) {
            uint32_t tmp = faces[i].idx[1];
            faces[i].idx[1] = faces[i].idx[2];
            faces[i].idx[2] = tmp;
            faces[i].normal = re_vec3_neg(faces[i].normal);
        }
        faces[i].dist = fmaxf(re_vec3_dot(faces[i].normal, poly->p[faces[i].idx[0]]), 0.0f);
    }
    *fcount = 4u;
    return true;
}

/** @brief True when w can see the face from outside. */
static bool epa_face_visible(const EpaFace *f, const EpaPolytope *poly, ReVec3 w)
{
    ReVec3 v = re_vec3_sub(w, poly->p[f->idx[0]]);
    return re_vec3_dot(f->normal, v) > 1.0e-9f;
}

/** @brief True when the reversed edge (v, u) is absent from the visible set. */
static bool epa_edge_is_horizon(const EpaFace *faces, const bool *visible, uint32_t fcount,
                                uint32_t u, uint32_t v)
{
    for (uint32_t i = 0; i < fcount; ++i) {
        if (!visible[i]) {
            continue;
        }
        for (uint32_t e = 0; e < 3u; ++e) {
            uint32_t a = faces[i].idx[e];
            uint32_t b = faces[i].idx[(e + 1u) % 3u];
            if (a == v && b == u) {
                return false;
            }
        }
    }
    return true;
}

/** @brief Collects the boundary edges of the visible face region. */
static bool epa_collect_horizon(const EpaFace *faces, const bool *visible, uint32_t fcount,
                                uint32_t *hu, uint32_t *hv, uint32_t *hcount)
{
    *hcount = 0u;
    for (uint32_t i = 0; i < fcount; ++i) {
        if (!visible[i]) {
            continue;
        }
        for (uint32_t e = 0; e < 3u; ++e) {
            uint32_t u = faces[i].idx[e];
            uint32_t v = faces[i].idx[(e + 1u) % 3u];
            if (!epa_edge_is_horizon(faces, visible, fcount, u, v)) {
                continue;
            }
            if (*hcount >= EPA_HORIZON_MAX) {
                return false;
            }
            hu[*hcount] = u;
            hv[*hcount] = v;
            *hcount += 1u;
        }
    }
    return true;
}

/**
 * @brief Rebuilds the face list without the visible faces, adding the
 *        fan of triangles between the horizon edges and the new vertex.
 */
static bool epa_rebuild(const EpaPolytope *poly, const EpaFace *faces, const bool *visible,
                        uint32_t fcount, uint32_t w_idx, const uint32_t *hu, const uint32_t *hv,
                        uint32_t hcount, EpaFace *out, uint32_t *out_count)
{
    uint32_t n = 0u;
    for (uint32_t i = 0; i < fcount; ++i) {
        if (visible[i]) {
            continue;
        }
        if (n >= RE_EPA_MAX_FACES) {
            return false;
        }
        out[n] = faces[i];
        n += 1u;
    }
    for (uint32_t h = 0; h < hcount; ++h) {
        if (n >= RE_EPA_MAX_FACES) {
            return false;
        }
        if (!epa_face_init(poly, hu[h], hv[h], w_idx, &out[n])) {
            return false;
        }
        if (out[n].dist < 0.0f) {
            return false; /* inverted or through-origin sliver: stop growing */
        }
        n += 1u;
    }
    *out_count = n;
    return n > 0u;
}

/** @brief Clamps barycentric weights to a usable convex combination. */
static void bary_clamp(float *u, float *v, float *w)
{
    if (*u < 0.0f) {
        *u = 0.0f;
    }
    if (*v < 0.0f) {
        *v = 0.0f;
    }
    if (*w < 0.0f) {
        *w = 0.0f;
    }
    float sum = *u + *v + *w;
    if (!(sum > 0.0f)) {
        *u = 1.0f;
        *v = 0.0f;
        *w = 0.0f;
        return;
    }
    *u /= sum;
    *v /= sum;
    *w /= sum;
}

/** @brief A non-penetrating EPA result (normal defaults to +X). */
static ReEpaResult epa_no_result(void)
{
    ReEpaResult res;
    memset(&res, 0, sizeof res);
    res.normal = re_vec3_make(1.0f, 0.0f, 0.0f);
    return res;
}

/** @brief Blends the face witnesses by the projected barycentric weights. */
static void epa_face_witness(const EpaPolytope *poly, const EpaFace *f, ReVec3 *pa, ReVec3 *pb)
{
    ReVec3 probe = re_vec3_scale(f->normal, f->dist);
    float u = 1.0f;
    float v = 0.0f;
    float w = 0.0f;
    if (!triangle_bary(poly->p[f->idx[0]], poly->p[f->idx[1]], poly->p[f->idx[2]], probe, &u, &v,
                       &w)) {
        u = 1.0f;
        v = 0.0f;
        w = 0.0f;
    }
    bary_clamp(&u, &v, &w);
    *pa = re_vec3_scale(poly->pa[f->idx[0]], u);
    *pa = re_vec3_add_scaled(*pa, poly->pa[f->idx[1]], v);
    *pa = re_vec3_add_scaled(*pa, poly->pa[f->idx[2]], w);
    *pb = re_vec3_scale(poly->pb[f->idx[0]], u);
    *pb = re_vec3_add_scaled(*pb, poly->pb[f->idx[1]], v);
    *pb = re_vec3_add_scaled(*pb, poly->pb[f->idx[2]], w);
}

/**
 * @brief Converts the closest face into the final penetration result.
 *
 * The normal is the negative of the closest-face normal: pushing A along
 * it translates the Minkowski difference by exactly one depth, moving the
 * origin onto (and, past that, out of) the boundary.
 */
static ReEpaResult epa_finalize(const EpaPolytope *poly, const EpaFace *faces, uint32_t fcount)
{
    ReEpaResult res = epa_no_result();
    const EpaFace *f = &faces[epa_best_face(faces, fcount)];
    epa_face_witness(poly, f, &res.point_a, &res.point_b);
    res.depth = fmaxf(f->dist, 0.0f);
    res.normal = re_vec3_neg(f->normal);
    res.ok = res.depth > EPA_MIN_DEPTH;
    return res;
}

/** @brief Removes the faces visible from vertex w_idx and fans the horizon to it. */
static bool epa_absorb_vertex(EpaPolytope *poly, EpaFace *faces, uint32_t *fcount, uint32_t w_idx,
                              EpaFace *scratch)
{
    ReVec3 w = poly->p[w_idx];
    bool visible[RE_EPA_MAX_FACES];
    for (uint32_t i = 0; i < *fcount; ++i) {
        visible[i] = epa_face_visible(&faces[i], poly, w);
    }
    uint32_t hu[EPA_HORIZON_MAX];
    uint32_t hv[EPA_HORIZON_MAX];
    uint32_t hcount = 0u;
    if (!epa_collect_horizon(faces, visible, *fcount, hu, hv, &hcount)) {
        return false;
    }
    uint32_t ncount = 0u;
    if (!epa_rebuild(poly, faces, visible, *fcount, w_idx, hu, hv, hcount, scratch, &ncount)) {
        return false;
    }
    memcpy(faces, scratch, (size_t)ncount * sizeof scratch[0]);
    *fcount = ncount;
    return true;
}

/**
 * @brief One EPA expansion: add the support point of the closest face and
 *        rebuild the visible region.
 * @return False when the polytope is converged, saturated or degenerate.
 */
static bool epa_step(const ReConvexShape *a, const ReConvexShape *b, EpaPolytope *poly,
                     EpaFace *faces, uint32_t *fcount, EpaFace *scratch)
{
    uint32_t bi = epa_best_face(faces, *fcount);
    ReVec3 wa = re_vec3_zero();
    ReVec3 wb = re_vec3_zero();
    if (!support_pair(a, b, faces[bi].normal, &wa, &wb)) {
        return false;
    }
    ReVec3 w = re_vec3_sub(wa, wb);
    float growth = re_vec3_dot(faces[bi].normal, w) - faces[bi].dist;
    if (growth < EPA_GROWTH_EPS || poly->vcount >= RE_EPA_MAX_VERTICES) {
        return false;
    }
    if (epa_is_duplicate(poly, w)) {
        return false; /* the support point is already a vertex: converged */
    }
    uint32_t w_idx = 0u;
    if (!epa_push_vertex(poly, w, wa, wb, &w_idx)) {
        return false;
    }
    return epa_absorb_vertex(poly, faces, fcount, w_idx, scratch);
}

/**
 * @brief Expands the polytope until the closest face stops moving.
 *        Saturation (vertex or face pool full) returns the current best face.
 */
static ReEpaResult epa_solve(const ReConvexShape *a, const ReConvexShape *b, EpaPolytope *poly)
{
    EpaFace faces[RE_EPA_MAX_FACES];
    EpaFace scratch[RE_EPA_MAX_FACES];
    memset(faces, 0, sizeof faces);
    memset(scratch, 0, sizeof scratch);
    uint32_t fcount = 0u;
    if (!epa_init_faces(poly, faces, &fcount)) {
        return epa_no_result();
    }
    for (uint32_t iter = 0; iter < RE_EPA_MAX_ITER; ++iter) {
        if (!epa_step(a, b, poly, faces, &fcount, scratch)) {
            break;
        }
    }
    return epa_finalize(poly, faces, fcount);
}

ReEpaResult re_gjk_penetration(const ReConvexShape *a, const ReConvexShape *b)
{
    ReEpaResult out = epa_no_result();
    if (a == NULL || b == NULL) {
        return out;
    }
    GjkState st = gjk_run(a, b);
    out.point_a = st.point_a;
    out.point_b = st.point_b;
    if (!st.hit) {
        return out;
    }
    EpaPolytope poly;
    if (epa_build(a, b, &st, &poly)) {
        return epa_solve(a, b, &poly);
    }
    /* Degenerate containment (flat Minkowski set): report a zero depth
       with the best available separation direction. */
    ReVec3 sep = re_vec3_sub(st.point_a, st.point_b);
    if (re_vec3_length_sq(sep) > 0.0f) {
        out.normal = re_vec3_normalize(sep);
    }
    return out;
}

/* ------------------------------------------------------------------ */
/* Conservative advancement                                            */
/* ------------------------------------------------------------------ */

/** @brief Builds a hit result, preferring the EPA normal. */
static ReCcdResult ccd_hit(const ReConvexShape *a, const ReConvexShape *b, float t, ReVec3 fallback)
{
    ReCcdResult res;
    memset(&res, 0, sizeof res);
    res.hit = true;
    res.toi = t;
    res.normal = fallback;
    ReEpaResult pen = re_gjk_penetration(a, b);
    if (pen.ok) {
        res.normal = pen.normal;
        return res;
    }
    ReGjkResult d = re_gjk_distance(a, b);
    ReVec3 sep = re_vec3_sub(d.point_a, d.point_b);
    if (re_vec3_length_sq(sep) > 1.0e-8f) {
        res.normal = re_vec3_normalize(sep);
    }
    return res;
}

/** @brief Shapes A and B translated to time t. */
static void ccd_poses(const ReConvexShape *a, const ReConvexShape *b, ReVec3 va, ReVec3 vb,
                      float t, ReConvexShape *sa, ReConvexShape *sb)
{
    *sa = shape_translated(*a, re_vec3_scale(va, t));
    *sb = shape_translated(*b, re_vec3_scale(vb, t));
}

/** @brief Outcome of one conservative-advancement step. */
typedef enum CcdStep {
    CCD_STEP_ADVANCE, /**< Time advanced; keep iterating. */
    CCD_STEP_END,     /**< Horizon exhausted or no progress: no impact. */
    CCD_STEP_HIT      /**< Impact stored in the result. */
} CcdStep;

/**
 * @brief One conservative-advancement step.
 *
 * The time increment is distance / speed bound, which can never jump over
 * a contact because the relative motion of the shapes is bounded by speed.
 */
static CcdStep ccd_step(const ReConvexShape *a, const ReConvexShape *b, ReVec3 va, ReVec3 vb,
                        float speed, float max_t, float *t, ReVec3 *last, ReCcdResult *res)
{
    ReConvexShape sa;
    ReConvexShape sb;
    ccd_poses(a, b, va, vb, *t, &sa, &sb);
    ReGjkResult d = re_gjk_distance(&sa, &sb);
    if (d.hit) {
        *res = ccd_hit(&sa, &sb, *t, *last);
        return CCD_STEP_HIT;
    }
    ReVec3 sep = re_vec3_sub(d.point_a, d.point_b);
    if (re_vec3_length_sq(sep) > 0.0f) {
        *last = re_vec3_normalize(sep);
    }
    float t_next = *t + d.distance / speed;
    if (!(t_next > *t)) {
        return CCD_STEP_END;
    }
    if (t_next < max_t) {
        *t = t_next;
        return CCD_STEP_ADVANCE;
    }
    ReConvexShape fa;
    ReConvexShape fb;
    ccd_poses(a, b, va, vb, max_t, &fa, &fb);
    if (re_gjk_distance(&fa, &fb).hit) {
        *res = ccd_hit(&fa, &fb, max_t, *last);
        return CCD_STEP_HIT;
    }
    return CCD_STEP_END;
}

ReCcdResult re_ccd_advance(const ReConvexShape *a, const ReConvexShape *b, ReVec3 va, ReVec3 vb,
                           float max_t)
{
    ReCcdResult res;
    memset(&res, 0, sizeof res);
    res.normal = re_vec3_make(1.0f, 0.0f, 0.0f);
    if (a == NULL || b == NULL) {
        return res;
    }
    float speed = re_vec3_length(re_vec3_sub(va, vb));
    if (!(max_t > 0.0f) || !(speed > 1.0e-8f)) {
        if (re_gjk_distance(a, b).hit) {
            return ccd_hit(a, b, 0.0f, res.normal);
        }
        return res;
    }
    float t = 0.0f;
    ReVec3 last = re_vec3_make(1.0f, 0.0f, 0.0f);
    for (uint32_t iter = 0; iter < RE_CCD_MAX_ITER; ++iter) {
        CcdStep step = ccd_step(a, b, va, vb, speed, max_t, &t, &last, &res);
        if (step == CCD_STEP_HIT) {
            return res;
        }
        if (step == CCD_STEP_END) {
            break;
        }
    }
    return res;
}

/* ------------------------------------------------------------------ */
/* Ray tests                                                           */
/* ------------------------------------------------------------------ */

/** @brief Analytic ray/sphere test (dir need not be normalized). */
static bool ray_sphere(const ReConvexShape *s, ReRay ray, float t_max, float *t_out,
                       ReVec3 *n_out)
{
    float r = magnitude(s->radius);
    ReVec3 oc = re_vec3_sub(ray.origin, s->p);
    float qa = re_vec3_dot(ray.dir, ray.dir);
    if (!(qa > 0.0f)) {
        return false;
    }
    float qb = re_vec3_dot(oc, ray.dir);
    float qc = re_vec3_dot(oc, oc) - r * r;
    float disc = qb * qb - qa * qc;
    if (!(disc >= 0.0f)) {
        return false;
    }
    float sq = sqrtf(disc);
    float t0 = (-qb - sq) / qa;
    float t1 = (-qb + sq) / qa;
    float t = (t0 >= 0.0f) ? t0 : t1;
    if (!(t >= 0.0f) || t > t_max) {
        return false;
    }
    *t_out = t;
    if (n_out != NULL) {
        ReVec3 hit = re_vec3_add_scaled(ray.origin, ray.dir, t);
        ReVec3 n = re_vec3_normalize(re_vec3_sub(hit, s->p));
        *n_out = (re_vec3_length_sq(n) > 0.0f) ? n : re_vec3_make(1.0f, 0.0f, 0.0f);
    }
    return true;
}

/**
 * @brief Slab clip state for the box ray test.
 */
typedef struct SlabClip {
    float t_min;   /**< Current entry distance. */
    float t_max;   /**< Current exit distance. */
    int axis_min;  /**< Axis that produced the entry. */
    int axis_max;  /**< Axis that produced the exit. */
    float sign_min;/**< Entry normal sign along axis_min. */
    float sign_max;/**< Exit normal sign along axis_max. */
} SlabClip;

/** @brief Clips one axis; false when the slab rejects the ray. */
static bool slab_axis(SlabClip *c, float o, float d, float hh, int axis)
{
    if (fabsf(d) < 1.0e-12f) {
        return fabsf(o) <= hh; /* parallel: inside the slab or missed */
    }
    float inv = 1.0f / d;
    float ta = (-hh - o) * inv;
    float tb = (hh - o) * inv;
    float near_t = (ta < tb) ? ta : tb;
    float far_t = (ta < tb) ? tb : ta;
    float near_sgn = (d > 0.0f) ? -1.0f : 1.0f;
    if (near_t > c->t_min) {
        c->t_min = near_t;
        c->axis_min = axis;
        c->sign_min = near_sgn;
    }
    if (far_t < c->t_max) {
        c->t_max = far_t;
        c->axis_max = axis;
        c->sign_max = -near_sgn;
    }
    return c->t_min <= c->t_max;
}

/**
 * @brief Slab test in the box's local frame.
 *
 * A ray that starts inside the box reports the exit distance, which is
 * the first surface crossing at t >= 0.
 */
static bool ray_box(const ReConvexShape *s, ReRay ray, float t_max, float *t_out, ReVec3 *n_out)
{
    ReVec3 lo = shape_to_local(s, re_vec3_sub(ray.origin, s->p));
    ReVec3 ld = shape_to_local(s, ray.dir);
    ReVec3 h = re_vec3_abs(s->half);
    SlabClip c;
    memset(&c, 0, sizeof c);
    c.t_min = -INFINITY;
    c.t_max = INFINITY;
    c.sign_min = 1.0f;
    c.sign_max = -1.0f;
    for (int i = 0; i < 3; ++i) {
        if (!slab_axis(&c, vec3_comp(lo, i), vec3_comp(ld, i), vec3_comp(h, i), i)) {
            return false;
        }
    }
    if (c.t_max < 0.0f) {
        return false;
    }
    bool inside = c.t_min < 0.0f;
    float t = inside ? c.t_max : c.t_min;
    if (!(t >= 0.0f) || t > t_max) {
        return false;
    }
    *t_out = t;
    if (n_out != NULL) {
        *n_out = shape_to_world(s, vec3_axis(inside ? c.axis_max : c.axis_min,
                                            inside ? c.sign_max : c.sign_min));
    }
    return true;
}

/**
 * @brief Nearest ray/shape surface hit accumulated across analytic tests.
 */
typedef struct RayShot {
    bool found;   /**< A candidate hit has been recorded. */
    float t;      /**< Distance along the ray. */
    ReVec3 normal;/**< Local surface normal. */
} RayShot;

/** @brief Keeps a candidate when it is the nearest hit so far. */
static void ray_shot_offer(RayShot *shot, float t, ReVec3 normal, float t_max)
{
    if (!(t >= 0.0f) || t > t_max || !(t < shot->t)) {
        return;
    }
    shot->found = true;
    shot->t = t;
    shot->normal = normal;
}

/** @brief Cylinder body of a capsule, clipped to the segment. */
static void capsule_body_hit(ReVec3 lo, ReVec3 ld, float hl, float r, float t_max, RayShot *shot)
{
    float qa = ld.y * ld.y + ld.z * ld.z;
    if (!(qa > 0.0f) || !(r > 0.0f)) {
        return;
    }
    float qb = 2.0f * (lo.y * ld.y + lo.z * ld.z);
    float qc = lo.y * lo.y + lo.z * lo.z - r * r;
    float disc = qb * qb - 4.0f * qa * qc;
    if (!(disc >= 0.0f)) {
        return;
    }
    float sq = sqrtf(disc);
    float roots[2] = {(-qb - sq) / (2.0f * qa), (-qb + sq) / (2.0f * qa)};
    for (int k = 0; k < 2; ++k) {
        float t = roots[k];
        if (fabsf(lo.x + ld.x * t) > hl) {
            continue;
        }
        ReVec3 n = re_vec3_make(0.0f, lo.y + ld.y * t, lo.z + ld.z * t);
        ray_shot_offer(shot, t, re_vec3_scale(n, 1.0f / r), t_max);
    }
}

/** @brief The two endpoint spheres of a capsule. */
static void capsule_cap_hit(ReVec3 lo, ReVec3 ld, float hl, float r, float t_max, RayShot *shot)
{
    float dir2 = re_vec3_length_sq(ld);
    for (int k = 0; k < 2; ++k) {
        ReVec3 centre = re_vec3_make((k == 0) ? hl : -hl, 0.0f, 0.0f);
        ReVec3 oc = re_vec3_sub(lo, centre);
        float qb = re_vec3_dot(oc, ld);
        float qc = re_vec3_dot(oc, oc) - r * r;
        float disc = qb * qb - dir2 * qc;
        if (!(disc >= 0.0f)) {
            continue;
        }
        float sq = sqrtf(disc);
        float t0 = (-qb - sq) / dir2;
        float t1 = (-qb + sq) / dir2;
        float t = (t0 >= 0.0f) ? t0 : t1;
        ReVec3 hit = re_vec3_add_scaled(lo, ld, t);
        ray_shot_offer(shot, t, re_vec3_normalize(re_vec3_sub(hit, centre)), t_max);
    }
}

/**
 * @brief Ray/capsule test: the capsule is the union of the clipped
 *        cylinder and the two endpoint spheres.
 */
static bool ray_capsule(const ReConvexShape *s, ReRay ray, float t_max, float *t_out,
                        ReVec3 *n_out)
{
    float r = magnitude(s->radius);
    float hl = magnitude(s->half.x);
    ReVec3 lo = shape_to_local(s, re_vec3_sub(ray.origin, s->p));
    ReVec3 ld = shape_to_local(s, ray.dir);
    if (!(re_vec3_length_sq(ld) > 0.0f)) {
        return false;
    }
    RayShot shot;
    memset(&shot, 0, sizeof shot);
    shot.t = INFINITY;
    shot.normal = re_vec3_make(1.0f, 0.0f, 0.0f);
    capsule_body_hit(lo, ld, hl, r, t_max, &shot);
    capsule_cap_hit(lo, ld, hl, r, t_max, &shot);
    if (!shot.found) {
        return false;
    }
    *t_out = shot.t;
    if (n_out != NULL) {
        *n_out = shape_to_world(s, shot.normal);
    }
    return true;
}

/** @brief Refines a contact time by halving the last conservative step. */
static float hull_ray_bisect(const ReConvexShape *s, ReRay ray, ReConvexShape point, float lo,
                             float hi)
{
    for (int i = 0; i < RAY_BISECT_STEPS; ++i) {
        float mid = 0.5f * (lo + hi);
        point.p = re_vec3_add_scaled(ray.origin, ray.dir, mid);
        ReGjkResult d = re_gjk_distance(s, &point);
        if (d.hit || d.distance <= RAY_CONTACT_EPS) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return hi;
}

/** @brief One hull-march sample result. */
typedef struct HullMarch {
    float step;   /**< Conservative distance to advance (ray-dir units). */
    ReVec3 normal;/**< Surface normal estimate at the sample. */
    bool contact; /**< The sample touches the hull. */
} HullMarch;

/** @brief Samples the hull at a ray point and reports the next step. */
static HullMarch hull_march(const ReConvexShape *s, ReVec3 sample, float len)
{
    HullMarch m;
    memset(&m, 0, sizeof m);
    m.normal = re_vec3_make(1.0f, 0.0f, 0.0f);
    ReConvexShape point = shape_point(sample);
    ReGjkResult d = re_gjk_distance(s, &point);
    if (d.hit || d.distance <= RAY_CONTACT_EPS) {
        m.contact = true;
        return m;
    }
    ReVec3 dir = re_vec3_sub(sample, d.point_a);
    if (re_vec3_length_sq(dir) > 0.0f) {
        m.normal = re_vec3_normalize(dir);
    }
    m.step = d.distance / len;
    return m;
}

/** @brief Hull ray test by conservative advancement plus bisection. */
static bool ray_hull(const ReConvexShape *s, ReRay ray, float t_max, float *t_out, ReVec3 *n_out)
{
    float len = re_vec3_length(ray.dir);
    if (!(len > 0.0f)) {
        return false;
    }
    ReConvexShape point = shape_point(ray.origin);
    float t = 0.0f;
    float t_prev = 0.0f;
    ReVec3 normal = re_vec3_make(1.0f, 0.0f, 0.0f);
    for (uint32_t iter = 0; iter < RE_GJK_MAX_ITER; ++iter) {
        point.p = re_vec3_add_scaled(ray.origin, ray.dir, t);
        HullMarch m = hull_march(s, point.p, len);
        if (m.contact) {
            *t_out = hull_ray_bisect(s, ray, point, t_prev, t);
            if (n_out != NULL) {
                *n_out = normal;
            }
            return true;
        }
        normal = m.normal;
        if (!(m.step > 0.0f) || !(t + m.step <= t_max)) {
            return false;
        }
        t_prev = t;
        t += m.step;
    }
    return false;
}

bool re_shape_ray(const ReConvexShape *s, ReRay ray, float t_max, float *t_out,
                  ReVec3 *normal_out)
{
    if (s == NULL || t_out == NULL || !(t_max >= 0.0f)) {
        return false;
    }
    *t_out = 0.0f;
    bool hit = false;
    if (s->kind == RE_SHAPE_SPHERE) {
        hit = ray_sphere(s, ray, t_max, t_out, normal_out);
    } else if (s->kind == RE_SHAPE_BOX) {
        hit = ray_box(s, ray, t_max, t_out, normal_out);
    } else if (s->kind == RE_SHAPE_CAPSULE) {
        hit = ray_capsule(s, ray, t_max, t_out, normal_out);
    } else if (s->kind == RE_SHAPE_HULL) {
        hit = ray_hull(s, ray, t_max, t_out, normal_out);
    }
    return hit;
}
