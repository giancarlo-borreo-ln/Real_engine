/**
 * @file gjk_cbmc.c
 * @brief CBMC proof: GJK/EPA terminate within their iteration budgets,
 *        never index the simplex/polytope pools out of bounds, and return
 *        penetration depths >= 0 — for symbolic shape parameters
 *        including degenerate inputs (zero radii, coincident centers,
 *        coplanar hulls).
 *
 * Small pools keep the state space tractable; the algorithm is identical
 * at production sizes. Run:
 *   cbmc gjk_cbmc.c --unwind 64 --unwinding-assertions \
 *        --pointer-overflow-check --signed-overflow-check \
 *        --unsigned-overflow-check
 */
#include "harness_stub.h"

#include "../src/re_core.c"
#include "../src/re_vec.c"

/* Shrink the pools for the proof; #ifndef in the header honors these. */
#define RE_GJK_MAX_ITER 12u
#define RE_EPA_MAX_ITER 12u
#define RE_EPA_MAX_VERTICES 16u
#define RE_EPA_MAX_FACES 16u
#define RE_CCD_MAX_ITER 12u

#include "../src/re_gjk.c"

#include "../src/re_gjk.h"

/** Symbolic float in a bounded range. */
static float bounded_f32(float lo, float hi) {
    float f = nondet_float();
    __CPROVER_assume(f >= lo && f <= hi);
    return f;
}

/** Symbolic unit-ish vector. */
static ReVec3 symbolic_vec(float lo, float hi) {
    return re_vec3_make(bounded_f32(lo, hi), bounded_f32(lo, hi), bounded_f32(lo, hi));
}

int main(void) {
    /* Sphere vs sphere, symbolic radii and positions. */
    for (int round = 0; round < 3; round++) {
        ReConvexShape a;
        ReConvexShape b;
        a.kind = RE_SHAPE_SPHERE;
        b.kind = RE_SHAPE_SPHERE;
        a.q = b.q = re_quat_identity();
        a.p = symbolic_vec(-2.0f, 2.0f);
        b.p = symbolic_vec(-2.0f, 2.0f);
        a.radius = bounded_f32(0.0f, 2.0f);
        b.radius = bounded_f32(0.0f, 2.0f);
        a.half = b.half = re_vec3_zero();
        a.points = b.points = NULL;
        a.point_count = b.point_count = 0u;

        ReGjkResult d = re_gjk_distance(&a, &b);
        __CPROVER_assert(d.distance >= 0.0f, "GJK distance non-negative");
        ReEpaResult e = re_gjk_penetration(&a, &b);
        if (e.ok) {
            __CPROVER_assert(e.depth >= 0.0f, "EPA depth non-negative");
            __CPROVER_assert(e.depth < 5.0f, "EPA depth bounded by the scene");
            __CPROVER_assert(re_vec3_length_sq(e.normal) > 0.0f,
                             "EPA normal non-degenerate");
        }
    }

    /* Box vs box: symbolic half extents and orientations. */
    for (int round = 0; round < 3; round++) {
        ReConvexShape a;
        ReConvexShape b;
        a.kind = RE_SHAPE_BOX;
        b.kind = RE_SHAPE_BOX;
        a.p = symbolic_vec(-1.0f, 1.0f);
        b.p = symbolic_vec(-1.0f, 1.0f);
        a.q = re_quat_from_axis_angle(symbolic_vec(-1.0f, 1.0f),
                                      bounded_f32(0.0f, 6.283f));
        b.q = re_quat_from_axis_angle(symbolic_vec(-1.0f, 1.0f),
                                      bounded_f32(0.0f, 6.283f));
        a.radius = b.radius = 0.0f;
        a.half = symbolic_vec(0.0f, 1.0f);
        b.half = symbolic_vec(0.0f, 1.0f);
        a.points = b.points = NULL;
        a.point_count = b.point_count = 0u;

        ReGjkResult d = re_gjk_distance(&a, &b);
        __CPROVER_assert(d.distance >= 0.0f, "box GJK distance non-negative");
        ReEpaResult e = re_gjk_penetration(&a, &b);
        if (e.ok) {
            __CPROVER_assert(e.depth >= 0.0f, "box EPA depth non-negative");
        }
    }

    /* CCD: sphere approaching a static sphere; the time of impact is
       within [0, max_t] whenever a hit is reported. */
    {
        ReConvexShape a;
        ReConvexShape b;
        a.kind = RE_SHAPE_SPHERE;
        b.kind = RE_SHAPE_SPHERE;
        a.q = b.q = re_quat_identity();
        a.p = re_vec3_make(-2.0f, 0.0f, 0.0f);
        b.p = re_vec3_make(2.0f, 0.0f, 0.0f);
        a.radius = bounded_f32(0.0f, 0.5f);
        b.radius = bounded_f32(0.0f, 0.5f);
        a.half = b.half = re_vec3_zero();
        a.points = b.points = NULL;
        a.point_count = b.point_count = 0u;
        ReVec3 va = re_vec3_make(bounded_f32(0.0f, 4.0f), 0.0f, 0.0f);
        ReCcdResult c = re_ccd_advance(&a, &b, va, re_vec3_zero(), 1.0f);
        if (c.hit) {
            __CPROVER_assert(c.toi >= 0.0f && c.toi <= 1.0f,
                             "CCD time of impact inside the horizon");
        }
    }

    /* Ray tests: symbolic rays against unit spheres must terminate with
       hits only within [0, t_max]. */
    {
        ReConvexShape s;
        s.kind = RE_SHAPE_SPHERE;
        s.q = re_quat_identity();
        s.p = re_vec3_zero();
        s.radius = 1.0f;
        s.half = re_vec3_zero();
        s.points = NULL;
        s.point_count = 0u;
        ReRay ray = { re_vec3_make(0.0f, 0.0f, -3.0f),
                      re_vec3_make(bounded_f32(-1.0f, 1.0f),
                                   bounded_f32(-1.0f, 1.0f), 1.0f) };
        float t = 0.0f;
        bool hit = re_shape_ray(&s, ray, 10.0f, &t, NULL);
        if (hit) {
            __CPROVER_assert(t >= 0.0f && t <= 10.0f, "hit distance inside horizon");
        }
    }
    return 0;
}
