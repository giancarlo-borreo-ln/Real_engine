/**
 * @file re_gjk.h
 * @brief Narrowphase: GJK distance, EPA penetration, conservative
 *        advancement CCD and ray tests for convex shapes.
 *
 * Shapes are described in world space by ReConvexShape; support
 * functions are analytic for spheres, boxes and capsules, and
 * linear-scan for point hulls. All functions are total: they terminate
 * within their iteration budgets even for degenerate inputs (zero
 * radius, coincident centers, coplanar hulls) and return conservative
 * results.
 */
#ifndef RE_GJK_H
#define RE_GJK_H

#include <stdbool.h>
#include <stdint.h>

#include "re_vec.h"

/** GJK iteration budget (smaller under __CPROVER__ for the harnesses). */
#ifndef RE_GJK_MAX_ITER
#define RE_GJK_MAX_ITER 64u
#endif

/** EPA iteration budget. */
#ifndef RE_EPA_MAX_ITER
#define RE_EPA_MAX_ITER 32u
#endif

/** EPA polytope vertex pool size. */
#ifndef RE_EPA_MAX_VERTICES
#define RE_EPA_MAX_VERTICES 64u
#endif

/** EPA face pool size. */
#ifndef RE_EPA_MAX_FACES
#define RE_EPA_MAX_FACES 64u
#endif

/** CCD conservative-advancement iteration budget. */
#ifndef RE_CCD_MAX_ITER
#define RE_CCD_MAX_ITER 32u
#endif

/**
 * @brief Convex shape kinds.
 */
typedef enum ReShapeKind {
    RE_SHAPE_SPHERE = 0, /**< Sphere (radius). */
    RE_SHAPE_BOX = 1,    /**< Oriented box (half extents). */
    RE_SHAPE_CAPSULE = 2,/**< Capsule: segment along local +X (half_length) with radius. */
    RE_SHAPE_HULL = 3    /**< Convex point cloud (points). */
} ReShapeKind;

/**
 * @brief World-space convex shape view.
 *
 * Built on the stack from body data; no ownership. For capsules the
 * segment runs from p - q*x*half_length to p + q*x*half_length.
 */
typedef struct ReConvexShape {
    ReShapeKind kind;   /**< Shape kind. */
    ReVec3 p;           /**< Center position (world). */
    ReQuat q;           /**< Orientation (world). */
    float radius;       /**< Sphere radius / capsule radius. */
    ReVec3 half;        /**< Box half extents; capsule: x = half length. */
    const ReVec3 *points; /**< Hull vertices in local space. */
    uint32_t point_count; /**< Hull vertex count. */
} ReConvexShape;

/**
 * @brief Support point of the shape in world space along direction d.
 *
 * For the capsule, the segment endpoints plus a sphere sweep; the
 * returned point is one of p +- q*(half_length,0,0) + radius*d/|d|.
 * A zero direction behaves as if d were (1, 0, 0) (deterministic).
 * @param s Shape.
 * @param d Direction (need not be normalized).
 * @param out Receives the support point.
 * @return True on success (always true for well-formed shapes).
 */
bool re_shape_support(const ReConvexShape *s, ReVec3 d, ReVec3 *out);

/**
 * @brief Result of the GJK distance query.
 */
typedef struct ReGjkResult {
    bool hit;        /**< True when the shapes overlap (distance == 0). */
    float distance;  /**< Minimum separation distance (>= 0). */
    ReVec3 point_a;  /**< Closest point on A (valid when !hit). */
    ReVec3 point_b;  /**< Closest point on B (valid when !hit). */
} ReGjkResult;

/**
 * @brief Minimum distance between two convex shapes.
 *
 * Terminates after at most RE_GJK_MAX_ITER iterations; overlapping
 * shapes return hit == true with distance 0.
 * @param a First shape.
 * @param b Second shape.
 */
ReGjkResult re_gjk_distance(const ReConvexShape *a, const ReConvexShape *b);

/**
 * @brief Result of the EPA penetration query.
 */
typedef struct ReEpaResult {
    bool ok;         /**< False only for degenerate cases (e.g. both
                          shapes are zero-size or EPA failed to expand). */
    ReVec3 normal;   /**< Contact normal pointing from B to A. */
    float depth;     /**< Penetration depth (>= 0). */
    ReVec3 point_a;  /**< Contact point on A (world). */
    ReVec3 point_b;  /**< Contact point on B (world). */
} ReEpaResult;

/**
 * @brief Penetration depth, normal and witness points for two
 *        overlapping shapes (GJK containment followed by EPA).
 *
 * For shape pairs that merely touch or are separated the result is
 * ok == false. Degenerate inputs (zero radii, coincident centers)
 * terminate within the budgets; the normal defaults to (1, 0, 0).
 * @param a First shape.
 * @param b Second shape.
 */
ReEpaResult re_gjk_penetration(const ReConvexShape *a, const ReConvexShape *b);

/**
 * @brief Result of conservative-advancement CCD.
 */
typedef struct ReCcdResult {
    bool hit;       /**< True when a time of impact was found in [0, max_t]. */
    float toi;      /**< Time of impact (valid when hit). */
    ReVec3 normal;  /**< Contact normal at impact, from B to A. */
} ReCcdResult;

/**
 * @brief Time of impact via conservative advancement.
 *
 * Assumes constant linear velocities va and vb over [0, max_t];
 * advances time by distance / speed bound each step, which can never
 * tunnel through a collision (the bound is |va - vb|, conservative).
 * @param a First shape (at t = 0 pose).
 * @param b Second shape (at t = 0 pose).
 * @param va Linear velocity of A.
 * @param vb Linear velocity of B.
 * @param max_t Time horizon.
 */
ReCcdResult re_ccd_advance(const ReConvexShape *a, const ReConvexShape *b,
                           ReVec3 va, ReVec3 vb, float max_t);

/**
 * @brief Ray test against a single shape.
 *
 * Spheres, boxes and capsules use analytic tests; hulls use GJK-based
 * conservative advancement along the ray (exact up to RE_GJK_MAX_ITER
 * bisection steps).
 * @param s Shape.
 * @param ray Ray.
 * @param t_max Maximum distance.
 * @param t_out Receives the entry distance when hit.
 * @param normal_out Receives the surface normal at the hit (optional).
 * @return True on hit.
 */
bool re_shape_ray(const ReConvexShape *s, ReRay ray, float t_max,
                  float *t_out, ReVec3 *normal_out);

#endif /* RE_GJK_H */
