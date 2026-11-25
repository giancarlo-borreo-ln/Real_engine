/**
 * @file re_vec.h
 * @brief Float linear algebra: vectors, matrices, quaternions, AABBs,
 *        rays and frustums.
 *
 * Matrices are column-major (m.col[0] is the first basis column), which
 * matches GLSL std140/std430 layout and the shader convention.
 *
 * Determinism contract: all functions use only IEEE-754 arithmetic with
 * no fused multiply-add; combined with -ffp-contract=off and the SSE2
 * baseline, results are bit-identical for the same binary. AVX2 paths
 * (enabled by compiling with RE_AVX2=1 / -mavx2) may differ from the
 * scalar paths in the last ulp of some functions; the scalar path is
 * always available and is the reference implementation.
 *
 * Structs are 16-byte aligned so the AVX2 paths can load whole
 * ReVec4 columns; AVX2 code uses unaligned 256-bit loads on those
 * 16-byte-aligned columns (a 256-bit aligned load would be UB).
 * sizeof(ReVec3) is therefore 16, not 12.
 */
#ifndef RE_VEC_H
#define RE_VEC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief 3D float vector (16-byte aligned; sizeof == 16).
 *
 * The alignment comes from the over-aligned first member: C23 grammar
 * places _Alignas among declaration specifiers only, so aligning the
 * leading member is the portable way to over-align a struct type.
 */
typedef struct ReVec3 {
    _Alignas(16) float x; /**< X component. */
    float y; /**< Y component. */
    float z; /**< Z component. */
} ReVec3;

/** Size of ReVec3 must be 16 bytes (alignment rounds it up). */
_Static_assert(sizeof(ReVec3) == 16, "ReVec3 must be 16 bytes");

/**
 * @brief 4D float vector (16-byte aligned; sizeof == 16).
 */
typedef struct ReVec4 {
    _Alignas(16) float x; /**< X component. */
    float y; /**< Y component. */
    float z; /**< Z component. */
    float w; /**< W component. */
} ReVec4;

/**
 * @brief 4x4 column-major float matrix (16-byte aligned; 64 bytes).
 *
 * Alignment follows from the over-aligned ReVec4 columns.
 */
typedef struct ReMat4 {
    ReVec4 col[4]; /**< Columns 0..3. */
} ReMat4;

/**
 * @brief Quaternion, Hamilton convention (x y z w), unit for rotations.
 */
typedef struct ReQuat {
    float x; /**< X component. */
    float y; /**< Y component. */
    float z; /**< Z component. */
    float w; /**< Scalar component. */
} ReQuat;

/**
 * @brief Axis-aligned bounding box.
 */
typedef struct ReAabb {
    ReVec3 min; /**< Minimum corner. */
    ReVec3 max; /**< Maximum corner. */
} ReAabb;

/**
 * @brief Ray with origin and direction (direction need not be normalized).
 */
typedef struct ReRay {
    ReVec3 origin; /**< Start point. */
    ReVec3 dir;    /**< Direction; t multiplies dir. */
} ReRay;

/**
 * @brief View frustum as six inward-facing planes.
 *
 * Plane i is ax + by + cz + d = 0 with the negative side outside;
 * a point (x,y,z) is inside when dot(plane.xyz, p) + plane.w >= 0
 * for all six planes. Order: left, right, bottom, top, near, far.
 */
typedef struct ReFrustum {
    ReVec4 planes[6]; /**< Plane equations (a, b, c, d). */
} ReFrustum;

/**
 * @brief Sphere shape data.
 */
typedef struct ReSphere {
    ReVec3 center; /**< Center point. */
    float radius;  /**< Radius. */
} ReSphere;

/* ------------------------------ ReVec3 ------------------------------ */

/** @return Zero vector. */
ReVec3 re_vec3_zero(void);

/** @return Vector with the given components. */
ReVec3 re_vec3_make(float x, float y, float z);

/** @return Component-wise sum. */
ReVec3 re_vec3_add(ReVec3 a, ReVec3 b);

/** @return Component-wise difference. */
ReVec3 re_vec3_sub(ReVec3 a, ReVec3 b);

/** @return Component-wise negation. */
ReVec3 re_vec3_neg(ReVec3 a);

/** @return Vector scaled by s. */
ReVec3 re_vec3_scale(ReVec3 v, float s);

/** @return a + b * s (one multiply, one add; no FMA, deterministic). */
ReVec3 re_vec3_add_scaled(ReVec3 a, ReVec3 b, float s);

/** @return Component-wise product. */
ReVec3 re_vec3_mul(ReVec3 a, ReVec3 b);

/** @return Dot product. */
float re_vec3_dot(ReVec3 a, ReVec3 b);

/** @return Cross product. */
ReVec3 re_vec3_cross(ReVec3 a, ReVec3 b);

/** @return Squared length. */
float re_vec3_length_sq(ReVec3 v);

/** @return Length. */
float re_vec3_length(ReVec3 v);

/** @return Normalized vector; the zero vector maps to zero. */
ReVec3 re_vec3_normalize(ReVec3 v);

/** @return Component-wise minimum. */
ReVec3 re_vec3_min(ReVec3 a, ReVec3 b);

/** @return Component-wise maximum. */
ReVec3 re_vec3_max(ReVec3 a, ReVec3 b);

/** @return Component-wise absolute value. */
ReVec3 re_vec3_abs(ReVec3 v);

/** @return Linear interpolation a + (b - a) * t. */
ReVec3 re_vec3_lerp(ReVec3 a, ReVec3 b, float t);

/** @return Distance between two points. */
float re_vec3_distance(ReVec3 a, ReVec3 b);

/** @return True when the component-wise difference is below eps. */
bool re_vec3_equal_eps(ReVec3 a, ReVec3 b, float eps);

/* ------------------------------ ReVec4 ------------------------------ */

/** @return Vector with the given components. */
ReVec4 re_vec4_make(float x, float y, float z, float w);

/** @return Dot product. */
float re_vec4_dot(ReVec4 a, ReVec4 b);

/** @return Component-wise sum. */
ReVec4 re_vec4_add(ReVec4 a, ReVec4 b);

/** @return Vector scaled by s. */
ReVec4 re_vec4_scale(ReVec4 v, float s);

/* ------------------------------ ReMat4 ------------------------------ */

/** @return Identity matrix. */
ReMat4 re_mat4_identity(void);

/** @return Matrix product a * b. */
ReMat4 re_mat4_mul(ReMat4 a, ReMat4 b);

/** @return Transposed matrix. */
ReMat4 re_mat4_transpose(ReMat4 m);

/**
 * @brief Right-handed perspective matrix with reverse Z.
 *
 * Maps view-space z in [near, far] to clip depth in [1, 0] (reverse Z,
 * far plane at 0), which matches Vulkan's depth range 0..1 and yields
 * the best float precision distribution.
 * @param fov_y Vertical field of view in radians.
 * @param aspect Width / height.
 * @param znear Near plane distance (> 0).
 * @param zfar Far plane distance (> znear; may be inf).
 */
ReMat4 re_mat4_perspective(float fov_y, float aspect, float znear, float zfar);

/** @return Orthographic matrix (right-handed, reverse Z). */
ReMat4 re_mat4_ortho(float left, float right, float bottom, float top, float znear, float zfar);

/**
 * @brief Right-handed view matrix looking from eye at target.
 * @param eye Camera position.
 * @param target Look-at point.
 * @param up Up direction (need not be normalized).
 */
ReMat4 re_mat4_look_at(ReVec3 eye, ReVec3 target, ReVec3 up);

/** @return Pure translation matrix. */
ReMat4 re_mat4_translate(ReVec3 t);

/** @return Pure rotation matrix around a normalized axis. */
ReMat4 re_mat4_rotate_axis(ReVec3 axis, float angle);

/** @return Pure scale matrix. */
ReMat4 re_mat4_scale(ReVec3 s);

/**
 * @brief TRS matrix: translate * rotate * scale.
 * @param t Translation.
 * @param r Rotation (unit quaternion).
 * @param s Scale.
 */
ReMat4 re_mat4_compose(ReVec3 t, ReQuat r, ReVec3 s);

/** @return Full inverse via Gauss-Jordan (any invertible matrix). */
ReMat4 re_mat4_inverse(ReMat4 m);

/** @return Fast inverse for affine (rotation/scale/translation) matrices. */
ReMat4 re_mat4_affine_inverse(ReMat4 m);

/** @return p transformed as a point (w = 1, perspective divide applied). */
ReVec3 re_mat4_transform_point(ReMat4 m, ReVec3 p);

/** @return v transformed as a direction (w = 0). */
ReVec3 re_mat4_transform_dir(ReMat4 m, ReVec3 v);

/** @return v transformed as a 4D point (w = 1), no divide. */
ReVec4 re_mat4_transform_vec4(ReMat4 m, ReVec4 v);

/** @return Matrix built from a unit quaternion. */
ReMat4 re_mat4_from_quat(ReQuat q);

/* ------------------------------ ReQuat ------------------------------ */

/** @return Identity quaternion. */
ReQuat re_quat_identity(void);

/** @return Hamilton product a * b. */
ReQuat re_quat_mul(ReQuat a, ReQuat b);

/** @return Normalized quaternion; the zero quaternion maps to identity. */
ReQuat re_quat_normalize(ReQuat q);

/** @return Conjugate. */
ReQuat re_quat_conjugate(ReQuat q);

/**
 * @brief Rotation quaternion from a normalized axis and an angle.
 * @param axis Axis (normalized by caller; defensively re-normalized).
 * @param angle Angle in radians.
 */
ReQuat re_quat_from_axis_angle(ReVec3 axis, float angle);

/** @return Vector rotated by the unit quaternion. */
ReVec3 re_quat_rotate_vec3(ReQuat q, ReVec3 v);

/** @return Rotation matrix from the unit quaternion. */
ReMat4 re_quat_to_mat4(ReQuat q);

/** @return Normalized linear interpolation (nlerp). */
ReQuat re_quat_nlerp(ReQuat a, ReQuat b, float t);

/** @return Spherical linear interpolation (slerp). */
ReQuat re_quat_slerp(ReQuat a, ReQuat b, float t);

/* ------------------------------ ReAabb ------------------------------ */

/** @return Empty AABB (min = +inf, max = -inf). */
ReAabb re_aabb_empty(void);

/** @return AABB containing a single point. */
ReAabb re_aabb_from_point(ReVec3 p);

/** @return AABB containing both points (p and q may be in any order). */
ReAabb re_aabb_from_two_points(ReVec3 p, ReVec3 q);

/** @return True when the AABB is empty (min > max on any axis). */
bool re_aabb_is_empty(ReAabb b);

/** @return Union of the two boxes. */
ReAabb re_aabb_merge(ReAabb a, ReAabb b);

/** @return Box expanded by e on every side. */
ReAabb re_aabb_expand(ReAabb b, float e);

/** @return Center point. */
ReVec3 re_aabb_center(ReAabb b);

/** @return Half extents ((max - min) / 2). */
ReVec3 re_aabb_extents(ReAabb b);

/** @return Surface area (0 for empty boxes). */
float re_aabb_surface_area(ReAabb b);

/** @return True when p is inside (inclusive) the box. */
bool re_aabb_contains_point(ReAabb b, ReVec3 p);

/** @return True when inner is fully inside outer (inclusive). */
bool re_aabb_contains_aabb(ReAabb outer, ReAabb inner);

/** @return True when the boxes overlap or touch. */
bool re_aabb_intersects(ReAabb a, ReAabb b);

/**
 * @brief Slab test of a ray against the box.
 * @param b Box.
 * @param ray Ray.
 * @param t_out Receives the entry distance along the ray when hit.
 * @return True when the ray hits the box at t >= 0 and within t_max.
 */
bool re_aabb_ray(ReAabb b, ReRay ray, float t_max, float *t_out);

/**
 * @brief Transformed box (Arvo's conservative method).
 * @param b Box.
 * @param m Affine transform (no projection).
 */
ReAabb re_aabb_transform(ReAabb b, ReMat4 m);

/* ------------------------------ ReRay ------------------------------- */

/** @return Point at distance t along the ray. */
ReVec3 re_ray_point_at(ReRay ray, float t);

/* ----------------------------- ReFrustum ---------------------------- */

/**
 * @brief Extracts the six frustum planes from a clip-space matrix
 *        (Gribb-Hartmann). Pass proj * view.
 * @param vp Combined view-projection matrix.
 */
ReFrustum re_frustum_from_mat4(ReMat4 vp);

/** @return True when the point is inside or on the frustum. */
bool re_frustum_contains_point(ReFrustum f, ReVec3 p);

/** @return True when the sphere intersects the frustum. */
bool re_frustum_contains_sphere(ReFrustum f, ReSphere s);

/**
 * @brief Conservative AABB-vs-frustum test.
 * @param f Frustum.
 * @param b Box.
 * @return RE_FRUSTUM_OUTSIDE, RE_FRUSTUM_INTERSECT or RE_FRUSTUM_INSIDE.
 */
int re_frustum_test_aabb(ReFrustum f, ReAabb b);

/** AABB fully outside the frustum. */
#define RE_FRUSTUM_OUTSIDE 0

/** AABB partially inside the frustum. */
#define RE_FRUSTUM_INTERSECT 1

/** AABB fully inside the frustum. */
#define RE_FRUSTUM_INSIDE 2

/* --------------------------- Fixed conversion ----------------------- */

/**
 * @brief Converts a Q32.32 fixed value to float (exact rounding).
 * @param v Fixed value.
 */
float re_fixed_to_f32(int64_t v);

/**
 * @brief Converts a float to Q32.32 fixed with saturation.
 * @param v Float value.
 */
int64_t re_f32_to_fixed(float v);

#endif /* RE_VEC_H */
