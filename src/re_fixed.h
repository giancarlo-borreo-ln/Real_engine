/**
 * @file re_fixed.h
 * @brief Deterministic Q32.32 fixed-point arithmetic.
 *
 * A re_fixed64_t stores the real value x * 2^32 as a signed 64-bit
 * integer, giving 32 fractional bits and a range of about
 * [-2^31, +2^31). Every operation is total: additions, subtractions and
 * multiplications saturate to RE_FIXED_MIN/RE_FIXED_MAX instead of
 * overflowing (which would be UB), and division by zero yields 0.
 * The saturation behavior is the documented determinism contract.
 *
 * Signed overflow is structurally impossible: all widening happens in
 * unsigned 128-bit space (re_i128.h) or through ckd_* checked
 * operations from <stdckdint.h>.
 */
#ifndef RE_FIXED_H
#define RE_FIXED_H

#include <stdbool.h>
#include <stdint.h>
#include <stdckdint.h>

#include "re_i128.h"

/** A Q32.32 fixed-point value (x * 2^32). */
typedef int64_t re_fixed64_t;

/** Fraction bits. */
#define RE_FIXED_SHIFT 32

/** The value 1.0 in Q32.32. */
#define RE_FIXED_ONE ((re_fixed64_t)INT64_C(0x100000000))

/** The value 0.5 in Q32.32. */
#define RE_FIXED_HALF ((re_fixed64_t)INT64_C(0x80000000))

/** Smallest representable positive value (1 ulp). */
#define RE_FIXED_EPS ((re_fixed64_t)1)

/** Smallest representable value. */
#define RE_FIXED_MIN INT64_MIN

/** Largest representable value. */
#define RE_FIXED_MAX INT64_MAX

/** pi in Q32.32 (floor(pi * 2^32) = 13493037704). */
#define RE_FIXED_PI ((re_fixed64_t)13493037704)

/** 2*pi in Q32.32 (floor(2 * pi * 2^32) = 26986075409). */
#define RE_FIXED_2PI ((re_fixed64_t)26986075409)

/** pi/2 in Q32.32 (floor(pi/2 * 2^32) = 6746518852). */
#define RE_FIXED_PI_2 ((re_fixed64_t)6746518852)

/**
 * @brief Converts an integer to Q32.32 with saturation.
 * @param v Integer value.
 */
re_fixed64_t re_fixed64_from_int(int64_t v);

/**
 * @brief Converts a float to Q32.32 with saturation.
 * @param v Float value (any value, including inf/nan).
 */
re_fixed64_t re_fixed64_from_f32(float v);

/**
 * @brief Converts Q32.32 to float (round to nearest).
 * @param v Fixed value.
 */
float re_fixed64_to_f32(re_fixed64_t v);

/**
 * @brief Saturating addition.
 * @param a First addend.
 * @param b Second addend.
 */
re_fixed64_t re_fixed64_add(re_fixed64_t a, re_fixed64_t b);

/**
 * @brief Saturating subtraction.
 * @param a Minuend.
 * @param b Subtrahend.
 */
re_fixed64_t re_fixed64_sub(re_fixed64_t a, re_fixed64_t b);

/**
 * @brief Saturating multiplication with full 128-bit intermediate.
 * @param a First factor.
 * @param b Second factor.
 */
re_fixed64_t re_fixed64_mul(re_fixed64_t a, re_fixed64_t b);

/**
 * @brief Saturating division with full 128-bit intermediate.
 *
 * Division by zero returns 0 (documented, deterministic).
 * @param a Dividend.
 * @param b Divisor.
 */
re_fixed64_t re_fixed64_div(re_fixed64_t a, re_fixed64_t b);

/**
 * @brief Floor of sqrt(a) in Q32.32.
 *
 * Negative inputs return 0 (documented, deterministic).
 * @param a Input value.
 */
re_fixed64_t re_fixed64_sqrt(re_fixed64_t a);

/**
 * @brief Sine of a Q32.32 angle in radians.
 *
 * Range reduction is exact integer arithmetic mod 2*pi (a remainder
 * operation on the Q32.32 grid); the polynomial evaluation is a
 * degree-11 odd series over the first octant with quadrant folding.
 * Maximum absolute error is below 3 ulp (~7e-10) for angles within a
 * few periods. Deterministic on every platform.
 * @param a Angle in Q32.32 radians (any value, even negative or huge;
 *          |a| beyond ~2^31 radians saturates the reduction).
 */
re_fixed64_t re_fixed64_sin(re_fixed64_t a);

/**
 * @brief Cosine of a Q32.32 angle in radians (see re_fixed64_sin).
 * @param a Angle in Q32.32 radians.
 */
re_fixed64_t re_fixed64_cos(re_fixed64_t a);

/**
 * @brief Absolute value (RE_FIXED_MIN maps to RE_FIXED_MAX).
 * @param a Input value.
 */
re_fixed64_t re_fixed64_abs(re_fixed64_t a);

/**
 * @brief Three-way comparison.
 * @param a First value.
 * @param b Second value.
 * @return Negative when a < b, 0 when equal, positive when a > b.
 */
int re_fixed64_cmp(re_fixed64_t a, re_fixed64_t b);

/**
 * @brief Clamps v into [lo, hi].
 * @param v Value.
 * @param lo Lower bound.
 * @param hi Upper bound.
 */
re_fixed64_t re_fixed64_clamp(re_fixed64_t v, re_fixed64_t lo, re_fixed64_t hi);

/**
 * @brief Fixed-point 3D vector (Q32.32 components).
 */
typedef struct ReVec3F {
    re_fixed64_t x; /**< X component. */
    re_fixed64_t y; /**< Y component. */
    re_fixed64_t z; /**< Z component. */
} ReVec3F;

/** @return Zero vector. */
ReVec3F re_vec3f_zero(void);

/** @return Vector with given components. */
ReVec3F re_vec3f_make(re_fixed64_t x, re_fixed64_t y, re_fixed64_t z);

/** @return Component-wise saturating sum. */
ReVec3F re_vec3f_add(ReVec3F a, ReVec3F b);

/** @return Component-wise saturating difference. */
ReVec3F re_vec3f_sub(ReVec3F a, ReVec3F b);

/** @return Vector scaled by s (saturating per component). */
ReVec3F re_vec3f_scale(ReVec3F v, re_fixed64_t s);

/** @return Negated vector. */
ReVec3F re_vec3f_neg(ReVec3F v);

/** @return Saturating dot product. */
re_fixed64_t re_vec3f_dot(ReVec3F a, ReVec3F b);

/** @return Cross product (saturating per component). */
ReVec3F re_vec3f_cross(ReVec3F a, ReVec3F b);

/** @return Saturating squared length (dot(v, v)). */
re_fixed64_t re_vec3f_length_sq(ReVec3F v);

/** @return Length (sqrt of squared length). */
re_fixed64_t re_vec3f_length(ReVec3F v);

/**
 * @brief Normalized vector.
 *
 * The zero vector normalizes to the zero vector (documented, avoids
 * division by zero entirely).
 * @param v Input vector.
 */
ReVec3F re_vec3f_normalize(ReVec3F v);

/**
 * @brief Fixed-point quaternion (Hamilton convention, x y z w).
 */
typedef struct ReQuatF {
    re_fixed64_t x; /**< X component. */
    re_fixed64_t y; /**< Y component. */
    re_fixed64_t z; /**< Z component. */
    re_fixed64_t w; /**< Scalar component. */
} ReQuatF;

/** @return Identity quaternion. */
ReQuatF re_quatf_identity(void);

/**
 * @brief Rotation quaternion from a normalized axis and an angle.
 *
 * The caller must pass a normalized axis; the function normalizes
 * defensively anyway (zero axis yields identity).
 * @param axis Rotation axis (will be normalized if needed).
 * @param angle Angle in Q32.32 radians.
 */
ReQuatF re_quatf_from_axis_angle(ReVec3F axis, re_fixed64_t angle);

/** @return Hamilton product a * b (saturating components). */
ReQuatF re_quatf_mul(ReQuatF a, ReQuatF b);

/** @return Normalized quaternion (zero quaternion maps to identity). */
ReQuatF re_quatf_normalize(ReQuatF a);

/** @return Vector rotated by the quaternion (unit quaternion assumed). */
ReVec3F re_quatf_rotate_vec3(ReQuatF q, ReVec3F v);

/**
 * @brief Fixed-point 4x4 matrix, column-major (m[col * 4 + row]).
 */
typedef struct ReMat4F {
    re_fixed64_t m[16]; /**< Column-major elements. */
} ReMat4F;

/** @return Identity matrix. */
ReMat4F re_mat4f_identity(void);

/** @return Matrix product a * b (saturating per element). */
ReMat4F re_mat4f_mul(ReMat4F a, ReMat4F b);

/** @return a transformed as a point (w = 1). */
ReVec3F re_mat4f_transform_point(ReMat4F m, ReVec3F p);

/** @return v transformed as a direction (w = 0). */
ReVec3F re_mat4f_transform_dir(ReMat4F m, ReVec3F v);

/**
 * @brief TRS matrix: translate * rotate * scale.
 * @param t Translation.
 * @param r Rotation (unit quaternion).
 * @param s Scale.
 */
ReMat4F re_mat4f_compose(ReVec3F t, ReQuatF r, ReVec3F s);

#endif /* RE_FIXED_H */
