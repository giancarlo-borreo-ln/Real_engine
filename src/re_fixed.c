/**
 * @file re_fixed.c
 * @brief Deterministic Q32.32 fixed-point arithmetic and fixed 3D math.
 *
 * Real Engine - C23 data-oriented game engine.
 *
 * Every operation is total: no signed overflow can occur because all
 * widening goes through <stdckdint.h> checked arithmetic or through the
 * unsigned 128-bit helpers in re_i128.h. Out-of-range results saturate to
 * RE_FIXED_MIN / RE_FIXED_MAX, which is the documented determinism
 * contract (see docs/ABI.md).
 */
#include "re_fixed.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdckdint.h>

/** 2^63 as an unsigned magnitude: |INT64_MIN|. */
#define RE_U64_SIGN_BIT UINT64_C(0x8000000000000000)

/** The value 2.0 in Q32.32. */
#define RE_FIXED_TWO ((re_fixed64_t)INT64_C(0x200000000))

/**
 * Quarter turn used for range reduction: nearest Q32.32 value of pi/2
 * (pi/2 * 2^32 = 6746518852.26). RE_FIXED_PI_2 in re_fixed.h is 7 ulp
 * larger, which would add up to 21 ulp of argument error while folding.
 */
#define RE_FIXED_QTR_TURN ((re_fixed64_t)INT64_C(6746518852))

/**
 * Full turn used for range reduction: nearest Q32.32 value of 2*pi
 * (2*pi * 2^32 = 26986075409.04). RE_FIXED_2PI in re_fixed.h is 27 ulp
 * larger, i.e. ~6e-9 rad of drift per reduced period.
 */
#define RE_FIXED_TURN ((re_fixed64_t)INT64_C(26986075409))

/* ------------------------- internal helpers ------------------------- */

/**
 * @brief Absolute value of a signed 64-bit integer as an unsigned magnitude.
 *
 * The unsigned negation wraps, so INT64_MIN maps to 2^63 instead of
 * overflowing.
 * @param x Input value.
 * @return |x| as an unsigned 64-bit magnitude.
 */
static uint64_t fixed_mag(int64_t x) {
    uint64_t u = (uint64_t)x;
    return x < 0 ? UINT64_C(0) - u : u;
}

/**
 * @brief Re-attaches a sign to a magnitude and saturates outside Q32.32.
 * @param magnitude Unsigned magnitude.
 * @param negative True when the result must be negative.
 * @return Signed value, saturated to RE_FIXED_MIN / RE_FIXED_MAX.
 */
static re_fixed64_t fixed_apply_sign(uint64_t magnitude, bool negative) {
    if (magnitude >= RE_U64_SIGN_BIT) {
        /* -2^63 is representable, anything larger is not. */
        return negative ? RE_FIXED_MIN : RE_FIXED_MAX;
    }
    return negative ? -(re_fixed64_t)magnitude : (re_fixed64_t)magnitude;
}

/**
 * @brief Floor of sqrt(value * 2^32) for a non-negative Q32.32 input.
 *
 * Digit-by-digit square root over the 96-bit radicand value * 2^32: the
 * 32 upper bit pairs come from value, the 16 lower pairs are zero. The
 * running remainder stays below 2^51 and the root below 2^48, so uint64_t
 * is enough and no intermediate can overflow.
 *
 * @param value Non-negative Q32.32 value.
 * @return floor(sqrt(real_value) * 2^32).
 */
static re_fixed64_t fixed_sqrt_nonneg(uint64_t value) {
    uint64_t root = 0u;
    uint64_t rem = 0u;
    for (int i = 47; i >= 0; i--) {
        if (i >= 16) {
            unsigned shift = (unsigned)(2 * (i - 16));
            rem = (rem << 2u) | ((value >> shift) & 3u);
        } else {
            rem <<= 2u;
        }
        uint64_t trial = (root << 2u) | 1u;
        if (trial <= rem) {
            rem -= trial;
            root = (root << 1u) | 1u;
        } else {
            root <<= 1u;
        }
    }
    return (re_fixed64_t)root;
}

/**
 * @brief sin(u) for u in [0, pi/4], degree-11 odd Taylor series.
 * @param u Angle in Q32.32 radians.
 * @return sin(u) with a few ulp of truncation error.
 */
static re_fixed64_t fixed_sin_series(re_fixed64_t u) {
    re_fixed64_t s = re_fixed64_mul(u, u);
    re_fixed64_t p = -108;                                                        /* -1/11! */
    p = re_fixed64_add(11836, re_fixed64_mul(p, s));                               /*  1/9!  */
    p = re_fixed64_add(-852176, re_fixed64_mul(p, s));                             /* -1/7!  */
    p = re_fixed64_add(35791394, re_fixed64_mul(p, s));                            /*  1/5!  */
    p = re_fixed64_add(-715827883, re_fixed64_mul(p, s));                          /* -1/3!  */
    return re_fixed64_mul(u, re_fixed64_add(RE_FIXED_ONE, re_fixed64_mul(s, p)));
}

/**
 * @brief cos(u) for u in [0, pi/4], degree-12 even Taylor series.
 * @param u Angle in Q32.32 radians.
 * @return cos(u) with a few ulp of truncation error.
 */
static re_fixed64_t fixed_cos_series(re_fixed64_t u) {
    re_fixed64_t s = re_fixed64_mul(u, u);
    re_fixed64_t p = 9;                                                           /*  1/12! */
    p = re_fixed64_add(-1184, re_fixed64_mul(p, s));                               /* -1/10! */
    p = re_fixed64_add(106522, re_fixed64_mul(p, s));                              /*  1/8!  */
    p = re_fixed64_add(-5965232, re_fixed64_mul(p, s));                            /* -1/6!  */
    p = re_fixed64_add(178956971, re_fixed64_mul(p, s));                           /*  1/4!  */
    p = re_fixed64_add(-2147483648, re_fixed64_mul(p, s));                         /* -1/2!  */
    return re_fixed64_add(RE_FIXED_ONE, re_fixed64_mul(s, p));
}

/**
 * @brief Row `row` of a column-major matrix dotted with (p, w).
 * @param m Matrix.
 * @param row Row index in [0, 4).
 * @param p Vector operand.
 * @param w Fourth operand (1 for points, 0 for directions).
 * @return Saturating dot product.
 */
static re_fixed64_t matrix_row_dot(const ReMat4F *m, unsigned row, ReVec3F p, re_fixed64_t w) {
    re_fixed64_t acc = re_fixed64_mul(m->m[row], p.x);
    acc = re_fixed64_add(acc, re_fixed64_mul(m->m[4u + row], p.y));
    acc = re_fixed64_add(acc, re_fixed64_mul(m->m[8u + row], p.z));
    acc = re_fixed64_add(acc, re_fixed64_mul(m->m[12u + row], w));
    return acc;
}

/**
 * @brief Column-major rotation matrix (rot[col * 3 + row]) of a quaternion.
 * @param q Quaternion (unit quaternion assumed).
 * @param rot Output array of 9 elements.
 */
static void quat_rotation_matrix(ReQuatF q, re_fixed64_t *rot) {
    re_fixed64_t xx = re_fixed64_mul(q.x, q.x);
    re_fixed64_t yy = re_fixed64_mul(q.y, q.y);
    re_fixed64_t zz = re_fixed64_mul(q.z, q.z);
    re_fixed64_t xy = re_fixed64_mul(q.x, q.y);
    re_fixed64_t xz = re_fixed64_mul(q.x, q.z);
    re_fixed64_t yz = re_fixed64_mul(q.y, q.z);
    re_fixed64_t wx = re_fixed64_mul(q.w, q.x);
    re_fixed64_t wy = re_fixed64_mul(q.w, q.y);
    re_fixed64_t wz = re_fixed64_mul(q.w, q.z);
    rot[0] = re_fixed64_sub(RE_FIXED_ONE, re_fixed64_mul(RE_FIXED_TWO, re_fixed64_add(yy, zz)));
    rot[1] = re_fixed64_mul(RE_FIXED_TWO, re_fixed64_add(xy, wz));
    rot[2] = re_fixed64_mul(RE_FIXED_TWO, re_fixed64_sub(xz, wy));
    rot[3] = re_fixed64_mul(RE_FIXED_TWO, re_fixed64_sub(xy, wz));
    rot[4] = re_fixed64_sub(RE_FIXED_ONE, re_fixed64_mul(RE_FIXED_TWO, re_fixed64_add(xx, zz)));
    rot[5] = re_fixed64_mul(RE_FIXED_TWO, re_fixed64_add(yz, wx));
    rot[6] = re_fixed64_mul(RE_FIXED_TWO, re_fixed64_add(xz, wy));
    rot[7] = re_fixed64_mul(RE_FIXED_TWO, re_fixed64_sub(yz, wx));
    rot[8] = re_fixed64_sub(RE_FIXED_ONE, re_fixed64_mul(RE_FIXED_TWO, re_fixed64_add(xx, yy)));
}

/**
 * @brief Squared length of the four quaternion components.
 * @param a Quaternion.
 * @return x^2 + y^2 + z^2 + w^2, saturating.
 */
static re_fixed64_t quatf_length_sq(ReQuatF a) {
    re_fixed64_t acc = re_fixed64_mul(a.x, a.x);
    acc = re_fixed64_add(acc, re_fixed64_mul(a.y, a.y));
    acc = re_fixed64_add(acc, re_fixed64_mul(a.z, a.z));
    acc = re_fixed64_add(acc, re_fixed64_mul(a.w, a.w));
    return acc;
}

/* ----------------------- scalar Q32.32 math ------------------------ */

/**
 * @brief Converts an integer to Q32.32 with saturation (see re_fixed.h).
 * @param v Integer value.
 */
re_fixed64_t re_fixed64_from_int(int64_t v) {
    /* INT64_MIN / 2^32 is exactly -2^31, INT64_MAX / 2^32 is 2^31 - 1. */
    if (v > RE_FIXED_MAX / RE_FIXED_ONE) {
        return RE_FIXED_MAX;
    }
    if (v < RE_FIXED_MIN / RE_FIXED_ONE) {
        return RE_FIXED_MIN;
    }
    /* |v| <= 2^31, so v * 2^32 is representable: no overflow. */
    return (re_fixed64_t)(v * RE_FIXED_ONE);
}

/**
 * @brief Converts a float to Q32.32 with saturation (see re_fixed.h).
 *
 * The scaled value is clamped as a double before the cast: casting an
 * out-of-range double to int64_t is undefined, and 2^31 itself is out of
 * range once multiplied by 2^32.
 * @param v Float value (any value, including inf/nan).
 */
re_fixed64_t re_fixed64_from_f32(float v) {
    if (isnan(v)) {
        return 0;
    }
    double d = (double)v * 4294967296.0;
    if (d >= 9223372036854775808.0) { /* 2^63 */
        return RE_FIXED_MAX;
    }
    if (d < -9223372036854775808.0) {
        return RE_FIXED_MIN;
    }
    return (re_fixed64_t)d;
}

/**
 * @brief Converts Q32.32 to float (see re_fixed.h).
 * @param v Fixed value.
 */
float re_fixed64_to_f32(re_fixed64_t v) {
    return (float)((double)v / 4294967296.0);
}

/**
 * @brief Saturating addition (see re_fixed.h).
 * @param a First addend.
 * @param b Second addend.
 */
re_fixed64_t re_fixed64_add(re_fixed64_t a, re_fixed64_t b) {
    re_fixed64_t r;
    if (!ckd_add(&r, a, b)) {
        return r;
    }
    /* Only same-sign operands can overflow: both negative or both positive. */
    return a < 0 ? RE_FIXED_MIN : RE_FIXED_MAX;
}

/**
 * @brief Saturating subtraction (see re_fixed.h).
 * @param a Minuend.
 * @param b Subtrahend.
 */
re_fixed64_t re_fixed64_sub(re_fixed64_t a, re_fixed64_t b) {
    re_fixed64_t r;
    if (!ckd_sub(&r, a, b)) {
        return r;
    }
    /* a >= 0, b < 0 overflows high; a < 0, b > 0 overflows low. */
    return a < 0 ? RE_FIXED_MIN : RE_FIXED_MAX;
}

/**
 * @brief Saturating multiplication with full 128-bit intermediate.
 * @param a First factor.
 * @param b Second factor.
 */
re_fixed64_t re_fixed64_mul(re_fixed64_t a, re_fixed64_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    bool negative = (a < 0) != (b < 0);
    ReU128 product = re_u128_mul64(fixed_mag(a), fixed_mag(b));
    ReU128 shifted = re_u128_shr(product, RE_FIXED_SHIFT);
    if (shifted.hi != 0u) {
        return negative ? RE_FIXED_MIN : RE_FIXED_MAX;
    }
    return fixed_apply_sign(shifted.lo, negative);
}

/**
 * @brief Saturating division with full 128-bit intermediate.
 *
 * Division by zero returns 0.
 * @param a Dividend.
 * @param b Divisor.
 */
re_fixed64_t re_fixed64_div(re_fixed64_t a, re_fixed64_t b) {
    if (b == 0) {
        return 0;
    }
    bool negative = (a < 0) != (b < 0);
    uint64_t ua = fixed_mag(a);
    uint64_t ub = fixed_mag(b);
    if (ub == RE_U64_SIGN_BIT) {
        /* re_u128_div64 shifts by 64 when the divisor's leading bit is set
           (it computes u.lo >> (64 - clz)), which is UB for exactly 2^63.
           Divide by 2^63 directly: (ua * 2^32) / 2^63 == ua >> 31. */
        return fixed_apply_sign(ua >> 31u, negative);
    }
    ReU128 numerator = re_u128_shl((ReU128){ua, 0u}, RE_FIXED_SHIFT);
    if (numerator.hi >= ub) {
        /* The quotient would not fit in 64 bits. */
        return negative ? RE_FIXED_MIN : RE_FIXED_MAX;
    }
    uint64_t quotient = re_u128_div64(numerator, ub, NULL);
    return fixed_apply_sign(quotient, negative);
}

/**
 * @brief Floor of sqrt(a) in Q32.32 (see re_fixed.h).
 * @param a Input value.
 */
re_fixed64_t re_fixed64_sqrt(re_fixed64_t a) {
    if (a < 0) {
        return 0;
    }
    return fixed_sqrt_nonneg((uint64_t)a);
}

/**
 * @brief Sine of a Q32.32 angle (see re_fixed.h).
 *
 * Reduction modulo 2*pi is exact integer arithmetic on the Q32.32 grid,
 * folding is done against an internal quarter-turn constant so that the
 * quadrant boundaries are exactly representable, and the series are
 * evaluated on [0, pi/4] where their truncation error stays below two ulp.
 * sin(RE_FIXED_PI_2) therefore evaluates to exactly 1.0. For |a| beyond
 * about 1e6 radians the reduction inherits the 2*pi constant's error.
 * @param a Angle in Q32.32 radians.
 */
re_fixed64_t re_fixed64_sin(re_fixed64_t a) {
    re_fixed64_t t = a % RE_FIXED_TURN; /* (-2*pi, 2*pi), exact */
    bool negative = t < 0;
    if (negative) {
        t = -t; /* |t| < 2*pi*2^32, no overflow */
    }
    re_fixed64_t quadrant = t / RE_FIXED_QTR_TURN; /* 0..3, 4 for t == TURN - 1 */
    re_fixed64_t r = t - quadrant * RE_FIXED_QTR_TURN;
    if (quadrant == 4) {
        /* 4 * QTR_TURN == TURN - 1: the last representable angle of the
           period is exactly 0 in this fold (r is already 0). */
        quadrant = 0;
    }
    bool swap = r > RE_FIXED_QTR_TURN / 2;
    re_fixed64_t u = swap ? RE_FIXED_QTR_TURN - r : r;
    re_fixed64_t s = swap ? fixed_cos_series(u) : fixed_sin_series(u);
    re_fixed64_t c = swap ? fixed_sin_series(u) : fixed_cos_series(u);
    re_fixed64_t result = s;
    if (quadrant == 1) {
        result = c;
    } else if (quadrant == 2) {
        result = -s;
    } else if (quadrant == 3) {
        result = -c;
    }
    return negative ? -result : result;
}

/**
 * @brief Cosine of a Q32.32 angle (see re_fixed.h).
 *
 * cos(a) == sin(a + pi/2); the internal quarter-turn constant is used so
 * that the shift itself does not inject the header constant's 7 ulp error.
 * @param a Angle in Q32.32 radians.
 */
re_fixed64_t re_fixed64_cos(re_fixed64_t a) {
    return re_fixed64_sin(re_fixed64_add(a, RE_FIXED_QTR_TURN));
}

/**
 * @brief Absolute value (see re_fixed.h).
 * @param a Input value.
 */
re_fixed64_t re_fixed64_abs(re_fixed64_t a) {
    if (a >= 0) {
        return a;
    }
    return a == RE_FIXED_MIN ? RE_FIXED_MAX : -a;
}

/**
 * @brief Three-way comparison (see re_fixed.h).
 * @param a First value.
 * @param b Second value.
 */
int re_fixed64_cmp(re_fixed64_t a, re_fixed64_t b) {
    return (a > b) - (a < b);
}

/**
 * @brief Clamps v into [lo, hi] (see re_fixed.h).
 * @param v Value.
 * @param lo Lower bound.
 * @param hi Upper bound.
 */
re_fixed64_t re_fixed64_clamp(re_fixed64_t v, re_fixed64_t lo, re_fixed64_t hi) {
    if (v < lo) {
        return lo;
    }
    return v > hi ? hi : v;
}

/* --------------------------- ReVec3F ------------------------------- */

/**
 * @brief Zero vector (see re_fixed.h).
 */
ReVec3F re_vec3f_zero(void) {
    ReVec3F v = {0, 0, 0};
    return v;
}

/**
 * @brief Vector with the given components (see re_fixed.h).
 * @param x X component.
 * @param y Y component.
 * @param z Z component.
 */
ReVec3F re_vec3f_make(re_fixed64_t x, re_fixed64_t y, re_fixed64_t z) {
    ReVec3F v = {x, y, z};
    return v;
}

/**
 * @brief Component-wise saturating sum (see re_fixed.h).
 * @param a First vector.
 * @param b Second vector.
 */
ReVec3F re_vec3f_add(ReVec3F a, ReVec3F b) {
    ReVec3F r = {re_fixed64_add(a.x, b.x), re_fixed64_add(a.y, b.y), re_fixed64_add(a.z, b.z)};
    return r;
}

/**
 * @brief Component-wise saturating difference (see re_fixed.h).
 * @param a First vector.
 * @param b Second vector.
 */
ReVec3F re_vec3f_sub(ReVec3F a, ReVec3F b) {
    ReVec3F r = {re_fixed64_sub(a.x, b.x), re_fixed64_sub(a.y, b.y), re_fixed64_sub(a.z, b.z)};
    return r;
}

/**
 * @brief Vector scaled by s (see re_fixed.h).
 * @param v Input vector.
 * @param s Scale factor.
 */
ReVec3F re_vec3f_scale(ReVec3F v, re_fixed64_t s) {
    ReVec3F r = {re_fixed64_mul(v.x, s), re_fixed64_mul(v.y, s), re_fixed64_mul(v.z, s)};
    return r;
}

/**
 * @brief Negated vector (see re_fixed.h).
 * @param v Input vector.
 */
ReVec3F re_vec3f_neg(ReVec3F v) {
    ReVec3F r = {re_fixed64_sub(0, v.x), re_fixed64_sub(0, v.y), re_fixed64_sub(0, v.z)};
    return r;
}

/**
 * @brief Saturating dot product (see re_fixed.h).
 * @param a First vector.
 * @param b Second vector.
 */
re_fixed64_t re_vec3f_dot(ReVec3F a, ReVec3F b) {
    re_fixed64_t acc = re_fixed64_mul(a.x, b.x);
    acc = re_fixed64_add(acc, re_fixed64_mul(a.y, b.y));
    acc = re_fixed64_add(acc, re_fixed64_mul(a.z, b.z));
    return acc;
}

/**
 * @brief Cross product (see re_fixed.h).
 * @param a First vector.
 * @param b Second vector.
 */
ReVec3F re_vec3f_cross(ReVec3F a, ReVec3F b) {
    ReVec3F r;
    r.x = re_fixed64_sub(re_fixed64_mul(a.y, b.z), re_fixed64_mul(a.z, b.y));
    r.y = re_fixed64_sub(re_fixed64_mul(a.z, b.x), re_fixed64_mul(a.x, b.z));
    r.z = re_fixed64_sub(re_fixed64_mul(a.x, b.y), re_fixed64_mul(a.y, b.x));
    return r;
}

/**
 * @brief Saturating squared length (see re_fixed.h).
 * @param v Input vector.
 */
re_fixed64_t re_vec3f_length_sq(ReVec3F v) {
    return re_vec3f_dot(v, v);
}

/**
 * @brief Length (see re_fixed.h).
 * @param v Input vector.
 */
re_fixed64_t re_vec3f_length(ReVec3F v) {
    return re_fixed64_sqrt(re_vec3f_length_sq(v));
}

/**
 * @brief Normalized vector; the zero vector stays zero (see re_fixed.h).
 * @param v Input vector.
 */
ReVec3F re_vec3f_normalize(ReVec3F v) {
    re_fixed64_t len = re_vec3f_length(v);
    if (len <= 0) {
        return re_vec3f_zero();
    }
    return re_vec3f_scale(v, re_fixed64_div(RE_FIXED_ONE, len));
}

/* --------------------------- ReQuatF ------------------------------- */

/**
 * @brief Identity quaternion (see re_fixed.h).
 */
ReQuatF re_quatf_identity(void) {
    ReQuatF q = {0, 0, 0, RE_FIXED_ONE};
    return q;
}

/**
 * @brief Rotation quaternion from an axis and an angle (see re_fixed.h).
 * @param axis Rotation axis (normalized defensively).
 * @param angle Angle in Q32.32 radians.
 */
ReQuatF re_quatf_from_axis_angle(ReVec3F axis, re_fixed64_t angle) {
    ReVec3F n = re_vec3f_normalize(axis);
    if (n.x == 0 && n.y == 0 && n.z == 0) {
        return re_quatf_identity();
    }
    re_fixed64_t half = re_fixed64_div(angle, RE_FIXED_TWO);
    re_fixed64_t s = re_fixed64_sin(half);
    re_fixed64_t c = re_fixed64_cos(half);
    ReQuatF q = {re_fixed64_mul(n.x, s), re_fixed64_mul(n.y, s), re_fixed64_mul(n.z, s), c};
    return q;
}

/**
 * @brief Hamilton product a * b (see re_fixed.h).
 * @param a Left factor.
 * @param b Right factor.
 */
ReQuatF re_quatf_mul(ReQuatF a, ReQuatF b) {
    ReQuatF q;
    q.w = re_fixed64_sub(re_fixed64_mul(a.w, b.w), re_fixed64_mul(a.x, b.x));
    q.w = re_fixed64_sub(q.w, re_fixed64_mul(a.y, b.y));
    q.w = re_fixed64_sub(q.w, re_fixed64_mul(a.z, b.z));
    q.x = re_fixed64_add(re_fixed64_mul(a.w, b.x), re_fixed64_mul(a.x, b.w));
    q.x = re_fixed64_sub(q.x, re_fixed64_mul(a.z, b.y));
    q.x = re_fixed64_add(q.x, re_fixed64_mul(a.y, b.z));
    q.y = re_fixed64_add(re_fixed64_mul(a.w, b.y), re_fixed64_mul(a.y, b.w));
    q.y = re_fixed64_sub(q.y, re_fixed64_mul(a.x, b.z));
    q.y = re_fixed64_add(q.y, re_fixed64_mul(a.z, b.x));
    q.z = re_fixed64_add(re_fixed64_mul(a.w, b.z), re_fixed64_mul(a.z, b.w));
    q.z = re_fixed64_add(q.z, re_fixed64_mul(a.x, b.y));
    q.z = re_fixed64_sub(q.z, re_fixed64_mul(a.y, b.x));
    return q;
}

/**
 * @brief Normalized quaternion; zero maps to identity (see re_fixed.h).
 * @param a Input quaternion.
 */
ReQuatF re_quatf_normalize(ReQuatF a) {
    re_fixed64_t len = re_fixed64_sqrt(quatf_length_sq(a));
    if (len <= 0) {
        return re_quatf_identity();
    }
    re_fixed64_t inv = re_fixed64_div(RE_FIXED_ONE, len);
    ReQuatF q = {re_fixed64_mul(a.x, inv), re_fixed64_mul(a.y, inv),
                 re_fixed64_mul(a.z, inv), re_fixed64_mul(a.w, inv)};
    return q;
}

/**
 * @brief Vector rotated by a unit quaternion (see re_fixed.h).
 *
 * v' = v + 2 * (q.xyz x (q.xyz x v + q.w * v)), which needs only mul,
 * add and cross.
 * @param q Unit quaternion.
 * @param v Vector to rotate.
 */
ReVec3F re_quatf_rotate_vec3(ReQuatF q, ReVec3F v) {
    ReVec3F axis = {q.x, q.y, q.z};
    ReVec3F inner = re_vec3f_add(re_vec3f_cross(axis, v), re_vec3f_scale(v, q.w));
    ReVec3F twist = re_vec3f_scale(re_vec3f_cross(axis, inner), RE_FIXED_TWO);
    return re_vec3f_add(v, twist);
}

/* --------------------------- ReMat4F ------------------------------- */

/**
 * @brief Identity matrix (see re_fixed.h).
 */
ReMat4F re_mat4f_identity(void) {
    ReMat4F m = {{RE_FIXED_ONE, 0, 0, 0,
                  0, RE_FIXED_ONE, 0, 0,
                  0, 0, RE_FIXED_ONE, 0,
                  0, 0, 0, RE_FIXED_ONE}};
    return m;
}

/**
 * @brief Matrix product a * b, column-major (see re_fixed.h).
 * @param a Left factor.
 * @param b Right factor.
 */
ReMat4F re_mat4f_mul(ReMat4F a, ReMat4F b) {
    ReMat4F out;
    for (unsigned c = 0; c < 4u; c++) {
        for (unsigned r = 0; r < 4u; r++) {
            re_fixed64_t sum = re_fixed64_mul(a.m[r], b.m[c * 4u]);
            for (unsigned k = 1u; k < 4u; k++) {
                sum = re_fixed64_add(sum, re_fixed64_mul(a.m[k * 4u + r], b.m[c * 4u + k]));
            }
            out.m[c * 4u + r] = sum;
        }
    }
    return out;
}

/**
 * @brief Transforms p as a point, w = 1 (see re_fixed.h).
 * @param m Matrix.
 * @param p Point.
 */
ReVec3F re_mat4f_transform_point(ReMat4F m, ReVec3F p) {
    ReVec3F r = {matrix_row_dot(&m, 0u, p, RE_FIXED_ONE),
                 matrix_row_dot(&m, 1u, p, RE_FIXED_ONE),
                 matrix_row_dot(&m, 2u, p, RE_FIXED_ONE)};
    return r;
}

/**
 * @brief Transforms v as a direction, w = 0 (see re_fixed.h).
 * @param m Matrix.
 * @param v Direction.
 */
ReVec3F re_mat4f_transform_dir(ReMat4F m, ReVec3F v) {
    ReVec3F r = {matrix_row_dot(&m, 0u, v, 0), matrix_row_dot(&m, 1u, v, 0),
                 matrix_row_dot(&m, 2u, v, 0)};
    return r;
}

/**
 * @brief TRS matrix: translate * rotate * scale (see re_fixed.h).
 * @param t Translation.
 * @param r Rotation (unit quaternion).
 * @param s Scale.
 */
ReMat4F re_mat4f_compose(ReVec3F t, ReQuatF r, ReVec3F s) {
    re_fixed64_t rot[9];
    quat_rotation_matrix(r, rot);
    /* m[col * 4 + row]: columns 0..2 are the rotation columns scaled by
       s.x/s.y/s.z, column 3 is the translation. */
    ReMat4F out = {{re_fixed64_mul(rot[0], s.x), re_fixed64_mul(rot[1], s.x),
                    re_fixed64_mul(rot[2], s.x), 0,
                    re_fixed64_mul(rot[3], s.y), re_fixed64_mul(rot[4], s.y),
                    re_fixed64_mul(rot[5], s.y), 0,
                    re_fixed64_mul(rot[6], s.z), re_fixed64_mul(rot[7], s.z),
                    re_fixed64_mul(rot[8], s.z), 0,
                    t.x, t.y, t.z, RE_FIXED_ONE}};
    return out;
}
