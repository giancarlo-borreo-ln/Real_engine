/**
 * @file re_i128.h
 * @brief Portable unsigned 128-bit arithmetic built from 64-bit halves.
 *
 * Pure ISO C23: no __int128, no compiler extensions. Every operation is
 * total (no undefined behavior for any input), which is what makes the
 * fixed-point subsystem and its CBMC/Frama-C proofs possible.
 *
 * Functions are static inline so the compiler sees through them; CBMC
 * treats them as ordinary code.
 */
#ifndef RE_I128_H
#define RE_I128_H

#include <stdint.h>

#include "re_core.h"

/**
 * @brief Unsigned 128-bit value, little-endian word order.
 */
typedef struct ReU128 {
    uint64_t lo; /**< Low 64 bits. */
    uint64_t hi; /**< High 64 bits. */
} ReU128;

/**
 * @brief Shifts a 128-bit value left by s bits (0..127).
 *
 * Bits shifted beyond bit 127 are discarded (documented wrap, matching
 * the behavior needed by fixed-point saturation checks that run before
 * any shift that could lose information).
 *
 * @param v Input value.
 * @param s Shift amount, 0..127.
 */
static inline ReU128 re_u128_shl(ReU128 v, unsigned s) {
    ReU128 r;
    if (s == 0u) {
        r = v;
    } else if (s < 64u) {
        r.lo = v.lo << s;
        r.hi = (v.hi << s) | (v.lo >> (64u - s));
    } else {
        r.lo = 0u;
        r.hi = v.lo << (s - 64u);
    }
    return r;
}

/**
 * @brief Shifts a 128-bit value right by s bits (0..127).
 * @param v Input value.
 * @param s Shift amount, 0..127.
 */
static inline ReU128 re_u128_shr(ReU128 v, unsigned s) {
    ReU128 r;
    if (s == 0u) {
        r = v;
    } else if (s < 64u) {
        r.lo = (v.lo >> s) | (v.hi << (64u - s));
        r.hi = v.hi >> s;
    } else {
        r.lo = v.hi >> (s - 64u);
        r.hi = 0u;
    }
    return r;
}

/**
 * @brief Adds two 128-bit values (modulo 2^128, exact).
 * @param a First addend.
 * @param b Second addend.
 * @return Exact 128-bit sum.
 */
static inline ReU128 re_u128_add(ReU128 a, ReU128 b) {
    ReU128 r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + ((r.lo < a.lo) ? 1u : 0u);
    return r;
}

/**
 * @brief 64x64 -> 128-bit unsigned multiplication.
 *
 * Splits both operands into 32-bit halves, forms the four partial
 * products, shifts them into place and adds them with re_u128_add.
 * Additions wrap modulo 2^128, which is well-defined in C, so the full
 * product is exact for every input.
 *
 * @param a First factor.
 * @param b Second factor.
 * @return Full 128-bit product.
 */
static inline ReU128 re_u128_mul64(uint64_t a, uint64_t b) {
    const uint64_t mask32 = UINT64_C(0xFFFFFFFF);
    uint64_t al = a & mask32, ah = a >> 32u;
    uint64_t bl = b & mask32, bh = b >> 32u;
    ReU128 p00 = {(uint64_t)(al * bl), 0u};
    ReU128 p01 = {(uint64_t)(al * bh), 0u};
    ReU128 p10 = {(uint64_t)(ah * bl), 0u};
    ReU128 p11 = {(uint64_t)(ah * bh), 0u};
    p01 = re_u128_shl(p01, 32u);
    p10 = re_u128_shl(p10, 32u);
    p11 = re_u128_shl(p11, 64u);
    ReU128 r = re_u128_add(p00, p01);
    r = re_u128_add(r, p10);
    r = re_u128_add(r, p11);
    return r;
}

/**
 * @brief Compares two 128-bit values.
 * @param a First value.
 * @param b Second value.
 * @return -1 when a < b, 0 when equal, 1 when a > b.
 */
static inline int re_u128_cmp(ReU128 a, ReU128 b) {
    if (a.hi != b.hi) {
        return a.hi < b.hi ? -1 : 1;
    }
    if (a.lo != b.lo) {
        return a.lo < b.lo ? -1 : 1;
    }
    return 0;
}

/**
 * @brief Divides a 128-bit value by a 64-bit divisor.
 *
 * Shift-and-subtract long division after Hacker's Delight, fig. 9-3
 * (divlu). Total for every input: v == 0 or a quotient that does not fit
 * in 64 bits (u.hi >= v) both saturate to UINT64_MAX, which the fixed
 * point layer treats as saturation before this function is reached.
 *
 * @param u Dividend.
 * @param v Divisor.
 * @param rem_out Optional remainder output (may be NULL).
 * @return Quotient, or UINT64_MAX on saturation.
 */
static inline uint64_t re_u128_div64(ReU128 u, uint64_t v, uint64_t *rem_out) {
    uint64_t zero = 0u;
    uint64_t *rem = rem_out != NULL ? rem_out : &zero;
    *rem = 0u;
    if (v == 0u || u.hi >= v) {
        return UINT64_MAX; /* saturated: caller checks before relying on it */
    }
    const int s = re_clz64(v);
    const uint64_t vn = v << s;
    const uint64_t vn1 = vn >> 32u;
    const uint64_t vn0 = vn & UINT64_C(0xFFFFFFFF);
    /* s == 0 (v >= 2^63) needs no normalization: guard the shifts. */
    const uint64_t un64 = (s == 0) ? u.hi : (u.hi << s) | (u.lo >> (64u - (unsigned)s));
    const uint64_t un10 = (s == 0) ? u.lo : u.lo << s;
    const uint64_t un1 = un10 >> 32u;
    const uint64_t un0 = un10 & UINT64_C(0xFFFFFFFF);
    uint64_t q1 = un64 / vn1;
    uint64_t rhat = un64 - q1 * vn1;
    /* Correction loop: runs at most twice; rhat < 2^32 on every entry. */
    for (;;) {
        uint64_t prod = q1 * vn0;              /* q1 < 2^32 here */
        uint64_t rhs = (rhat << 32u) + un1;    /* rhat < 2^32: no overflow */
        if (prod > rhs) {
            q1--;
            rhat += vn1;
            if (rhat < (UINT64_C(1) << 32u)) {
                continue;
            }
        }
        break;
    }
    const uint64_t un21 = (rhat << 32u) + un1 - q1 * vn0; /* >= 0 after correction */
    uint64_t q0 = un21 / vn1;
    rhat = un21 - q0 * vn1;
    for (;;) {
        uint64_t prod = q0 * vn0;              /* q0 < 2^32 here */
        uint64_t rhs = (rhat << 32u) + un0;
        if (prod > rhs) {
            q0--;
            rhat += vn1;
            if (rhat < (UINT64_C(1) << 32u)) {
                continue;
            }
        }
        break;
    }
    *rem = ((rhat << 32u) + un0 - q0 * vn0) >> s;
    return (q1 << 32u) | q0;
}

/**
 * @brief Integer square root of a 64-bit value (floor).
 *
 * Bit-by-bit algorithm over 2-bit digit pairs: uses only shifts, adds
 * and comparisons, so no intermediate value can overflow.
 *
 * @param x Input value.
 * @return Floor of sqrt(x).
 */
static inline uint64_t re_isqrt64(uint64_t x) {
    uint64_t r = 0u;
    uint64_t rem = 0u;
    for (int i = 62; i >= 0; i -= 2) {
        rem = (rem << 2u) | ((x >> (unsigned)i) & 3u);
        uint64_t trial = (r << 2u) | 1u;
        if (trial <= rem) {
            rem -= trial;
            r = (r << 1u) | 1u;
        } else {
            r <<= 1u;
        }
    }
    return r;
}

#endif /* RE_I128_H */
