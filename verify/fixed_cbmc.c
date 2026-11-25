/**
 * @file fixed_cbmc.c
 * @brief CBMC proof: Q32.32 arithmetic is free of signed-overflow UB,
 *        division-by-zero and shift UB across the full symbolic int64
 *        domain, and satisfies its saturation and identity contracts.
 *
 * Run: cbmc fixed_cbmc.c --unwind 64 --unwinding-assertions \
 *        --pointer-overflow-check --signed-overflow-check \
 *        --unsigned-overflow-check
 *
 * The signed/unsigned overflow checks make ANY UB a proof failure; the
 * asserts below additionally pin the documented contracts.
 */
#include "harness_stub.h"

#include "../src/re_core.c"
#include "harness_stub.h"

#include "../src/re_fixed.c"

#include "harness_stub.h"

#include "../src/re_fixed.h"

/** Full-domain symbolic 64-bit pattern. */
static int64_t nondet_i64(void) {
    return (int64_t)(((uint64_t)nondet_uint() << 32u) | (uint64_t)nondet_uint());
}

int main(void) {
    for (int round = 0; round < 4; round++) {
        re_fixed64_t a = nondet_i64();
        re_fixed64_t b = nondet_i64();

        /* Totality: every operation returns a well-defined value. */
        re_fixed64_t s = re_fixed64_add(a, b);
        re_fixed64_t d = re_fixed64_sub(a, b);
        re_fixed64_t m = re_fixed64_mul(a, b);
        re_fixed64_t q = re_fixed64_div(a, b);
        (void)s;
        (void)d;
        (void)m;
        (void)q;

        /* Division by zero and negative sqrt are defined (0). */
        __CPROVER_assert(re_fixed64_div(a, 0) == 0, "division by zero yields 0");
        re_fixed64_t r = re_fixed64_sqrt(a);
        __CPROVER_assert(r >= 0, "sqrt result non-negative");

        /* Saturation bounds of the products/sums. */
        __CPROVER_assert(m <= RE_FIXED_MAX && m >= RE_FIXED_MIN, "mul saturates");
        __CPROVER_assert(s <= RE_FIXED_MAX && s >= RE_FIXED_MIN, "add saturates");
        __CPROVER_assert(d <= RE_FIXED_MAX && d >= RE_FIXED_MIN, "sub saturates");

        /* Abs and sign symmetry. */
        re_fixed64_t ab = re_fixed64_abs(a);
        __CPROVER_assert(ab >= 0, "abs non-negative");
        __CPROVER_assert(re_fixed64_cmp(a, b) == -re_fixed64_cmp(b, a),
                         "cmp antisymmetric");

        /* Trigonometry is total and bounded. */
        re_fixed64_t sn = re_fixed64_sin(a);
        re_fixed64_t cs = re_fixed64_cos(a);
        __CPROVER_assert(sn <= RE_FIXED_ONE && sn >= -RE_FIXED_ONE, "sin bounded");
        __CPROVER_assert(cs <= RE_FIXED_ONE && cs >= -RE_FIXED_ONE, "cos bounded");

        /* Vector normalize of a zero vector is the zero vector. */
        ReVec3F z = re_vec3f_zero();
        ReVec3F nz = re_vec3f_normalize(z);
        __CPROVER_assert(nz.x == 0 && nz.y == 0 && nz.z == 0,
                         "zero vector normalizes to zero");
    }

    /* Exact ground truths (constant inputs: fully decided by CBMC). */
    __CPROVER_assert(re_fixed64_mul(RE_FIXED_ONE, RE_FIXED_ONE) == RE_FIXED_ONE,
                     "1 * 1 == 1");
    __CPROVER_assert(re_fixed64_div(RE_FIXED_ONE, RE_FIXED_ONE) == RE_FIXED_ONE,
                     "1 / 1 == 1");
    __CPROVER_assert(re_fixed64_div(RE_FIXED_ONE, 2) == RE_FIXED_HALF,
                     "1 / 2 == 0.5 exactly");
    __CPROVER_assert(re_fixed64_sqrt(4 * RE_FIXED_ONE) == 2 * RE_FIXED_ONE,
                     "sqrt(4) == 2 exactly");
    __CPROVER_assert(re_fixed64_sqrt(RE_FIXED_ONE) == RE_FIXED_ONE,
                     "sqrt(1) == 1 exactly");
    __CPROVER_assert(re_fixed64_sqrt(0) == 0, "sqrt(0) == 0");
    __CPROVER_assert(re_fixed64_sin(0) == 0, "sin(0) == 0");
    __CPROVER_assert(re_fixed64_cos(0) == RE_FIXED_ONE, "cos(0) == 1");
    __CPROVER_assert(re_fixed64_from_int(7) == 7 * RE_FIXED_ONE,
                     "from_int is an exact shift");
    __CPROVER_assert(re_fixed64_from_f32(0.0f) == 0, "from_f32(0) == 0");
    __CPROVER_assert(re_fixed64_to_f32(0) == 0.0f, "to_f32(0) == 0");

    /* 128-bit helpers: exact ground truths. */
    ReU128 p = re_u128_mul64(UINT64_MAX, UINT64_MAX);
    __CPROVER_assert(p.hi == UINT64_MAX - 1u && p.lo == 1u,
                     "(2^64-1)^2 == 2^128 - 2^65 + 1");
    uint64_t rem = 0u;
    uint64_t qd = re_u128_div64((ReU128){100u, 0u}, 7u, &rem);
    __CPROVER_assert(qd == 14u && rem == 2u, "100 / 7 == 14 rem 2");
    uint64_t qbig = re_u128_div64((ReU128){0u, 1u}, 2u, NULL);
    __CPROVER_assert(qbig == UINT64_C(0x8000000000000000), "2^64 / 2 == 2^63");
    __CPROVER_assert(re_isqrt64(0u) == 0u, "isqrt(0) == 0");
    __CPROVER_assert(re_isqrt64(1u) == 1u, "isqrt(1) == 1");
    __CPROVER_assert(re_isqrt64(0xFFFFFFFFFFFFFFFFULL) == 0xFFFFFFFFu,
                     "isqrt(2^64-1) == 2^32-1");
    __CPROVER_assert(re_clz64(1u) == 63, "clz64(1) == 63");
    __CPROVER_assert(re_clz64(0u) == 64, "clz64(0) == 64");
    return 0;
}
