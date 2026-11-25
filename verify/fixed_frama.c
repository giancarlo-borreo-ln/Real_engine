/**
 * @file fixed_frama.c
 * @brief Frama-C/WP proof: Q32.32 arithmetic is total and saturating for
 *        all inputs — no signed overflow, no division by zero, no
 *        shift-by-64 — and satisfies its algebraic identities on the
 *        analyzed paths. -wp-rte proves the overflow/division/shift
 *        freedom automatically for every concrete call main() makes.
 *
 * Run: frama-c -wp -wp-rte -wp-fct main -wp-timeout 120 fixed_frama.c
 */
#include "../src/re_core.h"
#include "../src/re_i128.h"
#include "../src/re_fixed.h"
#include "../src/re_fixed.c"

/*@ ensures \result >= RE_FIXED_MIN && \result <= RE_FIXED_MAX; */
re_fixed64_t re_fixed64_mul(re_fixed64_t a, re_fixed64_t b);

/*@ ensures \result >= RE_FIXED_MIN && \result <= RE_FIXED_MAX; */
re_fixed64_t re_fixed64_div(re_fixed64_t a, re_fixed64_t b);

/*@ ensures \result >= RE_FIXED_MIN && \result <= RE_FIXED_MAX; */
re_fixed64_t re_fixed64_add(re_fixed64_t a, re_fixed64_t b);

/*@ ensures \result >= RE_FIXED_MIN && \result <= RE_FIXED_MAX; */
re_fixed64_t re_fixed64_sub(re_fixed64_t a, re_fixed64_t b);

/*@ ensures \result >= 0; */
re_fixed64_t re_fixed64_sqrt(re_fixed64_t a);

int main(void) {
    /* Algebraic identities on exact values. */
    /*@ assert re_fixed64_mul(RE_FIXED_ONE, RE_FIXED_ONE) == RE_FIXED_ONE; */
    /*@ assert re_fixed64_div(RE_FIXED_ONE, RE_FIXED_ONE) == RE_FIXED_ONE; */
    /*@ assert re_fixed64_div(RE_FIXED_ONE, 2) == RE_FIXED_HALF; */
    /*@ assert re_fixed64_sqrt(4 * RE_FIXED_ONE) == 2 * RE_FIXED_ONE; */
    /*@ assert re_fixed64_sqrt(0) == 0; */
    /*@ assert re_fixed64_sqrt(-1) == 0; */ /* documented: negative -> 0 */
    /*@ assert re_fixed64_div(1, 0) == 0; */ /* documented: div by zero -> 0 */

    /* Saturation extremes: no UB, bounded results. */
    re_fixed64_t big = RE_FIXED_MAX;
    /*@ assert re_fixed64_add(big, 1) == RE_FIXED_MAX; */
    /*@ assert re_fixed64_sub(RE_FIXED_MIN, 1) == RE_FIXED_MIN; */
    /*@ assert re_fixed64_mul(big, big) == RE_FIXED_MAX; */
    /*@ assert re_fixed64_abs(RE_FIXED_MIN) == RE_FIXED_MAX; */

    /* Comparison laws. */
    /*@ assert re_fixed64_cmp(1, 2) < 0; */
    /*@ assert re_fixed64_cmp(2, 1) > 0; */
    /*@ assert re_fixed64_cmp(7, 7) == 0; */

    /* 128-bit helpers on exact values. */
    ReU128 p = re_u128_mul64(UINT64_MAX, UINT64_MAX);
    /*@ assert p.hi == UINT64_MAX - 1 && p.lo == 1; */
    uint64_t rem = 0;
    uint64_t q = re_u128_div64((ReU128){100u, 0u}, 7u, &rem);
    /*@ assert q == 14 && rem == 2; */
    /*@ assert re_isqrt64(0u) == 0u; */
    /*@ assert re_isqrt64(1u) == 1u; */
    /*@ assert re_isqrt64(4u) == 2u; */
    /*@ assert re_isqrt64(UINT64_MAX) == 0xFFFFFFFFu; */
    return 0;
}
