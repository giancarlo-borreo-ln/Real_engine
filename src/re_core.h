/**
 * @file re_core.h
 * @brief Engine-wide primitives: assertions, logging, 256-bit component
 *        masks, deterministic PRNG, monotonic time and alignment helpers.
 *
 * Real Engine - C23 data-oriented game engine.
 * Everything in this header is freestanding: no OS call leaks into
 * translation units that only include re_core.h.
 */
#ifndef RE_CORE_H
#define RE_CORE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/** Engine semantic version components. */
#define RE_VERSION_MAJOR 1u
#define RE_VERSION_MINOR 0u
#define RE_VERSION_PATCH 0u

/** Cache line size the allocators and queues pad to (bytes). */
#define RE_CACHE_LINE 64u

/** Number of elements of a C array. */
#define RE_ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

/** Format attribute for printf-like functions, portable across gcc/clang. */
#if defined(__GNUC__) || defined(__clang__)
#define RE_PRINTF_FMT(fmt_idx, var_idx) __attribute__((format(printf, fmt_idx, var_idx)))
#else
#define RE_PRINTF_FMT(fmt_idx, var_idx)
#endif

/** Hints for branch prediction (compiler builtins, accepted under -pedantic). */
#define RE_LIKELY(x) __builtin_expect(!!(x), 1)
#define RE_UNLIKELY(x) __builtin_expect(!!(x), 0)

/**
 * @brief Terminates the process after printing file, line and message.
 * @param file Source file name (__FILE__).
 * @param line Source line number (__LINE__).
 * @param message Human-readable failure description.
 */
_Noreturn void re_panic(const char *file, int line, const char *message);

/** Assertion that terminates the process with a message on failure. */
#define RE_ASSERT(cond) ((cond) ? (void)0 : re_panic(__FILE__, __LINE__, #cond))

/**
 * @brief Logs an informational message to stderr.
 * @param fmt printf-style format string.
 */
RE_PRINTF_FMT(1, 2) void re_log_info(const char *fmt, ...);

/**
 * @brief Logs a warning message to stderr.
 * @param fmt printf-style format string.
 */
RE_PRINTF_FMT(1, 2) void re_log_warn(const char *fmt, ...);

/**
 * @brief Logs an error message to stderr.
 * @param fmt printf-style format string.
 */
RE_PRINTF_FMT(1, 2) void re_log_error(const char *fmt, ...);

/**
 * @brief Returns a monotonically increasing timestamp in nanoseconds.
 */
uint64_t re_time_ns(void);

/**
 * @brief Sleeps for at least ms milliseconds.
 * @param ms Milliseconds to sleep.
 */
void re_sleep_ms(uint32_t ms);

/** @return Minimum of two signed 64-bit values. */
static inline int64_t re_min_i64(int64_t a, int64_t b) { return a < b ? a : b; }

/** @return Maximum of two signed 64-bit values. */
static inline int64_t re_max_i64(int64_t a, int64_t b) { return a > b ? a : b; }

/** @return Minimum of two 32-bit floats. */
static inline float re_min_f32(float a, float b) { return a < b ? a : b; }

/** @return Maximum of two 32-bit floats. */
static inline float re_max_f32(float a, float b) { return a > b ? a : b; }

/** @return v clamped to the inclusive range [lo, hi]. */
static inline float re_clamp_f32(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/** @return Minimum of two unsigned 32-bit values. */
static inline uint32_t re_min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

/** @return Maximum of two unsigned 32-bit values. */
static inline uint32_t re_max_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }

/** @return Minimum of two size_t values. */
static inline size_t re_min_size(size_t a, size_t b) { return a < b ? a : b; }

/**
 * @brief Rounds v up to the next multiple of align.
 *
 * align must be a power of two, at least 1. On overflow the result
 * saturates to the largest multiple of align representable in size_t,
 * so the returned value is always aligned and never wraps.
 *
 * @param v Value to round.
 * @param align Alignment (power of two, >= 1).
 * @return Smallest multiple of align greater than or equal to v.
 */
static inline size_t re_align_up(size_t v, size_t align) {
    size_t m = align - 1u;
    size_t sum = v + m;
    return (sum < v) ? (SIZE_MAX & ~m) : (sum & ~m);
}

/**
 * @brief Rounds v down to the previous multiple of align.
 * @param v Value to round.
 * @param align Alignment (power of two, >= 1).
 */
static inline size_t re_align_down(size_t v, size_t align) {
    return v & ~(align - 1u);
}

/** @return True when a is a power of two (non-zero). */
static inline bool re_is_pow2_u32(uint32_t a) { return a != 0u && (a & (a - 1u)) == 0u; }

/** @return True when a is a power of two (non-zero). */
static inline bool re_is_pow2_size(size_t a) { return a != 0u && (a & (a - 1u)) == 0u; }

/**
 * @brief Smallest power of two greater than or equal to v.
 * @param v Input value.
 * @return Next power of two, or 0 when it would overflow 32 bits.
 */
static inline uint32_t re_next_pow2_u32(uint32_t v) {
    if (v <= 1u) {
        return 1u;
    }
    if (v > 0x80000000u) {
        return 0u;
    }
    v--;
    v |= v >> 1u;
    v |= v >> 2u;
    v |= v >> 4u;
    v |= v >> 8u;
    v |= v >> 16u;
    return v + 1u;
}

/**
 * @brief Counts leading zero bits of a 32-bit value.
 * @param x Input value.
 * @return Number of leading zeros (32 when x is zero).
 */
static inline int re_clz32(uint32_t x) {
    if (x == 0u) {
        return 32;
    }
    int n = 0;
    if ((x & 0xFFFF0000u) == 0u) { n += 16; x <<= 16; }
    if ((x & 0xFF000000u) == 0u) { n += 8; x <<= 8; }
    if ((x & 0xF0000000u) == 0u) { n += 4; x <<= 4; }
    if ((x & 0xC0000000u) == 0u) { n += 2; x <<= 2; }
    if ((x & 0x80000000u) == 0u) { n += 1; }
    return n;
}

/**
 * @brief Counts leading zero bits of a 64-bit value.
 * @param x Input value.
 * @return Number of leading zeros (64 when x is zero).
 */
static inline int re_clz64(uint64_t x) {
    return (x >> 32u) != 0u ? re_clz32((uint32_t)(x >> 32u)) : 32 + re_clz32((uint32_t)x);
}

/**
 * @brief Population count of a 32-bit value.
 * @param x Input value.
 * @return Number of set bits.
 */
static inline uint32_t re_popcount32(uint32_t x) {
    x = x - ((x >> 1u) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2u) & 0x33333333u);
    x = (x + (x >> 4u)) & 0x0F0F0F0Fu;
    return (x * 0x01010101u) >> 24u;
}

/**
 * @brief Population count of a 64-bit value.
 * @param x Input value.
 * @return Number of set bits.
 */
static inline uint32_t re_popcount64(uint64_t x) {
    return re_popcount32((uint32_t)x) + re_popcount32((uint32_t)(x >> 32u));
}

/** Number of 64-bit words in a component mask (256 bits total). */
#define RE_MASK_WORDS 4u

/** Number of bits in a component mask. */
#define RE_MASK_BITS 256u

/**
 * @brief 256-bit component access mask used by the ECS and the scheduler.
 *
 * Bit i covers component id i; bit i lives in word i / 64 at shift i % 64.
 */
typedef struct ReMask {
    uint64_t w[RE_MASK_WORDS]; /**< Little-endian word layout: w[0] holds bits 0..63. */
} ReMask;

/** @return A mask with every bit cleared. */
static inline ReMask re_mask_zero(void) {
    ReMask m = {{0u, 0u, 0u, 0u}};
    return m;
}

/** @return A mask with bit set. Bits outside [0, RE_MASK_BITS) are ignored. */
static inline ReMask re_mask_bit(unsigned bit) {
    ReMask m = {{0u, 0u, 0u, 0u}};
    if (bit < RE_MASK_BITS) {
        m.w[bit / 64u] = UINT64_C(1) << (bit % 64u);
    }
    return m;
}

/** @brief Sets bit in the mask. Bits outside [0, RE_MASK_BITS) are ignored. */
static inline void re_mask_set(ReMask *m, unsigned bit) {
    if (bit < RE_MASK_BITS) {
        m->w[bit / 64u] |= UINT64_C(1) << (bit % 64u);
    }
}

/** @brief Clears bit in the mask. Bits outside [0, RE_MASK_BITS) are ignored. */
static inline void re_mask_clear(ReMask *m, unsigned bit) {
    if (bit < RE_MASK_BITS) {
        m->w[bit / 64u] &= ~(UINT64_C(1) << (bit % 64u));
    }
}

/** @return True when bit is set in the mask. */
static inline bool re_mask_test(const ReMask *m, unsigned bit) {
    return bit < RE_MASK_BITS && (m->w[bit / 64u] & (UINT64_C(1) << (bit % 64u))) != 0u;
}

/** @return Bitwise OR of a and b. */
static inline ReMask re_mask_or(ReMask a, ReMask b) {
    ReMask r;
    for (unsigned i = 0; i < RE_MASK_WORDS; i++) {
        r.w[i] = a.w[i] | b.w[i];
    }
    return r;
}

/** @return Bitwise AND of a and b. */
static inline ReMask re_mask_and(ReMask a, ReMask b) {
    ReMask r;
    for (unsigned i = 0; i < RE_MASK_WORDS; i++) {
        r.w[i] = a.w[i] & b.w[i];
    }
    return r;
}

/** @return Bitwise complement of a. */
static inline ReMask re_mask_not(ReMask a) {
    ReMask r;
    for (unsigned i = 0; i < RE_MASK_WORDS; i++) {
        r.w[i] = ~a.w[i];
    }
    return r;
}

/** @return True when the mask has at least one set bit. */
static inline bool re_mask_any(ReMask m) {
    return m.w[0] | m.w[1] | m.w[2] | m.w[3];
}

/** @return True when the two masks share at least one set bit. */
static inline bool re_mask_overlaps(ReMask a, ReMask b) {
    ReMask x = re_mask_and(a, b);
    return re_mask_any(x);
}

/** @return True when every bit set in subset is also set in superset. */
static inline bool re_mask_subset(ReMask subset, ReMask superset) {
    ReMask x = re_mask_and(subset, re_mask_not(superset));
    return !re_mask_any(x);
}

/** @return True when the masks are identical. */
static inline bool re_mask_equal(ReMask a, ReMask b) {
    return a.w[0] == b.w[0] && a.w[1] == b.w[1] && a.w[2] == b.w[2] && a.w[3] == b.w[3];
}

/** @return Number of set bits in the mask. */
static inline uint32_t re_mask_count(ReMask m) {
    return re_popcount64(m.w[0]) + re_popcount64(m.w[1]) +
           re_popcount64(m.w[2]) + re_popcount64(m.w[3]);
}

/**
 * @brief Counts trailing zero bits of a 32-bit value.
 * @param x Input value (must be non-zero).
 * @return Number of trailing zeros.
 */
static inline int re_ctz32(uint32_t x) {
    return 31 - re_clz32(x & (0u - x)); /* isolate the lowest set bit */
}

/**
 * @brief Counts trailing zero bits of a 64-bit value.
 * @param x Input value (must be non-zero).
 * @return Number of trailing zeros.
 */
static inline int re_ctz64(uint64_t x) {
    return (x & 0xFFFFFFFFu) != 0u ? re_ctz32((uint32_t)x)
                                   : 32 + re_ctz32((uint32_t)(x >> 32u));
}

/**
 * @brief Index of the lowest set bit of the mask.
 * @param m Input mask.
 * @return Bit index in [0, RE_MASK_BITS), or -1 when the mask is empty.
 */
static inline int re_mask_first_bit(ReMask m) {
    for (unsigned i = 0; i < RE_MASK_WORDS; i++) {
        if (m.w[i] != 0u) {
            return (int)(i * 64u) + re_ctz64(m.w[i]);
        }
    }
    return -1;
}

/**
 * @brief FNV-1a hash of the mask, used for hash tables keyed by masks.
 * @param m Input mask.
 * @return 64-bit hash value.
 */
static inline uint64_t re_mask_hash(ReMask m) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (unsigned i = 0; i < RE_MASK_WORDS; i++) {
        uint64_t x = m.w[i];
        for (unsigned b = 0; b < 8u; b++) {
            h ^= (uint8_t)(x >> (b * 8u));
            h *= UINT64_C(1099511628211);
        }
    }
    return h;
}

/**
 * @brief Deterministic splitmix64 PRNG state.
 *
 * Produces the same sequence on every platform and with every compiler,
 * which makes it the only source of randomness allowed in simulation code.
 */
typedef struct ReRng {
    uint64_t state; /**< Internal state, advanced by re_rng_next. */
} ReRng;

/**
 * @brief Seeds a PRNG. A seed of 0 is remapped to a non-zero constant.
 * @param seed Arbitrary seed value.
 * @return Initialized PRNG state.
 */
static inline ReRng re_rng_init(uint64_t seed) {
    ReRng rng;
    rng.state = seed == 0u ? UINT64_C(0x9E3779B97F4A7C15) : seed;
    return rng;
}

/**
 * @brief Advances the PRNG and returns the next value (splitmix64).
 * @param rng PRNG state.
 * @return Next 64-bit pseudo-random value.
 */
static inline uint64_t re_rng_next(ReRng *rng) {
    uint64_t z = (rng->state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30u)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27u)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31u);
}

/**
 * @brief Uniform 64-bit value in [0, bound) via modulo.
 *
 * Deterministic; carries the modulo bias of a simple modulus, which is
 * acceptable for engine purposes (documented trade-off, not an error).
 *
 * @param rng PRNG state.
 * @param bound Exclusive upper bound (must be non-zero).
 */
static inline uint64_t re_rng_range(ReRng *rng, uint64_t bound) {
    return re_rng_next(rng) % bound;
}

/**
 * @brief Uniform float in [0, 1) derived from the PRNG.
 * @param rng PRNG state.
 */
static inline float re_rng_f32(ReRng *rng) {
    return (float)(re_rng_next(rng) >> 40u) / 16777216.0f;
}

#endif /* RE_CORE_H */
