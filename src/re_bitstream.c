/**
 * @file re_bitstream.c
 * @brief Bounds-checked MSB-first bit serializer.
 *
 * Invariant shared by both directions: bit_pos/bit_len <= bit_cap always.
 * Every capacity test is written as `needed > cap - used` so the sum
 * needed + used can never overflow size_t, whatever the buffer size.
 * The writer normalizes the whole byte it is about to start, so padding
 * bits below re_bw_byte_len() are deterministic zeros rather than stale
 * buffer content (the reader ignores them).
 */
#include "re_bitstream.h"

#include <string.h>

/* Bits per byte, used for the byte<->bit index conversions. */
#define RE_BITS_PER_BYTE 8u

/* Largest bit count a single write_bits/read_bits call accepts. */
#define RE_MAX_BITS_PER_VALUE 64u

/* Bits consumed by the fixed-width wrappers. */
#define RE_BITS_U16 16u
#define RE_BITS_U32 32u

void re_bw_init(ReBitWriter *w, void *buf, size_t byte_cap)
{
    w->buf = (uint8_t *)buf;
    /* Documented assumption: a real buffer has byte_cap < 2^61, so the
     * conversion to a bit count cannot overflow size_t. */
    w->bit_cap = byte_cap * RE_BITS_PER_BYTE;
    w->bit_len = 0u;
    w->oom = false;
}

void re_bw_write_bits(ReBitWriter *w, uint64_t value, unsigned n)
{
    if (n == 0u) {
        return;
    }
    /* oom latches: after one rejected write the stream is dead, so a later
     * small write must not silently resume it (a failed write leaves
     * bit_len untouched, thus the capacity check alone would let it pass).
     * bit_len <= bit_cap, so the subtraction cannot underflow. */
    if (w->oom || n > RE_MAX_BITS_PER_VALUE || n > w->bit_cap - w->bit_len) {
        w->oom = true;
        return;
    }
    for (unsigned i = n; i >= 1u; --i) {
        size_t pos = w->bit_len;
        size_t byte = pos / RE_BITS_PER_BYTE;
        unsigned off = (unsigned)(pos % RE_BITS_PER_BYTE);
        uint8_t mask = (uint8_t)(1u << (7u - off));

        if (off == 0u) {
            w->buf[byte] = 0u; /* Deterministic padding for partial bytes. */
        }
        if (((value >> (i - 1u)) & UINT64_C(1)) != 0u) {
            w->buf[byte] |= mask;
        } else {
            w->buf[byte] &= (uint8_t)~mask;
        }
        ++w->bit_len;
    }
}

void re_bw_write_u32(ReBitWriter *w, uint32_t v)
{
    re_bw_write_bits(w, (uint64_t)v, RE_BITS_U32);
}

void re_bw_write_u64(ReBitWriter *w, uint64_t v)
{
    re_bw_write_bits(w, v, RE_MAX_BITS_PER_VALUE);
}

void re_bw_write_i16(ReBitWriter *w, int16_t v)
{
    /* Reinterpretation via unsigned is well-defined (modulo 2^16). */
    re_bw_write_bits(w, (uint64_t)(uint16_t)v, RE_BITS_U16);
}

void re_bw_write_f32(ReBitWriter *w, float v)
{
    uint32_t bits = 0u;

    memcpy(&bits, &v, sizeof bits);
    re_bw_write_bits(w, (uint64_t)bits, RE_BITS_U32);
}

size_t re_bw_byte_len(const ReBitWriter *w)
{
    return (w->bit_len + (RE_BITS_PER_BYTE - 1u)) / RE_BITS_PER_BYTE;
}

bool re_bw_oom(const ReBitWriter *w)
{
    return w->oom;
}

void re_br_init(ReBitReader *r, const void *buf, size_t byte_len)
{
    r->buf = (const uint8_t *)buf;
    /* Fail closed on absurd lengths: bit_cap = byte_len * 8 must not wrap
       (the recv path must clamp the length to the datagram anyway). */
    if (byte_len > SIZE_MAX / RE_BITS_PER_BYTE) {
        r->bit_cap = 0u;
        r->bit_pos = 0u;
        r->err = true;
        return;
    }
    r->bit_cap = byte_len * RE_BITS_PER_BYTE;
    r->bit_pos = 0u;
    r->err = false;
}

bool re_br_read_bits(ReBitReader *r, unsigned n, uint64_t *out)
{
    *out = 0u;
    if (n == 0u) {
        return true;
    }
    /* bit_pos <= bit_cap, so the subtraction cannot underflow. */
    if (n > RE_MAX_BITS_PER_VALUE || n > r->bit_cap - r->bit_pos) {
        r->err = true;
        return false;
    }
    for (unsigned i = 0; i < n; ++i) {
        size_t pos = r->bit_pos + i;
        unsigned bit = ((unsigned)r->buf[pos / RE_BITS_PER_BYTE]
                        >> (7u - (unsigned)(pos % RE_BITS_PER_BYTE))) & 1u;

        /* n <= 64, so the shift count stays in 0..63. */
        *out = (*out << 1) | (uint64_t)bit;
    }
    r->bit_pos += n;
    return true;
}

bool re_br_read_u32(ReBitReader *r, uint32_t *out)
{
    uint64_t value = 0u;
    bool ok = re_br_read_bits(r, RE_BITS_U32, &value);

    *out = (uint32_t)value; /* 0 when the read failed. */
    return ok;
}

bool re_br_read_u64(ReBitReader *r, uint64_t *out)
{
    uint64_t value = 0u;
    bool ok = re_br_read_bits(r, RE_MAX_BITS_PER_VALUE, &value);

    *out = value; /* 0 when the read failed. */
    return ok;
}

bool re_br_read_i16(ReBitReader *r, int16_t *out)
{
    uint64_t value = 0u;
    bool ok = re_br_read_bits(r, RE_BITS_U16, &value);

    *out = (int16_t)(uint16_t)value; /* 0 when the read failed. */
    return ok;
}

bool re_br_read_f32(ReBitReader *r, float *out)
{
    uint64_t value = 0u;
    bool ok = re_br_read_bits(r, RE_BITS_U32, &value);
    uint32_t bits = (uint32_t)value;
    float result = 0.0f;

    memcpy(&result, &bits, sizeof result);
    *out = result; /* +0.0f when the read failed. */
    return ok;
}

size_t re_br_remaining(const ReBitReader *r)
{
    return r->bit_pos < r->bit_cap ? r->bit_cap - r->bit_pos : 0u;
}

bool re_br_err(const ReBitReader *r)
{
    return r->err;
}
