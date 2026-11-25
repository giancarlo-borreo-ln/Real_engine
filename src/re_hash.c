/**
 * @file re_hash.c
 * @brief Bitwise FNV-1a and CRC-64 (ECMA-182) checksums.
 *
 * Both functions read the input exactly once, in increasing address order,
 * and use only fixed-width unsigned arithmetic, so the result depends on
 * the bytes alone: no tables, no libc state, no endianness. The CRC is the
 * reflected CRC-64/XZ variant: init and xorout are all-ones, polynomial
 * RE_CRC64_POLY (the bit reflection of ECMA-182's 0x42F0E1EBA9EA3693),
 * verified against the official check value
 * crc64("123456789") == 0x995DC9BBDF1939FA.
 */
#include "re_hash.h"

/* Number of bits folded per input byte by the reflected CRC-64 loop. */
#define RE_CRC64_BITS_PER_BYTE 8

uint64_t re_fnv1a64(const void *data, size_t len)
{
    const uint8_t *bytes = (const uint8_t *)data;
    uint64_t h = RE_FNV1A_BASIS;

    /* data may be NULL when len == 0: the loop body never runs. */
    for (size_t i = 0; i < len; ++i) {
        h ^= (uint64_t)bytes[i];
        h *= RE_FNV1A_PRIME;
    }
    return h;
}

uint64_t re_crc64_update(uint64_t crc, uint8_t byte)
{
    crc ^= (uint64_t)byte;
    for (int i = 0; i < RE_CRC64_BITS_PER_BYTE; ++i) {
        /* Reflected form: shift out the low bit, conditionally xor the poly. */
        if ((crc & UINT64_C(1)) != 0u) {
            crc = (crc >> 1) ^ RE_CRC64_POLY;
        } else {
            crc >>= 1;
        }
    }
    return crc;
}

uint64_t re_crc64(const void *data, size_t len)
{
    const uint8_t *bytes = (const uint8_t *)data;
    uint64_t crc = RE_CRC64_INIT;

    /* data may be NULL when len == 0: the loop body never runs. */
    for (size_t i = 0; i < len; ++i) {
        crc = re_crc64_update(crc, bytes[i]);
    }
    /* CRC-64/XZ finishes with xorout all-ones, as documented in re_hash.h;
     * the streaming update deliberately exposes the raw register instead. */
    return crc ^ UINT64_MAX;
}
