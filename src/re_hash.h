/**
 * @file re_hash.h
 * @brief FNV-1a and CRC-64 (ECMA-182) checksums for state verification.
 *
 * Both are bitwise implementations with no tables, so they are fully
 * deterministic and CBMC-friendly; they run on the rollback path, not
 * on the render hot path.
 */
#ifndef RE_HASH_H
#define RE_HASH_H

#include <stddef.h>
#include <stdint.h>

/** FNV-1a offset basis. */
#define RE_FNV1A_BASIS UINT64_C(14695981039346656037)

/** FNV-1a prime. */
#define RE_FNV1A_PRIME UINT64_C(1099511628211)

/** CRC-64 reflected polynomial (bit reflection of the ECMA-182
 *  polynomial 0x42F0E1EBA9EA3693; CRC-64/XZ, init/xorout all-ones). */
#define RE_CRC64_POLY UINT64_C(0xC96C5795D7870F42)

/** CRC-64 initial value. */
#define RE_CRC64_INIT UINT64_C(0xFFFFFFFFFFFFFFFF)

/**
 * @brief FNV-1a 64-bit hash of a byte range.
 * @param data Bytes to hash.
 * @param len Byte count.
 * @return 64-bit hash.
 */
uint64_t re_fnv1a64(const void *data, size_t len);

/**
 * @brief CRC-64 (ECMA-182, reflected, init/xorout all-ones) of a byte
 *        range, computed bitwise.
 * @param data Bytes to checksum.
 * @param len Byte count.
 * @return 64-bit checksum.
 */
uint64_t re_crc64(const void *data, size_t len);

/**
 * @brief Running CRC-64 update (same polynomial), for streaming use.
 * @param crc Previous checksum (RE_CRC64_INIT to start).
 * @param byte Next byte.
 * @return Updated checksum.
 */
uint64_t re_crc64_update(uint64_t crc, uint8_t byte);

#endif /* RE_HASH_H */
