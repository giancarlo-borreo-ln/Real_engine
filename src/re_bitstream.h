/**
 * @file re_bitstream.h
 * @brief Bounds-checked bit-level serializer.
 *
 * The writer packs values of 0..64 bits MSB-first into a caller
 * buffer; the reader unpacks them and validates every access against
 * the buffer end, setting an error flag instead of ever reading out
 * of bounds. No intermediate copies: the reader points directly into
 * the network buffer (zero-copy), and malformed input simply marks the
 * stream failed. Bit order is big-endian, which makes the wire format
 * endian-neutral by construction.
 *
 * Reading past the end is impossible by construction (every read
 * checks bit_pos + n <= bit_cap first); the CBMC harness proves it for
 * arbitrary buffers with __CPROVER_r_ok assertions.
 */
#ifndef RE_BITSTREAM_H
#define RE_BITSTREAM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Bit writer over a caller-provided buffer.
 */
typedef struct ReBitWriter {
    uint8_t *buf;    /**< Destination buffer. */
    size_t bit_cap;  /**< Buffer size in bits (byte_cap * 8). */
    size_t bit_len;  /**< Bits written so far. */
    bool oom;        /**< True after a write that did not fit. */
} ReBitWriter;

/**
 * @brief Initializes the writer.
 * @param w Writer.
 * @param buf Destination buffer (byte_cap bytes).
 * @param byte_cap Buffer size in bytes.
 */
void re_bw_init(ReBitWriter *w, void *buf, size_t byte_cap);

/**
 * @brief Writes the low n bits of value (n in 0..64), MSB-first.
 *
 * n == 0 is a no-op. Writes that would exceed the buffer set oom and
 * are ignored; check re_bw_oom before using the buffer.
 * @param w Writer.
 * @param value Bits to write.
 * @param n Bit count.
 */
void re_bw_write_bits(ReBitWriter *w, uint64_t value, unsigned n);

/** @brief Writes a full 32-bit value. */
void re_bw_write_u32(ReBitWriter *w, uint32_t v);

/** @brief Writes a full 64-bit value. */
void re_bw_write_u64(ReBitWriter *w, uint64_t v);

/** @brief Writes a 16-bit two's complement value. */
void re_bw_write_i16(ReBitWriter *w, int16_t v);

/** @brief Writes the 32-bit pattern of a float (endian-neutral). */
void re_bw_write_f32(ReBitWriter *w, float v);

/** @return Bytes written so far (rounded up). */
size_t re_bw_byte_len(const ReBitWriter *w);

/** @return True after a write exceeded the buffer. */
bool re_bw_oom(const ReBitWriter *w);

/**
 * @brief Bit reader over a caller-provided buffer (zero-copy).
 */
typedef struct ReBitReader {
    const uint8_t *buf; /**< Source buffer. */
    size_t bit_cap;     /**< Buffer size in bits (byte_len * 8). */
    size_t bit_pos;     /**< Next bit to read. */
    bool err;           /**< True after an out-of-bounds read attempt. */
} ReBitReader;

/**
 * @brief Initializes the reader.
 * @param r Reader.
 * @param buf Source buffer.
 * @param byte_len Buffer size in bytes.
 */
void re_br_init(ReBitReader *r, const void *buf, size_t byte_len);

/**
 * @brief Reads n bits (n in 0..64), MSB-first.
 * @param r Reader.
 * @param n Bit count.
 * @param out Receives the value (0 on failure).
 * @return False when the read would exceed the buffer (err is set).
 */
bool re_br_read_bits(ReBitReader *r, unsigned n, uint64_t *out);

/** @brief Reads a full 32-bit value. */
bool re_br_read_u32(ReBitReader *r, uint32_t *out);

/** @brief Reads a full 64-bit value. */
bool re_br_read_u64(ReBitReader *r, uint64_t *out);

/** @brief Reads a 16-bit two's complement value. */
bool re_br_read_i16(ReBitReader *r, int16_t *out);

/** @brief Reads the 32-bit pattern of a float (endian-neutral). */
bool re_br_read_f32(ReBitReader *r, float *out);

/** @return Bits remaining (0 when exhausted). */
size_t re_br_remaining(const ReBitReader *r);

/** @return True when a read failed (malformed or truncated input). */
bool re_br_err(const ReBitReader *r);

#endif /* RE_BITSTREAM_H */
