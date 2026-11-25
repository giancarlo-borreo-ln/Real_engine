/**
 * @file re_proto.h
 * @brief GGPO packet codec: input, state snapshot, ack and ping packets.
 *
 * Every field has a fixed bit width and the wire format is
 * endian-neutral (bitstream is MSB-first). Encoding is lossless, so
 * decode(encode(p)) == p for every packet: this is the round-trip
 * bijection the CBMC harness proves field by field. Decoding validates
 * all counts and lengths and returns false on malformed input without
 * ever reading out of bounds.
 */
#ifndef RE_PROTO_H
#define RE_PROTO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "re_bitstream.h"
#include "re_vec.h"

/** Packet types. */
typedef enum RePacketType {
    RE_PKT_INPUT = 0, /**< Input for one frame. */
    RE_PKT_STATE = 1, /**< Authoritative state snapshot. */
    RE_PKT_ACK = 2,   /**< Acknowledges a frame. */
    RE_PKT_PING = 3   /**< Latency probe. */
} RePacketType;

/** Maximum encoded packet size in bytes. */
#define RE_PKT_MAX_BYTES 8192u

/** Maximum bodies per state snapshot (bounded for proofs and memory). */
#define RE_NET_MAX_BODIES 128u

/**
 * @brief Decoded packet header.
 */
typedef struct RePacketHeader {
    RePacketType type; /**< Packet type. */
    uint16_t seq;      /**< Packet sequence number. */
    uint32_t frame;    /**< Game frame the packet refers to. */
} RePacketHeader;

/**
 * @brief Per-body state snapshot (lossless: 40 bytes per body on wire).
 */
typedef struct ReBodySnap {
    ReVec3 pos;     /**< Position. */
    ReQuat quat;    /**< Orientation. */
    ReVec3 lin_vel; /**< Linear velocity. */
} ReBodySnap;

/**
 * @brief Payload of a state packet.
 *
 * bodies is writable because decode fills caller storage through it;
 * encode treats it as read-only by convention.
 */
typedef struct ReStatePayload {
    uint32_t frame;       /**< Frame of the snapshot. */
    uint64_t checksum;    /**< CRC-64 of the full authoritative state. */
    uint32_t body_count;  /**< Bodies in bodies[]. */
    ReBodySnap *bodies;   /**< Snapshot array (body_count entries). */
} ReStatePayload;

/**
 * @brief Encodes an input packet.
 * @param buf Destination buffer.
 * @param cap Buffer capacity in bytes.
 * @param seq Sequence number.
 * @param frame Frame number.
 * @param inputs_bits Input bit flags (buttons).
 * @param move_x Horizontal move, two's complement 16-bit.
 * @param move_y Vertical move, two's complement 16-bit.
 * @return Encoded size in bytes, or 0 when cap is too small.
 */
size_t re_pkt_encode_input(uint8_t *buf, size_t cap, uint16_t seq, uint32_t frame,
                           uint32_t inputs_bits, int16_t move_x, int16_t move_y);

/**
 * @brief Decodes an input packet.
 * @param buf Packet bytes.
 * @param len Packet byte length.
 * @param hdr Receives the header.
 * @param inputs_bits Receives the input flags.
 * @param move_x Receives the horizontal move.
 * @param move_y Receives the vertical move.
 * @return False on malformed input (no out-of-bounds reads possible).
 */
bool re_pkt_decode_input(const uint8_t *buf, size_t len, RePacketHeader *hdr,
                         uint32_t *inputs_bits, int16_t *move_x, int16_t *move_y);

/**
 * @brief Encodes a state packet.
 * @param buf Destination buffer.
 * @param cap Buffer capacity in bytes.
 * @param seq Sequence number.
 * @param p Snapshot payload (body_count <= RE_NET_MAX_BODIES).
 * @return Encoded size in bytes, or 0 when cap is too small.
 */
size_t re_pkt_encode_state(uint8_t *buf, size_t cap, uint16_t seq, const ReStatePayload *p);

/**
 * @brief Decodes a state packet.
 * @param buf Packet bytes.
 * @param len Packet byte length.
 * @param hdr Receives the header.
 * @param out Receives the payload (bodies points at caller storage with
 *            RE_NET_MAX_BODIES entries).
 * @return False on malformed input or body_count > RE_NET_MAX_BODIES.
 */
bool re_pkt_decode_state(const uint8_t *buf, size_t len, RePacketHeader *hdr,
                         ReStatePayload *out);

/**
 * @brief Encodes an ack packet.
 * @param buf Destination buffer.
 * @param cap Buffer capacity in bytes.
 * @param seq Sequence number.
 * @param ack_frame Frame being acknowledged.
 * @param checksum Checksum of the acknowledged state.
 * @return Encoded size in bytes, or 0 when cap is too small.
 */
size_t re_pkt_encode_ack(uint8_t *buf, size_t cap, uint16_t seq,
                         uint32_t ack_frame, uint64_t checksum);

/**
 * @brief Decodes an ack packet.
 * @param buf Packet bytes.
 * @param len Packet byte length.
 * @param hdr Receives the header.
 * @param ack_frame Receives the acknowledged frame.
 * @param checksum Receives the checksum.
 * @return False on malformed input.
 */
bool re_pkt_decode_ack(const uint8_t *buf, size_t len, RePacketHeader *hdr,
                       uint32_t *ack_frame, uint64_t *checksum);

/**
 * @brief Encodes a ping packet.
 * @param buf Destination buffer.
 * @param cap Buffer capacity in bytes.
 * @param seq Sequence number.
 * @param client_ms Client monotonic timestamp (milliseconds).
 * @return Encoded size in bytes, or 0 when cap is too small.
 */
size_t re_pkt_encode_ping(uint8_t *buf, size_t cap, uint16_t seq, uint32_t client_ms);

/**
 * @brief Decodes a ping packet.
 * @param buf Packet bytes.
 * @param len Packet byte length.
 * @param hdr Receives the header.
 * @param client_ms Receives the client timestamp.
 * @return False on malformed input.
 */
bool re_pkt_decode_ping(const uint8_t *buf, size_t len, RePacketHeader *hdr,
                        uint32_t *client_ms);

#endif /* RE_PROTO_H */
