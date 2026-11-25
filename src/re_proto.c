/**
 * @file re_proto.c
 * @brief GGPO packet codec: fixed-width, MSB-first and lossless.
 *
 * Every packet starts with the same 64-bit header:
 *
 * @code
 *   8 bits type | 8 bits flags (reserved, always 0) | 16 bits seq | 32 bits frame
 * @endcode
 *
 * followed by the per-type payload:
 *
 * @code
 *   input : 32 inputs_bits | 16 move_x | 16 move_y
 *   state : 32 frame | 64 checksum | 32 body_count | body_count * 320 body bits
 *   ack   : 32 ack_frame | 64 checksum
 *   ping  : 32 client_ms
 * @endcode
 *
 * A body is 10 float bit patterns in order: pos.xyz, quat.xyzw,
 * lin_vel.xyz (40 bytes). Floats travel as their raw 32-bit pattern, so
 * the codec is bit-exact for every float, including NaN payloads.
 *
 * Decoding never reads past the buffer: the bit reader latches an error
 * flag instead of advancing, and every field read is checked. Trailing
 * bytes are ignored; truncated input fails. Output parameters are only
 * committed once the whole packet validates; a state packet reads body
 * bytes straight into the caller's array as it goes (rejected packets
 * may leave partial bodies there, never a claimed body_count).
 */
#include "re_proto.h"

/**
 * @brief Writes the 64-bit common header.
 * @param w Bit writer.
 * @param type Packet type.
 * @param seq Sequence number.
 * @param frame Frame number.
 */
static void write_header(ReBitWriter *w, RePacketType type, uint16_t seq, uint32_t frame) {
    re_bw_write_bits(w, (uint64_t)type, 8u);
    re_bw_write_bits(w, 0u, 8u); /* Flags byte: reserved, must stay 0. */
    re_bw_write_bits(w, (uint64_t)seq, 16u);
    re_bw_write_u32(w, frame);
}

/**
 * @brief Reads the common header and checks the packet type.
 *
 * The type is validated before the cast to RePacketType so that a
 * corrupt octet can never produce an out-of-range enum value.
 * @param r Bit reader.
 * @param expect Type the caller's decoder handles.
 * @param[out] hdr Receives the header on success.
 * @return False on truncation or on a packet of another type.
 */
static bool read_header(ReBitReader *r, RePacketType expect, RePacketHeader *hdr) {
    uint64_t raw_type = 0;
    uint64_t flags = 0;
    uint64_t seq = 0;
    uint32_t frame = 0;

    if (!re_br_read_bits(r, 8u, &raw_type)) {
        return false;
    }
    if (!re_br_read_bits(r, 8u, &flags)) { /* Reserved: read, then ignored. */
        return false;
    }
    if (!re_br_read_bits(r, 16u, &seq)) {
        return false;
    }
    if (!re_br_read_u32(r, &frame)) {
        return false;
    }
    if (raw_type != (uint64_t)expect) {
        return false;
    }
    hdr->type = expect;
    hdr->seq = (uint16_t)seq;
    hdr->frame = frame;
    return true;
}

/**
 * @brief Writes the 10 float bit patterns of one body.
 * @param w Bit writer.
 * @param b Body snapshot.
 */
static void write_body(ReBitWriter *w, const ReBodySnap *b) {
    re_bw_write_f32(w, b->pos.x);
    re_bw_write_f32(w, b->pos.y);
    re_bw_write_f32(w, b->pos.z);
    re_bw_write_f32(w, b->quat.x);
    re_bw_write_f32(w, b->quat.y);
    re_bw_write_f32(w, b->quat.z);
    re_bw_write_f32(w, b->quat.w);
    re_bw_write_f32(w, b->lin_vel.x);
    re_bw_write_f32(w, b->lin_vel.y);
    re_bw_write_f32(w, b->lin_vel.z);
}

/**
 * @brief Reads the 10 float bit patterns of one body.
 *
 * Fields are read in wire order; on the first failure the remaining
 * fields are skipped and the caller discards the packet anyway.
 * @param r Bit reader.
 * @param[out] b Body snapshot.
 * @return False when the buffer ends inside the body.
 */
static bool read_body(ReBitReader *r, ReBodySnap *b) {
    return re_br_read_f32(r, &b->pos.x) && re_br_read_f32(r, &b->pos.y) &&
           re_br_read_f32(r, &b->pos.z) && re_br_read_f32(r, &b->quat.x) &&
           re_br_read_f32(r, &b->quat.y) && re_br_read_f32(r, &b->quat.z) &&
           re_br_read_f32(r, &b->quat.w) && re_br_read_f32(r, &b->lin_vel.x) &&
           re_br_read_f32(r, &b->lin_vel.y) && re_br_read_f32(r, &b->lin_vel.z);
}

size_t re_pkt_encode_input(uint8_t *buf, size_t cap, uint16_t seq, uint32_t frame,
                           uint32_t inputs_bits, int16_t move_x, int16_t move_y) {
    ReBitWriter w;

    if (buf == NULL) {
        return 0u;
    }
    re_bw_init(&w, buf, cap);
    write_header(&w, RE_PKT_INPUT, seq, frame);
    re_bw_write_u32(&w, inputs_bits);
    re_bw_write_i16(&w, move_x);
    re_bw_write_i16(&w, move_y);
    return re_bw_oom(&w) ? 0u : re_bw_byte_len(&w);
}

bool re_pkt_decode_input(const uint8_t *buf, size_t len, RePacketHeader *hdr,
                         uint32_t *inputs_bits, int16_t *move_x, int16_t *move_y) {
    ReBitReader r;
    RePacketHeader local;
    uint32_t bits = 0;
    int16_t mx = 0;
    int16_t my = 0;

    if (buf == NULL || hdr == NULL || inputs_bits == NULL || move_x == NULL ||
        move_y == NULL) {
        return false;
    }
    re_br_init(&r, buf, len);
    if (!read_header(&r, RE_PKT_INPUT, &local)) {
        return false;
    }
    if (!re_br_read_u32(&r, &bits) || !re_br_read_i16(&r, &mx) || !re_br_read_i16(&r, &my)) {
        return false;
    }
    if (re_br_err(&r)) {
        return false;
    }
    *hdr = local;
    *inputs_bits = bits;
    *move_x = mx;
    *move_y = my;
    return true;
}

size_t re_pkt_encode_state(uint8_t *buf, size_t cap, uint16_t seq, const ReStatePayload *p) {
    ReBitWriter w;
    uint32_t i;

    if (buf == NULL || p == NULL) {
        return 0u;
    }
    if (p->body_count > RE_NET_MAX_BODIES) {
        return 0u;
    }
    if (p->body_count > 0u && p->bodies == NULL) {
        return 0u;
    }
    re_bw_init(&w, buf, cap);
    write_header(&w, RE_PKT_STATE, seq, p->frame);
    re_bw_write_u32(&w, p->frame);
    re_bw_write_u64(&w, p->checksum);
    re_bw_write_u32(&w, p->body_count);
    for (i = 0u; i < p->body_count; ++i) {
        write_body(&w, &p->bodies[i]);
    }
    return re_bw_oom(&w) ? 0u : re_bw_byte_len(&w);
}

bool re_pkt_decode_state(const uint8_t *buf, size_t len, RePacketHeader *hdr,
                         ReStatePayload *out) {
    ReBitReader r;
    RePacketHeader local;
    ReBodySnap *store;
    uint32_t frame = 0;
    uint32_t count = 0;
    uint32_t i;
    uint64_t checksum = 0;

    if (buf == NULL || hdr == NULL || out == NULL) {
        return false;
    }
    re_br_init(&r, buf, len);
    if (!read_header(&r, RE_PKT_STATE, &local)) {
        return false;
    }
    if (!re_br_read_u32(&r, &frame) || !re_br_read_u64(&r, &checksum) ||
        !re_br_read_u32(&r, &count)) {
        return false;
    }
    if (count > RE_NET_MAX_BODIES) {
        return false;
    }
    /* The payload struct exposes a writable bodies array (see the
     * re_pkt_decode_state contract): decoding fills the caller's storage. */
    store = out->bodies;
    if (count > 0u && store == NULL) {
        return false; /* No caller storage: cannot materialize the bodies. */
    }
    for (i = 0u; i < count; ++i) {
        if (!read_body(&r, &store[i])) {
            return false;
        }
    }
    if (re_br_err(&r)) {
        return false;
    }
    out->frame = frame;
    out->checksum = checksum;
    out->body_count = count;
    *hdr = local;
    return true;
}

size_t re_pkt_encode_ack(uint8_t *buf, size_t cap, uint16_t seq, uint32_t ack_frame,
                         uint64_t checksum) {
    ReBitWriter w;

    if (buf == NULL) {
        return 0u;
    }
    re_bw_init(&w, buf, cap);
    write_header(&w, RE_PKT_ACK, seq, ack_frame);
    re_bw_write_u32(&w, ack_frame);
    re_bw_write_u64(&w, checksum);
    return re_bw_oom(&w) ? 0u : re_bw_byte_len(&w);
}

bool re_pkt_decode_ack(const uint8_t *buf, size_t len, RePacketHeader *hdr,
                       uint32_t *ack_frame, uint64_t *checksum) {
    ReBitReader r;
    RePacketHeader local;
    uint32_t frame = 0;
    uint64_t sum = 0;

    if (buf == NULL || hdr == NULL || ack_frame == NULL || checksum == NULL) {
        return false;
    }
    re_br_init(&r, buf, len);
    if (!read_header(&r, RE_PKT_ACK, &local)) {
        return false;
    }
    if (!re_br_read_u32(&r, &frame) || !re_br_read_u64(&r, &sum)) {
        return false;
    }
    if (re_br_err(&r)) {
        return false;
    }
    *hdr = local;
    *ack_frame = frame;
    *checksum = sum;
    return true;
}

size_t re_pkt_encode_ping(uint8_t *buf, size_t cap, uint16_t seq, uint32_t client_ms) {
    ReBitWriter w;

    if (buf == NULL) {
        return 0u;
    }
    re_bw_init(&w, buf, cap);
    write_header(&w, RE_PKT_PING, seq, 0u);
    re_bw_write_u32(&w, client_ms);
    return re_bw_oom(&w) ? 0u : re_bw_byte_len(&w);
}

bool re_pkt_decode_ping(const uint8_t *buf, size_t len, RePacketHeader *hdr,
                        uint32_t *client_ms) {
    ReBitReader r;
    RePacketHeader local;
    uint32_t ms = 0;

    if (buf == NULL || hdr == NULL || client_ms == NULL) {
        return false;
    }
    re_br_init(&r, buf, len);
    if (!read_header(&r, RE_PKT_PING, &local)) {
        return false;
    }
    if (!re_br_read_u32(&r, &ms)) {
        return false;
    }
    if (re_br_err(&r)) {
        return false;
    }
    *hdr = local;
    *client_ms = ms;
    return true;
}
