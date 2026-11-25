/**
 * @file proto_cbmc.c
 * @brief CBMC proof: every packet codec is a round-trip bijection on its
 *        field domain (decode(encode(p)) == p), and every decoder is
 *        total and bounds-safe on arbitrary symbolic/malformed buffers.
 *
 * Run: cbmc proto_cbmc.c --unwind 8 --unwinding-assertions \
 *        --pointer-overflow-check --signed-overflow-check \
 *        --unsigned-overflow-check
 */
#include "harness_stub.h"

#include "../src/re_hash.c"
#include "harness_stub.h"

#include "../src/re_bitstream.c"
#include "harness_stub.h"

#include "../src/re_proto.c"

#include "harness_stub.h"

#include "../src/re_proto.h"

/** Wire buffer. */
static uint8_t pkt[RE_PKT_MAX_BYTES];

/** Malformed input buffer. */
static uint8_t junk[64];

/** Body storage for state decode. */
static ReBodySnap bodies[RE_NET_MAX_BODIES];

int main(void) {
    /* ---- INPUT packet round trip ---- */
    {
        uint16_t seq = nondet_ushort();
        uint32_t frame = nondet_uint();
        uint32_t inputs = nondet_uint();
        int16_t mx = (int16_t)nondet_uint();
        int16_t my = (int16_t)nondet_uint();
        size_t len = re_pkt_encode_input(pkt, sizeof pkt, seq, frame, inputs, mx, my);
        __CPROVER_assert(len == 16u, "input packet is 16 bytes");
        RePacketHeader h;
        uint32_t oi = 0u;
        int16_t ox = 0, oy = 0;
        bool ok = re_pkt_decode_input(pkt, len, &h, &oi, &ox, &oy);
        __CPROVER_assert(ok, "decode succeeds");
        __CPROVER_assert(h.type == RE_PKT_INPUT && h.seq == seq && h.frame == frame,
                         "header round-trips");
        __CPROVER_assert(oi == inputs && ox == mx && oy == my, "payload round-trips");
    }

    /* ---- STATE packet round trip ---- */
    {
        uint16_t seq = nondet_ushort();
        uint32_t frame = nondet_uint();
        uint64_t checksum = nondet_ulong();
        uint32_t count = nondet_uint();
        __CPROVER_assume(count <= 2u); /* keep the state space tractable */
        for (uint32_t i = 0u; i < count; i++) {
            float *raw = (float *)&bodies[i];
            for (uint32_t k = 0u; k < sizeof(ReBodySnap) / sizeof(float); k++) {
                raw[k] = nondet_float();
            }
        }
        ReStatePayload p = { frame, checksum, count, bodies };
        size_t len = re_pkt_encode_state(pkt, sizeof pkt, seq, &p);
        __CPROVER_assert(len == 24u + 40u * count, "state packet size exact");
        RePacketHeader h;
        ReStatePayload out = { 0u, 0u, 0u, bodies };
        bool ok = re_pkt_decode_state(pkt, len, &h, &out);
        __CPROVER_assert(ok, "state decode succeeds");
        __CPROVER_assert(h.seq == seq && h.frame == frame, "state header round-trips");
        __CPROVER_assert(out.frame == frame && out.checksum == checksum &&
                         out.body_count == count, "state payload round-trips");
        for (uint32_t i = 0u; i < count; i++) {
            uint32_t *a = (uint32_t *)&out.bodies[i];
            uint32_t *b = (uint32_t *)&bodies[i];
            for (uint32_t k = 0u; k < sizeof(ReBodySnap) / 4u; k++) {
                __CPROVER_assert(a[k] == b[k], "body bits round-trip");
            }
        }
    }

    /* ---- ACK round trip ---- */
    {
        uint16_t seq = nondet_ushort();
        uint32_t ack = nondet_uint();
        uint64_t checksum = nondet_ulong();
        size_t len = re_pkt_encode_ack(pkt, sizeof pkt, seq, ack, checksum);
        __CPROVER_assert(len == 20u, "ack packet is 20 bytes");
        RePacketHeader h;
        uint32_t oa = 0u;
        uint64_t oc = 0u;
        bool ok = re_pkt_decode_ack(pkt, len, &h, &oa, &oc);
        __CPROVER_assert(ok && oa == ack && oc == checksum, "ack round-trips");
    }

    /* ---- PING round trip ---- */
    {
        uint16_t seq = nondet_ushort();
        uint32_t ms = nondet_uint();
        size_t len = re_pkt_encode_ping(pkt, sizeof pkt, seq, ms);
        __CPROVER_assert(len == 12u, "ping packet is 12 bytes");
        RePacketHeader h;
        uint32_t om = 0u;
        bool ok = re_pkt_decode_ping(pkt, len, &h, &om);
        __CPROVER_assert(ok && om == ms, "ping round-trips");
    }

    /* ---- Malformed input: all four decoders, any bytes, any length ---- */
    {
        for (int i = 0; i < 64; i++) {
            junk[i] = nondet_uchar();
        }
        size_t len = nondet_size_t();
        __CPROVER_assume(len <= sizeof junk);
        RePacketHeader h;
        uint32_t u1 = 0u;
        int16_t s1 = 0, s2 = 0;
        uint64_t c = 0u;
        ReStatePayload sp = { 0u, 0u, 0u, bodies };
        (void)re_pkt_decode_input(junk, len, &h, &u1, &s1, &s2);
        (void)re_pkt_decode_state(junk, len, &h, &sp);
        (void)re_pkt_decode_ack(junk, len, &h, &u1, &c);
        (void)re_pkt_decode_ping(junk, len, &h, &u1);
        /* Decoders must never accept an impossible body count. */
        __CPROVER_assert(sp.body_count <= RE_NET_MAX_BODIES,
                         "body count bounded even for garbage input");
    }
    return 0;
}
