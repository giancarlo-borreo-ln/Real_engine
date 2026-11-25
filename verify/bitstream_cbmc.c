/**
 * @file bitstream_cbmc.c
 * @brief CBMC proof: the bit reader never reads outside the buffer for
 *        any symbolic buffer content, length and read sequence; the
 *        writer never writes outside its buffer and latches oom.
 *
 * CBMC's --pointer-overflow-check / bounds instrumentation makes any
 * out-of-bounds access a proof failure automatically; the asserts pin
 * the behavioral contract. Run:
 *   cbmc bitstream_cbmc.c --unwind 32 --unwinding-assertions \
 *        --pointer-overflow-check --unsigned-overflow-check \
 *        --signed-overflow-check
 */
#include "harness_stub.h"

#include "../src/re_bitstream.c"

#include "harness_stub.h"

#include "../src/re_bitstream.h"

/** Arbitrary untrusted input buffer. */
static uint8_t net_buf[16];

int main(void) {
    for (int i = 0; i < 16; i++) {
        net_buf[i] = nondet_uchar();
    }

    /* Reader: any length, any sequence of read widths. */
    {
        size_t len = nondet_size_t();
        __CPROVER_assume(len <= sizeof net_buf);
        ReBitReader r;
        re_br_init(&r, net_buf, len);
        __CPROVER_assert(r.bit_cap == len * 8u, "bit_cap is len*8");
        for (int i = 0; i < 8; i++) {
            unsigned n = nondet_uint();
            __CPROVER_assume(n <= 64u);
            uint64_t out = 0u;
            bool ok = re_br_read_bits(&r, n, &out);
            __CPROVER_assert(r.bit_pos <= r.bit_cap, "position never past the end");
            if (!ok) {
                __CPROVER_assert(r.err, "failure sets the error flag");
            }
        }
        __CPROVER_assert(re_br_remaining(&r) <= r.bit_cap, "remaining bounded");
    }

    /* Writer: any write sequence stays inside the buffer or sets oom. */
    {
        ReBitWriter w;
        re_bw_init(&w, net_buf, sizeof net_buf);
        for (int i = 0; i < 8; i++) {
            unsigned n = nondet_uint();
            __CPROVER_assume(n <= 64u);
            re_bw_write_bits(&w, nondet_ulong(), n);
            __CPROVER_assert(w.bit_len <= w.bit_cap || w.oom,
                             "writer never exceeds capacity");
            __CPROVER_assert(w.oom || w.bit_len <= 128u, "fits in 16 bytes");
        }
        __CPROVER_assert(re_bw_byte_len(&w) <= sizeof net_buf, "byte length bounded");
    }

    /* Round trip: any two consecutive writes reproduce on read. */
    {
        unsigned n1 = nondet_uint();
        unsigned n2 = nondet_uint();
        __CPROVER_assume(n1 <= 32u && n2 <= 32u);
        uint64_t v1 = nondet_ulong(), v2 = nondet_ulong();
        ReBitWriter w;
        re_bw_init(&w, net_buf, sizeof net_buf);
        re_bw_write_bits(&w, v1, n1);
        re_bw_write_bits(&w, v2, n2);
        __CPROVER_assume(!w.oom);
        ReBitReader r;
        re_br_init(&r, net_buf, re_bw_byte_len(&w));
        uint64_t o1 = 0u, o2 = 0u;
        bool ok1 = re_br_read_bits(&r, n1, &o1);
        bool ok2 = re_br_read_bits(&r, n2, &o2);
        __CPROVER_assert(ok1 && ok2, "reads of written bits succeed");
        if (n1 > 0u) {
            __CPROVER_assert(o1 == (v1 & ((n1 == 64u) ? ~0u : ((1u << n1) - 1u))),
                             "first value round-trips");
        }
        if (n2 > 0u) {
            __CPROVER_assert(o2 == (v2 & ((n2 == 64u) ? ~0u : ((1u << n2) - 1u))),
                             "second value round-trips");
        }
    }
    return 0;
}
