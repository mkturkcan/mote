// Prototype: repacked 3-bit 8x8 gemv inner loop for Cortex-A76.
// Goal: keep 8 output-row accumulators in vector lanes (ONE reduction at the end, unlike the
// per-row vec_dot's 16 addv/super-block) so the kernel becomes BANDWIDTH-bound at 3-bit.
// We measure the steady-state inner loop with llvm-mca; correctness vs scalar checked separately.
//
// Layout (repacked, per 32-input-column panel, 8 rows interleaved):
//   - q2[8 rows][32 cols] low-2-bits, packed 4 cols/byte  -> 8*8 = 64 bytes
//   - q1[8 rows][32 cols] high-1-bit, packed 8 cols/byte  -> 8*4 = 32 bytes
//   - one fp16 scale per (row, 32-col block)
// Activations: Q8 (int8), 32 values, broadcast across the 8 rows.
#include <arm_neon.h>
#include <stdint.h>

// process one 16-column chunk for 8 rows; accumulate into acc[0..7] (int32x4 each, lanes summed later)
// q2p: pointer to low-2-bit bytes (8 rows x 16 cols / 4 = 32 bytes), q1p: high-1-bit (8x16/8 = 16 bytes)
// a: int8x16 activation for these 16 columns (same for all rows)
void q3_8x16_chunk(const uint8_t* q2p, const uint8_t* q1p, int8x16_t a, int32x4_t acc[8]) {
    const uint8x16_t m2 = vdupq_n_u8(0x03);
    for (int r = 0; r < 8; r += 2) {
        // each iteration handles 2 rows (16 cols each) to amortise loads
        // low-2 bits: 8 bytes hold 2 rows x 16 cols (4 cols/byte). load 8 bytes -> 16 lanes via shifts.
        uint8x8_t lo8 = vld1_u8(q2p + r*8);           // 8 bytes = 2 rows x 16 cols low-2
        uint8x16_t lo = vcombine_u8(lo8, lo8);
        // hi-1 bit: 4 bytes hold 2 rows x 16 cols (8 cols/byte)
        uint8x8_t hi8 = vld1_u8(q1p + r*4);
        uint8x16_t hi = vcombine_u8(hi8, hi8);
        // unpack row r: low2 = (lo & 3) for the 16 cols laid 4/byte -> expand
        int8x16_t w0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(lo, m2),
                                                    vshlq_n_u8(vandq_u8(hi, vdupq_n_u8(0x01)), 2)));
        int8x16_t w1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(vshrq_n_u8(lo,2), m2),
                                                    vshlq_n_u8(vandq_u8(vshrq_n_u8(hi,1), vdupq_n_u8(0x01)), 2)));
        // center 3-bit [0..7] -> [-4..3]
        w0 = vsubq_s8(w0, vdupq_n_s8(4));
        w1 = vsubq_s8(w1, vdupq_n_s8(4));
        acc[r]   = vdotq_s32(acc[r],   w0, a);
        acc[r+1] = vdotq_s32(acc[r+1], w1, a);
    }
}
