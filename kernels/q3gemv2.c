// Tight repacked 3-bit 8x8 gemv: 8 accumulators register-resident across the K loop.
// Leaner unpack: one 16-byte low-2 load feeds 4 rows; high-1 handled with minimal ops.
#include <arm_neon.h>
#include <stdint.h>
// y[0..7] += dequant3(W[8 rows]) . x  over nk columns (multiple of 32)
void q3_gemv_8(const uint8_t* q2, const uint8_t* q1, const int8_t* x, int nk, int32_t* y) {
    int32x4_t a0=vdupq_n_s32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    const uint8x16_t m2=vdupq_n_u8(3); const int8x16_t bias=vdupq_n_s8(4);
    for (int k=0;k<nk;k+=16){
        int8x16_t xv=vld1q_s8(x+k);
        // 4 rows per 16-byte low-2 load (16 bytes = 4 rows x 16 cols, 4 cols/byte)
        #define ROW2(rr, ld2, ld1, A, B) do{ \
            uint8x16_t lo=vld1q_u8(q2+(ld2)); uint8x16_t hh=vld1q_u8(q1+(ld1)); \
            int8x16_t wa=vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(lo,m2), vshlq_n_u8(vandq_u8(hh,vdupq_n_u8(1)),2))),bias); \
            int8x16_t wb=vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(vshrq_n_u8(lo,2),m2), vshlq_n_u8(vandq_u8(vshrq_n_u8(hh,1),vdupq_n_u8(1)),2))),bias); \
            A=vdotq_s32(A,wa,xv); B=vdotq_s32(B,wb,xv); }while(0)
        ROW2(0, k*0+ (k/16)*64,      (k/16)*32,      a0,a1);
        ROW2(2, (k/16)*64+16,        (k/16)*32+8,    a2,a3);
        ROW2(4, (k/16)*64+32,        (k/16)*32+16,   a4,a5);
        ROW2(6, (k/16)*64+48,        (k/16)*32+24,   a6,a7);
    }
    y[0]=vaddvq_s32(a0); y[1]=vaddvq_s32(a1); y[2]=vaddvq_s32(a2); y[3]=vaddvq_s32(a3);
    y[4]=vaddvq_s32(a4); y[5]=vaddvq_s32(a5); y[6]=vaddvq_s32(a6); y[7]=vaddvq_s32(a7);
}
