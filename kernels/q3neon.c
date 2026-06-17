// Hand-written NEON+dotprod q3_K repacked gemv (8 columns, blocklen=4), mirroring the proven
// ggml_gemv_q6_K_8x4_q8_K ARM kernel. Validates BIT-EXACT vs the real ggml_vec_dot_q3_K_q8_K.
// Build:  aarch64-linux-gnu-gcc -O3 -mcpu=cortex-a76 -static q3neon.c -o /tmp/q3neon -lm
// Run:    qemu-aarch64-static /tmp/q3neon
//
// Key identity: q3_K weight = ((qs>>shift)&3) - (hbit?0:4) = (3bit value 0..7) - 4  (constant -4).
//   => dot = Σ_sb scale[sb]*(Σ_p val[p]*q8[p]) - 4*Σ_sb scale[sb]*bsum[sb]   (bias via q8 bsums)
//   where scale[sb] = (6bit unpacked) - 32 (signed), stored interleaved at repack time.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <arm_neon.h>
#define QK_K 256

typedef struct { uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; float d; } block_q3_K;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;

// repacked 8-col block, blocklen=4 interleave; scales kept PACKED (12B/col) so the block fits the
// original 8*block_q3_K size (880B; unpacked-128 would overflow). Unpacked in-kernel (scalar).
// d as float here (cancels on both sides, isolates integer path).
typedef struct {
    float   d[8];
    uint8_t hmask[8 * QK_K / 8];// 4-byte interleave
    uint8_t qs[8 * QK_K / 4];   // 4-byte interleave
    uint8_t scales[12 * 8];     // packed 6-bit, 12 bytes/col
} block_q3_Kx8;

// ---------- reference: real ggml_vec_dot_q3_K_q8_K over nb super-blocks ----------
static float ref_dot_q3K(const block_q3_K * xs, const block_q8_K * ys, int nb) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    int8_t aux8[QK_K]; int16_t aux16[8]; int32_t aux32[8]; float sums[8];
    uint32_t auxs[4]; const int8_t * sc = (const int8_t*)auxs;
    memset(sums, 0, sizeof(sums));
    for (int i = 0; i < nb; ++i) {
        const uint8_t *q3 = xs[i].qs, *hm = xs[i].hmask; const int8_t *q8 = ys[i].qs;
        memset(aux32, 0, sizeof(aux32)); int8_t *a = aux8; uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4); a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4); a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4); a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4); a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8; memcpy(auxs, xs[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (sc[j] - 32) * aux16[l]; q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (sc[j] - 32) * aux16[l]; q8 += 8; a += 8;
        }
        const float d = xs[i].d * ys[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    float sumf = 0; for (int l = 0; l < 8; ++l) sumf += sums[l]; return sumf;
}

// ---------- repack: 8 q3_K rows -> block_q3_Kx8 (blocklen=4, signed interleaved scales) ----------
static void make_block_q3_Kx8(const block_q3_K in[8], block_q3_Kx8 * o) {
    const int B = 4;
    for (int i = 0; i < 8; i++) o->d[i] = in[i].d;
    const int qs_end = (QK_K/4) * 8 / B;
    for (int i = 0; i < qs_end; i++) memcpy(&o->qs[i*B], &in[i%8].qs[(i/8)*B], B);
    const int hm_end = (QK_K/8) * 8 / B;
    for (int i = 0; i < hm_end; i++) memcpy(&o->hmask[i*B], &in[i%8].hmask[(i/8)*B], B);
    // COL-MAJOR scale storage: scales[col*12 + t] (12 contiguous bytes/col) -> cheap scalar gather.
    for (int col = 0; col < 8; col++)
        for (int t = 0; t < 12; t++) o->scales[col * 12 + t] = in[col].scales[t];
}

// scalar unpack of ONE column's 16 packed 6-bit scales (col-major) -> 16 UNSIGNED values. For the generic.
static inline void unpack_col_flat(const uint8_t * scales, int col, int8_t out16[16]) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    uint32_t auxs[4]; memcpy(auxs, scales + col * 12, 12);
    uint32_t tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t * s8 = (const int8_t *) auxs;
    for (int t = 0; t < 16; t++) out16[t] = s8[t];
}

// scalar unpack of ONE column into the interleaved SIGNED scbuf[sb*8 + col] (= sc - 32). For the hand kernel.
// This runs on the INTEGER pipes; software-pipelined into the NEON loop it hides under the vector work.
static inline void unpack_col_into(const uint8_t * scales, int col, int8_t * scbuf) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    uint32_t auxs[4]; memcpy(auxs, scales + col * 12, 12);
    uint32_t tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t * s8 = (const int8_t *) auxs;
    for (int sb = 0; sb < 16; sb++) scbuf[sb * 8 + col] = (int8_t) (s8[sb] - 32);
}

// ---------- the hand NEON+dotprod gemv ----------
// ---------- OPTIMIZED: load qs/hmask ONCE per byte, fan out the 4 shifts (fixed immediates) ----------
// Each qs byte holds 4 weights (shifts 0/2/4/6 -> 4 sub-blocks); each hmask byte holds 8 bits.
// Group the 16 sub-blocks into 4 "qgroups" sharing the same 16 qs bytes; one load feeds 4 sdots.
__attribute__((noinline)) static void q3_gemv_neon_opt(int nb, const block_q3_Kx8 * b_ptr, const block_q8_K * q8_ptr, float out[8]) {
    // qgroup table: {qbyte0, hbyte0, base_bit, sb_base};  sb = sb_base + 2*s, shift=2*s, bit=base_bit+s
    static const int QG[4][4] = { {0,0,0,0}, {16,16,0,1}, {32,0,4,8}, {48,16,4,9} };
    const uint8x16_t m3 = vdupq_n_u8(3), m1 = vdupq_n_u8(1);
    float32x4_t acc_f32[2] = { vdupq_n_f32(0), vdupq_n_f32(0) };

    // Double-buffered, SIGNED, interleaved scales (scbuf[buf][sb*8+col]). The scalar unpack of the NEXT
    // super-block is woven into THIS super-block's NEON loop -> hides on the idle integer pipes.
    int8_t scbuf[2][16*8];
    for (int col = 0; col < 8; col++) unpack_col_into(b_ptr[0].scales, col, scbuf[0]); // prologue: SB 0

    for (int b = 0; b < nb; b++) {
        const int cur = b & 1, nxt = (b + 1) & 1;
        const int8_t * sc = scbuf[cur];
        const block_q3_Kx8 * bn = (b + 1 < nb) ? &b_ptr[b + 1] : NULL;

        float32x4_t d0 = vmulq_f32(vld1q_f32(b_ptr[b].d),     vdupq_n_f32(q8_ptr[b].d));
        float32x4_t d1 = vmulq_f32(vld1q_f32(b_ptr[b].d + 4), vdupq_n_f32(q8_ptr[b].d));
        int32x4_t acc[2] = { vdupq_n_s32(0), vdupq_n_s32(0) };
        int32x4_t bias_lo = vdupq_n_s32(0), bias_hi = vdupq_n_s32(0);
        for (int sb = 0; sb < 16; sb++) {
            int16x8_t scv = vmovl_s8(vld1_s8(sc + sb*8));
            int16_t bs = q8_ptr[b].bsums[sb];
            bias_lo = vmlal_n_s16(bias_lo, vget_low_s16(scv),  bs);
            bias_hi = vmlal_n_s16(bias_hi, vget_high_s16(scv), bs);
        }
        bias_lo = vshlq_n_s32(bias_lo, 2);
        bias_hi = vshlq_n_s32(bias_hi, 2);

        for (int qg = 0; qg < 4; qg++) {
            const int qbyte0 = QG[qg][0], hbyte0 = QG[qg][1], base_bit = QG[qg][2], sb_base = QG[qg][3];
            for (int g = 0; g < 2; g++) {
                // --- INTERLEAVED scalar unpack: one column of the NEXT super-block (integer pipes) ---
                if (bn) unpack_col_into(bn->scales, qg * 2 + g, scbuf[nxt]);

                int32x4_t sbacc[4] = { vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0) };
                for (int q = 0; q < 4; q++) {
                    uint8x16_t qsv = vld1q_u8(b_ptr[b].qs    + ((qbyte0/4 + q)*8 + 4*g) * 4);
                    uint8x16_t hmv = vld1q_u8(b_ptr[b].hmask + ((hbyte0/4 + q)*8 + 4*g) * 4);
                    uint8x16_t hb = vshlq_u8(hmv, vdupq_n_s8(-(int8_t)base_bit));
                    const int8_t * q8b = q8_ptr[b].qs + 4*q;
#define DEQDOT(S, SH, HBV) do { \
        uint8x16_t lo = vandq_u8(vshrq_n_u8(qsv, (SH)), m3); \
        int8x16_t w = vreinterpretq_s8_u8(vsliq_n_u8(lo, vandq_u8((HBV), m1), 2)); \
        int8x16_t q8v = (int8x16_t) vld1q_dup_s32((const int32_t *)(q8b + (sb_base + 2*(S))*16)); \
        sbacc[(S)] = vdotq_s32(sbacc[(S)], w, q8v); \
    } while (0)
                    DEQDOT(0, 0, hb);
                    DEQDOT(1, 2, vshrq_n_u8(hb, 1));
                    DEQDOT(2, 4, vshrq_n_u8(hb, 2));
                    DEQDOT(3, 6, vshrq_n_u8(hb, 3));
#undef DEQDOT
                }
                for (int s = 0; s < 4; s++) {
                    const int sb = sb_base + 2*s;
                    int32x4_t scv = vmovl_s16(vget_low_s16(vmovl_s8(vld1_s8(sc + sb*8 + 4*g))));
                    acc[g] = vmlaq_s32(acc[g], sbacc[s], scv);
                }
            }
        }
        acc[0] = vsubq_s32(acc[0], bias_lo);
        acc[1] = vsubq_s32(acc[1], bias_hi);
        acc_f32[0] = vaddq_f32(acc_f32[0], vmulq_f32(vcvtq_f32_s32(acc[0]), d0));
        acc_f32[1] = vaddq_f32(acc_f32[1], vmulq_f32(vcvtq_f32_s32(acc[1]), d1));
    }
    vst1q_f32(out, acc_f32[0]);
    vst1q_f32(out + 4, acc_f32[1]);
}

// EXACT copy of the ggml generic gemv (blocklen=4 IDX, inline packed-scale unpack) to isolate bugs.
static void gen_gemv(int nb, const block_q3_Kx8 * b_ptr, const block_q8_K * a_ptr, float out[8]) {
    int8_t aux8[QK_K];
    for (int j = 0; j < 8; j++) {
        float sumf = 0.0f;
        for (int l = 0; l < nb; l++) {
            uint8_t m = 1; int wi = 0; int qbase = 0;
            for (int nn = 0; nn < QK_K; nn += 128) {
                int shift = 0;
                for (int jj = 0; jj < 4; jj++) {
                    for (int t = 0; t < 16; t++) {
                        const int pq = qbase + t;
                        const uint8_t q = b_ptr[l].qs[((pq / 4) * 8 + j) * 4 + (pq % 4)];
                        const uint8_t h = b_ptr[l].hmask[((t / 4) * 8 + j) * 4 + (t % 4)];
                        aux8[wi + t] = (int8_t) (((q >> shift) & 3) - ((h & m) ? 0 : 4));
                    }
                    for (int t = 0; t < 16; t++) {
                        const int pq = qbase + 16 + t; const int ph = 16 + t;
                        const uint8_t q = b_ptr[l].qs[((pq / 4) * 8 + j) * 4 + (pq % 4)];
                        const uint8_t h = b_ptr[l].hmask[((ph / 4) * 8 + j) * 4 + (ph % 4)];
                        aux8[wi + 16 + t] = (int8_t) (((q >> shift) & 3) - ((h & m) ? 0 : 4));
                    }
                    wi += 32; shift += 2; m <<= 1;
                }
                qbase += 32;
            }
            int8_t scales[16]; unpack_col_flat(b_ptr[l].scales, j, scales);
            int isum = 0;
            for (int sb = 0; sb < 16; sb++) {
                int dot = 0;
                for (int t = 0; t < 16; t++) dot += aux8[sb*16+t] * (int) a_ptr[l].qs[sb*16+t];
                isum += (scales[sb] - 32) * dot;
            }
            sumf += (float) isum * (b_ptr[l].d[j] * a_ptr[l].d);
        }
        out[j] = sumf;
    }
}

typedef struct { float d[4]; int8_t qs[QK_K*4]; int16_t bsums[QK_K/4]; } block_q8_Kx4;
// pack 4 q8_K rows -> q8_Kx4 with the q8_K_4x4 interleave: qs[j] = row[(j%16)/4][(j/16)*4 + (j%4)]
static void pack_q8_Kx4_4x4(const block_q8_K rows[4], block_q8_Kx4 * o) {
    for (int m = 0; m < 4; m++) o->d[m] = rows[m].d;
    for (int j = 0; j < QK_K*4; j++) {
        int src_id = (j % 16) / 4, src_offset = (j / 16) * 4 + (j % 4);
        o->qs[j] = rows[src_id].qs[src_offset];
    }
}
// EXACT ggml gemm (blocklen=4 weight dequant; q8_K_4x4 activation index 16*(p/4)+4*mm+(p%4))
static void gen_gemm(int nb, const block_q3_Kx8 * b_ptr, const block_q8_Kx4 * a_ptr, float out[4][8]) {
    int8_t aux8[QK_K];
    for (int j = 0; j < 8; j++) {
        float sumf[4]; for (int mm=0; mm<4; mm++) sumf[mm]=0.0f;
        for (int l = 0; l < nb; l++) {
            uint8_t m=1; int wi=0, qbase=0;
            for (int nn=0;nn<QK_K;nn+=128){ int shift=0;
                for(int jj=0;jj<4;jj++){
                    for(int t=0;t<16;t++){ int pq=qbase+t; uint8_t q=b_ptr[l].qs[((pq/4)*8+j)*4+(pq%4)]; uint8_t h=b_ptr[l].hmask[((t/4)*8+j)*4+(t%4)]; aux8[wi+t]=(int8_t)(((q>>shift)&3)-((h&m)?0:4)); }
                    for(int t=0;t<16;t++){ int pq=qbase+16+t,ph=16+t; uint8_t q=b_ptr[l].qs[((pq/4)*8+j)*4+(pq%4)]; uint8_t h=b_ptr[l].hmask[((ph/4)*8+j)*4+(ph%4)]; aux8[wi+16+t]=(int8_t)(((q>>shift)&3)-((h&m)?0:4)); }
                    wi+=32; shift+=2; m<<=1;
                }
                qbase+=32;
            }
            int8_t scales[16]; unpack_col_flat(b_ptr[l].scales, j, scales);
            for (int mm=0; mm<4; mm++) {
                int isum=0;
                for (int sb=0; sb<16; sb++) {
                    int dot=0;
                    for (int chunk=0; chunk<4; chunk++) {
                        int p0=sb*16+chunk*4, idx0=16*(p0/4)+4*mm;
                        for (int t=0;t<4;t++) dot += aux8[p0+t]*(int)a_ptr[l].qs[idx0+t];
                    }
                    isum += (scales[sb]-32)*dot;
                }
                sumf[mm] += (float)isum * (b_ptr[l].d[j]*a_ptr[l].d[mm]);
            }
        }
        for (int mm=0; mm<4; mm++) out[mm][j]=sumf[mm];
    }
}

static int rndb(void){ return rand() & 0xff; }
int main(void){
    srand(20260615); int fails = 0, total = 0;
    for (int nb = 1; nb <= 3; nb++) {
        for (int trial = 0; trial < 200; trial++) {
            block_q3_K rows[8*3]; block_q8_K act[3];
            for (int j = 0; j < 8; j++) for (int l = 0; l < nb; l++) {
                block_q3_K * r = &rows[j*nb + l];
                for (int i = 0; i < QK_K/8; i++) r->hmask[i] = rndb();
                for (int i = 0; i < QK_K/4; i++) r->qs[i] = rndb();
                for (int i = 0; i < 12; i++) r->scales[i] = rndb();
                r->d = ((rand()%2000)-1000)/1000.0f;
            }
            for (int l = 0; l < nb; l++) {
                int32_t bs[16]; memset(bs,0,sizeof(bs));
                for (int i = 0; i < QK_K; i++){ act[l].qs[i]=(rand()%255)-127; bs[i/16]+=act[l].qs[i]; }
                for (int i = 0; i < 16; i++) act[l].bsums[i]=(int16_t)bs[i];
                act[l].d = ((rand()%2000)-1000)/1000.0f;
            }
            block_q3_Kx8 rep[3];
            for (int l = 0; l < nb; l++) {
                block_q3_K tmp[8]; for (int j = 0; j < 8; j++) tmp[j] = rows[j*nb+l];
                make_block_q3_Kx8(tmp, &rep[l]);
            }
            float gopt[8], ggen[8]; q3_gemv_neon_opt(nb, rep, act, gopt); gen_gemv(nb, rep, act, ggen);
            for (int j = 0; j < 8; j++) {
                float ref = ref_dot_q3K(&rows[j*nb], act, nb);
                float eo = fabsf(ref-gopt[j]), eg = fabsf(ref-ggen[j]), den = fabsf(ref)+1e-6f;
                total++; if (eo/den > 1e-3f) { fails++; if (fails<6) printf("  MISS(opt) nb=%d t=%d j=%d ref=%.4f got=%.4f\n",nb,trial,j,ref,gopt[j]); }
                total++; if (eg/den > 1e-3f) { fails++; if (fails<6) printf("  MISS(gen) nb=%d t=%d j=%d ref=%.4f got=%.4f\n",nb,trial,j,ref,ggen[j]); }
            }
            // gemm test: 4 activation rows packed q8_K_4x4
            block_q8_K arows[4*3]; block_q8_Kx4 ax4[3];
            for (int mr=0; mr<4; mr++) for (int l=0;l<nb;l++) {
                block_q8_K * a = &arows[mr*nb+l];
                for (int i=0;i<QK_K;i++) a->qs[i]=(rand()%255)-127;
                a->d = ((rand()%2000)-1000)/1000.0f; for(int i=0;i<16;i++) a->bsums[i]=0;
            }
            for (int l=0;l<nb;l++){ block_q8_K four[4]; for(int mr=0;mr<4;mr++) four[mr]=arows[mr*nb+l]; pack_q8_Kx4_4x4(four,&ax4[l]); }
            float gm[4][8]; gen_gemm(nb, rep, ax4, gm);
            for (int mr=0; mr<4; mr++) for (int j=0;j<8;j++) {
                float ref = ref_dot_q3K(&rows[j*nb], &arows[mr*nb], nb);
                float e = fabsf(ref-gm[mr][j]), den = fabsf(ref)+1e-6f;
                total++; if (e/den > 1e-3f) { fails++; if (fails<6) printf("  MISS(gemm) nb=%d t=%d mr=%d j=%d ref=%.4f got=%.4f\n",nb,trial,mr,j,ref,gm[mr][j]); }
            }
        }
    }
    printf("%s  (%d/%d mismatched)\n", fails?"FAIL":"PASS - hand NEON dotprod q3_K gemv bit-exact vs ggml_vec_dot_q3_K", fails, total);
    return fails?1:0;
}
