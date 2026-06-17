// Step 3 gate: validate the EXACT q3_K repacked gemv+gemm code (the version going into
// ggml repack.cpp) against ggml's real ggml_vec_dot_q3_K_q8_K logic (copied verbatim from
// ggml-cpu/quants.c:566). Covers BOTH gemv (M=1, block_q8_K) and gemm (M=4, block_q8_Kx4).
// d stored as float here (cancels identically on both sides) so we isolate the integer path.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#define QK_K 256

typedef struct { uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; float d; } block_q3_K;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;
typedef struct { float d[4]; int8_t qs[QK_K*4]; int16_t bsums[QK_K/4]; } block_q8_Kx4;

// ---- repacked 8-col block (mirrors what will go into repack.h) ----
typedef struct { float d[8]; uint8_t hmask[8*QK_K/8]; uint8_t qs[8*QK_K/4]; uint8_t scales[8*12]; } block_q3_Kx8;

// ====================== REFERENCE (ggml_vec_dot_q3_K_q8_K_generic) ======================
// nb super-blocks x nb activations -> single float. Copied verbatim from quants.c:566
// (the real ggml q3_K dot over n=nb*256), d as float. Lane accumulation across super-blocks.
static float ref_dot_q3K(const block_q3_K * xs, const block_q8_K * ys, int nb) {
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    int8_t  aux8[QK_K];
    int16_t aux16[8];
    int32_t aux32[8];
    float   sums[8];
    uint32_t auxs[4];
    const int8_t * scales = (const int8_t*)auxs;
    memset(sums, 0, sizeof(sums));
    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const block_q3_K * x = &xs[i];
        const block_q8_K * y = &ys[i];
        const uint8_t * q3 = x->qs;
        const uint8_t * hm = x->hmask;
        const int8_t  * q8 = y->qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;
        memcpy(auxs, x->scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = x->d * y->d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

// ====================== CANDIDATE (the code for repack.cpp) ======================
// make_block_q3_Kx8: round-robin 8-byte interleave of qs/hmask; scales stored packed (12B/col).
static block_q3_Kx8 make_block_q3_Kx8(const block_q3_K * in, unsigned int blck_size_interleave) {
    block_q3_Kx8 out;
    for (int i = 0; i < 8; i++) out.d[i] = in[i].d;
    const int qs_end = (QK_K / 4) * 8 / blck_size_interleave;
    for (int i = 0; i < qs_end; i++) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;
        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_size_interleave);
    }
    const int hm_end = (QK_K / 8) * 8 / blck_size_interleave;
    for (int i = 0; i < hm_end; i++) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;
        memcpy(&out.hmask[dst_offset], &in[src_id].hmask[src_offset], blck_size_interleave);
    }
    for (int j = 0; j < 8; j++)
        for (int t = 0; t < 12; t++) out.scales[j * 12 + t] = in[j].scales[t];
    return out;
}

// dequantize column j of a repacked super-block into aux8[256] (canonical q3_K order).
static inline void dequant_q3Kx8_col(const block_q3_Kx8 * b, int j, int8_t aux8[QK_K]) {
    uint8_t m = 1;
    int wi = 0;
    int qbase = 0;
    for (int nn = 0; nn < QK_K; nn += 128) {
        int shift = 0;
        for (int jj = 0; jj < 4; jj++) {
            for (int t = 0; t < 16; t++) {
                int pq = qbase + t;
                const uint8_t q = b->qs[((pq / 8) * 8 + j) * 8 + (pq % 8)];
                const uint8_t h = b->hmask[((t / 8) * 8 + j) * 8 + (t % 8)];
                aux8[wi + t] = (int8_t)(((q >> shift) & 3) - ((h & m) ? 0 : 4));
            }
            for (int t = 0; t < 16; t++) {
                int pq = qbase + 16 + t;
                int ph = 16 + t;
                const uint8_t q = b->qs[((pq / 8) * 8 + j) * 8 + (pq % 8)];
                const uint8_t h = b->hmask[((ph / 8) * 8 + j) * 8 + (ph % 8)];
                aux8[wi + 16 + t] = (int8_t)(((q >> shift) & 3) - ((h & m) ? 0 : 4));
            }
            wi += 32; shift += 2; m <<= 1;
        }
        qbase += 32;
    }
}

static inline void unpack_scales_col(const block_q3_Kx8 * b, int j, int8_t scales[16]) {
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t auxs[4];
    memcpy(auxs, b->scales + j * 12, 12);
    uint32_t tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t * s8 = (const int8_t *) auxs;
    for (int t = 0; t < 16; t++) scales[t] = s8[t];
}

// gemv: M=1, activation = block_q8_K (natural order). out[8].
// sdot-friendly: int32 += int8*int8 reductions (no int16 intermediate) so -mcpu=cortex-a76 auto-emits
// SDOT. Integer-exact vs ggml q3_K dot; float accumulation per-superblock (in-tree q2_K/q6_K style, KL~0).
static void my_gemv(int nb, const block_q3_Kx8 * b_ptr, const block_q8_K * a_ptr, float out[8]) {
    int8_t aux8[QK_K];
    int8_t scales[16];
    for (int j = 0; j < 8; j++) {
        float sumf = 0.0f;
        for (int l = 0; l < nb; l++) {
            dequant_q3Kx8_col(&b_ptr[l], j, aux8);
            unpack_scales_col(&b_ptr[l], j, scales);
            int isum = 0;
            for (int sb = 0; sb < 16; sb++) {
                int s = 0;
                for (int t = 0; t < 16; t++) s += aux8[sb * 16 + t] * (int) a_ptr[l].qs[sb * 16 + t];
                isum += (scales[sb] - 32) * s;
            }
            sumf += (float) isum * (b_ptr[l].d[j] * a_ptr[l].d);
        }
        out[j] = sumf;
    }
}

// gemm: M=4, activation = block_q8_Kx4. out[4][8].  q8 index = 32*(p/8) + 8*m + (p%8).
// Each 8-element half of a sub-block maps to 8 contiguous q8_Kx4 indices -> two SDOT-friendly reductions.
static void my_gemm(int nb, const block_q3_Kx8 * b_ptr, const block_q8_Kx4 * a_ptr, float out[4][8]) {
    int8_t aux8[QK_K];
    int8_t scales[16];
    for (int j = 0; j < 8; j++) {
        float sumf[4]; for (int mm = 0; mm < 4; mm++) sumf[mm] = 0.0f;
        for (int l = 0; l < nb; l++) {
            dequant_q3Kx8_col(&b_ptr[l], j, aux8);
            unpack_scales_col(&b_ptr[l], j, scales);
            for (int mm = 0; mm < 4; mm++) {
                int isum = 0;
                for (int sb = 0; sb < 16; sb++) {
                    int s = 0;
                    for (int half = 0; half < 2; half++) {
                        const int p0  = sb * 16 + half * 8;
                        const int idx0 = 32 * (p0 / 8) + 8 * mm;  // 8 contiguous q8_Kx4 lanes
                        for (int t = 0; t < 8; t++) s += aux8[p0 + t] * (int) a_ptr[l].qs[idx0 + t];
                    }
                    isum += (scales[sb] - 32) * s;
                }
                sumf[mm] += (float) isum * (b_ptr[l].d[j] * a_ptr[l].d[mm]);
            }
        }
        for (int mm = 0; mm < 4; mm++) out[mm][j] = sumf[mm];
    }
}

// pack 4 block_q8_K rows into one block_q8_Kx4 using ggml's interleave (independent of read formula):
//   src_id = (j%32)/8 ; src_offset = (j/32)*8 + (j%8)
static void pack_q8_Kx4(const block_q8_K rows[4], block_q8_Kx4 * o) {
    for (int m = 0; m < 4; m++) o->d[m] = rows[m].d;
    for (int j = 0; j < QK_K * 4; j++) {
        int src_id = (j % 32) / 8;
        int src_offset = (j / 32) * 8 + (j % 8);
        o->qs[j] = rows[src_id].qs[src_offset];
    }
}

static int rndb(void){ return rand() & 0xff; }

int main(void) {
    srand(1234);
    int fails = 0, total = 0;

    for (int nb = 1; nb <= 2; nb++) {
        for (int trial = 0; trial < 150; trial++) {
            // 8 q3_K rows, each nb super-blocks
            block_q3_K rows[8 * 2];
            for (int j = 0; j < 8; j++) for (int l = 0; l < nb; l++) {
                block_q3_K * r = &rows[j * nb + l];
                for (int i = 0; i < QK_K/8; i++) r->hmask[i] = rndb();
                for (int i = 0; i < QK_K/4; i++) r->qs[i] = rndb();
                for (int i = 0; i < 12; i++) r->scales[i] = rndb();
                r->d = ((rand() % 2000) - 1000) / 1000.0f;
            }
            // repack into nb block_q3_Kx8 (col-major: b_rep[l] holds super-block l of all 8 cols)
            block_q3_Kx8 b_rep[2];
            for (int l = 0; l < nb; l++) {
                block_q3_K tmp[8];
                for (int j = 0; j < 8; j++) tmp[j] = rows[j * nb + l];
                b_rep[l] = make_block_q3_Kx8(tmp, 8);
            }

            // ---- gemv test ----
            block_q8_K act[2];
            for (int l = 0; l < nb; l++) {
                for (int i = 0; i < QK_K; i++) act[l].qs[i] = (rand() % 255) - 127;
                act[l].d = ((rand() % 2000) - 1000) / 1000.0f;
                for (int i = 0; i < QK_K/16; i++) act[l].bsums[i] = 0;
            }
            float gv[8];
            my_gemv(nb, b_rep, act, gv);
            for (int j = 0; j < 8; j++) {
                float ref = ref_dot_q3K(&rows[j * nb], act, nb);
                float e = fabsf(ref - gv[j]); float den = fabsf(ref) + 1e-6f;
                total++; if (e / den > 1e-3f) { fails++; if (fails < 5) printf("  GEMV mismatch nb=%d trial=%d j=%d ref=%.5f got=%.5f\n", nb, trial, j, ref, gv[j]); }
            }

            // ---- gemm test (4 activation rows) ----
            block_q8_K arows[4 * 2];
            for (int m = 0; m < 4; m++) for (int l = 0; l < nb; l++) {
                block_q8_K * a = &arows[m * nb + l];
                for (int i = 0; i < QK_K; i++) a->qs[i] = (rand() % 255) - 127;
                a->d = ((rand() % 2000) - 1000) / 1000.0f;
                for (int i = 0; i < QK_K/16; i++) a->bsums[i] = 0;
            }
            block_q8_Kx4 ax4[2];
            for (int l = 0; l < nb; l++) {
                block_q8_K four[4];
                for (int m = 0; m < 4; m++) four[m] = arows[m * nb + l];
                pack_q8_Kx4(four, &ax4[l]);
            }
            float gm[4][8];
            my_gemm(nb, b_rep, ax4, gm);
            for (int m = 0; m < 4; m++) for (int j = 0; j < 8; j++) {
                float ref = ref_dot_q3K(&rows[j * nb], &arows[m * nb], nb);
                float e = fabsf(ref - gm[m][j]); float den = fabsf(ref) + 1e-6f;
                total++; if (e / den > 1e-3f) { fails++; if (fails < 5) printf("  GEMM mismatch nb=%d trial=%d m=%d j=%d ref=%.5f got=%.5f\n", nb, trial, m, j, ref, gm[m][j]); }
            }
        }
    }
    printf("%s  (%d/%d dots mismatched)\n", fails ? "FAIL" : "PASS - q3_K repacked gemv+gemm bit-exact vs ggml_vec_dot_q3_K reference", fails, total);
    return fails ? 1 : 0;
}
