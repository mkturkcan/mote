/*
 * bench.c -- 2-bit weight packing + UDOT cvt-fold matvec microbenchmark.
 *
 * Validates the bandwidth-saving kernel idea for Gemma 4 E2B FFN matrices on
 * Cortex-A76 (Armv8.2-A: NEON + FEAT_DotProd, no i8mm, no SVE).
 *
 * Formats implemented (all matvec: y = W @ x, W is [out,in], x fp32 length in):
 *   - REF   : fp32 W @ fp32 x (ground truth).
 *   - Q40   : 4-bit symmetric, per-32 fp16 scale (Q4_0-style). UDOT int8 path.
 *   - U2     : uniform 2-bit symmetric, per-group fp16 scale ("cvt-fold" affine).
 *             UDOT int8 path (levels mapped to int8, dot product, rescale).
 *   - CB2    : 2-bit per-group 4-level codebook (Lloyd-Max MSE fit). Table dequant.
 *
 * The integer (UDOT) kernels quantize x once to int8 with a per-vector scale,
 * then accumulate integer dot products group-by-group, multiplying each group's
 * int32 partial sum by (w_scale * x_scale). This is the "cvt-fold" structure:
 * the dequant of W folds into a single fp multiply per group, off the hot path.
 *
 * Gated behind __ARM_FEATURE_DOTPROD; scalar fallback used on x86 host so the
 * same int kernels are exercised (and cross-checked NEON-vs-scalar on aarch64).
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#endif

/* ---- fp16 <-> fp32 (software, portable) ---------------------------------- */
typedef uint16_t f16_t;

static inline float f16_to_f32(f16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) { bits = sign; }
        else {
            /* subnormal */
            exp = 127 - 15 + 1;
            while ((man & 0x400) == 0) { man <<= 1; exp--; }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (man << 13);
    } else {
        exp = exp - 15 + 127;
        bits = sign | (exp << 23) | (man << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}

static inline f16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man  = x & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) return (f16_t)sign;          /* too small -> 0 */
        man |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = man >> shift;
        if ((man >> (shift - 1)) & 1) half++;       /* round to nearest */
        return (f16_t)(sign | half);
    } else if (exp >= 0x1F) {
        return (f16_t)(sign | 0x7C00);              /* inf/overflow */
    }
    uint32_t half = (uint32_t)exp << 10 | (man >> 13);
    if (man & 0x1000) half++;                        /* round to nearest */
    return (f16_t)(sign | half);
}

/* ---- group size ----------------------------------------------------------- */
#define GS 32  /* group/block size for all quant formats (matches Q4_0) */

/* ========================================================================== */
/* Format 1: Q4_0-style 4-bit symmetric, per-32 fp16 scale                    */
/* ========================================================================== */
/* Storage per group of 32: 1 fp16 scale (2 B) + 16 B packed nibbles = 18 B.  */
/* nibble in [0,15], value = (nibble - 8) * scale.                            */
typedef struct {
    f16_t  *scales;   /* [out * (in/GS)] */
    uint8_t *q;       /* [out * (in/GS) * 16]  packed nibbles, lo then hi */
    int out, in, ng;  /* ng = in/GS groups per row */
} Q40;

static void q40_pack(Q40 *q, const float *W, int out, int in) {
    q->out = out; q->in = in; q->ng = in / GS;
    q->scales = malloc((size_t)out * q->ng * sizeof(f16_t));
    q->q = malloc((size_t)out * q->ng * 16);
    for (int r = 0; r < out; r++) {
        const float *row = W + (size_t)r * in;
        for (int g = 0; g < q->ng; g++) {
            const float *blk = row + g * GS;
            float amax = 0.f;
            for (int i = 0; i < GS; i++) { float a = fabsf(blk[i]); if (a > amax) amax = a; }
            /* symmetric: 8 levels each side -> max nibble offset 8 (or 7) */
            float scale = amax / 8.0f;
            float inv = scale > 0 ? 1.0f / scale : 0.0f;
            f16_t hs = f32_to_f16(scale);
            float rs = f16_to_f32(hs);            /* use stored scale for fidelity */
            inv = rs > 0 ? 1.0f / rs : 0.0f;
            q->scales[(size_t)r * q->ng + g] = hs;
            uint8_t *dst = q->q + ((size_t)r * q->ng + g) * 16;
            for (int i = 0; i < 16; i++) {
                int v0 = (int)lrintf(blk[i] * inv) + 8;
                int v1 = (int)lrintf(blk[i + 16] * inv) + 8;
                if (v0 < 0) v0 = 0; if (v0 > 15) v0 = 15;
                if (v1 < 0) v1 = 0; if (v1 > 15) v1 = 15;
                dst[i] = (uint8_t)(v0 | (v1 << 4));
            }
        }
    }
}
static double q40_bytes_per_weight(const Q40 *q) {
    double bytes = (double)q->out * q->ng * (sizeof(f16_t) + 16);
    return bytes / ((double)q->out * q->in);
}
/* reconstruct one full row into dst (for RMSE) */
static void q40_unpack_row(const Q40 *q, int r, float *dst) {
    for (int g = 0; g < q->ng; g++) {
        float s = f16_to_f32(q->scales[(size_t)r * q->ng + g]);
        const uint8_t *src = q->q + ((size_t)r * q->ng + g) * 16;
        for (int i = 0; i < 16; i++) {
            dst[g*GS + i]      = ((int)(src[i] & 0xF) - 8) * s;
            dst[g*GS + i + 16] = ((int)(src[i] >> 4)  - 8) * s;
        }
    }
}

/* ========================================================================== */
/* Format 2: uniform 2-bit symmetric, per-group fp16 scale                    */
/* ========================================================================== */
/* 2 bits -> 4 levels. Symmetric uniform: indices {0,1,2,3} map to            */
/* {-2,-1,0,1}? We instead center: level = (idx - 1.5) so {-1.5,-.5,.5,1.5}*s.*/
/* That keeps it symmetric around 0 with 4 evenly spaced levels.              */
/* Storage per group of 32: 1 fp16 scale + 32*2bits = 8 B packed = 10 B.      */
typedef struct {
    f16_t *scales;    /* [out*ng] */
    uint8_t *q;       /* [out*ng*8] packed: 4 weights/byte */
    int out, in, ng;
} U2;

/* uniform symmetric levels for idx 0..3 : (idx - 1.5) */
static inline float u2_level(int idx) { return (float)idx - 1.5f; }

static void u2_pack(U2 *q, const float *W, int out, int in) {
    q->out = out; q->in = in; q->ng = in / GS;
    q->scales = malloc((size_t)out * q->ng * sizeof(f16_t));
    q->q = malloc((size_t)out * q->ng * 8);
    for (int r = 0; r < out; r++) {
        const float *row = W + (size_t)r * in;
        for (int g = 0; g < q->ng; g++) {
            const float *blk = row + g * GS;
            float amax = 0.f;
            for (int i = 0; i < GS; i++) { float a = fabsf(blk[i]); if (a > amax) amax = a; }
            /* largest magnitude level is 1.5, so scale = amax/1.5 */
            float scale = amax / 1.5f;
            f16_t hs = f32_to_f16(scale);
            float rs = f16_to_f32(hs);
            float inv = rs > 0 ? 1.0f / rs : 0.0f;
            q->scales[(size_t)r * q->ng + g] = hs;
            uint8_t *dst = q->q + ((size_t)r * q->ng + g) * 8;
            memset(dst, 0, 8);
            for (int i = 0; i < GS; i++) {
                /* idx = round(v/scale + 1.5), clamp 0..3 */
                int idx = (int)lrintf(blk[i] * inv + 1.5f);
                if (idx < 0) idx = 0; if (idx > 3) idx = 3;
                dst[i >> 2] |= (uint8_t)(idx << ((i & 3) * 2));
            }
        }
    }
}
static double u2_bytes_per_weight(const U2 *q) {
    double bytes = (double)q->out * q->ng * (sizeof(f16_t) + 8);
    return bytes / ((double)q->out * q->in);
}
static void u2_unpack_row(const U2 *q, int r, float *dst) {
    for (int g = 0; g < q->ng; g++) {
        float s = f16_to_f32(q->scales[(size_t)r * q->ng + g]);
        const uint8_t *src = q->q + ((size_t)r * q->ng + g) * 8;
        for (int i = 0; i < GS; i++) {
            int idx = (src[i >> 2] >> ((i & 3) * 2)) & 3;
            dst[g*GS + i] = u2_level(idx) * s;
        }
    }
}

/* ========================================================================== */
/* Format 3: 2-bit per-group 4-level codebook (Lloyd-Max MSE fit)             */
/* ========================================================================== */
/* Per group of 32: 4 fp16 codebook levels (8 B) + 8 B indices = 16 B.        */
/* (We keep a full per-group codebook to test the *best case* for VQ.)        */
typedef struct {
    f16_t *cb;        /* [out*ng*4]  4 levels per group */
    uint8_t *q;       /* [out*ng*8] */
    int out, in, ng;
} CB2;

/* 1-D Lloyd-Max / k-means with 4 centroids on a group's 32 values. */
static void lloyd4(const float *v, int n, float lev[4]) {
    /* init: spread over [min,max] */
    float mn = v[0], mx = v[0];
    for (int i = 1; i < n; i++) { if (v[i] < mn) mn = v[i]; if (v[i] > mx) mx = v[i]; }
    if (mx - mn < 1e-12f) { for (int k = 0; k < 4; k++) lev[k] = mn; return; }
    for (int k = 0; k < 4; k++) lev[k] = mn + (mx - mn) * (k + 0.5f) / 4.0f;
    for (int it = 0; it < 12; it++) {
        /* assign + accumulate in one pass (no per-point storage -> any n) */
        float sum[4] = {0}; int cnt[4] = {0};
        for (int i = 0; i < n; i++) {
            int best = 0; float bd = fabsf(v[i] - lev[0]);
            for (int k = 1; k < 4; k++) { float d = fabsf(v[i] - lev[k]); if (d < bd) { bd = d; best = k; } }
            sum[best] += v[i]; cnt[best]++;
        }
        for (int k = 0; k < 4; k++) if (cnt[k]) lev[k] = sum[k] / cnt[k];
    }
    /* sort levels ascending so indices are ordered (helps nothing functional,
       but makes the codebook deterministic) */
    for (int a = 0; a < 4; a++) for (int b = a+1; b < 4; b++)
        if (lev[b] < lev[a]) { float t = lev[a]; lev[a] = lev[b]; lev[b] = t; }
}

static void cb2_pack(CB2 *q, const float *W, int out, int in) {
    q->out = out; q->in = in; q->ng = in / GS;
    q->cb = malloc((size_t)out * q->ng * 4 * sizeof(f16_t));
    q->q  = malloc((size_t)out * q->ng * 8);
    for (int r = 0; r < out; r++) {
        const float *row = W + (size_t)r * in;
        for (int g = 0; g < q->ng; g++) {
            const float *blk = row + g * GS;
            float lev[4];
            lloyd4(blk, GS, lev);
            f16_t *cb = q->cb + ((size_t)r * q->ng + g) * 4;
            float levq[4];
            for (int k = 0; k < 4; k++) { cb[k] = f32_to_f16(lev[k]); levq[k] = f16_to_f32(cb[k]); }
            uint8_t *dst = q->q + ((size_t)r * q->ng + g) * 8;
            memset(dst, 0, 8);
            for (int i = 0; i < GS; i++) {
                int best = 0; float bd = fabsf(blk[i] - levq[0]);
                for (int k = 1; k < 4; k++) { float d = fabsf(blk[i] - levq[k]); if (d < bd) { bd = d; best = k; } }
                dst[i >> 2] |= (uint8_t)(best << ((i & 3) * 2));
            }
        }
    }
}
static double cb2_bytes_per_weight(const CB2 *q) {
    double bytes = (double)q->out * q->ng * (4 * sizeof(f16_t) + 8);
    return bytes / ((double)q->out * q->in);
}
static void cb2_unpack_row(const CB2 *q, int r, float *dst) {
    for (int g = 0; g < q->ng; g++) {
        const f16_t *cb = q->cb + ((size_t)r * q->ng + g) * 4;
        float lev[4]; for (int k = 0; k < 4; k++) lev[k] = f16_to_f32(cb[k]);
        const uint8_t *src = q->q + ((size_t)r * q->ng + g) * 8;
        for (int i = 0; i < GS; i++) {
            int idx = (src[i >> 2] >> ((i & 3) * 2)) & 3;
            dst[g*GS + i] = lev[idx];
        }
    }
}

/* ========================================================================== */
/* Format 4: per-TENSOR 4-level codebook + per-group fp16 scale (CBT)          */
/* ========================================================================== */
/* This is the bandwidth-honest VQ variant: one global 4-level shape codebook  */
/* shared by the whole tensor (negligible bytes), plus a per-group fp16 scale. */
/* Each weight = scale * cb[idx]. Storage per group of 32 = 2B scale + 8B idx  */
/* = 10 B -> 0.3125 B/w, identical to uniform 2-bit. So CBT-vs-U2 is an        */
/* apples-to-apples "does a learned 4-level shape beat the uniform grid?".     */
typedef struct {
    float   cb[4];    /* shared normalized levels (max |level| = 1) */
    f16_t  *scales;   /* [out*ng] */
    uint8_t *q;       /* [out*ng*8] */
    int out, in, ng;
} CBT;

static void cbt_pack(CBT *q, const float *W, int out, int in) {
    q->out = out; q->in = in; q->ng = in / GS;
    q->scales = malloc((size_t)out * q->ng * sizeof(f16_t));
    q->q = malloc((size_t)out * q->ng * 8);
    /* Fit shared codebook on normalized weights: w/group_amax in [-1,1].
       Gather a subsample of normalized values across the tensor, run lloyd4. */
    int ng = q->ng;
    int max_samp = 200000;
    float *samp = malloc((size_t)max_samp * sizeof(float));
    int ns = 0;
    int stride = ((size_t)out * ng) / (max_samp / GS) ; if (stride < 1) stride = 1;
    for (int r = 0; r < out && ns < max_samp - GS; r++) {
        const float *row = W + (size_t)r * in;
        for (int g = 0; g < ng && ns < max_samp - GS; g++) {
            if (((size_t)r * ng + g) % stride) continue;
            const float *blk = row + g * GS;
            float amax = 0.f; for (int i = 0; i < GS; i++){ float a=fabsf(blk[i]); if(a>amax)amax=a; }
            float inv = amax > 0 ? 1.0f/amax : 0.0f;
            for (int i = 0; i < GS; i++) samp[ns++] = blk[i] * inv;
        }
    }
    float lev[4]; lloyd4(samp, ns > 0 ? ns : 1, lev);
    free(samp);
    /* normalize so max |level| = 1 (the per-group scale absorbs magnitude) */
    float mx = 0; for (int k=0;k<4;k++){ float a=fabsf(lev[k]); if(a>mx)mx=a; }
    if (mx <= 0) mx = 1;
    for (int k = 0; k < 4; k++) q->cb[k] = lev[k] / mx;

    for (int r = 0; r < out; r++) {
        const float *row = W + (size_t)r * in;
        for (int g = 0; g < ng; g++) {
            const float *blk = row + g * GS;
            float amax = 0.f; for (int i = 0; i < GS; i++){ float a=fabsf(blk[i]); if(a>amax)amax=a; }
            /* scale chosen so cb[k]*scale covers the group's range */
            float scale = amax;  /* since max|cb|=1 */
            f16_t hs = f32_to_f16(scale); float rs = f16_to_f32(hs);
            q->scales[(size_t)r * ng + g] = hs;
            uint8_t *dst = q->q + ((size_t)r * ng + g) * 8; memset(dst, 0, 8);
            for (int i = 0; i < GS; i++) {
                int best = 0; float bd = fabsf(blk[i] - q->cb[0]*rs);
                for (int k = 1; k < 4; k++){ float d=fabsf(blk[i]-q->cb[k]*rs); if(d<bd){bd=d;best=k;} }
                dst[i >> 2] |= (uint8_t)(best << ((i & 3) * 2));
            }
        }
    }
}
static double cbt_bytes_per_weight(const CBT *q) {
    double bytes = (double)q->out * q->ng * (sizeof(f16_t) + 8);  /* codebook ~free */
    return bytes / ((double)q->out * q->in);
}
static void cbt_unpack_row(const CBT *q, int r, float *dst) {
    for (int g = 0; g < q->ng; g++) {
        float s = f16_to_f32(q->scales[(size_t)r * q->ng + g]);
        const uint8_t *src = q->q + ((size_t)r * q->ng + g) * 8;
        for (int i = 0; i < GS; i++) {
            int idx = (src[i >> 2] >> ((i & 3) * 2)) & 3;
            dst[g*GS + i] = q->cb[idx] * s;
        }
    }
}

/* ========================================================================== */
/* Matvec kernels                                                             */
/* ========================================================================== */

/* Reference fp32 */
static void matvec_ref(const float *W, const float *x, float *y, int out, int in) {
    for (int r = 0; r < out; r++) {
        const float *row = W + (size_t)r * in;
        float acc = 0.f;
        for (int i = 0; i < in; i++) acc += row[i] * x[i];
        y[r] = acc;
    }
}

/* ---- x quantization to int8 (per-group scale) ----------------------------- */
/* We quantize x per group of GS to int8 symmetric, returning per-group scale. */
static void quant_x_int8(const float *x, int in, int8_t *xq, float *xscale) {
    int ng = in / GS;
    for (int g = 0; g < ng; g++) {
        const float *b = x + g * GS;
        float amax = 0.f;
        for (int i = 0; i < GS; i++) { float a = fabsf(b[i]); if (a > amax) amax = a; }
        float s = amax / 127.0f;
        float inv = s > 0 ? 1.0f / s : 0.0f;
        xscale[g] = s;
        for (int i = 0; i < GS; i++) {
            int v = (int)lrintf(b[i] * inv);
            if (v < -127) v = -127; if (v > 127) v = 127;
            xq[g * GS + i] = (int8_t)v;
        }
    }
}

/* ---- Q4_0 UDOT matvec: int8 W * int8 x, per-group rescale ----------------- */
/* W nibbles -> int8 in [-8,7], dot with xq, *(wscale*xscale).                 */
static void matvec_q40_dot(const Q40 *q, const int8_t *xq, const float *xscale,
                           float *y) {
    int out = q->out, ng = q->ng;
    for (int r = 0; r < out; r++) {
        float acc = 0.f;
        for (int g = 0; g < ng; g++) {
            float ws = f16_to_f32(q->scales[(size_t)r * ng + g]);
            const uint8_t *src = q->q + ((size_t)r * ng + g) * 16;
            const int8_t *xg = xq + g * GS;
            int32_t isum;
#if defined(__ARM_FEATURE_DOTPROD)
            /* unpack 16 bytes (32 nibbles) -> two int8x16 (-8..7) */
            uint8x16_t packed = vld1q_u8(src);
            int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0F)));
            int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
            int8x16_t off = vdupq_n_s8(8);
            lo = vsubq_s8(lo, off);
            hi = vsubq_s8(hi, off);
            int8x16_t x0 = vld1q_s8(xg);
            int8x16_t x1 = vld1q_s8(xg + 16);
            int32x4_t a = vdupq_n_s32(0);
            a = vdotq_s32(a, lo, x0);
            a = vdotq_s32(a, hi, x1);
            isum = vaddvq_s32(a);
#else
            isum = 0;
            for (int i = 0; i < 16; i++) {
                int w0 = (int)(src[i] & 0xF) - 8;
                int w1 = (int)(src[i] >> 4)  - 8;
                isum += w0 * (int)xg[i];
                isum += w1 * (int)xg[i + 16];
            }
#endif
            acc += (float)isum * ws * xscale[g];
        }
        y[r] = acc;
    }
}

/* ---- U2 UDOT matvec ------------------------------------------------------- */
/* uniform levels (idx-1.5) -> to keep integers we use l_int = 2*idx-3 in      */
/* {-3,-1,1,3}, and fold the 0.5 via wscale_eff = ws*0.5. Then int dot.        */
static void matvec_u2_dot(const U2 *q, const int8_t *xq, const float *xscale,
                          float *y) {
    int out = q->out, ng = q->ng;
    for (int r = 0; r < out; r++) {
        float acc = 0.f;
        for (int g = 0; g < ng; g++) {
            float ws = f16_to_f32(q->scales[(size_t)r * ng + g]) * 0.5f;
            const uint8_t *src = q->q + ((size_t)r * ng + g) * 8;
            const int8_t *xg = xq + g * GS;
            int32_t isum;
#if defined(__ARM_FEATURE_DOTPROD)
            /* expand 8 bytes (32 2-bit idx) into int8x16 x2 of values 2*idx-3 */
            int8_t tmp[GS];
            for (int i = 0; i < GS; i++) {
                int idx = (src[i >> 2] >> ((i & 3) * 2)) & 3;
                tmp[i] = (int8_t)(2 * idx - 3);
            }
            int8x16_t w0 = vld1q_s8(tmp);
            int8x16_t w1 = vld1q_s8(tmp + 16);
            int8x16_t x0 = vld1q_s8(xg);
            int8x16_t x1 = vld1q_s8(xg + 16);
            int32x4_t a = vdupq_n_s32(0);
            a = vdotq_s32(a, w0, x0);
            a = vdotq_s32(a, w1, x1);
            isum = vaddvq_s32(a);
#else
            isum = 0;
            for (int i = 0; i < GS; i++) {
                int idx = (src[i >> 2] >> ((i & 3) * 2)) & 3;
                int w = 2 * idx - 3;        /* {-3,-1,1,3} */
                isum += w * (int)xg[i];
            }
#endif
            acc += (float)isum * ws * xscale[g];
        }
        y[r] = acc;
    }
}

/* ---- CB2 matvec (table dequant; arbitrary levels, no int dot) ------------- */
/* Codebook levels are arbitrary fp; we accumulate per-index x sums then dot   */
/* with the 4 levels. This is cheap: 4 accumulators + 4 madds per group.       */
static void matvec_cb2(const CB2 *q, const float *x, float *y) {
    int out = q->out, ng = q->ng;
    for (int r = 0; r < out; r++) {
        float acc = 0.f;
        for (int g = 0; g < ng; g++) {
            const f16_t *cb = q->cb + ((size_t)r * ng + g) * 4;
            const uint8_t *src = q->q + ((size_t)r * ng + g) * 8;
            const float *xg = x + g * GS;
            float xs[4] = {0,0,0,0};   /* sum of x where idx==k */
            for (int i = 0; i < GS; i++) {
                int idx = (src[i >> 2] >> ((i & 3) * 2)) & 3;
                xs[idx] += xg[i];
            }
            acc += f16_to_f32(cb[0]) * xs[0] + f16_to_f32(cb[1]) * xs[1]
                 + f16_to_f32(cb[2]) * xs[2] + f16_to_f32(cb[3]) * xs[3];
        }
        y[r] = acc;
    }
}

/* ---- CBT UDOT matvec ------------------------------------------------------ */
/* Shared 4-level codebook -> quantize the 4 levels to int8 (round(cb*127))    */
/* once. Per group, expand indices to int8 weights via a 4-entry LUT, UDOT     */
/* with int8 x, then rescale by scale*(1/127)*xscale. True integer hot loop.   */
static void matvec_cbt_dot(const CBT *q, const int8_t *xq, const float *xscale,
                           float *y) {
    int out = q->out, ng = q->ng;
    int8_t cbi[4];
    for (int k = 0; k < 4; k++) {
        int v = (int)lrintf(q->cb[k] * 127.0f);
        if (v < -127) v = -127; if (v > 127) v = 127;
        cbi[k] = (int8_t)v;
    }
    for (int r = 0; r < out; r++) {
        float acc = 0.f;
        for (int g = 0; g < ng; g++) {
            float ws = f16_to_f32(q->scales[(size_t)r * ng + g]) * (1.0f / 127.0f);
            const uint8_t *src = q->q + ((size_t)r * ng + g) * 8;
            const int8_t *xg = xq + g * GS;
            int32_t isum;
            int8_t tmp[GS];
            for (int i = 0; i < GS; i++) {
                int idx = (src[i >> 2] >> ((i & 3) * 2)) & 3;
                tmp[i] = cbi[idx];
            }
#if defined(__ARM_FEATURE_DOTPROD)
            int8x16_t w0 = vld1q_s8(tmp), w1 = vld1q_s8(tmp + 16);
            int8x16_t x0 = vld1q_s8(xg),  x1 = vld1q_s8(xg + 16);
            int32x4_t a = vdupq_n_s32(0);
            a = vdotq_s32(a, w0, x0); a = vdotq_s32(a, w1, x1);
            isum = vaddvq_s32(a);
#else
            isum = 0;
            for (int i = 0; i < GS; i++) isum += (int)tmp[i] * (int)xg[i];
#endif
            acc += (float)isum * ws * xscale[g];
        }
        y[r] = acc;
    }
}

/* ========================================================================== */
/* Metrics                                                                     */
/* ========================================================================== */
typedef struct { double rmse, maxabs; } WErr;

static WErr row_recon_err(const float *Wrow, const float *rec, int in) {
    double se = 0, mx = 0;
    for (int i = 0; i < in; i++) {
        double d = (double)Wrow[i] - rec[i];
        se += d * d; if (fabs(d) > mx) mx = fabs(d);
    }
    WErr e; e.rmse = sqrt(se / in); e.maxabs = mx; return e;
}

static double rel_err(const float *yref, const float *y, int out) {
    double num = 0, den = 0;
    for (int r = 0; r < out; r++) {
        double d = (double)yref[r] - y[r];
        num += d * d; den += (double)yref[r] * yref[r];
    }
    return den > 0 ? sqrt(num / den) : 0.0;
}
__attribute__((unused))
static double max_abs_diff(const float *a, const float *b, int n) {
    double m = 0; for (int i = 0; i < n; i++) { double d = fabs((double)a[i]-b[i]); if (d>m) m=d; } return m;
}

/* ========================================================================== */
/* Data loading                                                               */
/* ========================================================================== */
static float *load_f32(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    float *buf = malloc(n * sizeof(float));
    size_t got = fread(buf, sizeof(float), n, f);
    if (got != n) { fprintf(stderr, "short read %s: %zu/%zu\n", path, got, n); exit(1); }
    fclose(f);
    return buf;
}

/* simple xorshift RNG for reproducible x vectors */
static uint64_t rng_state = 0x123456789abcdefULL;
static double frand(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) / (double)(1ULL << 53);
}

/* ========================================================================== */
/* Per-tensor evaluation                                                      */
/* ========================================================================== */
static void eval_tensor(const char *name, const char *path, int out, int in) {
    printf("\n================ %s  [out=%d in=%d] ================\n", name, out, in);
    size_t n = (size_t)out * in;
    float *W = load_f32(path, n);

    /* pack all formats */
    Q40 q40; q40_pack(&q40, W, out, in);
    U2  u2;  u2_pack(&u2,  W, out, in);
    CB2 cb2; cb2_pack(&cb2, W, out, in);
    CBT cbt; cbt_pack(&cbt, W, out, in);

    /* ---- W reconstruction error (avg over rows) ---- */
    float *rec = malloc((size_t)in * sizeof(float));
    double q40_rmse=0,q40_mx=0,u2_rmse=0,u2_mx=0,cb_rmse=0,cb_mx=0,cbt_rmse=0,cbt_mx=0;
    for (int r = 0; r < out; r++) {
        const float *Wrow = W + (size_t)r * in;
        WErr e;
        q40_unpack_row(&q40, r, rec); e = row_recon_err(Wrow, rec, in);
        q40_rmse += e.rmse*e.rmse; if (e.maxabs > q40_mx) q40_mx = e.maxabs;
        u2_unpack_row(&u2, r, rec);  e = row_recon_err(Wrow, rec, in);
        u2_rmse  += e.rmse*e.rmse; if (e.maxabs > u2_mx) u2_mx = e.maxabs;
        cb2_unpack_row(&cb2, r, rec); e = row_recon_err(Wrow, rec, in);
        cb_rmse  += e.rmse*e.rmse; if (e.maxabs > cb_mx) cb_mx = e.maxabs;
        cbt_unpack_row(&cbt, r, rec); e = row_recon_err(Wrow, rec, in);
        cbt_rmse += e.rmse*e.rmse; if (e.maxabs > cbt_mx) cbt_mx = e.maxabs;
    }
    q40_rmse = sqrt(q40_rmse/out); u2_rmse = sqrt(u2_rmse/out);
    cb_rmse = sqrt(cb_rmse/out);   cbt_rmse = sqrt(cbt_rmse/out);
    free(rec);

    /* W stats for context */
    double wse=0; double wabs=0; for (size_t i=0;i<n;i++){ wse+=(double)W[i]*W[i]; double a=fabs(W[i]); if(a>wabs)wabs=a;}
    double wrms = sqrt(wse/n);

    printf("W stats: rms=%.6f maxabs=%.6f\n", wrms, wabs);
    printf("bytes/weight:  Q40=%.4f  U2=%.4f  CB2=%.4f  CBT=%.4f  (fp32=4.0)\n",
           q40_bytes_per_weight(&q40), u2_bytes_per_weight(&u2),
           cb2_bytes_per_weight(&cb2), cbt_bytes_per_weight(&cbt));
    printf("W recon RMSE:  Q40=%.6f  U2=%.6f  CB2=%.6f  CBT=%.6f\n",
           q40_rmse, u2_rmse, cb_rmse, cbt_rmse);
    printf("W recon maxabs:Q40=%.6f  U2=%.6f  CB2=%.6f  CBT=%.6f\n",
           q40_mx, u2_mx, cb_mx, cbt_mx);
    printf("W RMSE / W_rms:Q40=%.4f  U2=%.4f  CB2=%.4f  CBT=%.4f\n",
           q40_rmse/wrms, u2_rmse/wrms, cb_rmse/wrms, cbt_rmse/wrms);
    printf("CBT shared codebook levels: [%.4f %.4f %.4f %.4f]\n",
           cbt.cb[0], cbt.cb[1], cbt.cb[2], cbt.cb[3]);

    /* ---- output relative error over several x vectors ---- */
    float *x   = malloc((size_t)in * sizeof(float));
    float *yref= malloc((size_t)out * sizeof(float));
    float *yq40= malloc((size_t)out * sizeof(float));
    float *yu2 = malloc((size_t)out * sizeof(float));
    float *ycb = malloc((size_t)out * sizeof(float));
    float *ycbt= malloc((size_t)out * sizeof(float));
    int8_t *xq = malloc((size_t)in);
    float  *xscale = malloc((size_t)(in/GS) * sizeof(float));

    const int NV = 4;
    double sq40=0,su2=0,scb=0,scbt=0;
    for (int v = 0; v < NV; v++) {
        /* mix: random normal-ish (v<2) and a "spiky" real-ish activation (v>=2) */
        for (int i = 0; i < in; i++) {
            double u = frand()*2-1;
            if (v >= 2) {
                /* heavier tailed: occasional large activations like real FFN input */
                double s = frand();
                u = (s < 0.02) ? (frand()*8-4) : (frand()*1.0-0.5);
            }
            x[i] = (float)u;
        }
        matvec_ref(W, x, yref, out, in);
        quant_x_int8(x, in, xq, xscale);
        matvec_q40_dot(&q40, xq, xscale, yq40);
        matvec_u2_dot(&u2, xq, xscale, yu2);
        matvec_cb2(&cb2, x, ycb);
        matvec_cbt_dot(&cbt, xq, xscale, ycbt);
        sq40 += rel_err(yref, yq40, out);
        su2  += rel_err(yref, yu2,  out);
        scb  += rel_err(yref, ycb,  out);
        scbt += rel_err(yref, ycbt, out);
    }
    printf("output rel-err (avg of %d x-vecs):  Q40=%.5f  U2=%.5f  CB2=%.5f  CBT=%.5f\n",
           NV, sq40/NV, su2/NV, scb/NV, scbt/NV);

    /* ---- NEON-vs-scalar correctness of the int dot kernels ----
       We recompute the int kernels with a forced scalar path and compare.
       On x86 the kernels are already scalar; on aarch64 this proves the NEON
       UDOT output matches a scalar reference bit-for-fp-tolerance. */
#if defined(__ARM_FEATURE_DOTPROD)
    {
        /* scalar recompute for last x */
        float *yq40s = malloc((size_t)out*sizeof(float));
        float *yu2s  = malloc((size_t)out*sizeof(float));
        /* scalar Q40 */
        for (int r=0;r<out;r++){ float acc=0;
            for(int g=0;g<q40.ng;g++){ float ws=f16_to_f32(q40.scales[(size_t)r*q40.ng+g]);
                const uint8_t*src=q40.q+((size_t)r*q40.ng+g)*16; const int8_t*xg=xq+g*GS; int32_t s=0;
                for(int i=0;i<16;i++){ s+=((int)(src[i]&0xF)-8)*(int)xg[i]; s+=((int)(src[i]>>4)-8)*(int)xg[i+16];}
                acc+=(float)s*ws*xscale[g]; } yq40s[r]=acc; }
        /* scalar U2 */
        for (int r=0;r<out;r++){ float acc=0;
            for(int g=0;g<u2.ng;g++){ float ws=f16_to_f32(u2.scales[(size_t)r*u2.ng+g])*0.5f;
                const uint8_t*src=u2.q+((size_t)r*u2.ng+g)*8; const int8_t*xg=xq+g*GS; int32_t s=0;
                for(int i=0;i<GS;i++){int idx=(src[i>>2]>>((i&3)*2))&3; s+=(2*idx-3)*(int)xg[i];}
                acc+=(float)s*ws*xscale[g]; } yu2s[r]=acc; }
        /* scalar CBT */
        float *ycbts = malloc((size_t)out*sizeof(float));
        int8_t cbi[4]; for(int k=0;k<4;k++){int v=(int)lrintf(cbt.cb[k]*127.0f);
            if(v<-127)v=-127; if(v>127)v=127; cbi[k]=(int8_t)v;}
        for (int r=0;r<out;r++){ float acc=0;
            for(int g=0;g<cbt.ng;g++){ float ws=f16_to_f32(cbt.scales[(size_t)r*cbt.ng+g])*(1.0f/127.0f);
                const uint8_t*src=cbt.q+((size_t)r*cbt.ng+g)*8; const int8_t*xg=xq+g*GS; int32_t s=0;
                for(int i=0;i<GS;i++){int idx=(src[i>>2]>>((i&3)*2))&3; s+=(int)cbi[idx]*(int)xg[i];}
                acc+=(float)s*ws*xscale[g]; } ycbts[r]=acc; }
        double dq=max_abs_diff(yq40,yq40s,out), du=max_abs_diff(yu2,yu2s,out);
        double dc=max_abs_diff(ycbt,ycbts,out);
        printf("NEON-vs-scalar max abs diff:  Q40=%.3e  U2=%.3e  CBT=%.3e  %s\n",
               dq, du, dc, (dq<1e-3 && du<1e-3 && dc<1e-3) ? "[PASS]" : "[FAIL]");
        free(yq40s); free(yu2s); free(ycbts);
    }
#else
    printf("NEON-vs-scalar: (host x86, kernels already scalar -> N/A)\n");
#endif

    free(x); free(yref); free(yq40); free(yu2); free(ycb); free(ycbt);
    free(xq); free(xscale);
    free(q40.scales); free(q40.q);
    free(u2.scales); free(u2.q);
    free(cb2.cb); free(cb2.q);
    free(cbt.scales); free(cbt.q);
    free(W);
}

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : "data";
#if defined(__ARM_FEATURE_DOTPROD)
    printf("# build: aarch64, __ARM_FEATURE_DOTPROD ENABLED (NEON UDOT path)\n");
#else
    printf("# build: scalar fallback (no __ARM_FEATURE_DOTPROD)\n");
#endif
    char mpath[1024]; snprintf(mpath, sizeof mpath, "%s/manifest.txt", dir);
    FILE *mf = fopen(mpath, "r");
    if (!mf) { fprintf(stderr, "cannot open %s\n", mpath); return 1; }
    char shortn[128], fname[256]; int out, in;
    while (fscanf(mf, "%127s %d %d %255s", shortn, &out, &in, fname) == 4) {
        char path[1024]; snprintf(path, sizeof path, "%s/%s", dir, fname);
        eval_tensor(shortn, path, out, in);
    }
    fclose(mf);
    return 0;
}
