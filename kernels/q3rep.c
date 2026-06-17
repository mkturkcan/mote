// Step 1: bit-exact gate for the q3_K repacked gemv.
// Reference: q3_K vec_dot (int accumulate per sub-block * 6-bit scale * d, like ggml_vec_dot_q3_K_q8_K).
// Candidate: my make_block_q3_Kx8 (repack 8 rows) + my gemv -> must equal the reference for each row.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#define QK_K 256
typedef struct { uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; float d; } block_q3_K; // float d for test
typedef struct { int8_t qs[QK_K]; float d; } block_q8; // simplified q8 activation (one super-block)

// unpack the 16 signed 6-bit scales (value = s-32) from the 12-byte packed array (mirrors ggml)
static void unpack_scales(const uint8_t* sc, int8_t out[16]){
    const uint32_t k1=0x03030303,k2=0x0f0f0f0f; uint32_t aux[4]; memcpy(aux,sc,12);
    uint32_t tmp=aux[2];
    aux[2]=((aux[0]>>4)&k2)|(((tmp>>4)&k1)<<4);
    aux[3]=((aux[1]>>4)&k2)|(((tmp>>6)&k1)<<4);
    aux[0]=(aux[0]&k2)|(((tmp>>0)&k1)<<4);
    aux[1]=(aux[1]&k2)|(((tmp>>2)&k1)<<4);
    const int8_t* s=(const int8_t*)aux; for(int i=0;i<16;i++) out[i]=s[i];
}
// REFERENCE: dot one q3_K row with q8 activation, int-accumulate per 16-wide sub-block * scale.
static float ref_dot(const block_q3_K* x, const block_q8* a){
    int8_t sc[16]; unpack_scales(x->scales, sc);
    const uint8_t* q=x->qs; const uint8_t* hm=x->hmask; uint8_t m=1; int is=0; int wi=0; double acc=0;
    for(int n=0;n<QK_K;n+=128){ int shift=0;
        for(int j=0;j<4;j++){
            int s0=sc[is++]-32; long sum=0;
            for(int l=0;l<16;l++){ int w=((q[l]>>shift)&3) - ((hm[l]&m)?0:4); sum+= w*(long)a->qs[wi+l]; }
            acc += (double)s0*sum;
            int s1=sc[is++]-32; sum=0;
            for(int l=0;l<16;l++){ int w=((q[l+16]>>shift)&3) - ((hm[l+16]&m)?0:4); sum+= w*(long)a->qs[wi+16+l]; }
            acc += (double)s1*sum;
            wi+=32; shift+=2; m<<=1;
        }
        q+=32;
    }
    return (float)(acc * x->d * a->d);
}
// repacked 8-row block: simple side-by-side (correctness first; SIMD interleave is a later step)
// interleaved (q2_K-style): qs/hmask packed as 8-byte chunks round-robin across the 8 cols; scales unpacked+interleaved
typedef struct { float d[8]; uint8_t qs[8*QK_K/4]; uint8_t hmask[8*QK_K/8]; int8_t scales[8*16]; } block_q3_Kx8;
// interleaved byte index for column j, logical byte p of a plane with `nb` bytes/col (chunk=8)
static inline int IDX(int j,int p){ return ((p/8)*8 + j)*8 + (p%8); }
static void make_block_q3_Kx8(const block_q3_K in[8], block_q3_Kx8* o){
    for(int j=0;j<8;j++){ o->d[j]=in[j].d;
        for(int p=0;p<QK_K/4;p++) o->qs[IDX(j,p)]=in[j].qs[p];
        for(int p=0;p<QK_K/8;p++) o->hmask[IDX(j,p)]=in[j].hmask[p];
        int8_t sc[16]; unpack_scales(in[j].scales,sc); for(int s2=0;s2<16;s2++) o->scales[j*16+s2]=sc[s2]; }
}
// CANDIDATE gemv: 8 columns, reconstruct + dot, must match ref_dot per column.
static void my_gemv(const block_q3_Kx8* b, const block_q8* a, float out[8]){
    for(int j=0;j<8;j++){
        const int8_t* sc=&b->scales[j*16]; uint8_t m=1; int is=0; int wi=0; int qbase=0; double acc=0;
        for(int n=0;n<QK_K;n+=128){ int shift=0;
            for(int jj=0;jj<4;jj++){
                int s0=sc[is++]-32; long sum=0;
                for(int l=0;l<16;l++){ int q=b->qs[IDX(j,qbase+l)]; int h=b->hmask[IDX(j,l)]; int w=((q>>shift)&3)-((h&m)?0:4); sum+=w*(long)a->qs[wi+l]; }
                acc+=(double)s0*sum;
                int s1=sc[is++]-32; sum=0;
                for(int l=0;l<16;l++){ int q=b->qs[IDX(j,qbase+16+l)]; int h=b->hmask[IDX(j,16+l)]; int w=((q>>shift)&3)-((h&m)?0:4); sum+=w*(long)a->qs[wi+16+l]; }
                acc+=(double)s1*sum;
                wi+=32; shift+=2; m<<=1;
            }
            qbase+=32;
        }
        out[j]=(float)(acc*b->d[j]*a->d);
    }
}
int main(){
    srand(7); int fails=0;
    for(int t=0;t<200;t++){
        block_q3_K rows[8]; block_q8 a;
        for(int j=0;j<8;j++){ for(int i=0;i<QK_K/8;i++)rows[j].hmask[i]=rand()&0xff;
            for(int i=0;i<QK_K/4;i++)rows[j].qs[i]=rand()&0xff; for(int i=0;i<12;i++)rows[j].scales[i]=rand()&0xff;
            rows[j].d=((rand()%2000)-1000)/1000.0f; }
        for(int i=0;i<QK_K;i++)a.qs[i]=(rand()%255)-127; a.d=((rand()%2000)-1000)/1000.0f;
        float ref[8],mine[8]; for(int j=0;j<8;j++) ref[j]=ref_dot(&rows[j],&a);
        block_q3_Kx8 rp; make_block_q3_Kx8(rows,&rp); my_gemv(&rp,&a,mine);
        for(int j=0;j<8;j++){ float e=fabsf(ref[j]-mine[j]); float den=fabsf(ref[j])+1e-6f;
            if(e/den>1e-5f){ fails++; if(fails<4)printf("  MISMATCH t=%d j=%d ref=%.6f mine=%.6f\n",t,j,ref[j],mine[j]); } }
    }
    printf("%s  (%d/1600 column-dots mismatched)\n", fails? "FAIL":"PASS — repack+gemv bit-exact vs reference", fails);
    return fails?1:0;
}
