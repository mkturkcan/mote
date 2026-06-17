// Two-tier lossless sparse FFN vs dense, real Gemma blk.20 weights + real ffn_norm-20 inputs.
// gate/up: [OUT=12288][IN=1536] (neuron-major rows => skipping a neuron skips a contiguous row).
// down:    [OUT=1536][IN=12288] row-major; we transpose ONCE to downT[IN=12288][OUT=1536]
//          (neuron-major) so skipping neuron i skips a contiguous 1536-row. This is the key layout.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define IN 1536
#define MID 12288
static float *gate,*up,*down,*downT; // gate/up: MID*IN ; down: IN*MID ; downT: MID*IN
static inline float gelu(float x){const float k=0.7978845608028654f;return 0.5f*x*(1.f+tanhf(k*(x+0.044715f*x*x*x)));}
static float* load(const char*p,long n){FILE*f=fopen(p,"rb");if(!f){perror(p);exit(1);}float*b=malloc(n*4);fread(b,4,n,f);fclose(f);return b;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}

int main(int argc,char**argv){
  double thr1=argc>1?atof(argv[1]):0.01; // gelu(gate) tier-1 threshold (skip up+down)
  double thr2=argc>2?atof(argv[2]):0.01; // geglu tier-2 threshold (skip down)
  int nth=argc>3?atoi(argv[3]):8;
  omp_set_num_threads(nth);
  gate=load("data/gate.f32",(long)MID*IN); up=load("data/up.f32",(long)MID*IN); down=load("data/down.f32",(long)IN*MID);
  downT=malloc((long)MID*IN*4);
  for(int o=0;o<IN;o++)for(int i=0;i<MID;i++) downT[(long)i*IN+o]=down[(long)o*MID+i]; // transpose once
  // precompute downT row L2 norms (per-neuron output influence)
  float*dn=malloc(MID*4);
  for(int i=0;i<MID;i++){double s=0;const float*w=downT+(long)i*IN;for(int o=0;o<IN;o++)s+=(double)w[o]*w[o];dn[i]=sqrtf(s);}
  // load real inputs
  FILE*xf=fopen("/tmp/x20.f32","rb");fseek(xf,0,SEEK_END);long xb=ftell(xf);fseek(xf,0,SEEK_SET);
  int ntok=xb/4/IN; float*X=malloc(xb);fread(X,4,xb/4,xf);fclose(xf);
  printf("tokens=%d thr1=%.3g thr2=%.3g threads=%d\n",ntok,thr1,thr2,nth);

  float *outD=malloc(IN*4),*outS=malloc(IN*4),*g=malloc(MID*4),*ge=malloc(MID*4);
  double errsum=0,relsum=0; long bytesD=0,bytesS=0; double tD=0,tS=0; long act1=0,act2=0;

  for(int t=0;t<ntok;t++){
    float*x=X+(long)t*IN;
    // ---- DENSE ----
    double a=now();
    #pragma omp parallel for schedule(static)
    for(int o=0;o<MID;o++){float s=0,u=0;const float*wg=gate+(long)o*IN,*wu=up+(long)o*IN;
      for(int i=0;i<IN;i++){s+=wg[i]*x[i];u+=wu[i]*x[i];} ge[o]=gelu(s)*u;}
    memset(outD,0,IN*4);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<IN;o++){float s=0;const float*wd=down+(long)o*MID;for(int i=0;i<MID;i++)s+=wd[i]*ge[i];outD[o]=s;}
    tD+=now()-a; bytesD+=(long)3*MID*IN*4;

    // ---- SPARSE two-tier ----
    a=now();
    // gate (full) + gelu
    float gmax=0;
    #pragma omp parallel for schedule(static) reduction(max:gmax)
    for(int o=0;o<MID;o++){float s=0;const float*wg=gate+(long)o*IN;for(int i=0;i<IN;i++)s+=wg[i]*x[i];g[o]=gelu(s);float ag=fabsf(g[o]);if(ag>gmax)gmax=ag;}
    long lb=(long)MID*IN*4; // gate fully read
    // tier1 active (skip up+down where |gelu(gate)|<thr1*gmax); compute up+geglu for active1
    float impmax=0; long a1=0;
    #pragma omp parallel for schedule(static) reduction(max:impmax) reduction(+:a1,lb)
    for(int o=0;o<MID;o++){ if(fabsf(g[o])<thr1*gmax){ge[o]=0;continue;} a1++;
      float u=0;const float*wu=up+(long)o*IN;for(int i=0;i<IN;i++)u+=wu[i]*x[i]; lb+=IN*4; // read up row
      ge[o]=g[o]*u; float imp=fabsf(ge[o])*dn[o]; if(imp>impmax)impmax=imp; }
    // tier2: zero geglu below thr2*gemax (skip its down row); accumulate down over survivors
    memset(outS,0,IN*4); long a2=0;
    #pragma omp parallel
    { float*loc=calloc(IN,4);
      #pragma omp for schedule(dynamic,64) reduction(+:a2,lb) nowait
      for(int o=0;o<MID;o++){ if(ge[o]==0||fabsf(ge[o])*dn[o]<thr2*impmax) continue; a2++; lb+=IN*4;
        const float*wt=downT+(long)o*IN; float c=ge[o]; for(int i=0;i<IN;i++)loc[i]+=c*wt[i]; }
      #pragma omp critical
      for(int i=0;i<IN;i++)outS[i]+=loc[i]; free(loc);
    }
    tS+=now()-a; bytesS+=lb; act1+=a1; act2+=a2;
    // error
    double e=0,n=0; for(int i=0;i<IN;i++){double d=outS[i]-outD[i];e+=d*d;n+=(double)outD[i]*outD[i];}
    relsum+=sqrt(e/n);
  }
  printf("output rel-err (L2, vs dense): %.4f%%\n",100*relsum/ntok);
  printf("avg active: tier1(up+down)=%.1f%% (%ld/%d)  tier2(down)=%.1f%%\n",
     100.0*act1/ntok/MID, act1/ntok, MID, 100.0*act2/ntok/MID);
  printf("bytes/token: dense=%.1f MB  sparse=%.1f MB  (%.1f%% of dense)\n",
     (double)bytesD/ntok/1e6,(double)bytesS/ntok/1e6,100.0*bytesS/bytesD);
  printf("time/token:  dense=%.3f ms  sparse=%.3f ms  speedup=%.2fx\n",
     1e3*tD/ntok,1e3*tS/ntok,tD/tS);
  return 0;
}
