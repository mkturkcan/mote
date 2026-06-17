// Streaming memory-bandwidth microbenchmark for the Pi 5 (BCM2712 / Cortex-A76).
// Measures pure READ BW (NEON loads) and COPY BW (load+store) at 1..4 threads, on a buffer far larger than
// cache, to find the achievable ceiling vs the ~9-11 GB/s the decode actually gets.
// build:  gcc -O3 -march=armv8.2-a+fp16+dotprod -pthread membw.c -o membw
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#include <arm_neon.h>

#define GB (1024.0*1024.0*1024.0)
static size_t BYTES = (size_t)768*1024*1024;   // 768 MB per buffer, >> 2MB L2
static int    REPS  = 8;

typedef struct { char *a, *b; size_t off, len; int mode; uint64_t sink; } targ_t;

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

static void* worker(void *p){
    targ_t *t = (targ_t*)p;
    const char *a = t->a + t->off; char *b = t->b + t->off; size_t len = t->len;
    for (int r=0; r<REPS; ++r){
        if (t->mode==0){ // READ: sum via 4 NEON accumulators (64B/iter)
            uint32x4_t s0=vdupq_n_u32(0),s1=s0,s2=s0,s3=s0;
            for (size_t i=0;i+64<=len;i+=64){
                s0=vaddq_u32(s0,vld1q_u32((const uint32_t*)(a+i)));
                s1=vaddq_u32(s1,vld1q_u32((const uint32_t*)(a+i+16)));
                s2=vaddq_u32(s2,vld1q_u32((const uint32_t*)(a+i+32)));
                s3=vaddq_u32(s3,vld1q_u32((const uint32_t*)(a+i+48)));
            }
            uint32x4_t s=vaddq_u32(vaddq_u32(s0,s1),vaddq_u32(s2,s3));
            t->sink += vgetq_lane_u32(s,0)+vgetq_lane_u32(s,1)+vgetq_lane_u32(s,2)+vgetq_lane_u32(s,3);
        } else { // COPY
            memcpy(b, a, len);
            t->sink += b[r % len];
        }
    }
    return NULL;
}

static double run(char*a,char*b,int nth,int mode){
    pthread_t th[4]; targ_t ta[4];
    size_t chunk = BYTES/nth; chunk &= ~((size_t)63);
    double t0=now();
    for (int i=0;i<nth;i++){ ta[i]=(targ_t){a,b,i*chunk,chunk,mode,0}; pthread_create(&th[i],NULL,worker,&ta[i]); }
    uint64_t sink=0; for (int i=0;i<nth;i++){ pthread_join(th[i],NULL); sink+=ta[i].sink; }
    double dt=now()-t0;
    volatile uint64_t use=sink; (void)use;
    double moved = (double)chunk*nth*REPS*(mode==1?2.0:1.0); // copy moves 2x
    return moved/GB/dt;
}

int main(int argc,char**argv){
    if (argc>1) BYTES=(size_t)atoll(argv[1])*1024*1024;
    char *a=aligned_alloc(64,BYTES), *b=aligned_alloc(64,BYTES);
    memset(a,1,BYTES); memset(b,2,BYTES);
    printf("buffer=%zuMB reps=%d\n", BYTES/1024/1024, REPS);
    printf("%-6s %12s %12s\n","thr","READ GB/s","COPY GB/s");
    for (int nth=1;nth<=4;nth++){
        double rd=run(a,b,nth,0), cp=run(a,b,nth,1);
        printf("%-6d %12.2f %12.2f\n", nth, rd, cp);
    }
    free(a); free(b); return 0;
}
