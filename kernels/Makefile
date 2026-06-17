# 2-bit FFN kernel microbenchmark: native x86 + cross aarch64 (run under qemu).
CC_X86   ?= gcc
CC_ARM   ?= aarch64-linux-gnu-gcc
QEMU     ?= qemu-aarch64-static
ARM_SYSROOT ?= /usr/aarch64-linux-gnu

CFLAGS_COMMON = -O3 -std=c11 -Wall -Wextra -lm
# x86: no dotprod feature macro -> scalar fallback path is compiled.
CFLAGS_X86 = -O3 -march=native
# aarch64 Cortex-A76: enables NEON + FEAT_DotProd (defines __ARM_FEATURE_DOTPROD).
CFLAGS_ARM = -mcpu=cortex-a76 -O3 -static

.PHONY: all x86 arm run run-x86 run-arm clean data

all: bench_x86 bench_arm

data: data/manifest.txt
data/manifest.txt:
	python3 dump_weights.py

bench_x86: bench.c
	$(CC_X86) $(CFLAGS_X86) -O3 -std=c11 -Wall -o $@ bench.c -lm

bench_arm: bench.c
	$(CC_ARM) $(CFLAGS_ARM) -std=c11 -Wall -o $@ bench.c -lm

run-x86: bench_x86 data
	./bench_x86 data

run-arm: bench_arm data
	$(QEMU) ./bench_arm data

run: run-x86 run-arm

clean:
	rm -f bench_x86 bench_arm
