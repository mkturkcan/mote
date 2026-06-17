# CMake toolchain: cross-compile for Raspberry Pi 5 (BCM2712, Cortex-A76, Armv8.2-A)
# A76 includes: NEON, FEAT_DotProd (UDOT/SDOT), FEAT_FP16. It does NOT have i8mm/SVE.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Use -march (not -mcpu) so the ISA features appear as literal "+dotprod" in the flags: llama.cpp's KleidiAI
# cmake gates the A76 dotprod matmul ukernels on a STRING match for "+dotprod" in the arch flags (a plain
# -mcpu=cortex-a76 implies dotprod but lacks the literal, so the kai_*_neon_dotprod kernels get dropped and the
# link fails). -march=armv8.2-a+dotprod+fp16 + -mtune=cortex-a76 = identical A76 ISA/scheduling to -mcpu.
set(A76_FLAGS "-march=armv8.2-a+dotprod+fp16 -mtune=cortex-a76 -O3 -fno-finite-math-only")
set(CMAKE_C_FLAGS_INIT   "${A76_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${A76_FLAGS}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
