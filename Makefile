# Makefile -- FSRCNN video super-resolution: CPU/NEON, NPU RK3588 (RKNN),
# GPU Mali-G610 RK3588 (OpenCL) -- TIGA jalur eksekusi utk dibandingkan.
#
# Target:
#   make cpu-native      -> bin/fsrcnn_cpu_native      (arch host, utk dev/verifikasi)
#   make cpu-aarch64     -> bin/fsrcnn_cpu_aarch64      (cross-compile utk Orange Pi 5)
#   make npu-aarch64     -> bin/fsrcnn_npu_aarch64      (cross-compile, butuh RKNPU2_DIR)
#   make gpu-native      -> bin/fsrcnn_gpu_native       (arch host, butuh libOpenCL;
#                            lihat catatan pocl di include/fsrcnn_gpu.h utk dev tanpa GPU)
#   make gpu-aarch64     -> bin/fsrcnn_gpu_aarch64      (cross-compile utk Orange Pi 5)
#   make all-aarch64     -> ketiga binary aarch64 (cpu + npu + gpu)
#   make yuv-diff        -> bin/yuv_diff  (utilitas verifikasi numerik, TANPA
#                            dependency -- bisa juga langsung `gcc -O2 -o
#                            yuv_diff tools/yuv_diff.c` di board, lihat README
#                            bagian verifikasi hardware fisik)
#   make clean
#
# RKNPU2_DIR: folder berisi SDK Rockchip RKNN (TIDAK ikut di-vendor di repo
# ini karena proprietary -- lihat include/fsrcnn_npu.h untuk cara unduh).
# Override kalau lokasinya berbeda:
#   make npu-aarch64 RKNPU2_DIR=/path/ke/rknpu2
#
# GPU (OpenCL) TIDAK butuh variabel serupa RKNPU2_DIR: header (`CL/cl.h`) &
# ICD loader (`libOpenCL.so`) open-source, biasanya cukup `apt install
# opencl-headers ocl-icd-opencl-dev` di board -- lihat include/fsrcnn_gpu.h
# utk detail driver Mali & fallback pocl (dev tanpa GPU fisik).
# CATATAN cross-compile gpu-aarch64: perlu libOpenCL.so + CL/cl.h utk
# aarch64 di sysroot cross-compiler Anda -- kalau tidak tersedia, build
# LANGSUNG DI board (spt npu-aarch64 di board fisik pada proyek ini) dgn
# `make gpu-aarch64 CROSS_CC=gcc` (native gcc board sudah aarch64, tidak
# perlu cross-toolchain sama sekali -- lihat catatan di README bagian GPU).

CC          ?= gcc
CROSS_CC    ?= aarch64-linux-gnu-gcc
RKNPU2_DIR  ?= third_party/rknpu2

CFLAGS_COMMON := -std=c11 -O3 -Wall -Wextra -Iinclude
LDLIBS_COMMON := -lm

SRC_OPS   := src/fsrcnn_ops.c src/fsrcnn_cpu.c src/cpu_affinity.c
SRC_IO    := src/yuv_io.c
SRC_CPU   := $(SRC_OPS) $(SRC_IO) src/main_cpu.c
SRC_NPU   := src/fsrcnn_npu.c $(SRC_IO) src/fsrcnn_ops.c src/main_npu.c
SRC_GPU   := src/fsrcnn_gpu.c $(SRC_IO) src/fsrcnn_ops.c src/main_gpu.c

BIN_DIR := bin

.PHONY: all-aarch64 cpu-native cpu-aarch64 npu-aarch64 gpu-native gpu-aarch64 yuv-diff clean

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# --- CPU, native (arch mesin yang menjalankan make ini) ---
cpu-native: $(BIN_DIR)
	$(CC) $(CFLAGS_COMMON) $(SRC_CPU) -o $(BIN_DIR)/fsrcnn_cpu_native $(LDLIBS_COMMON)

# --- CPU, cross-compile aarch64 (utk deploy ke Orange Pi 5 / RK3588) ---
# -march=armv8-a: baseline AArch64, NEON/ASIMD wajib ada (bagian dari ISA
# dasar), jadi tidak perlu -mfpu=neon seperti di ARMv7.
cpu-aarch64: $(BIN_DIR)
	$(CROSS_CC) $(CFLAGS_COMMON) -march=armv8-a $(SRC_CPU) -o $(BIN_DIR)/fsrcnn_cpu_aarch64 $(LDLIBS_COMMON)

# --- NPU, cross-compile aarch64 (WAJIB RKNPU2_DIR berisi include/ & lib/aarch64/) ---
npu-aarch64: $(BIN_DIR)
	@test -f "$(RKNPU2_DIR)/include/rknn_api.h" || \
	  (echo "ERROR: $(RKNPU2_DIR)/include/rknn_api.h tidak ditemukan."; \
	   echo "Unduh RKNN SDK resmi Rockchip dulu -- lihat include/fsrcnn_npu.h"; exit 1)
	@test -f "$(RKNPU2_DIR)/lib/aarch64/librknnrt.so" || \
	  (echo "ERROR: $(RKNPU2_DIR)/lib/aarch64/librknnrt.so tidak ditemukan."; exit 1)
	$(CROSS_CC) $(CFLAGS_COMMON) -march=armv8-a \
	  -I$(RKNPU2_DIR)/include $(SRC_NPU) \
	  -L$(RKNPU2_DIR)/lib/aarch64 -lrknnrt \
	  -o $(BIN_DIR)/fsrcnn_npu_aarch64

# --- GPU, native (arch mesin yang menjalankan make ini; butuh libOpenCL) ---
gpu-native: $(BIN_DIR)
	$(CC) $(CFLAGS_COMMON) $(SRC_GPU) -o $(BIN_DIR)/fsrcnn_gpu_native $(LDLIBS_COMMON) -lOpenCL

# --- GPU, cross-compile aarch64 (utk deploy ke Orange Pi 5 / Mali-G610) ---
gpu-aarch64: $(BIN_DIR)
	$(CROSS_CC) $(CFLAGS_COMMON) -march=armv8-a $(SRC_GPU) -o $(BIN_DIR)/fsrcnn_gpu_aarch64 $(LDLIBS_COMMON) -lOpenCL

all-aarch64: cpu-aarch64 npu-aarch64 gpu-aarch64

# --- Utilitas verifikasi numerik (bandingkan dua file .yuv byte-level) ---
# Pure C, tanpa dependency (bukan bagian dari ke-3 engine) -- dipakai utk
# menutup loop verifikasi LANGSUNG DI board fisik (susi_aarch64.yuv vs
# susi_npu.yuv vs susi_gpu.yuv), tanpa perlu python/numpy di sana. Karena
# tidak butuh library apa pun, satu perintah `$(CC)` ini sudah cukup baik
# di host maupun native di board (tidak perlu target *-aarch64 terpisah).
yuv-diff: $(BIN_DIR)
	$(CC) $(CFLAGS_COMMON) tools/yuv_diff.c -o $(BIN_DIR)/yuv_diff

clean:
	rm -rf $(BIN_DIR)
