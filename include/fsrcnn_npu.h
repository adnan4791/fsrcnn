/*
 * fsrcnn_npu.h -- wrapper inference FSRCNN via NPU RK3588, memakai RKNN C
 * API resmi Rockchip (rknn_api.h). Fungsinya paralel dengan fsrcnn.h (versi
 * CPU) agar main_cpu.c dan main_npu.c bisa berbagi struktur & benchmark
 * harness yang sama.
 *
 * PENTING -- dependency eksternal (SENGAJA TIDAK di-vendor di repo ini):
 * File ini membutuhkan rknn_api.h dan librknnrt.so dari RKNN SDK resmi
 * Rockchip. SDK ini berlisensi proprietary (RKNN SDK License, bukan
 * open-source) sehingga TIDAK ikut didistribusikan bersama proyek ini.
 * Unduh dari repo resmi:
 *     https://github.com/airockchip/rknn-toolkit2
 *     -> rknpu2/runtime/Linux/librknn_api/include/rknn_api.h
 *     -> rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so
 * Letakkan di third_party/rknpu2/{include,lib/aarch64}/ sebelum build
 * (lihat Makefile target `npu`). PENTING: versi librknnrt.so HARUS cocok
 * dengan versi driver NPU kernel di board Orange Pi 5 Anda -- pakai file
 * yang sudah terpasang di image OS board (biasanya /usr/lib/librknnrt.so)
 * kalau tersedia, daripada sembarang versi dari GitHub.
 */
#ifndef FSRCNN_NPU_H
#define FSRCNN_NPU_H

#include "fsrcnn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fsrcnn_npu_ctx fsrcnn_npu_ctx_t; /* opaque */

/* Muat model .rknn (hasil convert_to_rknn.py) dan inisialisasi NPU. Model
 * ini SHAPE TETAP/STATIS (bukan dynamic_input) -- fsrcnn_npu_init() membaca
 * (query, bukan set) shape H x W yang sudah di-compile ke dalamnya, dipakai
 * utk validasi di fsrcnn_npu_forward().
 * Return NULL jika gagal (cek stderr untuk pesan RKNN). */
fsrcnn_npu_ctx_t *fsrcnn_npu_init(const char *rknn_path);

/* Jalankan forward pass pada NPU untuk plane Y (luma) SAJA -- model .rknn
 * ini HANYA dibangun untuk satu shape tetap (lihat scripts/convert_to_rknn.py
 * --luma-h/--luma-w). Plane U/V TIDAK pernah melewati fungsi ini; upscale-nya
 * murni replikasi nearest-neighbor di CPU (chroma_upsample_nn(), yuv_io.h).
 * in->height x in->width HARUS PERSIS SAMA dengan --luma-h/--luma-w yang
 * dipakai saat konversi -- kalau tidak, fungsi ini gagal cepat dgn pesan
 * jelas (validasi lokal terhadap shape hasil query di fsrcnn_npu_init(),
 * TIDAK memanggil rknn_set_input_shapes() -- API itu HANYA valid utk model
 * dynamic_input, dan akan ditolak runtime pada model statis seperti ini).
 * out harus sudah dialokasikan pemanggil sesuai fsrcnn_deconv_output_size().
 * Return 0 sukses. */
int fsrcnn_npu_forward(fsrcnn_npu_ctx_t *ctx, const fsrcnn_tensor_t *in, fsrcnn_tensor_t *out);

void fsrcnn_npu_destroy(fsrcnn_npu_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* FSRCNN_NPU_H */
