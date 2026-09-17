/*
 * fsrcnn_gpu.h -- wrapper inference FSRCNN via GPU (Mali-G610 di RK3588/
 * Orange Pi 5) memakai OpenCL. Fungsinya paralel dengan fsrcnn.h (CPU) dan
 * fsrcnn_npu.h (NPU) agar main_cpu.c/main_npu.c/main_gpu.c berbagi struktur
 * & benchmark harness yang sama -- proyek ini jadi punya TIGA jalur
 * akselerasi berbeda utk perbandingan langsung (studi kasus heterogen utk
 * buku ajar: CPU/NEON vs NPU/RKNN vs GPU/OpenCL, tiga unit compute berbeda
 * pada SoC yang sama).
 *
 * PENTING -- dependency eksternal:
 * Header OpenCL (`CL/cl.h`, dari paket `opencl-headers`/Khronos) dan
 * `libOpenCL.so` (ICD loader, dari paket `ocl-icd-opencl-dev` atau serupa)
 * BEBAS/open-source, TIDAK seperti RKNN SDK -- aman di-assume tersedia via
 * package manager. YANG proprietary & TIDAK di-vendor di sini adalah DRIVER
 * OpenCL Mali itu sendiri (`libmali-valhall-g610-*` atau ekuivalen) yang
 * mengimplementasikan ICD utk GPU Mali-G610 fisik -- ini biasanya SUDAH
 * TERPASANG di image OS Orange Pi 5 yang mendukung GPU (mis. Armbian/Ubuntu
 * varian "Mali" dari komunitas Orange Pi / Joshua-Riek), TAPI TIDAK ADA di
 * image server/minimal biasa. Cek dgn `clinfo -l` di board Anda -- kalau
 * kosong, install driver Mali (BUKAN mesa/Panfrost -- driver open-source
 * Panfrost TIDAK punya implementasi OpenCL yang lengkap/stabil per saat
 * proyek ini ditulis) sebelum memakai modul ini.
 *
 * FALLBACK BERGUNA UTK PENGEMBANGAN: kalau tidak ada GPU OpenCL sama sekali
 * (mis. develop di laptop x86 tanpa GPU vendor-supported), pasang
 * `pocl-opencl-icd` (implementasi OpenCL open-source yang jalan di CPU) --
 * modul ini akan tetap berfungsi (walau tanpa akselerasi GPU sungguhan),
 * berguna utk verifikasi numerik/logic kernel sebelum deploy ke board.
 */
#ifndef FSRCNN_GPU_H
#define FSRCNN_GPU_H

#include "fsrcnn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fsrcnn_gpu_ctx fsrcnn_gpu_ctx_t; /* opaque */

/* Inisialisasi OpenCL: pilih platform/device (prioritas: device GPU dgn
 * nama mengandung "Mali"; fallback: device GPU apa pun; fallback terakhir:
 * device APAPUN yg tersedia -- termasuk implementasi CPU spt pocl, lihat
 * catatan fallback di atas), compile kernel, alokasi buffer bobot (upload
 * sekali, dipakai ulang di setiap panggilan forward).
 * Return NULL jika gagal (cek stderr utk pesan OpenCL, termasuk build log
 * kernel kalau kompilasi kernel gagal). */
fsrcnn_gpu_ctx_t *fsrcnn_gpu_init(void);

/* Jalankan forward pass pada GPU utk plane Y (luma) SAJA -- identik dgn
 * fsrcnn_forward() (CPU) & fsrcnn_npu_forward() (NPU) secara matematis
 * (PReLU per-layer, replicate padding via clamp-addressing, custom deconv
 * via formulasi GATHER -- lihat catatan derivasi di fsrcnn_gpu.c), TIDAK
 * seperti NPU, engine GPU ini FULLY-CONVOLUTIONAL: H,W boleh berapa pun
 * (buffer device dialokasikan ulang otomatis kalau H,W berubah dari
 * panggilan sebelumnya -- lihat cur_h/cur_w di fsrcnn_gpu.c).
 * out harus sudah dialokasikan pemanggil sesuai fsrcnn_deconv_output_size().
 * Return 0 sukses. */
int fsrcnn_gpu_forward(fsrcnn_gpu_ctx_t *ctx, const fsrcnn_tensor_t *in, fsrcnn_tensor_t *out);

void fsrcnn_gpu_destroy(fsrcnn_gpu_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* FSRCNN_GPU_H */
