/*
 * yuv_io.h -- baca/tulis frame mentah YUV 4:2:0 planar (format ".yuv" umum:
 * urutan byte per frame = Y (H*W) lalu U ((H/2)*(W/2)) lalu V ((H/2)*(W/2)),
 * 8-bit per sampel, tanpa header -- format yang sama dipakai ffmpeg -f rawvideo
 * -pix_fmt yuv420p).
 *
 * Modul ini JUGA menjembatani antara domain uint8 [0,255] (representasi
 * file) dan domain float32 [0,1] (domain kerja FSRCNN, lihat fsrcnn.h) --
 * pemisahan ini membuat inference engine tidak perlu tahu apa pun soal
 * format file video.
 */
#ifndef YUV_IO_H
#define YUV_IO_H

#include <stdio.h>
#include "fsrcnn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int width;   /* lebar luma (Y) */
    int height;  /* tinggi luma (Y) */
} yuv420_dims_t;

/* Ukuran satu frame YUV 4:2:0 dalam byte. */
size_t yuv420_frame_size(const yuv420_dims_t *d);

/* Baca satu frame dari file mentah ke 3 tensor float [0,1] (channels=1
 * masing-masing). Tensor HARUS sudah dialokasikan oleh pemanggil dengan
 * ukuran yang sesuai (Y: HxW, U/V: (H/2)x(W/2) dibulatkan ke atas).
 * Return 0 sukses, 1 EOF (tidak ada frame lagi), -1 error I/O. */
int yuv420_read_frame(FILE *f, const yuv420_dims_t *d,
                       fsrcnn_tensor_t *y, fsrcnn_tensor_t *u, fsrcnn_tensor_t *v);

/* Tulis satu frame (Y,U,V dalam domain float [0,1]) ke file mentah,
 * meng-clamp balik ke uint8 [0,255]. Dimensi output HARUS sudah 2x dari
 * dimensi input asli (hasil fsrcnn_forward pada tiap plane). */
int yuv420_write_frame(FILE *f, const fsrcnn_tensor_t *y,
                        const fsrcnn_tensor_t *u, const fsrcnn_tensor_t *v);

/* Dimensi chroma (4:2:0) dari dimensi luma: ceil(H/2) x ceil(W/2). */
void yuv420_chroma_dims(int luma_h, int luma_w, int *chroma_h, int *chroma_w);

/*
 * Upscale plane CHROMA (U atau V) via replikasi nearest-neighbor blok
 * scale x scale -- BUKAN lewat FSRCNN. Ini PERSIS mereproduksi perilaku
 * blok "U Component"/"V Component" pada main() di source_channel_pixel.c
 * (setiap piksel sumber diduplikasi menjadi blok scale x scale piksel
 * identik pada output). Model FSRCNN yang tersedia hanya dilatih utk
 * plane Y -- JANGAN panggil fsrcnn_forward() pada U/V, pakai fungsi ini.
 *   in  : tensor (1,H,W). out : tensor (1,H*scale,W*scale), sudah
 *         dialokasikan pemanggil dgn fsrcnn_tensor_alloc().
 */
void chroma_upsample_nn(const fsrcnn_tensor_t *in, fsrcnn_tensor_t *out, int scale);

#ifdef __cplusplus
}
#endif

#endif /* YUV_IO_H */
