/*
 * fsrcnn.h -- API inti inference engine FSRCNN (versi CPU, C murni + NEON).
 *
 * Konvensi tensor: channel-major (C,H,W), setiap channel adalah blok
 * kontigu H*W float32. Ini SAMA dengan konvensi PyTorch/ONNX (NCHW dengan
 * N=1 diimplisitkan), sehingga bobot dari fsrcnn_weights.h (hasil generate
 * dari file .txt) bisa dipakai tanpa transformasi layout tambahan.
 *
 * Konvensi nilai piksel: input & output adalah float32 pada rentang [0,1]
 * (piksel_8bit / 255.0). Konversi ke/dari uint8 dilakukan di lapisan I/O
 * (yuv_io.c), BUKAN di dalam inference engine ini -- pemisahan tanggung
 * jawab ini penting agar fsrcnn_forward() bisa dipanggil untuk plane Y
 * MAUPUN plane U/V (resolusi berbeda) tanpa modifikasi.
 */
#ifndef FSRCNN_H
#define FSRCNN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *data;     /* contigu, layout (channels, height, width) */
    int channels;
    int height;
    int width;
} fsrcnn_tensor_t;

/* Alokasi/dealokasi tensor (data di-nol-kan via calloc). */
int  fsrcnn_tensor_alloc(fsrcnn_tensor_t *t, int channels, int height, int width);
void fsrcnn_tensor_free(fsrcnn_tensor_t *t);

/*
 * Jalankan forward pass FSRCNN lengkap (8 layer) pada SATU channel plane.
 *   in  : tensor (1, H, W), sudah dinormalisasi ke [0,1].
 *   out : tensor yang akan diisi, HARUS sudah dialokasikan dengan
 *         fsrcnn_tensor_alloc(out, 1, 2*H, 2*W) oleh pemanggil.
 * Fully-convolutional -> H,W boleh berapa pun.
 *
 * PENTING (sesuai referensi source_channel_pixel.c): model ini HANYA
 * dilatih & dipakai untuk plane Y (luma). Plane U/V TIDAK melewati
 * jaringan ini -- upscale-nya memakai replikasi nearest-neighbor sederhana
 * (lihat chroma_upsample_nn() di yuv_io.h). Memanggil fsrcnn_forward() pada
 * plane U/V akan menghasilkan output yang salah (memakai bobot yang
 * dilatih utk statistik luma, bukan chroma) -- JANGAN dilakukan.
 * Return 0 jika sukses.
 */
int fsrcnn_forward(const fsrcnn_tensor_t *in, fsrcnn_tensor_t *out);

/* --- Primitif conv, diekspos agar bisa diuji/dibandingkan unit-per-unit --- */

/*
 * Aktivasi PENTING: model ini dilatih dengan PReLU per-LAYER (satu skalar
 * slope per layer, BUKAN plain ReLU, dan bukan per-channel) --
 * v>=0 ? v : slope*v -- persis seperti diimplementasikan pada referensi
 * source_channel_pixel.c (fungsi Max/Min). Slope per layer sudah termasuk
 * NEGATIF (mis. layer1 = -0.8986); JANGAN diasumsikan >= 0.
 */
typedef enum {
    FSRCNN_ACT_NONE = 0,   /* tanpa aktivasi (dipakai layer 8 / deconv) */
    FSRCNN_ACT_PRELU = 1
} fsrcnn_activation_t;

/* Conv2D "same" (stride=1) + aktivasi opsional. in: (cin,H,W).
 * weight: (cout,cin,kh,kw). bias: (cout). out: (cout,H,W), harus sudah
 * dialokasikan oleh pemanggil. pad harus memenuhi kh == 2*pad+1 (dipakai
 * skema replicate-pad eksplisit -- lihat catatan besar di fsrcnn_ops.c).
 * prelu_slope dipakai hanya jika act==FSRCNN_ACT_PRELU. */
void fsrcnn_conv2d(const float *in, int cin, int H, int W,
                    const float *weight, const float *bias, int cout,
                    int kh, int kw, int pad,
                    fsrcnn_activation_t act, float prelu_slope,
                    float *out);

/*
 * Upsampling layer 8 -- BUKAN ConvTranspose2d generik/standar PyTorch.
 * Port langsung dari deconv_reduce_hwc() di source_channel_pixel.c: input
 * di-replicate-pad dgn `border`, "full transposed convolution" (scatter),
 * lalu crop jendela Hin*stride x Win*stride dari offset tetap. Lihat
 * penjelasan lengkap di fsrcnn_ops.c.
 *   in     : (cin,Hin,Win).
 *   weight : layout NATIVE (cin,1,fsize,fsize) row-major -- SAMA dgn W8 di
 *            fsrcnn_weights.h, TIDAK perlu repack.
 *   bias   : SATU skalar (bukan per-channel -- cout tersirat = 1).
 *   out    : (1,Hin*stride,Win*stride), harus sudah dialokasikan pemanggil.
 * Proyek ini HANYA memakai fsize=9, stride=FSRCNN_SCALE(=2), border=1
 * (konstanta arsitektur model, identik dgn referensi).
 */
void fsrcnn_deconv_reduce(const float *in, int cin, int Hin, int Win,
                           const float *weight, float bias,
                           int fsize, int stride, int border,
                           float *out);

/* Hout=Hin*stride, Wout=Win*stride (ukuran keluaran fsrcnn_deconv_reduce). */
void fsrcnn_deconv_output_size(int Hin, int Win, int fsize, int stride, int border,
                                int *Hout, int *Wout);

#ifdef __cplusplus
}
#endif

#endif /* FSRCNN_H */
