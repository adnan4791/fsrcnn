/*
 * fsrcnn_ops.c -- primitif Conv2D / deconv (upsampling) untuk FSRCNN.
 *
 * KOREKSI PENTING (hasil validasi terhadap referensi otoritatif
 * source_channel_pixel.c, yang jadi acuan pelatihan bobot model ini):
 *   1. Padding pada SEMUA layer conv memakai REPLICATE/EDGE padding
 *      (menduplikasi piksel tepi terdekat), BUKAN zero-padding seperti pada
 *      iterasi awal proyek ini. Perbedaan ini terdeteksi lewat validasi
 *      numerik langsung terhadap keluaran referensi (bin-compare) -- dengan
 *      zero-pad, selisih piksel dekat tepi gambar bisa mencapai >100 (dari
 *      skala 0-255), padahal bagian tengah gambar sudah nyaris identik.
 *   2. Layer 8 (upsampling) BUKAN ConvTranspose2d standar PyTorch (yang
 *      meng-crop dari hasil scatter atas input ASLI/tanpa-pad). Referensi
 *      memakai algoritma custom: (a) replicate-pad input dgn border=1,
 *      (b) scatter/"full" transposed-convolution atas input yang SUDAH
 *      di-pad tsb (tanpa crop), (c) ekstraksi jendela ukuran Hin*2 x Win*2
 *      dari kanvas hasil scatter, mulai offset tetap ((fsize+1)/2 +
 *      stride*border - 1) = 6 pada kedua dimensi. Diimplementasikan di
 *      fsrcnn_deconv_reduce() di bawah, PERSIS meniru deconv_reduce_hwc().
 *
 * Desain kunci #1 (menghilangkan cabang di hot loop, tetap berlaku):
 *   Padding eksplisit (kini replicate, bukan zero) dibuat SEKALI sebelum
 *   loop utama, sehingga loop konvolusi tidak perlu mengecek batas array
 *   sama sekali -- trade-off memori/copy vs branch mispredict.
 *
 * Desain kunci #2 (SIMD, tetap berlaku untuk Conv2D): loop innermost (dot
 * product sepanjang kw) divektorisasi NEON 4-lebar saat kw>=4.
 *
 * fsrcnn_deconv_reduce() SENGAJA belum divektorisasi NEON: aksesnya
 * ber-stride 81 float per kanal (layout weight native (D,1,9,9) dari
 * fsrcnn_weights.h, BUKAN layout "tap-major" hasil repack seperti pada
 * referensi HWC) -- didokumentasikan sbg latihan optimasi lanjutan
 * (bisa dipercepat dgn repack bobot ke tap-major sekali di awal, mirip
 * repack_weights_deconv() pada referensi, lalu loop channel jadi kontigu).
 */
#include "fsrcnn.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define FSRCNN_HAVE_NEON 1
#endif

int fsrcnn_tensor_alloc(fsrcnn_tensor_t *t, int channels, int height, int width) {
    t->channels = channels;
    t->height = height;
    t->width = width;
    size_t n = (size_t)channels * height * width;
    t->data = (float *)calloc(n, sizeof(float));
    return t->data ? 0 : -1;
}

void fsrcnn_tensor_free(fsrcnn_tensor_t *t) {
    free(t->data);
    t->data = NULL;
}

/*
 * Replicate/edge-pad (C,H,W) -> (C,H+2*pad,W+2*pad): border diisi dengan
 * menduplikasi piksel tepi terdekat -- padanan CHW dari pad_image_hwc()
 * pada source_channel_pixel.c. pad=0 valid (degenerasi jadi copy biasa).
 */
static float *pad_chw_replicate(const float *in, int C, int H, int W, int pad, int *PH_out, int *PW_out) {
    int PH = H + 2 * pad, PW = W + 2 * pad;
    float *out = (float *)malloc((size_t)C * PH * PW * sizeof(float));
    if (!out) return NULL;

    for (int c = 0; c < C; c++) {
        const float *src = in + (size_t)c * H * W;
        float *dst = out + (size_t)c * PH * PW;

        /* Tengah: copy langsung. */
        for (int y = 0; y < H; y++) {
            memcpy(dst + (size_t)(y + pad) * PW + pad, src + (size_t)y * W, (size_t)W * sizeof(float));
        }
        /* Baris atas/bawah: replikasi baris tepi pertama/terakhir. */
        for (int x = 0; x < W; x++) {
            float top = src[x];
            float bot = src[(size_t)(H - 1) * W + x];
            for (int k = 0; k < pad; k++) {
                dst[(size_t)k * PW + (pad + x)] = top;
                dst[(size_t)(PH - 1 - k) * PW + (pad + x)] = bot;
            }
        }
        /* Kolom kiri/kanan (baris tengah saja; sudut ditangani terpisah). */
        for (int y = 0; y < H; y++) {
            float left = src[(size_t)y * W];
            float right = src[(size_t)y * W + (W - 1)];
            for (int k = 0; k < pad; k++) {
                dst[(size_t)(y + pad) * PW + k] = left;
                dst[(size_t)(y + pad) * PW + (PW - 1 - k)] = right;
            }
        }
        /* Sudut: replikasi piksel pojok gambar ke blok pad x pad. */
        float tl = src[0], tr = src[W - 1];
        float bl = src[(size_t)(H - 1) * W], br = src[(size_t)(H - 1) * W + (W - 1)];
        for (int ky = 0; ky < pad; ky++) {
            for (int kx = 0; kx < pad; kx++) {
                dst[(size_t)ky * PW + kx] = tl;
                dst[(size_t)ky * PW + (PW - 1 - kx)] = tr;
                dst[(size_t)(PH - 1 - ky) * PW + kx] = bl;
                dst[(size_t)(PH - 1 - ky) * PW + (PW - 1 - kx)] = br;
            }
        }
    }
    *PH_out = PH;
    *PW_out = PW;
    return out;
}

/* Dot product sepanjang kw elemen kontigu (in_row · w_row). */
static inline float dot_kw(const float *in_row, const float *w_row, int kw) {
    int kx = 0;
    float acc = 0.0f;
#ifdef FSRCNN_HAVE_NEON
    if (kw >= 4) {
        float32x4_t vacc = vdupq_n_f32(0.0f);
        for (; kx + 4 <= kw; kx += 4) {
            float32x4_t vin = vld1q_f32(in_row + kx);
            float32x4_t vw  = vld1q_f32(w_row + kx);
            vacc = vmlaq_f32(vacc, vin, vw);
        }
#if defined(__aarch64__)
        acc += vaddvq_f32(vacc);
#else
        float32x2_t sum2 = vadd_f32(vget_low_f32(vacc), vget_high_f32(vacc));
        sum2 = vpadd_f32(sum2, sum2);
        acc += vget_lane_f32(sum2, 0);
#endif
    }
#endif
    for (; kx < kw; kx++) acc += in_row[kx] * w_row[kx];
    return acc;
}

void fsrcnn_conv2d(const float *in, int cin, int H, int W,
                    const float *weight, const float *bias, int cout,
                    int kh, int kw, int pad,
                    fsrcnn_activation_t act, float prelu_slope,
                    float *out) {
    int PH, PW;
    float *padded = pad_chw_replicate(in, cin, H, W, pad, &PH, &PW);
    if (!padded) return; /* alokasi gagal: keluar diam-diam, caller idealnya cek errno/OOM sebelumnya */

    for (int oc = 0; oc < cout; oc++) {
        const float *w_oc = weight + (size_t)oc * cin * kh * kw;
        float b = bias[oc];
        float *out_oc = out + (size_t)oc * H * W;
        for (int oy = 0; oy < H; oy++) {
            for (int ox = 0; ox < W; ox++) {
                float acc = b;
                for (int ic = 0; ic < cin; ic++) {
                    const float *in_c = padded + (size_t)ic * PH * PW;
                    const float *w_c = w_oc + (size_t)ic * kh * kw;
                    for (int ky = 0; ky < kh; ky++) {
                        const float *in_row = in_c + (size_t)(oy + ky) * PW + ox;
                        const float *w_row = w_c + (size_t)ky * kw;
                        acc += dot_kw(in_row, w_row, kw);
                    }
                }
                /* PReLU per-layer (BUKAN plain ReLU): v>=0 ? v : slope*v,
                 * persis Max(v,0)+slope*Min(v,0) pada source_channel_pixel.c.
                 * slope BOLEH negatif (mis. layer1 = -0.8986) -- jangan
                 * disederhanakan jadi fmaxf(acc,0.0f). */
                if (act == FSRCNN_ACT_PRELU) {
                    out_oc[(size_t)oy * W + ox] = (acc >= 0.0f) ? acc : prelu_slope * acc;
                } else {
                    out_oc[(size_t)oy * W + ox] = acc;
                }
            }
        }
    }
    free(padded);
}

void fsrcnn_deconv_output_size(int Hin, int Win, int fsize, int stride, int border,
                                int *Hout, int *Wout) {
    (void)fsize; (void)border; /* tidak memengaruhi ukuran keluaran -- lihat catatan fsrcnn_deconv_reduce() */
    *Hout = Hin * stride;
    *Wout = Win * stride;
}

/*
 * fsrcnn_deconv_reduce -- upsampling layer 8, PORT LANGSUNG dari
 * deconv_reduce_hwc() pada source_channel_pixel.c (adaptasi layout CHW,
 * bukan HWC -- lihat catatan header file). in: (cin,Hin,Win). weight:
 * layout NATIVE (cin,1,fsize,fsize) row-major (SAMA dengan yang dipakai
 * fsrcnn_weights.h utk W8, TIDAK perlu repack tap-major). out: (1,Hin*
 * stride,Win*stride), harus sudah dialokasikan pemanggil.
 */
void fsrcnn_deconv_reduce(const float *in, int cin, int Hin, int Win,
                           const float *weight, float bias,
                           int fsize, int stride, int border,
                           float *out) {
    int PH, PW;
    float *padded = pad_chw_replicate(in, cin, Hin, Win, border, &PH, &PW);
    if (!padded) return;

    int tmp_rows = PH * stride + fsize - 1;
    int tmp_cols = PW * stride + fsize - 1;
    float *tmp = (float *)calloc((size_t)tmp_rows * tmp_cols, sizeof(float));
    if (!tmp) { free(padded); return; }

    /* Fase scatter: utk tiap piksel input (SUDAH di-pad), sebarkan
     * dot-product-nya (direduksi atas SELURUH kanal cin) ke jendela
     * fsize x fsize pada kanvas akumulasi `tmp`, mulai posisi
     * (i*stride, j*stride). Ini "full transposed convolution", TANPA
     * crop apa pun di tahap ini (crop dilakukan terpisah di bawah). */
    for (int i = 0; i < PH; i++) {
        for (int j = 0; j < PW; j++) {
            int base_row = i * stride;
            int base_col = j * stride;
            for (int kr = 0; kr < fsize; kr++) {
                float *tmp_row = tmp + (size_t)(base_row + kr) * tmp_cols + base_col;
                for (int kc = 0; kc < fsize; kc++) {
                    int tap = kr * fsize + kc;
                    float dot = 0.0f;
                    for (int c = 0; c < cin; c++) {
                        float pix = padded[(size_t)c * PH * PW + (size_t)i * PW + j];
                        float w = weight[(size_t)c * fsize * fsize + tap];
                        dot += pix * w;
                    }
                    tmp_row[kc] += dot;
                }
            }
        }
    }
    free(padded);

    /* Fase crop: ekstrak jendela Hout x Wout dari kanvas, mulai offset
     * tetap (independen dari Hin/Win, hanya bergantung fsize/stride/border
     * yang semuanya konstanta arsitektural) -- lihat catatan header. */
    int Hout = Hin * stride, Wout = Win * stride;
    int offset = (fsize + 1) / 2 + stride * border - 1;
    for (int oy = 0; oy < Hout; oy++) {
        const float *tmp_row = tmp + (size_t)(oy + offset) * tmp_cols + offset;
        float *out_row = out + (size_t)oy * Wout;
        for (int ox = 0; ox < Wout; ox++) {
            out_row[ox] = tmp_row[ox] + bias;
        }
    }
    free(tmp);
}
