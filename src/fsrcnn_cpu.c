/*
 * fsrcnn_cpu.c -- forward pass FSRCNN (8 layer) di CPU, memakai primitif
 * dari fsrcnn_ops.c dan bobot ter-embed dari fsrcnn_weights.h.
 *
 * Arsitektur & aktivasi -- KOREKSI PENTING (mengikuti referensi otoritatif
 * source_channel_pixel.c yang dipakai untuk melatih bobot ini): aktivasinya
 * adalah PReLU PER-LAYER (satu skalar slope per layer, termasuk BOLEH
 * negatif), BUKAN plain ReLU seperti pada iterasi awal proyek ini.
 *
 *   x1 = PReLU(Conv5x5  (in, 1->D), slope=PRELU1)   feature extraction
 *   x2 = PReLU(Conv1x1  (x1, D->S), slope=PRELU2)   shrinking
 *   x3 = PReLU(Conv3x3  (x2, S->S), slope=PRELU3)   mapping 1/4
 *   x4 = PReLU(Conv3x3  (x3, S->S), slope=PRELU4)   mapping 2/4
 *   x5 = PReLU(Conv3x3  (x4, S->S), slope=PRELU5)   mapping 3/4
 *   x6 = PReLU(Conv3x3  (x5, S->S), slope=PRELU6)   mapping 4/4
 *   x7 = PReLU(Conv1x1  (x6, S->D), slope=PRELU7)   expanding
 *   out =      Deconv9x9(x7, stride=2, D->1)        upsampling (TANPA aktivasi)
 *
 * Fungsi ini HANYA valid untuk plane Y (luma) -- lihat catatan di fsrcnn.h.
 */
#include "fsrcnn.h"
#include "fsrcnn_weights.h"

#include <stddef.h>

int fsrcnn_forward(const fsrcnn_tensor_t *in, fsrcnn_tensor_t *out) {
    if (in->channels != 1 || in->data == NULL) return -1;
    const int H = in->height, W = in->width;

    int Hout, Wout;
    fsrcnn_deconv_output_size(H, W, 9, FSRCNN_SCALE, 1, &Hout, &Wout);
    if (out->data == NULL || out->channels != 1 || out->height != Hout || out->width != Wout) {
        return -2; /* pemanggil wajib alokasi out dgn fsrcnn_deconv_output_size() lebih dulu */
    }

    fsrcnn_tensor_t x1 = {0}, x2 = {0}, x3 = {0}, x4 = {0}, x5 = {0}, x6 = {0}, x7 = {0};
    int rc = 0;

    if (fsrcnn_tensor_alloc(&x1, FSRCNN_D, H, W)) { rc = -3; goto cleanup; }
    fsrcnn_conv2d(in->data, 1, H, W, W1, B1, FSRCNN_D, 5, 5, 2, FSRCNN_ACT_PRELU, FSRCNN_PRELU1, x1.data);

    if (fsrcnn_tensor_alloc(&x2, FSRCNN_S, H, W)) { rc = -3; goto cleanup; }
    fsrcnn_conv2d(x1.data, FSRCNN_D, H, W, W2, B2, FSRCNN_S, 1, 1, 0, FSRCNN_ACT_PRELU, FSRCNN_PRELU2, x2.data);

    if (fsrcnn_tensor_alloc(&x3, FSRCNN_S, H, W)) { rc = -3; goto cleanup; }
    fsrcnn_conv2d(x2.data, FSRCNN_S, H, W, W3, B3, FSRCNN_S, 3, 3, 1, FSRCNN_ACT_PRELU, FSRCNN_PRELU3, x3.data);

    if (fsrcnn_tensor_alloc(&x4, FSRCNN_S, H, W)) { rc = -3; goto cleanup; }
    fsrcnn_conv2d(x3.data, FSRCNN_S, H, W, W4, B4, FSRCNN_S, 3, 3, 1, FSRCNN_ACT_PRELU, FSRCNN_PRELU4, x4.data);

    if (fsrcnn_tensor_alloc(&x5, FSRCNN_S, H, W)) { rc = -3; goto cleanup; }
    fsrcnn_conv2d(x4.data, FSRCNN_S, H, W, W5, B5, FSRCNN_S, 3, 3, 1, FSRCNN_ACT_PRELU, FSRCNN_PRELU5, x5.data);

    if (fsrcnn_tensor_alloc(&x6, FSRCNN_S, H, W)) { rc = -3; goto cleanup; }
    fsrcnn_conv2d(x5.data, FSRCNN_S, H, W, W6, B6, FSRCNN_S, 3, 3, 1, FSRCNN_ACT_PRELU, FSRCNN_PRELU6, x6.data);

    if (fsrcnn_tensor_alloc(&x7, FSRCNN_D, H, W)) { rc = -3; goto cleanup; }
    fsrcnn_conv2d(x6.data, FSRCNN_S, H, W, W7, B7, FSRCNN_D, 1, 1, 0, FSRCNN_ACT_PRELU, FSRCNN_PRELU7, x7.data);

    fsrcnn_deconv_reduce(x7.data, FSRCNN_D, H, W, W8, B8[0], 9, FSRCNN_SCALE, 1, out->data);

cleanup:
    fsrcnn_tensor_free(&x1);
    fsrcnn_tensor_free(&x2);
    fsrcnn_tensor_free(&x3);
    fsrcnn_tensor_free(&x4);
    fsrcnn_tensor_free(&x5);
    fsrcnn_tensor_free(&x6);
    fsrcnn_tensor_free(&x7);
    return rc;
}
