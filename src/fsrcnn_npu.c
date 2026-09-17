/*
 * fsrcnn_npu.c -- implementasi fsrcnn_npu_* memakai RKNN C API resmi.
 *
 * KOREKSI PENTING (ditemukan saat uji pertama di board fisik Orange Pi 5):
 * `fsrcnn_rk3588.rknn` dibangun sebagai model SHAPE TETAP/STATIS
 * (scripts/convert_to_rknn.py TIDAK lagi memakai fitur eksperimental
 * `dynamic_input` -- lihat catatan koreksi Fase 3 proyek ini, U/V sudah
 * tidak lewat NPU sama sekali sehingga satu shape tetap sudah cukup).
 * Konsekuensinya: `rknn_set_input_shapes()` (API utk memilih salah satu dari
 * BEBERAPA shape terdaftar pada model dinamis) TIDAK BOLEH dipanggil pada
 * model statis -- runtime RKNN akan menolak dengan error eksplisit ("rknn
 * model is static shape type, please export rknn with dynamic_shapes").
 * Versi sebelumnya modul ini memanggilnya secara tak bersyarat pada
 * panggilan pertama (last_h/last_w = -1 awal), yang GAGAL di board fisik.
 *
 * KOREKSI PENTING #2 (ditemukan segera setelah #1, uji board fisik yg sama):
 * `rknn_query(RKNN_QUERY_INPUT_ATTR)` pada `fsrcnn_rk3588.rknn` melaporkan
 * `fmt=RKNN_TENSOR_NHWC` (=1) -- BUKAN NCHW seperti asumsi awal (meski graph
 * ONNX kita deklarasikan NCHW; kompiler RKNN RK3588 mengonversi ke layout
 * native NHWC utk compute NPU). Versi sebelumnya modul ini hardcode
 * `rk_in.fmt = RKNN_TENSOR_NCHW` tak peduli fmt asli model -- mismatch ini
 * memicu `rknn_inputs_set()` gagal di internal step "normalize" ("Meet
 * unsupported src layout for normalize"). Yang lebih berbahaya: pada uji di
 * board, kegagalan itu TIDAK selalu membuat `rknn_inputs_set()` mengembalikan
 * kode error != RKNN_SUCC (driver hanya mencetak log "E RKNN" tapi lanjut) --
 * artinya frame bisa "sukses" secara API padahal isi buffer normalize-nya
 * tidak terdefinisi/salah. Diperbaiki: fmt yang dipakai di `rknn_inputs_set()`
 * SEKARANG SELALU disamakan dgn fmt native hasil query di init (`ctx->in_fmt`),
 * bukan hardcode NCHW. Karena model ini SATU channel (C=1), NCHW (1,1,H,W)
 * dan NHWC (1,H,W,1) serialize ke byte yang PERSIS SAMA -- jadi ini murni
 * perbaikan metadata (memberi tahu driver formatnya dgn benar), bukan
 * perubahan tata-letak data di memori.
 *
 * Alur yang BENAR per panggilan fsrcnn_npu_forward():
 *   1. (hanya sekali, di fsrcnn_npu_init()) rknn_query(RKNN_QUERY_INPUT_ATTR)
 *      -- tanya runtime shape H x W DAN fmt native yang SUDAH DI-COMPILE ke
 *      dalam model (bukan "set", murni baca). Disimpan di ctx.
 *   2. fsrcnn_npu_forward() memvalidasi in->height/in->width TEPAT SAMA
 *      dengan shape hasil query #1 -- kalau tidak, gagal cepat dgn pesan
 *      jelas (dari pada gagal aneh di rknn_inputs_set/rknn_run).
 *   3. rknn_inputs_set()  -- salin data plane (float32, fmt = ctx->in_fmt
 *      hasil query, BUKAN hardcode) ke NPU.
 *   4. rknn_run()          -- jalankan inference di NPU (asynchronous
 *      terhadap CPU sampai rknn_outputs_get() dipanggil, yang blocking).
 *   5. rknn_outputs_get()  -- ambil hasil (want_float=1 supaya driver
 *      men-dekonversi dari layout native NPU/FP16 kembali ke float32).
 *   6. rknn_outputs_release().
 */
#include "fsrcnn_npu.h"
#include "rknn_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct fsrcnn_npu_ctx {
    rknn_context ctx;
    int in_h, in_w;             /* shape TETAP yang di-compile ke dalam model (hasil query saat init) */
    rknn_tensor_format in_fmt;  /* fmt NATIVE model (NCHW/NHWC), hasil query saat init -- SELALU
                                  * dipakai apa adanya di rknn_inputs_set(), jangan hardcode. */
};

static void print_rknn_error(const char *where, int ret) {
    fprintf(stderr, "[npu] %s gagal, kode error RKNN = %d\n", where, ret);
}

fsrcnn_npu_ctx_t *fsrcnn_npu_init(const char *rknn_path) {
    fsrcnn_npu_ctx_t *c = (fsrcnn_npu_ctx_t *)calloc(1, sizeof(*c));
    if (!c) return NULL;

    /* size=0 -> parameter model diinterpretasikan sebagai path file (lihat
     * dokumentasi rknn_init di rknn_api.h). */
    int ret = rknn_init(&c->ctx, (void *)rknn_path, 0, 0, NULL);
    if (ret != RKNN_SUCC) {
        print_rknn_error("rknn_init", ret);
        free(c);
        return NULL;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(c->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        print_rknn_error("rknn_query(IN_OUT_NUM)", ret);
        rknn_destroy(c->ctx);
        free(c);
        return NULL;
    }
    if (io_num.n_input != 1 || io_num.n_output != 1) {
        fprintf(stderr, "[npu] model tidak sesuai ekspektasi: n_input=%u n_output=%u (harap 1/1)\n",
                io_num.n_input, io_num.n_output);
        rknn_destroy(c->ctx);
        free(c);
        return NULL;
    }

    /* Query (BUKAN set) shape input yang sudah TETAP di-compile ke model --
     * dipakai HANYA utk validasi di fsrcnn_npu_forward(), tidak pernah
     * dipakai utk memilih/mengganti shape (model statis, cuma ada 1). */
    rknn_tensor_attr in_attr;
    memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    ret = rknn_query(c->ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    if (ret != RKNN_SUCC) {
        print_rknn_error("rknn_query(INPUT_ATTR)", ret);
        rknn_destroy(c->ctx);
        free(c);
        return NULL;
    }
    if (in_attr.n_dims != 4) {
        fprintf(stderr, "[npu] input model bukan 4D (n_dims=%u), model tidak sesuai ekspektasi\n", in_attr.n_dims);
        rknn_destroy(c->ctx);
        free(c);
        return NULL;
    }
    /* Layout dims tergantung fmt yang dilaporkan driver: NCHW -> [N,C,H,W],
     * NHWC -> [N,H,W,C]. Kita build model dgn NCHW (lihat build_fsrcnn_onnx.py),
     * tapi driver RK3588 kadang melaporkan native NHWC secara internal. */
    if (in_attr.fmt == RKNN_TENSOR_NHWC) {
        c->in_h = (int)in_attr.dims[1];
        c->in_w = (int)in_attr.dims[2];
    } else { /* RKNN_TENSOR_NCHW (default) */
        c->in_h = (int)in_attr.dims[2];
        c->in_w = (int)in_attr.dims[3];
    }
    c->in_fmt = in_attr.fmt; /* dipakai apa adanya di rknn_inputs_set(), lihat koreksi #2 di atas */
    fprintf(stderr, "[npu] model shape tetap terbaca: %dx%d (fmt=%s)\n",
            c->in_h, c->in_w, get_format_string(in_attr.fmt));

    return c;
}

int fsrcnn_npu_forward(fsrcnn_npu_ctx_t *ctx, const fsrcnn_tensor_t *in, fsrcnn_tensor_t *out) {
    if (!ctx || in->channels != 1) return -1;
    const int H = in->height, W = in->width;

    if (H != ctx->in_h || W != ctx->in_w) {
        fprintf(stderr, "[npu] shape input %dx%d TIDAK COCOK dgn shape tetap model (%dx%d) -- "
                        "bangun ulang fsrcnn_rk3588.rknn dgn scripts/convert_to_rknn.py "
                        "--luma-h %d --luma-w %d, atau jalankan program ini dgn <W> <H> yg sesuai model\n",
                        H, W, ctx->in_h, ctx->in_w, H, W);
        return -3;
    }

    int Hout, Wout;
    fsrcnn_deconv_output_size(H, W, 9, 2, 1, &Hout, &Wout);
    if (out->data == NULL || out->channels != 1 || out->height != Hout || out->width != Wout) {
        return -2;
    }

    rknn_input rk_in;
    memset(&rk_in, 0, sizeof(rk_in));
    rk_in.index = 0;
    rk_in.buf = in->data;
    rk_in.size = (uint32_t)((size_t)H * W * sizeof(float));
    rk_in.pass_through = 0;
    rk_in.type = RKNN_TENSOR_FLOAT32;
    rk_in.fmt = ctx->in_fmt; /* SELALU fmt native model (query saat init) -- lihat koreksi #2 */

    int ret = rknn_inputs_set(ctx->ctx, 1, &rk_in);
    if (ret != RKNN_SUCC) { print_rknn_error("rknn_inputs_set", ret); return -4; }

    ret = rknn_run(ctx->ctx, NULL);
    if (ret != RKNN_SUCC) { print_rknn_error("rknn_run", ret); return -5; }

    rknn_output rk_out;
    memset(&rk_out, 0, sizeof(rk_out));
    rk_out.index = 0;
    rk_out.want_float = 1;
    rk_out.is_prealloc = 0;

    ret = rknn_outputs_get(ctx->ctx, 1, &rk_out, NULL);
    if (ret != RKNN_SUCC) { print_rknn_error("rknn_outputs_get", ret); return -6; }

    size_t expected_bytes = (size_t)Hout * Wout * sizeof(float);
    if (rk_out.size != expected_bytes) {
        fprintf(stderr, "[npu] ukuran output tak terduga: dapat %u byte, diharapkan %zu byte\n",
                rk_out.size, expected_bytes);
        rknn_outputs_release(ctx->ctx, 1, &rk_out);
        return -7;
    }
    memcpy(out->data, rk_out.buf, expected_bytes);

    rknn_outputs_release(ctx->ctx, 1, &rk_out);
    return 0;
}

void fsrcnn_npu_destroy(fsrcnn_npu_ctx_t *ctx) {
    if (!ctx) return;
    rknn_destroy(ctx->ctx);
    free(ctx);
}
