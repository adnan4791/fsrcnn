/*
 * main_npu.c -- driver CLI: super-resolusi video YUV 4:2:0 mentah memakai
 * FSRCNN, dijalankan pada NPU RK3588 lewat RKNN C API.
 *
 * Pemakaian:
 *   ./fsrcnn_npu <model.rknn> <in.yuv> <width> <height> <out.yuv> [--max-frames N]
 *
 * width/height DI SINI ADALAH DIMENSI PLANE Y (luma) video sumber, dan
 * HARUS PERSIS SAMA dengan --luma-h/--luma-w yang dipakai saat
 * scripts/convert_to_rknn.py dijalankan -- karena NPU RKNN (berbeda dari
 * CPU) meng-compile graph untuk himpunan shape yang ditentukan di waktu
 * konversi, bukan benar-benar dinamis sembarang ukuran.
 *
 * KOREKSI channel U/V (mengikuti referensi otoritatif
 * source_channel_pixel.c): model .rknn HANYA dipakai utk plane Y. Plane
 * U/V di-upscale via replikasi nearest-neighbor (chroma_upsample_nn(), di
 * CPU, TANPA menyentuh NPU sama sekali) -- konsisten dgn main_cpu.c.
 */
#define _POSIX_C_SOURCE 199309L

#include "fsrcnn.h"
#include "fsrcnn_npu.h"
#include "yuv_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int sr_plane_y_npu(fsrcnn_npu_ctx_t *ctx, const fsrcnn_tensor_t *in, fsrcnn_tensor_t *out) {
    int Hout, Wout;
    fsrcnn_deconv_output_size(in->height, in->width, 9, 2, 1, &Hout, &Wout);
    if (fsrcnn_tensor_alloc(out, 1, Hout, Wout) != 0) return -1;
    return fsrcnn_npu_forward(ctx, in, out);
}

static int upscale_plane_chroma(const fsrcnn_tensor_t *in, fsrcnn_tensor_t *out, int scale) {
    if (fsrcnn_tensor_alloc(out, 1, in->height * scale, in->width * scale) != 0) return -1;
    chroma_upsample_nn(in, out, scale);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr,
            "Pemakaian: %s <model.rknn> <in.yuv> <width> <height> <out.yuv> [--max-frames N]\n",
            argv[0]);
        return 1;
    }
    const char *rknn_path = argv[1];
    const char *in_path = argv[2];
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);
    const char *out_path = argv[5];
    long max_frames = -1;

    for (int i = 6; i < argc; i++) {
        if (strcmp(argv[i], "--max-frames") == 0 && i + 1 < argc) {
            max_frames = atol(argv[++i]);
        }
    }
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "width/height tidak valid\n");
        return 1;
    }

    fsrcnn_npu_ctx_t *npu = fsrcnn_npu_init(rknn_path);
    if (!npu) {
        fprintf(stderr, "[error] gagal inisialisasi NPU/model %s\n", rknn_path);
        return 1;
    }

    FILE *fin = fopen(in_path, "rb");
    if (!fin) { perror("fopen input"); fsrcnn_npu_destroy(npu); return 1; }
    FILE *fout = fopen(out_path, "wb");
    if (!fout) { perror("fopen output"); fclose(fin); fsrcnn_npu_destroy(npu); return 1; }

    yuv420_dims_t dims = { .width = width, .height = height };
    int ch, cw;
    yuv420_chroma_dims(height, width, &ch, &cw);

    fsrcnn_tensor_t y_in = {0}, u_in = {0}, v_in = {0};
    fsrcnn_tensor_t y_out = {0}, u_out = {0}, v_out = {0};
    fsrcnn_tensor_alloc(&y_in, 1, height, width);
    fsrcnn_tensor_alloc(&u_in, 1, ch, cw);
    fsrcnn_tensor_alloc(&v_in, 1, ch, cw);

    long frame_idx = 0;
    double t_total = 0.0;

    for (;;) {
        if (max_frames >= 0 && frame_idx >= max_frames) break;
        int rc = yuv420_read_frame(fin, &dims, &y_in, &u_in, &v_in);
        if (rc == 1) break;
        if (rc != 0) { fprintf(stderr, "[error] gagal baca frame %ld\n", frame_idx); break; }

        double t0 = now_sec();
        /* Hanya Y yang lewat NPU; shape yg dikirim ke rknn_set_input_shapes()
         * jadi selalu SAMA antar-panggilan (tidak perlu lagi berpindah
         * shape Y<->chroma seperti versi sebelumnya). U/V murni di CPU. */
        if (sr_plane_y_npu(npu, &y_in, &y_out) != 0) { fprintf(stderr, "[error] SR plane Y (NPU) gagal\n"); break; }
        if (upscale_plane_chroma(&u_in, &u_out, 2) != 0) { fprintf(stderr, "[error] upscale plane U gagal\n"); break; }
        if (upscale_plane_chroma(&v_in, &v_out, 2) != 0) { fprintf(stderr, "[error] upscale plane V gagal\n"); break; }
        double dt = now_sec() - t0;
        t_total += dt;

        yuv420_write_frame(fout, &y_out, &u_out, &v_out);

        fprintf(stderr, "[frame %ld] %d x %d -> %d x %d | %.4f s (%.2f fps sesaat)\n",
                frame_idx, width, height, y_out.width, y_out.height,
                dt, dt > 0 ? 1.0 / dt : 0.0);

        fsrcnn_tensor_free(&y_out);
        fsrcnn_tensor_free(&u_out);
        fsrcnn_tensor_free(&v_out);
        frame_idx++;
    }

    if (frame_idx > 0) {
        fprintf(stderr, "\n[ringkasan NPU] %ld frame, total %.4f s, rata-rata %.4f s/frame (%.2f fps)\n",
                frame_idx, t_total, t_total / frame_idx, frame_idx / t_total);
    } else {
        fprintf(stderr, "[warn] tidak ada frame yang diproses\n");
    }

    fsrcnn_tensor_free(&y_in);
    fsrcnn_tensor_free(&u_in);
    fsrcnn_tensor_free(&v_in);
    fclose(fin);
    fclose(fout);
    fsrcnn_npu_destroy(npu);
    return 0;
}
