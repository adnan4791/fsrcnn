/*
 * main_gpu.c -- driver CLI: super-resolusi video YUV 4:2:0 mentah memakai
 * FSRCNN, dijalankan pada GPU (Mali-G610 RK3588/Orange Pi 5) lewat OpenCL.
 *
 * Pemakaian:
 *   ./fsrcnn_gpu <in.yuv> <width> <height> <out.yuv> [--max-frames N]
 *
 * Sama seperti main_cpu.c: FULLY-CONVOLUTIONAL, width/height boleh berapa
 * pun (TIDAK seperti main_npu.c yg terikat shape tetap hasil convert_to_rknn.py).
 *
 * Channel: hanya Y (luma) lewat FSRCNN/GPU; U/V via replikasi nearest-
 * neighbor di CPU (chroma_upsample_nn(), TANPA menyentuh GPU) -- identik
 * konvensi main_cpu.c/main_npu.c, mengikuti referensi otoritatif
 * source_channel_pixel.c.
 */
#define _POSIX_C_SOURCE 199309L

#include "fsrcnn.h"
#include "fsrcnn_gpu.h"
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

static int sr_plane_y_gpu(fsrcnn_gpu_ctx_t *ctx, const fsrcnn_tensor_t *in, fsrcnn_tensor_t *out) {
    int Hout, Wout;
    fsrcnn_deconv_output_size(in->height, in->width, 9, 2, 1, &Hout, &Wout);
    if (fsrcnn_tensor_alloc(out, 1, Hout, Wout) != 0) return -1;
    return fsrcnn_gpu_forward(ctx, in, out);
}

static int upscale_plane_chroma(const fsrcnn_tensor_t *in, fsrcnn_tensor_t *out, int scale) {
    if (fsrcnn_tensor_alloc(out, 1, in->height * scale, in->width * scale) != 0) return -1;
    chroma_upsample_nn(in, out, scale);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
            "Pemakaian: %s <in.yuv> <width> <height> <out.yuv> [--max-frames N]\n",
            argv[0]);
        return 1;
    }
    const char *in_path = argv[1];
    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    const char *out_path = argv[4];
    long max_frames = -1;

    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "--max-frames") == 0 && i + 1 < argc) {
            max_frames = atol(argv[++i]);
        }
    }
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "width/height tidak valid\n");
        return 1;
    }

    fsrcnn_gpu_ctx_t *gpu = fsrcnn_gpu_init();
    if (!gpu) {
        fprintf(stderr, "[error] gagal inisialisasi OpenCL/GPU\n");
        return 1;
    }

    FILE *fin = fopen(in_path, "rb");
    if (!fin) { perror("fopen input"); fsrcnn_gpu_destroy(gpu); return 1; }
    FILE *fout = fopen(out_path, "wb");
    if (!fout) { perror("fopen output"); fclose(fin); fsrcnn_gpu_destroy(gpu); return 1; }

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
        if (sr_plane_y_gpu(gpu, &y_in, &y_out) != 0) { fprintf(stderr, "[error] SR plane Y (GPU) gagal\n"); break; }
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
        fprintf(stderr, "\n[ringkasan GPU] %ld frame, total %.4f s, rata-rata %.4f s/frame (%.2f fps)\n",
                frame_idx, t_total, t_total / frame_idx, frame_idx / t_total);
    } else {
        fprintf(stderr, "[warn] tidak ada frame yang diproses\n");
    }

    fsrcnn_tensor_free(&y_in);
    fsrcnn_tensor_free(&u_in);
    fsrcnn_tensor_free(&v_in);
    fclose(fin);
    fclose(fout);
    fsrcnn_gpu_destroy(gpu);
    return 0;
}
