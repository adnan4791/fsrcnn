/*
 * main_cpu.c -- driver CLI: super-resolusi video YUV 4:2:0 mentah memakai
 * FSRCNN, dijalankan murni di CPU (referensi / fallback tanpa NPU).
 *
 * Pemakaian:
 *   ./fsrcnn_cpu <in.yuv> <width> <height> <out.yuv> [--pin-big] [--max-frames N]
 *
 * KOREKSI channel U/V (mengikuti referensi otoritatif
 * source_channel_pixel.c, yang jadi acuan pelatihan bobot ini): FSRCNN
 * HANYA dijalankan pada plane Y (luma). Plane U/V di-upscale dengan
 * replikasi nearest-neighbor blok 2x2 sederhana (chroma_upsample_nn(),
 * TANPA neural network) -- persis meniru blok "U/V Component" pada main()
 * referensi tsb. Iterasi awal proyek ini SEMPAT menjalankan FSRCNN pada
 * ketiga plane; itu SALAH, sudah dikoreksi di sini.
 */
#define _POSIX_C_SOURCE 199309L

#include "fsrcnn.h"
#include "yuv_io.h"
#include "cpu_affinity.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* SR plane Y lewat FSRCNN (satu-satunya plane yg model ini dilatih untuknya). */
static int sr_plane_y(const fsrcnn_tensor_t *in, fsrcnn_tensor_t *out) {
    int Hout, Wout;
    fsrcnn_deconv_output_size(in->height, in->width, 9, 2, 1, &Hout, &Wout);
    if (fsrcnn_tensor_alloc(out, 1, Hout, Wout) != 0) return -1;
    return fsrcnn_forward(in, out);
}

/* Upscale plane chroma (U/V) via replikasi nearest-neighbor, BUKAN FSRCNN. */
static int upscale_plane_chroma(const fsrcnn_tensor_t *in, fsrcnn_tensor_t *out, int scale) {
    if (fsrcnn_tensor_alloc(out, 1, in->height * scale, in->width * scale) != 0) return -1;
    chroma_upsample_nn(in, out, scale);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
            "Pemakaian: %s <in.yuv> <width> <height> <out.yuv> [--pin-big] [--max-frames N]\n",
            argv[0]);
        return 1;
    }
    const char *in_path = argv[1];
    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    const char *out_path = argv[4];
    int pin_big = 0;
    long max_frames = -1;

    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "--pin-big") == 0) {
            pin_big = 1;
        } else if (strcmp(argv[i], "--max-frames") == 0 && i + 1 < argc) {
            max_frames = atol(argv[++i]);
        }
    }

    if (width <= 0 || height <= 0) {
        fprintf(stderr, "width/height tidak valid\n");
        return 1;
    }

    if (pin_big) {
        int n = cpu_affinity_pin_to_big_cores();
        if (n > 0) {
            fprintf(stderr, "[info] dipin ke %d core \"big\" (frekuensi tertinggi)\n", n);
        } else {
            fprintf(stderr, "[warn] deteksi/pinning core big gagal (bukan Linux ARM? berjalan tanpa pinning)\n");
        }
    }

    FILE *fin = fopen(in_path, "rb");
    if (!fin) { perror("fopen input"); return 1; }
    FILE *fout = fopen(out_path, "wb");
    if (!fout) { perror("fopen output"); fclose(fin); return 1; }

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
        if (rc == 1) break;       /* EOF wajar */
        if (rc != 0) { fprintf(stderr, "[error] gagal baca frame %ld (file terpotong?)\n", frame_idx); break; }

        double t0 = now_sec();
        if (sr_plane_y(&y_in, &y_out) != 0) { fprintf(stderr, "[error] SR plane Y (FSRCNN) gagal\n"); break; }
        if (upscale_plane_chroma(&u_in, &u_out, 2) != 0) { fprintf(stderr, "[error] upscale plane U gagal\n"); break; }
        if (upscale_plane_chroma(&v_in, &v_out, 2) != 0) { fprintf(stderr, "[error] upscale plane V gagal\n"); break; }
        double dt = now_sec() - t0;
        t_total += dt;

        yuv420_write_frame(fout, &y_out, &u_out, &v_out);

        fprintf(stderr, "[frame %ld] %.1f x %.1f -> %d x %d | %.3f s (%.2f fps sesaat)\n",
                frame_idx, (double)width, (double)height, y_out.width, y_out.height,
                dt, dt > 0 ? 1.0 / dt : 0.0);

        fsrcnn_tensor_free(&y_out);
        fsrcnn_tensor_free(&u_out);
        fsrcnn_tensor_free(&v_out);
        frame_idx++;
    }

    if (frame_idx > 0) {
        fprintf(stderr, "\n[ringkasan] %ld frame, total %.3f s, rata-rata %.3f s/frame (%.2f fps)\n",
                frame_idx, t_total, t_total / frame_idx, frame_idx / t_total);
    } else {
        fprintf(stderr, "[warn] tidak ada frame yang diproses (file kosong/dimensi salah?)\n");
    }

    fsrcnn_tensor_free(&y_in);
    fsrcnn_tensor_free(&u_in);
    fsrcnn_tensor_free(&v_in);
    fclose(fin);
    fclose(fout);
    return 0;
}
