/*
 * yuv_diff.c -- bandingkan dua file YUV420p (keluaran fsrcnn_cpu/npu/gpu)
 * secara byte-level, TANPA dependency (murni C, tanpa python/numpy) --
 * dipakai utk verifikasi numerik LANGSUNG DI board (mis. bandingkan
 * susi_cpu.yuv vs susi_npu.yuv vs susi_gpu.yuv pada resolusi OUTPUT yang
 * SAMA, hasil salah satu dari bin/fsrcnn_{cpu,npu,gpu}_aarch64).
 *
 * Pemakaian:
 *   ./yuv_diff a.yuv b.yuv <Wout> <Hout> [--frames N] [--tol-y N]
 *
 * <Wout> <Hout> adalah dimensi PLANE Y KELUARAN (mis. 352 288 utk input
 * 176x144, scale=2) -- BUKAN dimensi input. Dimensi chroma keluaran
 * diturunkan otomatis (Wout/2 x Hout/2, konsisten dgn chroma_upsample_nn()
 * scale=2 pada plane chroma INPUT yg sudah setengah ukuran luma input).
 * --frames: jumlah frame yg dibandingkan (default: auto dari ukuran file
 * terkecil di antara kedua file).
 * --tol-y: toleransi selisih maksimum plane Y sebelum dianggap GAGAL
 * (default 1 -- wajar akibat reasosiasi floating-point antar
 * implementasi/compiler/device, lihat README proyek). Plane U/V SELALU
 * harus 0 (replikasi nearest-neighbor murni, tanpa floating point) --
 * tidak ada opsi toleransi utknya.
 *
 * Exit code: 0 kalau semua frame dalam toleransi, 1 kalau ada yang GAGAL
 * atau file tidak bisa dibuka/ukurannya tidak cocok.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long file_size(FILE *f) {
    long cur = ftell(f);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, cur, SEEK_SET);
    return sz;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Pemakaian: %s a.yuv b.yuv <Wout> <Hout> [--frames N] [--tol-y N]\n", argv[0]);
        return 1;
    }
    const char *pa = argv[1], *pb = argv[2];
    int Wout = atoi(argv[3]), Hout = atoi(argv[4]);
    long force_frames = -1;
    int tol_y = 1;
    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) force_frames = atol(argv[++i]);
        else if (strcmp(argv[i], "--tol-y") == 0 && i + 1 < argc) tol_y = atoi(argv[++i]);
    }
    if (Wout <= 0 || Hout <= 0) { fprintf(stderr, "Wout/Hout tidak valid\n"); return 1; }

    int Cw = Wout / 2, Ch = Hout / 2; /* dimensi chroma keluaran, lihat catatan header */
    size_t y_sz = (size_t)Wout * Hout;
    size_t c_sz = (size_t)Cw * Ch;
    size_t frame_sz = y_sz + 2 * c_sz;

    FILE *fa = fopen(pa, "rb");
    FILE *fb = fopen(pb, "rb");
    if (!fa) { perror(pa); return 1; }
    if (!fb) { perror(pb); fclose(fa); return 1; }

    long sa = file_size(fa), sb = file_size(fb);
    long n_avail = (sa < sb ? sa : sb) / (long)frame_sz;
    long n_frames = (force_frames >= 0) ? force_frames : n_avail;
    if (n_frames > n_avail) {
        fprintf(stderr, "PERINGATAN: --frames %ld melebihi frame tersedia (%ld), dipotong.\n", n_frames, n_avail);
        n_frames = n_avail;
    }
    if (n_frames <= 0) {
        fprintf(stderr, "GAGAL: tidak ada frame lengkap (ukuran file a=%ld byte, b=%ld byte, frame_size=%zu byte)\n",
                sa, sb, frame_sz);
        fclose(fa); fclose(fb);
        return 1;
    }

    unsigned char *bufa = (unsigned char *)malloc(frame_sz);
    unsigned char *bufb = (unsigned char *)malloc(frame_sz);
    if (!bufa || !bufb) { fprintf(stderr, "GAGAL alokasi memori\n"); return 1; }

    long overall_max_y = 0, overall_max_uv = 0;
    long total_y_over_tol = 0;
    int any_read_error = 0;

    for (long fr = 0; fr < n_frames; fr++) {
        if (fread(bufa, 1, frame_sz, fa) != frame_sz) { any_read_error = 1; break; }
        if (fread(bufb, 1, frame_sz, fb) != frame_sz) { any_read_error = 1; break; }

        int max_y = 0, max_u = 0, max_v = 0;
        for (size_t i = 0; i < y_sz; i++) {
            int d = abs((int)bufa[i] - (int)bufb[i]);
            if (d > max_y) max_y = d;
        }
        for (size_t i = 0; i < c_sz; i++) {
            int d = abs((int)bufa[y_sz + i] - (int)bufb[y_sz + i]);
            if (d > max_u) max_u = d;
        }
        for (size_t i = 0; i < c_sz; i++) {
            int d = abs((int)bufa[y_sz + c_sz + i] - (int)bufb[y_sz + c_sz + i]);
            if (d > max_v) max_v = d;
        }

        int max_uv = (max_u > max_v) ? max_u : max_v;
        if (max_y > overall_max_y) overall_max_y = max_y;
        if (max_uv > overall_max_uv) overall_max_uv = max_uv;
        if (max_y > tol_y) total_y_over_tol++;

        printf("frame %ld: maxdiff Y=%d U=%d V=%d\n", fr, max_y, max_u, max_v);
    }

    free(bufa); free(bufb);
    fclose(fa); fclose(fb);

    if (any_read_error) {
        fprintf(stderr, "GAGAL: error baca file sebelum %ld frame selesai\n", n_frames);
        return 1;
    }

    printf("\n=== Ringkasan (%ld frame, Wout=%d Hout=%d) ===\n", n_frames, Wout, Hout);
    printf("Selisih maksimum keseluruhan: Y=%ld (toleransi %d)  U/V=%ld (toleransi 0)\n",
           overall_max_y, tol_y, overall_max_uv);

    if (overall_max_y > tol_y || overall_max_uv > 0 || total_y_over_tol > 0) {
        printf("GAGAL: ada selisih melebihi toleransi.\n");
        return 1;
    }
    printf("OK: dalam toleransi.\n");
    return 0;
}
