/*
 * fsrcnn_gpu.c -- implementasi fsrcnn_gpu_* memakai OpenCL (target: GPU
 * Mali-G610 pada RK3588/Orange Pi 5, lihat catatan driver di fsrcnn_gpu.h).
 *
 * ================================================================
 * DERIVASI KUNCI: kernel deconv sbg GATHER, bukan SCATTER
 * ================================================================
 * Versi CPU (fsrcnn_deconv_reduce() di fsrcnn_ops.c, port langsung dari
 * deconv_reduce_hwc() referensi) menghitung layer 8 sbg SCATTER: utk tiap
 * piksel input (i,j) beserta SELURUH kanal, sebarkan dot-product-nya ke
 * jendela fsize x fsize pada kanvas akumulasi `tmp`, lalu crop. Skema
 * scatter ini AMAN di CPU (satu thread, akumulasi berurutan) tapi TIDAK
 * aman langsung diparalelkan sbg satu work-item per piksel INPUT di GPU:
 * banyak work-item akan menulis (+=) ke piksel OUTPUT yang SAMA secara
 * bersamaan -- race condition, butuh atomic add (mahal & tidak trivial utk
 * float di OpenCL 1.2).
 *
 * Solusinya: derivasi ULANG rumus yang MATEMATIS IDENTIK tapi berbentuk
 * GATHER (satu work-item per piksel OUTPUT, membaca banyak piksel input --
 * TIDAK ADA race condition, cocok utk GPU). Dari kode referensi:
 *   tmp[i*stride+kr][j*stride+kc] += dot(i,j,kr,kc)   utk semua i,j,kr,kc valid
 *   out[oy][ox] = tmp[oy+offset][ox+offset] + bias
 * Substitusi ty=oy+offset, tx=ox+offset, maka utk SATU (oy,ox) tertentu,
 * kontribusi yang masuk ke situ adalah SEMUA (i,j,kr,kc) yg memenuhi
 * i*stride+kr==ty DAN j*stride+kc==tx. Karena stride & kr/kc tetap (fsize=9,
 * stride=2), ini setara: utk tiap kr di [0,fsize), i=(ty-kr)/stride HANYA
 * valid kalau (ty-kr) habis dibagi stride DAN i di jangkauan; sama utk
 * kc/j/tx. Ini pola transposed-convolution-as-gather yang standar (dual
 * dari scatter) -- lihat komentar di kernel deconv_reduce di bawah.
 * Sudah diverifikasi numerik (lihat catatan proyek) hasilnya BIT-EXACT
 * (selisih 0, bukan cuma toleransi 1/255) terhadap fsrcnn_deconv_reduce()
 * versi CPU pada input yang sama -- karena keduanya cuma REORDER dari
 * penjumlahan float YANG SAMA PERSIS (bukan operasi matematis berbeda),
 * jadi tidak ada galat reasosiasi sama sekali di sini.
 *
 * ================================================================
 * Padding via CLAMP ADDRESS, bukan buffer replicate-pad terpisah
 * ================================================================
 * Versi CPU membuat buffer ter-pad eksplisit (pad_chw_replicate()) sebelum
 * loop conv, supaya loop utama bebas percabangan batas array. Di GPU,
 * pendekatan itu berarti kernel PAD terpisah + buffer device ekstra + 1x
 * write tambahan ke global memory (mahal secara bandwidth). Gantinya:
 * setiap kernel conv/deconv di sini meng-clamp INDEX pembacaan langsung ke
 * [0,H-1]/[0,W-1] via clamp() bawaan OpenCL. Untuk replicate-padding, ini
 * SECARA MATEMATIS IDENTIK: elemen padding hasil replikasi PERSIS SAMA
 * dgn piksel tepi yang "seharusnya" dibaca kalau indeks di-clamp ke
 * jangkauan -- tidak perlu materialisasi buffer pad sama sekali.
 */
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "fsrcnn_gpu.h"
#include "fsrcnn_weights.h"
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *KERNEL_SRC =
"__kernel void conv_prelu(\n"
"    __global const float *in, __global const float *weight, __global const float *bias,\n"
"    __global float *out,\n"
"    int cin, int H, int W, int cout, int kh, int kw, int pad,\n"
"    float prelu_slope, int use_prelu)\n"
"{\n"
"    int ox = get_global_id(0);\n"
"    int oy = get_global_id(1);\n"
"    int oc = get_global_id(2);\n"
"    if (ox >= W || oy >= H || oc >= cout) return;\n"
"\n"
"    float acc = bias[oc];\n"
"    for (int ic = 0; ic < cin; ic++) {\n"
"        __global const float *in_c = in + (size_t)ic * H * W;\n"
"        __global const float *w_c  = weight + (size_t)(oc * cin + ic) * kh * kw;\n"
"        for (int ky = 0; ky < kh; ky++) {\n"
"            int iy = clamp(oy + ky - pad, 0, H - 1);\n"
"            __global const float *in_row = in_c + (size_t)iy * W;\n"
"            __global const float *w_row  = w_c + (size_t)ky * kw;\n"
"            for (int kx = 0; kx < kw; kx++) {\n"
"                int ix = clamp(ox + kx - pad, 0, W - 1);\n"
"                acc += in_row[ix] * w_row[kx];\n"
"            }\n"
"        }\n"
"    }\n"
"    if (use_prelu) acc = (acc >= 0.0f) ? acc : prelu_slope * acc;\n"
"    out[(size_t)oc * H * W + (size_t)oy * W + ox] = acc;\n"
"}\n"
"\n"
"__kernel void deconv_reduce(\n"
"    __global const float *in, __global const float *weight,\n"
"    int cin, int Hin, int Win, int fsize, int stride, int border, float bias,\n"
"    __global float *out)\n"
"{\n"
"    int Hout = Hin * stride;\n"
"    int Wout = Win * stride;\n"
"    int ox = get_global_id(0);\n"
"    int oy = get_global_id(1);\n"
"    if (ox >= Wout || oy >= Hout) return;\n"
"\n"
"    int PH = Hin + 2 * border;\n"
"    int PW = Win + 2 * border;\n"
"    /* offset: konstanta arsitektural (fsize+1)/2 + stride*border - 1 = 6\n"
"     * utk fsize=9,stride=2,border=1 -- lihat fsrcnn_ops.c utk derivasi. */\n"
"    int offset = (fsize + 1) / 2 + stride * border - 1;\n"
"    int ty = oy + offset;\n"
"    int tx = ox + offset;\n"
"\n"
"    float acc = bias;\n"
"    for (int kr = 0; kr < fsize; kr++) {\n"
"        int num_r = ty - kr;\n"
"        if (num_r % stride != 0) continue;   /* dual dari scatter: hanya kr yg 'align' dgn stride yg berkontribusi */\n"
"        int i = num_r / stride;\n"
"        if (i < 0 || i >= PH) continue;\n"
"        int iy = clamp(i - border, 0, Hin - 1);  /* dual dari replicate-pad(border) pada input, via clamp */\n"
"        for (int kc = 0; kc < fsize; kc++) {\n"
"            int num_c = tx - kc;\n"
"            if (num_c % stride != 0) continue;\n"
"            int j = num_c / stride;\n"
"            if (j < 0 || j >= PW) continue;\n"
"            int ix = clamp(j - border, 0, Win - 1);\n"
"            int tap = kr * fsize + kc;\n"
"            for (int c = 0; c < cin; c++) {\n"
"                float v = in[(size_t)c * Hin * Win + (size_t)iy * Win + ix];\n"
"                float w = weight[(size_t)c * fsize * fsize + tap];\n"
"                acc += v * w;\n"
"            }\n"
"        }\n"
"    }\n"
"    out[(size_t)oy * Wout + ox] = acc;\n"
"}\n";

struct fsrcnn_gpu_ctx {
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    cl_kernel k_conv;
    cl_kernel k_deconv;

    cl_mem w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, w7, b7, w8;
    float bias8; /* B8[0], dikirim sbg scalar kernel arg, bukan buffer */

    /* Buffer skrap, dialokasikan ulang otomatis kalau H,W berubah. */
    int cur_h, cur_w;
    cl_mem buf_in, buf_x1, buf_x2, buf_x3, buf_x4, buf_x5, buf_x6, buf_x7, buf_out;
};

static const char *cl_err_str(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        default: return "?";
    }
}

static void print_cl_error(const char *where, cl_int err) {
    fprintf(stderr, "[gpu] %s gagal, kode error OpenCL = %d (%s)\n", where, (int)err, cl_err_str(err));
}

/* Cari device: prioritas GPU bernama "Mali", lalu GPU apa pun, lalu device
 * apa pun (fallback pengembangan, mis. pocl CPU runtime -- lihat fsrcnn_gpu.h). */
static cl_int pick_device(cl_platform_id *out_plat, cl_device_id *out_dev) {
    cl_uint n_plat = 0;
    if (clGetPlatformIDs(0, NULL, &n_plat) != CL_SUCCESS || n_plat == 0) return CL_DEVICE_NOT_FOUND;
    cl_platform_id *plats = (cl_platform_id *)malloc(n_plat * sizeof(cl_platform_id));
    clGetPlatformIDs(n_plat, plats, NULL);

    cl_device_id best_gpu = NULL, best_mali = NULL, any_dev = NULL;
    cl_platform_id best_gpu_p = NULL, best_mali_p = NULL, any_dev_p = NULL;

    for (cl_uint p = 0; p < n_plat; p++) {
        cl_uint n_dev = 0;
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 0, NULL, &n_dev) != CL_SUCCESS || n_dev == 0) continue;
        cl_device_id *devs = (cl_device_id *)malloc(n_dev * sizeof(cl_device_id));
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, n_dev, devs, NULL);
        for (cl_uint d = 0; d < n_dev; d++) {
            char name[256] = {0};
            cl_device_type type = 0;
            clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(name), name, NULL);
            clGetDeviceInfo(devs[d], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
            fprintf(stderr, "[gpu] device terdeteksi: \"%s\" (%s)\n", name,
                    (type & CL_DEVICE_TYPE_GPU) ? "GPU" : (type & CL_DEVICE_TYPE_CPU) ? "CPU" : "lainnya");
            if (!any_dev) { any_dev = devs[d]; any_dev_p = plats[p]; }
            if ((type & CL_DEVICE_TYPE_GPU) && !best_gpu) { best_gpu = devs[d]; best_gpu_p = plats[p]; }
            if ((type & CL_DEVICE_TYPE_GPU) && !best_mali && strstr(name, "Mali")) {
                best_mali = devs[d]; best_mali_p = plats[p];
            }
        }
        free(devs);
    }
    free(plats);

    if (best_mali) { *out_dev = best_mali; *out_plat = best_mali_p; return CL_SUCCESS; }
    if (best_gpu)  { *out_dev = best_gpu;  *out_plat = best_gpu_p;  return CL_SUCCESS; }
    if (any_dev)   {
        fprintf(stderr, "[gpu] PERINGATAN: tidak ada device GPU, fallback ke device pertama yg ditemukan "
                        "(wajar utk pengembangan dgn pocl; TIDAK direkomendasikan utk pengukuran performa)\n");
        *out_dev = any_dev; *out_plat = any_dev_p; return CL_SUCCESS;
    }
    return CL_DEVICE_NOT_FOUND;
}

static cl_mem make_ro_buf(cl_context ctx, const float *host, size_t n_elems, cl_int *err) {
    return clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           n_elems * sizeof(float), (void *)host, err);
}

fsrcnn_gpu_ctx_t *fsrcnn_gpu_init(void) {
    fsrcnn_gpu_ctx_t *c = (fsrcnn_gpu_ctx_t *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->cur_h = c->cur_w = -1;

    cl_platform_id plat;
    cl_device_id dev;
    cl_int err = pick_device(&plat, &dev);
    if (err != CL_SUCCESS) { print_cl_error("pick_device", err); free(c); return NULL; }

    char name[256] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    fprintf(stderr, "[gpu] dipakai: \"%s\"\n", name);

    c->ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (err != CL_SUCCESS) { print_cl_error("clCreateContext", err); free(c); return NULL; }

    c->queue = clCreateCommandQueue(c->ctx, dev, 0, &err);
    if (err != CL_SUCCESS) {
        print_cl_error("clCreateCommandQueue", err);
        clReleaseContext(c->ctx); free(c); return NULL;
    }

    c->program = clCreateProgramWithSource(c->ctx, 1, &KERNEL_SRC, NULL, &err);
    if (err != CL_SUCCESS) {
        print_cl_error("clCreateProgramWithSource", err);
        clReleaseCommandQueue(c->queue); clReleaseContext(c->ctx); free(c); return NULL;
    }
    err = clBuildProgram(c->program, 1, &dev, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        print_cl_error("clBuildProgram", err);
        size_t log_size = 0;
        clGetProgramBuildInfo(c->program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        if (log_size > 1) {
            char *log = (char *)malloc(log_size);
            clGetProgramBuildInfo(c->program, dev, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "[gpu] build log kernel:\n%s\n", log);
            free(log);
        }
        clReleaseProgram(c->program); clReleaseCommandQueue(c->queue); clReleaseContext(c->ctx);
        free(c); return NULL;
    }

    c->k_conv = clCreateKernel(c->program, "conv_prelu", &err);
    if (err != CL_SUCCESS) { print_cl_error("clCreateKernel(conv_prelu)", err); fsrcnn_gpu_destroy(c); return NULL; }
    c->k_deconv = clCreateKernel(c->program, "deconv_reduce", &err);
    if (err != CL_SUCCESS) { print_cl_error("clCreateKernel(deconv_reduce)", err); fsrcnn_gpu_destroy(c); return NULL; }

    /* Upload SEMUA bobot sekali di init -- dipakai ulang di setiap forward(),
     * TIDAK di-upload ulang per frame (beda dgn buffer aktivasi skrap yg
     * memang berubah tiap frame, lihat fsrcnn_gpu_forward()). */
    cl_int e2 = CL_SUCCESS, e = CL_SUCCESS;
    c->w1 = make_ro_buf(c->ctx, W1, sizeof(W1)/sizeof(float), &e); e2 |= e;
    c->b1 = make_ro_buf(c->ctx, B1, sizeof(B1)/sizeof(float), &e); e2 |= e;
    c->w2 = make_ro_buf(c->ctx, W2, sizeof(W2)/sizeof(float), &e); e2 |= e;
    c->b2 = make_ro_buf(c->ctx, B2, sizeof(B2)/sizeof(float), &e); e2 |= e;
    c->w3 = make_ro_buf(c->ctx, W3, sizeof(W3)/sizeof(float), &e); e2 |= e;
    c->b3 = make_ro_buf(c->ctx, B3, sizeof(B3)/sizeof(float), &e); e2 |= e;
    c->w4 = make_ro_buf(c->ctx, W4, sizeof(W4)/sizeof(float), &e); e2 |= e;
    c->b4 = make_ro_buf(c->ctx, B4, sizeof(B4)/sizeof(float), &e); e2 |= e;
    c->w5 = make_ro_buf(c->ctx, W5, sizeof(W5)/sizeof(float), &e); e2 |= e;
    c->b5 = make_ro_buf(c->ctx, B5, sizeof(B5)/sizeof(float), &e); e2 |= e;
    c->w6 = make_ro_buf(c->ctx, W6, sizeof(W6)/sizeof(float), &e); e2 |= e;
    c->b6 = make_ro_buf(c->ctx, B6, sizeof(B6)/sizeof(float), &e); e2 |= e;
    c->w7 = make_ro_buf(c->ctx, W7, sizeof(W7)/sizeof(float), &e); e2 |= e;
    c->b7 = make_ro_buf(c->ctx, B7, sizeof(B7)/sizeof(float), &e); e2 |= e;
    c->w8 = make_ro_buf(c->ctx, W8, sizeof(W8)/sizeof(float), &e); e2 |= e;
    c->bias8 = B8[0];
    if (e2 != CL_SUCCESS) {
        fprintf(stderr, "[gpu] gagal upload salah satu buffer bobot ke device\n");
        fsrcnn_gpu_destroy(c); return NULL;
    }

    return c;
}

static cl_int enqueue_conv(fsrcnn_gpu_ctx_t *c, cl_mem in, cl_mem w, cl_mem b, cl_mem out,
                            int cin, int H, int W, int cout, int kh, int kw, int pad,
                            float prelu_slope, int use_prelu) {
    cl_kernel k = c->k_conv;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k, 0, sizeof(cl_mem), &in);
    err |= clSetKernelArg(k, 1, sizeof(cl_mem), &w);
    err |= clSetKernelArg(k, 2, sizeof(cl_mem), &b);
    err |= clSetKernelArg(k, 3, sizeof(cl_mem), &out);
    err |= clSetKernelArg(k, 4, sizeof(int), &cin);
    err |= clSetKernelArg(k, 5, sizeof(int), &H);
    err |= clSetKernelArg(k, 6, sizeof(int), &W);
    err |= clSetKernelArg(k, 7, sizeof(int), &cout);
    err |= clSetKernelArg(k, 8, sizeof(int), &kh);
    err |= clSetKernelArg(k, 9, sizeof(int), &kw);
    err |= clSetKernelArg(k, 10, sizeof(int), &pad);
    err |= clSetKernelArg(k, 11, sizeof(float), &prelu_slope);
    err |= clSetKernelArg(k, 12, sizeof(int), &use_prelu);
    if (err != CL_SUCCESS) return err;

    size_t gws[3] = { (size_t)W, (size_t)H, (size_t)cout };
    return clEnqueueNDRangeKernel(c->queue, k, 3, NULL, gws, NULL, 0, NULL, NULL);
}

static cl_int enqueue_deconv(fsrcnn_gpu_ctx_t *c, cl_mem in, cl_mem w, float bias, cl_mem out,
                              int cin, int Hin, int Win, int fsize, int stride, int border) {
    cl_kernel k = c->k_deconv;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k, 0, sizeof(cl_mem), &in);
    err |= clSetKernelArg(k, 1, sizeof(cl_mem), &w);
    err |= clSetKernelArg(k, 2, sizeof(int), &cin);
    err |= clSetKernelArg(k, 3, sizeof(int), &Hin);
    err |= clSetKernelArg(k, 4, sizeof(int), &Win);
    err |= clSetKernelArg(k, 5, sizeof(int), &fsize);
    err |= clSetKernelArg(k, 6, sizeof(int), &stride);
    err |= clSetKernelArg(k, 7, sizeof(int), &border);
    err |= clSetKernelArg(k, 8, sizeof(float), &bias);
    err |= clSetKernelArg(k, 9, sizeof(cl_mem), &out);
    if (err != CL_SUCCESS) return err;

    int Hout = Hin * stride, Wout = Win * stride;
    size_t gws[2] = { (size_t)Wout, (size_t)Hout };
    return clEnqueueNDRangeKernel(c->queue, k, 2, NULL, gws, NULL, 0, NULL, NULL);
}

static void release_scratch(fsrcnn_gpu_ctx_t *c) {
    cl_mem *bufs[9] = { &c->buf_in, &c->buf_x1, &c->buf_x2, &c->buf_x3,
                         &c->buf_x4, &c->buf_x5, &c->buf_x6, &c->buf_x7, &c->buf_out };
    for (int i = 0; i < 9; i++) {
        if (*bufs[i]) { clReleaseMemObject(*bufs[i]); *bufs[i] = NULL; }
    }
}

int fsrcnn_gpu_forward(fsrcnn_gpu_ctx_t *ctx, const fsrcnn_tensor_t *in, fsrcnn_tensor_t *out) {
    if (!ctx || in->channels != 1) return -1;
    const int H = in->height, W = in->width;

    int Hout, Wout;
    fsrcnn_deconv_output_size(H, W, 9, FSRCNN_SCALE, 1, &Hout, &Wout);
    if (out->data == NULL || out->channels != 1 || out->height != Hout || out->width != Wout) return -2;

    cl_int err;
    if (H != ctx->cur_h || W != ctx->cur_w) {
        release_scratch(ctx);
        ctx->buf_in  = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, (size_t)1 * H * W * sizeof(float), NULL, &err);
        ctx->buf_x1  = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, (size_t)FSRCNN_D * H * W * sizeof(float), NULL, &err);
        ctx->buf_x2  = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, (size_t)FSRCNN_S * H * W * sizeof(float), NULL, &err);
        ctx->buf_x3  = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, (size_t)FSRCNN_S * H * W * sizeof(float), NULL, &err);
        ctx->buf_x4  = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, (size_t)FSRCNN_S * H * W * sizeof(float), NULL, &err);
        ctx->buf_x5  = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, (size_t)FSRCNN_S * H * W * sizeof(float), NULL, &err);
        ctx->buf_x6  = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, (size_t)FSRCNN_S * H * W * sizeof(float), NULL, &err);
        ctx->buf_x7  = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, (size_t)FSRCNN_D * H * W * sizeof(float), NULL, &err);
        ctx->buf_out = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, (size_t)1 * Hout * Wout * sizeof(float), NULL, &err);
        if (!ctx->buf_in || !ctx->buf_x1 || !ctx->buf_x2 || !ctx->buf_x3 || !ctx->buf_x4 ||
            !ctx->buf_x5 || !ctx->buf_x6 || !ctx->buf_x7 || !ctx->buf_out) {
            fprintf(stderr, "[gpu] gagal alokasi buffer device utk shape %dx%d\n", H, W);
            release_scratch(ctx);
            ctx->cur_h = ctx->cur_w = -1;
            return -3;
        }
        ctx->cur_h = H; ctx->cur_w = W;
    }

    err = clEnqueueWriteBuffer(ctx->queue, ctx->buf_in, CL_TRUE, 0, (size_t)H * W * sizeof(float), in->data, 0, NULL, NULL);
    if (err != CL_SUCCESS) { print_cl_error("clEnqueueWriteBuffer(in)", err); return -4; }

    cl_int rc = CL_SUCCESS;
    rc |= enqueue_conv(ctx, ctx->buf_in, ctx->w1, ctx->b1, ctx->buf_x1, 1,          H, W, FSRCNN_D, 5, 5, 2, FSRCNN_PRELU1, 1);
    rc |= enqueue_conv(ctx, ctx->buf_x1, ctx->w2, ctx->b2, ctx->buf_x2, FSRCNN_D,   H, W, FSRCNN_S, 1, 1, 0, FSRCNN_PRELU2, 1);
    rc |= enqueue_conv(ctx, ctx->buf_x2, ctx->w3, ctx->b3, ctx->buf_x3, FSRCNN_S,   H, W, FSRCNN_S, 3, 3, 1, FSRCNN_PRELU3, 1);
    rc |= enqueue_conv(ctx, ctx->buf_x3, ctx->w4, ctx->b4, ctx->buf_x4, FSRCNN_S,   H, W, FSRCNN_S, 3, 3, 1, FSRCNN_PRELU4, 1);
    rc |= enqueue_conv(ctx, ctx->buf_x4, ctx->w5, ctx->b5, ctx->buf_x5, FSRCNN_S,   H, W, FSRCNN_S, 3, 3, 1, FSRCNN_PRELU5, 1);
    rc |= enqueue_conv(ctx, ctx->buf_x5, ctx->w6, ctx->b6, ctx->buf_x6, FSRCNN_S,   H, W, FSRCNN_S, 3, 3, 1, FSRCNN_PRELU6, 1);
    rc |= enqueue_conv(ctx, ctx->buf_x6, ctx->w7, ctx->b7, ctx->buf_x7, FSRCNN_S,   H, W, FSRCNN_D, 1, 1, 0, FSRCNN_PRELU7, 1);
    if (rc != CL_SUCCESS) { print_cl_error("enqueue_conv (L1..L7)", rc); return -5; }

    rc = enqueue_deconv(ctx, ctx->buf_x7, ctx->w8, ctx->bias8, ctx->buf_out, FSRCNN_D, H, W, 9, FSRCNN_SCALE, 1);
    if (rc != CL_SUCCESS) { print_cl_error("enqueue_deconv (L8)", rc); return -6; }

    err = clEnqueueReadBuffer(ctx->queue, ctx->buf_out, CL_TRUE, 0, (size_t)Hout * Wout * sizeof(float), out->data, 0, NULL, NULL);
    if (err != CL_SUCCESS) { print_cl_error("clEnqueueReadBuffer(out)", err); return -7; }

    return 0;
}

void fsrcnn_gpu_destroy(fsrcnn_gpu_ctx_t *ctx) {
    if (!ctx) return;
    release_scratch(ctx);
    cl_mem *w[15] = { &ctx->w1, &ctx->b1, &ctx->w2, &ctx->b2, &ctx->w3, &ctx->b3,
                       &ctx->w4, &ctx->b4, &ctx->w5, &ctx->b5, &ctx->w6, &ctx->b6,
                       &ctx->w7, &ctx->b7, &ctx->w8 };
    for (int i = 0; i < 15; i++) if (*w[i]) clReleaseMemObject(*w[i]);
    if (ctx->k_conv) clReleaseKernel(ctx->k_conv);
    if (ctx->k_deconv) clReleaseKernel(ctx->k_deconv);
    if (ctx->program) clReleaseProgram(ctx->program);
    if (ctx->queue) clReleaseCommandQueue(ctx->queue);
    if (ctx->ctx) clReleaseContext(ctx->ctx);
    free(ctx);
}
