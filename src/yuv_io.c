#include "yuv_io.h"

#include <stdlib.h>

void yuv420_chroma_dims(int luma_h, int luma_w, int *chroma_h, int *chroma_w) {
    *chroma_h = (luma_h + 1) / 2;
    *chroma_w = (luma_w + 1) / 2;
}

size_t yuv420_frame_size(const yuv420_dims_t *d) {
    int ch, cw;
    yuv420_chroma_dims(d->height, d->width, &ch, &cw);
    return (size_t)d->height * d->width + 2 * (size_t)ch * cw;
}

static int read_plane_u8_to_f32(FILE *f, fsrcnn_tensor_t *t) {
    size_t n = (size_t)t->height * t->width;
    unsigned char *buf = (unsigned char *)malloc(n);
    if (!buf) return -1;
    size_t got = fread(buf, 1, n, f);
    if (got != n) {
        free(buf);
        return (got == 0) ? 1 : -1; /* 0 byte terbaca dianggap EOF bersih */
    }
    for (size_t i = 0; i < n; i++) t->data[i] = (float)buf[i] / 255.0f;
    free(buf);
    return 0;
}

int yuv420_read_frame(FILE *f, const yuv420_dims_t *d,
                       fsrcnn_tensor_t *y, fsrcnn_tensor_t *u, fsrcnn_tensor_t *v) {
    int ch, cw;
    yuv420_chroma_dims(d->height, d->width, &ch, &cw);
    if (y->height != d->height || y->width != d->width) return -1;
    if (u->height != ch || u->width != cw) return -1;
    if (v->height != ch || v->width != cw) return -1;

    int rc = read_plane_u8_to_f32(f, y);
    if (rc != 0) return rc; /* EOF di awal frame = wajar (akhir video) */
    rc = read_plane_u8_to_f32(f, u);
    if (rc != 0) return -1; /* EOF di tengah frame = file korup/terpotong */
    rc = read_plane_u8_to_f32(f, v);
    if (rc != 0) return -1;
    return 0;
}

static void write_plane_f32_to_u8(FILE *f, const fsrcnn_tensor_t *t) {
    size_t n = (size_t)t->height * t->width;
    unsigned char *buf = (unsigned char *)malloc(n);
    if (!buf) return;
    for (size_t i = 0; i < n; i++) {
        float v = t->data[i] * 255.0f;
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        buf[i] = (unsigned char)(v + 0.5f); /* round-to-nearest */
    }
    fwrite(buf, 1, n, f);
    free(buf);
}

int yuv420_write_frame(FILE *f, const fsrcnn_tensor_t *y,
                        const fsrcnn_tensor_t *u, const fsrcnn_tensor_t *v) {
    write_plane_f32_to_u8(f, y);
    write_plane_f32_to_u8(f, u);
    write_plane_f32_to_u8(f, v);
    return 0;
}

void chroma_upsample_nn(const fsrcnn_tensor_t *in, fsrcnn_tensor_t *out, int scale) {
    const int H = in->height, W = in->width;
    const int Wout = out->width;
    for (int i = 0; i < H; i++) {
        const float *src_row = in->data + (size_t)i * W;
        for (int j = 0; j < W; j++) {
            float val = src_row[j];
            for (int di = 0; di < scale; di++) {
                float *dst_row = out->data + (size_t)(i * scale + di) * Wout;
                for (int dj = 0; dj < scale; dj++) {
                    dst_row[j * scale + dj] = val;
                }
            }
        }
    }
}
