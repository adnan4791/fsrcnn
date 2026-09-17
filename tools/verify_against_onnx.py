#!/usr/bin/env python3
"""
verify_against_onnx.py -- verifikasi korektnes numerik: bandingkan output
biner dari fsrcnn_cpu / fsrcnn_npu terhadap referensi.

KOREKSI PENTING (per source_channel_pixel.c, referensi otoritatif):
  - Plane Y  : dibandingkan terhadap onnxruntime menjalankan fsrcnn.onnx
               (model FSRCNN sesungguhnya).
  - Plane U/V: TIDAK melewati FSRCNN sama sekali -- dibandingkan terhadap
               replikasi nearest-neighbor 2x2 murni dari input (bukan
               dijalankan lewat model). Ini meniru persis apa yang dilakukan
               chroma_upsample_nn() di yuv_io.c / blok "U/V Component" pada
               source_channel_pixel.c.

Dipakai untuk regresi setelah mengubah kode C (mis. optimasi NEON baru) --
kalau selisihnya melonjak, ada bug.

Pemakaian:
    python3 tools/verify_against_onnx.py \
        --in-yuv test_in.yuv --out-yuv test_out.yuv \
        --onnx fsrcnn.onnx --width 176 --height 144 --frames 3
"""
import argparse
import numpy as np
import onnxruntime as ort


def sr_plane_y_ref(sess, oin, oout, plane_u8):
    """Jalankan FSRCNN (via onnxruntime) pada satu plane Y. Return uint8 2x upscale."""
    x = (plane_u8.astype(np.float32) / 255.0)[None, None, :, :]
    y = sess.run([oout], {oin: x})[0][0, 0]
    y = np.clip(y * 255.0, 0, 255)
    return np.round(y).astype(np.uint8)


def upscale_chroma_nn_ref(plane_u8, scale=2):
    """Replikasi nearest-neighbor blok scale x scale -- BUKAN via model,
    identik dengan chroma_upsample_nn() di yuv_io.c."""
    return np.repeat(np.repeat(plane_u8, scale, axis=0), scale, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-yuv", required=True)
    ap.add_argument("--out-yuv", required=True)
    ap.add_argument("--onnx", required=True, help="fsrcnn.onnx (model Y-only, shape tetap)")
    ap.add_argument("--width", type=int, required=True, help="lebar plane Y (HARUS sama dgn shape build fsrcnn.onnx)")
    ap.add_argument("--height", type=int, required=True, help="tinggi plane Y (HARUS sama dgn shape build fsrcnn.onnx)")
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--tolerance", type=int, default=1,
                     help="toleransi selisih maksimum dalam satuan uint8 (default 1, "
                          "wajar akibat reasosiasi floating-point antar urutan "
                          "penjumlahan berbeda -- lihat catatan di source_channel_pixel.c; "
                          "plane U/V harus SELALU bit-exact/0 karena replikasi murni)")
    args = ap.parse_args()

    W, H = args.width, args.height
    cw, ch = (W + 1) // 2, (H + 1) // 2  # dimensi chroma sumber (4:2:0)
    frame_in_size = H * W + 2 * ch * cw
    Yh, Yw = 2 * H, 2 * W          # luma keluaran: FSRCNN scale=2
    Ch, Cw = 2 * ch, 2 * cw        # chroma keluaran: replikasi NN scale=2
    frame_out_size = Yh * Yw + 2 * Ch * Cw

    raw_in = open(args.in_yuv, "rb").read()
    raw_out = open(args.out_yuv, "rb").read()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    oin, oout = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    worst_y, worst_uv = 0, 0
    for fi in range(args.frames):
        off_in = fi * frame_in_size
        y = np.frombuffer(raw_in, dtype=np.uint8, count=H * W, offset=off_in).reshape(H, W)
        u = np.frombuffer(raw_in, dtype=np.uint8, count=ch * cw, offset=off_in + H * W).reshape(ch, cw)
        v = np.frombuffer(raw_in, dtype=np.uint8, count=ch * cw, offset=off_in + H * W + ch * cw).reshape(ch, cw)

        y_ref = sr_plane_y_ref(sess, oin, oout, y)
        u_ref = upscale_chroma_nn_ref(u, 2)
        v_ref = upscale_chroma_nn_ref(v, 2)

        off_out = fi * frame_out_size
        y_c = np.frombuffer(raw_out, dtype=np.uint8, count=Yh * Yw, offset=off_out).reshape(Yh, Yw)
        u_c = np.frombuffer(raw_out, dtype=np.uint8, count=Ch * Cw, offset=off_out + Yh * Yw).reshape(Ch, Cw)
        v_c = np.frombuffer(raw_out, dtype=np.uint8, count=Ch * Cw, offset=off_out + Yh * Yw + Ch * Cw).reshape(Ch, Cw)

        dY = int(np.abs(y_c.astype(int) - y_ref.astype(int)).max())
        dU = int(np.abs(u_c.astype(int) - u_ref.astype(int)).max())
        dV = int(np.abs(v_c.astype(int) - v_ref.astype(int)).max())
        worst_y = max(worst_y, dY)
        worst_uv = max(worst_uv, dU, dV)
        print(f"frame {fi}: maxdiff Y(vs onnxruntime)={dY}  U,V(vs replikasi-NN)={dU},{dV}")

    print(f"\nSelisih maksimum Y: {worst_y} (toleransi: {args.tolerance})")
    print(f"Selisih maksimum U/V: {worst_uv} (harus 0 -- replikasi murni, tanpa floating point)")
    if worst_y > args.tolerance or worst_uv > 0:
        print("GAGAL: selisih melebihi toleransi.")
        raise SystemExit(1)
    print("OK: dalam toleransi.")


if __name__ == "__main__":
    main()
