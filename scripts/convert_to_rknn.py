"""
Konversi fsrcnn_dynamic.onnx -> fsrcnn_rk3588.rknn menggunakan rknn-toolkit2.

KOREKSI PENTING (per source_channel_pixel.c, referensi otoritatif): model
FSRCNN ini HANYA dipakai untuk plane Y (luma). Plane U/V di-upscale dengan
replikasi nearest-neighbor sederhana di CPU (lihat chroma_upsample_nn di
yuv_io.c/.h) -- BUKAN lewat NPU. Karena itu file .rknn ini HANYA perlu
mendukung SATU shape (Y-plane), bukan dua (Y + chroma) seperti versi
sebelumnya -- lebih sederhana & (secara umum) lebih cepat di NPU
dibanding mode `dynamic_input` (yang mem-build banyak subgraph sekaligus).

PENTING -- resolusi harus ditentukan di waktu konversi (khas NPU):
Berbeda dari CPU, backend NPU RKNN mengkompilasi graph untuk shape statis
tertentu. Kalau video Anda beresolusi lain dari default di bawah, panggil
ulang skrip ini dengan --luma-h/--luma-w yang sesuai (hasilkan .rknn baru
per resolusi -- JANGAN dipakai lintas resolusi).

Default (--luma-h 144 --luma-w 176) mengikuti resolusi QCIF yang di-hardcode
di source_channel_pixel.c (inRows=144, inCols=176).

Penggunaan:
    python3 convert_to_rknn.py fsrcnn_dynamic.onnx fsrcnn_rk3588.rknn \
        --luma-h 144 --luma-w 176 [--quantize --dataset calib_list.txt]
"""
import argparse
import sys
from rknn.api import RKNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path")
    ap.add_argument("rknn_path")
    ap.add_argument("--luma-h", type=int, default=144, help="Tinggi plane Y (default 144, sesuai QCIF di source_channel_pixel.c)")
    ap.add_argument("--luma-w", type=int, default=176, help="Lebar plane Y (default 176)")
    ap.add_argument("--target-platform", default="rk3588")
    ap.add_argument("--quantize", action="store_true",
                     help="Aktifkan kuantisasi int8 (w8a8). Default: OFF (float16) "
                          "karena FSRCNN adalah jaringan regresi presisi-piksel -- "
                          "kuantisasi int8 tanpa kalibrasi yang matang berisiko "
                          "menurunkan PSNR/SSIM output secara nyata.")
    ap.add_argument("--dataset", default=None,
                     help="Path file daftar gambar kalibrasi (wajib jika --quantize dipakai)")
    args = ap.parse_args()

    luma_h, luma_w = args.luma_h, args.luma_w
    print(f"[cfg] Y-plane (satu-satunya input NPU): {luma_h}x{luma_w}")
    print("[cfg] U/V TIDAK dikonversi -- ditangani via replikasi nearest-neighbor di CPU (yuv_io.c)")

    if args.quantize and not args.dataset:
        print("ERROR: --quantize membutuhkan --dataset (file daftar gambar kalibrasi representatif).", file=sys.stderr)
        sys.exit(1)

    rknn = RKNN(verbose=True)

    # Tidak ada normalisasi mean/std di sisi NPU: pixel sudah dinormalisasi ke
    # [0,1] (piksel/255.0) SEBELUM masuk ke model, persis sama dengan konvensi
    # pada sisi CPU/ONNX -- lihat catatan di build_fsrcnn_onnx_dynamic.py.
    print("[1/4] config()")
    ret = rknn.config(
        target_platform=args.target_platform,
        mean_values=[[0]],
        std_values=[[1]],
    )
    if ret != 0:
        print("config() gagal"); sys.exit(1)

    print("[2/4] load_onnx()")
    ret = rknn.load_onnx(
        model=args.onnx_path,
        inputs=["input"],
        input_size_list=[[1, 1, luma_h, luma_w]],
    )
    if ret != 0:
        print("load_onnx() gagal"); sys.exit(1)

    print(f"[3/4] build(do_quantization={args.quantize})")
    ret = rknn.build(do_quantization=args.quantize, dataset=args.dataset)
    if ret != 0:
        print("build() gagal"); sys.exit(1)

    print("[4/4] export_rknn()")
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print("export_rknn() gagal"); sys.exit(1)

    print(f"Sukses: {args.rknn_path}")
    rknn.release()


if __name__ == "__main__":
    main()
