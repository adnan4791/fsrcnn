"""
Membangun graph ONNX FSRCNN (8 layer) dari bobot .txt, untuk SATU shape
INPUT TETAP (H,W ditentukan di command line -- lihat alasan di bawah).

MODEL INI HANYA UNTUK PLANE Y (luma) -- lihat catatan otoritatif di
source_channel_pixel.c: plane U/V TIDAK melewati FSRCNN, di-upscale dengan
replikasi nearest-neighbor sederhana di sisi C (yuv_io.c: chroma_upsample_nn).

KOREKSI PENTING (hasil validasi bin-compare terhadap keluaran
source_channel_pixel.c -- lihat catatan verifikasi proyek):
  1. Padding pada SEMUA layer conv adalah REPLICATE/EDGE (bukan zero-pad
     implisit dari atribut `pads` pada operator Conv). Direpresentasikan
     eksplisit dengan node Pad(mode="edge") SEBELUM setiap Conv(pads=0).
  2. Aktivasi adalah PReLU PER-LAYER (slope skalar per layer, BOLEH
     negatif), bukan plain ReLU -- operator ONNX "PRelu".
  3. Layer 8 (upsampling) BUKAN ConvTranspose2d standar: input di-
     replicate-pad dgn border=1, lalu "full transposed convolution" (TANPA
     crop implisit dari atribut `pads`), lalu di-crop eksplisit dengan node
     Slice pada offset TETAP (konstanta 6, diturunkan dari
     (fsize+1)/2 + stride*border - 1 = 5+2-1, utk fsize=9,stride=2,border=1).

Kenapa SHAPE TETAP (bukan dinamis H,W seperti versi awal proyek ini)?
Offset crop pada Slice utk layer 8 adalah konstanta, tapi UKURAN jendela
crop (Hout=2*Hin, Wout=2*Win) bergantung pada Hin/Win. Untuk graph yang
truly dinamis, ukuran ini harus dihitung via node Shape/Mul saat runtime --
bisa dilakukan, tapi menambah kompleksitas yang tidak sepadan mengingat
backend NPU (RKNN, lihat convert_to_rknn.py) TETAP mengharuskan satu shape
tetap per model. Jadi baik jalur ONNX/onnxruntime (CPU) maupun RKNN (NPU)
kini konsisten: satu resolusi tetap per build, ganti resolusi = build ulang.

Konvensi numerik (konsisten dgn sisi C, fsrcnn.h/fsrcnn_ops.c):
- Tensor float32 NCHW, C=1. Piksel dinormalisasi ke [0,1] SEBELUM masuk
  model (tidak ada node Div/Sub di dalam graph).

Penggunaan:
    python3 build_fsrcnn_onnx.py <folder_txt> fsrcnn.onnx --height 144 --width 176
"""
import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto

PRELU_SLOPES = {
    1: -0.8986, 2: 0.3236, 3: 0.2288, 4: 0.2476,
    5: 0.3495, 6: 0.7806, 7: 0.0087,
}


def load_txt(path, shape):
    data = np.loadtxt(path, dtype=np.float32)
    n_expected = int(np.prod(shape))
    if data.size != n_expected:
        raise ValueError(f"{path}: jumlah angka={data.size}, diharapkan={n_expected} untuk shape={shape}")
    return data.reshape(shape)


def build(weights_dir, out_path, height, width):
    w1 = load_txt(f"{weights_dir}/weights_layer1.txt", (56, 1, 5, 5));  b1 = load_txt(f"{weights_dir}/biasess_layer1.txt", (56,))
    w2 = load_txt(f"{weights_dir}/weights_layer2.txt", (12, 56, 1, 1)); b2 = load_txt(f"{weights_dir}/biasess_layer2.txt", (12,))
    w3 = load_txt(f"{weights_dir}/weights_layer3.txt", (12, 12, 3, 3)); b3 = load_txt(f"{weights_dir}/biasess_layer3.txt", (12,))
    w4 = load_txt(f"{weights_dir}/weights_layer4.txt", (12, 12, 3, 3)); b4 = load_txt(f"{weights_dir}/biasess_layer4.txt", (12,))
    w5 = load_txt(f"{weights_dir}/weights_layer5.txt", (12, 12, 3, 3)); b5 = load_txt(f"{weights_dir}/biasess_layer5.txt", (12,))
    w6 = load_txt(f"{weights_dir}/weights_layer6.txt", (12, 12, 3, 3)); b6 = load_txt(f"{weights_dir}/biasess_layer6.txt", (12,))
    w7 = load_txt(f"{weights_dir}/weights_layer7.txt", (56, 12, 1, 1)); b7 = load_txt(f"{weights_dir}/biasess_layer7.txt", (56,))
    w8 = load_txt(f"{weights_dir}/weights_layer8.txt", (56, 1, 9, 9))
    b8 = np.array([-0.03262640000], dtype=np.float32)  # dikonfirmasi thd source_channel_pixel.c

    print("Semua file teks berhasil dimuat ke dalam struktur model!")

    def T(name, arr, dtype=TensorProto.FLOAT):
        return helper.make_tensor(name, dtype, list(arr.shape), arr.flatten().tolist())

    initializers = [
        T("w1", w1), T("b1", b1), T("w2", w2), T("b2", b2),
        T("w3", w3), T("b3", b3), T("w4", w4), T("b4", b4),
        T("w5", w5), T("b5", b5), T("w6", w6), T("b6", b6),
        T("w7", w7), T("b7", b7), T("w8", w8), T("b8", b8),
    ]
    for idx, slope in PRELU_SLOPES.items():
        initializers.append(T(f"prelu{idx}_slope", np.array([slope], dtype=np.float32)))

    nodes = []

    def pad_edge(idx, x_in, x_out, p):
        pads = np.array([0, 0, p, p, 0, 0, p, p], dtype=np.int64)
        initializers.append(T(f"pad{idx}_amount", pads, dtype=TensorProto.INT64))
        nodes.append(helper.make_node("Pad", [x_in, f"pad{idx}_amount"], [x_out],
                                       mode="edge", name=f"Pad{idx}"))

    def conv_prelu(idx, x_in, x_out, w, b, k, pad):
        cur = x_in
        if pad > 0:
            padded = x_out + "_padded"
            pad_edge(idx, cur, padded, pad)
            cur = padded
        conv_out = x_out + "_pre"
        nodes.append(helper.make_node("Conv", [cur, w, b], [conv_out],
                                       kernel_shape=[k, k], pads=[0, 0, 0, 0],
                                       strides=[1, 1], name=f"Conv{idx}"))
        nodes.append(helper.make_node("PRelu", [conv_out, f"prelu{idx}_slope"], [x_out], name=f"PRelu{idx}"))

    conv_prelu(1, "input", "x1", "w1", "b1", 5, 2)
    conv_prelu(2, "x1", "x2", "w2", "b2", 1, 0)
    conv_prelu(3, "x2", "x3", "w3", "b3", 3, 1)
    conv_prelu(4, "x3", "x4", "w4", "b4", 3, 1)
    conv_prelu(5, "x4", "x5", "w5", "b5", 3, 1)
    conv_prelu(6, "x5", "x6", "w6", "b6", 3, 1)
    conv_prelu(7, "x6", "x7", "w7", "b7", 1, 0)

    # --- Layer 8: upsampling custom (lihat docstring modul) ---
    FSIZE, STRIDE, BORDER = 9, 2, 1
    pad_edge(8, "x7", "x7_padded", BORDER)  # (1,56,H+2,W+2)
    nodes.append(helper.make_node(
        "ConvTranspose", ["x7_padded", "w8", "b8"], ["deconv_full"],
        kernel_shape=[FSIZE, FSIZE], pads=[0, 0, 0, 0], strides=[STRIDE, STRIDE],
        output_padding=[0, 0], name="Deconv8"))
    # deconv_full: (1,1, (H+2-1)*2+9, (W+2-1)*2+9) = (1,1, 2H+11, 2W+11)
    offset = (FSIZE + 1) // 2 + STRIDE * BORDER - 1  # = 6, konstanta
    h_out, w_out = height * STRIDE, width * STRIDE
    starts = np.array([offset, offset], dtype=np.int64)
    ends = np.array([offset + h_out, offset + w_out], dtype=np.int64)
    axes = np.array([2, 3], dtype=np.int64)
    initializers.append(T("crop_starts", starts, dtype=TensorProto.INT64))
    initializers.append(T("crop_ends", ends, dtype=TensorProto.INT64))
    initializers.append(T("crop_axes", axes, dtype=TensorProto.INT64))
    nodes.append(helper.make_node("Slice", ["deconv_full", "crop_starts", "crop_ends", "crop_axes"],
                                   ["output"], name="CropDeconv8"))

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, height, width])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, h_out, w_out])

    graph = helper.make_graph(nodes, "FSRCNN", [input_info], [output_info], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"Model tersimpan: {out_path} (input {height}x{width} -> output {h_out}x{w_out})")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights_dir")
    ap.add_argument("out_path")
    ap.add_argument("--height", type=int, default=144, help="Tinggi plane Y (default 144, QCIF)")
    ap.add_argument("--width", type=int, default=176, help="Lebar plane Y (default 176, QCIF)")
    args = ap.parse_args()
    build(args.weights_dir, args.out_path, args.height, args.width)
