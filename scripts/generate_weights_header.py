"""
Menghasilkan include/fsrcnn_weights.h: bobot & bias FSRCNN sebagai array C
`static const float` yang di-embed langsung di source (bukan file biner
eksternal). Untuk jaringan sekecil ini (~12.6K float / ~50KB), embedding
sebagai source jauh lebih sederhana daripada menulis parser biner kustom --
tidak ada isu endianness, tidak ada dependency path file runtime, dan bobotnya
bisa langsung dibaca/diperiksa manusia (relevan untuk konteks buku ajar).

Layout tensor MENGIKUTI KONVENSI PyTorch/ONNX: bobot conv = (out_ch, in_ch,
kH, kW), row-major/C-order flatten -- identik dengan urutan angka di file
.txt sumber, jadi tidak ada reshuffle di sini.
"""
import sys
import numpy as np


def load_txt(path, shape):
    data = np.loadtxt(path, dtype=np.float32)
    n_expected = int(np.prod(shape))
    if data.size != n_expected:
        raise ValueError(f"{path}: jumlah angka={data.size}, diharapkan={n_expected} untuk shape={shape}")
    return data.reshape(shape)


def emit_array(f, c_name, arr):
    flat = arr.flatten()
    f.write(f"static const float {c_name}[{flat.size}] = {{\n")
    for i in range(0, flat.size, 8):
        chunk = flat[i:i+8]
        f.write("    " + ", ".join(f"{v:.9g}f" for v in chunk) + ",\n")
    f.write("};\n\n")


def main():
    weights_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out_path = sys.argv[2] if len(sys.argv) > 2 else "fsrcnn_weights.h"

    D, S, M = 56, 12, 4  # feature dims, sesuai fsrcnn_pytorch.py asli

    # Slope PReLU per-layer -- KONSTANTA, diambil PERSIS dari referensi
    # otoritatif source_channel_pixel.c (bukan hasil pelatihan yang dibaca
    # dari file, karena memang tidak ada file eksternal utk ini). Boleh
    # negatif (lihat prelu1) -- JANGAN disamakan dengan ReLU biasa.
    prelu_slopes = {
        "PRELU1": -0.8986, "PRELU2": 0.3236, "PRELU3": 0.2288, "PRELU4": 0.2476,
        "PRELU5": 0.3495, "PRELU6": 0.7806, "PRELU7": 0.0087,
    }

    layers = [
        ("W1", "weights_layer1.txt", (D, 1, 5, 5)), ("B1", "biasess_layer1.txt", (D,)),
        ("W2", "weights_layer2.txt", (S, D, 1, 1)), ("B2", "biasess_layer2.txt", (S,)),
        ("W3", "weights_layer3.txt", (S, S, 3, 3)), ("B3", "biasess_layer3.txt", (S,)),
        ("W4", "weights_layer4.txt", (S, S, 3, 3)), ("B4", "biasess_layer4.txt", (S,)),
        ("W5", "weights_layer5.txt", (S, S, 3, 3)), ("B5", "biasess_layer5.txt", (S,)),
        ("W6", "weights_layer6.txt", (S, S, 3, 3)), ("B6", "biasess_layer6.txt", (S,)),
        ("W7", "weights_layer7.txt", (D, S, 1, 1)), ("B7", "biasess_layer7.txt", (D,)),
        ("W8", "weights_layer8.txt", (D, 1, 9, 9)),
    ]

    arrays = {}
    for c_name, fname, shape in layers:
        arrays[c_name] = load_txt(f"{weights_dir}/{fname}", shape)
    # B8: tidak ada file sumber -- konstanta identik dgn skrip asli pengguna.
    arrays["B8"] = np.array([-0.03262640000], dtype=np.float32)

    with open(out_path, "w") as f:
        f.write("/* File hasil generate OTOMATIS oleh generate_weights_header.py -- JANGAN EDIT MANUAL.\n")
        f.write(" * Sumber: 15 file weights_layer*.txt / biasess_layer*.txt (lihat catatan proyek).\n")
        f.write(" * Layout tensor conv: (out_ch, in_ch, kH, kW), row-major (konvensi PyTorch/ONNX). */\n")
        f.write("#ifndef FSRCNN_WEIGHTS_H\n#define FSRCNN_WEIGHTS_H\n\n")
        f.write(f"#define FSRCNN_D {D}   /* feature extraction / expanding channels */\n")
        f.write(f"#define FSRCNN_S {S}   /* shrinking / mapping channels */\n")
        f.write(f"#define FSRCNN_M {M}   /* jumlah layer mapping (conv 3x3) */\n")
        f.write("#define FSRCNN_SCALE 2 /* faktor upscale (stride deconv) */\n\n")
        f.write("/* Slope PReLU per-layer (v>=0 ? v : slope*v) -- lihat catatan di atas. */\n")
        for name, val in prelu_slopes.items():
            f.write(f"#define FSRCNN_{name} {val}f\n")
        f.write("\n")
        for c_name, _, _ in layers:
            emit_array(f, c_name, arrays[c_name])
        f.write("/* B8: konstanta hardcode (tidak ada biasess_layer8.txt di sumber data).\n")
        f.write(" * Sudah DIKONFIRMASI identik dgn biases_layer8 di source_channel_pixel.c. */\n")
        emit_array(f, "B8", arrays["B8"])
        f.write("#endif /* FSRCNN_WEIGHTS_H */\n")

    total = sum(a.size for a in arrays.values())
    print(f"Ditulis: {out_path} ({total} float, ~{total*4/1024:.1f} KB data biner setara)")


if __name__ == "__main__":
    main()
