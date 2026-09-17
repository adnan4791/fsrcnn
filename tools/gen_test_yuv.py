#!/usr/bin/env python3
"""
gen_test_yuv.py -- generate file YUV 4:2:0 mentah berisi noise acak, untuk
uji cepat pipeline tanpa perlu video sungguhan.

Pemakaian:
    python3 gen_test_yuv.py out.yuv --width 64 --height 48 --frames 3
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_path")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cw, ch = (args.width + 1) // 2, (args.height + 1) // 2

    with open(args.out_path, "wb") as f:
        for _ in range(args.frames):
            y = rng.integers(0, 256, (args.height, args.width), dtype=np.uint8)
            u = rng.integers(0, 256, (ch, cw), dtype=np.uint8)
            v = rng.integers(0, 256, (ch, cw), dtype=np.uint8)
            f.write(y.tobytes())
            f.write(u.tobytes())
            f.write(v.tobytes())

    print(f"Ditulis {args.out_path}: {args.width}x{args.height}, {args.frames} frame, format yuv420p")
    print(f"(chroma {cw}x{ch} per plane U/V)")


if __name__ == "__main__":
    main()
