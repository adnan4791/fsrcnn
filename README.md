# FSRCNN Video Super-Resolution -- CPU (NEON), NPU (RKNN), & GPU (OpenCL) di RK3588

Proyek ini melengkapi konversi bobot `.txt` FSRCNN menjadi TIGA jalur
eksekusi yang bisa dibandingkan langsung secara numerik pada SoC heterogen
RK3588 (Orange Pi 5) -- studi kasus akselerasi lintas unit compute:
1. **CPU** (`fsrcnn_cpu.c`/`fsrcnn_ops.c`): C murni, dioptimasi NEON, sadar
   topologi big.LITTLE.
2. **NPU** (`fsrcnn_npu.c`): offload ke NPU RK3588 lewat RKNN C API resmi.
3. **GPU** (`fsrcnn_gpu.c`): offload ke GPU Mali-G610 lewat OpenCL.

Ketiganya memproses video YUV 4:2:0 mentah dan menulis ulang frame hasil
upscale 2x, dgn algoritma inti yang identik (lihat bagian 0) -- perbedaannya
murni DI MANA komputasi dijalankan, bukan APA yang dihitung.

**Algoritma & pembagian kerja per channel MENGIKUTI PERSIS referensi otoritatif
`source_channel_pixel.c`** yang disediakan peneliti -- bobot yang dipakai di
sini dilatih khusus untuk mereproduksi program tersebut, sehingga setiap
detail numerik (aktivasi, padding, upsampling, pembagian channel) HARUS
identik, bukan hanya "mirip FSRCNN pada umumnya". Lihat bagian 0 di bawah.

## Struktur direktori

```
include/            fsrcnn.h, fsrcnn_npu.h, fsrcnn_gpu.h, yuv_io.h, cpu_affinity.h, fsrcnn_weights.h (auto-generate)
src/                 implementasi C: ops, forward pass CPU, wrapper NPU, wrapper GPU (OpenCL), I/O YUV, driver CLI
scripts/             skrip Python tahap konversi (.txt -> .onnx -> .rknn) & generator header bobot
tools/               utilitas bantu (generate YUV uji, verifikasi numerik vs onnxruntime,
                     yuv_diff.c -- bandingkan 2 file .yuv byte-level TANPA dependency,
                     dipakai langsung di board utk verifikasi CPU vs NPU vs GPU)
Makefile             build native, cross-compile aarch64 (CPU, NPU, & GPU)
fsrcnn.onnx          model ONNX Y-only, shape TETAP (hasil scripts/build_fsrcnn_onnx.py)
fsrcnn_rk3588.rknn   model NPU siap pakai, dibangun utk Y=144x176 (QCIF, default source_channel_pixel.c)
```

Catatan: GPU (`fsrcnn_gpu.c`) TIDAK butuh file model terpisah spt `.rknn` --
kernel OpenCL-nya di-compile at-runtime dari source string ter-embed di
`fsrcnn_gpu.c` sendiri (JIT via driver, bukan AOT spt RKNN), dan bobot
dibaca langsung dari `fsrcnn_weights.h` yang sama dipakai jalur CPU.

## 0. Konvensi & koreksi penting (WAJIB dibaca sebelum mengubah apa pun)

Iterasi awal proyek ini memakai asumsi FSRCNN generik (paper Dong et al.,
2016) yang TERNYATA tidak cocok dengan bobot yang tersedia. Setelah
membandingkan bit-demi-bit terhadap keluaran `source_channel_pixel.c` yang
dikompilasi & dijalankan sebagai *ground truth*, ditemukan 4 penyimpangan,
semuanya sudah dikoreksi di kode saat ini:

1. **Hanya plane Y (luma) yang melewati FSRCNN.** Plane U dan V di-upscale
   dengan **replikasi nearest-neighbor blok 2x2 sederhana**
   (`chroma_upsample_nn()` di `yuv_io.c`) -- TANPA jaringan saraf sama
   sekali. Ini persis meniru blok "U/V Component" pada `main()` di
   `source_channel_pixel.c`. Konsekuensinya: NPU dan GPU (dan sebagian besar
   biaya komputasi) hanya menyentuh plane Y; U/V murni operasi memori di CPU.
2. **Aktivasi adalah PReLU PER-LAYER** (satu skalar slope per layer, BOLEH
   negatif: layer1=-0.8986, layer2=0.3236, ..., layer7=0.0087) -- BUKAN
   plain ReLU. Nilai-nilai ini di-hardcode di `fsrcnn_weights.h` (via
   `scripts/generate_weights_header.py`) dan di ONNX (initializer
   `preluN_slope` + node `PRelu`), identik dengan `source_channel_pixel.c`.
3. **Padding pada semua layer conv adalah REPLICATE/EDGE** (menduplikasi
   piksel tepi), bukan zero-pad. Diimplementasikan eksplisit sebagai
   `pad_chw_replicate()` (C) / node `Pad(mode="edge")` (ONNX) sebelum tiap
   konvolusi -- lihat `fsrcnn_ops.c`.
4. **Layer 8 (upsampling) BUKAN `ConvTranspose2d` standar PyTorch.**
   Algoritmanya: replicate-pad input dengan `border=1` -> "full transposed
   convolution" (scatter, TANPA crop implisit dari `pads`) -> crop eksplisit
   pada offset TETAP `(fsize+1)/2 + stride*border - 1 = 6` (utk fsize=9,
   stride=2, border=1). Diimplementasikan sebagai `fsrcnn_deconv_reduce()`
   (C, port langsung dari `deconv_reduce_hwc()`) / `Pad+ConvTranspose(full)
   +Slice` (ONNX).

Konvensi lain yang TIDAK berubah dari desain awal:

- **Layout tensor**: channel-major `(C,H,W)`, bobot conv `(out_ch,in_ch,kH,kW)`
  -- identik konvensi PyTorch/ONNX, jadi bobot dari `.txt` dipakai langsung
  tanpa transpose. Bobot layer 8 (`(56,1,9,9)`) dipakai NATIVE tanpa repack.
- **Domain nilai piksel**: seluruh inference (CPU, NPU, maupun GPU) bekerja pada
  float32 `[0,1]` (piksel_8bit/255.0). Konversi ke/dari `uint8` HANYA terjadi
  di `yuv_io.c` (baca) dan `yuv_io.c` (tulis, dengan clamp+round). Inference
  engine sendiri tidak tahu apa-apa soal format file video.
- **Bias layer 8**: tidak ada file `biasess_layer8.txt` sumber -- nilai
  `-0.03262640000` yang dipakai di seluruh jalur (C, ONNX, RKNN) sudah
  DIKONFIRMASI identik dengan `biases_layer8` hardcode di
  `source_channel_pixel.c` (bukan lagi asumsi/TODO).

## 1. Alur konversi bobot -> model (Python, sekali jalan)

```bash
pip install onnx numpy onnxruntime rknn-toolkit2 --break-system-packages

# .txt (15 file weights_layer*/biasess_layer*) -> ONNX Y-only, shape TETAP
python3 scripts/build_fsrcnn_onnx.py <folder_txt> fsrcnn.onnx --height 144 --width 176

# ONNX -> RKNN, HARUS sebutkan resolusi Y video Anda (satu shape tetap, U/V tidak ikut)
python3 scripts/convert_to_rknn.py fsrcnn.onnx fsrcnn_rk3588.rknn \
    --luma-h 144 --luma-w 176

# .txt -> header C ter-embed (dipakai src/fsrcnn_cpu.c), termasuk konstanta PReLU
python3 scripts/generate_weights_header.py <folder_txt> include/fsrcnn_weights.h
```

**PENTING soal resolusi & NPU**: berbeda dari CPU (fully-convolutional,
menerima H×W berapa pun tanpa build ulang), backend NPU RKNN meng-compile
graph untuk SATU shape tetap yang ditentukan SAAT KONVERSI
(`--luma-h/--luma-w` di atas). Kalau video Anda beresolusi Y lain, ulangi
KEDUA langkah `build_fsrcnn_onnx.py` dan `convert_to_rknn.py` dengan angka
`--height/--width` (resp. `--luma-h/--luma-w`) yang sama dan sesuai. U/V
tidak perlu dikonversi ke NPU sama sekali karena tidak pernah melewatinya.

Default `144x176` mengikuti resolusi QCIF yang di-hardcode di
`source_channel_pixel.c` (`inRows=144, inCols=176`) -- cocok untuk
reproduksi/verifikasi langsung terhadap program referensi tersebut.

## 2. Build C

```bash
make cpu-native          # build & jalankan di mesin dev Anda (x86/ARM apa saja)
make cpu-aarch64         # cross-compile utk Orange Pi 5 (CPU/NEON)
make npu-aarch64 RKNPU2_DIR=/path/ke/sdk/rknpu2   # lihat bagian 3 di bawah
make gpu-native           # build & jalankan di mesin dev Anda (butuh libOpenCL)
make gpu-aarch64          # cross-compile utk Orange Pi 5 (GPU Mali) -- lihat bagian 3b
```

Jalankan:
```bash
# CPU
bin/fsrcnn_cpu_native  in.yuv <W> <H> out.yuv [--pin-big] [--max-frames N]
bin/fsrcnn_cpu_aarch64 in.yuv <W> <H> out.yuv [--pin-big] [--max-frames N]

# NPU (di board Orange Pi 5, W/H harus persis sama dgn --luma-h/--luma-w saat konversi)
bin/fsrcnn_npu_aarch64 fsrcnn_rk3588.rknn in.yuv <W> <H> out.yuv [--max-frames N]

# GPU (fully-convolutional spt CPU -- <W> <H> BOLEH resolusi berapa pun, tidak
# terikat shape spt NPU; auto-pilih device Mali kalau ada, lihat bagian 3b)
bin/fsrcnn_gpu_aarch64 in.yuv <W> <H> out.yuv [--max-frames N]
```
`<W> <H>` adalah dimensi PLANE Y (luma) video sumber; dimensi chroma
dihitung otomatis (`ceil(H/2) x ceil(W/2)`, standar 4:2:0). Hanya Y yang
dikirim ke FSRCNN (CPU, NPU, maupun GPU); U/V diproses di CPU lewat
`chroma_upsample_nn()` pada ketiga binary.

`--pin-big` (khusus CPU): pin thread ke core "big" (Cortex-A76) lewat
deteksi frekuensi maksimum di sysfs -- lihat `src/cpu_affinity.c`. Ini
contoh langsung dari kendala arsitektur asimetris: kalau scheduler Linux
membiarkan thread compute-bound ini "mengembara" ke core A55 (LITTLE),
throughput turun signifikan meski jumlah instruksi yang dieksekusi sama --
sebab-akibatnya murni soal *di core mana* instruksi itu dijalankan.

## 3. Dependency NPU (RKNN SDK) -- TIDAK ikut di-vendor di sini

`fsrcnn_npu.c` membutuhkan `rknn_api.h` dan `librknnrt.so` dari RKNN SDK
resmi Rockchip. SDK ini berlisensi **proprietary** (RKNN SDK License, bukan
open-source) sehingga sengaja TIDAK disertakan dalam paket file ini. Ambil
dari repo resmi:

```
https://github.com/airockchip/rknn-toolkit2
  -> rknpu2/runtime/Linux/librknn_api/include/rknn_api.h
  -> rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so
```

Taruh sebagai `third_party/rknpu2/include/rknn_api.h` dan
`third_party/rknpu2/lib/aarch64/librknnrt.so`, lalu `make npu-aarch64`.
**Prioritaskan** file `librknnrt.so` yang SUDAH TERPASANG di image OS board
Orange Pi 5 Anda (biasanya `/usr/lib/librknnrt.so`) dibanding versi acak dari
GitHub -- versi runtime harus cocok dengan versi driver kernel NPU yang
aktif, kalau tidak `rknn_init()` bisa gagal atau berperilaku tidak terduga.

Kode `fsrcnn_npu.c` sudah diverifikasi **compile + link bersih** (tanpa
warning, tanpa undefined symbol) terhadap header & `librknnrt.so` aarch64
resmi memakai cross-compiler `aarch64-linux-gnu-gcc`. Eksekusi sesungguhnya
tidak bisa diuji di sini karena butuh driver kernel NPU RK3588 fisik --
WAJIB diuji langsung di board Orange Pi 5 Anda sebelum dipakai produksi
(lihat bagian 5).

## 3b. Dependency GPU (OpenCL) -- JAUH lebih ringan dari RKNN SDK

Berbeda dari NPU, dependency GPU **sebagian besar open-source**:

- `CL/cl.h` (header Khronos OpenCL) -- paket `opencl-headers`.
- `libOpenCL.so` (ICD loader, memuat driver vendor apa pun yang terdaftar
  di `/etc/OpenCL/vendors/`) -- paket `ocl-icd-opencl-dev` (dev) atau
  `ocl-icd-libopencl1` (runtime saja).

```bash
sudo apt install opencl-headers ocl-icd-opencl-dev
```

Yang TETAP proprietary & TIDAK di-vendor di sini hanyalah **driver Mali
itu sendiri** (mis. `libmali-valhall-g610-*`) -- implementasi OpenCL
sesungguhnya utk GPU Mali-G610 fisik. Ini BIASANYA SUDAH TERPASANG di image
OS Orange Pi 5 varian "Mali"/desktop dari komunitas (Joshua-Riek
Ubuntu-Rockchip, Armbian dgn paket mali, dll) -- TIDAK ADA di image
server/minimal biasa. Cek dgn:
```bash
clinfo -l
```
Kalau kosong atau cuma menunjukkan device CPU, install driver Mali sesuai
image OS Anda (BUKAN Panfrost/mesa -- driver open-source Panfrost belum
punya implementasi OpenCL yang matang per proyek ini ditulis) sebelum
memakai `fsrcnn_gpu_aarch64`.

**Fallback berguna utk pengembangan tanpa GPU fisik**: paket `pocl-opencl-icd`
(implementasi OpenCL open-source yang jalan di CPU) membuat `fsrcnn_gpu_*`
tetap BISA di-build & dijalankan tanpa GPU vendor apa pun -- dipakai persis
dgn cara ini utk memverifikasi KERNEL OpenCL proyek ini (lihat bagian 4a).
`fsrcnn_gpu_init()` otomatis mendeteksi & memilih device: prioritas GPU
bernama "Mali", lalu GPU apa pun, baru fallback ke device pertama yang
ditemukan (dgn peringatan eksplisit di stderr).

**Cross-compile `gpu-aarch64` dari mesin x86**: BUTUH `libOpenCL.so` &
`CL/cl.h` utk aarch64 di sysroot cross-compiler Anda -- pada banyak setup
ini TIDAK tersedia begitu saja (beda dari header CPU standar). Kalau
`make gpu-aarch64` gagal linking `-lOpenCL`, cara paling praktis (SAMA
seperti pola yang sudah terbukti berhasil utk NPU pada proyek ini) adalah
build LANGSUNG DI board:
```bash
sudo apt install opencl-headers ocl-icd-opencl-dev
make gpu-aarch64 CROSS_CC=gcc   # native gcc board = aarch64, tak perlu cross-toolchain
```

## 4. Verifikasi & benchmark

### 4a. Verifikasi terhadap ground truth `source_channel_pixel.c` (dilakukan di sesi ini)

Metodologi: kompilasi `source_channel_pixel.c` apa adanya (`gcc -O2 -fopenmp
-lm`), jalankan pada file YUV 4:2:0 QCIF (176x144) sintetis 150 frame acak
sebagai **ground truth**, lalu bandingkan bit-demi-bit terhadap keluaran
`bin/fsrcnn_cpu_native` pada input yang SAMA:

| Jalur                         | Plane | Hasil                                            |
|-------------------------------|-------|---------------------------------------------------|
| `bin/fsrcnn_cpu_native` (skalar+NEON, x86 native) | Y | maks selisih **1/255** dari 15.206.400 piksel (150 frame), 0 piksel berselisih >1 |
| `bin/fsrcnn_cpu_native`        | U, V  | **bit-exact (selisih 0)** -- replikasi murni, tanpa floating point |
| `fsrcnn.onnx` via onnxruntime  | Y     | maks selisih **1/255** |
| `fsrcnn_rk3588.rknn` via simulator RKNN (`rknn.init_runtime()` tanpa target fisik) | Y | maks selisih **1/255** |
| `bin/fsrcnn_cpu_aarch64` (cross-compiled, dijalankan via `qemu-aarch64`) vs native x86 | Y | selisih 1 pada segelintir piksel (reasosiasi SIMD NEON) |
| `bin/fsrcnn_gpu_native` (kernel OpenCL SESUNGGUHNYA, via `pocl-opencl-icd` sbg device pengganti Mali fisik) vs ground truth | Y | maks selisih **1/255** dari 15.206.400 piksel (150 frame), 0 piksel berselisih >1 |
| `bin/fsrcnn_gpu_native` (pocl) vs `bin/fsrcnn_cpu_native` | Y | maks selisih **1/255** (5 frame) |

**Catatan penting soal jalur GPU**: verifikasi di atas menjalankan KERNEL
OpenCL YANG PERSIS SAMA yang akan jalan di Mali-G610 (source string di
`fsrcnn_gpu.c` tidak diubah sama sekali), hanya di-eksekusi lewat device
CPU (`pocl`) krn tidak ada GPU fisik di lingkungan pengembangan ini -- BUKAN
simulasi/pemodelan matematis terpisah spt pada verifikasi RKNN (bagian NPU),
jadi ini verifikasi lebih kuat thd LOGIKA kernel (derivasi gather utk
deconv, clamp-addressing utk padding -- lihat komentar derivasi panjang di
`fsrcnn_gpu.c`) dibanding yang bisa dicapai utk jalur NPU dari lingkungan
pengembangan ini. TAPI ini TETAP BUKAN pengganti uji di Mali fisik: device
GPU sungguhan bisa punya batasan work-group-size, dukungan ekstensi, atau
perilaku driver berbeda dari pocl -- lihat bagian 5.

Selisih 1/255 (~0,0039 pada skala uint8) ini **BUKAN bug** -- ini adalah
galat reasosiasi floating-point (urutan penjumlahan berbeda antar
implementasi menghasilkan bit terakhir mantissa yang berbeda), fenomena yang
secara eksplisit didokumentasikan pada komentar header
`source_channel_pixel.c` sendiri sebagai perilaku yang diharapkan, dan jauh
di bawah step kuantisasi uint8 (1/255 ≈ 3,9×10⁻³). Untuk mereproduksi:

```bash
# 1) kompilasi & jalankan program referensi Anda sbg ground truth
gcc -O2 -fopenmp -Wall source_channel_pixel.c -o ref_bin -lm
python3 tools/gen_test_yuv.py test_in.yuv --width 176 --height 144 --frames 150
./ref_bin   # sesuaikan I/O program referensi Anda dgn test_in.yuv -> ref_out.yuv

# 2) jalankan jalur kita pada input yang SAMA
bin/fsrcnn_cpu_native test_in.yuv 176 144 our_out.yuv

# 3) bandingkan bit-demi-bit (Y toleransi 1, U/V harus 0) -- lihat skrip
#    perbandingan biner sederhana atau tools/verify_against_onnx.py di bawah
```

### 4b. Verifikasi cepat terhadap onnxruntime (regresi harian)

```bash
python3 tools/gen_test_yuv.py test_in.yuv --width 176 --height 144 --frames 3
bin/fsrcnn_cpu_native test_in.yuv 176 144 test_out.yuv
python3 tools/verify_against_onnx.py --in-yuv test_in.yuv --out-yuv test_out.yuv \
    --onnx fsrcnn.onnx --width 176 --height 144 --frames 3 --tolerance 1
```
Skrip ini membandingkan plane Y terhadap `fsrcnn.onnx` (onnxruntime, toleransi
1/255 wajar akibat reasosiasi float) dan plane U/V terhadap replikasi
nearest-neighbor murni (harus SELALU bit-exact/0, tanpa toleransi).

### 4c. Benchmark CPU vs NPU vs GPU (di board fisik)
```bash
bin/fsrcnn_cpu_aarch64 --pin-big in.yuv <W> <H> out_cpu.yuv   # catat fps dari stderr
bin/fsrcnn_npu_aarch64 fsrcnn_rk3588.rknn in.yuv <W> <H> out_npu.yuv   # catat fps dari stderr
bin/fsrcnn_gpu_aarch64 in.yuv <W> <H> out_gpu.yuv   # catat fps dari stderr; device yg dipakai dicetak di baris pertama
```
Baris pertama stderr `fsrcnn_gpu_aarch64` mencetak nama device OpenCL yang
dipakai (mis. `"Mali-G610"`) -- SELALU periksa baris ini utk memastikan
benar-benar jalan di GPU, bukan fallback CPU (lihat peringatan di bagian
3b kalau driver Mali belum terpasang).

**Hasil terukur di board fisik (Orange Pi 5, RK3588, QCIF 176x144 ->
352x288, 150 frame, `suzie_qcif.yuv`)**:

| Jalur | fps (steady-state) | Device terpakai |
|-------|---------------------|------------------|
| CPU (`fsrcnn_cpu_aarch64`, single-thread, tanpa `--pin-big`) | ~1,43-1,48 fps | CPU (baseline) |
| NPU (`fsrcnn_npu_aarch64`) | ~28-32 fps | NPU RK3588 |
| GPU (`fsrcnn_gpu_aarch64`) | ~7,7-8,0 fps | `"Mali-LODX r0p0"` (dikonfirmasi via `clinfo -l` & baris stderr) |

Ini mengonfirmasi ketiga jalur **berjalan bersih** di board fisik (tidak
crash, NPU/GPU memilih device yang benar). fps di atas **BELUM** berarti
keluarannya BENAR secara numerik -- lihat 4d utk menutup loop verifikasi
itu, dan bagian 5 utk status verifikasi per jalur.

### 4d. Verifikasi numerik langsung di board fisik (`tools/yuv_diff.c`)

fps saja tidak membuktikan korektnes -- exit-code sukses `rknn_inputs_set()`
bahkan pernah "berhasil" padahal internalnya gagal (lihat bug #2 di bagian
5). Untuk menutup loop verifikasi TANPA perlu Python/numpy di board, dipakai
`tools/yuv_diff.c`: utilitas C murni (tanpa dependency eksternal) yang
membandingkan dua file `.yuv` byte-demi-byte per frame, dgn toleransi 1/255
utk plane Y (reasosiasi floating-point, lihat bagian 4a) dan toleransi 0 utk
plane U/V (replikasi nearest-neighbor murni, tanpa floating point).

Build (di board, native gcc -- tidak butuh cross-compiler, sama spt
`gpu-aarch64 CROSS_CC=gcc`):
```bash
make yuv-diff CC=gcc
# atau langsung tanpa make sama sekali (tidak ada dependency):
gcc -O2 -Wall -Wextra -o bin/yuv_diff tools/yuv_diff.c
```

Pemakaian -- setelah menjalankan ketiga engine pada INPUT YANG SAMA (lihat
4c), bandingkan berpasangan pada resolusi KELUARAN (`Wout Hout`, bukan
resolusi input; utk QCIF 176x144 dgn scale=2 berarti `352 288`):
```bash
bin/yuv_diff susi_aarch64.yuv susi_npu.yuv 352 288   # CPU vs NPU
bin/yuv_diff susi_aarch64.yuv susi_gpu.yuv 352 288   # CPU vs GPU
bin/yuv_diff susi_npu.yuv     susi_gpu.yuv 352 288   # NPU vs GPU
```
Opsional: `--frames N` (batasi jumlah frame yg dibandingkan, default: auto
dari file terkecil), `--tol-y N` (default 1). Exit code `0` = OK/dalam
toleransi, `1` = GAGAL (ada piksel Y berselisih >toleransi, ATAU ada piksel
U/V berselisih sama sekali, ATAU file tidak terbaca/ukuran tidak cocok).

Diuji di sandbox pengembangan ini (bukan di board, krn tidak ada board fisik
di sini) memakai keluaran `fsrcnn_gpu_native` (jalur `pocl`) vs ground truth
`source_channel_pixel.c` (150 frame, 352x288): hasil `Y=1 (toleransi 1)
U/V=0 (toleransi 0) -- OK`, cocok persis dgn angka yang sudah dilaporkan di
bagian 4a lewat skrip Python -- mengonfirmasi `yuv_diff.c` menghitung
selisih yang sama dgn `verify_against_onnx.py`, hanya lebih ringan (tanpa
Python/numpy) utk dipakai langsung di board. Juga diuji: file identik (harus
selisih 0 di semua plane), file tidak ditemukan (harus gagal dgn pesan
jelas, exit 1), serta bersih dari `valgrind --leak-check=full` (0 leak, 0
error).

**Ekspektasi hasil di board fisik**: kalau NPU dan GPU benar-benar sudah
lolos verifikasi numerik (bukan cuma "jalan tanpa crash"), ketiga
perbandingan di atas harus melaporkan `OK: dalam toleransi` dengan Y maks
selisih ≤1 dan U/V maks selisih =0 -- persis pola yang sudah dikonfirmasi
utk setiap jalur lain di proyek ini (bagian 4a). Kalau salah satu
perbandingan GAGAL (terutama plane U/V yang seharusnya SELALU 0 krn murni
replikasi tanpa floating point), itu indikasi kuat ada bug numerik yang
BELUM terlihat dari sekadar fps -- lihat bagian 5 sebelum menganggap jalur
tsb siap dipakai utk eksperimen/publikasi.

## 5. Hal yang WAJIB diverifikasi ulang oleh peneliti (belum bisa dicek dari sini)

1. **Eksekusi NPU nyata**: kode sudah link-clean terhadap RKNN C API resmi
   DAN sudah diverifikasi lewat simulator RKNN (`rknn.init_runtime()` tanpa
   target fisik, selisih 1/255 vs ground truth). Uji di board fisik Orange
   Pi 5 sempat menemukan 2 bug runtime di `fsrcnn_npu.c`, KEDUANYA sudah
   diperbaiki (bukan masalah pada `fsrcnn_rk3588.rknn`):
   - `rknn_set_input_shapes()` dipanggil tanpa syarat (warisan desain
     `dynamic_input` lama) padahal model sekarang shape TETAP/statis --
     runtime menolak eksplisit ("rknn model is static shape type"). Fix:
     init HANYA *query* (baca) shape tetap model; forward pass memvalidasi
     dimensi input terhadap hasil query itu, tanpa pernah memanggil
     `rknn_set_input_shapes()`.
   - `rknn_inputs_set()` di-hardcode `fmt=RKNN_TENSOR_NCHW`, padahal model
     ini ternyata di-compile RKNN dgn layout native NHWC (`rknn_query`
     melaporkan `fmt=NHWC`) -- mismatch ini memicu error internal driver
     "Meet unsupported src layout for normalize", DAN pada sebagian kasus
     `rknn_inputs_set()` tetap mengembalikan sukses meski internal-nya
     gagal (silent-wrong-result, bukan crash bersih). Fix: `fmt` yang
     dikirim SELALU disamakan dgn fmt native hasil query di init --
     aman krn model 1-channel (NCHW & NHWC byte-identical utk C=1).
   **Status terkini**: kedua fix di atas sudah dikonfirmasi jalan bersih
   di board fisik (`fsrcnn_npu_aarch64` menghasilkan `[frame N] ... fps`
   selama 150 frame tanpa error `E RKNN`, ~28-32 fps steady-state -- lihat
   tabel di 4c). **TAPI ini baru mengonfirmasi EKSEKUSI, BUKAN korektnes
   numerik** -- ingat bug #2 di atas: `rknn_inputs_set()` sempat
   mengembalikan sukses PADAHAL internalnya gagal, jadi "jalan tanpa error"
   saja tidak cukup. **Masih WAJIB** membandingkan `susi_npu.yuv` thd
   `susi_aarch64.yuv` (CPU, dari input yang SAMA) memakai `bin/yuv_diff`
   (bagian 4d) sebelum dipakai sbg bagian eksperimen/publikasi -- termasuk
   mengonfirmasi tidak ada penurunan presisi tambahan akibat kuantisasi
   FLOAT16 internal NPU (di luar reasosiasi float32 yang sudah diukur di
   simulator).
2. **Resolusi model `.rknn` yang disertakan** (`fsrcnn_rk3588.rknn`, dibangun
   untuk Y=144x176/QCIF) HARUS DIBANGUN ULANG (`scripts/convert_to_rknn.py
   --luma-h/--luma-w`) kalau video riset Anda beresolusi lain -- satu file
   `.rknn` hanya valid untuk SATU shape.
3. **Throughput CPU** yang terukur di sesi ini (~3,2 fps untuk 176x144->
   352x288, single-thread, mesin x86 dev container) HANYA baseline
   fungsional -- BUKAN representasi performa RK3588 sesungguhnya (beda ISA,
   beda clock, beda cache). Ukur ulang di board fisik dengan `--pin-big` utk
   angka yang bisa dipakai di publikasi.
4. **Eksekusi GPU nyata**: kode `fsrcnn_gpu.c` sudah diverifikasi numerik
   KUAT lewat `pocl` (kernel OpenCL SESUNGGUHNYA, bukan simulasi terpisah --
   lihat catatan di bagian 4a) DAN sudah diverifikasi bebas leak/error
   `clRelease*` lewat `valgrind` (0 leak dari kode proyek ini; leak kecil
   yg terdeteksi murni dari internal `libpocl`/`dlopen`/thread-pool, BUKAN
   dari `fsrcnn_gpu.c`). **Status terkini**: sudah dikonfirmasi jalan di
   GPU Mali-G610 fisik -- `clinfo -l` di board menunjukkan device asli
   `"Mali-LODX r0p0"` (platform `ARM Platform`/libmali, BUKAN `Clover`), dan
   `fsrcnn_gpu_aarch64` (dibangun via `make gpu-aarch64 CROSS_CC=gcc`
   langsung di board -- lihat catatan cross-compile di bagian 3b) mencetak
   `[gpu] dipakai: "Mali-LODX r0p0"` lalu berjalan 150 frame bersih di
   ~7,7-8,0 fps steady-state (lihat tabel di 4c). **TAPI ini baru
   mengonfirmasi EKSEKUSI, BUKAN korektnes numerik pada silikon fisik** --
   driver Mali sungguhan bisa saja berperilaku beda dari `pocl` (mis.
   `-cl-fast-relaxed-math`, penanganan subnormal, batasan
   `CL_DEVICE_MAX_WORK_GROUP_SIZE`/dukungan tipe data) meski logika kernel
   (derivasi gather + clamp-addressing) sudah diverifikasi benar scr
   matematis. **Masih WAJIB** membandingkan `susi_gpu.yuv` thd
   `susi_aarch64.yuv` (CPU) memakai `bin/yuv_diff` (bagian 4d) sebelum
   dipakai sbg bagian eksperimen/publikasi.
