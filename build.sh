#!/usr/bin/env bash
# fix_npu_build.sh -- diagnosa & siapkan third_party/rknpu2/ agar cocok
# dengan konvensi path yang diasumsikan Makefile ($(RKNPU2_DIR)/include/rknn_api.h
# dan $(RKNPU2_DIR)/lib/aarch64/librknnrt.so), APAPUN struktur asli SDK Anda.
# Jalankan dari dalam folder fsrcnn_npu_c di board.
set -e

echo "=== 1) Cek toolchain (native vs cross) ==="
echo "Anda sedang compile LANGSUNG DI board RK3588 (aarch64), bukan cross-compile"
echo "dari mesin x86 -- jadi cross-compiler aarch64-linux-gnu-gcc mungkin memang"
echo "TIDAK terpasang (dan tidak perlu). gcc native di board ini SUDAH aarch64."
CROSS_OK=0
if command -v aarch64-linux-gnu-gcc >/dev/null 2>&1; then
  echo "aarch64-linux-gnu-gcc DITEMUKAN: $(command -v aarch64-linux-gnu-gcc)"
  CROSS_OK=1
else
  echo "aarch64-linux-gnu-gcc TIDAK ada -- akan pakai gcc native (CROSS_CC=gcc)."
fi
echo "gcc native: $(command -v gcc || echo 'TIDAK ADA -- install dulu: sudo apt install gcc')"

echo
echo "=== 2) Cari rknn_api.h ==="
HDR=$(find "$HOME/rknn-toolkit2" -iname 'rknn_api.h' 2>/dev/null | head -1)
if [ -z "$HDR" ]; then
  HDR=$(find "$PWD/include" -iname 'rknn_api.h' 2>/dev/null | head -1)
fi
echo "Dipakai: ${HDR:-TIDAK KETEMU}"
[ -n "$HDR" ] || { echo "GAGAL: rknn_api.h tidak ditemukan di ~/rknn-toolkit2 atau ./include"; exit 1; }

echo
echo "=== 3) Cari librknnrt.so (prioritas: yang SUDAH TERPASANG di OS board) ==="
SOLIB=$(find /usr/lib /usr/local/lib /lib -iname 'librknnrt.so*' 2>/dev/null | head -1)
SOURCE_DESC="sistem (image OS board -- prioritas, cocok versi driver kernel)"
if [ -z "$SOLIB" ]; then
  SOLIB=$(find "$HOME/rknn-toolkit2" -iname 'librknnrt.so*' 2>/dev/null | grep -i aarch64 | head -1)
  SOURCE_DESC="repo rknn-toolkit2 (fallback -- pastikan versi cocok driver kernel Anda)"
fi
if [ -z "$SOLIB" ]; then
  SOLIB=$(find "$PWD/lib" -iname 'librknnrt.so*' 2>/dev/null | head -1)
  SOURCE_DESC="folder lib/ proyek ini (fallback terakhir)"
fi
echo "Dipakai: ${SOLIB:-TIDAK KETEMU}  (sumber: $SOURCE_DESC)"
[ -n "$SOLIB" ] || { echo "GAGAL: librknnrt.so tidak ditemukan di /usr/lib, ~/rknn-toolkit2, atau ./lib"; exit 1; }

echo
echo "=== 4) Susun third_party/rknpu2/ sesuai konvensi Makefile ==="
mkdir -p third_party/rknpu2/include third_party/rknpu2/lib/aarch64
ln -sf "$HDR"   third_party/rknpu2/include/rknn_api.h
ln -sf "$SOLIB" third_party/rknpu2/lib/aarch64/librknnrt.so
ls -la third_party/rknpu2/include third_party/rknpu2/lib/aarch64

echo
echo "=== 5) Build ==="
if [ "$CROSS_OK" = "1" ]; then
  make npu-aarch64 RKNPU2_DIR=third_party/rknpu2
else
  make npu-aarch64 RKNPU2_DIR=third_party/rknpu2 CROSS_CC=gcc
fi

echo
echo "=== SELESAI: bin/fsrcnn_npu_aarch64 ==="
ls -la bin/fsrcnn_npu_aarch64
ldd bin/fsrcnn_npu_aarch64 | grep -i rknn
