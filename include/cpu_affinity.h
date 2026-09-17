/*
 * cpu_affinity.h -- deteksi & pinning core "big" pada CPU heterogen
 * (big.LITTLE), khususnya RK3588: 4x Cortex-A55 (LITTLE) + 4x Cortex-A76
 * (big). Alih-alih HARDCODE indeks core (rawan salah antar board/kernel),
 * deteksi dilakukan lewat scaling_max_freq di sysfs -- core dgn frekuensi
 * maksimum lebih tinggi diasumsikan core "big". Ini konsisten dengan
 * prinsip kendala hardware asimetris: thread compute-bound (inference CNN)
 * harus dipin ke core besar agar tidak "diacak" scheduler ke core LITTLE
 * yang jauh lebih lambat untuk beban floating-point berat.
 */
#ifndef CPU_AFFINITY_H
#define CPU_AFFINITY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Isi big_cpus[] (kapasitas max_count) dengan ID core "big" (frekuensi max
 * tertinggi). Return jumlah core big yang ditemukan (0 jika deteksi gagal,
 * mis. bukan Linux/tanpa sysfs cpufreq -- caller sebaiknya fallback ke
 * "tidak melakukan pinning" dalam kasus itu). */
int cpu_affinity_detect_big_cores(int *big_cpus, int max_count);

/* Pin thread pemanggil (saat ini) ke himpunan core yang diberikan.
 * Return 0 sukses, -1 gagal (mis. platform tidak mendukung sched_setaffinity). */
int cpu_affinity_pin_to(const int *cpus, int count);

/* Kombinasi praktis: deteksi core big lalu pin thread saat ini ke situ.
 * Return jumlah core yang berhasil dipin (0 = gagal/tidak didukung). */
int cpu_affinity_pin_to_big_cores(void);

#ifdef __cplusplus
}
#endif

#endif /* CPU_AFFINITY_H */
