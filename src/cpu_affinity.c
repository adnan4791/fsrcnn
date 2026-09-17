#define _GNU_SOURCE
#include "cpu_affinity.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#if defined(__linux__)
#include <sched.h>
#define CPU_AFFINITY_LINUX 1
#endif

#define MAX_CPUS 32

static long read_max_freq_khz(int cpu) {
    char path[128];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    long v = -1;
    if (fscanf(f, "%ld", &v) != 1) v = -1;
    fclose(f);
    return v;
}

int cpu_affinity_detect_big_cores(int *big_cpus, int max_count) {
    long freqs[MAX_CPUS];
    int n = 0;
    long max_freq = 0;

    long ncpu = sysconf(_SC_NPROCESSORS_CONF);
    if (ncpu <= 0 || ncpu > MAX_CPUS) return 0;

    for (int c = 0; c < ncpu; c++) {
        freqs[c] = read_max_freq_khz(c);
        if (freqs[c] > max_freq) max_freq = freqs[c];
        n = (int)ncpu;
    }
    if (max_freq <= 0) return 0; /* sysfs cpufreq tidak tersedia (mis. bukan target ARM) */

    /* Core "big" = core dengan cpuinfo_max_freq dalam 90% dari frekuensi
     * tertinggi yang terdeteksi di sistem (toleransi kecil terhadap
     * perbedaan minor antar-core dalam cluster yang sama). */
    long threshold = (max_freq * 9) / 10;
    int found = 0;
    for (int c = 0; c < n && found < max_count; c++) {
        if (freqs[c] >= threshold) {
            big_cpus[found++] = c;
        }
    }
    return found;
}

int cpu_affinity_pin_to(const int *cpus, int count) {
#ifdef CPU_AFFINITY_LINUX
    if (count <= 0) return -1;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < count; i++) CPU_SET(cpus[i], &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) return -1;
    return 0;
#else
    (void)cpus; (void)count;
    return -1; /* tidak didukung di luar Linux */
#endif
}

int cpu_affinity_pin_to_big_cores(void) {
    int big[MAX_CPUS];
    int n = cpu_affinity_detect_big_cores(big, MAX_CPUS);
    if (n <= 0) return 0;
    if (cpu_affinity_pin_to(big, n) != 0) return 0;
    return n;
}
