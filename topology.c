/* topology.c — hardware bandwidth prober for BREAD inference engine.
 *
 * Measures:
 *   1. RAM→VRAM bandwidth via cudaMemcpyAsync (64 MB, 256 MB, 512 MB, 1 GB)
 *   2. SSD→RAM bandwidth via sequential 64 MB fread of the model blob
 *
 * Analyses the GGUF model to derive scheduler decisions:
 *   - Does the full model fit in RAM?
 *   - Which weights pin permanently in VRAM (non-expert)?
 *   - How do expert weights get streamed (RAM→VRAM per layer)?
 *
 * C99 + CUDA runtime API.  Compile with nvcc.
 */

#ifdef _WIN32
#  define _CRT_SECURE_NO_WARNINGS
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "gguf.h"
#include "topology.h"

/* ------------------------------------------------------------------ */
/* CUDA error checking                                                  */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _err = (call);                                         \
        if (_err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_err));         \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

/* ------------------------------------------------------------------ */
/* High-resolution wall-clock timer                                    */
/* ------------------------------------------------------------------ */

static double wall_sec(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/* ------------------------------------------------------------------ */
/* System RAM query (Windows)                                          */
/* ------------------------------------------------------------------ */

static uint64_t query_ram_total_mb(void) {
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms))
        return ms.ullTotalPhys / (1024 * 1024);
#endif
    return 0;   /* fallback: unknown */
}

/* ------------------------------------------------------------------ */
/* RAM → VRAM benchmark (single transfer size)                        */
/* ------------------------------------------------------------------ */

static double bench_h2d_once(size_t nbytes) {
    void *h_buf = NULL, *d_buf = NULL;
    cudaStream_t stream;
    cudaEvent_t  ev0, ev1;

    CUDA_CHECK(cudaMallocHost(&h_buf, nbytes));
    CUDA_CHECK(cudaMalloc    (&d_buf, nbytes));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    /* Touch every page of the host buffer to commit physical memory */
    memset(h_buf, 0xAB, nbytes);

    /* Warmup — ensures driver and PCIe link are fully spun up */
    CUDA_CHECK(cudaMemcpyAsync(d_buf, h_buf, nbytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* Timed runs */
    const int REPS = 5;
    CUDA_CHECK(cudaEventRecord(ev0, stream));
    for (int i = 0; i < REPS; i++)
        CUDA_CHECK(cudaMemcpyAsync(d_buf, h_buf, nbytes,
                                   cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(ev1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    double bw_gbs = (double)nbytes * REPS / (ms * 1e-3) / 1e9;

    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFreeHost(h_buf));

    return bw_gbs;
}

/* ------------------------------------------------------------------ */
/* RAM → VRAM: sweep transfer sizes, return peak bandwidth            */
/* ------------------------------------------------------------------ */

static double probe_ram_to_vram_bw(uint64_t vram_free_mb) {
    static const size_t SIZES[] = {
        64ULL  * 1024 * 1024,
        256ULL * 1024 * 1024,
        512ULL * 1024 * 1024,
        1024ULL* 1024 * 1024,
    };
    static const char *LABELS[] = { "  64 MB", " 256 MB", " 512 MB", "1024 MB" };

    uint64_t free_bytes = vram_free_mb * 1024 * 1024;
    double   peak = 0.0;

    printf("\n  RAM → VRAM bandwidth (pinned cudaMemcpyAsync, 5 reps each):\n");

    for (size_t i = 0; i < sizeof(SIZES) / sizeof(SIZES[0]); i++) {
        size_t sz = SIZES[i];
        if (sz > free_bytes * 9 / 10) {
            printf("    %s : skipped (not enough free VRAM)\n", LABELS[i]);
            continue;
        }
        double bw = bench_h2d_once(sz);
        printf("    %s : %6.2f GB/s\n", LABELS[i], bw);
        if (bw > peak) peak = bw;
    }

    printf("    peak     : %6.2f GB/s\n", peak);
    return peak;
}

/* ------------------------------------------------------------------ */
/* SSD → RAM: sequential fread of model blob, report sustained GB/s   */
/* ------------------------------------------------------------------ */

static double probe_ssd_to_ram_bw(const char *path) {
    const size_t   CHUNK    = 64ULL * 1024 * 1024;   /* 64 MB per read */
    const uint64_t CAP      = 4ULL  * 1024 * 1024 * 1024;  /* read up to 4 GB */

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "  SSD probe: cannot open '%s'\n", path);
        return 0.0;
    }

    void *buf = malloc(CHUNK);
    if (!buf) {
        fclose(fp);
        fprintf(stderr, "  SSD probe: out of memory\n");
        return 0.0;
    }

    uint64_t total_bytes = 0;
    size_t   n;

    printf("\n  SSD → RAM bandwidth (64 MB chunks, up to 4 GB):\n");
    printf("    reading...");
    fflush(stdout);

    double t0 = wall_sec();

    while (total_bytes < CAP && (n = fread(buf, 1, CHUNK, fp)) > 0) {
        total_bytes += n;
        if (n < CHUNK) break;   /* short read = EOF */
    }

    double elapsed = wall_sec() - t0;
    free(buf);
    fclose(fp);

    double bw_gbs = (double)total_bytes / elapsed / 1e9;
    printf(" read %.2f GiB in %.2f s = %6.2f GB/s",
           total_bytes / (double)(1ULL << 30), elapsed, bw_gbs);
    if (total_bytes < CAP * 9 / 10)
        printf(" (file smaller than 4 GB cap)");
    else
        printf(" (note: OS page cache may inflate repeated runs)");
    printf("\n");

    return bw_gbs;
}

/* ------------------------------------------------------------------ */
/* Model analysis: separate expert vs non-expert bytes                */
/* ------------------------------------------------------------------ */

static void analyze_model(const char *path, topo_t *t) {
    gguf_ctx_t *ctx = gguf_open(path);
    if (!ctx) {
        fprintf(stderr, "topology: gguf_open failed\n");
        return;
    }

    uint64_t n = gguf_num_tensors(ctx);

    t->model_total_bytes     = 0;
    t->model_expert_bytes    = 0;
    t->model_nonexpert_bytes = 0;
    t->n_moe_layers          = 0;
    t->layer_expert_bytes    = 0;

    /* Accumulate per-blk.0 expert sizes to compute per-layer budget */
    uint64_t blk0_expert_bytes = 0;

    for (uint64_t i = 0; i < n; i++) {
        const gguf_tensor_t *ten = gguf_tensor_by_idx(ctx, i);
        t->model_total_bytes += ten->size;

        if (strstr(ten->name, "_exps")) {
            t->model_expert_bytes += ten->size;
            /* Count MoE layers by counting ffn_gate_exps tensors */
            if (strstr(ten->name, "ffn_gate_exps"))
                t->n_moe_layers++;
            /* Capture per-layer sizes from blk.0 */
            if (strncmp(ten->name, "blk.0.", 6) == 0)
                blk0_expert_bytes += ten->size;
        } else {
            t->model_nonexpert_bytes += ten->size;
        }
    }

    /* blk.0 has gate+up+down expert tensors, each layer is the same size */
    t->layer_expert_bytes = blk0_expert_bytes;

    gguf_close(ctx);
}

/* ------------------------------------------------------------------ */
/* Formatting helpers                                                   */
/* ------------------------------------------------------------------ */

static void fmt_gib(char *buf, size_t buflen, uint64_t bytes) {
    if      (bytes >= (1ULL << 30))
        snprintf(buf, buflen, "%.2f GiB", bytes / (double)(1ULL << 30));
    else if (bytes >= (1ULL << 20))
        snprintf(buf, buflen, "%.2f MiB", bytes / (double)(1ULL << 20));
    else
        snprintf(buf, buflen, "%llu KiB", (unsigned long long)(bytes >> 10));
}

/* ------------------------------------------------------------------ */
/* Scheduler decision logic                                            */
/* ------------------------------------------------------------------ */

static void print_scheduler(const topo_t *t, const char *model_path) {
    uint64_t vram_bytes     = t->vram_total_mb * 1024 * 1024;
    uint64_t vram_free      = t->vram_free_mb  * 1024 * 1024;
    uint64_t ram_bytes      = t->ram_total_mb  * 1024 * 1024;

    /* Reserve 15% of VRAM for KV cache + activations + CUDA overhead */
    uint64_t vram_usable    = vram_free * 85 / 100;

    /* Model fits in RAM? */
    int in_ram  = (t->model_total_bytes <= ram_bytes  * 90 / 100);
    /* Non-expert weights fit permanently in VRAM? */
    int nexp_in_vram = (t->model_nonexpert_bytes <= vram_usable);

    /* Double-buffer two layers of experts simultaneously */
    uint64_t double_buf_bytes = t->layer_expert_bytes * 2;

    char s_model[32], s_nexp[32], s_exp[32], s_layer[32], s_dbuf[32];
    fmt_gib(s_model, sizeof(s_model), t->model_total_bytes);
    fmt_gib(s_nexp,  sizeof(s_nexp),  t->model_nonexpert_bytes);
    fmt_gib(s_exp,   sizeof(s_exp),   t->model_expert_bytes);
    fmt_gib(s_layer, sizeof(s_layer), t->layer_expert_bytes);
    fmt_gib(s_dbuf,  sizeof(s_dbuf),  double_buf_bytes);

    /* Extract filename from path for display */
    const char *fname = model_path;
    const char *p = model_path;
    while (*p) { if (*p=='/'||*p=='\\') fname=p+1; p++; }

    printf("\n  Scheduler decision for %s (%s):\n", fname, s_model);

    if (!in_ram) {
        printf("  → Model does NOT fit in RAM — SSD streaming required\n");
        printf("    SSD→RAM→VRAM pipeline bandwidth limited to %.2f GB/s\n",
               t->bw_ssd_to_ram_gbs);
    } else {
        printf("  → Full model fits in RAM  (%.0f GiB RAM available)\n",
               ram_bytes / (double)(1ULL << 30));
        printf("    SSD streaming: not needed for this model\n");
    }

    printf("\n");

    if (nexp_in_vram) {
        uint64_t vram_after_nexp = vram_free > t->model_nonexpert_bytes
                                   ? vram_free - t->model_nonexpert_bytes : 0;
        char s_spare[32];
        fmt_gib(s_spare, sizeof(s_spare), vram_after_nexp);
        printf("  → Non-expert weights (%s): pin permanently in VRAM\n", s_nexp);
        printf("    Remaining VRAM after pinning: %s"
               "  (KV cache + active experts)\n", s_spare);
    } else {
        printf("  → Non-expert weights (%s): exceed usable VRAM budget\n", s_nexp);
        printf("    Some non-expert layers must also be streamed.\n");
    }

    printf("\n");
    printf("  → Expert weights (%s across %llu MoE layers):"
           " stream RAM→VRAM per layer\n",
           s_exp, (unsigned long long)t->n_moe_layers);
    printf("    Per-layer expert budget : %s\n", s_layer);
    printf("    Double-buffer (2 layers): %s\n", s_dbuf);

    double layer_load_ms = t->layer_expert_bytes / (t->bw_ram_to_vram_gbs * 1e9) * 1e3;
    printf("    Est. layer load time   : %.2f ms @ %.2f GB/s RAM→VRAM\n",
           layer_load_ms, t->bw_ram_to_vram_gbs);
}

/* ------------------------------------------------------------------ */
/* Public: topology_probe                                              */
/* ------------------------------------------------------------------ */

topo_t topology_probe(const char *model_path) {
    topo_t t;
    memset(&t, 0, sizeof(t));

    /* ---- GPU info ---- */
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    struct cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    strncpy(t.gpu_name, prop.name, sizeof(t.gpu_name) - 1);
    t.vram_total_mb = prop.totalGlobalMem / (1024 * 1024);

    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    t.vram_free_mb = free_bytes / (1024 * 1024);

    /* ---- System RAM ---- */
    t.ram_total_mb = query_ram_total_mb();

    /* ---- Print header ---- */
    printf("========================================\n");
    printf("  BREAD Topology Probe\n");
    printf("========================================\n");
    printf("\n");
    printf("  GPU   : %s\n", t.gpu_name);
    printf("  VRAM  : %llu MB total,  %llu MB free\n",
           (unsigned long long)t.vram_total_mb,
           (unsigned long long)t.vram_free_mb);
    printf("  RAM   : %llu MB  (~%.0f GB)\n",
           (unsigned long long)t.ram_total_mb,
           t.ram_total_mb / 1024.0);

    /* ---- Model analysis (done first so we can display model info early) ---- */
    printf("\n  Analysing model: %s\n", model_path);
    analyze_model(model_path, &t);

    char s_tot[32], s_nexp[32], s_exp[32];
    fmt_gib(s_tot,  sizeof(s_tot),  t.model_total_bytes);
    fmt_gib(s_nexp, sizeof(s_nexp), t.model_nonexpert_bytes);
    fmt_gib(s_exp,  sizeof(s_exp),  t.model_expert_bytes);
    printf("  Model : %s total  |  non-expert %s  |  expert %s  |  %llu MoE layers\n",
           s_tot, s_nexp, s_exp, (unsigned long long)t.n_moe_layers);

    /* ---- Bandwidth benchmarks ---- */
    t.bw_ssd_to_ram_gbs   = probe_ssd_to_ram_bw(model_path);
    t.bw_ram_to_vram_gbs  = probe_ram_to_vram_bw(t.vram_free_mb);

    /* ---- Summary ---- */
    printf("\n========================================\n");
    printf("  BREAD Topology Summary\n");
    printf("========================================\n");
    printf("\n");
    printf("  GPU              : %s\n",     t.gpu_name);
    printf("  VRAM             : %llu MB\n", (unsigned long long)t.vram_total_mb);
    printf("  RAM              : ~%.0f GB\n", t.ram_total_mb / 1024.0);
    printf("  RAM → VRAM BW    : %.2f GB/s\n", t.bw_ram_to_vram_gbs);
    printf("  SSD → RAM BW     : %.2f GB/s\n", t.bw_ssd_to_ram_gbs);

    print_scheduler(&t, model_path);

    printf("\n========================================\n\n");

    return t;
}

/* ------------------------------------------------------------------ */
/* Standalone entry point                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: topology <gguf-path>\n");
        return 1;
    }
    topology_probe(argv[1]);
    return 0;
}
