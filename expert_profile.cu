#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <chrono>

#include "bread.h"
#include "gguf.h"
#include "loader.h"
#include "layer_ops.h"

extern void bread_matvec(void *w, half *x, half *y,
                         int rows, int cols, int qtype, cudaStream_t stream);

#define QTYPE_Q4_K 12
#define QTYPE_Q6_K 14

static __global__ void silu_mul_kernel(half *gate, const half *up, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        gate[i] = __float2half((g / (1.0f + expf(-g))) * u);
    }
}

static __global__ void scale_accum_kernel(half *dst, const half *src, float scale, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float d = __half2float(dst[i]);
        float s = __half2float(src[i]);
        dst[i] = __float2half(d + scale * s);
    }
}

static double now_ms(void) {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(
        clock::now().time_since_epoch()).count();
}

struct kernel_timing {
    double gate_time;
    double up_time;
    double silu_time;
    double down_time;
    double accum_time;
    double total_per_expert;
    double total_all_experts;
};

int profile_gpu_expert_path(const bread_model_config_t *cfg,
                            const loader_t *L,
                            const weight_cache_t *wc,
                            kernel_timing *out_timing)
{
    if (!cfg || !L || !wc) {
        fprintf(stderr, "profile_gpu_expert_path: invalid inputs\n");
        return 1;
    }

    /* Find first valid MoE layer */
    int layer_idx = -1;
    for (int i = 0; i < cfg->num_layers; i++) {
        if (L->layers[i].valid) {
            layer_idx = i;
            break;
        }
    }
    if (layer_idx < 0) {
        fprintf(stderr, "profile_gpu_expert_path: no valid MoE layer\n");
        return 1;
    }

    const loader_layer_info_t *li = &L->layers[layer_idx];
    const wc_layer_t *wcl = wc ? &wc->layers[layer_idx] : NULL;

    if (!wcl || !wcl->experts.gate_ptrs) {
        fprintf(stderr, "profile_gpu_expert_path: expert cache unavailable\n");
        return 1;
    }

    const int H = cfg->hidden_dim;
    const int E = cfg->expert_inter;
    const int K = cfg->top_k;
    const int repeats = 20;  /* Warmup + measurement */

    /* Allocate I/O */
    half *d_x = NULL, *d_eg = NULL, *d_eu = NULL, *d_eo = NULL, *d_hidden = NULL;
    float *x_fp32 = (float *)malloc((size_t)H * sizeof(float));

    for (int i = 0; i < H; i++) {
        x_fp32[i] = sinf(0.01f * (float)i);
    }

    half *x_fp16 = (half *)malloc((size_t)H * sizeof(half));
    for (int i = 0; i < H; i++) x_fp16[i] = __float2half(x_fp32[i]);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc((void **)&d_x, (size_t)H * sizeof(half));
    cudaMalloc((void **)&d_eg, (size_t)E * sizeof(half));
    cudaMalloc((void **)&d_eu, (size_t)E * sizeof(half));
    cudaMalloc((void **)&d_eo, (size_t)H * sizeof(half));
    cudaMalloc((void **)&d_hidden, (size_t)H * sizeof(half));

    cudaMemcpy(d_x, x_fp16, (size_t)H * sizeof(half), cudaMemcpyHostToDevice);

    /* Warmup */
    for (int k = 0; k < 2; k++) {
        cudaMemsetAsync(d_hidden, 0, (size_t)H * sizeof(half), stream);
        for (int expert = 0; expert < K; expert++) {
            bread_matvec(wcl->experts.gate_ptrs[expert], d_x, d_eg, E, H, QTYPE_Q4_K, stream);
            bread_matvec(wcl->experts.up_ptrs[expert], d_x, d_eu, E, H, QTYPE_Q4_K, stream);
            silu_mul_kernel<<<(E + 255) / 256, 256, 0, stream>>>(d_eg, d_eu, E);
            bread_matvec(wcl->experts.down_ptrs[expert], d_eg, d_eo, H, E, QTYPE_Q6_K, stream);
            scale_accum_kernel<<<(H + 255) / 256, 256, 0, stream>>>(d_hidden, d_eo, 1.0f / K, H);
        }
        cudaStreamSynchronize(stream);
    }

    /* Profile: measure each kernel type separately */
    memset(out_timing, 0, sizeof(*out_timing));

    for (int rep = 0; rep < repeats; rep++) {
        cudaMemsetAsync(d_hidden, 0, (size_t)H * sizeof(half), stream);

        for (int expert = 0; expert < K; expert++) {
            double t0, t1;

            /* Gate matvec */
            cudaStreamSynchronize(stream);
            t0 = now_ms();
            bread_matvec(wcl->experts.gate_ptrs[expert], d_x, d_eg, E, H, QTYPE_Q4_K, stream);
            cudaStreamSynchronize(stream);
            t1 = now_ms();
            out_timing->gate_time += (t1 - t0);

            /* Up matvec */
            t0 = now_ms();
            bread_matvec(wcl->experts.up_ptrs[expert], d_x, d_eu, E, H, QTYPE_Q4_K, stream);
            cudaStreamSynchronize(stream);
            t1 = now_ms();
            out_timing->up_time += (t1 - t0);

            /* SiLU+mul */
            t0 = now_ms();
            silu_mul_kernel<<<(E + 255) / 256, 256, 0, stream>>>(d_eg, d_eu, E);
            cudaStreamSynchronize(stream);
            t1 = now_ms();
            out_timing->silu_time += (t1 - t0);

            /* Down matvec */
            t0 = now_ms();
            bread_matvec(wcl->experts.down_ptrs[expert], d_eg, d_eo, H, E, QTYPE_Q6_K, stream);
            cudaStreamSynchronize(stream);
            t1 = now_ms();
            out_timing->down_time += (t1 - t0);

            /* Accumulate */
            t0 = now_ms();
            scale_accum_kernel<<<(H + 255) / 256, 256, 0, stream>>>(d_hidden, d_eo, 1.0f / K, H);
            cudaStreamSynchronize(stream);
            t1 = now_ms();
            out_timing->accum_time += (t1 - t0);
        }
    }

    /* Average */
    out_timing->gate_time /= (repeats * K);
    out_timing->up_time /= (repeats * K);
    out_timing->silu_time /= (repeats * K);
    out_timing->down_time /= (repeats * K);
    out_timing->accum_time /= (repeats * K);
    out_timing->total_per_expert = out_timing->gate_time + out_timing->up_time +
                                    out_timing->silu_time + out_timing->down_time +
                                    out_timing->accum_time;
    out_timing->total_all_experts = out_timing->total_per_expert * K;

    /* Cleanup */
    cudaFree(d_x);
    cudaFree(d_eg);
    cudaFree(d_eu);
    cudaFree(d_eo);
    cudaFree(d_hidden);
    cudaStreamDestroy(stream);
    free(x_fp32);
    free(x_fp16);

    return 0;
}

int bread_profile_gpu_experts(const bread_model_config_t *cfg,
                              const loader_t *L,
                              const weight_cache_t *wc)
{
    kernel_timing timing;
    if (profile_gpu_expert_path(cfg, L, wc, &timing) != 0) {
        return 1;
    }

    printf("\n=== GPU Expert Kernel Profiling ===\n");
    printf("Layer 0, K=%d experts\n\n", cfg->top_k);

    printf("Per-Expert Timing (K=%d parallel serial loops):\n", cfg->top_k);
    printf("  Gate matvec (Q4_K):       %.4f ms  [%5.1f%%]\n",
           timing.gate_time, 100.0 * timing.gate_time / timing.total_per_expert);
    printf("  Up matvec (Q4_K):         %.4f ms  [%5.1f%%]\n",
           timing.up_time, 100.0 * timing.up_time / timing.total_per_expert);
    printf("  SiLU+mul kernel:          %.4f ms  [%5.1f%%]\n",
           timing.silu_time, 100.0 * timing.silu_time / timing.total_per_expert);
    printf("  Down matvec (Q6_K):       %.4f ms  [%5.1f%%]\n",
           timing.down_time, 100.0 * timing.down_time / timing.total_per_expert);
    printf("  Accumulate kernel:        %.4f ms  [%5.1f%%]\n",
           timing.accum_time, 100.0 * timing.accum_time / timing.total_per_expert);
    printf("  ─────────────────────────────────\n");
    printf("  Total per expert:         %.4f ms\n", timing.total_per_expert);
    printf("  Total K=%d experts:        %.4f ms\n\n", cfg->top_k, timing.total_all_experts);

    /* Analysis */
    double mvec_time = timing.gate_time + timing.up_time + timing.down_time;
    double mvec_pct = 100.0 * mvec_time / timing.total_per_expert;
    printf("Analysis:\n");
    printf("  Matvec (3 kernels): %.4f ms [%.1f%% of per-expert time]\n",
           mvec_time, mvec_pct);
    printf("  Elementwise ops:    %.4f ms [%.1f%% of per-expert time]\n",
           timing.silu_time + timing.accum_time,
           100.0 * (timing.silu_time + timing.accum_time) / timing.total_per_expert);

    if (mvec_pct > 80.0) {
        printf("\n⚠ Bottleneck: MATVEC KERNELS (dequantization-limited)\n");
        printf("  → Optimization: Fuse gate+up, or implement Q4_K→Q6_K kernel fusion\n");
    } else if (timing.down_time > timing.gate_time || timing.down_time > timing.up_time) {
        printf("\n⚠ Bottleneck: DOWN MATVEC (Q6_K slower than Q4_K)\n");
        printf("  → Optimization: Improve Q6_K dequant, or fuse down+accumulate\n");
    } else if (timing.accum_time > 0.5) {
        printf("\n⚠ Bottleneck: ACCUMULATION KERNEL\n");
        printf("  → Optimization: Fuse scale_accum into down matvec output\n");
    }

    printf("\nFor 40 layers: %.1f ms total expert compute (est.)\n",
           timing.total_all_experts * 40);

    return 0;
}
