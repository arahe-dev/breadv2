#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
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

static __global__ void silu_mul_inplace_bench(half *gate, const half *up, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        gate[i] = __float2half((g / (1.0f + expf(-g))) * u);
    }
}

static __global__ void scale_accum_bench(half *dst, const half *src, float scale, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float d = __half2float(dst[i]);
        float s = __half2float(src[i]);
        dst[i] = __float2half(d + scale * s);
    }
}

static double now_ms_bench(void) {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(
        clock::now().time_since_epoch()).count();
}

static void fill_input(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float a = sinf(0.013f * (float)i);
        float b = cosf(0.007f * (float)(i + 11));
        x[i] = 0.5f * a + 0.5f * b;
    }
}

static int first_valid_moe_layer(const loader_t *L, int num_layers) {
    for (int i = 0; i < num_layers; i++) {
        if (L->layers[i].valid) return i;
    }
    return -1;
}

int bread_benchmark_expert_block(const bread_model_config_t *cfg,
                                 const loader_t *L,
                                 const weight_cache_t *wc) {
    if (!cfg || !L) {
        fprintf(stderr, "expert_bench: invalid inputs\n");
        return 1;
    }

    const int layer_idx = first_valid_moe_layer(L, cfg->num_layers);
    if (layer_idx < 0) {
        fprintf(stderr, "expert_bench: no valid MoE layer found\n");
        return 1;
    }

    const loader_layer_info_t *li = &L->layers[layer_idx];
    const wc_layer_t *wcl = wc ? &wc->layers[layer_idx] : NULL;
    if (!li->valid) {
        fprintf(stderr, "expert_bench: selected layer has no experts\n");
        return 1;
    }
    if (!wcl || !wcl->experts.gate_ptrs || !wcl->experts.up_ptrs || !wcl->experts.down_ptrs) {
        fprintf(stderr, "expert_bench: GPU expert cache unavailable; benchmark requires preloaded experts\n");
        return 1;
    }

    const int H = cfg->hidden_dim;
    const int E = cfg->expert_inter;
    const int K = cfg->top_k;
    const int repeats = 10;

    int *expert_indices = (int *)malloc((size_t)K * sizeof(int));
    float *expert_weights = (float *)malloc((size_t)K * sizeof(float));
    float *x_fp32 = (float *)malloc((size_t)H * sizeof(float));
    half *x_fp16 = (half *)malloc((size_t)H * sizeof(half));
    float *cpu_out = (float *)malloc((size_t)H * sizeof(float));
    float *cpu_tmp = (float *)malloc((size_t)K * (size_t)H * sizeof(float));
    float *cpu_gate = (float *)malloc((size_t)K * (size_t)E * sizeof(float));
    float *cpu_up = (float *)malloc((size_t)K * (size_t)E * sizeof(float));
    float *gpu_out = (float *)malloc((size_t)H * sizeof(float));

    half *d_x = NULL, *d_eg = NULL, *d_eu = NULL, *d_eo = NULL, *d_hidden = NULL;
    cudaStream_t stream = NULL;

    if (!expert_indices || !expert_weights || !x_fp32 || !x_fp16 || !cpu_out ||
        !cpu_tmp || !cpu_gate || !cpu_up || !gpu_out) {
        fprintf(stderr, "expert_bench: host alloc failed\n");
        return 1;
    }

    for (int k = 0; k < K; k++) {
        expert_indices[k] = k;
        expert_weights[k] = 1.0f / (float)K;
    }
    fill_input(x_fp32, H);
    for (int i = 0; i < H; i++) x_fp16[i] = __float2half(x_fp32[i]);

    cudaStreamCreate(&stream);
    cudaMalloc((void **)&d_x, (size_t)H * sizeof(half));
    cudaMalloc((void **)&d_eg, (size_t)E * sizeof(half));
    cudaMalloc((void **)&d_eu, (size_t)E * sizeof(half));
    cudaMalloc((void **)&d_eo, (size_t)H * sizeof(half));
    cudaMalloc((void **)&d_hidden, (size_t)H * sizeof(half));
    cudaMemcpy(d_x, x_fp16, (size_t)H * sizeof(half), cudaMemcpyHostToDevice);

    for (int warm = 0; warm < 2; warm++) {
        memset(cpu_out, 0, (size_t)H * sizeof(float));
#pragma omp parallel for schedule(static, 1) num_threads(8)
        for (int k = 0; k < K; k++) {
            float *tmp_out = cpu_tmp + (size_t)k * H;
            float *tmp_gate = cpu_gate + (size_t)k * E;
            float *tmp_up = cpu_up + (size_t)k * E;
            const uint8_t *gate_src = li->gate_base + (uint64_t)expert_indices[k] * li->gate_expert_bytes;
            const uint8_t *up_src = li->up_base + (uint64_t)expert_indices[k] * li->up_expert_bytes;
            const uint8_t *down_src = li->down_base + (uint64_t)expert_indices[k] * li->down_expert_bytes;
            memset(tmp_out, 0, (size_t)H * sizeof(float));
            cpu_tensor_matvec(gate_src, li->gate_type, x_fp32, tmp_gate, E, H);
            cpu_tensor_matvec(up_src, li->up_type, x_fp32, tmp_up, E, H);
            cpu_swiglu(tmp_gate, tmp_up, tmp_gate, E);
            cpu_tensor_matvec(down_src, li->down_type, tmp_gate, tmp_out, H, E);
            for (int i = 0; i < H; i++) tmp_out[i] *= expert_weights[k];
        }
        for (int k = 0; k < K; k++) {
            float *tmp_out = cpu_tmp + (size_t)k * H;
            for (int i = 0; i < H; i++) cpu_out[i] += tmp_out[i];
        }

        cudaMemsetAsync(d_hidden, 0, (size_t)H * sizeof(half), stream);
        for (int k = 0; k < K; k++) {
            bread_matvec(wcl->experts.gate_ptrs[expert_indices[k]], d_x, d_eg, E, H, QTYPE_Q4_K, stream);
            bread_matvec(wcl->experts.up_ptrs[expert_indices[k]], d_x, d_eu, E, H, QTYPE_Q4_K, stream);
            silu_mul_inplace_bench<<<(E + 255) / 256, 256, 0, stream>>>(d_eg, d_eu, E);
            bread_matvec(wcl->experts.down_ptrs[expert_indices[k]], d_eg, d_eo, H, E, QTYPE_Q6_K, stream);
            scale_accum_bench<<<(H + 255) / 256, 256, 0, stream>>>(d_hidden, d_eo, expert_weights[k], H);
        }
        cudaStreamSynchronize(stream);
    }

    double t0 = now_ms_bench();
    for (int rep = 0; rep < repeats; rep++) {
        memset(cpu_out, 0, (size_t)H * sizeof(float));
#pragma omp parallel for schedule(static, 1) num_threads(8)
        for (int k = 0; k < K; k++) {
            float *tmp_out = cpu_tmp + (size_t)k * H;
            float *tmp_gate = cpu_gate + (size_t)k * E;
            float *tmp_up = cpu_up + (size_t)k * E;
            const uint8_t *gate_src = li->gate_base + (uint64_t)expert_indices[k] * li->gate_expert_bytes;
            const uint8_t *up_src = li->up_base + (uint64_t)expert_indices[k] * li->up_expert_bytes;
            const uint8_t *down_src = li->down_base + (uint64_t)expert_indices[k] * li->down_expert_bytes;
            memset(tmp_out, 0, (size_t)H * sizeof(float));
            cpu_tensor_matvec(gate_src, li->gate_type, x_fp32, tmp_gate, E, H);
            cpu_tensor_matvec(up_src, li->up_type, x_fp32, tmp_up, E, H);
            cpu_swiglu(tmp_gate, tmp_up, tmp_gate, E);
            cpu_tensor_matvec(down_src, li->down_type, tmp_gate, tmp_out, H, E);
            for (int i = 0; i < H; i++) tmp_out[i] *= expert_weights[k];
        }
        for (int k = 0; k < K; k++) {
            float *tmp_out = cpu_tmp + (size_t)k * H;
            for (int i = 0; i < H; i++) cpu_out[i] += tmp_out[i];
        }
    }
    double t1 = now_ms_bench();

    double t2 = now_ms_bench();
    for (int rep = 0; rep < repeats; rep++) {
        cudaMemsetAsync(d_hidden, 0, (size_t)H * sizeof(half), stream);
        for (int k = 0; k < K; k++) {
            bread_matvec(wcl->experts.gate_ptrs[expert_indices[k]], d_x, d_eg, E, H, QTYPE_Q4_K, stream);
            bread_matvec(wcl->experts.up_ptrs[expert_indices[k]], d_x, d_eu, E, H, QTYPE_Q4_K, stream);
            silu_mul_inplace_bench<<<(E + 255) / 256, 256, 0, stream>>>(d_eg, d_eu, E);
            bread_matvec(wcl->experts.down_ptrs[expert_indices[k]], d_eg, d_eo, H, E, QTYPE_Q6_K, stream);
            scale_accum_bench<<<(H + 255) / 256, 256, 0, stream>>>(d_hidden, d_eo, expert_weights[k], H);
        }
        cudaStreamSynchronize(stream);
    }
    double t3 = now_ms_bench();

    half *gpu_half = (half *)malloc((size_t)H * sizeof(half));
    cudaMemcpy(gpu_half, d_hidden, (size_t)H * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < H; i++) gpu_out[i] = __half2float(gpu_half[i]);
    free(gpu_half);

    double mse = 0.0;
    double max_abs = 0.0;
    for (int i = 0; i < H; i++) {
        double diff = (double)cpu_out[i] - (double)gpu_out[i];
        double ad = fabs(diff);
        mse += diff * diff;
        if (ad > max_abs) max_abs = ad;
    }
    mse /= (double)H;

    printf("\n=== Expert Block Benchmark ===\n");
    printf("Layer             : %d\n", layer_idx);
    printf("Experts           : %d\n", K);
    printf("Hidden dim        : %d\n", H);
    printf("Expert inter      : %d\n", E);
    printf("CPU path          : %.3f ms/block (OpenMP over %d experts)\n", (t1 - t0) / repeats, K);
    printf("GPU path          : %.3f ms/block (current serial GPU loop)\n", (t3 - t2) / repeats);
    printf("CPU/GPU ratio     : %.2fx\n", ((t1 - t0) / repeats) / ((t3 - t2) / repeats));
    printf("Output MSE        : %.6e\n", mse);
    printf("Output max abs    : %.6e\n", max_abs);

    cudaFree(d_x);
    cudaFree(d_eg);
    cudaFree(d_eu);
    cudaFree(d_eo);
    cudaFree(d_hidden);
    cudaStreamDestroy(stream);
    free(expert_indices);
    free(expert_weights);
    free(x_fp32);
    free(x_fp16);
    free(cpu_out);
    free(cpu_tmp);
    free(cpu_gate);
    free(cpu_up);
    free(gpu_out);

    return 0;
}
