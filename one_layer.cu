/* one_layer.cu — BREAD Step 2: single transformer layer forward pass
 *
 * Implements the CMD1 / CPU-attn / CMD2 / CPU-routing / CMD3 pipeline
 * for one Qwen3.5-35B-A3B transformer layer.
 *
 * Full-attention path (layer % 4 == 3): real Q/K/V proj, GQA expand, o_proj.
 * GatedDeltaNet/SSM path: stub (attention contribution = 0).
 *
 * Both paths exercise the full MoE path:
 *   CMD2  — post-attn RMSNorm + shared-expert gate/up projections
 *   CPU   — router softmax + topK
 *   DMA   — loader_request / loader_sync
 *   CMD3  — K expert gate/up/SwiGLU/down + shared expert + weighted combine
 *
 * Build: build_layer.bat
 * Run:   one_layer_test.exe [model_path]
 *
 * Expected output:
 *   Layer 0 (SSM stub) and Layer 3 (full attn) both print non-zero output
 *   that differs from the random input → PASS.
 */

#ifdef _WIN32
#  define _CRT_SECURE_NO_WARNINGS
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "bread.h"
#include "gguf.h"
#include "loader.h"
#include "bread_utils.h"
#include "layer_ops.h"
#include "buffer_pool.h"

/* ------------------------------------------------------------------ */
/* External: bread_matvec from kernels.cu                              */
/* ------------------------------------------------------------------ */

extern void bread_matvec(void *w, half *x, half *y,
                          int rows, int cols, int qtype, cudaStream_t stream);

/* ================================================================== */
/* Phase 2: GPU-side routing kernels                                  */
/* ================================================================== */

/* Router matmul + softmax in one kernel for efficiency
 * Computes: logits[i] = sum_j(router_w[i*H + j] * normed2[j])
 * Then softmax(logits) → probs[i]
 */
static __global__ void router_matmul_softmax(
    const half *d_normed2,           /* [H] input (half) */
    const float *d_router_w,         /* [num_experts × H] weights (float) */
    float *d_logits,                 /* [num_experts] logits (output) */
    int H, int num_experts)
{
    int expert_idx = blockIdx.x;
    if (expert_idx >= num_experts) return;

    /* Each block handles one expert: compute dot product */
    float sum = 0.0f;
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        sum += d_router_w[expert_idx * H + j] * __half2float(d_normed2[j]);
    }

    /* Warp reduce */
    for (int delta = 16; delta > 0; delta /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, delta);

    if (threadIdx.x == 0) {
        d_logits[expert_idx] = sum;
    }
}

/* Softmax in-place on GPU */
static __global__ void softmax_inplace(float *logits, int n)
{
    int tid = threadIdx.x;
    __shared__ float max_val, sum_exp;

    /* Step 1: find maximum for numerical stability */
    float thread_max = (tid < n) ? logits[tid] : -1e30f;
    for (int delta = 16; delta > 0; delta /= 2)
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, delta));

    if (tid == 0) max_val = thread_max;
    __syncthreads();

    /* Step 2: compute exp and sum */
    float exp_val = (tid < n) ? expf(logits[tid] - max_val) : 0.0f;
    float thread_sum = exp_val;
    for (int delta = 16; delta > 0; delta /= 2)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, delta);

    if (tid == 0) sum_exp = thread_sum;
    __syncthreads();

    /* Step 3: normalize */
    if (tid < n) {
        logits[tid] = exp_val / (sum_exp + 1e-6f);
    }
}

/* TopK selection: find K largest values and their indices */
static __global__ void topk_select(
    const float *d_probs,           /* [num_experts] values */
    int *d_indices,                 /* [top_k] output indices */
    float *d_weights,               /* [top_k] output values */
    int num_experts, int top_k)
{
    /* Simplified: use first K threads to find their best */
    int k = threadIdx.x;
    if (k >= top_k) return;

    float best = -1.0f;
    int best_id = -1;

    /* Each thread finds the k-th largest by simple scan */
    for (int i = 0; i < num_experts; i++) {
        float val = d_probs[i];
        bool taken = false;
        for (int j = 0; j < k; j++) {
            if (d_indices[j] == i) {
                taken = true;
                break;
            }
        }
        if (!taken && val > best) {
            best = val;
            best_id = i;
        }
    }

    d_indices[k] = best_id;
    d_weights[k] = best;
}

/* ================================================================== */
/* File-scope scratch buffers for CPU conversion functions             */
/* ================================================================== */

static half *s_vram2cpu_tmp = NULL;  /* scratch for vram_half_to_cpu_float  */

/* ------------------------------------------------------------------ */
/* Kernel 1: RMSNorm in-place                                           */
/*                                                                      */
/* x[n] (half, in VRAM) is normalised and multiplied by w[n] (float). */
/* One block, 256 threads, tree reduction in shared memory.            */
/* Works for any n; threads stride through n in steps of 256.         */
/* ------------------------------------------------------------------ */

static __global__ void rmsnorm_inplace(half *x, const float *w, int n, float eps)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    /* accumulate x^2 */
    float acc = 0.0f;
    for (int i = tid; i < n; i += 256)
        acc += __half2float(x[i]) * __half2float(x[i]);
    sdata[tid] = acc;
    __syncthreads();

    /* tree reduction */
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / (float)n + eps);

    /* scale each element */
    for (int i = tid; i < n; i += 256)
        x[i] = __float2half(__half2float(x[i]) * rms * w[i]);
}

/* ------------------------------------------------------------------ */
/* Kernel 2: SwiGLU in-place on gate[]                                 */
/*                                                                      */
/* gate[i] = silu(gate[i]) * up[i]   where silu(x) = x * sigmoid(x)  */
/* Each thread handles one element.                                     */
/* ------------------------------------------------------------------ */

static __global__ void silu_mul_inplace(half *gate, const half *up, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    gate[i] = __float2half(g / (1.0f + expf(-g)) * u);
}

/* ------------------------------------------------------------------ */
/* Kernel 4: Scaled accumulate — dst[i] += scale * src[i]             */
/* ------------------------------------------------------------------ */

static __global__ void scale_accum(half *dst, const half *src, float scale, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + scale * __half2float(src[i]));
}

/* ------------------------------------------------------------------ */
/* Kernel 5: memcpy half[] device→device (avoids cudaMemcpy overhead) */
/* ------------------------------------------------------------------ */

static __global__ void copy_half(half *dst, const half *src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

/* ------------------------------------------------------------------ */
/* Weight loading helpers                                               */
/* ------------------------------------------------------------------ */

/* Pointer to tensor data in pinned host RAM */
const gguf_tensor_t *require_tensor(const gguf_ctx_t *g, const char *name)
{
    const gguf_tensor_t *t = gguf_find_tensor(g, name);
    if (!t) {
        fprintf(stderr, "tensor not found: %s\n", name);
        exit(1);
    }
    return t;
}

uint8_t *tensor_ram(const loader_t *L, const gguf_ctx_t *g,
                    const char *name)
{
    const gguf_tensor_t *t = require_tensor(g, name);
    return L->pinned_data + L->data_offset + t->offset;
}

static const float *tensor_ram_f32(const loader_t *L, const gguf_ctx_t *g,
                                   const char *name)
{
    const gguf_tensor_t *t = require_tensor(g, name);
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "tensor type mismatch: %s expected F32 got %s\n",
                name, ggml_type_name(t->type));
        exit(1);
    }
    return (const float *)(L->pinned_data + L->data_offset + t->offset);
}

static const half *tensor_ram_f16(const loader_t *L, const gguf_ctx_t *g,
                                  const char *name)
{
    const gguf_tensor_t *t = require_tensor(g, name);
    if (t->type != GGML_TYPE_F16) {
        fprintf(stderr, "tensor type mismatch: %s expected F16 got %s\n",
                name, ggml_type_name(t->type));
        exit(1);
    }
    return (const half *)(L->pinned_data + L->data_offset + t->offset);
}

static void *load_expert_tensor_vram(const loader_t *L, const gguf_ctx_t *g,
                                     const char *name, int expert_idx,
                                     uint32_t *type_out)
{
    const gguf_tensor_t *t = require_tensor(g, name);
    int n_experts = (int)t->dims[t->n_dims - 1];
    uint64_t expert_bytes;
    uint8_t *src;
    void *d;

    if (expert_idx < 0 || expert_idx >= n_experts) {
        fprintf(stderr, "load_expert_tensor_vram: %s expert %d out of range [0,%d)\n",
                name, expert_idx, n_experts);
        exit(1);
    }

    expert_bytes = t->size / (uint64_t)n_experts;
    src = L->pinned_data + L->data_offset + t->offset + (uint64_t)expert_idx * expert_bytes;
    CUDA_CHECK(cudaMalloc(&d, (size_t)expert_bytes));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)expert_bytes, cudaMemcpyHostToDevice));
    if (type_out) *type_out = t->type;
    return d;
}

/* Allocate VRAM and copy a tensor from pinned RAM.
 * Used as fallback when weight cache is not available. */
static void *load_vram(const loader_t *L, const gguf_ctx_t *g,
                        const char *name)
{
    const gguf_tensor_t *t = require_tensor(g, name);
    void *d;
    CUDA_CHECK(cudaMalloc(&d, t->size));
    CUDA_CHECK(cudaMemcpy(d, L->pinned_data + L->data_offset + t->offset,
                           t->size, cudaMemcpyHostToDevice));
    return d;
}

/* ------------------------------------------------------------------ */
/* CPU helpers                                                          */
/* ------------------------------------------------------------------ */

/* Convert VRAM half[] to CPU float[] */
static void vram_half_to_cpu_float(const half *d_x, float *h_f, int n)
{
    /* Allocate scratch buffer on first use if not already done */
    if (!s_vram2cpu_tmp) {
        const bread_model_config_t *cfg = bread_model_config_get();
        s_vram2cpu_tmp = (half *)malloc(cfg->ssm_qkv_dim * sizeof(half));
        if (!s_vram2cpu_tmp) {
            fprintf(stderr, "vram_half_to_cpu_float: malloc failed for s_vram2cpu_tmp\n");
            exit(1);
        }
    }
    CUDA_CHECK(cudaMemcpy(s_vram2cpu_tmp, d_x, n * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) h_f[i] = __half2float(s_vram2cpu_tmp[i]);
}

#define CPU_QK_BLOCK_ELEMS  256
#define CPU_Q4K_BLOCK_BYTES 144
#define CPU_Q6K_BLOCK_BYTES 210


static void get_scale_min_k4_host(int j, const uint8_t *scales,
                                  uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = scales[j]     & 63;
        *m = scales[j + 4] & 63;
    } else {
        *d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m = (scales[j + 4] >>   4) | ((scales[j    ] >> 6) << 4);
    }
}

static void cpu_dequant_q4k_block(const uint8_t *block, float *out)
{
    uint16_t d_raw, dmin_raw;
    const uint8_t *scales = block + 4;
    const uint8_t *qs     = block + 16;
    memcpy(&d_raw,    block + 0, 2);
    memcpy(&dmin_raw, block + 2, 2);
    {
        const float d    = bread_h2f(d_raw);
        const float dmin = bread_h2f(dmin_raw);
        int is = 0;
        float *y = out;
        const uint8_t *q = qs;
        for (int j = 0; j < CPU_QK_BLOCK_ELEMS; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4_host(is + 0, scales, &sc, &m);
            {
                const float d1 = d * sc;
                const float m1 = dmin * m;
                get_scale_min_k4_host(is + 1, scales, &sc, &m);
                {
                    const float d2 = d * sc;
                    const float m2 = dmin * m;
                    for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
                    for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
                }
            }
            q  += 32;
            is += 2;
        }
    }
}

static void cpu_dequant_q6k_block(const uint8_t *block, float *out)
{
    uint16_t d_raw;
    const uint8_t *ql = block + 0;
    const uint8_t *qh = block + 128;
    const int8_t  *sc = (const int8_t *)(block + 192);
    memcpy(&d_raw, block + 208, 2);
    {
        const float d = bread_h2f(d_raw);
        float *y = out;
        for (int n = 0; n < CPU_QK_BLOCK_ELEMS; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static void cpu_matvec_f32(const uint8_t *w, const float *x, float *y, int rows, int cols)
{
    const float *wf = (const float *)w;
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const float *row = wf + (size_t)r * cols;
        for (int c = 0; c < cols; c++) sum += row[c] * x[c];
        y[r] = sum;
    }
}

static void cpu_matvec_f16(const uint8_t *w, const float *x, float *y, int rows, int cols)
{
    const half *wh = (const half *)w;
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const half *row = wh + (size_t)r * cols;
        for (int c = 0; c < cols; c++) sum += __half2float(row[c]) * x[c];
        y[r] = sum;
    }
}

static void cpu_matvec_q4k(const uint8_t *w, const float *x, float *y, int rows, int cols)
{
    const int row_bytes = (int)ggml_tensor_nbytes((uint64_t)cols, GGML_TYPE_Q4_K);
    const int n_blocks = cols / CPU_QK_BLOCK_ELEMS;
    float block[CPU_QK_BLOCK_ELEMS];
    for (int r = 0; r < rows; r++) {
        const uint8_t *row = w + (size_t)r * row_bytes;
        float sum = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            cpu_dequant_q4k_block(row + (size_t)b * CPU_Q4K_BLOCK_BYTES, block);
            for (int i = 0; i < CPU_QK_BLOCK_ELEMS; i++) {
                sum += block[i] * x[b * CPU_QK_BLOCK_ELEMS + i];
            }
        }
        y[r] = sum;
    }
}

static void cpu_matvec_q6k(const uint8_t *w, const float *x, float *y, int rows, int cols)
{
    const int row_bytes = (int)ggml_tensor_nbytes((uint64_t)cols, GGML_TYPE_Q6_K);
    const int n_blocks = cols / CPU_QK_BLOCK_ELEMS;
    float block[CPU_QK_BLOCK_ELEMS];
    for (int r = 0; r < rows; r++) {
        const uint8_t *row = w + (size_t)r * row_bytes;
        float sum = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            cpu_dequant_q6k_block(row + (size_t)b * CPU_Q6K_BLOCK_BYTES, block);
            for (int i = 0; i < CPU_QK_BLOCK_ELEMS; i++) {
                sum += block[i] * x[b * CPU_QK_BLOCK_ELEMS + i];
            }
        }
        y[r] = sum;
    }
}

static void cpu_tensor_matvec(const uint8_t *w, uint32_t type,
                              const float *x, float *y, int rows, int cols)
{
    switch (type) {
        case GGML_TYPE_F32:
            cpu_matvec_f32(w, x, y, rows, cols);
            break;
        case GGML_TYPE_F16:
            cpu_matvec_f16(w, x, y, rows, cols);
            break;
        case GGML_TYPE_Q4_K:
            cpu_matvec_q4k(w, x, y, rows, cols);
            break;
        case GGML_TYPE_Q6_K:
            cpu_matvec_q6k(w, x, y, rows, cols);
            break;
        default:
            fprintf(stderr, "cpu_tensor_matvec: unsupported type %s\n", ggml_type_name(type));
            exit(1);
    }
}

static void cpu_named_matvec(const loader_t *L, const gguf_ctx_t *g,
                             const char *name, const float *x, float *y,
                             int rows, int cols)
{
    const gguf_tensor_t *t = require_tensor(g, name);
    cpu_tensor_matvec(tensor_ram(L, g, name), t->type, x, y, rows, cols);
}

static void cpu_rms_norm_weighted(const float *x, const float *w,
                                  float *out, int n, float eps)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    {
        const float inv_rms = 1.0f / sqrtf(sum_sq / (float)n + eps);
        for (int i = 0; i < n; i++) out[i] = x[i] * inv_rms * w[i];
    }
}

static void cpu_swiglu(const float *gate, const float *up, float *out, int n)
{
    for (int i = 0; i < n; i++) {
        const float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

/* Softmax in-place on x[n] */
static void cpu_softmax(float *x, int n)
{
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* topK: fill indices[K] and weights[K] (un-normalised expert weights) */
static void cpu_topk(const float *probs, int n, int K,
                      int *indices, float *weights)
{
    /* Simple O(n*K) selection — fine for n=256, K=8 */
    int used[256];
    memset(used, 0, n * sizeof(int));
    float w_sum = 0.0f;
    for (int k = 0; k < K; k++) {
        int best = -1;
        float bv = -1e30f;
        for (int i = 0; i < n; i++) {
            if (!used[i] && probs[i] > bv) { bv = probs[i]; best = i; }
        }
        indices[k] = best;
        weights[k] = bv;
        w_sum += bv;
        used[best] = 1;
    }
    /* Normalise weights to sum to 1 */
    if (w_sum > 0.0f)
        for (int k = 0; k < K; k++) weights[k] /= w_sum;
}

static float cpu_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static float cpu_softplus(float x)
{
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return logf(1.0f + expf(x));
}

static void cpu_rms_norm_bare(float *x, int n, float eps)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / (float)n + eps);
    for (int i = 0; i < n; i++) x[i] *= inv_rms;
}

static void cpu_l2_norm_bare(float *x, int n, float eps)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    {
        float inv_norm = 1.0f / sqrtf(sum_sq + eps);
        for (int i = 0; i < n; i++) x[i] *= inv_norm;
    }
}

static void cpu_conv1d_step(const float *conv_state,
                            const float *new_input,
                            const float *weight,
                            float *out,
                            int channels,
                            int kernel_size)
{
    for (int c = 0; c < channels; c++) {
        float acc = 0.0f;
        for (int k = 0; k < kernel_size - 1; k++)
            acc += conv_state[k * channels + c] * weight[c * kernel_size + k];
        acc += new_input[c] * weight[c * kernel_size + (kernel_size - 1)];
        out[c] = acc;
    }
}

static void cpu_silu_inplace(float *x, int n)
{
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

static void cpu_gated_rms_norm(const float *x, const float *z,
                               const float *w, float *out, int n, float eps)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / (float)n + eps);
    for (int i = 0; i < n; i++) {
        float silu_z = z[i] / (1.0f + expf(-z[i]));
        out[i] = x[i] * inv_rms * w[i] * silu_z;
    }
}

static int ssm_k_head_for_v_head(int vh, int num_k_heads, int num_v_heads)
{
    if (num_k_heads <= 0 || num_v_heads <= 0) return 0;
    if (num_k_heads == num_v_heads) return vh;
    /* Match ggml_repeat-style head broadcast semantics from llama.cpp. */
    return vh % num_k_heads;
}

static void cpu_repeat_heads(float *dst,
                             const float *src,
                             int src_heads,
                             int dst_heads,
                             int head_dim)
{
    if (src_heads <= 0 || dst_heads <= 0 || head_dim <= 0) return;
    if (src_heads == dst_heads) {
        memcpy(dst, src, (size_t)dst_heads * head_dim * sizeof(float));
        return;
    }
    for (int h = 0; h < dst_heads; h++) {
        const float *src_h = src + (size_t)(h % src_heads) * head_dim;
        memcpy(dst + (size_t)h * head_dim, src_h, (size_t)head_dim * sizeof(float));
    }
}

static void cpu_delta_net_autoregressive_step(const float *q,
                                              const float *k,
                                              const float *v,
                                              float gate,
                                              float beta,
                                              float *state,
                                              float *out,
                                              int value_dim,
                                              int key_dim,
                                              float *sk_buf,
                                              float *d_buf)
{
    float decay = expf(gate);

    for (int vi = 0; vi < value_dim; vi++) {
        float *row = state + (size_t)vi * key_dim;
        for (int ki = 0; ki < key_dim; ki++) {
            row[ki] *= decay;
        }
    }

    for (int vi = 0; vi < value_dim; vi++) {
        const float *row = state + (size_t)vi * key_dim;
        float sk = 0.0f;
        for (int ki = 0; ki < key_dim; ki++) {
            sk += row[ki] * k[ki];
        }
        sk_buf[vi] = sk;
        d_buf[vi] = (v[vi] - sk) * beta;
    }

    for (int vi = 0; vi < value_dim; vi++) {
        float *row = state + (size_t)vi * key_dim;
        float d = d_buf[vi];
        for (int ki = 0; ki < key_dim; ki++) {
            row[ki] += k[ki] * d;
        }
    }

    for (int vi = 0; vi < value_dim; vi++) {
        const float *row = state + (size_t)vi * key_dim;
        float acc = 0.0f;
        for (int ki = 0; ki < key_dim; ki++) {
            acc += row[ki] * q[ki];
        }
        out[vi] = acc;
    }
}

static void trace_vec_stats(const char *label, const float *x, int n, int max_show)
{
    float min_v = 1e30f, max_v = -1e30f, sum_sq = 0.0f;
    fprintf(stderr, "TRACE %s: n=%d", label, n);
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum_sq += v * v;
    }
    fprintf(stderr, " rms=%.6f min=%.6f max=%.6f first=", sqrtf(sum_sq / (float)n), min_v, max_v);
    for (int i = 0; i < n && i < max_show; i++) {
        fprintf(stderr, "%s%.6f", i ? "," : "", x[i]);
    }
    fprintf(stderr, "\n");
}

static void trace_sigmoid_stats(const char *label, const float *x, int n, int max_show)
{
    float min_v = 1e30f, max_v = -1e30f, sum_sq = 0.0f;
    fprintf(stderr, "TRACE %s: n=%d", label, n);
    for (int i = 0; i < n; i++) {
        float v = cpu_sigmoid(x[i]);
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum_sq += v * v;
    }
    fprintf(stderr, " rms=%.6f min=%.6f max=%.6f first=", sqrtf(sum_sq / (float)n), min_v, max_v);
    for (int i = 0; i < n && i < max_show; i++) {
        fprintf(stderr, "%s%.6f", i ? "," : "", cpu_sigmoid(x[i]));
    }
    fprintf(stderr, "\n");
}

static int rope_select_stream_for_pair(const bread_model_config_t *cfg, int pair_idx, int total_pairs)
{
    const int *sections = cfg->rope_sections;
    const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    if (sect_dims <= 0 || total_pairs <= 0) {
        return 0;
    }

    const int sector = pair_idx % sect_dims;

    if (cfg->rope_mrope_interleaved) {
        if ((sector % 3) == 1 && sector < 3 * sections[1]) {
            return 1;
        } else if ((sector % 3) == 2 && sector < 3 * sections[2]) {
            return 2;
        } else if ((sector % 3) == 0 && sector < 3 * sections[0]) {
            return 0;
        }
        return 3;
    }

    {
        const int sec_w = sections[0] + sections[1];
        const int sec_e = sec_w + sections[2];
        if (sector >= sections[0] && sector < sec_w) {
            return 1;
        } else if (sector >= sec_w && sector < sec_e) {
            return 2;
        } else if (sector >= sec_e) {
            return 3;
        }
    }

    return 0;
}

static void apply_rotary_emb(const bread_model_config_t *cfg,
                             float *q, float *k, int pos)
{
    if (bread_get_disable_rope()) return;
    const int rotary_dim = cfg->head_dim_rope;
    const int half = rotary_dim / 2;
    const int num_q_heads = cfg->num_q_heads;
    const int num_kv_heads = cfg->num_kv_heads;
    const int head_dim = cfg->head_dim_qk;
    const float rope_freq_base = cfg->rope_freq_base;
    const float pos_streams[4] = { (float) pos, (float) pos, (float) pos, 0.0f };

    for (int h = 0; h < num_q_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            const int stream = rope_select_stream_for_pair(cfg, i, half);
            const float freq = 1.0f / powf(rope_freq_base, (float)(2 * i) / rotary_dim);
            const float angle = pos_streams[stream] * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float q0 = qh[i];
            float q1 = qh[i + half];
            qh[i]        = q0 * cos_a - q1 * sin_a;
            qh[i + half] = q0 * sin_a + q1 * cos_a;
        }
    }
    for (int h = 0; h < num_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < half; i++) {
            const int stream = rope_select_stream_for_pair(cfg, i, half);
            const float freq = 1.0f / powf(rope_freq_base, (float)(2 * i) / rotary_dim);
            const float angle = pos_streams[stream] * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float k0 = kh[i];
            float k1 = kh[i + half];
            kh[i]        = k0 * cos_a - k1 * sin_a;
            kh[i + half] = k0 * sin_a + k1 * cos_a;
        }
    }
}

static void split_full_attn_q(const bread_model_config_t *cfg,
                              const float *q_full,
                              float *q_score,
                              float *q_gate)
{
    const int per_head_q = cfg->head_dim_qk + cfg->head_dim_qgate;

    for (int h = 0; h < cfg->num_q_heads; h++) {
        const float *src = q_full + h * per_head_q;
        memcpy(q_score + h * cfg->head_dim_qk,
               src,
               cfg->head_dim_qk * sizeof(float));
        memcpy(q_gate + h * cfg->head_dim_v,
               src + cfg->head_dim_qk,
               cfg->head_dim_v * sizeof(float));
    }
}

static void apply_per_head_rms_norm(float *x,
                                    const float *w,
                                    int num_heads,
                                    int head_dim,
                                    float eps)
{
    for (int h = 0; h < num_heads; h++) {
        float *xh = x + h * head_dim;
        float sum_sq = 0.0f;

        for (int i = 0; i < head_dim; i++) sum_sq += xh[i] * xh[i];
        {
            float inv_rms = 1.0f / sqrtf(sum_sq / (float)head_dim + eps);
            for (int i = 0; i < head_dim; i++) {
                xh[i] = xh[i] * inv_rms * w[i];
            }
        }
    }
}

/* ================================================================== */
/*  one_layer_forward                                                   */
/*                                                                      */
/* Runs one transformer block (CMD1/CPU/CMD2/CPU/CMD3).               */
/*                                                                      */
/* d_hidden [BREAD_HIDDEN_DIM] half — input, modified in-place.       */
/* layer_idx     — which transformer block (0-39).                    */
/* pos           — token position (0 for first token).                 */
/* L             — loader (model in pinned RAM, expert VRAM cache).   */
/* g             — GGUF context (tensor metadata).                     */
/* stream_a      — CUDA stream for compute kernels.                   */
/* ================================================================== */

/* File-scope so one_layer_cpu_hidden_rms() can access it */
static float *g_h_hidden = NULL;
static float g_last_branch_rms = -1.0f;

static float cpu_rms_f32(const float *x, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += x[i] * x[i];
    return sqrtf(sum / (float)n);
}

static float device_half_rms(const half *d_x, int n)
{
    half *tmp = (half *)malloc((size_t)n * sizeof(half));
    float sum = 0.0f;
    if (!tmp) {
        fprintf(stderr, "device_half_rms: alloc failed\n");
        exit(1);
    }
    CUDA_CHECK(cudaMemcpy(tmp, d_x, (size_t)n * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
        float v = __half2float(tmp[i]);
        sum += v * v;
    }
    free(tmp);
    return sqrtf(sum / (float)n);
}

/* Returns RMS of CPU-side hidden state (valid only in boring/minimal mode,
 * after at least one layer has run). Returns -1 if not available. */
float one_layer_cpu_hidden_rms(int hidden_dim)
{
    if (!g_h_hidden) return -1.0f;
    float sum = 0.0f;
    for (int i = 0; i < hidden_dim; i++)
        sum += g_h_hidden[i] * g_h_hidden[i];
    return sqrtf(sum / hidden_dim);
}

float one_layer_last_branch_rms(void)
{
    return g_last_branch_rms;
}

/* ================================================================== */
/* ================================================================== */
/* Phase 2: GPU routing wrapper - keeps routing on device              */
/* ================================================================== */
static void route_layer_gpu(const bread_model_config_t *cfg, const half *d_normed2,
                            const float *d_router_w, const half *d_shared_gate_w,
                            int *d_expert_indices, float *d_expert_weights,
                            float *h_shared_gate_score, cudaStream_t stream)
{
    int H = cfg->hidden_dim;
    int num_experts = cfg->num_experts;
    int top_k = cfg->top_k;

    /* Allocate temporary device buffers for logits */
    float *d_logits = NULL;
    CUDA_CHECK(cudaMalloc(&d_logits, num_experts * sizeof(float)));

    /* Step 1: Router matmul → logits[num_experts] */
    router_matmul_softmax<<<num_experts, 256, 0, stream>>>(
        d_normed2, d_router_w, d_logits, H, num_experts);

    /* Step 2: Softmax (in-place on d_logits) */
    softmax_inplace<<<1, 256, 0, stream>>>(d_logits, num_experts);

    /* Step 3: TopK selection → expert_indices, expert_weights (device) */
    topk_select<<<1, top_k, 0, stream>>>(d_logits, d_expert_indices,
                                         d_expert_weights, num_experts, top_k);

    /* Step 4: Shared gate score (if exists) - still computed on device via kernel */
    /* For now, set to 0.0; could add a device kernel for this later */
    *h_shared_gate_score = 0.0f;

    CUDA_CHECK(cudaFree(d_logits));
}

/* ================================================================== */
/* route_layer: MoE routing logic (extracted for prefetching)         */
/*                                                                    */
/* Takes the post-attn normalised hidden state (d_normed2) and       */
/* computes expert routing: router matmul + softmax + topK.          */
/* Output: expert_indices, expert_weights (pre-allocated buffers).   */
/* Also computes shared_gate_score (returned, not used by route).    */
/* ================================================================== */

float route_layer(loader_t *L, gguf_ctx_t *g,
                  int layer_idx, const half *d_normed2,
                  int *expert_indices, float *expert_weights)
{
    const bread_model_config_t *cfg = bread_model_config_get();
    int H = cfg->hidden_dim;
    char nm[128];
    float shared_gate_score = 0.0f;

    /* Pre-allocated static buffers (shared with one_layer_forward) */
    static float *h_normed2 = NULL;
    static float *h_logits = NULL;
    static int   *h_expert_indices = NULL;
    static float *h_expert_weights = NULL;

    /* Allocate once */
    if (!h_normed2) {
        h_normed2 = (float *)malloc((size_t)H * sizeof(float));
        h_logits = (float *)malloc((size_t)cfg->num_experts * sizeof(float));
        h_expert_indices = (int *)malloc((size_t)cfg->top_k * sizeof(int));
        h_expert_weights = (float *)malloc((size_t)cfg->top_k * sizeof(float));
    }

    /* Copy d_normed2 (VRAM half) → CPU float */
    vram_half_to_cpu_float(d_normed2, h_normed2, H);

    /* Load router weights */
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", layer_idx);
    const float *router_w = tensor_ram_f32(L, g, nm);

    /* Router matmul: logits[i] = Σ_j router_w[i * H + j] * normed2[j] */
    memset(h_logits, 0, (size_t)cfg->num_experts * sizeof(float));
    for (int i = 0; i < cfg->num_experts; i++)
        for (int j = 0; j < H; j++)
            h_logits[i] += router_w[i * H + j] * h_normed2[j];

    /* Shared gate score (optional) */
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp_shexp.weight", layer_idx);
    const half *shared_gate_w = NULL;
    if (gguf_find_tensor(g, nm)) {
        shared_gate_w = tensor_ram_f16(L, g, nm);
        for (int j = 0; j < H; j++)
            shared_gate_score += __half2float(shared_gate_w[j]) * h_normed2[j];
    }

    /* Softmax + topK */
    cpu_softmax(h_logits, cfg->num_experts);
    cpu_topk(h_logits, cfg->num_experts, cfg->top_k,
             h_expert_indices, h_expert_weights);

    /* Copy results to output buffers */
    memcpy(expert_indices, h_expert_indices, (size_t)cfg->top_k * sizeof(int));
    memcpy(expert_weights, h_expert_weights, (size_t)cfg->top_k * sizeof(float));

    return shared_gate_score;
}

void one_layer_forward(half *d_hidden, int layer_idx, int pos,
                        loader_t *L, gguf_ctx_t *g,
                        weight_cache_t *wc,
                        cudaStream_t stream_a)
{
    const bread_model_config_t *cfg = bread_model_config_get();
    char nm[128];   /* tensor name buffer */
    int H = cfg->hidden_dim;
    int is_full = bread_layer_is_full_attention(layer_idx);

    /* ---- Persistent scratch VRAM (allocated once, reused) ---- */
    static half *d_normed   = NULL;   /* H — pre-attn normalised state  */
    static half *d_normed2  = NULL;   /* H — pre-FFN normalised state   */
    static half *d_q        = NULL;   /* Q_PROJ_DIM                     */
    static half *d_k        = NULL;   /* KV_PROJ_DIM                    */
    static half *d_v        = NULL;   /* KV_PROJ_DIM                    */
    static half *d_attn_out = NULL;   /* ATTN_OUT_DIM                   */
    static half *d_o_out    = NULL;   /* H — o_proj output              */
    static half *d_sg       = NULL;   /* SHARED_INTER — shexp gate      */
    static half *d_su       = NULL;   /* SHARED_INTER — shexp up        */
    static half *d_sh_out   = NULL;   /* H — shexp final output         */
    static half *d_eg       = NULL;   /* EXPERT_INTER — expert gate     */
    static half *d_eu       = NULL;   /* EXPERT_INTER — expert up       */
    static half *d_eo       = NULL;   /* H — expert final output        */
    static half *d_qkv      = NULL;   /* SSM QKV projection             */
    static half *d_z        = NULL;   /* SSM gate projection            */
    static half *d_alpha    = NULL;   /* SSM alpha projection           */
    static half *d_beta     = NULL;   /* SSM beta projection            */

    static float *h_qkv      = NULL;
    static float *h_z        = NULL;
    static float *h_alpha    = NULL;
    static float *h_beta     = NULL;
    static float *h_conv_out = NULL;
    static float *h_attn_out = NULL;
    static float *h_head_tmp = NULL;
    static float *h_ssm_qrep = NULL;
    static float *h_ssm_krep = NULL;
    static float *h_ssm_sk   = NULL;
    static float *h_ssm_d    = NULL;
    static float *h_q_full   = NULL;
    static float *h_kv_k     = NULL;
    static float *h_kv_v     = NULL;
    static float *h_q_score  = NULL;
    static float *h_q_gate   = NULL;
    static float *h_scores   = NULL;
    static float *h_normed   = NULL;
    static float *h_normed2  = NULL;
    static float *h_o_cpu    = NULL;
    static float *h_sg_cpu   = NULL;
    static float *h_su_cpu   = NULL;
    static float *h_sh_cpu   = NULL;
    static float *h_eg_cpu   = NULL;
    static float *h_eu_cpu   = NULL;
    static float *h_eo_cpu   = NULL;
    static half  *h_hidden_half = NULL;
    static half  *h_attn_half_buf = NULL;
    static int   *h_expert_indices = NULL;
    static float *h_expert_weights = NULL;
    static float *h_cpu_expert_delta = NULL;
    static float *h_cpu_thread_out[8] = {NULL};
    static float *h_cpu_thread_gate[8] = {NULL};
    static float *h_cpu_thread_up[8] = {NULL};

    /* Pipelined routing: compute at layer N-1 end, use at layer N start */
    static int   *h_expert_indices_current = NULL;
    static float *h_expert_weights_current = NULL;
    static int   *h_expert_indices_next = NULL;
    static float *h_expert_weights_next = NULL;
    static int    h_expert_current_valid = 0;
    static int    h_expert_next_valid = 0;

    static float *h_logits = NULL;

    float *&h_hidden = g_h_hidden;  /* alias to file-scope for RMS access */

    /* Get the pre-allocated buffer pool (initialized in main) */
    static const bread_buffer_pool_t *pool = NULL;
    if (!pool) {
        pool = bread_buffer_pool_get();
        if (!pool) {
            fprintf(stderr, "one_layer_forward: buffer pool not initialized\n");
            exit(1);
        }
    }

    /* Initialize pipelined routing buffers if SSD streaming mode enabled */
    if (bread_get_ssd_streaming_mode() && !h_expert_indices_current) {
        h_expert_indices_current = (int *)malloc((size_t)cfg->top_k * sizeof(int));
        h_expert_weights_current = (float *)malloc((size_t)cfg->top_k * sizeof(float));
        h_expert_indices_next = (int *)malloc((size_t)cfg->top_k * sizeof(int));
        h_expert_weights_next = (float *)malloc((size_t)cfg->top_k * sizeof(float));
        if (!h_expert_indices_current || !h_expert_weights_current ||
            !h_expert_indices_next || !h_expert_weights_next) {
            fprintf(stderr, "one_layer_forward: failed to allocate pipelined routing buffers\n");
            exit(1);
        }
        h_expert_current_valid = 0;
        h_expert_next_valid = 0;
    }

    /* Alias device and host buffers from pool on every call
       (they're static pointers in the function, but pool provides them) */
    if (!d_normed) {  /* Only alias once on first call */
        d_normed   = pool->d_normed;
        d_normed2  = pool->d_normed2;
        d_q        = pool->d_q;
        d_k        = pool->d_k;
        d_v        = pool->d_v;
        d_attn_out = pool->d_attn_out;
        d_o_out    = pool->d_o_out;
        d_sg       = pool->d_sg;
        d_su       = pool->d_su;
        d_sh_out   = pool->d_sh_out;
        d_eg       = pool->d_eg[0];
        d_eu       = pool->d_eu[0];
        d_eo       = pool->d_eo[0];
        d_qkv      = pool->d_qkv;
        d_z        = pool->d_z;
        d_alpha    = pool->d_alpha;
        d_beta     = pool->d_beta;

        h_qkv      = pool->h_qkv;
        h_z        = pool->h_z;
        h_alpha    = pool->h_alpha;
        h_beta     = pool->h_beta;
        h_conv_out = pool->h_conv_out;
        h_attn_out = pool->h_attn_out;
        h_head_tmp = pool->h_head_tmp;
        h_ssm_qrep = pool->h_ssm_qrep;
        h_ssm_krep = pool->h_ssm_krep;
        h_ssm_sk   = pool->h_ssm_sk;
        h_ssm_d    = pool->h_ssm_d;
        h_q_full   = pool->h_q_full;
        h_kv_k     = pool->h_kv_k;
        h_kv_v     = pool->h_kv_v;
        h_q_score  = pool->h_q_score;
        h_q_gate   = pool->h_q_gate;
        h_scores   = pool->h_scores;
        h_normed   = pool->h_normed;
        h_normed2  = pool->h_normed2;
        h_o_cpu    = pool->h_o_cpu;
        h_sg_cpu   = pool->h_sg_cpu;
        h_su_cpu   = pool->h_su_cpu;
        h_sh_cpu   = pool->h_sh_cpu;
        h_eg_cpu   = pool->h_eg_cpu;
        h_eu_cpu   = pool->h_eu_cpu;
        h_eo_cpu   = pool->h_eo_cpu;
        h_hidden_half  = pool->h_hidden_half;
        h_attn_half_buf = pool->h_attn_half_buf;
        h_expert_indices = pool->h_expert_indices;
        h_expert_weights = pool->h_expert_weights;
        h_logits   = pool->h_logits;
    }

    /* Per-layer state (still static, allocated once) */
    static float *ssm_conv_state[LOADER_MAX_LAYERS] = {0};
    static float *ssm_state[LOADER_MAX_LAYERS] = {0};
    static float *kv_k_cache[LOADER_MAX_LAYERS] = {0};
    static float *kv_v_cache[LOADER_MAX_LAYERS] = {0};
    static int    kv_cache_len[LOADER_MAX_LAYERS] = {0};
    static int    cpu_hidden_pos = -1;
    static int    cpu_hidden_layer = -1;
    static int    last_pos = -1;  /* Detect new query */

    /* Reset KV cache counters when starting a new query (pos decreases or resets) */
    if (layer_idx == 0 && (last_pos >= 0 && pos <= last_pos)) {
        fprintf(stderr, "[RESET] Query state detected: last_pos=%d pos=%d - resetting KV cache\n",
                last_pos, pos);
        memset(kv_cache_len, 0, sizeof(kv_cache_len));
    }
    if (layer_idx == 0) last_pos = pos;

    /* Initialize per-layer state buffers on first call */
    static int initialized = 0;
    if (!initialized) {
        for (int layer = 0; layer < cfg->num_layers; layer++) {
            if (bread_layer_is_full_attention(layer)) {
                kv_k_cache[layer] = (float *)calloc(
                    cfg->kv_cache_len * cfg->kv_proj_dim, sizeof(float));
                kv_v_cache[layer] = (float *)calloc(
                    cfg->kv_cache_len * cfg->kv_proj_dim, sizeof(float));
                if (!kv_k_cache[layer] || !kv_v_cache[layer]) {
                    fprintf(stderr, "one_layer_forward: KV cache alloc failed for layer %d\n", layer);
                    exit(1);
                }
            } else {
                ssm_conv_state[layer] = (float *)calloc(
                    (cfg->ssm_conv_kernel - 1) * cfg->ssm_qkv_dim, sizeof(float));
                ssm_state[layer] = (float *)calloc(
                    cfg->ssm_num_v_heads * cfg->ssm_head_dim * cfg->ssm_head_dim, sizeof(float));
                if (!ssm_conv_state[layer] || !ssm_state[layer]) {
                    fprintf(stderr, "one_layer_forward: SSM state alloc failed for layer %d\n", layer);
                    exit(1);
                }
            }
        }
        h_cpu_expert_delta = (float *)malloc(H * sizeof(float));
        if (!h_cpu_expert_delta) {
            fprintf(stderr, "one_layer_forward: CPU expert delta alloc failed\n");
            exit(1);
        }
        for (int k = 0; k < 8; k++) {
            h_cpu_thread_out[k] = (float *)malloc(H * sizeof(float));
            h_cpu_thread_gate[k] = (float *)malloc(cfg->expert_inter * sizeof(float));
            h_cpu_thread_up[k] = (float *)malloc(cfg->expert_inter * sizeof(float));
            if (!h_cpu_thread_out[k] || !h_cpu_thread_gate[k] || !h_cpu_thread_up[k]) {
                fprintf(stderr, "one_layer_forward: CPU expert scratch alloc failed\n");
                exit(1);
            }
        }
        initialized = 1;
    }

    if (bread_get_boring_mode()) {
        int   *expert_indices = NULL;
        float *expert_weights = NULL;
        float shared_gate_score = 0.0f;

        if (layer_idx == 0 || pos != cpu_hidden_pos || cpu_hidden_layer < 0 || cpu_hidden_layer >= cfg->num_layers - 1) {
            vram_half_to_cpu_float(d_hidden, h_hidden, H);
        }

        snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", layer_idx);
        cpu_rms_norm_weighted(h_hidden, tensor_ram_f32(L, g, nm),
                              h_normed, H, cfg->rms_eps);

        if (is_full) {
            int heads_per_kv = cfg->num_q_heads / cfg->num_kv_heads;
            int kv_len;
            const int attn_trace = bread_get_trace_debug() &&
                                   bread_get_trace_pos() == pos &&
                                   layer_idx == 3;

            snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", layer_idx);
            cpu_named_matvec(L, g, nm, h_normed, h_q_full, cfg->q_proj_dim, H);
            snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", layer_idx);
            cpu_named_matvec(L, g, nm, h_normed, h_kv_k, cfg->kv_proj_dim, H);
            snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", layer_idx);
            cpu_named_matvec(L, g, nm, h_normed, h_kv_v, cfg->kv_proj_dim, H);

            split_full_attn_q(cfg, h_q_full, h_q_score, h_q_gate);

            snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", layer_idx);
            apply_per_head_rms_norm(h_q_score, tensor_ram_f32(L, g, nm),
                                    cfg->num_q_heads, cfg->head_dim_qk, cfg->rms_eps);
            snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", layer_idx);
            apply_per_head_rms_norm(h_kv_k, tensor_ram_f32(L, g, nm),
                                    cfg->num_kv_heads, cfg->head_dim_qk, cfg->rms_eps);

            apply_rotary_emb(cfg, h_q_score, h_kv_k, pos);

            if (attn_trace) {
                trace_vec_stats("bread.layer3.q_score_head0", h_q_score, cfg->head_dim_qk, 8);
                trace_vec_stats("bread.layer3.k_head0", h_kv_k, cfg->head_dim_qk, 8);
                trace_vec_stats("bread.layer3.v_head0", h_kv_v, cfg->head_dim_v, 8);
                trace_vec_stats("bread.layer3.q_gate_raw_head0", h_q_gate, cfg->head_dim_v, 8);
                trace_sigmoid_stats("bread.layer3.q_gate_sigmoid_head0", h_q_gate, cfg->head_dim_v, 8);
            }

            if (kv_cache_len[layer_idx] >= cfg->kv_cache_len) {
                fprintf(stderr, "one_layer_forward: KV cache full at layer %d\n", layer_idx);
                exit(1);
            }

            memcpy(kv_k_cache[layer_idx] + (size_t)kv_cache_len[layer_idx] * cfg->kv_proj_dim,
                   h_kv_k, cfg->kv_proj_dim * sizeof(float));
            memcpy(kv_v_cache[layer_idx] + (size_t)kv_cache_len[layer_idx] * cfg->kv_proj_dim,
                   h_kv_v, cfg->kv_proj_dim * sizeof(float));
            kv_cache_len[layer_idx]++;
            kv_len = kv_cache_len[layer_idx];

            memset(h_attn_out, 0, cfg->attn_out_dim * sizeof(float));
            for (int h = 0; h < cfg->num_q_heads; h++) {
                int kv_h = h / heads_per_kv;
                float *qh = h_q_score + h * cfg->head_dim_qk;
                float *oh = h_attn_out + h * cfg->head_dim_v;
                for (int p = 0; p < kv_len; p++) {
                    float *kp = kv_k_cache[layer_idx] + (size_t)p * cfg->kv_proj_dim + kv_h * cfg->head_dim_qk;
                    float dot = 0.0f;
                    for (int d = 0; d < cfg->head_dim_qk; d++) dot += qh[d] * kp[d];
                    h_scores[p] = dot / sqrtf((float)cfg->head_dim_qk);
                }
                cpu_softmax(h_scores, kv_len);
                if (attn_trace && h == 0) {
                    trace_vec_stats("bread.layer3.head0.attn_scores_softmax", h_scores, kv_len, 8);
                    trace_vec_stats("bread.layer3.head0.v_cache_current", kv_v_cache[layer_idx] + (size_t)(kv_len - 1) * cfg->kv_proj_dim + kv_h * cfg->head_dim_v,
                                    cfg->head_dim_v, 8);
                }
                for (int p = 0; p < kv_len; p++) {
                    float *vp = kv_v_cache[layer_idx] + (size_t)p * cfg->kv_proj_dim + kv_h * cfg->head_dim_v;
                    for (int d = 0; d < cfg->head_dim_v; d++) oh[d] += h_scores[p] * vp[d];
                }
                if (attn_trace && h == 0) {
                    memcpy(h_head_tmp, oh, cfg->head_dim_v * sizeof(float));
                    trace_vec_stats("bread.layer3.head0.attn_pre_gate", h_head_tmp, cfg->head_dim_v, 8);
                }
                for (int d = 0; d < cfg->head_dim_v; d++) {
                    oh[d] *= cpu_sigmoid(h_q_gate[h * cfg->head_dim_v + d]);
                }
                if (attn_trace && h == 0) {
                    trace_vec_stats("bread.layer3.head0.attn_post_gate", oh, cfg->head_dim_v, 8);
                }
            }

            if (attn_trace) {
                trace_vec_stats("bread.layer3.attn_out_all_heads_post_gate", h_attn_out, cfg->attn_out_dim, 8);
            }

            snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", layer_idx);
            cpu_named_matvec(L, g, nm, h_attn_out, h_o_cpu, H, cfg->attn_out_dim);
            if (attn_trace) {
                trace_vec_stats("bread.layer3.o_proj_out", h_o_cpu, H, 8);
            }
            g_last_branch_rms = cpu_rms_f32(h_o_cpu, H);
        } else {
            const float *conv_w;
            const float *ssm_a;
            const float *ssm_dt;
            const float *ssm_norm_w;
            float *conv_state;
            float *layer_state;
            const int key_dim = cfg->ssm_head_dim;
            const int value_dim = cfg->ssm_head_dim;
            const int num_k_heads = cfg->ssm_num_k_heads;
            const int num_v_heads = cfg->ssm_num_v_heads;
            const float inv_scale = 1.0f / sqrtf((float)key_dim);
            const int ssm_trace = bread_get_trace_debug() &&
                                  bread_get_trace_pos() == pos &&
                                  layer_idx == 0;

            snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight", layer_idx);
            cpu_named_matvec(L, g, nm, h_normed, h_qkv, cfg->ssm_qkv_dim, H);
            snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight", layer_idx);
            cpu_named_matvec(L, g, nm, h_normed, h_z, cfg->ssm_z_dim, H);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_alpha.weight", layer_idx);
            cpu_named_matvec(L, g, nm, h_normed, h_alpha, cfg->ssm_num_v_heads, H);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_beta.weight", layer_idx);
            cpu_named_matvec(L, g, nm, h_normed, h_beta, cfg->ssm_num_v_heads, H);

            snprintf(nm, sizeof(nm), "blk.%d.ssm_conv1d.weight", layer_idx);
            conv_w = tensor_ram_f32(L, g, nm);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_a", layer_idx);
            ssm_a = tensor_ram_f32(L, g, nm);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_dt", layer_idx);
            ssm_dt = tensor_ram_f32(L, g, nm);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_norm.weight", layer_idx);
            ssm_norm_w = tensor_ram_f32(L, g, nm);
            conv_state = ssm_conv_state[layer_idx];
            layer_state = ssm_state[layer_idx];

            cpu_conv1d_step(conv_state, h_qkv, conv_w, h_conv_out,
                            cfg->ssm_qkv_dim, cfg->ssm_conv_kernel);
            if (ssm_trace) {
                trace_vec_stats("bread.layer0.qkv_proj", h_qkv, cfg->ssm_qkv_dim, 8);
                trace_vec_stats("bread.layer0.z_proj", h_z, cfg->ssm_z_dim, 8);
                trace_vec_stats("bread.layer0.alpha_proj", h_alpha, cfg->ssm_num_v_heads, 8);
                trace_vec_stats("bread.layer0.beta_proj", h_beta, cfg->ssm_num_v_heads, 8);
                trace_vec_stats("bread.layer0.conv_raw", h_conv_out, cfg->ssm_qkv_dim, 8);
            }
            memmove(conv_state, conv_state + cfg->ssm_qkv_dim,
                    (cfg->ssm_conv_kernel - 2) * cfg->ssm_qkv_dim * sizeof(float));
            memcpy(conv_state + (cfg->ssm_conv_kernel - 2) * cfg->ssm_qkv_dim,
                   h_qkv, cfg->ssm_qkv_dim * sizeof(float));
            cpu_silu_inplace(h_conv_out, cfg->ssm_qkv_dim);

            {
                float *lin_q = h_conv_out;
                float *lin_k = h_conv_out + num_k_heads * key_dim;
                float *lin_v = h_conv_out + 2 * num_k_heads * key_dim;
                for (int h = 0; h < num_k_heads; h++) {
                    float *qh = lin_q + h * key_dim;
                    cpu_l2_norm_bare(qh, key_dim, cfg->rms_eps);
                    for (int d = 0; d < key_dim; d++) qh[d] *= inv_scale;
                }
                for (int h = 0; h < num_k_heads; h++) {
                    float *kh = lin_k + h * key_dim;
                    cpu_l2_norm_bare(kh, key_dim, cfg->rms_eps);
                }
                cpu_repeat_heads(h_ssm_qrep, lin_q, num_k_heads, num_v_heads, key_dim);
                cpu_repeat_heads(h_ssm_krep, lin_k, num_k_heads, num_v_heads, key_dim);
                if (ssm_trace) {
                    trace_vec_stats("bread.layer0.q_conv_norm_head0", h_ssm_qrep, key_dim, 8);
                    trace_vec_stats("bread.layer0.k_conv_norm_head0", h_ssm_krep, key_dim, 8);
                    trace_vec_stats("bread.layer0.v_conv_head0", lin_v, value_dim, 8);
                }

                memset(h_attn_out, 0, cfg->ssm_z_dim * sizeof(float));
                for (int vh = 0; vh < num_v_heads; vh++) {
                    float gate = cpu_softplus(h_alpha[vh] + ssm_dt[vh]) * ssm_a[vh];
                    float decay = expf(gate);
                    float beta_gate = cpu_sigmoid(h_beta[vh]);
                    float *S = layer_state + (size_t)vh * value_dim * key_dim;
                    float *q_h = h_ssm_qrep + (size_t)vh * key_dim;
                    float *k_h = h_ssm_krep + (size_t)vh * key_dim;
                    float *v_h = lin_v + vh * value_dim;
                    float *o_h = h_attn_out + vh * value_dim;

                    if (ssm_trace && vh == 0) {
                        float gate_buf[1] = { gate };
                        float decay_buf[1] = { decay };
                        float beta_buf[1] = { beta_gate };
                        trace_vec_stats("bread.layer0.vh0.state_before_row0", S, key_dim, 8);
                        trace_vec_stats("bread.layer0.vh0.q_head", q_h, key_dim, 8);
                        trace_vec_stats("bread.layer0.vh0.k_head", k_h, key_dim, 8);
                        trace_vec_stats("bread.layer0.vh0.v_head", v_h, value_dim, 8);
                        trace_vec_stats("bread.layer0.vh0.gate", gate_buf, 1, 1);
                        trace_vec_stats("bread.layer0.vh0.decay", decay_buf, 1, 1);
                        trace_vec_stats("bread.layer0.vh0.beta", beta_buf, 1, 1);
                    }

                    cpu_delta_net_autoregressive_step(
                        q_h, k_h, v_h, gate, beta_gate,
                        S, h_head_tmp, value_dim, key_dim,
                        h_ssm_sk, h_ssm_d);
                    if (ssm_trace && vh == 0) {
                        trace_vec_stats("bread.layer0.vh0.state_after_row0", S, key_dim, 8);
                        trace_vec_stats("bread.layer0.vh0.readout_pre_norm", h_head_tmp, value_dim, 8);
                    }
                    cpu_gated_rms_norm(h_head_tmp, h_z + vh * value_dim,
                                       ssm_norm_w, o_h, value_dim, cfg->rms_eps);
                    if (ssm_trace && vh == 0) {
                        trace_vec_stats("bread.layer0.vh0.readout_post_norm", o_h, value_dim, 8);
                    }
                }
            }

            snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight", layer_idx);
            cpu_named_matvec(L, g, nm, h_attn_out, h_o_cpu, H, cfg->ssm_z_dim);
            if (bread_get_force_ssm_zero()) {
                memset(h_o_cpu, 0, (size_t)H * sizeof(float));
            }
            g_last_branch_rms = cpu_rms_f32(h_o_cpu, H);
        }

        for (int i = 0; i < H; i++) h_hidden[i] += h_o_cpu[i];

        snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", layer_idx);
        cpu_rms_norm_weighted(h_hidden, tensor_ram_f32(L, g, nm),
                              h_normed2, H, cfg->rms_eps);

        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_shexp.weight", layer_idx);
        cpu_named_matvec(L, g, nm, h_normed2, h_sg_cpu, cfg->shared_inter, H);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_up_shexp.weight", layer_idx);
        cpu_named_matvec(L, g, nm, h_normed2, h_su_cpu, cfg->shared_inter, H);

        expert_indices = h_expert_indices;     /* pre-allocated static buffer */
        expert_weights = h_expert_weights;     /* pre-allocated static buffer */
        {
            float *logits = h_logits;          /* pre-allocated static buffer */
            const float *router_w;
            const half *shared_gate_w = NULL;
            snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", layer_idx);
            router_w = tensor_ram_f32(L, g, nm);
            snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp_shexp.weight", layer_idx);
            if (gguf_find_tensor(g, nm)) shared_gate_w = tensor_ram_f16(L, g, nm);
            memset(logits, 0, cfg->num_experts * sizeof(float));
            for (int i = 0; i < cfg->num_experts; i++) {
                for (int j = 0; j < H; j++) logits[i] += router_w[i * H + j] * h_normed2[j];
            }
            if (shared_gate_w) {
                for (int j = 0; j < H; j++) shared_gate_score += __half2float(shared_gate_w[j]) * h_normed2[j];
            }
            cpu_softmax(logits, cfg->num_experts);
            cpu_topk(logits, cfg->num_experts, cfg->top_k, expert_indices, expert_weights);
        }

        cpu_swiglu(h_sg_cpu, h_su_cpu, h_sg_cpu, cfg->shared_inter);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_down_shexp.weight", layer_idx);
        cpu_named_matvec(L, g, nm, h_sg_cpu, h_sh_cpu, H, cfg->shared_inter);
        {
            float shared_weight = cpu_sigmoid(shared_gate_score);
            for (int i = 0; i < H; i++) h_hidden[i] += shared_weight * h_sh_cpu[i];
        }

        {
            const loader_layer_info_t *li = &L->layers[layer_idx];
            for (int k = 0; k < cfg->top_k; k++) {
                const int expert_idx = expert_indices[k];
                const uint8_t *gate_src;
                const uint8_t *up_src;
                const uint8_t *down_src;
                if (!li->valid) {
                    fprintf(stderr, "one_layer_forward: layer %d has no expert tensors\n", layer_idx);
                    exit(1);
                }
                gate_src = li->gate_base + (uint64_t)expert_idx * li->gate_expert_bytes;
                up_src   = li->up_base   + (uint64_t)expert_idx * li->up_expert_bytes;
                down_src = li->down_base + (uint64_t)expert_idx * li->down_expert_bytes;
                cpu_tensor_matvec(gate_src, li->gate_type, h_normed2, h_eg_cpu, cfg->expert_inter, H);
                cpu_tensor_matvec(up_src,   li->up_type,   h_normed2, h_eu_cpu, cfg->expert_inter, H);
                cpu_swiglu(h_eg_cpu, h_eu_cpu, h_eg_cpu, cfg->expert_inter);
                cpu_tensor_matvec(down_src, li->down_type, h_eg_cpu, h_eo_cpu, H, cfg->expert_inter);
                for (int i = 0; i < H; i++) h_hidden[i] += expert_weights[k] * h_eo_cpu[i];
            }
        }

        cpu_hidden_pos = pos;
        cpu_hidden_layer = layer_idx;
        if (layer_idx == cfg->num_layers - 1) {
            for (int i = 0; i < H; i++) h_hidden_half[i] = __float2half(h_hidden[i]);
            CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden_half, H * sizeof(half), cudaMemcpyHostToDevice));
        }
        return;
    }

    /* ================================================================
     * PHASE 0: pre-attention RMSNorm
     *   d_normed = rmsnorm(d_hidden, attn_norm_w)
     * d_hidden is kept as the residual.
     * ================================================================ */
    {
        float *d_attn_norm_w;
        if (wc) {
            d_attn_norm_w = (float *)wc->layers[layer_idx].attn_norm_w;
        } else {
            snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", layer_idx);
            d_attn_norm_w = (float *)load_vram(L, g, nm);
        }

        /* Copy hidden → normed, then normalise normed in-place */
        int blocks = (H + 255) / 256;
        copy_half<<<blocks, 256, 0, stream_a>>>(d_normed, d_hidden, H);
        rmsnorm_inplace<<<1, 256, 0, stream_a>>>(d_normed, d_attn_norm_w,
                                                  H, cfg->rms_eps);
        /* Record event but don't sync yet; let GPU continue if possible */
        CUDA_CHECK(cudaEventRecord(pool->normed2_ready_event, stream_a));
        if (!wc) cudaFree(d_attn_norm_w);
    }

    /* ================================================================
     * CMD1: Attention projections
     *
     * Full-attn: Q/K/V matvec on d_normed.
     * SSM stub:  skip — d_attn_out will be zeroed so residual unchanged.
     * ================================================================ */
    if (is_full) {
        void *d_qw, *d_kw, *d_vw;
        if (wc) {
            d_qw = wc->layers[layer_idx].attn_q_w;
            d_kw = wc->layers[layer_idx].attn_k_w;
            d_vw = wc->layers[layer_idx].attn_v_w;
        } else {
            snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", layer_idx);
            d_qw = load_vram(L, g, nm);
            snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", layer_idx);
            d_kw = load_vram(L, g, nm);
            snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", layer_idx);
            d_vw = load_vram(L, g, nm);
        }

        bread_matvec(d_qw, d_normed, d_q,
                     cfg->q_proj_dim,  H, GGML_TYPE_Q4_K, stream_a);
        bread_matvec(d_kw, d_normed, d_k,
                     cfg->kv_proj_dim, H, GGML_TYPE_Q4_K, stream_a);
        bread_matvec(d_vw, d_normed, d_v,
                     cfg->kv_proj_dim, H, GGML_TYPE_Q6_K, stream_a);

        /* Sync only when CPU needs Q/K/V data below */
        if (!wc) { cudaFree(d_qw); cudaFree(d_kw); cudaFree(d_vw); }
    }

    /* ================================================================
     * CPU attention
     *
     * Full-attn layers use CPU-side incremental RoPE + KV cache.
     * SSM layers use the gated delta recurrence.
     * ================================================================ */
    if (is_full) {
        const float *q_norm_w;
        const float *k_norm_w;
        int heads_per_kv = cfg->num_q_heads / cfg->num_kv_heads;
        int kv_len;

        /* Wait for Q/K/V projections to complete before copying to host */
        CUDA_CHECK(cudaStreamSynchronize(stream_a));

        vram_half_to_cpu_float(d_q, h_q_full, cfg->q_proj_dim);
        vram_half_to_cpu_float(d_k, h_kv_k, cfg->kv_proj_dim);
        vram_half_to_cpu_float(d_v, h_kv_v, cfg->kv_proj_dim);
        split_full_attn_q(cfg, h_q_full, h_q_score, h_q_gate);

        snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", layer_idx);
        q_norm_w = tensor_ram_f32(L, g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", layer_idx);
        k_norm_w = tensor_ram_f32(L, g, nm);
        apply_per_head_rms_norm(h_q_score, q_norm_w,
                                cfg->num_q_heads, cfg->head_dim_qk, cfg->rms_eps);
        apply_per_head_rms_norm(h_kv_k, k_norm_w,
                                cfg->num_kv_heads, cfg->head_dim_qk, cfg->rms_eps);

        apply_rotary_emb(cfg, h_q_score, h_kv_k, pos);

        if (kv_cache_len[layer_idx] >= cfg->kv_cache_len) {
            fprintf(stderr, "one_layer_forward: KV cache full at layer %d\n", layer_idx);
            exit(1);
        }

        memcpy(kv_k_cache[layer_idx] + (size_t)kv_cache_len[layer_idx] * cfg->kv_proj_dim,
               h_kv_k, cfg->kv_proj_dim * sizeof(float));
        memcpy(kv_v_cache[layer_idx] + (size_t)kv_cache_len[layer_idx] * cfg->kv_proj_dim,
               h_kv_v, cfg->kv_proj_dim * sizeof(float));
        kv_cache_len[layer_idx]++;
        kv_len = kv_cache_len[layer_idx];

        memset(h_attn_out, 0, cfg->attn_out_dim * sizeof(float));
        for (int h = 0; h < cfg->num_q_heads; h++) {
            int kv_h = h / heads_per_kv;
            float *qh = h_q_score + h * cfg->head_dim_qk;
            float *oh = h_attn_out + h * cfg->head_dim_v;
            for (int p = 0; p < kv_len; p++) {
                float *kp = kv_k_cache[layer_idx] + (size_t)p * cfg->kv_proj_dim + kv_h * cfg->head_dim_qk;
                float dot = 0.0f;
                for (int d = 0; d < cfg->head_dim_qk; d++)
                    dot += qh[d] * kp[d];
                h_scores[p] = dot / sqrtf((float)cfg->head_dim_qk);
            }
            cpu_softmax(h_scores, kv_len);
            for (int p = 0; p < kv_len; p++) {
                float *vp = kv_v_cache[layer_idx] + (size_t)p * cfg->kv_proj_dim + kv_h * cfg->head_dim_v;
                for (int d = 0; d < cfg->head_dim_v; d++)
                    oh[d] += h_scores[p] * vp[d];
            }
            for (int d = 0; d < cfg->head_dim_v; d++) {
                float gate = cpu_sigmoid(h_q_gate[h * cfg->head_dim_v + d]);
                oh[d] *= gate;
            }
        }

        {
            half *h_attn_half = h_attn_half_buf;  /* pre-allocated static buffer */
            for (int i = 0; i < cfg->attn_out_dim; i++)
                h_attn_half[i] = __float2half(h_attn_out[i]);
            CUDA_CHECK(cudaMemcpy(d_attn_out, h_attn_half,
                                  cfg->attn_out_dim * sizeof(half),
                                  cudaMemcpyHostToDevice));
        }
        /* Don't sync; GPU will naturally synchronize when it needs attn_out */
    } else {
        const float *conv_w;
        const float *ssm_a;
        const float *ssm_dt;
        const float *ssm_norm_w;
        float *conv_state;
        float *layer_state;

        void *d_qkv_w, *d_gate_w, *d_alpha_w, *d_beta_w;
        if (wc) {
            d_qkv_w = wc->layers[layer_idx].attn_qkv_w;
            d_gate_w = wc->layers[layer_idx].attn_gate_w;
            d_alpha_w = wc->layers[layer_idx].ssm_alpha_w;
            d_beta_w = wc->layers[layer_idx].ssm_beta_w;
        } else {
            snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight", layer_idx);
            d_qkv_w = load_vram(L, g, nm);
            snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight", layer_idx);
            d_gate_w = load_vram(L, g, nm);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_alpha.weight", layer_idx);
            d_alpha_w = load_vram(L, g, nm);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_beta.weight", layer_idx);
            d_beta_w = load_vram(L, g, nm);
        }

        bread_matvec(d_qkv_w, d_normed, d_qkv,
                     cfg->ssm_qkv_dim, H, GGML_TYPE_Q4_K, stream_a);
        bread_matvec(d_gate_w, d_normed, d_z,
                     cfg->ssm_z_dim, H, GGML_TYPE_Q4_K, stream_a);
        bread_matvec(d_alpha_w, d_normed, d_alpha,
                     cfg->ssm_num_v_heads, H, GGML_TYPE_Q4_K, stream_a);
        bread_matvec(d_beta_w, d_normed, d_beta,
                     cfg->ssm_num_v_heads, H, GGML_TYPE_Q4_K, stream_a);

        /* Sync before D2H copy of SSM projections */
        CUDA_CHECK(cudaStreamSynchronize(stream_a));

        if (!wc) {
            cudaFree(d_qkv_w);
            cudaFree(d_gate_w);
            cudaFree(d_alpha_w);
            cudaFree(d_beta_w);
        }

        vram_half_to_cpu_float(d_qkv, h_qkv, cfg->ssm_qkv_dim);
        vram_half_to_cpu_float(d_z, h_z, cfg->ssm_z_dim);
        vram_half_to_cpu_float(d_alpha, h_alpha, cfg->ssm_num_v_heads);
        vram_half_to_cpu_float(d_beta, h_beta, cfg->ssm_num_v_heads);

        snprintf(nm, sizeof(nm), "blk.%d.ssm_conv1d.weight", layer_idx);
        conv_w = tensor_ram_f32(L, g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_a", layer_idx);
        ssm_a = tensor_ram_f32(L, g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_dt", layer_idx);
        ssm_dt = tensor_ram_f32(L, g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_norm.weight", layer_idx);
        ssm_norm_w = tensor_ram_f32(L, g, nm);
        conv_state = ssm_conv_state[layer_idx];
        layer_state = ssm_state[layer_idx];
        if (!conv_state || !layer_state) {
            fprintf(stderr, "one_layer_forward: missing SSM state for layer %d\n", layer_idx);
            exit(1);
        }

        cpu_conv1d_step(conv_state, h_qkv, conv_w, h_conv_out,
                        cfg->ssm_qkv_dim, cfg->ssm_conv_kernel);
        memmove(conv_state, conv_state + cfg->ssm_qkv_dim,
                (cfg->ssm_conv_kernel - 2) * cfg->ssm_qkv_dim * sizeof(float));
        memcpy(conv_state + (cfg->ssm_conv_kernel - 2) * cfg->ssm_qkv_dim,
               h_qkv, cfg->ssm_qkv_dim * sizeof(float));
        cpu_silu_inplace(h_conv_out, cfg->ssm_qkv_dim);

        {
            const int key_dim = cfg->ssm_head_dim;
            const int value_dim = cfg->ssm_head_dim;
            const int num_k_heads = cfg->ssm_num_k_heads;
            const int num_v_heads = cfg->ssm_num_v_heads;
            const float inv_scale = 1.0f / sqrtf((float)key_dim);
            float *lin_q = h_conv_out;
            float *lin_k = h_conv_out + num_k_heads * key_dim;
            float *lin_v = h_conv_out + 2 * num_k_heads * key_dim;

            for (int h = 0; h < num_k_heads; h++) {
                float *qh = lin_q + h * key_dim;
                cpu_l2_norm_bare(qh, key_dim, cfg->rms_eps);
                for (int d = 0; d < key_dim; d++) qh[d] *= inv_scale;
            }
            for (int h = 0; h < num_k_heads; h++) {
                float *kh = lin_k + h * key_dim;
                cpu_l2_norm_bare(kh, key_dim, cfg->rms_eps);
            }
            cpu_repeat_heads(h_ssm_qrep, lin_q, num_k_heads, num_v_heads, key_dim);
            cpu_repeat_heads(h_ssm_krep, lin_k, num_k_heads, num_v_heads, key_dim);

            for (int vh = 0; vh < num_v_heads; vh++) {
                /* Bounds validation for SSM state access */
                size_t expected_state_size = (size_t)num_v_heads * value_dim * key_dim;
                size_t offset = (size_t)vh * value_dim * key_dim;
                if (offset + value_dim * key_dim > expected_state_size) {
                    fprintf(stderr, "ERROR: SSM state out of bounds at token %d layer %d vh %d\n"
                            "  offset=%zu expected_size=%zu total_needed=%zu\n",
                            pos, layer_idx, vh, offset, expected_state_size, offset + value_dim * key_dim);
                    exit(1);
                }

                float gate = cpu_softplus(h_alpha[vh] + ssm_dt[vh]) * ssm_a[vh];
                float beta_gate = cpu_sigmoid(h_beta[vh]);
                float *S = layer_state + offset;
                float *q_h = h_ssm_qrep + (size_t)vh * key_dim;
                float *k_h = h_ssm_krep + (size_t)vh * key_dim;
                float *v_h = lin_v + vh * value_dim;
                float *o_h = h_attn_out + vh * value_dim;

                cpu_delta_net_autoregressive_step(
                    q_h, k_h, v_h, gate, beta_gate,
                    S, h_head_tmp, value_dim, key_dim,
                    h_ssm_sk, h_ssm_d);

                cpu_gated_rms_norm(h_head_tmp, h_z + vh * value_dim,
                                   ssm_norm_w, o_h, value_dim, cfg->rms_eps);
            }
        }

        {
            half *h_attn_half = h_attn_half_buf;  /* pre-allocated static buffer */
            for (int i = 0; i < cfg->ssm_z_dim; i++)
                h_attn_half[i] = __float2half(h_attn_out[i]);
            CUDA_CHECK(cudaMemcpy(d_attn_out, h_attn_half,
                                  cfg->ssm_z_dim * sizeof(half),
                                  cudaMemcpyHostToDevice));
        }
        /* Don't sync; GPU will naturally synchronize when needed */
    }

    /* ================================================================
     * CMD2a: o_proj + residual add (full-attn only)
     *
     * attn_out [ATTN_OUT_DIM] → o_proj → [H]
     * hidden += o_proj_out
     * ================================================================ */
    if (is_full) {
        void *d_ow;
        if (wc) {
            d_ow = wc->layers[layer_idx].attn_output_w;
        } else {
            snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", layer_idx);
            d_ow = load_vram(L, g, nm);
        }

        bread_matvec(d_ow, d_attn_out, d_o_out,
                     H, cfg->attn_out_dim, GGML_TYPE_Q4_K, stream_a);
        if (!wc) cudaFree(d_ow);
        if (bread_get_trace_debug()) {
            /* Sync for RMS measurement */
            CUDA_CHECK(cudaStreamSynchronize(stream_a));
            g_last_branch_rms = device_half_rms(d_o_out, H);
        }

        int blocks = (H + 255) / 256;
        scale_accum<<<blocks, 256, 0, stream_a>>>(d_hidden, d_o_out, 1.0f, H);
        /* Don't sync; let subsequent work depend naturally */
    } else {
        void *d_sw;
        if (wc) {
            d_sw = wc->layers[layer_idx].ssm_out_w;
        } else {
            snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight", layer_idx);
            d_sw = load_vram(L, g, nm);
        }

        bread_matvec(d_sw, d_attn_out, d_o_out,
                     H, cfg->ssm_z_dim, GGML_TYPE_Q4_K, stream_a);
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
        if (!wc) cudaFree(d_sw);
        if (bread_get_force_ssm_zero()) {
            CUDA_CHECK(cudaMemset(d_o_out, 0, (size_t)H * sizeof(half)));
        }
        if (bread_get_trace_debug()) {
            g_last_branch_rms = device_half_rms(d_o_out, H);
        }

        int blocks = (H + 255) / 256;
        scale_accum<<<blocks, 256, 0, stream_a>>>(d_hidden, d_o_out, 1.0f, H);
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
    }

    /* ================================================================
     * CMD2b: post-attention RMSNorm → d_normed2 (pre-FFN input)
     * ================================================================ */
    {
        float *d_pan_w;
        if (wc) {
            d_pan_w = (float *)wc->layers[layer_idx].post_attn_norm_w;
        } else {
            snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", layer_idx);
            d_pan_w = (float *)load_vram(L, g, nm);
        }

        int blocks = (H + 255) / 256;
        copy_half<<<blocks, 256, 0, stream_a>>>(d_normed2, d_hidden, H);
        rmsnorm_inplace<<<1, 256, 0, stream_a>>>(d_normed2, d_pan_w,
                                                  H, cfg->rms_eps);
        /* Record event; CPU routing will implicitly wait via host copy if needed */
        if (!wc) cudaFree(d_pan_w);
    }

    /* ================================================================
     * Phase 4 Optimization: Overlap CPU routing with GPU work
     *
     * P4-3: Submit shared gate+up to stream_a WITHOUT syncing,
     * then run CPU routing (vram_copy + router matmul) while GPU executes.
     * Sync stream_a AFTER routing completes.
     * ================================================================ */

    /* CMD2c: submit shared gate+up to GPU — NO sync here */
    void *d_sg_w, *d_su_w;
    if (wc) {
        d_sg_w = wc->layers[layer_idx].ffn_gate_shexp_w;
        d_su_w = wc->layers[layer_idx].ffn_up_shexp_w;
    } else {
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_shexp.weight", layer_idx);
        d_sg_w = load_vram(L, g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_up_shexp.weight", layer_idx);
        d_su_w = load_vram(L, g, nm);
    }

    bread_matvec(d_sg_w, d_normed2, d_sg,
                 cfg->shared_inter, H, GGML_TYPE_Q4_K, stream_a);
    bread_matvec(d_su_w, d_normed2, d_su,
                 cfg->shared_inter, H, GGML_TYPE_Q4_K, stream_a);
    /* NOTE: no sync yet — gate+up runs on GPU while CPU does routing below */

    /* CPU routing: extract to route_layer() for prefetching support.
     * route_layer runs while stream_a executes shared gate+up matvecs (~8 μs GPU work).
     * vram_half_to_cpu_float uses cudaMemcpy(DeviceToHost) which syncs the default
     * stream only, not stream_a. Both can read d_normed2 concurrently (safe).
     */
    int   *expert_indices = h_expert_indices;   /* pre-allocated static buffer */
    float *expert_weights = h_expert_weights;   /* pre-allocated static buffer */
    float shared_gate_score = route_layer(L, g, layer_idx, d_normed2,
                                           expert_indices, expert_weights);

    /* In SSD streaming mode, initialize h_expert_indices_current for layer 0 */
    if (bread_get_ssd_streaming_mode() && layer_idx == 0 && h_expert_indices_current) {
        memcpy(h_expert_indices_current, expert_indices, (size_t)cfg->top_k * sizeof(int));
        memcpy(h_expert_weights_current, expert_weights, (size_t)cfg->top_k * sizeof(float));
        h_expert_current_valid = 1;
    }

    /* Sync stream_a for shared gate+up (should complete in ~8 μs, routing took ~500 μs → 0 wait) */
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
    if (!wc) { cudaFree(d_sg_w); cudaFree(d_su_w); }

    /* ================================================================
     * CMD3a: shared expert SwiGLU + down projection → accumulate
     * All expert weights are pre-cached in VRAM, so no DMA needed
     * ================================================================ */
    {
        int n_blocks_si = (cfg->shared_inter + 255) / 256;
        silu_mul_inplace<<<n_blocks_si, 256, 0, stream_a>>>(
            d_sg, d_su, cfg->shared_inter);

        void *d_sd_w;
        if (wc) {
            d_sd_w = wc->layers[layer_idx].ffn_down_shexp_w;
        } else {
            snprintf(nm, sizeof(nm), "blk.%d.ffn_down_shexp.weight", layer_idx);
            d_sd_w = load_vram(L, g, nm);
        }

        bread_matvec(d_sd_w, d_sg, d_sh_out,
                     H, cfg->shared_inter, GGML_TYPE_Q6_K, stream_a);
        if (!wc) cudaFree(d_sd_w);

        {
            float shared_weight = cpu_sigmoid(shared_gate_score);
            int blocks = (H + 255) / 256;
            scale_accum<<<blocks, 256, 0, stream_a>>>(d_hidden, d_sh_out, shared_weight, H);
            /* Don't sync; let expert loop work proceed on same stream */
        }
    }

    /* ================================================================
     * PIPELINED PREFETCH: Fire next layer's expert DMA on stream_c
     *
     * In SSD streaming mode, load next layer's experts in parallel with
     * this layer's computation on stream_a, hiding DMA latency.
     * ================================================================ */
    if (bread_get_ssd_streaming_mode() && L && layer_idx >= 0 &&
        layer_idx + 1 < cfg->num_layers && h_expert_indices_next && h_expert_next_valid) {
        loader_request_on_stream(L, layer_idx + 1, h_expert_indices_next,
                                 cfg->top_k, L->stream_c);
    }

    /* ================================================================
     * CMD3b: K active expert forwards → weighted accumulate into hidden
     * ================================================================ */
    {
        int n_blocks_ei = (cfg->expert_inter + 255) / 256;
        int n_blocks_h  = (H + 255) / 256;

        /* In SSD streaming mode, use pre-computed routing from previous layer
           Otherwise compute routing on-the-fly */
        int *active_expert_indices = expert_indices;
        float *active_expert_weights = expert_weights;

        if (bread_get_ssd_streaming_mode() && h_expert_indices_current && h_expert_current_valid) {
            active_expert_indices = h_expert_indices_current;
            active_expert_weights = h_expert_weights_current;
        }

        /* In SSD streaming mode, load required experts from host RAM before using them */
        if (bread_get_ssd_streaming_mode() && L) {
            loader_request(L, layer_idx, active_expert_indices, cfg->top_k);
            loader_sync(L);  /* Wait for DMA to complete */
        }

        if (bread_get_cpu_experts_mode()) {
            if (!L || !L->layers[layer_idx].valid) {
                fprintf(stderr, "ERROR: CPU experts mode requires valid host expert tensors for layer %d\n", layer_idx);
            } else {
                const loader_layer_info_t *li = &L->layers[layer_idx];
                vram_half_to_cpu_float(d_normed2, h_normed2, H);
                memset(h_cpu_expert_delta, 0, H * sizeof(float));
#pragma omp parallel for schedule(static, 1) num_threads(8)
                for (int k = 0; k < cfg->top_k; k++) {
                    const int expert_idx = active_expert_indices[k];
                    float *tmp_out = h_cpu_thread_out[k];
                    float *tmp_gate = h_cpu_thread_gate[k];
                    float *tmp_up = h_cpu_thread_up[k];
                    const uint8_t *gate_src = li->gate_base + (uint64_t)expert_idx * li->gate_expert_bytes;
                    const uint8_t *up_src = li->up_base + (uint64_t)expert_idx * li->up_expert_bytes;
                    const uint8_t *down_src = li->down_base + (uint64_t)expert_idx * li->down_expert_bytes;
                    cpu_tensor_matvec(gate_src, li->gate_type, h_normed2, tmp_gate, cfg->expert_inter, H);
                    cpu_tensor_matvec(up_src, li->up_type, h_normed2, tmp_up, cfg->expert_inter, H);
                    cpu_swiglu(tmp_gate, tmp_up, tmp_gate, cfg->expert_inter);
                    cpu_tensor_matvec(down_src, li->down_type, tmp_gate, tmp_out, H, cfg->expert_inter);
                    for (int i = 0; i < H; i++) tmp_out[i] *= active_expert_weights[k];
                }
                for (int k = 0; k < cfg->top_k; k++) {
                    float *tmp_out = h_cpu_thread_out[k];
                    for (int i = 0; i < H; i++) h_cpu_expert_delta[i] += tmp_out[i];
                }
                for (int i = 0; i < H; i++) h_hidden_half[i] = __float2half(h_cpu_expert_delta[i]);
                CUDA_CHECK(cudaMemcpyAsync(d_eo, h_hidden_half, H * sizeof(half),
                                           cudaMemcpyHostToDevice, stream_a));
                scale_accum<<<n_blocks_h, 256, 0, stream_a>>>(d_hidden, d_eo, 1.0f, H);
                CUDA_CHECK(cudaStreamSynchronize(stream_a));
            }
        } else {
            for (int k = 0; k < cfg->top_k; k++) {
                int expert_idx = active_expert_indices[k];

                void *d_gate = NULL;
                void *d_up = NULL;
                void *d_down = NULL;

                /* Load expert weights from either pre-cache or on-demand */
                if (bread_get_ssd_streaming_mode()) {
                    /* SSD streaming mode: load experts on-demand from loader */
                    if (!L) {
                        fprintf(stderr, "ERROR: loader not initialized in streaming mode\n");
                        continue;
                    }
                    expert_ptrs_t ptrs = loader_get_expert(L, layer_idx, expert_idx);
                    d_gate = ptrs.gate;
                    d_up = ptrs.up;
                    d_down = ptrs.down;
                    if (!d_gate || !d_up || !d_down) {
                        fprintf(stderr, "ERROR: expert (%d,%d) not loaded by loader\n",
                                layer_idx, expert_idx);
                        continue;
                    }
                } else {
                    /* Default mode: use pre-cached weights from weight_cache */
                    if (!wc || !wc->layers[layer_idx].experts.gate_ptrs) {
                        fprintf(stderr, "ERROR: expert weights not cached (wc=%p)\n", wc);
                        continue;
                    }
                    d_gate = wc->layers[layer_idx].experts.gate_ptrs[expert_idx];
                    d_up = wc->layers[layer_idx].experts.up_ptrs[expert_idx];
                    d_down = wc->layers[layer_idx].experts.down_ptrs[expert_idx];
                }

                /* gate/up projections: normed2[H] → gate/up[EXPERT_INTER] */
                bread_matvec(d_gate, d_normed2, d_eg,
                             cfg->expert_inter, H, GGML_TYPE_Q4_K, stream_a);
                bread_matvec(d_up,  d_normed2, d_eu,
                             cfg->expert_inter, H, GGML_TYPE_Q4_K, stream_a);

                /* SwiGLU in-place on gate */
                silu_mul_inplace<<<n_blocks_ei, 256, 0, stream_a>>>(
                    d_eg, d_eu, cfg->expert_inter);

                /* down projection: gate[EXPERT_INTER] → expert_out[H] */
                bread_matvec(d_down, d_eg, d_eo,
                             H, cfg->expert_inter, GGML_TYPE_Q6_K, stream_a);

                /* Weighted accumulate into hidden */
                scale_accum<<<n_blocks_h, 256, 0, stream_a>>>(
                    d_hidden, d_eo, active_expert_weights[k], H);
                /* No intermediate syncs — all on same stream, GPU handles ordering */
            }
            /* Flush all expert kernels to GPU (single sync instead of per-expert) */
            CUDA_CHECK(cudaStreamSynchronize(stream_a));
        }
    }

    /* ================================================================
     * ROUTING FOR NEXT LAYER (Pipelined)
     *
     * Compute routing at end of this layer, for use at START of next layer.
     * CPU work (routing matmul + softmax + topK) happens off critical path
     * while prefetch DMA runs on stream_c.
     * ================================================================ */
    if (bread_get_ssd_streaming_mode() && layer_idx + 1 < cfg->num_layers) {
        /* Use d_normed2 (post-attn norm hidden state) for routing.
           This is the standard routing input for MoE FFN. */
        route_layer(L, g, layer_idx + 1, d_normed2,
                    h_expert_indices_next, h_expert_weights_next);
        h_expert_next_valid = 1;
    }

    /* ================================================================
     * SYNC: Wait for compute and prefetch to complete
     * ================================================================ */
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
    if (bread_get_ssd_streaming_mode() && L) {
        loader_sync(L);  /* Waits for stream_c prefetch DMA to complete */
    }

    /* ================================================================
     * BUFFER SWAP: Current becomes previous, next becomes current for
     * next layer's expert computation
     * ================================================================ */
    if (bread_get_ssd_streaming_mode() && layer_idx + 1 < cfg->num_layers) {
        int *tmp_idx = h_expert_indices_current;
        float *tmp_wt = h_expert_weights_current;
        h_expert_indices_current = h_expert_indices_next;
        h_expert_weights_current = h_expert_weights_next;
        h_expert_indices_next = tmp_idx;
        h_expert_weights_next = tmp_wt;
        h_expert_current_valid = h_expert_next_valid;
        h_expert_next_valid = 0;
    }

    /* d_hidden now contains the full transformer block output */
}

/* ================================================================== */
/*  S E L F T E S T                                                    */
/* ================================================================== */
#ifdef SELFTEST_MAIN

#ifdef _WIN32
#  include <windows.h>
static double now_ms(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#  include <sys/time.h>
static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif

static void print_half_sample(const char *label, const half *d_x, int n,
                               int count)
{
    half *tmp = (half *)malloc(count * sizeof(half));
    if (n < count) count = n;
    CUDA_CHECK(cudaMemcpy(tmp, d_x, count * sizeof(half),
                           cudaMemcpyDeviceToHost));
    printf("  %s: [", label);
    for (int i = 0; i < count; i++)
        printf("%s%.4f", i ? ", " : "", __half2float(tmp[i]));
    printf("]\n");
    free(tmp);
}

static float max_diff_half(const half *a, const half *b, int n)
{
    half *ha = (half *)malloc(n * sizeof(half));
    half *hb = (half *)malloc(n * sizeof(half));
    CUDA_CHECK(cudaMemcpy(ha, a, n * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb, b, n * sizeof(half), cudaMemcpyDeviceToHost));
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(__half2float(ha[i]) - __half2float(hb[i]));
        if (d > mx) mx = d;
    }
    free(ha); free(hb);
    return mx;
}

/* Simple LCG for deterministic random half values in [-0.5, 0.5] */
static void fill_random_half(half *d_x, int n, unsigned seed)
{
    half *h = (half *)malloc(n * sizeof(half));
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        float v = ((float)(s >> 1) / (float)0x7FFFFFFF) - 1.0f;
        v *= 0.5f;
        h[i] = __float2half(v);
    }
    CUDA_CHECK(cudaMemcpy(d_x, h, n * sizeof(half), cudaMemcpyHostToDevice));
    free(h);
}

int main(int argc, char **argv)
{
    const char *model_path = (argc > 1) ? argv[1] : BREAD_MODEL_PATH;

    printf("=== one_layer_test ===\n");
    printf("Model: %s\n\n", model_path);

    /* ---- Load model ---- */
    printf("Loading model into pinned RAM (22 GB)...\n");
    double t0 = now_ms();
    loader_t *L = loader_init(model_path);
    if (!L) { fprintf(stderr, "loader_init failed\n"); return 1; }
    printf("  done in %.1f s\n\n", (now_ms() - t0) / 1000.0);

    /* ---- Open GGUF for metadata ---- */
    gguf_ctx_t *g = gguf_open(model_path);
    if (!g) { fprintf(stderr, "gguf_open failed\n"); return 1; }

    /* ---- CUDA stream for compute ---- */
    cudaStream_t stream_a;
    CUDA_CHECK(cudaStreamCreate(&stream_a));

    /* ---- Persistent input/reference buffers ---- */
    half *d_hidden = NULL;
    half *d_input_copy = NULL;
    CUDA_CHECK(cudaMalloc(&d_hidden,     BREAD_HIDDEN_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_input_copy, BREAD_HIDDEN_DIM * sizeof(half)));

    int all_pass = 1;

    /* ==============================================================
     * Test 1: Layer 0 (SSM stub — exercises MoE only)
     * ============================================================== */
    {
        int layer = 0;
        printf("--- Layer %d (SSM stub, MoE exercised) ---\n", layer);

        /* Random hidden state */
        fill_random_half(d_hidden, BREAD_HIDDEN_DIM, 42);
        CUDA_CHECK(cudaMemcpy(d_input_copy, d_hidden,
                               BREAD_HIDDEN_DIM * sizeof(half),
                               cudaMemcpyDeviceToDevice));

        print_half_sample("input ", d_hidden, BREAD_HIDDEN_DIM, 8);

        double t1 = now_ms();
        one_layer_forward(d_hidden, layer, /*pos=*/0, L, g, NULL, stream_a);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("  forward time: %.1f ms\n", now_ms() - t1);

        print_half_sample("output", d_hidden, BREAD_HIDDEN_DIM, 8);

        float diff = max_diff_half(d_hidden, d_input_copy, BREAD_HIDDEN_DIM);
        printf("  max_diff = %.4f  %s\n\n",
               diff, diff > 1e-3f ? "PASS (output differs from input)" : "FAIL");
        if (diff <= 1e-3f) all_pass = 0;
    }

    /* ==============================================================
     * Test 2: Layer 3 (first full-attention layer)
     * ============================================================== */
    {
        int layer = 3;
        printf("--- Layer %d (full attention + MoE) ---\n", layer);

        fill_random_half(d_hidden, BREAD_HIDDEN_DIM, 99);
        CUDA_CHECK(cudaMemcpy(d_input_copy, d_hidden,
                               BREAD_HIDDEN_DIM * sizeof(half),
                               cudaMemcpyDeviceToDevice));

        print_half_sample("input ", d_hidden, BREAD_HIDDEN_DIM, 8);

        double t1 = now_ms();
        one_layer_forward(d_hidden, layer, /*pos=*/0, L, g, NULL, stream_a);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("  forward time: %.1f ms\n", now_ms() - t1);

        print_half_sample("output", d_hidden, BREAD_HIDDEN_DIM, 8);

        float diff = max_diff_half(d_hidden, d_input_copy, BREAD_HIDDEN_DIM);
        printf("  max_diff = %.4f  %s\n\n",
               diff, diff > 1e-3f ? "PASS (output differs from input)" : "FAIL");
        if (diff <= 1e-3f) all_pass = 0;
    }

    /* ---- Cleanup ---- */
    cudaFree(d_hidden);
    cudaFree(d_input_copy);
    cudaStreamDestroy(stream_a);
    gguf_close(g);
    loader_free(L);

    printf("=== one_layer_test: %s ===\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}

#endif /* SELFTEST_MAIN */
