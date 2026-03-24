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

#include "bread.h"
#include "gguf.h"
#include "loader.h"

/* ------------------------------------------------------------------ */
/* CUDA error check                                                     */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call) do {                                           \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d — %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));            \
        exit(1);                                                        \
    }                                                                   \
} while (0)

/* ------------------------------------------------------------------ */
/* External: bread_matvec from kernels.cu                              */
/* ------------------------------------------------------------------ */

extern void bread_matvec(void *w, half *x, half *y,
                          int rows, int cols, int qtype);

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
/* Kernel 3: GQA expand — copy KV-head values to all Q-head slots     */
/*                                                                      */
/* For single-token (pos=0) attention the output is just V expanded:  */
/*   Q-head h gets the value of KV-head  h / (NUM_Q / NUM_KV).       */
/* Grid: (NUM_Q_HEADS,)   Block: (HEAD_DIM_V,)                        */
/* ------------------------------------------------------------------ */

static __global__ void gqa_expand_v(half *out, const half *v,
                               int num_q, int num_kv, int head_dim)
{
    int q_head  = blockIdx.x;
    int kv_head = q_head * num_kv / num_q;
    int i       = threadIdx.x;
    if (i < head_dim)
        out[q_head * head_dim + i] = v[kv_head * head_dim + i];
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
static const gguf_tensor_t *require_tensor(const gguf_ctx_t *g, const char *name)
{
    const gguf_tensor_t *t = gguf_find_tensor(g, name);
    if (!t) {
        fprintf(stderr, "tensor not found: %s\n", name);
        exit(1);
    }
    return t;
}

static uint8_t *tensor_ram(const loader_t *L, const gguf_ctx_t *g,
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

/* Allocate VRAM and copy a tensor from pinned RAM */
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
    half *tmp = (half *)malloc(n * sizeof(half));
    CUDA_CHECK(cudaMemcpy(tmp, d_x, n * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) h_f[i] = __half2float(tmp[i]);
    free(tmp);
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

static void apply_rotary_emb(float *q, float *k, int pos, float rope_freq_base,
                             int num_q_heads, int num_kv_heads,
                             int head_dim, int rotary_dim)
{
    int half = rotary_dim / 2;
    for (int h = 0; h < num_q_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(rope_freq_base, (float)(2 * i) / rotary_dim);
            float angle = (float)pos * freq;
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
            float freq = 1.0f / powf(rope_freq_base, (float)(2 * i) / rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float k0 = kh[i];
            float k1 = kh[i + half];
            kh[i]        = k0 * cos_a - k1 * sin_a;
            kh[i + half] = k0 * sin_a + k1 * cos_a;
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

void one_layer_forward(half *d_hidden, int layer_idx, int pos,
                        loader_t *L, gguf_ctx_t *g,
                        cudaStream_t stream_a)
{
    const bread_model_config_t *cfg = bread_model_config_get();
    char nm[128];   /* tensor name buffer */
    int H = cfg->hidden_dim;
    int is_full = ((layer_idx % 4) == 3);

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
    static float *h_q_full   = NULL;
    static float *h_kv_k     = NULL;
    static float *h_kv_v     = NULL;
    static float *h_q_score  = NULL;
    static float *h_q_gate   = NULL;
    static float *h_scores   = NULL;
    static float *ssm_conv_state[LOADER_MAX_LAYERS] = {0};
    static float *ssm_state[LOADER_MAX_LAYERS] = {0};
    static float *kv_k_cache[LOADER_MAX_LAYERS] = {0};
    static float *kv_v_cache[LOADER_MAX_LAYERS] = {0};
    static int    kv_cache_len[LOADER_MAX_LAYERS] = {0};

    if (!d_normed) {
        CUDA_CHECK(cudaMalloc(&d_normed,   H                          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_normed2,  H                          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_q,        cfg->q_proj_dim            * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_k,        cfg->kv_proj_dim           * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_v,        cfg->kv_proj_dim           * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_attn_out, cfg->attn_out_dim          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_o_out,    H                          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_sg,       cfg->shared_inter          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_su,       cfg->shared_inter          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_sh_out,   H                          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_eg,       cfg->expert_inter          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_eu,       cfg->expert_inter          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_eo,       H                          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_qkv,      cfg->ssm_qkv_dim           * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_z,        cfg->ssm_z_dim             * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_alpha,    cfg->ssm_num_v_heads       * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_beta,     cfg->ssm_num_v_heads       * sizeof(half)));

        h_qkv      = (float *)malloc(cfg->ssm_qkv_dim * sizeof(float));
        h_z        = (float *)malloc(cfg->ssm_z_dim * sizeof(float));
        h_alpha    = (float *)malloc(cfg->ssm_num_v_heads * sizeof(float));
        h_beta     = (float *)malloc(cfg->ssm_num_v_heads * sizeof(float));
        h_conv_out = (float *)malloc(cfg->ssm_qkv_dim * sizeof(float));
        h_attn_out = (float *)malloc(cfg->attn_out_dim * sizeof(float));
        h_head_tmp = (float *)malloc(cfg->ssm_head_dim * sizeof(float));
        h_q_full   = (float *)malloc(cfg->q_proj_dim * sizeof(float));
        h_kv_k     = (float *)malloc(cfg->kv_proj_dim * sizeof(float));
        h_kv_v     = (float *)malloc(cfg->kv_proj_dim * sizeof(float));
        h_q_score  = (float *)malloc(cfg->attn_out_dim * sizeof(float));
        h_q_gate   = (float *)malloc(cfg->attn_out_dim * sizeof(float));
        h_scores   = (float *)malloc(cfg->kv_cache_len * sizeof(float));
        if (!h_qkv || !h_z || !h_alpha || !h_beta || !h_conv_out || !h_attn_out ||
            !h_head_tmp || !h_q_full || !h_kv_k || !h_kv_v || !h_q_score || !h_q_gate || !h_scores) {
            fprintf(stderr, "one_layer_forward: host scratch alloc failed\n");
            exit(1);
        }

        for (int layer = 0; layer < cfg->num_layers; layer++) {
            if (((layer % 4) == 3)) {
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
    }

    /* ================================================================
     * PHASE 0: pre-attention RMSNorm
     *   d_normed = rmsnorm(d_hidden, attn_norm_w)
     * d_hidden is kept as the residual.
     * ================================================================ */
    {
        snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", layer_idx);
        float *d_attn_norm_w = (float *)load_vram(L, g, nm);

        /* Copy hidden → normed, then normalise normed in-place */
        int blocks = (H + 255) / 256;
        copy_half<<<blocks, 256, 0, stream_a>>>(d_normed, d_hidden, H);
        rmsnorm_inplace<<<1, 256, 0, stream_a>>>(d_normed, d_attn_norm_w,
                                                  H, cfg->rms_eps);
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
        cudaFree(d_attn_norm_w);
    }

    /* ================================================================
     * CMD1: Attention projections
     *
     * Full-attn: Q/K/V matvec on d_normed.
     * SSM stub:  skip — d_attn_out will be zeroed so residual unchanged.
     * ================================================================ */
    if (is_full) {
        snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", layer_idx);
        void *d_qw = load_vram(L, g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", layer_idx);
        void *d_kw = load_vram(L, g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", layer_idx);
        void *d_vw = load_vram(L, g, nm);

        bread_matvec(d_qw, d_normed, d_q,
                     cfg->q_proj_dim,  H, GGML_TYPE_Q4_K);
        bread_matvec(d_kw, d_normed, d_k,
                     cfg->kv_proj_dim, H, GGML_TYPE_Q4_K);
        bread_matvec(d_vw, d_normed, d_v,
                     cfg->kv_proj_dim, H, GGML_TYPE_Q6_K);

        CUDA_CHECK(cudaStreamSynchronize(stream_a));
        cudaFree(d_qw); cudaFree(d_kw); cudaFree(d_vw);
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

        vram_half_to_cpu_float(d_q, h_q_full, cfg->q_proj_dim);
        vram_half_to_cpu_float(d_k, h_kv_k, cfg->kv_proj_dim);
        vram_half_to_cpu_float(d_v, h_kv_v, cfg->kv_proj_dim);

        for (int h = 0; h < cfg->num_q_heads; h++) {
            memcpy(h_q_score + h * cfg->head_dim_qk,
                   h_q_full + h * (cfg->head_dim_qk + cfg->head_dim_qgate),
                   cfg->head_dim_qk * sizeof(float));
            memcpy(h_q_gate + h * cfg->head_dim_v,
                   h_q_full + h * (cfg->head_dim_qk + cfg->head_dim_qgate) + cfg->head_dim_qk,
                   cfg->head_dim_v * sizeof(float));
        }

        snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", layer_idx);
        q_norm_w = tensor_ram_f32(L, g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", layer_idx);
        k_norm_w = tensor_ram_f32(L, g, nm);

        for (int h = 0; h < cfg->num_q_heads; h++) {
            float *qh = h_q_score + h * cfg->head_dim_qk;
            float sum_sq = 0.0f;
            for (int i = 0; i < cfg->head_dim_qk; i++) sum_sq += qh[i] * qh[i];
            {
                float inv_rms = 1.0f / sqrtf(sum_sq / (float)cfg->head_dim_qk + cfg->rms_eps);
                for (int i = 0; i < cfg->head_dim_qk; i++)
                    qh[i] = qh[i] * inv_rms * q_norm_w[i];
            }
        }
        for (int h = 0; h < cfg->num_kv_heads; h++) {
            float *kh = h_kv_k + h * cfg->head_dim_qk;
            float sum_sq = 0.0f;
            for (int i = 0; i < cfg->head_dim_qk; i++) sum_sq += kh[i] * kh[i];
            {
                float inv_rms = 1.0f / sqrtf(sum_sq / (float)cfg->head_dim_qk + cfg->rms_eps);
                for (int i = 0; i < cfg->head_dim_qk; i++)
                    kh[i] = kh[i] * inv_rms * k_norm_w[i];
            }
        }

        apply_rotary_emb(h_q_score, h_kv_k, pos, cfg->rope_freq_base,
                         cfg->num_q_heads, cfg->num_kv_heads,
                         cfg->head_dim_qk, cfg->head_dim_rope);

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
            half *h_attn_half = (half *)malloc(cfg->attn_out_dim * sizeof(half));
            if (!h_attn_half) { fprintf(stderr, "one_layer_forward: attn half alloc failed\n"); exit(1); }
            for (int i = 0; i < cfg->attn_out_dim; i++)
                h_attn_half[i] = __float2half(h_attn_out[i]);
            CUDA_CHECK(cudaMemcpy(d_attn_out, h_attn_half,
                                  cfg->attn_out_dim * sizeof(half),
                                  cudaMemcpyHostToDevice));
            free(h_attn_half);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
    } else {
        const float *conv_w;
        const float *ssm_a;
        const float *ssm_dt;
        const float *ssm_norm_w;
        float *conv_state;
        float *layer_state;

        snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight", layer_idx);
        void *d_qkv_w = load_vram(L, g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight", layer_idx);
        void *d_gate_w = load_vram(L, g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_alpha.weight", layer_idx);
        void *d_alpha_w = load_vram(L, g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_beta.weight", layer_idx);
        void *d_beta_w = load_vram(L, g, nm);

        bread_matvec(d_qkv_w, d_normed, d_qkv,
                     cfg->ssm_qkv_dim, H, GGML_TYPE_Q4_K);
        bread_matvec(d_gate_w, d_normed, d_z,
                     cfg->ssm_z_dim, H, GGML_TYPE_Q4_K);
        bread_matvec(d_alpha_w, d_normed, d_alpha,
                     cfg->ssm_num_v_heads, H, GGML_TYPE_Q4_K);
        bread_matvec(d_beta_w, d_normed, d_beta,
                     cfg->ssm_num_v_heads, H, GGML_TYPE_Q4_K);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_qkv_w);
        cudaFree(d_gate_w);
        cudaFree(d_alpha_w);
        cudaFree(d_beta_w);

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

        {
            const int key_dim = cfg->ssm_head_dim;
            const int value_dim = cfg->ssm_head_dim;
            const int num_k_heads = cfg->ssm_num_k_heads;
            const int num_v_heads = cfg->ssm_num_v_heads;
            const int k_heads_per_v = num_v_heads / num_k_heads;
            const float inv_scale = 1.0f / sqrtf((float)key_dim);
            float *lin_q = h_conv_out;
            float *lin_k = h_conv_out + num_k_heads * key_dim;
            float *lin_v = h_conv_out + 2 * num_k_heads * key_dim;

            for (int h = 0; h < num_k_heads; h++) {
                float *qh = lin_q + h * key_dim;
                cpu_rms_norm_bare(qh, key_dim, cfg->rms_eps);
                for (int d = 0; d < key_dim; d++) qh[d] *= inv_scale * inv_scale;
            }
            for (int h = 0; h < num_k_heads; h++) {
                float *kh = lin_k + h * key_dim;
                cpu_rms_norm_bare(kh, key_dim, cfg->rms_eps);
                for (int d = 0; d < key_dim; d++) kh[d] *= inv_scale;
            }

            for (int vh = 0; vh < num_v_heads; vh++) {
                int kh = vh / k_heads_per_v;
                float decay = expf(-expf(ssm_a[vh]) * cpu_softplus(h_alpha[vh] + ssm_dt[vh]));
                float beta_gate = cpu_sigmoid(h_beta[vh]);
                float *S = layer_state + (size_t)vh * value_dim * key_dim;
                float *q_h = lin_q + kh * key_dim;
                float *k_h = lin_k + kh * key_dim;
                float *v_h = lin_v + vh * value_dim;
                float *o_h = h_attn_out + vh * value_dim;

                for (int vi = 0; vi < value_dim; vi++) {
                    float *row = S + (size_t)vi * key_dim;
                    for (int ki = 0; ki < key_dim; ki++)
                        row[ki] *= decay;
                }

                for (int vi = 0; vi < value_dim; vi++) {
                    float *row = S + (size_t)vi * key_dim;
                    float kv_mem = 0.0f;
                    for (int ki = 0; ki < key_dim; ki++)
                        kv_mem += row[ki] * k_h[ki];

                    {
                        float delta = (v_h[vi] - kv_mem) * beta_gate;
                        for (int ki = 0; ki < key_dim; ki++)
                            row[ki] += delta * k_h[ki];
                    }
                }

                for (int vi = 0; vi < value_dim; vi++) {
                    float *row = S + (size_t)vi * key_dim;
                    float out = 0.0f;
                    for (int ki = 0; ki < key_dim; ki++)
                        out += row[ki] * q_h[ki];
                    h_head_tmp[vi] = out;
                }

                cpu_gated_rms_norm(h_head_tmp, h_z + vh * value_dim,
                                   ssm_norm_w, o_h, value_dim, cfg->rms_eps);
            }
        }

        {
            half *h_attn_half = (half *)malloc(cfg->ssm_z_dim * sizeof(half));
            if (!h_attn_half) { fprintf(stderr, "one_layer_forward: ssm half alloc failed\n"); exit(1); }
            for (int i = 0; i < cfg->ssm_z_dim; i++)
                h_attn_half[i] = __float2half(h_attn_out[i]);
            CUDA_CHECK(cudaMemcpy(d_attn_out, h_attn_half,
                                  cfg->ssm_z_dim * sizeof(half),
                                  cudaMemcpyHostToDevice));
            free(h_attn_half);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
    }

    /* ================================================================
     * CMD2a: o_proj + residual add (full-attn only)
     *
     * attn_out [ATTN_OUT_DIM] → o_proj → [H]
     * hidden += o_proj_out
     * ================================================================ */
    if (is_full) {
        snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", layer_idx);
        void *d_ow = load_vram(L, g, nm);

        bread_matvec(d_ow, d_attn_out, d_o_out,
                     H, cfg->attn_out_dim, GGML_TYPE_Q4_K);
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
        cudaFree(d_ow);

        int blocks = (H + 255) / 256;
        scale_accum<<<blocks, 256, 0, stream_a>>>(d_hidden, d_o_out, 1.0f, H);
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
    } else {
        snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight", layer_idx);
        void *d_sw = load_vram(L, g, nm);

        bread_matvec(d_sw, d_attn_out, d_o_out,
                     H, cfg->ssm_z_dim, GGML_TYPE_Q4_K);
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
        cudaFree(d_sw);

        int blocks = (H + 255) / 256;
        scale_accum<<<blocks, 256, 0, stream_a>>>(d_hidden, d_o_out, 1.0f, H);
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
    }

    /* ================================================================
     * CMD2b: post-attention RMSNorm → d_normed2 (pre-FFN input)
     * ================================================================ */
    {
        snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", layer_idx);
        float *d_pan_w = (float *)load_vram(L, g, nm);

        int blocks = (H + 255) / 256;
        copy_half<<<blocks, 256, 0, stream_a>>>(d_normed2, d_hidden, H);
        rmsnorm_inplace<<<1, 256, 0, stream_a>>>(d_normed2, d_pan_w,
                                                  H, cfg->rms_eps);
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
        cudaFree(d_pan_w);
    }

    /* ================================================================
     * CMD2c: shared-expert gate/up projections
     * ================================================================ */
    {
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_shexp.weight", layer_idx);
        void *d_sg_w = load_vram(L, g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_up_shexp.weight", layer_idx);
        void *d_su_w = load_vram(L, g, nm);

        bread_matvec(d_sg_w, d_normed2, d_sg,
                     cfg->shared_inter, H, GGML_TYPE_Q4_K);
        bread_matvec(d_su_w, d_normed2, d_su,
                     cfg->shared_inter, H, GGML_TYPE_Q4_K);
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
        cudaFree(d_sg_w); cudaFree(d_su_w);
    }

    /* ================================================================
     * CPU routing: router matmul (F32) + softmax + topK
     *
     * router weight: blk.N.ffn_gate_inp.weight [H × NUM_EXPERTS] F32
     * ================================================================ */
    int   *expert_indices = NULL;
    float *expert_weights = NULL;
    float shared_gate_score = 0.0f;
    {
        expert_indices = (int *)malloc(cfg->top_k * sizeof(int));
        expert_weights = (float *)malloc(cfg->top_k * sizeof(float));
        if (!expert_indices || !expert_weights) {
            fprintf(stderr, "one_layer_forward: router alloc failed\n");
            exit(1);
        }
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", layer_idx);
        const float *router_w = tensor_ram_f32(L, g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp_shexp.weight", layer_idx);
        const half *shared_gate_w = NULL;
        if (gguf_find_tensor(g, nm)) shared_gate_w = tensor_ram_f16(L, g, nm);

        /* Copy d_normed2 → CPU float */
        float *normed_f32 = (float *)malloc(H * sizeof(float));
        float *logits = (float *)malloc(cfg->num_experts * sizeof(float));
        if (!normed_f32 || !logits) {
            fprintf(stderr, "one_layer_forward: router scratch alloc failed\n");
            exit(1);
        }
        vram_half_to_cpu_float(d_normed2, normed_f32, H);

        /* Dense F32 matmul: logits[i] = router_w[i * H + j] * normed[j] */
        memset(logits, 0, cfg->num_experts * sizeof(float));
        for (int i = 0; i < cfg->num_experts; i++)
            for (int j = 0; j < H; j++)
                logits[i] += router_w[i * H + j] * normed_f32[j];

        if (shared_gate_w) {
            for (int j = 0; j < H; j++)
                shared_gate_score += __half2float(shared_gate_w[j]) * normed_f32[j];
        }

        cpu_softmax(logits, cfg->num_experts);
        cpu_topk(logits, cfg->num_experts, cfg->top_k,
                 expert_indices, expert_weights);
        free(normed_f32);
        free(logits);
    }

    /* ================================================================
     * DMA: Stream B — load K expert weight sets into VRAM
     * ================================================================ */
    loader_request(L, layer_idx, expert_indices, cfg->top_k);
    loader_sync(L);

    /* ================================================================
     * CMD3a: shared expert SwiGLU + down projection → accumulate
     * ================================================================ */
    {
        int n_blocks_si = (cfg->shared_inter + 255) / 256;
        silu_mul_inplace<<<n_blocks_si, 256, 0, stream_a>>>(
            d_sg, d_su, cfg->shared_inter);

        snprintf(nm, sizeof(nm), "blk.%d.ffn_down_shexp.weight", layer_idx);
        void *d_sd_w = load_vram(L, g, nm);

        bread_matvec(d_sd_w, d_sg, d_sh_out,
                     H, cfg->shared_inter, GGML_TYPE_Q6_K);
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
        cudaFree(d_sd_w);

        {
            float shared_weight = cpu_sigmoid(shared_gate_score);
            int blocks = (H + 255) / 256;
            scale_accum<<<blocks, 256, 0, stream_a>>>(d_hidden, d_sh_out, shared_weight, H);
            CUDA_CHECK(cudaStreamSynchronize(stream_a));
        }
    }

    /* ================================================================
     * CMD3b: K active expert forwards → weighted accumulate into hidden
     * ================================================================ */
    {
        int n_blocks_ei = (cfg->expert_inter + 255) / 256;
        int n_blocks_h  = (H + 255) / 256;

        for (int k = 0; k < cfg->top_k; k++) {
            expert_ptrs_t ep = loader_get_expert(L, layer_idx,
                                                  expert_indices[k]);
            if (!ep.gate) {
                fprintf(stderr, "expert(%d,%d) not in cache after sync\n",
                        layer_idx, expert_indices[k]);
                continue;
            }

            /* gate/up projections: normed2[H] → gate/up[EXPERT_INTER] */
            bread_matvec(ep.gate, d_normed2, d_eg,
                         cfg->expert_inter, H, (int)ep.gate_type);
            bread_matvec(ep.up,  d_normed2, d_eu,
                         cfg->expert_inter, H, (int)ep.up_type);

            /* SwiGLU in-place on gate */
            silu_mul_inplace<<<n_blocks_ei, 256, 0, stream_a>>>(
                d_eg, d_eu, cfg->expert_inter);

            /* down projection: gate[EXPERT_INTER] → expert_out[H] */
            bread_matvec(ep.down, d_eg, d_eo,
                         H, cfg->expert_inter, (int)ep.down_type);

            CUDA_CHECK(cudaStreamSynchronize(stream_a));

            /* Weighted accumulate into hidden */
            scale_accum<<<n_blocks_h, 256, 0, stream_a>>>(
                d_hidden, d_eo, expert_weights[k], H);
            CUDA_CHECK(cudaStreamSynchronize(stream_a));
        }
    }
    free(expert_indices);
    free(expert_weights);
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
        one_layer_forward(d_hidden, layer, /*pos=*/0, L, g, stream_a);
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
        one_layer_forward(d_hidden, layer, /*pos=*/0, L, g, stream_a);
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
