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
static uint8_t *tensor_ram(const loader_t *L, const gguf_ctx_t *g,
                             const char *name)
{
    const gguf_tensor_t *t = gguf_find_tensor(g, name);
    if (!t) { fprintf(stderr, "tensor not found: %s\n", name); return NULL; }
    return L->pinned_data + L->data_offset + t->offset;
}

/* Allocate VRAM and copy a tensor from pinned RAM */
static void *load_vram(const loader_t *L, const gguf_ctx_t *g,
                        const char *name)
{
    const gguf_tensor_t *t = gguf_find_tensor(g, name);
    if (!t) { fprintf(stderr, "tensor not found: %s\n", name); return NULL; }
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
        out[c] = acc / (1.0f + expf(-acc));
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
    char nm[128];   /* tensor name buffer */
    int H = BREAD_HIDDEN_DIM;
    int is_full = BREAD_IS_FULL_ATTN(layer_idx);

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
    static float *ssm_conv_state[BREAD_NUM_LAYERS] = {0};
    static float *ssm_state[BREAD_NUM_LAYERS] = {0};

    if (!d_normed) {
        CUDA_CHECK(cudaMalloc(&d_normed,   H                          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_normed2,  H                          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_q,        BREAD_Q_PROJ_DIM           * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_k,        BREAD_KV_PROJ_DIM          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_v,        BREAD_KV_PROJ_DIM          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_attn_out, BREAD_ATTN_OUT_DIM         * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_o_out,    H                          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_sg,       BREAD_SHARED_INTER         * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_su,       BREAD_SHARED_INTER         * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_sh_out,   H                          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_eg,       BREAD_EXPERT_INTER         * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_eu,       BREAD_EXPERT_INTER         * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_eo,       H                          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_qkv,      BREAD_SSM_QKV_DIM          * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_z,        BREAD_SSM_Z_DIM            * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_alpha,    BREAD_SSM_NUM_V_HEADS      * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_beta,     BREAD_SSM_NUM_V_HEADS      * sizeof(half)));

        h_qkv      = (float *)malloc(BREAD_SSM_QKV_DIM * sizeof(float));
        h_z        = (float *)malloc(BREAD_SSM_Z_DIM * sizeof(float));
        h_alpha    = (float *)malloc(BREAD_SSM_NUM_V_HEADS * sizeof(float));
        h_beta     = (float *)malloc(BREAD_SSM_NUM_V_HEADS * sizeof(float));
        h_conv_out = (float *)malloc(BREAD_SSM_QKV_DIM * sizeof(float));
        h_attn_out = (float *)malloc(BREAD_SSM_Z_DIM * sizeof(float));
        h_head_tmp = (float *)malloc(BREAD_SSM_HEAD_DIM * sizeof(float));
        if (!h_qkv || !h_z || !h_alpha || !h_beta || !h_conv_out || !h_attn_out || !h_head_tmp) {
            fprintf(stderr, "one_layer_forward: host scratch alloc failed\n");
            exit(1);
        }

        for (int layer = 0; layer < BREAD_NUM_LAYERS; layer++) {
            if (BREAD_IS_FULL_ATTN(layer)) continue;
            ssm_conv_state[layer] = (float *)calloc(
                (BREAD_SSM_CONV_KERNEL - 1) * BREAD_SSM_QKV_DIM, sizeof(float));
            ssm_state[layer] = (float *)calloc(
                BREAD_SSM_NUM_V_HEADS * BREAD_SSM_HEAD_DIM * BREAD_SSM_HEAD_DIM, sizeof(float));
            if (!ssm_conv_state[layer] || !ssm_state[layer]) {
                fprintf(stderr, "one_layer_forward: SSM state alloc failed for layer %d\n", layer);
                exit(1);
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
                                                  H, BREAD_RMS_EPS);
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
                     BREAD_Q_PROJ_DIM,  H, GGML_TYPE_Q4_K);
        bread_matvec(d_kw, d_normed, d_k,
                     BREAD_KV_PROJ_DIM, H, GGML_TYPE_Q4_K);
        bread_matvec(d_vw, d_normed, d_v,
                     BREAD_KV_PROJ_DIM, H, GGML_TYPE_Q6_K);

        CUDA_CHECK(cudaStreamSynchronize(stream_a));
        cudaFree(d_qw); cudaFree(d_kw); cudaFree(d_vw);
    }

    /* ================================================================
     * CPU attention: single-token (pos=0)
     *
     * For pos=0 there is exactly one KV entry.
     * softmax([[score]]) = [[1.0]] → attn_out = V expanded to Q heads.
     * Q and K are computed but their dot-product is irrelevant (scalar
     * softmax = 1.0 regardless).  Skip RoPE and QK-norm for Step 2.
     * ================================================================ */
    if (is_full) {
        /* GQA expand: tile V across Q heads (2 KV heads → 16 Q heads) */
        gqa_expand_v<<<BREAD_NUM_Q_HEADS, BREAD_HEAD_DIM_V, 0, stream_a>>>(
            d_attn_out, d_v,
            BREAD_NUM_Q_HEADS, BREAD_NUM_KV_HEADS, BREAD_HEAD_DIM_V);
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
                     BREAD_SSM_QKV_DIM, H, GGML_TYPE_Q4_K);
        bread_matvec(d_gate_w, d_normed, d_z,
                     BREAD_SSM_Z_DIM, H, GGML_TYPE_Q4_K);
        bread_matvec(d_alpha_w, d_normed, d_alpha,
                     BREAD_SSM_NUM_V_HEADS, H, GGML_TYPE_Q4_K);
        bread_matvec(d_beta_w, d_normed, d_beta,
                     BREAD_SSM_NUM_V_HEADS, H, GGML_TYPE_Q4_K);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_qkv_w);
        cudaFree(d_gate_w);
        cudaFree(d_alpha_w);
        cudaFree(d_beta_w);

        vram_half_to_cpu_float(d_qkv, h_qkv, BREAD_SSM_QKV_DIM);
        vram_half_to_cpu_float(d_z, h_z, BREAD_SSM_Z_DIM);
        vram_half_to_cpu_float(d_alpha, h_alpha, BREAD_SSM_NUM_V_HEADS);
        vram_half_to_cpu_float(d_beta, h_beta, BREAD_SSM_NUM_V_HEADS);

        snprintf(nm, sizeof(nm), "blk.%d.ssm_conv1d.weight", layer_idx);
        conv_w = (const float *)tensor_ram(L, g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_a", layer_idx);
        ssm_a = (const float *)tensor_ram(L, g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_dt", layer_idx);
        ssm_dt = (const float *)tensor_ram(L, g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_norm.weight", layer_idx);
        ssm_norm_w = (const float *)tensor_ram(L, g, nm);
        conv_state = ssm_conv_state[layer_idx];
        layer_state = ssm_state[layer_idx];

        if (!conv_w || !ssm_a || !ssm_dt || !ssm_norm_w || !conv_state || !layer_state) {
            fprintf(stderr, "one_layer_forward: missing SSM tensors for layer %d\n", layer_idx);
            exit(1);
        }

        cpu_conv1d_step(conv_state, h_qkv, conv_w, h_conv_out,
                        BREAD_SSM_QKV_DIM, BREAD_SSM_CONV_KERNEL);
        memmove(conv_state, conv_state + BREAD_SSM_QKV_DIM,
                (BREAD_SSM_CONV_KERNEL - 2) * BREAD_SSM_QKV_DIM * sizeof(float));
        memcpy(conv_state + (BREAD_SSM_CONV_KERNEL - 2) * BREAD_SSM_QKV_DIM,
               h_qkv, BREAD_SSM_QKV_DIM * sizeof(float));

        {
            const int key_dim = BREAD_SSM_HEAD_DIM;
            const int value_dim = BREAD_SSM_HEAD_DIM;
            const int num_k_heads = BREAD_SSM_NUM_K_HEADS;
            const int num_v_heads = BREAD_SSM_NUM_V_HEADS;
            const int k_heads_per_v = num_v_heads / num_k_heads;
            const float inv_scale = 1.0f / sqrtf((float)key_dim);
            float *lin_q = h_conv_out;
            float *lin_k = h_conv_out + num_k_heads * key_dim;
            float *lin_v = h_conv_out + 2 * num_k_heads * key_dim;

            for (int h = 0; h < num_k_heads; h++) {
                float *qh = lin_q + h * key_dim;
                cpu_rms_norm_bare(qh, key_dim, BREAD_RMS_EPS);
                for (int d = 0; d < key_dim; d++) qh[d] *= inv_scale * inv_scale;
            }
            for (int h = 0; h < num_k_heads; h++) {
                float *kh = lin_k + h * key_dim;
                cpu_rms_norm_bare(kh, key_dim, BREAD_RMS_EPS);
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
                    float kv_mem = 0.0f;
                    for (int ki = 0; ki < key_dim; ki++) {
                        row[ki] *= decay;
                        kv_mem += row[ki] * k_h[ki];
                    }

                    float delta = (v_h[vi] - kv_mem) * beta_gate;
                    float out = 0.0f;
                    for (int ki = 0; ki < key_dim; ki++) {
                        row[ki] += delta * k_h[ki];
                        out += row[ki] * q_h[ki];
                    }
                    h_head_tmp[vi] = out;
                }

                cpu_gated_rms_norm(h_head_tmp, h_z + vh * value_dim,
                                   ssm_norm_w, o_h, value_dim, BREAD_RMS_EPS);
            }
        }

        {
            half h_attn_half[BREAD_SSM_Z_DIM];
            for (int i = 0; i < BREAD_SSM_Z_DIM; i++)
                h_attn_half[i] = __float2half(h_attn_out[i]);
            CUDA_CHECK(cudaMemcpy(d_attn_out, h_attn_half,
                                  BREAD_SSM_Z_DIM * sizeof(half),
                                  cudaMemcpyHostToDevice));
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
                     H, BREAD_ATTN_OUT_DIM, GGML_TYPE_Q4_K);
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
        cudaFree(d_ow);

        int blocks = (H + 255) / 256;
        scale_accum<<<blocks, 256, 0, stream_a>>>(d_hidden, d_o_out, 1.0f, H);
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
    } else {
        snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight", layer_idx);
        void *d_sw = load_vram(L, g, nm);

        bread_matvec(d_sw, d_attn_out, d_o_out,
                     H, BREAD_SSM_Z_DIM, GGML_TYPE_Q4_K);
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
                                                  H, BREAD_RMS_EPS);
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
                     BREAD_SHARED_INTER, H, GGML_TYPE_Q4_K);
        bread_matvec(d_su_w, d_normed2, d_su,
                     BREAD_SHARED_INTER, H, GGML_TYPE_Q4_K);
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
        cudaFree(d_sg_w); cudaFree(d_su_w);
    }

    /* ================================================================
     * CPU routing: router matmul (F32) + softmax + topK
     *
     * router weight: blk.N.ffn_gate_inp.weight [H × NUM_EXPERTS] F32
     * ================================================================ */
    int   expert_indices[BREAD_TOP_K];
    float expert_weights[BREAD_TOP_K];
    {
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", layer_idx);
        const gguf_tensor_t *rt = gguf_find_tensor(g, nm);
        if (!rt) { fprintf(stderr, "router tensor not found\n"); exit(1); }
        const float *router_w = (const float *)
            (L->pinned_data + L->data_offset + rt->offset);

        /* Copy d_normed2 → CPU float */
        float normed_f32[BREAD_HIDDEN_DIM];
        vram_half_to_cpu_float(d_normed2, normed_f32, H);

        /* Dense F32 matmul: logits[i] = router_w[i * H + j] * normed[j] */
        float logits[BREAD_NUM_EXPERTS];
        memset(logits, 0, sizeof(logits));
        for (int i = 0; i < BREAD_NUM_EXPERTS; i++)
            for (int j = 0; j < H; j++)
                logits[i] += router_w[i * H + j] * normed_f32[j];

        cpu_softmax(logits, BREAD_NUM_EXPERTS);
        cpu_topk(logits, BREAD_NUM_EXPERTS, BREAD_TOP_K,
                 expert_indices, expert_weights);
    }

    /* ================================================================
     * DMA: Stream B — load K expert weight sets into VRAM
     * ================================================================ */
    loader_request(L, layer_idx, expert_indices, BREAD_TOP_K);
    loader_sync(L);

    /* ================================================================
     * CMD3a: shared expert SwiGLU + down projection → accumulate
     * ================================================================ */
    {
        int n_blocks_si = (BREAD_SHARED_INTER + 255) / 256;
        silu_mul_inplace<<<n_blocks_si, 256, 0, stream_a>>>(
            d_sg, d_su, BREAD_SHARED_INTER);

        snprintf(nm, sizeof(nm), "blk.%d.ffn_down_shexp.weight", layer_idx);
        void *d_sd_w = load_vram(L, g, nm);

        bread_matvec(d_sd_w, d_sg, d_sh_out,
                     H, BREAD_SHARED_INTER, GGML_TYPE_Q6_K);
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
        cudaFree(d_sd_w);

        int blocks = (H + 255) / 256;
        scale_accum<<<blocks, 256, 0, stream_a>>>(d_hidden, d_sh_out, 1.0f, H);
        CUDA_CHECK(cudaStreamSynchronize(stream_a));
    }

    /* ================================================================
     * CMD3b: K active expert forwards → weighted accumulate into hidden
     * ================================================================ */
    {
        int n_blocks_ei = (BREAD_EXPERT_INTER + 255) / 256;
        int n_blocks_h  = (H + 255) / 256;

        for (int k = 0; k < BREAD_TOP_K; k++) {
            expert_ptrs_t ep = loader_get_expert(L, layer_idx,
                                                  expert_indices[k]);
            if (!ep.gate) {
                fprintf(stderr, "expert(%d,%d) not in cache after sync\n",
                        layer_idx, expert_indices[k]);
                continue;
            }

            /* gate/up projections: normed2[H] → gate/up[EXPERT_INTER] */
            bread_matvec(ep.gate, d_normed2, d_eg,
                         BREAD_EXPERT_INTER, H, (int)ep.gate_type);
            bread_matvec(ep.up,  d_normed2, d_eu,
                         BREAD_EXPERT_INTER, H, (int)ep.up_type);

            /* SwiGLU in-place on gate */
            silu_mul_inplace<<<n_blocks_ei, 256, 0, stream_a>>>(
                d_eg, d_eu, BREAD_EXPERT_INTER);

            /* down projection: gate[EXPERT_INTER] → expert_out[H] */
            bread_matvec(ep.down, d_eg, d_eo,
                         H, BREAD_EXPERT_INTER, (int)ep.down_type);

            CUDA_CHECK(cudaStreamSynchronize(stream_a));

            /* Weighted accumulate into hidden */
            scale_accum<<<n_blocks_h, 256, 0, stream_a>>>(
                d_hidden, d_eo, expert_weights[k], H);
            CUDA_CHECK(cudaStreamSynchronize(stream_a));
        }
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
