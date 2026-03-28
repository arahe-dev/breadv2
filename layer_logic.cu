/* layer_logic.cu — GPU + CPU logic for attention/SSM/MoE in one_layer_forward
 *
 * Extracted from one_layer.cu to improve readability and modularity.
 * Each function handles one major branch of the transformer block.
 */

#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bread_utils.h"
#include "bread.h"
#include "gguf.h"
#include "loader.h"
#include "layer_ops.h"

/* External declarations (defined in one_layer.cu) */
extern void bread_matvec(void *w, half *x, half *y,
                         int rows, int cols, int qtype, cudaStream_t stream);
extern void vram_half_to_cpu_float(const half *d_x, float *h_f, int n);
extern void load_expert_tensor_vram(const loader_t *L, const gguf_ctx_t *g,
                                     const char *name, int expert_idx, void **out_ptr, uint32_t *out_type);
extern void *load_vram(const loader_t *L, const gguf_ctx_t *g, const char *name);
extern const float *tensor_ram_f32(const loader_t *L, const gguf_ctx_t *g, const char *name);
extern uint8_t *tensor_ram(const loader_t *L, const gguf_ctx_t *g, const char *name);

/* Kernel launchers (defined in kernels.cu) */
extern void bread_rmsnorm_half(half *x, const float *w, int n, float eps, cudaStream_t s);
extern void bread_silu_mul(half *gate, const half *up, int n, cudaStream_t s);
extern void bread_scale_accum(half *dst, const half *src, float scale, int n, cudaStream_t s);
extern void bread_copy_half(half *dst, const half *src, int n, cudaStream_t s);

/* Device kernels for internal use only */
static __global__ void scale_accum_kern(half *dst, const half *src, float scale, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + scale * __half2float(src[i]));
}

static __global__ void silu_mul_kern(half *gate, const half *up, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g_f = __half2float(gate[i]);
    float up_f = __half2float(up[i]);
    gate[i] = __float2half((g_f / (1.0f + expf(-g_f))) * up_f);
}

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

/* Forward declarations matching signatures used in one_layer_forward */
void layer_full_attn_forward_gpu_cpu(
    half *d_attn_out,
    const half *d_normed,
    const half *d_q, const half *d_k, const half *d_v,
    float *h_q_full, float *h_q_score, float *h_q_gate,
    float *h_kv_k, float *h_kv_v, float *h_attn_out, float *h_scores,
    int layer_idx, int pos,
    loader_t *L, gguf_ctx_t *g, weight_cache_t *wc,
    cudaStream_t stream_a)
{
    const bread_model_config_t *cfg = bread_model_config_get();
    char nm[128];
    int H = cfg->hidden_dim;

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

    bread_matvec(d_qw, d_normed, (half*)d_q, cfg->q_proj_dim, H, GGML_TYPE_Q4_K, stream_a);
    bread_matvec(d_kw, d_normed, (half*)d_k, cfg->kv_proj_dim, H, GGML_TYPE_Q4_K, stream_a);
    bread_matvec(d_vw, d_normed, (half*)d_v, cfg->kv_proj_dim, H, GGML_TYPE_Q6_K, stream_a);

    CUDA_CHECK(cudaStreamSynchronize(stream_a));
    if (!wc) { cudaFree(d_qw); cudaFree(d_kw); cudaFree(d_vw); }

    /* Note: Full CPU attention logic from one_layer.cu lines 1301-1366 would go here
     * For now, we reference it stays in one_layer_forward() directly.
     * Task 5 Phase 2 can extract this into a separate helper function.
     */
}

void layer_ssm_forward_gpu_cpu(
    half *d_attn_out,
    const half *d_normed,
    const half *d_qkv, const half *d_z, const half *d_alpha, const half *d_beta,
    float *h_qkv, float *h_z, float *h_alpha, float *h_beta,
    int layer_idx, int pos,
    loader_t *L, gguf_ctx_t *g, weight_cache_t *wc,
    cudaStream_t stream_a)
{
    const bread_model_config_t *cfg = bread_model_config_get();
    char nm[128];
    int H = cfg->hidden_dim;

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

    bread_matvec(d_qkv_w, d_normed, (half*)d_qkv, cfg->ssm_qkv_dim, H, GGML_TYPE_Q4_K, stream_a);
    bread_matvec(d_gate_w, d_normed, (half*)d_z, cfg->ssm_z_dim, H, GGML_TYPE_Q4_K, stream_a);
    bread_matvec(d_alpha_w, d_normed, (half*)d_alpha, cfg->ssm_num_v_heads, H, GGML_TYPE_Q4_K, stream_a);
    bread_matvec(d_beta_w, d_normed, (half*)d_beta, cfg->ssm_num_v_heads, H, GGML_TYPE_Q4_K, stream_a);

    CUDA_CHECK(cudaDeviceSynchronize());
    if (!wc) {
        cudaFree(d_qkv_w);
        cudaFree(d_gate_w);
        cudaFree(d_alpha_w);
        cudaFree(d_beta_w);
    }

    /* Note: Full CPU SSM recurrence logic from one_layer.cu lines 1368-1534 would go here
     * For now, it remains in one_layer_forward() directly.
     */
}
