/* sriracha.cu — Step 1: Draft model runner for speculative decoding.
 *
 * Qwen3.5 0.8B = same hybrid SSM/attention architecture as 35B-A3B
 * but scaled down with dense FFN (no MoE routing), all Q8_0.
 *
 * Forward pass per layer:
 *   1. pre-attn RMSNorm
 *   2. SSM layer: GPU(qkv/gate/alpha/beta proj) + CPU GatedDeltaNet recurrence
 *      Full-attn layer: GPU(Q/K/V proj) + CPU GQA + sigmoid gate
 *   3. residual add
 *   4. post-attn RMSNorm (blk.N.post_attention_norm.weight)
 *   5. dense FFN: GPU gate+up+silu+down, residual add
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

#include "sriracha.h"
#include "bread_utils.h"
#include "layer_ops.h"
#include "gguf.h"
#include "loader.h"

/* ------------------------------------------------------------------ */
/* External: bread_matvec (kernels.cu) — supports Q4_K, Q6_K, Q8_0   */
/* ------------------------------------------------------------------ */
extern void bread_matvec(void *w, half *x, half *y,
                          int rows, int cols, int qtype, cudaStream_t stream);

/* ------------------------------------------------------------------ */
/* Quant type constants (match kernels.cu / ggml_type values)          */
/* ------------------------------------------------------------------ */
#define QTYPE_Q8_0   8
#define QTYPE_F32    0
#define QTYPE_F16    1

/* Q8_0 embedding dequant */
#define Q8_0_BLOCK_BYTES  34
#define Q8_0_BLOCK_ELEMS  32

/* Q4_K embedding dequant */
#define Q4K_BLOCK_BYTES  144
#define Q4K_BLOCK_ELEMS  256

/* ================================================================== */
/* Static GPU kernels                                                  */
/* ================================================================== */

/* RMSNorm of d_x[n] (half), weighted by d_w[n] (float), in-place */
static __global__ void sr_rmsnorm(half *x, const float *w, int n, float eps)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < n; i += 256)
        acc += __half2float(x[i]) * __half2float(x[i]);
    sdata[tid] = acc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / (float)n + eps);
    for (int i = tid; i < n; i += 256)
        x[i] = __float2half(__half2float(x[i]) * rms * w[i]);
}

/* SiLU(gate[i]) * up[i]  →  gate[i], in-place */
static __global__ void sr_silu_mul(half *gate, const half *up, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    gate[i] = __float2half((g / (1.0f + expf(-g))) * __half2float(up[i]));
}

/* dst[i] += scale * src[i] */
static __global__ void sr_scale_accum(half *dst, const half *src, float scale, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + scale * __half2float(src[i]));
}

/* ================================================================== */
/* Helpers: upload tensors to VRAM                                     */
/* ================================================================== */

static float *upload_f32(const loader_t *L, const gguf_ctx_t *g, const char *name)
{
    const gguf_tensor_t *t = gguf_find_tensor(g, name);
    if (!t) return NULL;
    float *d;
    CUDA_CHECK(cudaMalloc((void **)&d, t->size));
    CUDA_CHECK(cudaMemcpy(d, L->pinned_data + L->data_offset + t->offset,
                          t->size, cudaMemcpyHostToDevice));
    return d;
}

static void *upload_any(const loader_t *L, const gguf_ctx_t *g,
                        const char *name, uint32_t *type_out)
{
    const gguf_tensor_t *t = gguf_find_tensor(g, name);
    if (!t) { if (type_out) *type_out = 0; return NULL; }
    void *d;
    CUDA_CHECK(cudaMalloc(&d, t->size));
    CUDA_CHECK(cudaMemcpy(d, L->pinned_data + L->data_offset + t->offset,
                          t->size, cudaMemcpyHostToDevice));
    if (type_out) *type_out = t->type;
    return d;
}

/* Direct pointer into pinned RAM (F32, no copy) */
static const float *pinned_f32(const loader_t *L, const gguf_ctx_t *g, const char *name)
{
    const gguf_tensor_t *t = gguf_find_tensor(g, name);
    if (!t) return NULL;
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "sriracha: %s expected F32\n", name);
        return NULL;
    }
    return (const float *)(L->pinned_data + L->data_offset + t->offset);
}

/* ================================================================== */
/* Config inference from GGUF tensor shapes                            */
/* ================================================================== */

static void infer_config(const loader_t *L, const gguf_ctx_t *g,
                         sriracha_cfg_t *cfg)
{
    char nm[128];
    const gguf_tensor_t *t;
    memset(cfg, 0, sizeof(*cfg));

    /* hidden_dim, vocab_size from embedding table */
    t = gguf_find_tensor(g, "token_embd.weight");
    if (t && t->n_dims >= 2) {
        cfg->hidden_dim = (int)t->dims[0];   /* cols = hidden_dim */
        cfg->vocab_size = (int)t->dims[1];   /* rows = vocab_size */
    }

    /* Count layers */
    for (cfg->num_layers = 0; ; cfg->num_layers++) {
        snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", cfg->num_layers);
        if (!gguf_find_tensor(g, nm)) break;
    }

    /* Detect full_attn_interval by finding first layer with attn_q.weight */
    cfg->full_attn_interval = 4;  /* Qwen3.5 default */
    for (int trial = 1; trial <= 8; trial++) {
        snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", trial);
        if (gguf_find_tensor(g, nm)) {
            cfg->full_attn_interval = trial + 1;
            break;
        }
    }

    /* Full-attention config from first full-attn layer */
    {
        int fa_idx = cfg->full_attn_interval - 1;

        snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", fa_idx);
        t = gguf_find_tensor(g, nm);
        cfg->has_qk_norm = (t != NULL);
        cfg->head_dim_qk = t ? (int)t->dims[0] : 64;

        snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", fa_idx);
        t = gguf_find_tensor(g, nm);
        if (t && t->n_dims >= 2) cfg->q_proj_dim = (int)t->dims[1];

        snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", fa_idx);
        t = gguf_find_tensor(g, nm);
        if (t && t->n_dims >= 2) {
            cfg->kv_proj_dim = (int)t->dims[1];
            cfg->num_kv_heads = cfg->kv_proj_dim / cfg->head_dim_qk;
        }

        /* attn_output.weight: dims[0]=attn_out_dim(input), dims[1]=hidden(output) */
        snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", fa_idx);
        t = gguf_find_tensor(g, nm);
        if (t && t->n_dims >= 2) {
            cfg->attn_out_dim = (int)t->dims[0];
            cfg->num_q_heads  = cfg->attn_out_dim / cfg->head_dim_qk;
        }

        cfg->head_dim_v    = cfg->head_dim_qk;
        cfg->head_dim_rope = cfg->head_dim_qk;

        /* Gate: q_proj_dim = num_q_heads * (head_dim_qk + head_dim_gate) */
        if (cfg->num_q_heads > 0) {
            int per_head = cfg->q_proj_dim / cfg->num_q_heads;
            if (per_head > cfg->head_dim_qk) {
                cfg->head_dim_gate = per_head - cfg->head_dim_qk;
                cfg->has_attn_gate = 1;
            }
        }
    }

    /* SSM config from first SSM layer (layer 0) */
    {
        snprintf(nm, sizeof(nm), "blk.0.attn_qkv.weight");
        t = gguf_find_tensor(g, nm);
        if (t && t->n_dims >= 2) cfg->ssm_qkv_dim = (int)t->dims[1];

        snprintf(nm, sizeof(nm), "blk.0.attn_gate.weight");
        t = gguf_find_tensor(g, nm);
        if (t && t->n_dims >= 2) cfg->ssm_z_dim = (int)t->dims[1];

        snprintf(nm, sizeof(nm), "blk.0.ssm_norm.weight");
        t = gguf_find_tensor(g, nm);
        cfg->ssm_head_dim = t ? (int)t->dims[0] : 128;

        if (cfg->ssm_head_dim > 0 && cfg->ssm_z_dim > 0)
            cfg->ssm_num_v_heads = cfg->ssm_z_dim / cfg->ssm_head_dim;
        if (cfg->ssm_head_dim > 0 && cfg->ssm_qkv_dim > 0)
            cfg->ssm_num_k_heads = (cfg->ssm_qkv_dim / 3) / cfg->ssm_head_dim;

        /* conv kernel from ssm_conv1d.weight dims[0] */
        snprintf(nm, sizeof(nm), "blk.0.ssm_conv1d.weight");
        t = gguf_find_tensor(g, nm);
        cfg->ssm_conv_kernel = t ? (int)t->dims[0] : 4;
    }

    /* FFN intermediate from first layer */
    snprintf(nm, sizeof(nm), "blk.0.ffn_gate.weight");
    t = gguf_find_tensor(g, nm);
    if (t && t->n_dims >= 2) cfg->ffn_intermediate = (int)t->dims[1];

    cfg->kv_cache_len   = 4096;
    cfg->rms_eps        = 1e-6f;
    cfg->rope_freq_base = 1000000.0f;

    (void)L; /* unused — metadata read directly from g */
}

/* ================================================================== */
/* Embedding dequant: one token row → host half[]                     */
/* ================================================================== */

static void embed_draft(const loader_t *L, const gguf_ctx_t *g,
                         int32_t token, half *h_out, int hidden_dim)
{
    const gguf_tensor_t *t = gguf_find_tensor(g, "token_embd.weight");
    if (!t) { fprintf(stderr, "sriracha: token_embd.weight missing\n"); exit(1); }
    const uint8_t *base = L->pinned_data + L->data_offset + t->offset;

    if (t->type == GGML_TYPE_Q8_0) {
        int nblocks = hidden_dim / Q8_0_BLOCK_ELEMS;
        size_t row_bytes = (size_t)nblocks * Q8_0_BLOCK_BYTES;
        const uint8_t *row = base + (size_t)token * row_bytes;
        for (int b = 0; b < nblocks; b++) {
            const uint8_t *blk = row + b * Q8_0_BLOCK_BYTES;
            uint16_t d_raw; memcpy(&d_raw, blk, 2);
            float d = bread_h2f(d_raw);
            const int8_t *qs = (const int8_t *)(blk + 2);
            for (int i = 0; i < Q8_0_BLOCK_ELEMS; i++)
                h_out[b * Q8_0_BLOCK_ELEMS + i] = __float2half(d * (float)qs[i]);
        }
    } else if (t->type == GGML_TYPE_Q4_K) {
        int nblocks = hidden_dim / Q4K_BLOCK_ELEMS;
        size_t row_bytes = (size_t)nblocks * Q4K_BLOCK_BYTES;
        const uint8_t *row = base + (size_t)token * row_bytes;
        for (int b = 0; b < nblocks; b++) {
            const uint8_t *blk    = row + b * Q4K_BLOCK_BYTES;
            const uint8_t *scales = blk + 4;
            const uint8_t *qs     = blk + 16;
            uint16_t d_raw, dmin_raw;
            memcpy(&d_raw,    blk,     2);
            memcpy(&dmin_raw, blk + 2, 2);
            float d = bread_h2f(d_raw), dmin = bread_h2f(dmin_raw);
            int is = 0;
            const uint8_t *q = qs;
            half *dst = h_out + b * Q4K_BLOCK_ELEMS;
            for (int grp = 0; grp < 4; grp++) {
                uint8_t sc0, mn0, sc1, mn1;
                if (is < 4) { sc0 = scales[is] & 63; mn0 = scales[is+4] & 63; }
                else { sc0 = (scales[is+4] & 0x0F) | ((scales[is-4] >> 6) << 4);
                       mn0 = (scales[is+4] >>   4) | ((scales[is  ] >> 6) << 4); }
                is++;
                if (is < 4) { sc1 = scales[is] & 63; mn1 = scales[is+4] & 63; }
                else { sc1 = (scales[is+4] & 0x0F) | ((scales[is-4] >> 6) << 4);
                       mn1 = (scales[is+4] >>   4) | ((scales[is  ] >> 6) << 4); }
                is++;
                float d0 = d * sc0, m0 = dmin * mn0;
                float d1 = d * sc1, m1 = dmin * mn1;
                for (int l = 0; l < 32; l++) dst[grp*64+l]    = __float2half(d0*(q[l]&0xF)-m0);
                for (int l = 0; l < 32; l++) dst[grp*64+32+l] = __float2half(d1*(q[l]>>4) -m1);
                q += 32;
            }
        }
    } else if (t->type == GGML_TYPE_F16) {
        const half *src = (const half *)base + (size_t)token * hidden_dim;
        memcpy(h_out, src, (size_t)hidden_dim * sizeof(half));
    } else {
        fprintf(stderr, "sriracha: unsupported emb type %u\n", t->type);
        exit(1);
    }
}

/* ================================================================== */
/* Half ↔ float conversion (CPU ↔ VRAM)                               */
/* ================================================================== */

/* VRAM half[] → host float[] (synchronous, default stream) */
static void vram_to_host_f32(const half *d_src, float *h_dst, int n, half *h_tmp)
{
    CUDA_CHECK(cudaMemcpy(h_tmp, d_src, (size_t)n * sizeof(half),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) h_dst[i] = __half2float(h_tmp[i]);
}

/* host float[] → VRAM half[] (synchronous, default stream) */
static void host_f32_to_vram(const float *h_src, half *d_dst, int n, half *h_tmp)
{
    for (int i = 0; i < n; i++) h_tmp[i] = __float2half(h_src[i]);
    CUDA_CHECK(cudaMemcpy(d_dst, h_tmp, (size_t)n * sizeof(half),
                          cudaMemcpyHostToDevice));
}

/* ================================================================== */
/* Simple RoPE (standard, no mrope sections)                           */
/* ================================================================== */

static void cpu_rope_simple(float *x, int n_heads, int head_dim,
                             int rope_dim, float freq_base, int pos)
{
    int half = rope_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float *xh = x + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq  = 1.0f / powf(freq_base, (float)(2 * i) / (float)rope_dim);
            float angle = (float)pos * freq;
            float c = cosf(angle), s = sinf(angle);
            float x0 = xh[i], x1 = xh[i + half];
            xh[i]        = x0 * c - x1 * s;
            xh[i + half] = x0 * s + x1 * c;
        }
    }
}

/* ================================================================== */
/* Layer type helpers                                                   */
/* ================================================================== */

static int sr_is_full_attn(int layer_idx, int interval)
{
    return ((layer_idx + 1) % interval) == 0;
}

/* Count full-attn layers in [0, layer_idx) */
static int sr_kv_idx(int layer_idx, int interval)
{
    /* Full-attn layers: interval-1, 2*interval-1, ...
     * Index among full-attn layers = floor(layer_idx / interval) */
    return layer_idx / interval;
}

static int sr_ssm_idx(int layer_idx, int interval)
{
    /* Count SSM layers before layer_idx (inclusive) would be:
     * layer_idx + 1 - (layer_idx/interval + 1) for full-attn
     * For strictly before: layer_idx - (layer_idx / interval) */
    int full_before = layer_idx / interval;   /* full-attn layers in [0..layer_idx) */
    return layer_idx - full_before;
}

/* ================================================================== */
/* Single-token forward pass                                           */
/* ================================================================== */

static int32_t sriracha_forward(sriracha_t *sr, int32_t token, int pos)
{
    const sriracha_cfg_t *cfg = &sr->cfg;
    const int H = cfg->hidden_dim;
    const int interval = cfg->full_attn_interval;

    /* --- Embed ---------------------------------------------------- */
    embed_draft(sr->L, sr->g, token, sr->h_emb_row, H);
    CUDA_CHECK(cudaMemcpy(sr->d_hidden, sr->h_emb_row,
                           (size_t)H * sizeof(half), cudaMemcpyHostToDevice));

    /* --- Transformer layers --------------------------------------- */
    for (int l = 0; l < cfg->num_layers; l++) {
        sriracha_layer_t *lw = &sr->layers[l];
        int is_full = sr_is_full_attn(l, interval);
        int nblk_h  = (H + 255) / 256;

        /* == Step 1: pre-attention RMSNorm ======================== */
        CUDA_CHECK(cudaMemcpyAsync(sr->d_normed, sr->d_hidden,
                                    (size_t)H * sizeof(half),
                                    cudaMemcpyDeviceToDevice, sr->stream));
        sr_rmsnorm<<<1, 256, 0, sr->stream>>>(sr->d_normed, lw->attn_norm_w,
                                               H, cfg->rms_eps);
        CUDA_CHECK(cudaStreamSynchronize(sr->stream));

        /* == Step 2: attention branch ============================= */
        if (is_full) {
            /* -- Full-attention layer -- */
            int kv_idx = sr_kv_idx(l, interval);

            /* Q/K/V projections on GPU */
            bread_matvec(lw->attn_q_w, sr->d_normed, sr->d_q,
                         cfg->q_proj_dim,  H, (int)lw->attn_q_type, sr->stream);
            bread_matvec(lw->attn_k_w, sr->d_normed, sr->d_k,
                         cfg->kv_proj_dim, H, (int)lw->attn_k_type, sr->stream);
            bread_matvec(lw->attn_v_w, sr->d_normed, sr->d_v,
                         cfg->kv_proj_dim, H, (int)lw->attn_v_type, sr->stream);
            CUDA_CHECK(cudaStreamSynchronize(sr->stream));

            /* Download to host */
            vram_to_host_f32(sr->d_q, sr->h_q, cfg->q_proj_dim, sr->h_half_tmp);
            vram_to_host_f32(sr->d_k, sr->h_k, cfg->kv_proj_dim, sr->h_half_tmp);
            vram_to_host_f32(sr->d_v, sr->h_v, cfg->kv_proj_dim, sr->h_half_tmp);

            /* Split Q into q_score | q_gate (interleaved per head)
             * Use h_z (ssm_z_dim=2048) for q_score and h_conv_out (ssm_qkv_dim=6144)
             * for q_gate — these are unused during full-attn layers. */
            int per_head = cfg->head_dim_qk + cfg->head_dim_gate;
            float *q_score_buf = sr->h_z;        /* num_q_heads * head_dim_qk floats */
            float *q_gate_buf  = sr->h_conv_out;  /* num_q_heads * head_dim_gate floats */

            for (int qh = 0; qh < cfg->num_q_heads; qh++) {
                const float *src = sr->h_q + qh * per_head;
                memcpy(q_score_buf + qh * cfg->head_dim_qk,
                       src, (size_t)cfg->head_dim_qk * sizeof(float));
                if (cfg->has_attn_gate)
                    memcpy(q_gate_buf + qh * cfg->head_dim_gate,
                           src + cfg->head_dim_qk,
                           (size_t)cfg->head_dim_gate * sizeof(float));
            }

            /* Per-head QK norm */
            if (cfg->has_qk_norm && lw->h_q_norm_w && lw->h_k_norm_w) {
                apply_per_head_rms_norm(q_score_buf, lw->h_q_norm_w,
                                        cfg->num_q_heads, cfg->head_dim_qk,
                                        cfg->rms_eps);
                apply_per_head_rms_norm(sr->h_k, lw->h_k_norm_w,
                                        cfg->num_kv_heads, cfg->head_dim_qk,
                                        cfg->rms_eps);
            }

            /* RoPE */
            cpu_rope_simple(q_score_buf, cfg->num_q_heads, cfg->head_dim_qk,
                             cfg->head_dim_rope, cfg->rope_freq_base, pos);
            cpu_rope_simple(sr->h_k, cfg->num_kv_heads, cfg->head_dim_qk,
                             cfg->head_dim_rope, cfg->rope_freq_base, pos);

            /* Append K, V to KV cache */
            if (sr->kv_len[kv_idx] >= cfg->kv_cache_len) {
                fprintf(stderr, "sriracha: KV cache full layer %d\n", l); exit(1);
            }
            memcpy(sr->kv_k[kv_idx] +
                   (size_t)sr->kv_len[kv_idx] * cfg->kv_proj_dim,
                   sr->h_k, (size_t)cfg->kv_proj_dim * sizeof(float));
            memcpy(sr->kv_v[kv_idx] +
                   (size_t)sr->kv_len[kv_idx] * cfg->kv_proj_dim,
                   sr->h_v, (size_t)cfg->kv_proj_dim * sizeof(float));
            sr->kv_len[kv_idx]++;
            int kv_len = sr->kv_len[kv_idx];

            /* GQA attention — output into h_attn (attn_out_dim = 2048 floats) */
            int heads_per_kv = cfg->num_q_heads / cfg->num_kv_heads;
            float *attn_out = sr->h_attn;   /* num_q_heads * head_dim_v floats */
            memset(attn_out, 0, (size_t)cfg->attn_out_dim * sizeof(float));

            for (int qh = 0; qh < cfg->num_q_heads; qh++) {
                int kv_h = qh / heads_per_kv;
                float *qhp = q_score_buf + qh * cfg->head_dim_qk;
                float *ohp = attn_out    + qh * cfg->head_dim_v;

                for (int p = 0; p < kv_len; p++) {
                    float *kp = sr->kv_k[kv_idx]
                                + (size_t)p * cfg->kv_proj_dim
                                + kv_h * cfg->head_dim_qk;
                    float dot = 0.0f;
                    for (int d = 0; d < cfg->head_dim_qk; d++)
                        dot += qhp[d] * kp[d];
                    sr->h_scores[p] = dot / sqrtf((float)cfg->head_dim_qk);
                }
                cpu_softmax(sr->h_scores, kv_len);

                for (int p = 0; p < kv_len; p++) {
                    float *vp = sr->kv_v[kv_idx]
                                + (size_t)p * cfg->kv_proj_dim
                                + kv_h * cfg->head_dim_v;
                    for (int d = 0; d < cfg->head_dim_v; d++)
                        ohp[d] += sr->h_scores[p] * vp[d];
                }

                /* Sigmoid gate (element-wise from q_gate) */
                if (cfg->has_attn_gate) {
                    float *ghp = q_gate_buf + qh * cfg->head_dim_gate;
                    int gd = cfg->head_dim_gate < cfg->head_dim_v
                           ? cfg->head_dim_gate : cfg->head_dim_v;
                    for (int d = 0; d < gd; d++)
                        ohp[d] *= cpu_sigmoid(ghp[d]);
                }
            }

            /* Upload attn output → d_attn_out */
            host_f32_to_vram(attn_out, sr->d_attn_out,
                              cfg->attn_out_dim, sr->h_half_tmp);

            /* O projection: d_attn_out[attn_out_dim] → d_normed[H] */
            bread_matvec(lw->attn_out_w, sr->d_attn_out, sr->d_normed,
                         H, cfg->attn_out_dim, (int)lw->attn_out_type, sr->stream);
            CUDA_CHECK(cudaStreamSynchronize(sr->stream));

        } else {
            /* -- SSM (GatedDeltaNet) layer -- */
            int ssm_idx = sr_ssm_idx(l, interval);

            /* GPU: project qkv, gate, alpha, beta */
            bread_matvec(lw->ssm_qkv_w, sr->d_normed, sr->d_qkv,
                         cfg->ssm_qkv_dim, H, (int)lw->ssm_qkv_type, sr->stream);
            bread_matvec(lw->ssm_gate_w, sr->d_normed, sr->d_z,
                         cfg->ssm_z_dim, H, (int)lw->ssm_gate_type, sr->stream);
            bread_matvec(lw->ssm_alpha_w, sr->d_normed, sr->d_alpha,
                         cfg->ssm_num_v_heads, H, (int)lw->ssm_alpha_type, sr->stream);
            bread_matvec(lw->ssm_beta_w, sr->d_normed, sr->d_beta,
                         cfg->ssm_num_v_heads, H, (int)lw->ssm_beta_type, sr->stream);
            CUDA_CHECK(cudaStreamSynchronize(sr->stream));

            /* Download to host */
            vram_to_host_f32(sr->d_qkv,   sr->h_qkv,   cfg->ssm_qkv_dim,    sr->h_half_tmp);
            vram_to_host_f32(sr->d_z,     sr->h_z,     cfg->ssm_z_dim,      sr->h_half_tmp);
            vram_to_host_f32(sr->d_alpha, sr->h_alpha,  cfg->ssm_num_v_heads, sr->h_half_tmp);
            vram_to_host_f32(sr->d_beta,  sr->h_beta,   cfg->ssm_num_v_heads, sr->h_half_tmp);

            /* Conv1d step */
            float *conv_state = sr->ssm_conv[ssm_idx];
            cpu_conv1d_step(conv_state, sr->h_qkv, lw->ssm_conv_w,
                            sr->h_conv_out, cfg->ssm_qkv_dim, cfg->ssm_conv_kernel);
            /* Update conv state (shift left, append new) */
            memmove(conv_state, conv_state + cfg->ssm_qkv_dim,
                    (size_t)(cfg->ssm_conv_kernel - 2) * cfg->ssm_qkv_dim * sizeof(float));
            memcpy(conv_state + (cfg->ssm_conv_kernel - 2) * cfg->ssm_qkv_dim,
                   sr->h_qkv, (size_t)cfg->ssm_qkv_dim * sizeof(float));

            /* SiLU on conv output */
            cpu_silu_inplace(sr->h_conv_out, cfg->ssm_qkv_dim);

            /* Split conv output into Q / K / V */
            float *lin_q = sr->h_conv_out;
            float *lin_k = sr->h_conv_out + cfg->ssm_num_k_heads * cfg->ssm_head_dim;
            float *lin_v = sr->h_conv_out + 2 * cfg->ssm_num_k_heads * cfg->ssm_head_dim;

            /* L2-normalize Q heads + scale */
            float inv_scale = 1.0f / sqrtf((float)cfg->ssm_head_dim);
            for (int h = 0; h < cfg->ssm_num_k_heads; h++) {
                float *qh = lin_q + h * cfg->ssm_head_dim;
                cpu_l2_norm_bare(qh, cfg->ssm_head_dim, cfg->rms_eps);
                for (int d = 0; d < cfg->ssm_head_dim; d++) qh[d] *= inv_scale;
            }

            /* L2-normalize K heads */
            for (int h = 0; h < cfg->ssm_num_k_heads; h++) {
                float *kh = lin_k + h * cfg->ssm_head_dim;
                cpu_l2_norm_bare(kh, cfg->ssm_head_dim, cfg->rms_eps);
            }

            /* Repeat heads: K/V from num_k_heads to num_v_heads */
            cpu_repeat_heads(sr->h_ssm_qrep, lin_q,
                             cfg->ssm_num_k_heads, cfg->ssm_num_v_heads,
                             cfg->ssm_head_dim);
            cpu_repeat_heads(sr->h_ssm_krep, lin_k,
                             cfg->ssm_num_k_heads, cfg->ssm_num_v_heads,
                             cfg->ssm_head_dim);

            /* GatedDeltaNet per v_head */
            float *layer_state = sr->ssm_state[ssm_idx];
            memset(sr->h_attn, 0, (size_t)cfg->ssm_z_dim * sizeof(float));

            for (int vh = 0; vh < cfg->ssm_num_v_heads; vh++) {
                float gate     = cpu_softplus(sr->h_alpha[vh] + lw->ssm_dt[vh])
                                 * lw->ssm_a[vh];
                float beta_g   = cpu_sigmoid(sr->h_beta[vh]);
                float *S       = layer_state
                                 + (size_t)vh * cfg->ssm_head_dim * cfg->ssm_head_dim;
                float *q_h     = sr->h_ssm_qrep + (size_t)vh * cfg->ssm_head_dim;
                float *k_h     = sr->h_ssm_krep + (size_t)vh * cfg->ssm_head_dim;
                float *v_h     = lin_v           + vh * cfg->ssm_head_dim;
                float *o_h     = sr->h_attn      + vh * cfg->ssm_head_dim;

                cpu_delta_net_autoregressive_step(
                    q_h, k_h, v_h, gate, beta_g,
                    S, sr->h_head_tmp, cfg->ssm_head_dim, cfg->ssm_head_dim,
                    sr->h_ssm_sk, sr->h_ssm_d);

                cpu_gated_rms_norm(sr->h_head_tmp,
                                   sr->h_z + vh * cfg->ssm_head_dim,
                                   lw->ssm_norm_w, o_h,
                                   cfg->ssm_head_dim, cfg->rms_eps);
            }

            /* Upload SSM attention output → d_attn_out */
            host_f32_to_vram(sr->h_attn, sr->d_attn_out,
                              cfg->ssm_z_dim, sr->h_half_tmp);

            /* SSM output projection: d_attn_out[ssm_z_dim] → d_normed[H] */
            bread_matvec(lw->ssm_out_w, sr->d_attn_out, sr->d_normed,
                         H, cfg->ssm_z_dim, (int)lw->ssm_out_type, sr->stream);
            CUDA_CHECK(cudaStreamSynchronize(sr->stream));
        }

        /* == Step 3: residual add d_hidden += d_normed ============ */
        sr_scale_accum<<<nblk_h, 256, 0, sr->stream>>>(
            sr->d_hidden, sr->d_normed, 1.0f, H);

        /* == Step 4: post-attn RMSNorm (pre-FFN) ================== */
        CUDA_CHECK(cudaMemcpyAsync(sr->d_normed, sr->d_hidden,
                                    (size_t)H * sizeof(half),
                                    cudaMemcpyDeviceToDevice, sr->stream));
        sr_rmsnorm<<<1, 256, 0, sr->stream>>>(
            sr->d_normed, lw->post_attn_norm_w, H, cfg->rms_eps);
        CUDA_CHECK(cudaStreamSynchronize(sr->stream));

        /* == Step 5: dense FFN ==================================== */
        bread_matvec(lw->ffn_gate_w, sr->d_normed, sr->d_gate,
                     cfg->ffn_intermediate, H, (int)lw->ffn_gate_type, sr->stream);
        bread_matvec(lw->ffn_up_w,   sr->d_normed, sr->d_up,
                     cfg->ffn_intermediate, H, (int)lw->ffn_up_type,   sr->stream);

        {
            int nblk_ffn = (cfg->ffn_intermediate + 255) / 256;
            sr_silu_mul<<<nblk_ffn, 256, 0, sr->stream>>>(
                sr->d_gate, sr->d_up, cfg->ffn_intermediate);
        }

        bread_matvec(lw->ffn_down_w, sr->d_gate, sr->d_normed,
                     H, cfg->ffn_intermediate, (int)lw->ffn_down_type, sr->stream);

        sr_scale_accum<<<nblk_h, 256, 0, sr->stream>>>(
            sr->d_hidden, sr->d_normed, 1.0f, H);
        CUDA_CHECK(cudaStreamSynchronize(sr->stream));
    }

    /* --- Output norm ------------------------------------------- */
    sr_rmsnorm<<<1, 256, 0, sr->stream>>>(sr->d_hidden, sr->d_out_norm_w,
                                           H, cfg->rms_eps);

    /* --- Logit projection -------------------------------------- */
    bread_matvec(sr->d_out_w, sr->d_hidden, sr->d_logits,
                 cfg->vocab_size, H, (int)sr->out_w_type, sr->stream);
    CUDA_CHECK(cudaStreamSynchronize(sr->stream));

    /* --- Greedy argmax ----------------------------------------- */
    CUDA_CHECK(cudaMemcpy(sr->h_logits, sr->d_logits,
                           (size_t)cfg->vocab_size * sizeof(half),
                           cudaMemcpyDeviceToHost));
    int32_t best = 0;
    float best_v = __half2float(sr->h_logits[0]);
    for (int32_t i = 1; i < cfg->vocab_size; i++) {
        float v = __half2float(sr->h_logits[i]);
        if (v > best_v) { best_v = v; best = i; }
    }
    return best;
}

/* ================================================================== */
/* Public API                                                          */
/* ================================================================== */

sriracha_t *sriracha_init(const char *draft_path, int spec_depth)
{
    sriracha_t *sr = (sriracha_t *)calloc(1, sizeof(*sr));
    if (!sr) return NULL;

    printf("[SRIRACHA] Loading draft model: %s\n", draft_path);

    sr->g = gguf_open(draft_path);
    if (!sr->g) { fprintf(stderr, "sriracha: gguf_open failed\n"); free(sr); return NULL; }

    sr->L = loader_init(draft_path);
    if (!sr->L) {
        fprintf(stderr, "sriracha: loader_init failed\n");
        gguf_close(sr->g); free(sr); return NULL;
    }

    /* Config inference */
    infer_config(sr->L, sr->g, &sr->cfg);
    {
        const sriracha_cfg_t *c = &sr->cfg;
        printf("[SRIRACHA] hidden=%d layers=%d vocab=%d interval=%d\n",
               c->hidden_dim, c->num_layers, c->vocab_size, c->full_attn_interval);
        printf("[SRIRACHA] Q=%d KV=%d head_qk=%d head_gate=%d has_gate=%d has_qknorm=%d\n",
               c->num_q_heads, c->num_kv_heads, c->head_dim_qk,
               c->head_dim_gate, c->has_attn_gate, c->has_qk_norm);
        printf("[SRIRACHA] ssm_qkv=%d ssm_z=%d ssm_k_heads=%d ssm_v_heads=%d ssm_hdim=%d ssm_conv=%d\n",
               c->ssm_qkv_dim, c->ssm_z_dim, c->ssm_num_k_heads,
               c->ssm_num_v_heads, c->ssm_head_dim, c->ssm_conv_kernel);
        printf("[SRIRACHA] ffn_intermediate=%d\n", c->ffn_intermediate);
    }

    sr->spec_depth = spec_depth;
    const sriracha_cfg_t *cfg = &sr->cfg;
    const int H = cfg->hidden_dim;

    /* ── VRAM working buffers ──────────────────────────────────── */
    int attn_buf  = cfg->attn_out_dim > cfg->ssm_z_dim
                  ? cfg->attn_out_dim : cfg->ssm_z_dim;
    CUDA_CHECK(cudaMalloc((void **)&sr->d_hidden,   (size_t)H * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&sr->d_normed,   (size_t)H * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&sr->d_qkv,      (size_t)cfg->ssm_qkv_dim    * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&sr->d_z,        (size_t)cfg->ssm_z_dim      * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&sr->d_alpha,    (size_t)cfg->ssm_num_v_heads * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&sr->d_beta,     (size_t)cfg->ssm_num_v_heads * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&sr->d_q,        (size_t)cfg->q_proj_dim     * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&sr->d_k,        (size_t)cfg->kv_proj_dim    * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&sr->d_v,        (size_t)cfg->kv_proj_dim    * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&sr->d_attn_out, (size_t)attn_buf            * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&sr->d_gate,     (size_t)cfg->ffn_intermediate * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&sr->d_up,       (size_t)cfg->ffn_intermediate * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&sr->d_logits,   (size_t)cfg->vocab_size     * sizeof(half)));

    /* ── Output head ───────────────────────────────────────────── */
    sr->d_out_norm_w = upload_f32(sr->L, sr->g, "output_norm.weight");
    sr->d_out_w      = upload_any(sr->L, sr->g, "output.weight", &sr->out_w_type);
    if (!sr->d_out_w)
        sr->d_out_w = upload_any(sr->L, sr->g, "token_embd.weight", &sr->out_w_type);
    if (!sr->d_out_norm_w || !sr->d_out_w) {
        fprintf(stderr, "sriracha: missing output head weights\n"); return NULL;
    }

    /* ── Per-layer weight upload ───────────────────────────────── */
    sr->layers = (sriracha_layer_t *)calloc(cfg->num_layers, sizeof(sriracha_layer_t));
    if (!sr->layers) { fprintf(stderr, "sriracha: OOM layers\n"); return NULL; }

    printf("[SRIRACHA] Uploading %d layers to VRAM...\n", cfg->num_layers);
    uint64_t total_vram = 0;
    char nm[128];

#define UP_F32(fld, fmt, ...) do { \
    snprintf(nm, sizeof(nm), fmt, ##__VA_ARGS__); \
    lw->fld = upload_f32(sr->L, sr->g, nm); \
    const gguf_tensor_t *_t = gguf_find_tensor(sr->g, nm); \
    if (_t) total_vram += _t->size; \
} while (0)

#define UP_ANY(fld, typ, fmt, ...) do { \
    snprintf(nm, sizeof(nm), fmt, ##__VA_ARGS__); \
    lw->fld = upload_any(sr->L, sr->g, nm, &lw->typ); \
    const gguf_tensor_t *_t = gguf_find_tensor(sr->g, nm); \
    if (_t) total_vram += _t->size; \
} while (0)

#define PIN_F32(fld, fmt, ...) do { \
    snprintf(nm, sizeof(nm), fmt, ##__VA_ARGS__); \
    lw->fld = pinned_f32(sr->L, sr->g, nm); \
} while (0)

    for (int l = 0; l < cfg->num_layers; l++) {
        sriracha_layer_t *lw = &sr->layers[l];
        int is_full = sr_is_full_attn(l, cfg->full_attn_interval);

        /* Common to all layers */
        UP_F32(attn_norm_w,     "blk.%d.attn_norm.weight",         l);
        UP_F32(post_attn_norm_w,"blk.%d.post_attention_norm.weight",l);
        UP_ANY(ffn_gate_w, ffn_gate_type, "blk.%d.ffn_gate.weight", l);
        UP_ANY(ffn_up_w,   ffn_up_type,   "blk.%d.ffn_up.weight",   l);
        UP_ANY(ffn_down_w, ffn_down_type, "blk.%d.ffn_down.weight", l);

        if (is_full) {
            /* Full-attention weights */
            UP_ANY(attn_q_w,   attn_q_type,   "blk.%d.attn_q.weight",      l);
            UP_ANY(attn_k_w,   attn_k_type,   "blk.%d.attn_k.weight",      l);
            UP_ANY(attn_v_w,   attn_v_type,   "blk.%d.attn_v.weight",      l);
            UP_ANY(attn_out_w, attn_out_type, "blk.%d.attn_output.weight", l);

            /* QK norm weights — pinned RAM (F32, direct pointer) */
            if (cfg->has_qk_norm) {
                PIN_F32(h_q_norm_w, "blk.%d.attn_q_norm.weight", l);
                PIN_F32(h_k_norm_w, "blk.%d.attn_k_norm.weight", l);
            }
        } else {
            /* SSM weights — GPU for projections */
            UP_ANY(ssm_qkv_w,   ssm_qkv_type,   "blk.%d.attn_qkv.weight",   l);
            UP_ANY(ssm_gate_w,  ssm_gate_type,   "blk.%d.attn_gate.weight",  l);
            UP_ANY(ssm_alpha_w, ssm_alpha_type,  "blk.%d.ssm_alpha.weight",  l);
            UP_ANY(ssm_beta_w,  ssm_beta_type,   "blk.%d.ssm_beta.weight",   l);
            UP_ANY(ssm_out_w,   ssm_out_type,    "blk.%d.ssm_out.weight",    l);

            /* SSM CPU weights — pinned RAM pointers (F32) */
            PIN_F32(ssm_conv_w, "blk.%d.ssm_conv1d.weight", l);
            PIN_F32(ssm_a,      "blk.%d.ssm_a",              l);
            PIN_F32(ssm_dt,     "blk.%d.ssm_dt",             l);
            PIN_F32(ssm_norm_w, "blk.%d.ssm_norm.weight",    l);
        }
    }

#undef UP_F32
#undef UP_ANY
#undef PIN_F32

    printf("[SRIRACHA] Layer weights: %.1f MB VRAM\n",
           (double)total_vram / (1024.0 * 1024.0));

    /* ── KV cache (host RAM) — one per full-attn layer ─────────── */
    sr->n_full_attn_layers = cfg->num_layers / cfg->full_attn_interval;
    sr->kv_k   = (float **)calloc(sr->n_full_attn_layers, sizeof(float *));
    sr->kv_v   = (float **)calloc(sr->n_full_attn_layers, sizeof(float *));
    sr->kv_len = (int    *)calloc(sr->n_full_attn_layers, sizeof(int));
    if (!sr->kv_k || !sr->kv_v || !sr->kv_len) {
        fprintf(stderr, "sriracha: OOM KV arrays\n"); return NULL;
    }
    for (int i = 0; i < sr->n_full_attn_layers; i++) {
        size_t kv_bytes = (size_t)cfg->kv_cache_len * cfg->kv_proj_dim * sizeof(float);
        sr->kv_k[i] = (float *)malloc(kv_bytes);
        sr->kv_v[i] = (float *)malloc(kv_bytes);
        if (!sr->kv_k[i] || !sr->kv_v[i]) {
            fprintf(stderr, "sriracha: OOM KV cache\n"); return NULL;
        }
    }

    /* ── SSM recurrent state (host RAM) — one per SSM layer ──────── */
    sr->n_ssm_layers = cfg->num_layers - sr->n_full_attn_layers;
    sr->ssm_state = (float **)calloc(sr->n_ssm_layers, sizeof(float *));
    sr->ssm_conv  = (float **)calloc(sr->n_ssm_layers, sizeof(float *));
    if (!sr->ssm_state || !sr->ssm_conv) {
        fprintf(stderr, "sriracha: OOM SSM arrays\n"); return NULL;
    }
    for (int i = 0; i < sr->n_ssm_layers; i++) {
        size_t state_bytes = (size_t)cfg->ssm_num_v_heads
                           * cfg->ssm_head_dim * cfg->ssm_head_dim * sizeof(float);
        size_t conv_bytes  = (size_t)(cfg->ssm_conv_kernel - 1)
                           * cfg->ssm_qkv_dim * sizeof(float);
        sr->ssm_state[i] = (float *)calloc(1, state_bytes);
        sr->ssm_conv[i]  = (float *)calloc(1, conv_bytes);
        if (!sr->ssm_state[i] || !sr->ssm_conv[i]) {
            fprintf(stderr, "sriracha: OOM SSM state\n"); return NULL;
        }
    }

    /* ── Host temp buffers ───────────────────────────────────────── */
    int half_tmp_n = cfg->ssm_qkv_dim > cfg->q_proj_dim
                   ? cfg->ssm_qkv_dim : cfg->q_proj_dim;
    half_tmp_n = half_tmp_n > attn_buf ? half_tmp_n : attn_buf;

    sr->h_emb_row    = (half  *)malloc((size_t)H                        * sizeof(half));
    sr->h_qkv        = (float *)malloc((size_t)cfg->ssm_qkv_dim         * sizeof(float));
    sr->h_z          = (float *)malloc((size_t)cfg->ssm_z_dim           * sizeof(float));
    sr->h_alpha      = (float *)malloc((size_t)cfg->ssm_num_v_heads     * sizeof(float));
    sr->h_beta       = (float *)malloc((size_t)cfg->ssm_num_v_heads     * sizeof(float));
    sr->h_conv_out   = (float *)malloc((size_t)cfg->ssm_qkv_dim         * sizeof(float));
    sr->h_ssm_qrep   = (float *)malloc((size_t)cfg->ssm_num_v_heads     * cfg->ssm_head_dim * sizeof(float));
    sr->h_ssm_krep   = (float *)malloc((size_t)cfg->ssm_num_v_heads     * cfg->ssm_head_dim * sizeof(float));
    sr->h_head_tmp   = (float *)malloc((size_t)cfg->ssm_head_dim        * sizeof(float));
    sr->h_ssm_sk     = (float *)malloc((size_t)cfg->ssm_head_dim        * sizeof(float));
    sr->h_ssm_d      = (float *)malloc((size_t)cfg->ssm_head_dim        * sizeof(float));
    sr->h_attn       = (float *)malloc((size_t)attn_buf                 * sizeof(float));
    sr->h_q          = (float *)malloc((size_t)cfg->q_proj_dim          * sizeof(float));
    sr->h_k          = (float *)malloc((size_t)cfg->kv_proj_dim         * sizeof(float));
    sr->h_v          = (float *)malloc((size_t)cfg->kv_proj_dim         * sizeof(float));
    sr->h_scores     = (float *)malloc((size_t)cfg->kv_cache_len        * sizeof(float));
    sr->h_half_tmp   = (half  *)malloc((size_t)half_tmp_n               * sizeof(half));
    sr->h_logits     = (half  *)malloc((size_t)cfg->vocab_size          * sizeof(half));

    if (!sr->h_emb_row || !sr->h_qkv || !sr->h_z || !sr->h_alpha || !sr->h_beta ||
        !sr->h_conv_out || !sr->h_ssm_qrep || !sr->h_ssm_krep || !sr->h_head_tmp ||
        !sr->h_ssm_sk || !sr->h_ssm_d || !sr->h_attn || !sr->h_q || !sr->h_k ||
        !sr->h_v || !sr->h_scores || !sr->h_half_tmp || !sr->h_logits) {
        fprintf(stderr, "sriracha: OOM host buffers\n"); return NULL;
    }

    CUDA_CHECK(cudaStreamCreate(&sr->stream));

    printf("[SRIRACHA] Ready. spec_depth=%d  full_attn_layers=%d  ssm_layers=%d\n",
           spec_depth, sr->n_full_attn_layers, sr->n_ssm_layers);
    return sr;
}

/* ------------------------------------------------------------------ */

void sriracha_prefill(sriracha_t *sr, const int32_t *tokens, int n)
{
    /* Reset all state */
    for (int i = 0; i < sr->n_full_attn_layers; i++) sr->kv_len[i] = 0;
    for (int i = 0; i < sr->n_ssm_layers; i++) {
        size_t state_bytes = (size_t)sr->cfg.ssm_num_v_heads
                           * sr->cfg.ssm_head_dim * sr->cfg.ssm_head_dim * sizeof(float);
        size_t conv_bytes  = (size_t)(sr->cfg.ssm_conv_kernel - 1)
                           * sr->cfg.ssm_qkv_dim * sizeof(float);
        memset(sr->ssm_state[i], 0, state_bytes);
        memset(sr->ssm_conv[i],  0, conv_bytes);
    }

    for (int p = 0; p < n; p++)
        sriracha_forward(sr, tokens[p], p);
}

/* ------------------------------------------------------------------ */

int sriracha_draft_from(sriracha_t *sr, int32_t seed_token, int cur_pos,
                         int32_t *out_tokens)
{
    int32_t tok = seed_token;
    for (int i = 0; i < sr->spec_depth; i++) {
        tok = sriracha_forward(sr, tok, cur_pos + i);
        out_tokens[i] = tok;
    }
    sr->n_drafted += sr->spec_depth;
    return sr->spec_depth;
}

/* ------------------------------------------------------------------ */

void sriracha_rewind(sriracha_t *sr, int pos)
{
    /* Rewind KV cache to position pos */
    for (int i = 0; i < sr->n_full_attn_layers; i++)
        if (sr->kv_len[i] > pos) sr->kv_len[i] = pos;
    /* Note: SSM state is stateful; rolling back requires checkpointing.
     * Step 2 will add SSM state snapshots before each draft round. */
}

/* ------------------------------------------------------------------ */

void sriracha_free(sriracha_t *sr)
{
    if (!sr) return;
    const sriracha_cfg_t *cfg = &sr->cfg;
    int interval = cfg->full_attn_interval;

    for (int l = 0; l < cfg->num_layers; l++) {
        sriracha_layer_t *lw = &sr->layers[l];
        cudaFree(lw->attn_norm_w);
        cudaFree(lw->post_attn_norm_w);
        cudaFree(lw->ffn_gate_w);
        cudaFree(lw->ffn_up_w);
        cudaFree(lw->ffn_down_w);
        if (sr_is_full_attn(l, interval)) {
            cudaFree(lw->attn_q_w);
            cudaFree(lw->attn_k_w);
            cudaFree(lw->attn_v_w);
            cudaFree(lw->attn_out_w);
            /* h_q_norm_w / h_k_norm_w are pinned RAM pointers — not freed here */
        } else {
            cudaFree(lw->ssm_qkv_w);
            cudaFree(lw->ssm_gate_w);
            cudaFree(lw->ssm_alpha_w);
            cudaFree(lw->ssm_beta_w);
            cudaFree(lw->ssm_out_w);
            /* ssm_conv_w / ssm_a / ssm_dt / ssm_norm_w are pinned RAM — not freed */
        }
    }
    free(sr->layers);

    for (int i = 0; i < sr->n_full_attn_layers; i++) {
        free(sr->kv_k[i]); free(sr->kv_v[i]);
    }
    free(sr->kv_k); free(sr->kv_v); free(sr->kv_len);

    for (int i = 0; i < sr->n_ssm_layers; i++) {
        free(sr->ssm_state[i]); free(sr->ssm_conv[i]);
    }
    free(sr->ssm_state); free(sr->ssm_conv);

    cudaFree(sr->d_hidden); cudaFree(sr->d_normed);
    cudaFree(sr->d_qkv);   cudaFree(sr->d_z);
    cudaFree(sr->d_alpha); cudaFree(sr->d_beta);
    cudaFree(sr->d_q);     cudaFree(sr->d_k);     cudaFree(sr->d_v);
    cudaFree(sr->d_attn_out);
    cudaFree(sr->d_gate);  cudaFree(sr->d_up);    cudaFree(sr->d_logits);
    cudaFree(sr->d_out_norm_w); cudaFree(sr->d_out_w);

    free(sr->h_emb_row);   free(sr->h_qkv);    free(sr->h_z);
    free(sr->h_alpha);     free(sr->h_beta);   free(sr->h_conv_out);
    free(sr->h_ssm_qrep);  free(sr->h_ssm_krep);
    free(sr->h_head_tmp);  free(sr->h_ssm_sk); free(sr->h_ssm_d);
    free(sr->h_attn);      free(sr->h_q);      free(sr->h_k);
    free(sr->h_v);         free(sr->h_scores); free(sr->h_half_tmp);
    free(sr->h_logits);

    cudaStreamDestroy(sr->stream);
    loader_free(sr->L);
    gguf_close(sr->g);
    free(sr);
}
