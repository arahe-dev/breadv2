#ifndef SRIRACHA_H
#define SRIRACHA_H

/* sriracha.h — Draft model runner for speculative decoding (SRIRACHA Step 1).
 *
 * The Qwen3.5 0.8B draft model is the SAME hybrid SSM/attention architecture
 * as Qwen3.5 35B-A3B, scaled down, with dense FFN (no MoE routing), all Q8_0.
 *
 * Architecture per layer (0.8B params):
 *   hidden_dim=1024, layers=24, full_attn_interval=4
 *   SSM layers (0,1,2,4,5,6,...): GatedDeltaNet recurrence
 *   Full-attn layers (3,7,11,...): GQA + sigmoid gate
 *   FFN: dense SwiGLU (gate+up+silu+down, no MoE routing)
 *   Weights: Q8_0 throughout (supported by updated bread_matvec)
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "gguf.h"
#include "loader.h"

/* ------------------------------------------------------------------ */
/* Architecture config (inferred from tensor shapes at init)           */
/* ------------------------------------------------------------------ */

typedef struct {
    int hidden_dim;
    int num_layers;
    int vocab_size;
    int full_attn_interval;   /* every Nth layer is full attention (4 for 0.8B) */

    /* Full-attention params */
    int num_q_heads;
    int num_kv_heads;
    int head_dim_qk;          /* per-head key/query dim (256) */
    int head_dim_gate;        /* per-head sigmoid gate dim (256, if has_attn_gate) */
    int head_dim_v;           /* per-head value dim (= head_dim_qk for Qwen3.5) */
    int has_attn_gate;        /* 1 if Q proj includes gate component */
    int has_qk_norm;          /* 1 if attn_q_norm.weight present */
    int q_proj_dim;           /* num_q_heads * (head_dim_qk + head_dim_gate) */
    int kv_proj_dim;          /* num_kv_heads * head_dim_qk */
    int attn_out_dim;         /* num_q_heads * head_dim_v */
    int kv_cache_len;         /* max KV context positions */
    int head_dim_rope;

    /* SSM (GatedDeltaNet) params */
    int ssm_qkv_dim;          /* combined Q+K+V output (6144) */
    int ssm_z_dim;            /* gate/value output dim (2048) */
    int ssm_num_k_heads;      /* number of K heads (16) */
    int ssm_num_v_heads;      /* number of V heads (16) */
    int ssm_head_dim;         /* head dimension for SSM (128) */
    int ssm_conv_kernel;      /* conv1d kernel size (4) */

    /* Dense FFN */
    int ffn_intermediate;     /* FFN hidden dimension (3584) */

    float rms_eps;
    float rope_freq_base;
} sriracha_cfg_t;

/* ------------------------------------------------------------------ */
/* Per-layer VRAM weight cache                                          */
/* ------------------------------------------------------------------ */

typedef struct {
    /* Common to all layers */
    float *attn_norm_w;           /* F32 */
    float *post_attn_norm_w;      /* F32 */
    void  *ffn_gate_w;  uint32_t ffn_gate_type;
    void  *ffn_up_w;    uint32_t ffn_up_type;
    void  *ffn_down_w;  uint32_t ffn_down_type;

    /* SSM layers only (NULL for full-attn layers) */
    void  *ssm_qkv_w;   uint32_t ssm_qkv_type;
    void  *ssm_gate_w;  uint32_t ssm_gate_type;
    void  *ssm_alpha_w; uint32_t ssm_alpha_type;
    void  *ssm_beta_w;  uint32_t ssm_beta_type;
    void  *ssm_out_w;   uint32_t ssm_out_type;

    /* SSM CPU-only weights (pinned RAM pointers — no extra copy) */
    const float *ssm_conv_w;   /* F32 [conv_kernel × qkv_dim] */
    const float *ssm_a;        /* F32 [num_v_heads] */
    const float *ssm_dt;       /* F32 [num_v_heads] */
    const float *ssm_norm_w;   /* F32 [head_dim] */

    /* Full-attention layers only (NULL for SSM layers) */
    void  *attn_q_w;   uint32_t attn_q_type;
    void  *attn_k_w;   uint32_t attn_k_type;
    void  *attn_v_w;   uint32_t attn_v_type;
    void  *attn_out_w; uint32_t attn_out_type;

    /* QK norm (host RAM, shared pointer into model file) */
    const float *h_q_norm_w;   /* F32 [head_dim_qk], NULL if !has_qk_norm */
    const float *h_k_norm_w;   /* F32 [head_dim_qk], NULL if !has_qk_norm */
} sriracha_layer_t;

/* ------------------------------------------------------------------ */
/* Main context                                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    loader_t         *L;
    gguf_ctx_t       *g;
    sriracha_cfg_t    cfg;
    sriracha_layer_t *layers;  /* [num_layers] */

    /* VRAM working buffers */
    half *d_hidden;
    half *d_normed;      /* temp: copy of hidden for norm, also output scratch */
    /* SSM projections */
    half *d_qkv;         /* [ssm_qkv_dim] */
    half *d_z;           /* [ssm_z_dim] — SSM gate */
    half *d_alpha;       /* [ssm_num_v_heads] */
    half *d_beta;        /* [ssm_num_v_heads] */
    /* Full-attn projections */
    half *d_q;           /* [q_proj_dim] */
    half *d_k;           /* [kv_proj_dim] */
    half *d_v;           /* [kv_proj_dim] */
    half *d_attn_out;    /* [max(attn_out_dim, ssm_z_dim)] */
    /* FFN */
    half *d_gate;        /* [ffn_intermediate] */
    half *d_up;          /* [ffn_intermediate] */
    /* Output */
    half *d_logits;      /* [vocab_size] */

    /* Output head (VRAM) */
    float   *d_out_norm_w;
    void    *d_out_w;
    uint32_t out_w_type;

    /* KV cache per full-attn layer (host RAM) */
    int     n_full_attn_layers;
    float **kv_k;     /* [n_full_attn_layers][kv_cache_len * kv_proj_dim] */
    float **kv_v;     /* [n_full_attn_layers][kv_cache_len * kv_proj_dim] */
    int    *kv_len;   /* [n_full_attn_layers] current fill depth */

    /* SSM recurrent state per SSM layer (host RAM) */
    int     n_ssm_layers;
    float **ssm_state;   /* [n_ssm_layers][v_heads * head_dim * head_dim] */
    float **ssm_conv;    /* [n_ssm_layers][conv_kernel * qkv_dim] */

    /* Host temp buffers */
    half  *h_emb_row;       /* [hidden_dim] for embedding dequant */
    float *h_qkv;           /* [ssm_qkv_dim] */
    float *h_z;             /* [ssm_z_dim] */
    float *h_alpha;         /* [ssm_num_v_heads] */
    float *h_beta;          /* [ssm_num_v_heads] */
    float *h_conv_out;      /* [ssm_qkv_dim] */
    float *h_ssm_qrep;      /* [ssm_num_v_heads * ssm_head_dim] */
    float *h_ssm_krep;      /* [ssm_num_v_heads * ssm_head_dim] */
    float *h_head_tmp;      /* [ssm_head_dim] */
    float *h_ssm_sk;        /* [ssm_head_dim] scratch for delta-net */
    float *h_ssm_d;         /* [ssm_head_dim] scratch for delta-net */
    float *h_attn;          /* [max(attn_out_dim, ssm_z_dim)] */
    float *h_q;             /* [q_proj_dim] */
    float *h_k;             /* [kv_proj_dim] */
    float *h_v;             /* [kv_proj_dim] */
    float *h_scores;        /* [kv_cache_len] */
    half  *h_half_tmp;      /* [max(q_proj_dim, ssm_qkv_dim)] for conversions */
    half  *h_logits;        /* [vocab_size] */

    cudaStream_t stream;
    int spec_depth;
    int n_drafted;
    int n_accepted;
} sriracha_t;

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

#ifdef __cplusplus
extern "C" {
#endif

/* Load 0.8B draft model, allocate all state, pre-upload weights.
 * spec_depth: K tokens to draft per round (typically 5).
 * Returns NULL on failure. */
sriracha_t *sriracha_init(const char *draft_path, int spec_depth);

/* Run forward pass for all prompt tokens to seed KV/SSM state. */
void sriracha_prefill(sriracha_t *sr, const int32_t *tokens, int n);

/* Generate spec_depth draft tokens from seed_token at cur_pos.
 * out_tokens[]: receives draft IDs (must hold spec_depth entries).
 * Returns number of tokens drafted. */
int sriracha_draft_from(sriracha_t *sr, int32_t seed_token, int cur_pos,
                         int32_t *out_tokens);

/* Rewind KV/SSM state to position pos (after rejection). */
void sriracha_rewind(sriracha_t *sr, int pos);

/* Free all resources. */
void sriracha_free(sriracha_t *sr);

#ifdef __cplusplus
}
#endif

#endif /* SRIRACHA_H */
