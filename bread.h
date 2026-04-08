#ifndef BREAD_H
#define BREAD_H

#include "gguf.h"
#include "kernel_tasks.h"
#include "error_classification.h"
#include "progress_tracking.h"
#include "hooks.h"

/* bread.h — model constants for Qwen3.5-35B-A3B (GGUF verified)
 *
 * All values derived directly from GGUF metadata (config_reader.exe)
 * and confirmed against tensor shapes in bread_info.exe output.
 */

/* ------------------------------------------------------------------ */
/* Model architecture                                                   */
/* ------------------------------------------------------------------ */

#define BREAD_HIDDEN_DIM      2048     /* embedding / hidden size              */
#define BREAD_NUM_LAYERS      40       /* transformer blocks blk.0 .. blk.39   */
#define BREAD_VOCAB_SIZE      248320   /* token vocabulary                     */

/* ------------------------------------------------------------------ */
/* Attention                                                            */
/*                                                                      */
/* Full-attention layers: 3, 7, 11, … (every 4th, i.e. layer%4==3)   */
/* GatedDeltaNet/SSM layers: all others.                               */
/*                                                                      */
/* Q is projected to Q_PROJ_DIM = 8192 (16 heads × 512 per head).     */
/* Attention SCORING uses the first HEAD_DIM_QK=256 dims per Q head.  */
/* Value output per Q head = HEAD_DIM_V = 256.                        */
/* o_proj input = NUM_Q_HEADS × HEAD_DIM_V = 4096.                    */
/* ------------------------------------------------------------------ */

#define BREAD_NUM_Q_HEADS     16       /* query attention heads                */
#define BREAD_NUM_KV_HEADS    2        /* KV heads for full-attention layers   */
#define BREAD_HEAD_DIM_QK     256      /* key/query head dim (key_length)      */
#define BREAD_HEAD_DIM_V      256      /* value head dim  (value_length)       */
#define BREAD_HEAD_DIM_QGATE  256      /* per-head sigmoid gate dim            */
#define BREAD_Q_PROJ_DIM      8192     /* Q projection output: 16 × 512        */
#define BREAD_KV_PROJ_DIM     512      /* K or V projection output: 2 × 256    */
#define BREAD_ATTN_OUT_DIM    4096     /* o_proj input: 16 × 256               */
#define BREAD_HEAD_DIM_ROPE   64       /* rotary dims per head (Qwen35MoE)     */
#define BREAD_KV_CACHE_LEN    8192     /* host KV cache capacity per FA layer  */
#define BREAD_FULL_ATTN_INTERVAL 4     /* every 4th layer is full attention    */

/* ------------------------------------------------------------------ */
/* GatedDeltaNet / SSM                                                 */
/* ------------------------------------------------------------------ */

#define BREAD_SSM_NUM_K_HEADS 16
#define BREAD_SSM_NUM_V_HEADS 32
#define BREAD_SSM_HEAD_DIM    128
#define BREAD_SSM_QKV_DIM     8192
#define BREAD_SSM_Z_DIM       4096
#define BREAD_SSM_CONV_KERNEL 4

/* ------------------------------------------------------------------ */
/* MoE FFN                                                              */
/* ------------------------------------------------------------------ */

#define BREAD_EXPERT_INTER    512      /* expert intermediate dim              */
#define BREAD_SHARED_INTER    512      /* shared-expert intermediate dim       */
#define BREAD_NUM_EXPERTS     256      /* routed experts per layer             */
#define BREAD_TOP_K           8        /* active experts per token (GGUF: 8)  */

/* ------------------------------------------------------------------ */
/* Numerics                                                             */
/* ------------------------------------------------------------------ */

#define BREAD_RMS_EPS         1e-6f    /* RMSNorm epsilon                      */
#define BREAD_ROPE_FREQ_BASE  1e7f     /* RoPE theta                           */

/* ------------------------------------------------------------------ */
/* Default model path                                                   */
/* ------------------------------------------------------------------ */

#define BREAD_MODEL_PATH \
    "C:\\Users\\arahe\\.ollama\\models\\blobs\\" \
    "sha256-900dde62fb7ebe8a5a25e35d5b7633f403f226a310965fed51d50f5238ba145a"

typedef struct bread_model_config {
    int hidden_dim;
    int num_layers;
    int vocab_size;

    int num_q_heads;
    int num_kv_heads;
    int head_dim_qk;
    int head_dim_v;
    int head_dim_qgate;
    int q_proj_dim;
    int kv_proj_dim;
    int attn_out_dim;
    int head_dim_rope;
    int kv_cache_len;
    int full_attention_interval;
    int rope_sections[4];

    int ssm_num_k_heads;
    int ssm_num_v_heads;
    int ssm_head_dim;
    int ssm_qkv_dim;
    int ssm_z_dim;
    int ssm_conv_kernel;
    int ssm_inner_size;
    int ssm_state_size;
    int ssm_time_step_rank;
    int ssm_group_count;
    int ssm_v_head_reordered;
    int rope_mrope_interleaved;

    int expert_inter;
    int shared_inter;
    int num_experts;
    int top_k;

    float rms_eps;
    float rope_freq_base;
} bread_model_config_t;

#include "buffer_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

int bread_model_config_init(const char *model_path, const gguf_ctx_t *g);
const bread_model_config_t *bread_model_config_get(void);
int bread_layer_is_recurrent(int layer_idx);
int bread_layer_is_full_attention(int layer_idx);
void bread_set_boring_mode(int enabled);
int bread_get_boring_mode(void);
void bread_set_force_ssm_zero(int enabled);
int bread_get_force_ssm_zero(void);
void bread_set_disable_rope(int enabled);
int bread_get_disable_rope(void);
void bread_set_trace_debug(int enabled);
int bread_get_trace_debug(void);
void bread_set_trace_pos(int pos);
int bread_get_trace_pos(void);
void bread_set_prefetch_mode(int enabled);
int bread_get_prefetch_mode(void);
void bread_init_progress(void);

#ifdef __cplusplus
}
#endif

#endif /* BREAD_H */
