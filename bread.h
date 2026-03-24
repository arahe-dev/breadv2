#ifndef BREAD_H
#define BREAD_H

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
#define BREAD_Q_PROJ_DIM      8192     /* Q projection output: 16 × 512        */
#define BREAD_KV_PROJ_DIM     512      /* K or V projection output: 2 × 256    */
#define BREAD_ATTN_OUT_DIM    4096     /* o_proj input: 16 × 256               */
#define BREAD_HEAD_DIM_ROPE   128      /* rotary dims per head (partial RoPE)  */

/* True for full-attention layers, false for SSM/GatedDeltaNet        */
#define BREAD_IS_FULL_ATTN(layer)  (((layer) % 4) == 3)

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

#endif /* BREAD_H */
