#ifndef LAYER_OPS_H
#define LAYER_OPS_H

#include <stdint.h>
#include <cuda_fp16.h>
#include "bread.h"
#include "gguf.h"
#include "loader.h"
#include "bread_utils.h"

/* ================================================================== */
/* CPU Math & Quantization Primitives                                */
/* ================================================================== */

/* Dequantization blocks */
void cpu_dequant_q4k_block(const uint8_t *block, float *out);
void cpu_dequant_q6k_block(const uint8_t *block, float *out);

/* CPU matvec dispatch (host-side matrix-vector products) */
void cpu_matvec_f32(const uint8_t *w, const float *x, float *y, int rows, int cols);
void cpu_matvec_f16(const uint8_t *w, const float *x, float *y, int rows, int cols);
void cpu_matvec_q4k(const uint8_t *w, const float *x, float *y, int rows, int cols);
void cpu_matvec_q6k(const uint8_t *w, const float *x, float *y, int rows, int cols);
void cpu_tensor_matvec(const uint8_t *w, uint32_t type,
                       const float *x, float *y, int rows, int cols);
void cpu_named_matvec(const loader_t *L, const gguf_ctx_t *g,
                      const char *name, const float *x, float *y,
                      int rows, int cols);

/* ================================================================== */
/* CPU Math Primitives                                               */
/* ================================================================== */

/* Normalization */
void cpu_rms_norm_weighted(const float *x, const float *w,
                          float *out, int n, float eps);
void cpu_rms_norm_bare(float *x, int n, float eps);
void cpu_l2_norm_bare(float *x, int n, float eps);
void cpu_gated_rms_norm(const float *x, const float *z,
                       const float *w, float *out, int n, float eps);

/* Activation & gating */
void cpu_swiglu(const float *gate, const float *up, float *out, int n);
void cpu_silu_inplace(float *x, int n);
void cpu_softmax(float *x, int n);
void cpu_topk(const float *probs, int n, int K, int *indices, float *weights);

/* Scalar functions */
float cpu_sigmoid(float x);
float cpu_softplus(float x);

/* Convolution & state ops */
void cpu_conv1d_step(const float *conv_state, const float *new_input,
                     const float *weight, float *out,
                     int channels, int kernel_size);
void cpu_repeat_heads(float *dst, const float *src,
                     int src_heads, int dst_heads, int head_dim);
int ssm_k_head_for_v_head(int vh, int num_k_heads, int num_v_heads);

/* GatedDeltaNet (SSM) core */
void cpu_delta_net_autoregressive_step(const float *q, const float *k,
                                       const float *v, float gate, float beta,
                                       float *state, float *out,
                                       int value_dim, int key_dim,
                                       float *sk_buf, float *d_buf);

/* ================================================================== */
/* Attention & Positional Encoding                                   */
/* ================================================================== */

void apply_rotary_emb(const bread_model_config_t *cfg,
                     float *q, float *k, int pos);
void split_full_attn_q(const bread_model_config_t *cfg,
                      const float *q_full, float *q_score, float *q_gate);
void apply_per_head_rms_norm(float *x, const float *w,
                            int num_heads, int head_dim, float eps);
int rope_select_stream_for_pair(const bread_model_config_t *cfg,
                               int pair_idx, int total_pairs);

/* ================================================================== */
/* Debug/Trace Utilities                                             */
/* ================================================================== */

void trace_vec_stats(const char *label, const float *x, int n, int max_show);
void trace_sigmoid_stats(const char *label, const float *x, int n, int max_show);

#endif /* LAYER_OPS_H */
