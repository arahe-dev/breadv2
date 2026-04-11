/* buffer_pool.h — Pre-allocated layer computation buffers
 *
 * Manages persistent VRAM buffers for one_layer_forward().
 * Eliminates repeated cudaMalloc/cudaFree overhead (60-80ms per token).
 *
 * Usage:
 *   bread_buffer_pool_init(cfg);  // Call once after model config
 *   const bread_buffer_pool_t *pool = bread_buffer_pool_get();
 *   // Use pool->d_normed, pool->d_q, etc. in layer forward
 *   bread_buffer_pool_free();      // Call on shutdown
 */

#ifndef BUFFER_POOL_H
#define BUFFER_POOL_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>

/* Include bread.h for bread_model_config_t definition */
#include "bread.h"

#define BREAD_MAX_TOP_K 8  /* max concurrent experts (top_k = 8) */

/* Pool of persistent layer computation buffers */
typedef struct {
    /* FP16 device buffers */
    half *d_normed;      /* H — normalized state */
    half *d_normed2;     /* H — normalized state 2 */
    half *d_q;           /* q_proj_dim */
    half *d_k;           /* kv_proj_dim */
    half *d_v;           /* kv_proj_dim */
    half *d_attn_out;    /* attn_out_dim */
    half *d_o_out;       /* H */
    half *d_sg;          /* shared_inter — shared expert gate */
    half *d_su;          /* shared_inter — shared expert up */
    half *d_sh_out;      /* H — shared expert output */
    half *d_eg[BREAD_MAX_TOP_K];    /* expert_inter — expert gate (per-stream) */
    half *d_eu[BREAD_MAX_TOP_K];    /* expert_inter — expert up (per-stream) */
    half *d_eo[BREAD_MAX_TOP_K];    /* H — expert output (per-stream) */
    half *d_qkv;         /* ssm_qkv_dim */
    half *d_z;           /* ssm_z_dim */
    half *d_alpha;       /* ssm_num_v_heads */
    half *d_beta;        /* ssm_num_v_heads */

    /* FP32 host buffers (CPU-side calculations) */
    float *h_qkv;
    float *h_z;
    float *h_alpha;
    float *h_beta;
    float *h_conv_out;
    float *h_attn_out;
    float *h_head_tmp;
    float *h_ssm_qrep;
    float *h_ssm_krep;
    float *h_ssm_sk;
    float *h_ssm_d;
    float *h_q_full;
    float *h_kv_k;
    float *h_kv_v;
    float *h_q_score;
    float *h_q_gate;
    float *h_scores;
    float *h_normed;
    float *h_normed2;
    float *h_o_cpu;
    float *h_sg_cpu;
    float *h_su_cpu;
    float *h_sh_cpu;
    float *h_eg_cpu;
    float *h_eu_cpu;
    float *h_eo_cpu;

    /* FP16 host buffers */
    half *h_hidden_half;
    half *h_attn_half_buf;

    /* Integer buffers */
    int *h_expert_indices;
    float *h_expert_weights;
    float *h_logits;

    /* CUDA streams and events for expert parallelism */
    cudaStream_t expert_streams[BREAD_MAX_TOP_K];   /* one stream per concurrent expert */
    cudaEvent_t  expert_events[BREAD_MAX_TOP_K];    /* fan-in synchronization (no timing) */
    cudaEvent_t  normed2_ready_event;               /* signals d_normed2 is ready */

    /* Runtime config */
    int top_k_actual;  /* actual top_k at init time (for freeing correct count) */

} bread_buffer_pool_t;

/* Initialize buffer pool with sizes from model config */
int bread_buffer_pool_init(const bread_model_config_t *cfg);

/* Get the pool instance (read-only) */
const bread_buffer_pool_t *bread_buffer_pool_get(void);

/* Free all buffers and release pool */
void bread_buffer_pool_free(void);

#endif
