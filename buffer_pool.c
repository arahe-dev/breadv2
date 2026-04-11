/* buffer_pool.c — Persistent layer computation buffer management
 *
 * Pre-allocates all working buffers needed by one_layer_forward()
 * at model initialization time. Eliminates repeated malloc/free during
 * inference, saving 60-80ms per token.
 */

#ifdef _WIN32
#  define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "bread.h"
#include "buffer_pool.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in buffer_pool: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/* Global pool instance */
static bread_buffer_pool_t g_pool = {0};
static int g_pool_initialized = 0;

int bread_buffer_pool_init(const bread_model_config_t *cfg)
{
    if (g_pool_initialized) {
        fprintf(stderr, "buffer_pool already initialized\n");
        return -1;
    }

    int H = cfg->hidden_dim;

    /* Allocate FP16 device buffers */
    CUDA_CHECK(cudaMalloc((void **)&g_pool.d_normed,   (size_t)H * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&g_pool.d_normed2,  (size_t)H * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&g_pool.d_q,        (size_t)cfg->q_proj_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&g_pool.d_k,        (size_t)cfg->kv_proj_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&g_pool.d_v,        (size_t)cfg->kv_proj_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&g_pool.d_attn_out, (size_t)cfg->attn_out_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&g_pool.d_o_out,    (size_t)H * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&g_pool.d_sg,       (size_t)cfg->shared_inter * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&g_pool.d_su,       (size_t)cfg->shared_inter * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&g_pool.d_sh_out,   (size_t)H * sizeof(half)));

    /* Allocate expert buffers (8 sets for parallel dispatch) */
    g_pool.top_k_actual = cfg->top_k;
    for (int k = 0; k < cfg->top_k; k++) {
        CUDA_CHECK(cudaMalloc((void **)&g_pool.d_eg[k], (size_t)cfg->expert_inter * sizeof(half)));
        CUDA_CHECK(cudaMalloc((void **)&g_pool.d_eu[k], (size_t)cfg->expert_inter * sizeof(half)));
        CUDA_CHECK(cudaMalloc((void **)&g_pool.d_eo[k], (size_t)H * sizeof(half)));
        CUDA_CHECK(cudaStreamCreate(&g_pool.expert_streams[k]));
        CUDA_CHECK(cudaEventCreateWithFlags(&g_pool.expert_events[k], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&g_pool.normed2_ready_event, cudaEventDisableTiming));

    CUDA_CHECK(cudaMalloc((void **)&g_pool.d_qkv,      (size_t)cfg->ssm_qkv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&g_pool.d_z,        (size_t)cfg->ssm_z_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&g_pool.d_alpha,    (size_t)cfg->ssm_num_v_heads * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&g_pool.d_beta,     (size_t)cfg->ssm_num_v_heads * sizeof(half)));

    /* Allocate FP32 host buffers */
    g_pool.h_qkv      = (float *)malloc((size_t)cfg->ssm_qkv_dim * sizeof(float));
    g_pool.h_z        = (float *)malloc((size_t)cfg->ssm_z_dim * sizeof(float));
    g_pool.h_alpha    = (float *)malloc((size_t)cfg->ssm_num_v_heads * sizeof(float));
    g_pool.h_beta     = (float *)malloc((size_t)cfg->ssm_num_v_heads * sizeof(float));
    g_pool.h_conv_out = (float *)malloc((size_t)cfg->ssm_qkv_dim * sizeof(float));
    g_pool.h_attn_out = (float *)malloc((size_t)cfg->attn_out_dim * sizeof(float));
    int head_dim_max = cfg->head_dim_qk > cfg->head_dim_v ? cfg->head_dim_qk : cfg->head_dim_v;
    g_pool.h_head_tmp = (float *)malloc((size_t)head_dim_max * sizeof(float));
    g_pool.h_ssm_qrep = (float *)malloc((size_t)cfg->ssm_num_v_heads * cfg->ssm_head_dim * sizeof(float));
    g_pool.h_ssm_krep = (float *)malloc((size_t)cfg->ssm_num_v_heads * cfg->ssm_head_dim * sizeof(float));
    g_pool.h_ssm_sk   = (float *)malloc((size_t)cfg->ssm_state_size * sizeof(float));
    g_pool.h_ssm_d    = (float *)malloc((size_t)cfg->ssm_state_size * sizeof(float));
    g_pool.h_q_full   = (float *)malloc((size_t)cfg->q_proj_dim * sizeof(float));
    g_pool.h_kv_k     = (float *)malloc((size_t)cfg->kv_proj_dim * sizeof(float));
    g_pool.h_kv_v     = (float *)malloc((size_t)cfg->kv_proj_dim * sizeof(float));
    g_pool.h_q_score  = (float *)malloc((size_t)cfg->q_proj_dim * sizeof(float));
    g_pool.h_q_gate   = (float *)malloc((size_t)cfg->q_proj_dim * sizeof(float));
    g_pool.h_scores   = (float *)malloc((size_t)cfg->kv_cache_len * cfg->num_kv_heads * sizeof(float));
    g_pool.h_normed   = (float *)malloc((size_t)H * sizeof(float));
    /* Use pinned memory for h_normed2 to speed up GPU↔Host transfers in CPU expert path */
    CUDA_CHECK(cudaMallocHost((void **)&g_pool.h_normed2, (size_t)H * sizeof(float)));
    g_pool.h_o_cpu    = (float *)malloc((size_t)H * sizeof(float));
    g_pool.h_sg_cpu   = (float *)malloc((size_t)cfg->shared_inter * sizeof(float));
    g_pool.h_su_cpu   = (float *)malloc((size_t)cfg->shared_inter * sizeof(float));
    g_pool.h_sh_cpu   = (float *)malloc((size_t)H * sizeof(float));
    g_pool.h_eg_cpu   = (float *)malloc((size_t)cfg->expert_inter * sizeof(float));
    g_pool.h_eu_cpu   = (float *)malloc((size_t)cfg->expert_inter * sizeof(float));
    g_pool.h_eo_cpu   = (float *)malloc((size_t)H * sizeof(float));

    /* FP16 host buffers */
    g_pool.h_hidden_half = (half *)malloc((size_t)H * sizeof(half));
    int attn_ssm_max = cfg->attn_out_dim > cfg->ssm_z_dim ? cfg->attn_out_dim : cfg->ssm_z_dim;
    g_pool.h_attn_half_buf = (half *)malloc((size_t)attn_ssm_max * sizeof(half));

    /* Integer buffers */
    g_pool.h_expert_indices = (int *)malloc((size_t)cfg->top_k * sizeof(int));
    g_pool.h_expert_weights = (float *)malloc((size_t)cfg->top_k * sizeof(float));
    g_pool.h_logits = (float *)malloc((size_t)cfg->num_experts * sizeof(float));

    g_pool_initialized = 1;
    return 0;
}

const bread_buffer_pool_t *bread_buffer_pool_get(void)
{
    if (!g_pool_initialized) {
        fprintf(stderr, "buffer_pool not initialized\n");
        return NULL;
    }
    return &g_pool;
}

void bread_buffer_pool_free(void)
{
    if (!g_pool_initialized) return;

    /* Free device memory */
    if (g_pool.d_normed)   cudaFree(g_pool.d_normed);
    if (g_pool.d_normed2)  cudaFree(g_pool.d_normed2);
    if (g_pool.d_q)        cudaFree(g_pool.d_q);
    if (g_pool.d_k)        cudaFree(g_pool.d_k);
    if (g_pool.d_v)        cudaFree(g_pool.d_v);
    if (g_pool.d_attn_out) cudaFree(g_pool.d_attn_out);
    if (g_pool.d_o_out)    cudaFree(g_pool.d_o_out);
    if (g_pool.d_sg)       cudaFree(g_pool.d_sg);
    if (g_pool.d_su)       cudaFree(g_pool.d_su);
    if (g_pool.d_sh_out)   cudaFree(g_pool.d_sh_out);

    /* Free expert buffers and streams/events */
    for (int k = 0; k < g_pool.top_k_actual; k++) {
        if (g_pool.d_eg[k]) cudaFree(g_pool.d_eg[k]);
        if (g_pool.d_eu[k]) cudaFree(g_pool.d_eu[k]);
        if (g_pool.d_eo[k]) cudaFree(g_pool.d_eo[k]);
        cudaStreamDestroy(g_pool.expert_streams[k]);
        cudaEventDestroy(g_pool.expert_events[k]);
    }
    if (g_pool.normed2_ready_event) cudaEventDestroy(g_pool.normed2_ready_event);

    if (g_pool.d_qkv)      cudaFree(g_pool.d_qkv);
    if (g_pool.d_z)        cudaFree(g_pool.d_z);
    if (g_pool.d_alpha)    cudaFree(g_pool.d_alpha);
    if (g_pool.d_beta)     cudaFree(g_pool.d_beta);

    /* Free host memory */
    if (g_pool.h_qkv)           free(g_pool.h_qkv);
    if (g_pool.h_z)             free(g_pool.h_z);
    if (g_pool.h_alpha)         free(g_pool.h_alpha);
    if (g_pool.h_beta)          free(g_pool.h_beta);
    if (g_pool.h_conv_out)      free(g_pool.h_conv_out);
    if (g_pool.h_attn_out)      free(g_pool.h_attn_out);
    if (g_pool.h_head_tmp)      free(g_pool.h_head_tmp);
    if (g_pool.h_ssm_qrep)      free(g_pool.h_ssm_qrep);
    if (g_pool.h_ssm_krep)      free(g_pool.h_ssm_krep);
    if (g_pool.h_ssm_sk)        free(g_pool.h_ssm_sk);
    if (g_pool.h_ssm_d)         free(g_pool.h_ssm_d);
    if (g_pool.h_q_full)        free(g_pool.h_q_full);
    if (g_pool.h_kv_k)          free(g_pool.h_kv_k);
    if (g_pool.h_kv_v)          free(g_pool.h_kv_v);
    if (g_pool.h_q_score)       free(g_pool.h_q_score);
    if (g_pool.h_q_gate)        free(g_pool.h_q_gate);
    if (g_pool.h_scores)        free(g_pool.h_scores);
    if (g_pool.h_normed)        free(g_pool.h_normed);
    if (g_pool.h_normed2)       cudaFreeHost(g_pool.h_normed2);  /* pinned memory */
    if (g_pool.h_o_cpu)         free(g_pool.h_o_cpu);
    if (g_pool.h_sg_cpu)        free(g_pool.h_sg_cpu);
    if (g_pool.h_su_cpu)        free(g_pool.h_su_cpu);
    if (g_pool.h_sh_cpu)        free(g_pool.h_sh_cpu);
    if (g_pool.h_eg_cpu)        free(g_pool.h_eg_cpu);
    if (g_pool.h_eu_cpu)        free(g_pool.h_eu_cpu);
    if (g_pool.h_eo_cpu)        free(g_pool.h_eo_cpu);
    if (g_pool.h_hidden_half)   free(g_pool.h_hidden_half);
    if (g_pool.h_attn_half_buf) free(g_pool.h_attn_half_buf);
    if (g_pool.h_expert_indices) free(g_pool.h_expert_indices);
    if (g_pool.h_expert_weights) free(g_pool.h_expert_weights);
    if (g_pool.h_logits)        free(g_pool.h_logits);

    memset(&g_pool, 0, sizeof(g_pool));
    g_pool_initialized = 0;
}
