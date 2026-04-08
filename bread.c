#ifdef _WIN32
#  define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "bread.h"

#ifdef _WIN32
#  define fseek64(f,o,w) _fseeki64((f),(__int64)(o),(w))
#else
#  define fseek64(f,o,w) fseeko((f),(off_t)(o),(w))
#endif

static bread_model_config_t g_cfg;
static int g_cfg_ready = 0;
static int g_boring_mode = 0;
static int g_force_ssm_zero = 0;
static int g_disable_rope = 0;
static int g_trace_debug = 0;
static int g_trace_pos = -1;
static int g_prefetch_mode = 0;  /* Layer prefetch optimization (off by default) */

static void xread(FILE *fp, void *buf, size_t n) {
    if (fread(buf, 1, n, fp) != n) {
        fprintf(stderr, "bread_config: read error\n");
        exit(1);
    }
}

static uint8_t  r_u8 (FILE *fp) { uint8_t  v; xread(fp, &v, 1); return v; }
static uint16_t r_u16(FILE *fp) { uint16_t v; xread(fp, &v, 2); return v; }
static uint32_t r_u32(FILE *fp) { uint32_t v; xread(fp, &v, 4); return v; }
static uint64_t r_u64(FILE *fp) { uint64_t v; xread(fp, &v, 8); return v; }
static float    r_f32(FILE *fp) { float    v; xread(fp, &v, 4); return v; }

static char *read_str(FILE *fp) {
    uint64_t len = r_u64(fp);
    char *s = (char *)malloc((size_t)len + 1);
    xread(fp, s, (size_t)len);
    s[len] = '\0';
    return s;
}

static void skip_bytes(FILE *fp, uint64_t n) {
    fseek64(fp, (int64_t)n, SEEK_CUR);
}

static void skip_str(FILE *fp) {
    skip_bytes(fp, r_u64(fp));
}

static void skip_val(FILE *fp, uint32_t vt);

static void read_u32_array(FILE *fp, int *dst, int max_count, int *out_count) {
    uint32_t et = r_u32(fp);
    uint64_t cnt = r_u64(fp);
    int n = 0;

    if (et == 4) {
        for (uint64_t i = 0; i < cnt; i++) {
            uint32_t v = r_u32(fp);
            if (n < max_count) dst[n++] = (int)v;
        }
    } else {
        for (uint64_t i = 0; i < cnt; i++) skip_val(fp, et);
    }

    if (out_count) *out_count = n;
}

static void skip_val(FILE *fp, uint32_t vt) {
    switch (vt) {
        case 0: case 1: case 7: r_u8(fp);  return;
        case 2: case 3:         r_u16(fp); return;
        case 4: case 5: case 6: r_u32(fp); return;
        case 10: case 11: case 12: r_u64(fp); return;
        case 8: skip_str(fp); return;
        case 9: {
            uint32_t et = r_u32(fp);
            uint64_t cnt = r_u64(fp);
            size_t esz = 0;
            switch (et) {
                case 0: case 1: case 7: esz = 1; break;
                case 2: case 3:         esz = 2; break;
                case 4: case 5: case 6: esz = 4; break;
                case 10: case 11: case 12: esz = 8; break;
            }
            if (esz > 0) skip_bytes(fp, cnt * esz);
            else for (uint64_t i = 0; i < cnt; i++) skip_val(fp, et);
            return;
        }
    }
}

static void bread_cfg_defaults(bread_model_config_t *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->hidden_dim = BREAD_HIDDEN_DIM;
    cfg->num_layers = BREAD_NUM_LAYERS;
    cfg->vocab_size = BREAD_VOCAB_SIZE;
    cfg->num_q_heads = BREAD_NUM_Q_HEADS;
    cfg->num_kv_heads = BREAD_NUM_KV_HEADS;
    cfg->head_dim_qk = BREAD_HEAD_DIM_QK;
    cfg->head_dim_v = BREAD_HEAD_DIM_V;
    cfg->head_dim_qgate = BREAD_HEAD_DIM_QGATE;
    cfg->q_proj_dim = BREAD_Q_PROJ_DIM;
    cfg->kv_proj_dim = BREAD_KV_PROJ_DIM;
    cfg->attn_out_dim = BREAD_ATTN_OUT_DIM;
    cfg->head_dim_rope = BREAD_HEAD_DIM_ROPE;
    cfg->kv_cache_len = BREAD_KV_CACHE_LEN;
    cfg->full_attention_interval = BREAD_FULL_ATTN_INTERVAL;
    memset(cfg->rope_sections, 0, sizeof(cfg->rope_sections));
    cfg->ssm_num_k_heads = BREAD_SSM_NUM_K_HEADS;
    cfg->ssm_num_v_heads = BREAD_SSM_NUM_V_HEADS;
    cfg->ssm_head_dim = BREAD_SSM_HEAD_DIM;
    cfg->ssm_qkv_dim = BREAD_SSM_QKV_DIM;
    cfg->ssm_z_dim = BREAD_SSM_Z_DIM;
    cfg->ssm_conv_kernel = BREAD_SSM_CONV_KERNEL;
    cfg->ssm_inner_size = BREAD_SSM_Z_DIM;
    cfg->ssm_state_size = BREAD_SSM_HEAD_DIM;
    cfg->ssm_time_step_rank = BREAD_SSM_NUM_V_HEADS;
    cfg->ssm_group_count = BREAD_SSM_NUM_K_HEADS;
    cfg->ssm_v_head_reordered = 0;
    cfg->rope_mrope_interleaved = 0;
    cfg->expert_inter = BREAD_EXPERT_INTER;
    cfg->shared_inter = BREAD_SHARED_INTER;
    cfg->num_experts = BREAD_NUM_EXPERTS;
    cfg->top_k = BREAD_TOP_K;
    cfg->rms_eps = BREAD_RMS_EPS;
    cfg->rope_freq_base = BREAD_ROPE_FREQ_BASE;
}

static void bread_cfg_from_metadata(const char *model_path, bread_model_config_t *cfg) {
    FILE *fp = fopen(model_path, "rb");
    if (!fp) return;

    if (r_u32(fp) != 0x46554747u) { fclose(fp); return; }
    (void)r_u32(fp);
    (void)r_u64(fp);
    uint64_t n_kv = r_u64(fp);

    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = read_str(fp);
        uint32_t vt = r_u32(fp);

        if (!strcmp(key, "qwen35moe.block_count") && vt == 4) {
            cfg->num_layers = (int)r_u32(fp);
        } else if (!strcmp(key, "qwen35moe.embedding_length") && vt == 4) {
            cfg->hidden_dim = (int)r_u32(fp);
        } else if (!strcmp(key, "qwen35moe.attention.head_count") && vt == 4) {
            cfg->num_q_heads = (int)r_u32(fp);
        } else if (!strcmp(key, "qwen35moe.attention.head_count_kv") && vt == 9) {
            int kv_arr[BREAD_NUM_LAYERS];
            int n_kv = 0;
            read_u32_array(fp, kv_arr, BREAD_NUM_LAYERS, &n_kv);
            if (n_kv > 0) cfg->num_kv_heads = kv_arr[0];
        } else if (!strcmp(key, "qwen35moe.attention.key_length") && vt == 4) {
            cfg->head_dim_qk = (int)r_u32(fp);
        } else if (!strcmp(key, "qwen35moe.attention.value_length") && vt == 4) {
            cfg->head_dim_v = (int)r_u32(fp);
        } else if (!strcmp(key, "qwen35moe.full_attention_interval") && vt == 4) {
            cfg->full_attention_interval = (int)r_u32(fp);
        } else if (!strcmp(key, "qwen35moe.expert_feed_forward_length") && vt == 4) {
            cfg->expert_inter = (int)r_u32(fp);
        } else if (!strcmp(key, "qwen35moe.expert_shared_feed_forward_length") && vt == 4) {
            cfg->shared_inter = (int)r_u32(fp);
        } else if (!strcmp(key, "qwen35moe.expert_count") && vt == 4) {
            cfg->num_experts = (int)r_u32(fp);
        } else if (!strcmp(key, "qwen35moe.expert_used_count") && vt == 4) {
            cfg->top_k = (int)r_u32(fp);
        } else if (!strcmp(key, "qwen35moe.rope.dimension_count") && vt == 4) {
            cfg->head_dim_rope = (int)r_u32(fp);
        } else if (!strcmp(key, "qwen35moe.rope.dimension_sections") && vt == 9) {
            int n_sec = 0;
            read_u32_array(fp, cfg->rope_sections, 4, &n_sec);
            (void)n_sec;
        } else if (!strcmp(key, "qwen35moe.attention.layer_norm_rms_epsilon") && vt == 6) {
            cfg->rms_eps = r_f32(fp);
        } else if (!strcmp(key, "qwen35moe.rope.freq_base") && vt == 6) {
            cfg->rope_freq_base = r_f32(fp);
        } else if (!strcmp(key, "qwen35moe.ssm.conv_kernel") && vt == 4) {
            cfg->ssm_conv_kernel = (int)r_u32(fp);
        } else if (!strcmp(key, "qwen35moe.ssm.inner_size") && vt == 4) {
            cfg->ssm_inner_size = (int)r_u32(fp);
        } else if (!strcmp(key, "qwen35moe.ssm.state_size") && vt == 4) {
            cfg->ssm_state_size = (int)r_u32(fp);
        } else if (!strcmp(key, "qwen35moe.ssm.time_step_rank") && vt == 4) {
            cfg->ssm_time_step_rank = (int)r_u32(fp);
        } else if (!strcmp(key, "qwen35moe.ssm.group_count") && vt == 4) {
            cfg->ssm_group_count = (int)r_u32(fp);
        } else if (!strcmp(key, "qwen35moe.ssm.v_head_reordered") && vt == 7) {
            cfg->ssm_v_head_reordered = (int)r_u8(fp);
        } else if (!strcmp(key, "qwen35moe.rope.mrope_interleaved") && vt == 7) {
            cfg->rope_mrope_interleaved = (int)r_u8(fp);
        } else {
            skip_val(fp, vt);
        }
        free(key);
    }

    fclose(fp);
}

int bread_model_config_init(const char *model_path, const gguf_ctx_t *g)
{
    const gguf_tensor_t *t;
    int layer;

    bread_cfg_defaults(&g_cfg);
    bread_cfg_from_metadata(model_path, &g_cfg);

    t = gguf_find_tensor(g, "token_embd.weight");
    if (t && t->n_dims >= 2) {
        g_cfg.hidden_dim = (int)t->dims[0];
        g_cfg.vocab_size = (int)t->dims[1];
    }

    for (layer = 0; ; layer++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", layer);
        if (!gguf_find_tensor(g, nm)) break;
    }
    if (layer > 0) g_cfg.num_layers = layer;

    if (g_cfg.ssm_state_size > 0) g_cfg.ssm_head_dim = g_cfg.ssm_state_size;
    if (g_cfg.ssm_group_count > 0) g_cfg.ssm_num_k_heads = g_cfg.ssm_group_count;
    if (g_cfg.ssm_time_step_rank > 0) g_cfg.ssm_num_v_heads = g_cfg.ssm_time_step_rank;
    if (g_cfg.ssm_inner_size > 0) g_cfg.ssm_z_dim = g_cfg.ssm_inner_size;
    if (g_cfg.ssm_num_k_heads > 0 && g_cfg.ssm_head_dim > 0 && g_cfg.ssm_z_dim > 0) {
        g_cfg.ssm_qkv_dim = g_cfg.ssm_z_dim + 2 * g_cfg.ssm_num_k_heads * g_cfg.ssm_head_dim;
    }

    for (layer = 0; layer < g_cfg.num_layers; layer++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", layer);
        t = gguf_find_tensor(g, nm);
        if (t) {
            const gguf_tensor_t *t_k, *t_v, *t_o, *t_qn, *t_kn;
            snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", layer);
            t_k = gguf_find_tensor(g, nm);
            snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", layer);
            t_v = gguf_find_tensor(g, nm);
            snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", layer);
            t_o = gguf_find_tensor(g, nm);
            snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", layer);
            t_qn = gguf_find_tensor(g, nm);
            snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", layer);
            t_kn = gguf_find_tensor(g, nm);
            if (t->n_dims >= 2 && t_k && t_v && t_o && t_qn && t_kn) {
                const int q_proj_dim = (int)t->dims[1];
                const int kv_proj_dim = (int)t_k->dims[1];
                const int attn_out_dim = (int)t_o->dims[0];
                const int head_dim_qk = (int)t_qn->dims[0];
                const int num_kv_heads = kv_proj_dim / head_dim_qk;
                const int head_dim_v = kv_proj_dim / num_kv_heads;
                const int num_q_heads = attn_out_dim / head_dim_v;

                g_cfg.q_proj_dim = q_proj_dim;
                g_cfg.kv_proj_dim = kv_proj_dim;
                g_cfg.attn_out_dim = attn_out_dim;
                g_cfg.head_dim_qk = head_dim_qk;
                g_cfg.num_kv_heads = num_kv_heads;
                g_cfg.head_dim_v = head_dim_v;
                g_cfg.num_q_heads = num_q_heads;
                g_cfg.head_dim_qgate = q_proj_dim / num_q_heads - head_dim_qk;
                break;
            }
        }
    }

    for (layer = 0; layer < g_cfg.num_layers; layer++) {
        char nm[64];
        const gguf_tensor_t *t_alpha, *t_gate, *t_conv, *t_norm;
        snprintf(nm, sizeof(nm), "blk.%d.ssm_alpha.weight", layer);
        t_alpha = gguf_find_tensor(g, nm);
        if (!t_alpha) continue;
        snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight", layer);
        t_gate = gguf_find_tensor(g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight", layer);
        t = gguf_find_tensor(g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_conv1d.weight", layer);
        t_conv = gguf_find_tensor(g, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_norm.weight", layer);
        t_norm = gguf_find_tensor(g, nm);
        if (t && t_gate && t_conv && t_norm) {
            if (g_cfg.ssm_qkv_dim <= 0) g_cfg.ssm_qkv_dim = (int)t->dims[1];
            if (g_cfg.ssm_z_dim <= 0) g_cfg.ssm_z_dim = (int)t_gate->dims[1];
            if (g_cfg.ssm_num_v_heads <= 0) g_cfg.ssm_num_v_heads = (int)t_alpha->dims[1];
            if (g_cfg.ssm_head_dim <= 0) g_cfg.ssm_head_dim = (int)t_norm->dims[0];
            if (g_cfg.ssm_num_k_heads <= 0) {
                g_cfg.ssm_num_k_heads =
                    (g_cfg.ssm_qkv_dim - g_cfg.ssm_z_dim) / (2 * g_cfg.ssm_head_dim);
            }
            if (g_cfg.ssm_conv_kernel <= 0) g_cfg.ssm_conv_kernel = (int)t_conv->dims[0];
            break;
        }
    }

    if (g_cfg.full_attention_interval <= 0) g_cfg.full_attention_interval = BREAD_FULL_ATTN_INTERVAL;
    if (g_cfg.kv_cache_len <= 0) g_cfg.kv_cache_len = BREAD_KV_CACHE_LEN;
    if (g_cfg.top_k <= 0) g_cfg.top_k = BREAD_TOP_K;

    g_cfg_ready = 1;
    return 0;
}

const bread_model_config_t *bread_model_config_get(void)
{
    return g_cfg_ready ? &g_cfg : NULL;
}

int bread_layer_is_recurrent(int layer_idx)
{
    const bread_model_config_t *cfg = bread_model_config_get();
    int interval = cfg ? cfg->full_attention_interval : BREAD_FULL_ATTN_INTERVAL;
    return ((layer_idx + 1) % interval) != 0;
}

int bread_layer_is_full_attention(int layer_idx)
{
    return !bread_layer_is_recurrent(layer_idx);
}

void bread_set_boring_mode(int enabled)
{
    g_boring_mode = enabled ? 1 : 0;
}

int bread_get_boring_mode(void)
{
    return g_boring_mode;
}

void bread_set_force_ssm_zero(int enabled)
{
    g_force_ssm_zero = enabled ? 1 : 0;
}

int bread_get_force_ssm_zero(void)
{
    return g_force_ssm_zero;
}

void bread_set_disable_rope(int enabled)
{
    g_disable_rope = enabled ? 1 : 0;
}

int bread_get_disable_rope(void)
{
    return g_disable_rope;
}

void bread_set_trace_debug(int enabled)
{
    g_trace_debug = enabled ? 1 : 0;
}

int bread_get_trace_debug(void)
{
    return g_trace_debug;
}

void bread_set_trace_pos(int pos)
{
    g_trace_pos = pos;
}

int bread_get_trace_pos(void)
{
    return g_trace_pos;
}

void bread_set_prefetch_mode(int enabled)
{
    g_prefetch_mode = enabled ? 1 : 0;
}

int bread_get_prefetch_mode(void)
{
    return g_prefetch_mode;
}

/* Initialize default progress callback */
void bread_init_progress(void)
{
    bread_progress_init_default();
}
