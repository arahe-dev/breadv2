/* layer_ops.cu — CPU math primitives for BREAD layer computation
 *
 * Extracted from one_layer.cu: quantization, matvec, normalization,
 * activation, attention, and SSM operations.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_fp16.h>
#include "bread_utils.h"
#include "bread.h"
#include "gguf.h"
#include "loader.h"

#define CPU_QK_BLOCK_ELEMS  256
#define CPU_Q4K_BLOCK_BYTES 144
#define CPU_Q6K_BLOCK_BYTES 210

/* ================================================================== */
/* Weight loading helpers (used by cpu_named_matvec)                 */
/* ================================================================== */

extern const gguf_tensor_t *require_tensor(const gguf_ctx_t *g, const char *name);
extern uint8_t *tensor_ram(const loader_t *L, const gguf_ctx_t *g, const char *name);

/* ================================================================== */
/* Dequantization blocks                                             */
/* ================================================================== */

static void get_scale_min_k4_host(int j, const uint8_t *scales,
                                  uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = scales[j]     & 63;
        *m = scales[j + 4] & 63;
    } else {
        *d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m = (scales[j + 4] >>   4) | ((scales[j    ] >> 6) << 4);
    }
}

void cpu_dequant_q4k_block(const uint8_t *block, float *out)
{
    uint16_t d_raw, dmin_raw;
    const uint8_t *scales = block + 4;
    const uint8_t *qs     = block + 16;
    memcpy(&d_raw,    block + 0, 2);
    memcpy(&dmin_raw, block + 2, 2);
    {
        const float d    = bread_h2f(d_raw);
        const float dmin = bread_h2f(dmin_raw);
        int is = 0;
        float *y = out;
        const uint8_t *q = qs;
        for (int j = 0; j < CPU_QK_BLOCK_ELEMS; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4_host(is + 0, scales, &sc, &m);
            {
                const float d1 = d * sc;
                const float m1 = dmin * m;
                get_scale_min_k4_host(is + 1, scales, &sc, &m);
                {
                    const float d2 = d * sc;
                    const float m2 = dmin * m;
                    for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
                    for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
                }
            }
            q  += 32;
            is += 2;
        }
    }
}

void cpu_dequant_q6k_block(const uint8_t *block, float *out)
{
    uint16_t d_raw;
    const uint8_t *ql = block + 0;
    const uint8_t *qh = block + 128;
    const int8_t  *sc = (const int8_t *)(block + 192);
    memcpy(&d_raw, block + 208, 2);
    {
        const float d = bread_h2f(d_raw);
        float *y = out;
        for (int n = 0; n < CPU_QK_BLOCK_ELEMS; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* ================================================================== */
/* CPU matvec primitives                                             */
/* ================================================================== */

void cpu_matvec_f32(const uint8_t *w, const float *x, float *y, int rows, int cols)
{
    const float *wf = (const float *)w;
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const float *row = wf + (size_t)r * cols;
        for (int c = 0; c < cols; c++) sum += row[c] * x[c];
        y[r] = sum;
    }
}

void cpu_matvec_f16(const uint8_t *w, const float *x, float *y, int rows, int cols)
{
    const half *wh = (const half *)w;
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const half *row = wh + (size_t)r * cols;
        for (int c = 0; c < cols; c++) sum += __half2float(row[c]) * x[c];
        y[r] = sum;
    }
}

void cpu_matvec_q4k(const uint8_t *w, const float *x, float *y, int rows, int cols)
{
    const int row_bytes = (int)ggml_tensor_nbytes((uint64_t)cols, GGML_TYPE_Q4_K);
    const int n_blocks = cols / CPU_QK_BLOCK_ELEMS;
    float block[CPU_QK_BLOCK_ELEMS];
    for (int r = 0; r < rows; r++) {
        const uint8_t *row = w + (size_t)r * row_bytes;
        float sum = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            cpu_dequant_q4k_block(row + (size_t)b * CPU_Q4K_BLOCK_BYTES, block);
            for (int i = 0; i < CPU_QK_BLOCK_ELEMS; i++) {
                sum += block[i] * x[b * CPU_QK_BLOCK_ELEMS + i];
            }
        }
        y[r] = sum;
    }
}

void cpu_matvec_q6k(const uint8_t *w, const float *x, float *y, int rows, int cols)
{
    const int row_bytes = (int)ggml_tensor_nbytes((uint64_t)cols, GGML_TYPE_Q6_K);
    const int n_blocks = cols / CPU_QK_BLOCK_ELEMS;
    float block[CPU_QK_BLOCK_ELEMS];
    for (int r = 0; r < rows; r++) {
        const uint8_t *row = w + (size_t)r * row_bytes;
        float sum = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            cpu_dequant_q6k_block(row + (size_t)b * CPU_Q6K_BLOCK_BYTES, block);
            for (int i = 0; i < CPU_QK_BLOCK_ELEMS; i++) {
                sum += block[i] * x[b * CPU_QK_BLOCK_ELEMS + i];
            }
        }
        y[r] = sum;
    }
}

void cpu_tensor_matvec(const uint8_t *w, uint32_t type,
                       const float *x, float *y, int rows, int cols)
{
    switch (type) {
        case GGML_TYPE_F32:
            cpu_matvec_f32(w, x, y, rows, cols);
            break;
        case GGML_TYPE_F16:
            cpu_matvec_f16(w, x, y, rows, cols);
            break;
        case GGML_TYPE_Q4_K:
            cpu_matvec_q4k(w, x, y, rows, cols);
            break;
        case GGML_TYPE_Q6_K:
            cpu_matvec_q6k(w, x, y, rows, cols);
            break;
        default:
            fprintf(stderr, "cpu_tensor_matvec: unsupported type %s\n", ggml_type_name(type));
            exit(1);
    }
}

void cpu_named_matvec(const loader_t *L, const gguf_ctx_t *g,
                      const char *name, const float *x, float *y,
                      int rows, int cols)
{
    const gguf_tensor_t *t = require_tensor(g, name);
    cpu_tensor_matvec(tensor_ram(L, g, name), t->type, x, y, rows, cols);
}

/* ================================================================== */
/* CPU Math Primitives                                               */
/* ================================================================== */

void cpu_rms_norm_weighted(const float *x, const float *w,
                          float *out, int n, float eps)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    {
        const float inv_rms = 1.0f / sqrtf(sum_sq / (float)n + eps);
        for (int i = 0; i < n; i++) out[i] = x[i] * inv_rms * w[i];
    }
}

void cpu_swiglu(const float *gate, const float *up, float *out, int n)
{
    for (int i = 0; i < n; i++) {
        const float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

void cpu_softmax(float *x, int n)
{
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

void cpu_topk(const float *probs, int n, int K,
              int *indices, float *weights)
{
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
    if (w_sum > 0.0f)
        for (int k = 0; k < K; k++) weights[k] /= w_sum;
}

float cpu_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float cpu_softplus(float x)
{
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return logf(1.0f + expf(x));
}

void cpu_rms_norm_bare(float *x, int n, float eps)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / (float)n + eps);
    for (int i = 0; i < n; i++) x[i] *= inv_rms;
}

void cpu_l2_norm_bare(float *x, int n, float eps)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    {
        float inv_norm = 1.0f / sqrtf(sum_sq + eps);
        for (int i = 0; i < n; i++) x[i] *= inv_norm;
    }
}

void cpu_conv1d_step(const float *conv_state, const float *new_input,
                     const float *weight, float *out,
                     int channels, int kernel_size)
{
    for (int c = 0; c < channels; c++) {
        float acc = 0.0f;
        for (int k = 0; k < kernel_size - 1; k++)
            acc += conv_state[k * channels + c] * weight[c * kernel_size + k];
        acc += new_input[c] * weight[c * kernel_size + (kernel_size - 1)];
        out[c] = acc;
    }
}

void cpu_silu_inplace(float *x, int n)
{
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

void cpu_gated_rms_norm(const float *x, const float *z,
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

int ssm_k_head_for_v_head(int vh, int num_k_heads, int num_v_heads)
{
    if (num_k_heads <= 0 || num_v_heads <= 0) return 0;
    if (num_k_heads == num_v_heads) return vh;
    return vh % num_k_heads;
}

void cpu_repeat_heads(float *dst, const float *src,
                     int src_heads, int dst_heads, int head_dim)
{
    if (src_heads <= 0 || dst_heads <= 0 || head_dim <= 0) return;
    if (src_heads == dst_heads) {
        memcpy(dst, src, (size_t)dst_heads * head_dim * sizeof(float));
        return;
    }
    for (int h = 0; h < dst_heads; h++) {
        const float *src_h = src + (size_t)(h % src_heads) * head_dim;
        memcpy(dst + (size_t)h * head_dim, src_h, (size_t)head_dim * sizeof(float));
    }
}

/* ================================================================== */
/* AVX2 Horizontal Sum Helper                                         */
/* ================================================================== */
#ifdef __AVX2__
#include <immintrin.h>

static inline float m256_hsum(__m256 v) {
    __m256 shuf = _mm256_permute2f128_ps(v, v, 0x1);
    v = _mm256_add_ps(v, shuf);
    shuf = _mm256_shuffle_ps(v, v, 0x4E);
    v = _mm256_add_ps(v, shuf);
    shuf = _mm256_shuffle_ps(v, v, 0xB1);
    v = _mm256_add_ps(v, shuf);
    return _mm256_cvtss_f32(v);
}
#endif

void cpu_delta_net_autoregressive_step(const float *q, const float *k,
                                       const float *v, float gate, float beta,
                                       float *state, float *out,
                                       int value_dim, int key_dim,
                                       float *sk_buf, float *d_buf)
{
    /* Bounds checking: validate dimensions before processing */
    if (value_dim <= 0 || key_dim <= 0) {
        fprintf(stderr, "ERROR: cpu_delta_net_autoregressive_step invalid dims: value_dim=%d key_dim=%d\n",
                value_dim, key_dim);
        exit(1);
    }
    if (!state || !out || !q || !k || !v || !sk_buf || !d_buf) {
        fprintf(stderr, "ERROR: cpu_delta_net_autoregressive_step NULL pointer\n");
        exit(1);
    }

#ifdef __AVX2__
    /* SIMD-accelerated version for key_dim >= 8 */
    __m256 decay_v = _mm256_set1_ps(expf(gate));
    __m256 beta_v = _mm256_set1_ps(beta);

    /* Phase 1: State decay (multiply by scalar) */
    for (int vi = 0; vi < value_dim; vi++) {
        float *row = state + (size_t)vi * key_dim;
        for (int ki = 0; ki < key_dim; ki += 8) {
            __m256 state_v = _mm256_loadu_ps(row + ki);
            state_v = _mm256_mul_ps(state_v, decay_v);
            _mm256_storeu_ps(row + ki, state_v);
        }
    }

    /* Phase 2: Compute sk = state[vi] · k, then delta[vi] = (v[vi] - sk) * beta */
    for (int vi = 0; vi < value_dim; vi++) {
        const float *row = state + (size_t)vi * key_dim;
        __m256 sk_v = _mm256_setzero_ps();

        /* Vectorized dot product */
        for (int ki = 0; ki < key_dim; ki += 8) {
            __m256 row_v = _mm256_loadu_ps(row + ki);
            __m256 k_v = _mm256_loadu_ps(k + ki);
            sk_v = _mm256_fmadd_ps(row_v, k_v, sk_v);
        }

        /* Horizontal sum */
        float sk = m256_hsum(sk_v);
        sk_buf[vi] = sk;
        d_buf[vi] = (v[vi] - sk) * beta;
    }

    /* Phase 3: State update with FMA (state[vi][k] += k[k] * delta[vi]) */
    for (int vi = 0; vi < value_dim; vi++) {
        float *row = state + (size_t)vi * key_dim;
        float d = d_buf[vi];
        __m256 d_v = _mm256_set1_ps(d);

        for (int ki = 0; ki < key_dim; ki += 8) {
            __m256 state_v = _mm256_loadu_ps(row + ki);
            __m256 k_v = _mm256_loadu_ps(k + ki);
            /* state += k * d using FMA */
            state_v = _mm256_fmadd_ps(k_v, d_v, state_v);
            _mm256_storeu_ps(row + ki, state_v);
        }
    }

    /* Phase 4: Compute output (state[vi] · q) */
    for (int vi = 0; vi < value_dim; vi++) {
        const float *row = state + (size_t)vi * key_dim;
        __m256 acc_v = _mm256_setzero_ps();

        /* Vectorized dot product */
        for (int ki = 0; ki < key_dim; ki += 8) {
            __m256 row_v = _mm256_loadu_ps(row + ki);
            __m256 q_v = _mm256_loadu_ps(q + ki);
            acc_v = _mm256_fmadd_ps(row_v, q_v, acc_v);
        }

        /* Horizontal sum */
        out[vi] = m256_hsum(acc_v);
    }

#else
    /* Scalar fallback for systems without AVX2 */
    float decay = expf(gate);
    for (int vi = 0; vi < value_dim; vi++) {
        float *row = state + (size_t)vi * key_dim;
        for (int ki = 0; ki < key_dim; ki++) row[ki] *= decay;
    }
    for (int vi = 0; vi < value_dim; vi++) {
        const float *row = state + (size_t)vi * key_dim;
        float sk = 0.0f;
        for (int ki = 0; ki < key_dim; ki++) sk += row[ki] * k[ki];
        sk_buf[vi] = sk;
        d_buf[vi] = (v[vi] - sk) * beta;
    }
    for (int vi = 0; vi < value_dim; vi++) {
        float *row = state + (size_t)vi * key_dim;
        float d = d_buf[vi];
        for (int ki = 0; ki < key_dim; ki++) row[ki] += k[ki] * d;
    }
    for (int vi = 0; vi < value_dim; vi++) {
        const float *row = state + (size_t)vi * key_dim;
        float acc = 0.0f;
        for (int ki = 0; ki < key_dim; ki++) acc += row[ki] * q[ki];
        out[vi] = acc;
    }
#endif
}

/* ================================================================== */
/* Attention & RoPE                                                  */
/* ================================================================== */

int rope_select_stream_for_pair(const bread_model_config_t *cfg,
                               int pair_idx, int total_pairs)
{
    const int *sections = cfg->rope_sections;
    const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    if (sect_dims <= 0 || total_pairs <= 0) return 0;
    const int sector = pair_idx % sect_dims;
    if (cfg->rope_mrope_interleaved) {
        if ((sector % 3) == 1 && sector < 3 * sections[1]) return 1;
        else if ((sector % 3) == 2 && sector < 3 * sections[2]) return 2;
        else if ((sector % 3) == 0 && sector < 3 * sections[0]) return 0;
        return 3;
    }
    const int sec_w = sections[0] + sections[1];
    const int sec_e = sec_w + sections[2];
    if (sector >= sections[0] && sector < sec_w) return 1;
    else if (sector >= sec_w && sector < sec_e) return 2;
    else if (sector >= sec_e) return 3;
    return 0;
}

void apply_rotary_emb(const bread_model_config_t *cfg,
                     float *q, float *k, int pos)
{
    if (bread_get_disable_rope()) return;
    const int rotary_dim = cfg->head_dim_rope;
    const int half = rotary_dim / 2;
    const int num_q_heads = cfg->num_q_heads;
    const int num_kv_heads = cfg->num_kv_heads;
    const int head_dim = cfg->head_dim_qk;
    const float rope_freq_base = cfg->rope_freq_base;
    const float pos_streams[4] = { (float) pos, (float) pos, (float) pos, 0.0f };

    for (int h = 0; h < num_q_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            const int stream = rope_select_stream_for_pair(cfg, i, half);
            const float freq = 1.0f / powf(rope_freq_base, (float)(2 * i) / rotary_dim);
            const float angle = pos_streams[stream] * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float q0 = qh[i];
            float q1 = qh[i + half];
            qh[i]        = q0 * cos_a - q1 * sin_a;
            qh[i + half] = q0 * sin_a + q1 * cos_a;
        }
    }
    for (int h = 0; h < num_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < half; i++) {
            const int stream = rope_select_stream_for_pair(cfg, i, half);
            const float freq = 1.0f / powf(rope_freq_base, (float)(2 * i) / rotary_dim);
            const float angle = pos_streams[stream] * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float k0 = kh[i];
            float k1 = kh[i + half];
            kh[i]        = k0 * cos_a - k1 * sin_a;
            kh[i + half] = k0 * sin_a + k1 * cos_a;
        }
    }
}

void split_full_attn_q(const bread_model_config_t *cfg,
                      const float *q_full, float *q_score, float *q_gate)
{
    const int per_head_q = cfg->head_dim_qk + cfg->head_dim_qgate;
    for (int h = 0; h < cfg->num_q_heads; h++) {
        const float *src = q_full + h * per_head_q;
        memcpy(q_score + h * cfg->head_dim_qk,
               src, cfg->head_dim_qk * sizeof(float));
        memcpy(q_gate + h * cfg->head_dim_v,
               src + cfg->head_dim_qk, cfg->head_dim_v * sizeof(float));
    }
}

void apply_per_head_rms_norm(float *x, const float *w,
                            int num_heads, int head_dim, float eps)
{
    for (int h = 0; h < num_heads; h++) {
        float *xh = x + h * head_dim;
        float sum_sq = 0.0f;
        for (int i = 0; i < head_dim; i++) sum_sq += xh[i] * xh[i];
        {
            float inv_rms = 1.0f / sqrtf(sum_sq / (float)head_dim + eps);
            for (int i = 0; i < head_dim; i++) {
                xh[i] = xh[i] * inv_rms * w[i];
            }
        }
    }
}

/* ================================================================== */
/* Debug Tracing                                                     */
/* ================================================================== */

void trace_vec_stats(const char *label, const float *x, int n, int max_show)
{
    float min_v = 1e30f, max_v = -1e30f, sum_sq = 0.0f;
    fprintf(stderr, "TRACE %s: n=%d", label, n);
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum_sq += v * v;
    }
    fprintf(stderr, " rms=%.6f min=%.6f max=%.6f first=", sqrtf(sum_sq / (float)n), min_v, max_v);
    for (int i = 0; i < n && i < max_show; i++) {
        fprintf(stderr, "%s%.6f", i ? "," : "", x[i]);
    }
    fprintf(stderr, "\n");
}

void trace_sigmoid_stats(const char *label, const float *x, int n, int max_show)
{
    float min_v = 1e30f, max_v = -1e30f, sum_sq = 0.0f;
    fprintf(stderr, "TRACE %s: n=%d", label, n);
    for (int i = 0; i < n; i++) {
        float v = cpu_sigmoid(x[i]);
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum_sq += v * v;
    }
    fprintf(stderr, " rms=%.6f min=%.6f max=%.6f first=", sqrtf(sum_sq / (float)n), min_v, max_v);
    for (int i = 0; i < n && i < max_show; i++) {
        fprintf(stderr, "%s%.6f", i ? "," : "", cpu_sigmoid(x[i]));
    }
    fprintf(stderr, "\n");
}
