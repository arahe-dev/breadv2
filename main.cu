/* main.cu — BREAD inference loop
 *
 * Embeds input tokens → runs 40 transformer layers → final norm →
 * lm_head logits → greedy sampling → streaming output.
 *
 * Build:   build_main.bat
 * Run:     bread.exe [--prompt "..."] [--tokens N] [--model PATH]
 *
 * Pipeline per token:
 *   embed_token()          — Q4K row dequant → VRAM
 *   one_layer_forward()    × 40 layers (CMD1/CPU/CMD2/CPU/CMD3)
 *   apply_output_norm()    — RMSNorm with output_norm.weight
 *   compute_logits()       — bread_matvec with output.weight
 *   greedy_sample()        — argmax on CPU
 */

#ifdef _WIN32
#  define _CRT_SECURE_NO_WARNINGS
#  include <windows.h>
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "bread.h"
#include "gguf.h"
#include "loader.h"
#include "tokenizer.h"

/* ------------------------------------------------------------------ */
/* CUDA error check                                                     */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call) do {                                           \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d — %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));            \
        exit(1);                                                        \
    }                                                                   \
} while (0)

/* ------------------------------------------------------------------ */
/* Timing (Windows high-resolution)                                     */
/* ------------------------------------------------------------------ */
static double now_ms(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1000.0 / (double)freq.QuadPart;
}

/* ------------------------------------------------------------------ */
/* External functions                                                   */
/* ------------------------------------------------------------------ */

/* one_layer.cu */
extern void one_layer_forward(half *d_hidden, int layer_idx, int pos,
                               loader_t *L, gguf_ctx_t *g,
                               cudaStream_t stream_a);

/* kernels.cu */
extern void bread_matvec(void *w, half *x, half *y,
                          int rows, int cols, int qtype);

/* ------------------------------------------------------------------ */
/* Q4K constants (must match kernels.cu)                               */
/* ------------------------------------------------------------------ */
#define Q4K_BLOCK_BYTES   144
#define Q4K_BLOCK_ELEMS   256
#define QTYPE_Q4_K        12
#define QTYPE_Q6_K        14

/* ------------------------------------------------------------------ */
/* Host fp16 → float (needed for Q4K block header decoding)           */
/* ------------------------------------------------------------------ */
static float h2f_host(uint16_t h)
{
    uint32_t sign     = (uint32_t)(h >> 15) << 31;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x03FF;
    uint32_t bits;
    if      (exponent ==  0) bits = sign;
    else if (exponent == 31) bits = sign | 0x7F800000u | (mantissa << 13);
    else                     bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

/* ------------------------------------------------------------------ */
/* Q4K row dequant on host → half[]                                    */
/*                                                                      */
/* Dequantises hidden_dim elements from one embedding row (Q4_K).     */
/* row_data points to the first of (hidden_dim/256) Q4K blocks.       */
/* Logic mirrors cpu_dequant_q4k in kernels.cu.                        */
/* ------------------------------------------------------------------ */
static void dequant_q4k_row(const uint8_t *row_data, half *out_half,
                              int hidden_dim)
{
    int n_blocks = hidden_dim / Q4K_BLOCK_ELEMS;

    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk    = row_data + (size_t)b * Q4K_BLOCK_BYTES;
        const uint8_t *scales = blk + 4;
        const uint8_t *qs     = blk + 16;
        half          *dst    = out_half + b * Q4K_BLOCK_ELEMS;

        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw,    blk + 0, 2);
        memcpy(&dmin_raw, blk + 2, 2);
        float d    = h2f_host(d_raw);
        float dmin = h2f_host(dmin_raw);

        /* 4 groups × 64 elements; each group uses 32 qs bytes */
        int is = 0;
        const uint8_t *q = qs;
        for (int grp = 0; grp < 4; grp++) {
            /* get_scale_min_k4(is) — low nibble sub-group */
            uint8_t sc0, mn0;
            if (is < 4) {
                sc0 = scales[is]     & 63;
                mn0 = scales[is + 4] & 63;
            } else {
                sc0 = (scales[is + 4] & 0x0F) | ((scales[is - 4] >> 6) << 4);
                mn0 = (scales[is + 4] >>   4) | ((scales[is    ] >> 6) << 4);
            }
            float d_lo = d * sc0;
            float m_lo = dmin * mn0;
            is++;

            /* get_scale_min_k4(is) — high nibble sub-group */
            uint8_t sc1, mn1;
            if (is < 4) {
                sc1 = scales[is]     & 63;
                mn1 = scales[is + 4] & 63;
            } else {
                sc1 = (scales[is + 4] & 0x0F) | ((scales[is - 4] >> 6) << 4);
                mn1 = (scales[is + 4] >>   4) | ((scales[is    ] >> 6) << 4);
            }
            float d_hi = d * sc1;
            float m_hi = dmin * mn1;
            is++;

            for (int l = 0; l < 32; l++)
                dst[grp * 64 + l]      = __float2half(d_lo * (float)(q[l] & 0xF) - m_lo);
            for (int l = 0; l < 32; l++)
                dst[grp * 64 + 32 + l] = __float2half(d_hi * (float)(q[l] >>  4) - m_hi);
            q += 32;
        }
    }
}

/* ------------------------------------------------------------------ */
/* GPU kernel: RMSNorm for final output norm (static — no collision    */
/* with same-named kernel in one_layer.cu which is also static)       */
/* ------------------------------------------------------------------ */
static __global__ void rmsnorm_output(half *x, const float *w,
                                       int n, float eps)
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

/* ------------------------------------------------------------------ */
/* GPU kernel: F16 matvec — fallback for F16 lm_head                  */
/*                                                                      */
/* Each block handles one output row.  Each thread accumulates cols   */
/* at stride 256, then block reduces.                                  */
/* ------------------------------------------------------------------ */
static __global__ void f16_matvec(const half *w, const half *x,
                                    half *y, int rows, int cols)
{
    __shared__ float sdata[256];
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int j = tid; j < cols; j += 256)
        sum += __half2float(w[(size_t)row * cols + j]) * __half2float(x[j]);

    sdata[tid] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[row] = __float2half(sdata[0]);
}

/* ================================================================== */
/*  High-level inference helpers                                        */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/* embed_token: copy one token's embedding row to d_hidden (VRAM)     */
/*                                                                      */
/* Supports Q4_K (primary) and F16 (fallback) embedding types.       */
/* h_emb_row: caller-allocated host buffer of BREAD_HIDDEN_DIM halfs. */
/* ------------------------------------------------------------------ */
static void embed_token(int32_t token_id,
                         const loader_t *L, const gguf_ctx_t *g,
                         half *d_hidden, half *h_emb_row)
{
    const gguf_tensor_t *et = gguf_find_tensor(g, "token_embd.weight");
    if (!et) { fprintf(stderr, "token_embd.weight not found\n"); exit(1); }

    const uint8_t *emb_base = L->pinned_data + L->data_offset + et->offset;

    if (et->type == GGML_TYPE_Q4_K) {
        size_t row_bytes = (size_t)(BREAD_HIDDEN_DIM / Q4K_BLOCK_ELEMS)
                           * Q4K_BLOCK_BYTES;
        dequant_q4k_row(emb_base + (size_t)token_id * row_bytes,
                         h_emb_row, BREAD_HIDDEN_DIM);
    } else if (et->type == GGML_TYPE_F16) {
        memcpy(h_emb_row,
               emb_base + (size_t)token_id * BREAD_HIDDEN_DIM * sizeof(half),
               (size_t)BREAD_HIDDEN_DIM * sizeof(half));
    } else {
        fprintf(stderr, "embed_token: unsupported type %u\n", et->type);
        exit(1);
    }

    CUDA_CHECK(cudaMemcpy(d_hidden, h_emb_row,
                           BREAD_HIDDEN_DIM * sizeof(half),
                           cudaMemcpyHostToDevice));
}

/* ------------------------------------------------------------------ */
/* apply_output_norm: final RMSNorm in-place with output_norm.weight  */
/*                                                                      */
/* d_norm_w: pre-loaded F32 weights already in VRAM.                  */
/* ------------------------------------------------------------------ */
static void apply_output_norm(half *d_hidden, const float *d_norm_w,
                               cudaStream_t stream_a)
{
    rmsnorm_output<<<1, 256, 0, stream_a>>>(
        d_hidden, d_norm_w, BREAD_HIDDEN_DIM, BREAD_RMS_EPS);
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
}

/* ------------------------------------------------------------------ */
/* compute_logits: lm_head matvec → d_logits[BREAD_VOCAB_SIZE]        */
/*                                                                      */
/* d_output_w: pre-loaded lm_head weights in VRAM.                    */
/* output_w_type: GGML type of d_output_w.                            */
/* ------------------------------------------------------------------ */
static void compute_logits(half *d_hidden,
                             void *d_output_w, uint32_t output_w_type,
                             half *d_logits, cudaStream_t stream_a)
{
    if (output_w_type == GGML_TYPE_Q4_K) {
        bread_matvec(d_output_w, d_hidden, d_logits,
                     BREAD_VOCAB_SIZE, BREAD_HIDDEN_DIM, QTYPE_Q4_K);
    } else if (output_w_type == GGML_TYPE_Q6_K) {
        bread_matvec(d_output_w, d_hidden, d_logits,
                     BREAD_VOCAB_SIZE, BREAD_HIDDEN_DIM, QTYPE_Q6_K);
    } else if (output_w_type == GGML_TYPE_F16) {
        f16_matvec<<<BREAD_VOCAB_SIZE, 256, 0, stream_a>>>(
            (const half *)d_output_w, d_hidden, d_logits,
            BREAD_VOCAB_SIZE, BREAD_HIDDEN_DIM);
    } else {
        fprintf(stderr, "compute_logits: unsupported weight type %u\n",
                output_w_type);
        exit(1);
    }
    /* Sync all streams (bread_matvec uses null stream; f16_matvec uses stream_a) */
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* ------------------------------------------------------------------ */
/* greedy_sample: copy d_logits to host, return argmax token id       */
/* ------------------------------------------------------------------ */
static int32_t greedy_sample(const half *d_logits, half *h_logits)
{
    CUDA_CHECK(cudaMemcpy(h_logits, d_logits,
                           (size_t)BREAD_VOCAB_SIZE * sizeof(half),
                           cudaMemcpyDeviceToHost));
    int32_t best_id  = 0;
    float   best_val = __half2float(h_logits[0]);
    for (int i = 1; i < BREAD_VOCAB_SIZE; i++) {
        float v = __half2float(h_logits[i]);
        if (v > best_val) { best_val = v; best_id = (int32_t)i; }
    }
    return best_id;
}

/* ================================================================== */
/*  main                                                                */
/* ================================================================== */

int main(int argc, char **argv)
{
    const char *model_path = BREAD_MODEL_PATH;
    const char *prompt     = "Hello, I am";
    int         max_tokens = 50;

    /* -- Parse args ------------------------------------------------- */
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--prompt") && i+1 < argc) prompt     = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i+1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--model")  && i+1 < argc) model_path = argv[++i];
    }

    printf("=== BREAD inference ===\n");
    printf("Prompt   : \"%s\"\n", prompt);
    printf("MaxTokens: %d\n\n", max_tokens);

    /* -- Load model into pinned RAM --------------------------------- */
    printf("[1/4] Loading model into pinned RAM (~22 GB)...\n");
    double t0 = now_ms();
    loader_t *L = loader_init(model_path);
    if (!L) { fprintf(stderr, "loader_init failed\n"); return 1; }
    printf("      done in %.1f s\n\n", (now_ms() - t0) / 1000.0);

    /* -- Open GGUF for metadata ------------------------------------- */
    gguf_ctx_t *g = gguf_open(model_path);
    if (!g) { fprintf(stderr, "gguf_open failed\n"); return 1; }

    /* -- Load tokenizer --------------------------------------------- */
    printf("[2/4] Loading tokenizer...\n");
    tokenizer_t *tok = tokenizer_load(model_path);
    if (!tok) { fprintf(stderr, "tokenizer_load failed\n"); return 1; }
    printf("      vocab size: %d\n\n", tokenizer_vocab_size(tok));

    /* -- Pre-load output_norm.weight and output.weight to VRAM ------ */
    printf("[3/4] Uploading output_norm + lm_head to VRAM...\n");

    const gguf_tensor_t *norm_t = gguf_find_tensor(g, "output_norm.weight");
    if (!norm_t) { fprintf(stderr, "output_norm.weight not found\n"); return 1; }
    float *d_norm_w = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_norm_w, norm_t->size));
    CUDA_CHECK(cudaMemcpy(d_norm_w,
                           L->pinned_data + L->data_offset + norm_t->offset,
                           norm_t->size, cudaMemcpyHostToDevice));
    printf("      output_norm.weight: %.1f KB (F32)\n",
           norm_t->size / 1024.0);

    /* Try output.weight; fall back to token_embd.weight (tied) */
    const gguf_tensor_t *out_t = gguf_find_tensor(g, "output.weight");
    if (!out_t) {
        printf("      output.weight not found — using token_embd.weight (tied)\n");
        out_t = gguf_find_tensor(g, "token_embd.weight");
    }
    if (!out_t) { fprintf(stderr, "no lm_head weight found\n"); return 1; }
    void *d_output_w = NULL;
    CUDA_CHECK(cudaMalloc(&d_output_w, out_t->size));
    CUDA_CHECK(cudaMemcpy(d_output_w,
                           L->pinned_data + L->data_offset + out_t->offset,
                           out_t->size, cudaMemcpyHostToDevice));
    printf("      output.weight:      %.1f MB (type %s)\n",
           out_t->size / 1024.0 / 1024.0,
           ggml_type_name(out_t->type));

    /* -- Encode prompt ---------------------------------------------- */
    printf("\n[4/4] Encoding prompt...\n");
    int32_t token_buf[4096];
    int32_t bos = tokenizer_bos(tok);
    int n_prompt;
    if (bos >= 0) {
        token_buf[0] = bos;
        n_prompt = 1 + tokenizer_encode(tok, prompt, token_buf + 1, 4095);
    } else {
        /* No BOS token for this model (Qwen3.5 style) — encode prompt directly */
        n_prompt = tokenizer_encode(tok, prompt, token_buf, 4096);
    }
    printf("      %d tokens: [", n_prompt);
    for (int i = 0; i < n_prompt && i < 8; i++)
        printf("%s%d", i ? " " : "", token_buf[i]);
    if (n_prompt > 8) printf(" ...");
    printf("]\n\n");

    /* -- CUDA resources --------------------------------------------- */
    cudaStream_t stream_a;
    CUDA_CHECK(cudaStreamCreate(&stream_a));

    half *d_hidden = NULL;
    half *d_logits = NULL;
    CUDA_CHECK(cudaMalloc(&d_hidden, BREAD_HIDDEN_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_logits,
                           (size_t)BREAD_VOCAB_SIZE * sizeof(half)));

    /* Host buffers reused across tokens */
    half *h_emb_row = (half *)malloc(BREAD_HIDDEN_DIM * sizeof(half));
    half *h_logits  = (half *)malloc((size_t)BREAD_VOCAB_SIZE * sizeof(half));
    if (!h_emb_row || !h_logits) {
        fprintf(stderr, "malloc failed\n"); return 1;
    }

    /* ================================================================
     * Prompt prefill: feed each prompt token through all 40 layers.
     * The final hidden state of the last token seeds generation.
     * (No KV cache — each token sees only itself; layers apply norms
     *  and MoE FFN correctly.  SSM state is stubbed.)
     * ================================================================ */
    printf("--- Prompt prefill (%d tokens) ---\n", n_prompt);
    double t_prefill_start = now_ms();

    for (int p = 0; p < n_prompt; p++) {
        embed_token(token_buf[p], L, g, d_hidden, h_emb_row);
        for (int layer = 0; layer < BREAD_NUM_LAYERS; layer++)
            one_layer_forward(d_hidden, layer, p, L, g, stream_a);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double t_prefill_done = now_ms();
    double prefill_ms = t_prefill_done - t_prefill_start;
    printf("    prefill: %.0f ms  (%.1f ms/tok)\n\n",
           prefill_ms, prefill_ms / n_prompt);

    /* ================================================================
     * First token: apply output norm + lm_head on last prompt state.
     * ================================================================ */
    double t_ttft_start = now_ms();

    apply_output_norm(d_hidden, d_norm_w, stream_a);
    compute_logits(d_hidden, d_output_w, out_t->type, d_logits, stream_a);
    int32_t next_tok = greedy_sample(d_logits, h_logits);

    double t_ttft_done = now_ms();
    double ttft_ms = t_ttft_done - t_ttft_start;

    /* Stream first token */
    printf("--- Generated output ---\n");
    {
        char *s = tokenizer_decode(tok, &next_tok, 1);
        printf("%s", s);
        fflush(stdout);
        free(s);
    }

    /* ================================================================
     * Autoregressive decode loop
     * ================================================================ */
    int32_t eos       = tokenizer_eos(tok);
    int     n_gen     = 1;   /* first token already done */
    double  t_gen_start = now_ms();

    while (n_gen < max_tokens && next_tok != eos) {
        embed_token(next_tok, L, g, d_hidden, h_emb_row);

        for (int layer = 0; layer < BREAD_NUM_LAYERS; layer++)
            one_layer_forward(d_hidden, layer, n_prompt + n_gen, L, g, stream_a);
        CUDA_CHECK(cudaDeviceSynchronize());

        apply_output_norm(d_hidden, d_norm_w, stream_a);
        compute_logits(d_hidden, d_output_w, out_t->type, d_logits, stream_a);
        next_tok = greedy_sample(d_logits, h_logits);

        if (next_tok != eos) {
            char *s = tokenizer_decode(tok, &next_tok, 1);
            printf("%s", s);
            fflush(stdout);
            free(s);
        }
        n_gen++;
    }

    double t_gen_done = now_ms();
    printf("\n\n");

    /* ================================================================
     * Benchmark report
     * ================================================================ */
    double decode_ms  = (t_gen_done - t_gen_start) + ttft_ms;
    double tok_per_s  = (n_gen > 0) ? (double)n_gen / (decode_ms / 1000.0) : 0.0;

    printf("=== Benchmark ===\n");
    printf("  Prompt tokens    : %d\n",    n_prompt);
    printf("  Generated tokens : %d\n",    n_gen);
    printf("  Prefill          : %.0f ms  (%.1f ms/tok)\n",
           prefill_ms, prefill_ms / n_prompt);
    printf("  Time to 1st tok  : %.1f ms\n",   ttft_ms);
    printf("  Total decode     : %.0f ms  (%.2f tok/s)\n",
           decode_ms, tok_per_s);
    printf("\n");
    printf("  NOTE: one_layer_forward() allocates/frees VRAM per call —\n");
    printf("  bulk of time is cudaMalloc overhead, not compute.\n");
    printf("  Next step: cache non-expert weights in VRAM at startup.\n");

    /* -- Cleanup ---------------------------------------------------- */
    free(h_emb_row);
    free(h_logits);
    cudaFree(d_hidden);
    cudaFree(d_logits);
    cudaFree(d_norm_w);
    cudaFree(d_output_w);
    cudaStreamDestroy(stream_a);
    tokenizer_free(tok);
    gguf_close(g);
    loader_free(L);

    return 0;
}
