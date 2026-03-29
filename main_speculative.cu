/* main_speculative.cu — SRIRACHA Step 2: Speculative decoding driver.
 *
 * Loads BREAD (35B-A3B target) and SRIRACHA (0.8B draft).
 * Runs greedy spec-decode loop:
 *   1. SRIRACHA drafts K tokens from last confirmed token.
 *   2. BREAD verifies sequentially; accept longest prefix where
 *      BREAD's argmax matches the draft at each position.
 *   3. Output bonus token from BREAD on first mismatch (or after all K).
 *   4. Rewind SRIRACHA's KV cache on partial rejection.
 *
 * Step 2 goal: correctness + acceptance rate measurement.
 * Performance optimisation (DMA amortisation) is Step 3.
 *
 * Build: build_speculative.ps1
 * Run:   speculative.exe [--prompt "..."] [--tokens N] [--spec-depth K]
 *                        [--model PATH] [--draft PATH]
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
#include "bread_utils.h"
#include "sriracha.h"

/* ------------------------------------------------------------------ */
/* Quant constants (must match kernels.cu)                             */
/* ------------------------------------------------------------------ */
#define Q4K_BLOCK_BYTES  144
#define Q4K_BLOCK_ELEMS  256
#define QTYPE_Q4_K       12
#define QTYPE_Q6_K       14

/* ------------------------------------------------------------------ */
/* Default model paths                                                 */
/* ------------------------------------------------------------------ */
#define TARGET_MODEL_PATH \
    "C:\\Users\\arahe\\.ollama\\models\\blobs\\" \
    "sha256-900dde62fb7ebe8a5a25e35d5b7633f403f226a310965fed51d50f5238ba145a"

#define DRAFT_MODEL_PATH \
    "C:\\Users\\arahe\\.ollama\\models\\blobs\\" \
    "sha256-afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5"

/* ------------------------------------------------------------------ */
/* Timing                                                               */
/* ------------------------------------------------------------------ */
static double now_ms(void)
{
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1000.0 / (double)freq.QuadPart;
}

/* ------------------------------------------------------------------ */
/* External functions                                                  */
/* ------------------------------------------------------------------ */
extern void one_layer_forward(half *d_hidden, int layer_idx, int pos,
                               loader_t *L, gguf_ctx_t *g,
                               weight_cache_t *wc, cudaStream_t stream_a);
extern void bread_matvec(void *w, half *x, half *y,
                          int rows, int cols, int qtype, cudaStream_t stream);

/* ------------------------------------------------------------------ */
/* Q4K embedding row dequant (duplicated from main.cu — static there) */
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
        float d    = bread_h2f(d_raw);
        float dmin = bread_h2f(dmin_raw);

        int is = 0;
        const uint8_t *q = qs;
        for (int grp = 0; grp < 4; grp++) {
            uint8_t sc0, mn0;
            if (is < 4) {
                sc0 = scales[is]     & 63;
                mn0 = scales[is + 4] & 63;
            } else {
                sc0 = (scales[is + 4] & 0x0F) | ((scales[is - 4] >> 6) << 4);
                mn0 = (scales[is + 4] >>   4) | ((scales[is    ] >> 6) << 4);
            }
            float d_lo = d * sc0, m_lo = dmin * mn0;
            is++;

            uint8_t sc1, mn1;
            if (is < 4) {
                sc1 = scales[is]     & 63;
                mn1 = scales[is + 4] & 63;
            } else {
                sc1 = (scales[is + 4] & 0x0F) | ((scales[is - 4] >> 6) << 4);
                mn1 = (scales[is + 4] >>   4) | ((scales[is    ] >> 6) << 4);
            }
            float d_hi = d * sc1, m_hi = dmin * mn1;
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
/* RMSNorm GPU kernel for final output norm                            */
/* ------------------------------------------------------------------ */
static __global__ void spec_rmsnorm_output(half *x, const float *w,
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

/* F16 matvec fallback for lm_head */
static __global__ void spec_f16_matvec(const half *w, const half *x,
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
/* BREAD context and per-token helpers                                 */
/* ================================================================== */

typedef struct {
    loader_t              *L;
    gguf_ctx_t            *g;
    weight_cache_t        *wc;
    const bread_model_config_t *cfg;
    float                 *d_norm_w;
    void                  *d_out_w;
    uint32_t               out_w_type;
    half                  *d_hidden;
    half                  *d_logits;
    half                  *h_emb_row;
    half                  *h_logits;
    cudaStream_t           stream;
} bread_ctx_t;

/* Embed one token into d_hidden (VRAM). */
static void bc_embed(bread_ctx_t *b, int32_t token_id)
{
    const gguf_tensor_t *et = gguf_find_tensor(b->g, "token_embd.weight");
    if (!et) { fprintf(stderr, "token_embd.weight not found\n"); exit(1); }
    const uint8_t *base = b->L->pinned_data + b->L->data_offset + et->offset;
    int H = b->cfg->hidden_dim;

    if (et->type == GGML_TYPE_Q4_K) {
        size_t rb = (size_t)(H / Q4K_BLOCK_ELEMS) * Q4K_BLOCK_BYTES;
        dequant_q4k_row(base + (size_t)token_id * rb, b->h_emb_row, H);
    } else if (et->type == GGML_TYPE_F16) {
        memcpy(b->h_emb_row,
               base + (size_t)token_id * H * sizeof(half),
               (size_t)H * sizeof(half));
    } else {
        fprintf(stderr, "bc_embed: unsupported type %u\n", et->type);
        exit(1);
    }
    CUDA_CHECK(cudaMemcpy(b->d_hidden, b->h_emb_row,
                           (size_t)H * sizeof(half), cudaMemcpyHostToDevice));
}

/* Apply output RMSNorm in-place. */
static void bc_output_norm(bread_ctx_t *b)
{
    spec_rmsnorm_output<<<1, 256, 0, b->stream>>>(
        b->d_hidden, b->d_norm_w, b->cfg->hidden_dim, b->cfg->rms_eps);
    CUDA_CHECK(cudaStreamSynchronize(b->stream));
}

/* Compute logits d_hidden → d_logits. */
static void bc_logits(bread_ctx_t *b)
{
    if (b->out_w_type == GGML_TYPE_Q4_K) {
        bread_matvec(b->d_out_w, b->d_hidden, b->d_logits,
                     b->cfg->vocab_size, b->cfg->hidden_dim, QTYPE_Q4_K, b->stream);
    } else if (b->out_w_type == GGML_TYPE_Q6_K) {
        bread_matvec(b->d_out_w, b->d_hidden, b->d_logits,
                     b->cfg->vocab_size, b->cfg->hidden_dim, QTYPE_Q6_K, b->stream);
    } else if (b->out_w_type == GGML_TYPE_F16) {
        spec_f16_matvec<<<b->cfg->vocab_size, 256, 0, b->stream>>>(
            (const half *)b->d_out_w, b->d_hidden, b->d_logits,
            b->cfg->vocab_size, b->cfg->hidden_dim);
    } else {
        fprintf(stderr, "bc_logits: unsupported type %u\n", b->out_w_type);
        exit(1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* Greedy argmax over d_logits. */
static int32_t bc_greedy(bread_ctx_t *b)
{
    CUDA_CHECK(cudaMemcpy(b->h_logits, b->d_logits,
                           (size_t)b->cfg->vocab_size * sizeof(half),
                           cudaMemcpyDeviceToHost));
    int32_t best_id  = 0;
    float   best_val = __half2float(b->h_logits[0]);
    for (int i = 1; i < b->cfg->vocab_size; i++) {
        float v = __half2float(b->h_logits[i]);
        if (v > best_val) { best_val = v; best_id = (int32_t)i; }
    }
    return best_id;
}

/*
 * bread_step: embed token at pos, run all layers, apply norm + logits,
 * return greedy next token.
 */
static int32_t bread_step(bread_ctx_t *b, int32_t token, int pos)
{
    bc_embed(b, token);
    for (int l = 0; l < b->cfg->num_layers; l++)
        one_layer_forward(b->d_hidden, l, pos, b->L, b->g, b->wc, b->stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    bc_output_norm(b);
    bc_logits(b);
    return bc_greedy(b);
}

/* ================================================================== */
/* Prompt helpers                                                       */
/* ================================================================== */

static char *wrap_prompt(const tokenizer_t *tok, const char *prompt)
{
    const char *pre = tokenizer_pre(tok);
    if (!pre || strcmp(pre, "qwen35") != 0)
        return _strdup(prompt);
    if (strncmp(prompt, "<|im_start|>", 12) == 0)
        return _strdup(prompt);
    size_t n = strlen(prompt) + 128;
    char *buf = (char *)malloc(n);
    if (!buf) return NULL;
    snprintf(buf, n,
             "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
             prompt);
    return buf;
}

/* ================================================================== */
/* main                                                                 */
/* ================================================================== */

int main(int argc, char **argv)
{
    const char *target_path = TARGET_MODEL_PATH;
    const char *draft_path  = DRAFT_MODEL_PATH;
    const char *prompt      = "The capital of France is";
    int         max_tokens  = 60;
    int         spec_depth  = 5;   /* K draft tokens per round */

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model")      && i+1 < argc) target_path = argv[++i];
        else if (!strcmp(argv[i], "--draft")      && i+1 < argc) draft_path  = argv[++i];
        else if (!strcmp(argv[i], "--prompt")     && i+1 < argc) prompt      = argv[++i];
        else if (!strcmp(argv[i], "--tokens")     && i+1 < argc) max_tokens  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--spec-depth") && i+1 < argc) spec_depth  = atoi(argv[++i]);
    }

    if (spec_depth > 15) spec_depth = 15;  /* sanity cap */

    printf("=== SRIRACHA Step 2: Speculative Decoding ===\n");
    printf("Target : %s\n", target_path);
    printf("Draft  : %s\n", draft_path);
    printf("Prompt : \"%s\"\n", prompt);
    printf("MaxToks: %d  SpecDepth: K=%d\n\n", max_tokens, spec_depth);

    /* ---- Load target (BREAD 35B-A3B) ---- */
    printf("[1/5] Loading target model (BREAD 35B-A3B)...\n");
    double t0 = now_ms();
    loader_t *L = loader_init(target_path);
    if (!L) { fprintf(stderr, "loader_init failed\n"); return 1; }
    printf("      %.1f s\n", (now_ms() - t0) / 1000.0);

    gguf_ctx_t *g = gguf_open(target_path);
    if (!g) { fprintf(stderr, "gguf_open failed\n"); return 1; }
    if (bread_model_config_init(target_path, g) != 0) {
        fprintf(stderr, "bread_model_config_init failed\n"); return 1;
    }
    const bread_model_config_t *cfg = bread_model_config_get();

    weight_cache_t *wc = weight_cache_init(L, g, cfg->num_layers,
                                            bread_layer_is_full_attention);
    if (!wc) { fprintf(stderr, "weight_cache_init failed\n"); return 1; }

    /* ---- Load tokenizer ---- */
    printf("[2/5] Loading tokenizer...\n");
    tokenizer_t *tok = tokenizer_load(target_path);
    if (!tok) { fprintf(stderr, "tokenizer_load failed\n"); return 1; }
    printf("      vocab=%d\n\n", tokenizer_vocab_size(tok));

    /* ---- Upload BREAD output head ---- */
    printf("[3/5] Uploading BREAD output head to VRAM...\n");
    const gguf_tensor_t *norm_t = gguf_find_tensor(g, "output_norm.weight");
    if (!norm_t) { fprintf(stderr, "output_norm.weight not found\n"); return 1; }
    float *d_norm_w = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_norm_w, norm_t->size));
    CUDA_CHECK(cudaMemcpy(d_norm_w,
                           L->pinned_data + L->data_offset + norm_t->offset,
                           norm_t->size, cudaMemcpyHostToDevice));

    const gguf_tensor_t *out_t = gguf_find_tensor(g, "output.weight");
    if (!out_t) out_t = gguf_find_tensor(g, "token_embd.weight");
    if (!out_t) { fprintf(stderr, "no lm_head weight found\n"); return 1; }
    void *d_out_w = NULL;
    CUDA_CHECK(cudaMalloc(&d_out_w, out_t->size));
    CUDA_CHECK(cudaMemcpy(d_out_w,
                           L->pinned_data + L->data_offset + out_t->offset,
                           out_t->size, cudaMemcpyHostToDevice));
    printf("      norm=%.1f KB  lm_head=%.1f MB  type=%s\n\n",
           norm_t->size / 1024.0,
           out_t->size / 1024.0 / 1024.0,
           ggml_type_name(out_t->type));

    /* ---- Init SRIRACHA draft model ---- */
    printf("[4/5] Initializing SRIRACHA draft model...\n");
    double t_sr = now_ms();
    sriracha_t *sr = sriracha_init(draft_path, spec_depth);
    if (!sr) { fprintf(stderr, "sriracha_init failed\n"); return 1; }
    printf("      %.1f s\n\n", (now_ms() - t_sr) / 1000.0);

    /* ---- Encode prompt ---- */
    printf("[5/5] Encoding prompt...\n");
    int32_t token_buf[4096];
    char *wprompt = wrap_prompt(tok, prompt);
    if (!wprompt) { fprintf(stderr, "prompt wrap failed\n"); return 1; }
    int32_t bos = tokenizer_bos(tok);
    int n_prompt;
    if (bos >= 0) {
        token_buf[0] = bos;
        n_prompt = 1 + tokenizer_encode(tok, wprompt, token_buf + 1, 4095);
    } else {
        n_prompt = tokenizer_encode(tok, wprompt, token_buf, 4096);
    }
    free(wprompt);
    printf("      %d tokens\n\n", n_prompt);

    /* ---- CUDA resources for BREAD ---- */
    cudaStream_t stream_a;
    CUDA_CHECK(cudaStreamCreate(&stream_a));
    half *d_hidden = NULL, *d_logits = NULL;
    CUDA_CHECK(cudaMalloc(&d_hidden, (size_t)cfg->hidden_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)cfg->vocab_size * sizeof(half)));
    half *h_emb_row = (half *)malloc((size_t)cfg->hidden_dim * sizeof(half));
    half *h_logits  = (half *)malloc((size_t)cfg->vocab_size * sizeof(half));
    if (!h_emb_row || !h_logits) { fprintf(stderr, "malloc failed\n"); return 1; }

    bread_ctx_t b = {
        L, g, wc, cfg,
        d_norm_w, d_out_w, out_t->type,
        d_hidden, d_logits,
        h_emb_row, h_logits,
        stream_a
    };

    int32_t eos = tokenizer_eos(tok);

    /* ================================================================
     * Prefill both models on the prompt.
     * After prefill, BREAD's logits give us first_tok (verified).
     * SRIRACHA's state is ready to draft from that seed.
     * ================================================================ */
    printf("--- Prefilling both models on %d prompt tokens ---\n", n_prompt);
    double t_pf = now_ms();

    /* BREAD prefill */
    for (int p = 0; p < n_prompt; p++) {
        bc_embed(&b, token_buf[p]);
        for (int l = 0; l < cfg->num_layers; l++)
            one_layer_forward(d_hidden, l, p, L, g, wc, stream_a);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    bc_output_norm(&b);
    bc_logits(&b);
    int32_t first_tok = bc_greedy(&b);

    /* SRIRACHA prefill */
    sriracha_prefill(sr, token_buf, n_prompt);

    double t_pf_done = now_ms();
    printf("    done in %.0f ms\n\n", t_pf_done - t_pf);

    /* ================================================================
     * Spec decode loop
     *
     * Position semantics (mirrors BREAD's main.cu generation loop):
     *   prompt positions: 0 .. n_prompt-1
     *   first_tok confirmed at position n_prompt (virtual, skipped in loop)
     *   first_tok processed (to predict next) at position n_prompt+1
     *
     * confirmed_tok: last verified token (seed for next draft)
     * confirmed_pos: position where confirmed_tok will be processed next
     * ================================================================ */
    int confirmed_pos = n_prompt + 1;
    int32_t confirmed_tok = first_tok;

    /* Acceptance stats: accept_count[i] = times draft[i] was accepted */
    int accept_count[16] = {0};
    int rounds     = 0;
    int total_gen  = 0;          /* tokens output so far */
    int total_acc  = 0;          /* total draft tokens accepted */
    int total_draft = 0;         /* total draft tokens offered */

    printf("--- Speculative decode output ---\n");
    /* Print first confirmed token */
    {
        char *s = tokenizer_decode(tok, &first_tok, 1);
        printf("%s", s);
        fflush(stdout);
        free(s);
    }
    total_gen = 1;

    double t_gen_start = now_ms();

    while (total_gen < max_tokens && confirmed_tok != eos) {

        /* 1. Draft K tokens from confirmed_tok */
        int32_t draft[16];
        int n_drafted = sriracha_draft_from(sr, confirmed_tok, confirmed_pos, draft);
        if (n_drafted <= 0) break;
        total_draft += n_drafted;

        /* 2. Verify sequentially with BREAD */
        int32_t target_pred = bread_step(&b, confirmed_tok, confirmed_pos);
        int n_accept = 0;

        for (int i = 0; i < n_drafted; i++) {
            if (target_pred == draft[i] && target_pred != eos) {
                accept_count[i]++;
                n_accept++;
                total_acc++;
                /* Print accepted draft token */
                char *s = tokenizer_decode(tok, &draft[i], 1);
                printf("%s", s);
                fflush(stdout);
                free(s);
                total_gen++;
                if (total_gen >= max_tokens) break;
                target_pred = bread_step(&b, draft[i], confirmed_pos + 1 + i);
            } else {
                break;
            }
        }

        /* 3. Output bonus token (BREAD's prediction at first mismatch or after all K) */
        if (total_gen < max_tokens && target_pred != eos) {
            char *s = tokenizer_decode(tok, &target_pred, 1);
            printf("%s", s);
            fflush(stdout);
            free(s);
            total_gen++;
        }

        /* 4. Update confirmed state */
        int new_pos = confirmed_pos + n_accept + 1;
        if (n_accept < n_drafted) {
            /* Partial rejection — rewind SRIRACHA KV to discard rejected drafts */
            sriracha_rewind(sr, new_pos);
        }
        confirmed_tok = target_pred;
        confirmed_pos = new_pos;
        rounds++;

        if (confirmed_tok == eos) break;
    }

    double t_gen_done = now_ms();
    printf("\n\n");

    /* ================================================================
     * Stats
     * ================================================================ */
    double gen_ms   = t_gen_done - t_gen_start;
    double tok_per_s = (total_gen > 1 && gen_ms > 0)
                     ? (double)(total_gen - 1) / (gen_ms / 1000.0) : 0.0;
    double acc_rate = (total_draft > 0)
                    ? 100.0 * total_acc / total_draft : 0.0;

    printf("=== Speculative Decode Stats ===\n");
    printf("  Prompt tokens    : %d\n",   n_prompt);
    printf("  Generated tokens : %d\n",   total_gen);
    printf("  Spec rounds      : %d\n",   rounds);
    printf("  Draft tokens     : %d (offered), %d (accepted)\n",
           total_draft, total_acc);
    printf("  Acceptance rate  : %.1f%%\n", acc_rate);
    printf("  Throughput       : %.2f tok/s\n", tok_per_s);
    printf("  Spec K           : %d\n\n", spec_depth);

    printf("  Acceptance by draft position:\n");
    for (int i = 0; i < spec_depth; i++) {
        int offered = rounds;   /* each round offers at least 1 draft */
        double pct = (offered > 0) ? 100.0 * accept_count[i] / offered : 0.0;
        printf("    [%d] %d/%d = %.1f%%\n", i, accept_count[i], offered, pct);
    }
    printf("\n");

    /* ---- SRIRACHA stats ---- */
    if (sr->n_drafted > 0) {
        printf("  SRIRACHA internal: %d drafted, %d accepted (%.1f%%)\n\n",
               sr->n_drafted, sr->n_accepted,
               100.0 * sr->n_accepted / sr->n_drafted);
    }

    /* ---- Cleanup ---- */
    sriracha_free(sr);
    free(h_emb_row);
    free(h_logits);
    cudaFree(d_hidden);
    cudaFree(d_logits);
    cudaFree(d_norm_w);
    cudaFree(d_out_w);
    cudaStreamDestroy(stream_a);
    weight_cache_free(wc);
    tokenizer_free(tok);
    gguf_close(g);
    loader_free(L);

    return 0;
}
