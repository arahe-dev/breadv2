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
#include "bread_utils.h"
#include "buffer_pool.h"
#include "hooks.h"
#include "progress_tracking.h"

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
                               weight_cache_t *wc,
                               cudaStream_t stream_a);
extern float route_layer(loader_t *L, gguf_ctx_t *g, int layer_idx,
                         const half *d_normed2,
                         int *expert_indices, float *expert_weights);
extern float one_layer_cpu_hidden_rms(int hidden_dim);
extern float one_layer_last_branch_rms(void);

/* kernels.cu */
extern void bread_matvec(void *w, half *x, half *y,
                          int rows, int cols, int qtype, cudaStream_t stream);
extern int bread_benchmark_expert_block(const bread_model_config_t *cfg,
                                        const loader_t *L,
                                        const weight_cache_t *wc);

/* expert_profile.cu */
extern int bread_profile_gpu_experts(const bread_model_config_t *cfg,
                                     const loader_t *L,
                                     const weight_cache_t *wc);

/* ------------------------------------------------------------------ */
/* Q4K constants (must match kernels.cu)                               */
/* ------------------------------------------------------------------ */
#define Q4K_BLOCK_BYTES   144
#define Q4K_BLOCK_ELEMS   256
#define QTYPE_Q4_K        12
#define QTYPE_Q6_K        14


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
        float d    = bread_h2f(d_raw);
        float dmin = bread_h2f(dmin_raw);

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
/* compute_rms_host: compute RMS norm of half tensor (on host)        */
/* ================================================================== */
static float compute_rms_host(const half *d_tensor, int n,
                              half *h_buf, cudaStream_t stream)
{
    /* Copy to host */
    CUDA_CHECK(cudaMemcpyAsync(h_buf, d_tensor, (size_t)n * sizeof(half),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* Compute RMS on host */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float f = __half2float(h_buf[i]);
        sum += f * f;
    }
    return sqrtf(sum / n);
}

/* ------------------------------------------------------------------ */
/* embed_token: copy one token's embedding row to d_hidden (VRAM)     */
/*                                                                      */
/* Supports Q4_K (primary) and F16 (fallback) embedding types.       */
/* h_emb_row: caller-allocated host buffer of BREAD_HIDDEN_DIM halfs. */
/* ------------------------------------------------------------------ */
static void embed_token(int32_t token_id,
                         const bread_model_config_t *cfg,
                         const loader_t *L, const gguf_ctx_t *g,
                         half *d_hidden, half *h_emb_row)
{
    const gguf_tensor_t *et = gguf_find_tensor(g, "token_embd.weight");
    if (!et) { fprintf(stderr, "token_embd.weight not found\n"); exit(1); }

    const uint8_t *emb_base = L->pinned_data + L->data_offset + et->offset;

    if (et->type == GGML_TYPE_Q4_K) {
        size_t row_bytes = (size_t)(cfg->hidden_dim / Q4K_BLOCK_ELEMS)
                           * Q4K_BLOCK_BYTES;
        dequant_q4k_row(emb_base + (size_t)token_id * row_bytes,
                         h_emb_row, cfg->hidden_dim);
    } else if (et->type == GGML_TYPE_F16) {
        memcpy(h_emb_row,
               emb_base + (size_t)token_id * cfg->hidden_dim * sizeof(half),
               (size_t)cfg->hidden_dim * sizeof(half));
    } else {
        fprintf(stderr, "embed_token: unsupported type %u\n", et->type);
        exit(1);
    }

    CUDA_CHECK(cudaMemcpy(d_hidden, h_emb_row,
                           cfg->hidden_dim * sizeof(half),
                           cudaMemcpyHostToDevice));
}

static char *format_prompt_for_model(const tokenizer_t *tok, const char *prompt)
{
    const char *pre = tokenizer_pre(tok);
    size_t n;
    char *wrapped;

    if (!pre || strcmp(pre, "qwen35") != 0) {
        return _strdup(prompt);
    }
    if (strncmp(prompt, "<|im_start|>", 12) == 0) {
        return _strdup(prompt);
    }

    n = strlen(prompt) + 128;
    wrapped = (char *)malloc(n);
    if (!wrapped) return NULL;
    snprintf(wrapped, n,
             "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
             prompt);
    return wrapped;
}

/* ------------------------------------------------------------------ */
/* apply_output_norm: final RMSNorm in-place with output_norm.weight  */
/*                                                                      */
/* d_norm_w: pre-loaded F32 weights already in VRAM.                  */
/* ------------------------------------------------------------------ */
static void apply_output_norm(const bread_model_config_t *cfg,
                               half *d_hidden, const float *d_norm_w,
                               cudaStream_t stream_a)
{
    rmsnorm_output<<<1, 256, 0, stream_a>>>(
        d_hidden, d_norm_w, cfg->hidden_dim, cfg->rms_eps);
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
}

/* ------------------------------------------------------------------ */
/* compute_logits: lm_head matvec → d_logits[BREAD_VOCAB_SIZE]        */
/*                                                                      */
/* d_output_w: pre-loaded lm_head weights in VRAM.                    */
/* output_w_type: GGML type of d_output_w.                            */
/* ------------------------------------------------------------------ */
static void compute_logits(const bread_model_config_t *cfg,
                             half *d_hidden,
                             void *d_output_w, uint32_t output_w_type,
                             half *d_logits, cudaStream_t stream_a)
{
    if (output_w_type == GGML_TYPE_Q4_K) {
        bread_matvec(d_output_w, d_hidden, d_logits,
                     cfg->vocab_size, cfg->hidden_dim, QTYPE_Q4_K, stream_a);
    } else if (output_w_type == GGML_TYPE_Q6_K) {
        bread_matvec(d_output_w, d_hidden, d_logits,
                     cfg->vocab_size, cfg->hidden_dim, QTYPE_Q6_K, stream_a);
    } else if (output_w_type == GGML_TYPE_F16) {
        f16_matvec<<<cfg->vocab_size, 256, 0, stream_a>>>(
            (const half *)d_output_w, d_hidden, d_logits,
            cfg->vocab_size, cfg->hidden_dim);
    } else {
        fprintf(stderr, "compute_logits: unsupported weight type %u\n",
                output_w_type);
        exit(1);
    }
    /* Stream-scoped sync instead of global device sync (bread_matvec uses stream_a) */
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
}

/* ================================================================== */
/* greedy_sample: copy d_logits to host, return argmax token id       */
/* ================================================================== */
static int32_t greedy_sample(const half *d_logits, half *h_logits)
{
    const bread_model_config_t *cfg = bread_model_config_get();
    CUDA_CHECK(cudaMemcpy(h_logits, d_logits,
                           (size_t)cfg->vocab_size * sizeof(half),
                           cudaMemcpyDeviceToHost));
    int32_t best_id  = 0;
    float   best_val = __half2float(h_logits[0]);
    for (int i = 1; i < cfg->vocab_size; i++) {
        float v = __half2float(h_logits[i]);
        if (v > best_val) { best_val = v; best_id = (int32_t)i; }
    }
    return best_id;
}

static void print_top5_logits(const half *h_logits, int vocab_size)
{
    int ids[5] = {0, 0, 0, 0, 0};
    float vals[5] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};

    for (int i = 0; i < vocab_size; i++) {
        float v = __half2float(h_logits[i]);
        for (int k = 0; k < 5; k++) {
            if (v > vals[k]) {
                for (int s = 4; s > k; s--) {
                    vals[s] = vals[s - 1];
                    ids[s] = ids[s - 1];
                }
                vals[k] = v;
                ids[k] = i;
                break;
            }
        }
    }

    fprintf(stderr, "Top-5 logits:");
    for (int k = 0; k < 5; k++) {
        fprintf(stderr, " [%d]=%.5f", ids[k], vals[k]);
    }
    fprintf(stderr, "\n");
}

/* ================================================================== */
/*  main                                                                */
/* ================================================================== */

int main(int argc, char **argv)
{
    const char *model_path = BREAD_MODEL_PATH;
    const char *prompt     = "Hello, I am";
    int         max_tokens = 50;
    int         minimal_mode = 0;
    int         debug_rms   = 0;
    int         force_ssm_zero = 0;
    int         disable_rope = 0;
    int         prefetch_mode = 0;
    int         ssd_streaming_mode = 0;
    int         cpu_experts_mode = 0;
    int         server_mode = 0;
    int         hooks_debug = 0;
    int         no_progress = 0;
    int         bench_experts = 0;
    int         profile_gpu_experts = 0;

    /* -- Parse args ------------------------------------------------- */
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--prompt") && i+1 < argc) prompt     = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i+1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--model")  && i+1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--minimal")) minimal_mode = 1;
        else if (!strcmp(argv[i], "--boring"))  minimal_mode = 1;
        else if (!strcmp(argv[i], "--debug"))   debug_rms = 1;
        else if (!strcmp(argv[i], "--force-ssm-zero")) force_ssm_zero = 1;
        else if (!strcmp(argv[i], "--disable-rope"))   disable_rope = 1;
        else if (!strcmp(argv[i], "--prefetch")) prefetch_mode = 1;
        else if (!strcmp(argv[i], "--ssd-streaming")) ssd_streaming_mode = 1;
        else if (!strcmp(argv[i], "--cpu-experts")) cpu_experts_mode = 1;
        else if (!strcmp(argv[i], "--server"))  server_mode = 1;
        else if (!strcmp(argv[i], "--hooks-debug")) hooks_debug = 1;
        else if (!strcmp(argv[i], "--no-progress")) no_progress = 1;
        else if (!strcmp(argv[i], "--bench-experts")) bench_experts = 1;
        else if (!strcmp(argv[i], "--profile-gpu-experts")) profile_gpu_experts = 1;
    }

    bread_set_boring_mode(minimal_mode);
    bread_set_force_ssm_zero(force_ssm_zero);
    bread_set_disable_rope(disable_rope);
    bread_set_trace_debug(debug_rms);
    bread_set_trace_pos(-1);
    bread_set_prefetch_mode(prefetch_mode);
    bread_set_ssd_streaming_mode(ssd_streaming_mode);
    bread_set_cpu_experts_mode(cpu_experts_mode);

    /* Initialize progress tracking with default callback */
    if (!no_progress) {
        bread_progress_init_default();
    }

    /* Enable built-in hooks if requested */
    if (hooks_debug) {
        bread_hooks_enable_layer_timing();
    }

    if (!server_mode) {
        printf("=== BREAD inference ===\n");
        printf("Prompt   : \"%s\"\n", prompt);
        printf("MaxTokens: %d\n\n", max_tokens);
        printf("Mode     : %s\n\n", minimal_mode ? "minimal-core" : "orchestrated");
        if (force_ssm_zero) printf("Experiment: force SSM branch output to zero\n");
        if (disable_rope)   printf("Experiment: disable RoPE\n");
        if (debug_rms)      printf("Trace    : per-layer hidden RMS + branch RMS + top-5 logits\n");
        if (force_ssm_zero || disable_rope || debug_rms) printf("\n");
    }

    /* -- Load model into pinned RAM --------------------------------- */
    if (!server_mode) printf("[1/4] Loading model into pinned RAM (~22 GB)...\n");
    double t0 = now_ms();
    loader_t *L = loader_init(model_path);
    if (!L) { fprintf(stderr, "loader_init failed\n"); return 1; }
    if (!server_mode) printf("      done in %.1f s\n\n", (now_ms() - t0) / 1000.0);

    /* -- Open GGUF for metadata ------------------------------------- */
    gguf_ctx_t *g = gguf_open(model_path);
    if (!g) { fprintf(stderr, "gguf_open failed\n"); return 1; }
    if (bread_model_config_init(model_path, g) != 0) {
        fprintf(stderr, "bread_model_config_init failed\n");
        return 1;
    }
    const bread_model_config_t *cfg = bread_model_config_get();

    /* -- Pre-load non-expert weights to VRAM cache ------------------- */
    if (!server_mode) printf("      Initializing weight cache...\n");
    weight_cache_t *wc = weight_cache_init(L, g, cfg->num_layers,
                                           bread_layer_is_full_attention);
    if (!wc) { fprintf(stderr, "weight_cache_init failed\n"); return 1; }

    /* -- Pre-load all expert weights to VRAM cache ------------------- */
    /* In default mode: pre-cache all experts at startup for fast access
       In SSD streaming mode: skip pre-cache, load experts on-demand for DMA overlap */
    if (!ssd_streaming_mode && !cpu_experts_mode) {
        if (!server_mode) printf("      Loading expert weights to VRAM...\n");
        if (weight_cache_load_experts(wc, L, cfg->num_layers) != 0) {
            fprintf(stderr, "weight_cache_load_experts failed\n");
            return 1;
        }
    } else {
        if (!server_mode) {
            if (cpu_experts_mode) {
                printf("      CPU experts mode: routed experts run from host RAM\n");
            } else {
                printf("      SSD streaming mode: expert weights loaded on-demand\n");
            }
        }
    }

    /* -- Pre-allocate layer computation buffers ---------------------- */
    if (!server_mode) printf("      Initializing layer buffer pool...\n");
    if (bread_buffer_pool_init(cfg) != 0) {
        fprintf(stderr, "bread_buffer_pool_init failed\n");
        return 1;
    }

    if (bench_experts) {
        int rc = bread_benchmark_expert_block(cfg, L, wc);
        bread_buffer_pool_free();
        weight_cache_free(wc);
        loader_free(L);
        return rc;
    }

    if (profile_gpu_experts) {
        int rc = bread_profile_gpu_experts(cfg, L, wc);
        bread_buffer_pool_free();
        weight_cache_free(wc);
        loader_free(L);
        return rc;
    }

    /* -- Load tokenizer --------------------------------------------- */
    if (!server_mode) printf("[2/4] Loading tokenizer...\n");
    tokenizer_t *tok = tokenizer_load(model_path);
    if (!tok) { fprintf(stderr, "tokenizer_load failed\n"); return 1; }
    if (!server_mode) printf("      vocab size: %d\n\n", tokenizer_vocab_size(tok));

    /* -- Pre-load output_norm.weight and output.weight to VRAM ------ */
    if (!server_mode) printf("[3/4] Uploading output_norm + lm_head to VRAM...\n");

    const gguf_tensor_t *norm_t = gguf_find_tensor(g, "output_norm.weight");
    if (!norm_t) { fprintf(stderr, "output_norm.weight not found\n"); return 1; }
    float *d_norm_w = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_norm_w, norm_t->size));
    CUDA_CHECK(cudaMemcpy(d_norm_w,
                           L->pinned_data + L->data_offset + norm_t->offset,
                           norm_t->size, cudaMemcpyHostToDevice));
    if (!server_mode) printf("      output_norm.weight: %.1f KB (F32)\n",
           norm_t->size / 1024.0);

    /* Try output.weight; fall back to token_embd.weight (tied) */
    const gguf_tensor_t *out_t = gguf_find_tensor(g, "output.weight");
    if (!out_t) {
        if (!server_mode) printf("      output.weight not found — using token_embd.weight (tied)\n");
        out_t = gguf_find_tensor(g, "token_embd.weight");
    }
    if (!out_t) { fprintf(stderr, "no lm_head weight found\n"); return 1; }
    void *d_output_w = NULL;
    CUDA_CHECK(cudaMalloc(&d_output_w, out_t->size));
    CUDA_CHECK(cudaMemcpy(d_output_w,
                           L->pinned_data + L->data_offset + out_t->offset,
                           out_t->size, cudaMemcpyHostToDevice));
    if (!server_mode) printf("      output.weight:      %.1f MB (type %s)\n",
           out_t->size / 1024.0 / 1024.0,
           ggml_type_name(out_t->type));

    /* -- CUDA resources --------------------------------------------- */
    cudaStream_t stream_a;
    CUDA_CHECK(cudaStreamCreate(&stream_a));

    half *d_hidden = NULL;
    half *d_logits = NULL;
    CUDA_CHECK(cudaMalloc(&d_hidden, cfg->hidden_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_logits,
                           (size_t)cfg->vocab_size * sizeof(half)));

    /* Host buffers reused across tokens */
    half *h_emb_row = (half *)malloc(cfg->hidden_dim * sizeof(half));
    half *h_logits  = (half *)malloc((size_t)cfg->vocab_size * sizeof(half));
    if (!h_emb_row || !h_logits) {
        fprintf(stderr, "malloc failed\n"); return 1;
    }

    /* -- Server mode initialization --------------------------------- */
    if (server_mode) {
        printf("BREAD_READY\n");
        fflush(stdout);
    }

    char stdin_buf[8192];
    if (!server_mode) {
        /* Single-mode: use argv prompt */
        strncpy(stdin_buf, prompt, sizeof(stdin_buf) - 1);
        stdin_buf[sizeof(stdin_buf) - 1] = '\0';
    }

    /* -- Server/single-prompt loop --------------------------------- */
    do {
        if (server_mode) {
            /* Read full multi-line prompt until double newline (empty line) */
            int pos = 0;
            int empty_lines = 0;
            while (pos < (int)sizeof(stdin_buf) - 1) {
                int c = fgetc(stdin);
                if (c == EOF) {
                    stdin_buf[pos] = '\0';
                    empty_lines = 2;  /* Force exit */
                    break;
                }
                stdin_buf[pos++] = (char)c;

                /* Track consecutive newlines to detect double-newline (empty line) */
                if (c == '\n') {
                    empty_lines++;
                    if (empty_lines == 2) {
                        /* Double newline found; remove both trailing newlines */
                        pos -= 2;
                        stdin_buf[pos] = '\0';
                        break;
                    }
                } else if (c != '\n') {
                    empty_lines = 0;
                }
            }
            if (pos == 0) break;  /* EOF or error */
        }
        prompt = stdin_buf;

        /* -- Encode prompt ---------------------------------------------- */
        if (!server_mode) {
            printf("\n[4/4] Encoding prompt...\n");
        }
        int32_t token_buf[4096];
        int32_t bos = tokenizer_bos(tok);
        int n_prompt;
        char *model_prompt = format_prompt_for_model(tok, prompt);
        if (!model_prompt) { fprintf(stderr, "prompt formatting failed\n"); return 1; }
        if (bos >= 0) {
            token_buf[0] = bos;
            n_prompt = 1 + tokenizer_encode(tok, model_prompt, token_buf + 1, 4095);
        } else {
            /* No BOS token for this model (Qwen3.5 style) — encode prompt directly */
            n_prompt = tokenizer_encode(tok, model_prompt, token_buf, 4096);
        }
        if (!server_mode) {
            printf("      %d tokens: [", n_prompt);
            for (int i = 0; i < n_prompt && i < 8; i++)
                printf("%s%d", i ? " " : "", token_buf[i]);
            if (n_prompt > 8) printf(" ...");
            printf("]\n\n");
        }
        if (debug_rms && n_prompt > 0) bread_set_trace_pos(n_prompt - 1);
        free(model_prompt);

    /* ================================================================
     * Prompt prefill: feed each prompt token through all 40 layers.
     * The final hidden state of the last token seeds generation.
     * (No KV cache — each token sees only itself; layers apply norms
     *  and MoE FFN correctly.  SSM state is stubbed.)
     * ================================================================ */
    if (!server_mode) printf("--- Prompt prefill (%d tokens) ---\n", n_prompt);
    double t_prefill_start = now_ms();

    for (int p = 0; p < n_prompt; p++) {
        embed_token(token_buf[p], cfg, L, g, d_hidden, h_emb_row);

        if (debug_rms && p == n_prompt - 1) {
            float rms = compute_rms_host(d_hidden, cfg->hidden_dim, h_emb_row, stream_a);
            fprintf(stderr, "Embed  : rms=%f\n", rms);
        }
        for (int layer = 0; layer < cfg->num_layers; layer++)
        {
            bread_fire_hook(BREAD_HOOK_PRE_LAYER, p, layer, d_hidden, 0.0);
            one_layer_forward(d_hidden, layer, p, L, g, wc, stream_a);
            bread_fire_hook(BREAD_HOOK_POST_LAYER, p, layer, d_hidden, 0.0);
            if (debug_rms && p == n_prompt - 1) {
                float hidden_rms = minimal_mode
                    ? one_layer_cpu_hidden_rms(cfg->hidden_dim)
                    : compute_rms_host(d_hidden, cfg->hidden_dim, h_emb_row, stream_a);
                float branch_rms = one_layer_last_branch_rms();
                fprintf(stderr, "Layer %2d [%s]: hidden_rms=%f branch_rms=%f\n",
                        layer,
                        bread_layer_is_full_attention(layer) ? "attn" : "ssm",
                        hidden_rms, branch_rms);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double t_prefill_done = now_ms();
    double prefill_ms = t_prefill_done - t_prefill_start;
    if (!server_mode) printf("    prefill: %.0f ms  (%.1f ms/tok)\n\n",
           prefill_ms, prefill_ms / n_prompt);

    /* ================================================================
     * First token: apply output norm + lm_head on last prompt state.
     * ================================================================ */
    double t_ttft_start = now_ms();

    apply_output_norm(cfg, d_hidden, d_norm_w, stream_a);
    compute_logits(cfg, d_hidden, d_output_w, out_t->type, d_logits, stream_a);
    int32_t next_tok = greedy_sample(d_logits, h_logits);
    if (debug_rms) {
        print_top5_logits(h_logits, cfg->vocab_size);
    }

    double t_ttft_done = now_ms();
    double ttft_ms = t_ttft_done - t_ttft_start;

    /* Stream first token */
    if (!server_mode) printf("--- Generated output ---\n");
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
        double t_token_start = now_ms();
        embed_token(next_tok, cfg, L, g, d_hidden, h_emb_row);

        bread_fire_hook(BREAD_HOOK_PRE_TOKEN, n_prompt + n_gen, -1, d_hidden, 0.0);

        for (int layer = 0; layer < cfg->num_layers; layer++) {
            bread_fire_hook(BREAD_HOOK_PRE_LAYER, n_prompt + n_gen, layer, d_hidden, 0.0);
            one_layer_forward(d_hidden, layer, n_prompt + n_gen, L, g, wc, stream_a);
            bread_fire_hook(BREAD_HOOK_POST_LAYER, n_prompt + n_gen, layer, d_hidden, 0.0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        bread_fire_hook(BREAD_HOOK_PRE_SAMPLE, n_prompt + n_gen, -1, d_hidden, 0.0);
        apply_output_norm(cfg, d_hidden, d_norm_w, stream_a);
        compute_logits(cfg, d_hidden, d_output_w, out_t->type, d_logits, stream_a);
        next_tok = greedy_sample(d_logits, h_logits);
        bread_fire_hook(BREAD_HOOK_POST_SAMPLE, n_prompt + n_gen, -1, d_hidden, 0.0);

        if (next_tok != eos) {
            char *s = tokenizer_decode(tok, &next_tok, 1);
            printf("%s", s);
            fflush(stdout);
            free(s);
        }

        bread_fire_hook(BREAD_HOOK_POST_TOKEN, n_prompt + n_gen, -1, d_hidden, 0.0);

        /* Report progress periodically (every 10 tokens) — skip in server mode */
        if (!server_mode && !no_progress && (n_gen % 10 == 0 || n_gen == max_tokens - 1)) {
            double elapsed = now_ms() - t_gen_start;
            double tok_per_s = (n_gen > 0) ? (double)n_gen / (elapsed / 1000.0) : 0.0;
            bread_progress_report(BREAD_PROGRESS_DECODE, n_gen, max_tokens, elapsed, tok_per_s, -1);
        }

        n_gen++;
    }

    double t_gen_done = now_ms();
    printf("\n");

    if (!server_mode) {
        /* ================================================================
         * Benchmark report (only for single-mode)
         * ================================================================ */
        double decode_ms  = (t_gen_done - t_gen_start) + ttft_ms;
        double tok_per_s  = (n_gen > 0) ? (double)n_gen / (decode_ms / 1000.0) : 0.0;

        printf("\n=== Benchmark ===\n");
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

        /* Report layer timing if hooks were enabled */
        bread_hooks_report_layer_timing();
    } else {
        /* Server mode: print sentinel to signal end of generation */
        printf("BREAD_END\n");
        fflush(stdout);
    }

    } while (server_mode);  /* End of server/single-prompt loop */

    /* -- Cleanup ---------------------------------------------------- */
    free(h_emb_row);
    free(h_logits);
    cudaFree(d_hidden);
    cudaFree(d_logits);
    cudaFree(d_norm_w);
    cudaFree(d_output_w);
    cudaStreamDestroy(stream_a);
    bread_buffer_pool_free();
    weight_cache_free(wc);
    tokenizer_free(tok);
    gguf_close(g);
    loader_free(L);

    return 0;
}
