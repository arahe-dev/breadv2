/* main_sriracha.cu — Step 1 test driver for SRIRACHA draft model runner.
 *
 * Loads the Qwen3.5 0.8B draft model, runs prefill on a prompt,
 * then generates K=5 draft tokens and prints them with timing.
 *
 * "Done when: sriracha_draft_from() returns 5 coherent tokens."
 *
 * Build: see build_sriracha.ps1
 * Run:   sriracha.exe [--prompt "..."] [--tokens N] [--draft PATH]
 */

#ifdef _WIN32
#  define _CRT_SECURE_NO_WARNINGS
#  include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "sriracha.h"
#include "tokenizer.h"

/* Default path for Qwen3.5 0.8B blob from Ollama */
#define DRAFT_MODEL_PATH \
    "C:\\Users\\arahe\\.ollama\\models\\blobs\\" \
    "sha256-afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5"

static double now_ms(void)
{
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    return 0.0;
#endif
}

int main(int argc, char **argv)
{
    const char *draft_path = DRAFT_MODEL_PATH;
    const char *prompt     = "The capital of France is";
    int         max_draft  = 5;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--draft")  && i+1 < argc) draft_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i+1 < argc) prompt     = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i+1 < argc) max_draft  = atoi(argv[++i]);
    }

    printf("=== SRIRACHA Step 1 Test ===\n");
    printf("Draft model : %s\n", draft_path);
    printf("Prompt      : \"%s\"\n", prompt);
    printf("Draft K     : %d\n\n", max_draft);

    /* Load tokenizer */
    printf("[1/3] Loading tokenizer...\n");
    tokenizer_t *tok = tokenizer_load(draft_path);
    if (!tok) { fprintf(stderr, "tokenizer_load failed\n"); return 1; }
    printf("      vocab=%d\n\n", tokenizer_vocab_size(tok));

    /* Encode prompt */
    int32_t token_buf[4096];
    int n_prompt = tokenizer_encode(tok, prompt, token_buf, 4096);
    printf("      %d tokens: [", n_prompt);
    for (int i = 0; i < n_prompt && i < 8; i++)
        printf("%s%d", i ? " " : "", token_buf[i]);
    if (n_prompt > 8) printf(" ...");
    printf("]\n\n");

    /* Init SRIRACHA */
    printf("[2/3] Initializing SRIRACHA draft runner...\n");
    double t0 = now_ms();
    sriracha_t *sr = sriracha_init(draft_path, max_draft);
    if (!sr) { fprintf(stderr, "sriracha_init failed\n"); return 1; }
    printf("      init: %.1f s\n\n", (now_ms() - t0) / 1000.0);

    /* Prefill */
    printf("[3/3] Prefill + draft generation...\n");
    double t_prefill = now_ms();
    sriracha_prefill(sr, token_buf, n_prompt);
    double t_prefill_done = now_ms();
    printf("      prefill: %.0f ms (%d tokens, %.1f ms/tok)\n",
           t_prefill_done - t_prefill,
           n_prompt,
           (t_prefill_done - t_prefill) / n_prompt);

    /* Draft K tokens */
    int32_t draft_tokens[16];
    int32_t seed = token_buf[n_prompt - 1];

    double t_draft = now_ms();
    int n = sriracha_draft_from(sr, seed, n_prompt, draft_tokens);
    double t_draft_done = now_ms();

    double draft_ms = t_draft_done - t_draft;
    printf("      drafted %d tokens in %.1f ms (%.2f ms/tok)\n\n",
           n, draft_ms, draft_ms / n);

    /* Decode and print draft tokens */
    printf("--- Draft tokens ---\n");
    printf("Prompt: \"%s\"\n", prompt);
    printf("Draft : \"");
    for (int i = 0; i < n; i++) {
        char *s = tokenizer_decode(tok, &draft_tokens[i], 1);
        printf("%s", s);
        free(s);
    }
    printf("\"\n\n");

    printf("--- Token IDs ---\n");
    for (int i = 0; i < n; i++)
        printf("  [%d] id=%d\n", i, draft_tokens[i]);
    printf("\n");

    /* Throughput */
    printf("=== Summary ===\n");
    printf("  Draft tok/s  : %.2f\n", (double)n / (draft_ms / 1000.0));
    printf("  Target (BREAD): ~5.5 tok/s\n");
    printf("  Expected speedup with spec-decode: ~3x (if acceptance >= 60%%)\n");

    sriracha_free(sr);
    tokenizer_free(tok);
    return 0;
}
