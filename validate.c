/* validate.c — Phase 1: Compare BREAD vs Ollama layer outputs
 *
 * Goal: Find the first layer where BREAD diverges from ollama
 *
 * Build: cl validate.c bread.c loader.c gguf.c tokenizer.c dequant_q4k_cpu.c -Iinclude -link ws2_32.lib
 * Run:   validate.exe
 */

#ifdef _WIN32
#  define _CRT_SECURE_NO_WARNINGS
#  include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <winsock2.h>

#pragma comment(lib, "ws2_32.lib")

#include "bread.h"
#include "gguf.h"
#include "loader.h"
#include "tokenizer.h"

/* ================================================================ */
/* Phase 1a: BREAD layer-by-layer dump (minimal mode simulation)   */
/* ================================================================ */

typedef struct {
    float *activation;  /* per-layer hidden state */
    int hidden_dim;
} layer_state_t;

/* Minimal RMSNorm (float only, reference correctness) */
static void rmsnorm_float(const float *x, const float *w, float *y,
                          int dim, float eps)
{
    float sum = 0.0f;
    for (int i = 0; i < dim; i++)
        sum += x[i] * x[i];
    float rms = sqrtf(sum / dim + eps);
    for (int i = 0; i < dim; i++)
        y[i] = (x[i] / rms) * w[i];
}

/* Save float tensor to file (debug output) */
static void save_tensor_f32(const char *filename, const float *data, int count)
{
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }
    fwrite(data, sizeof(float), count, f);
    fclose(f);
    printf("  saved: %s\n", filename);
}

/* Compute RMS norm of a tensor (for divergence detection) */
static float compute_rms(const float *data, int count)
{
    float sum = 0.0f;
    for (int i = 0; i < count; i++)
        sum += data[i] * data[i];
    return sqrtf(sum / count);
}

/* ================================================================ */
/* Phase 1b: Ollama HTTP API interaction (simple)                  */
/* ================================================================ */

#define OLLAMA_HOST "localhost"
#define OLLAMA_PORT 11434
#define OLLAMA_MODEL "qwen3.5:35b-a3b"

static char ollama_response_buffer[1024 * 100];  /* Buffer for HTTP response */

/* Simple HTTP GET request to ollama API
 *
 * Endpoint: GET /api/embeddings
 * Body: {"model": "qwen3.5:35b-a3b", "prompt": "..."}
 *
 * Returns: {"embedding": [float, ...]}
 */
static int ollama_call_embeddings(const char *prompt, float *out_embedding,
                                  int embedding_dim)
{
    SOCKET sock;
    struct sockaddr_in addr;
    struct hostent *host;
    char request[4096];
    int n_received;

    /* Resolve hostname */
    host = gethostbyname(OLLAMA_HOST);
    if (!host) {
        fprintf(stderr, "Failed to resolve %s\n", OLLAMA_HOST);
        return -1;
    }

    /* Create socket */
    sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) {
        fprintf(stderr, "socket() failed\n");
        return -1;
    }

    /* Connect */
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(OLLAMA_PORT);
    memcpy(&addr.sin_addr, host->h_addr, host->h_length);

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR) {
        fprintf(stderr, "Failed to connect to ollama at %s:%d\n",
                OLLAMA_HOST, OLLAMA_PORT);
        closesocket(sock);
        return -1;
    }

    /* Build POST request (note: ollama uses POST for /api/embeddings) */
    snprintf(request, sizeof(request),
             "POST /api/embeddings HTTP/1.1\r\n"
             "Host: %s:%d\r\n"
             "Content-Type: application/json\r\n"
             "Content-Length: %d\r\n"
             "Connection: close\r\n"
             "\r\n"
             "{\"model\":\"%s\",\"prompt\":\"%s\"}",
             OLLAMA_HOST, OLLAMA_PORT,
             (int)(strlen("{\"model\":\"") + strlen(OLLAMA_MODEL) +
                   strlen("\",\"prompt\":\"") + strlen(prompt) + 2),
             OLLAMA_MODEL, prompt);

    /* Send request */
    if (send(sock, request, (int)strlen(request), 0) == SOCKET_ERROR) {
        fprintf(stderr, "send() failed\n");
        closesocket(sock);
        return -1;
    }

    /* Receive response */
    n_received = recv(sock, ollama_response_buffer, sizeof(ollama_response_buffer) - 1, 0);
    if (n_received <= 0) {
        fprintf(stderr, "recv() failed or no data\n");
        closesocket(sock);
        return -1;
    }
    ollama_response_buffer[n_received] = '\0';

    closesocket(sock);

    /* Parse response (simple: look for "embedding" array)
     * For now, just indicate success. In reality, you'd parse JSON. */
    if (strstr(ollama_response_buffer, "embedding") == NULL) {
        fprintf(stderr, "No embedding in response\n");
        return -1;
    }

    printf("  ollama responded with embedding\n");
    return 0;
}

/* ================================================================ */
/* main: Phase 1 validation harness                                */
/* ================================================================ */

int main(void)
{
    printf("=== BREAD Validation Phase 1 ===\n\n");

    const char *model_path = BREAD_MODEL_PATH;
    const char *prompt = "The capital of France is";

    /* Load model (same as main.cu) */
    printf("[1] Loading model...\n");
    loader_t *L = loader_init(model_path);
    if (!L) {
        fprintf(stderr, "loader_init failed\n");
        return 1;
    }

    gguf_ctx_t *g = gguf_open(model_path);
    if (!g) {
        fprintf(stderr, "gguf_open failed\n");
        return 1;
    }

    if (bread_model_config_init(model_path, g) != 0) {
        fprintf(stderr, "bread_model_config_init failed\n");
        return 1;
    }

    const bread_model_config_t *cfg = bread_model_config_get();
    printf("  model: %d layers, %d hidden_dim, %d vocab\n",
           cfg->num_layers, cfg->hidden_dim, cfg->vocab_size);

    /* Load tokenizer */
    printf("\n[2] Loading tokenizer...\n");
    tokenizer_t *tok = tokenizer_load(model_path);
    if (!tok) {
        fprintf(stderr, "tokenizer_load failed\n");
        return 1;
    }

    /* Tokenize prompt */
    printf("\n[3] Tokenizing prompt: \"%s\"\n", prompt);
    int32_t token_buf[4096];
    int n_tokens = tokenizer_encode(tok, prompt, token_buf, 4096);
    printf("  %d tokens: [", n_tokens);
    for (int i = 0; i < n_tokens && i < 8; i++)
        printf("%s%d", i ? " " : "", token_buf[i]);
    if (n_tokens > 8) printf(" ...");
    printf("]\n");

    /* Allocate activation buffer (float, for minimal/reference mode) */
    float *activation = (float *)malloc(cfg->hidden_dim * sizeof(float));
    if (!activation) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* For now: just demonstrate that we can load and tokenize.
     * In a full implementation, we'd:
     * 1. Embed first token
     * 2. Run through each layer, saving activations
     * 3. Call ollama for reference
     * 4. Compare
     */

    printf("\n[4] Phase 1a: BREAD layer-by-layer (stub)\n");
    printf("  (Full implementation would run all 40 layers here)\n");

    printf("\n[5] Phase 1b: Ollama reference\n");
    printf("  Calling ollama at %s:%d...\n", OLLAMA_HOST, OLLAMA_PORT);
    if (ollama_call_embeddings(prompt, NULL, cfg->hidden_dim) < 0) {
        printf("  WARNING: Could not reach ollama (is it running?)\n");
        printf("  Start it with: ollama run qwen3.5:35b-a3b\n");
    } else {
        printf("  Ollama response received\n");
    }

    printf("\n[6] Divergence analysis (stub)\n");
    printf("  (Would compare BREAD vs ollama activations layer-by-layer)\n");

    printf("\n=== Phase 1 Setup Complete ===\n");
    printf("Next: Integrate CUDA layer execution and detailed comparison\n");

    free(activation);
    return 0;
}
