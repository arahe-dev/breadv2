/* probe_tensors.cu — dump tensor names + types + dims from a GGUF */
#ifdef _WIN32
#  define _CRT_SECURE_NO_WARNINGS
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gguf.h"

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] :
        "C:\\Users\\arahe\\.ollama\\models\\blobs\\"
        "sha256-afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5";

    gguf_ctx_t *g = gguf_open(path);
    if (!g) { fprintf(stderr, "gguf_open failed\n"); return 1; }

    uint64_t n = gguf_num_tensors(g);
    printf("Total tensors: %llu\n\n", (unsigned long long)n);

    /* Print first 80 */
    uint64_t limit = n < 80 ? n : 80;
    for (uint64_t i = 0; i < limit; i++) {
        const gguf_tensor_t *t = gguf_tensor_by_idx(g, i);
        if (!t) continue;
        printf("%-50s  type=%-6s  dims=[", t->name, ggml_type_name(t->type));
        for (uint32_t d = 0; d < t->n_dims; d++)
            printf("%s%llu", d ? ", " : "", (unsigned long long)t->dims[d]);
        printf("]\n");
    }

    /* Print all attn-related from blk.0 */
    printf("\n--- blk.0 tensors ---\n");
    for (uint64_t i = 0; i < n; i++) {
        const gguf_tensor_t *t = gguf_tensor_by_idx(g, i);
        if (!t || strncmp(t->name, "blk.0.", 6) != 0) continue;
        printf("  %-50s  type=%-6s  dims=[", t->name, ggml_type_name(t->type));
        for (uint32_t d = 0; d < t->n_dims; d++)
            printf("%s%llu", d ? ", " : "", (unsigned long long)t->dims[d]);
        printf("]\n");
    }

    gguf_close(g);
    return 0;
}
