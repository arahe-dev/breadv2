/* bread_info.c — dump every tensor in a GGUF model file.
 *
 * Usage:  bread_info <path-to-gguf-file>
 *
 * Prints: index, name, type, shape, absolute file offset, byte size.
 * At the end: per-type counts and total size, plus MoE expert tensor count.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gguf.h"

/* ------------------------------------------------------------------ */
/* Formatting helpers                                                   */
/* ------------------------------------------------------------------ */

static void fmt_size(char *buf, size_t buflen, uint64_t bytes) {
    if      (bytes >= (1ull << 30))
        snprintf(buf, buflen, "%.2f GiB", bytes / (double)(1ull << 30));
    else if (bytes >= (1ull << 20))
        snprintf(buf, buflen, "%.2f MiB", bytes / (double)(1ull << 20));
    else if (bytes >= (1ull << 10))
        snprintf(buf, buflen, "%.2f KiB", bytes / (double)(1ull << 10));
    else
        snprintf(buf, buflen, "%llu B", (unsigned long long)bytes);
}

static void fmt_shape(char *buf, size_t buflen, const gguf_tensor_t *t) {
    int off = snprintf(buf, buflen, "[");
    for (uint32_t d = 0; d < t->n_dims && (size_t)off < buflen - 1; d++) {
        off += snprintf(buf + off, buflen - (size_t)off,
                        "%llu", (unsigned long long)t->dims[d]);
        if (d + 1 < t->n_dims)
            off += snprintf(buf + off, buflen - (size_t)off, " x ");
    }
    snprintf(buf + off, buflen - (size_t)off, "]");
}

/* ------------------------------------------------------------------ */
/* Per-type accumulator for summary                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t type;
    uint64_t count;
    uint64_t bytes;
} type_stats_t;

static int stats_cmp(const void *a, const void *b) {
    const type_stats_t *ta = (const type_stats_t *)a;
    const type_stats_t *tb = (const type_stats_t *)b;
    if (tb->bytes > ta->bytes) return  1;
    if (tb->bytes < ta->bytes) return -1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: bread_info <gguf-path>\n");
        return 1;
    }

    gguf_ctx_t *ctx = gguf_open(argv[1]);
    if (!ctx) return 1;

    uint64_t n        = gguf_num_tensors(ctx);
    uint64_t data_off = gguf_data_offset(ctx);

    /* --- File summary header --- */
    char data_off_sz[32];
    fmt_size(data_off_sz, sizeof(data_off_sz), data_off);

    printf("\n");
    printf("  File      : %s\n",   argv[1]);
    printf("  GGUF ver  : %u\n",   gguf_version(ctx));
    printf("  Tensors   : %llu\n", (unsigned long long)n);
    printf("  Data at   : 0x%llx  (%s header)\n\n",
           (unsigned long long)data_off, data_off_sz);

    /* --- Column headers --- */
    /*  idx(6)  name(52)  type(8)  shape(28)  file_offset(18)  size(12) */
    printf("%-6s  %-52s  %-8s  %-28s  %-18s  %s\n",
           "idx", "name", "type", "shape", "file_offset", "size");
    printf("%-6s  %-52s  %-8s  %-28s  %-18s  %s\n",
           "------",
           "----------------------------------------------------",
           "--------",
           "----------------------------",
           "------------------",
           "------------");

    /* --- One row per tensor --- */
    uint64_t      total_bytes = 0;
    uint64_t      n_moe       = 0;
    type_stats_t  stats[GGML_TYPE_COUNT];
    memset(stats, 0, sizeof(stats));
    for (int i = 0; i < GGML_TYPE_COUNT; i++) stats[i].type = (uint32_t)i;

    for (uint64_t i = 0; i < n; i++) {
        const gguf_tensor_t *t = gguf_tensor_by_idx(ctx, i);

        char shape[64], sz[24];
        fmt_shape(shape, sizeof(shape), t);
        fmt_size(sz, sizeof(sz), t->size);

        uint64_t abs_off = data_off + t->offset;

        printf("%6llu  %-52s  %-8s  %-28s  0x%016llx  %s\n",
               (unsigned long long)i,
               t->name,
               ggml_type_name(t->type),
               shape,
               (unsigned long long)abs_off,
               sz);

        total_bytes += t->size;

        if (strstr(t->name, "_exps")) n_moe++;

        if (t->type < GGML_TYPE_COUNT) {
            stats[t->type].count++;
            stats[t->type].bytes += t->size;
        }
    }

    /* --- Totals --- */
    char total_sz[32];
    fmt_size(total_sz, sizeof(total_sz), total_bytes);
    printf("\n");
    printf("  %-30s : %s  (%llu bytes)\n",
           "Total tensor data",
           total_sz,
           (unsigned long long)total_bytes);
    printf("  %-30s : %llu\n",
           "MoE expert tensors (*_exps)",
           (unsigned long long)n_moe);

    /* --- Per-type breakdown --- */
    /* Sort by descending byte count, print only used types */
    qsort(stats, GGML_TYPE_COUNT, sizeof(type_stats_t), stats_cmp);

    printf("\n  By type:\n");
    printf("    %-10s  %8s  %14s\n", "type", "tensors", "total size");
    printf("    %-10s  %8s  %14s\n", "----------", "-------", "--------------");
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        if (stats[i].count == 0) break;
        char tsz[24];
        fmt_size(tsz, sizeof(tsz), stats[i].bytes);
        printf("    %-10s  %8llu  %14s\n",
               ggml_type_name(stats[i].type),
               (unsigned long long)stats[i].count,
               tsz);
    }
    printf("\n");

    gguf_close(ctx);
    return 0;
}
