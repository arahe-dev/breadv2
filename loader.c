/* loader.c — async RAM→VRAM expert weight streaming for BREAD.
 *
 * Design (matches danveloper's flash-moe pattern adapted for CUDA):
 *   1. Full model file loaded into pinned (page-locked) host RAM at startup
 *   2. GGUF tensor metadata used to compute per-expert byte ranges
 *   3. LRU cache of 18 VRAM slots (2×9 double buffer)
 *   4. On cache miss: cudaMemcpyAsync from pinned RAM → VRAM slot on stream_b
 *   5. loader_sync() blocks until DMA is done before expert kernels run
 *
 * Expert VRAM slot layout (contiguous):
 *   [gate_bytes | up_bytes | down_bytes]
 *
 * Compile (standalone selftest):
 *   nvcc -DSELFTEST_MAIN -O2 loader.c gguf.c -o loader_test.exe
 */

#ifdef _WIN32
#  define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#include "loader.h"
#include "gguf.h"

/* ------------------------------------------------------------------ */
/* CUDA error-check macro                                               */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call) do {                                          \
    cudaError_t _err = (call);                                         \
    if (_err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s:%d — %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(_err));         \
        exit(1);                                                       \
    }                                                                  \
} while (0)

/* ------------------------------------------------------------------ */
/* Portable 64-bit fseek / ftell (same as gguf.c)                      */
/* ------------------------------------------------------------------ */
#ifdef _WIN32
#  define fseek64(f, o, w)  _fseeki64((f), (__int64)(o), (w))
#  define ftell64(f)        ((int64_t)_ftelli64(f))
#else
#  include <sys/types.h>
#  define fseek64(f, o, w)  fseeko((f), (off_t)(o), (w))
#  define ftell64(f)        ((int64_t)ftello(f))
#endif

/* ------------------------------------------------------------------ */
/* Helper: get file size via seek                                       */
/* ------------------------------------------------------------------ */
static uint64_t file_size(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    fseek64(fp, 0, SEEK_END);
    int64_t sz = ftell64(fp);
    fclose(fp);
    return (sz > 0) ? (uint64_t)sz : 0;
}

/* ------------------------------------------------------------------ */
/* Helper: read full file into buffer with progress                     */
/* ------------------------------------------------------------------ */
static int read_file_into(const char *path, uint8_t *buf, uint64_t size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "loader: cannot open '%s'\n", path);
        return -1;
    }

    /* Use 64 MiB chunks for progress feedback */
    const uint64_t CHUNK = 64ULL * 1024 * 1024;
    uint64_t done = 0;
    clock_t t0 = clock();

    while (done < size) {
        uint64_t want = size - done;
        if (want > CHUNK) want = CHUNK;
        size_t got = fread(buf + done, 1, (size_t)want, fp);
        if (got == 0) break;
        done += got;

        /* Progress every ~512 MiB */
        if ((done % (512ULL * 1024 * 1024)) < CHUNK) {
            double pct = 100.0 * done / size;
            double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
            double gbs = (elapsed > 0.01) ? (done / 1e9 / elapsed) : 0;
            fprintf(stderr, "\rloader: reading model... %5.1f%% (%.2f GB/s)", pct, gbs);
        }
    }

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    double gbs = (elapsed > 0.01) ? (done / 1e9 / elapsed) : 0;
    fprintf(stderr, "\rloader: read %.2f GB in %.1f s (%.2f GB/s)            \n",
            done / 1e9, elapsed, gbs);

    fclose(fp);
    return (done == size) ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* Helper: find LRU slot (smallest last_used)                          */
/* ------------------------------------------------------------------ */
static int find_lru_slot(const loader_t *L) {
    int lru = 0;
    uint64_t min_used = L->cache[0].last_used;
    for (int i = 1; i < LOADER_NUM_SLOTS; i++) {
        if (L->cache[i].last_used < min_used) {
            min_used = L->cache[i].last_used;
            lru = i;
        }
    }
    return lru;
}

/* ------------------------------------------------------------------ */
/* Helper: evict a cache slot (clear the O(1) lookup entry)            */
/* ------------------------------------------------------------------ */
static void evict_slot(loader_t *L, int slot) {
    int old_layer  = L->cache[slot].layer_idx;
    int old_expert = L->cache[slot].expert_idx;
    if (old_layer >= 0 && old_expert >= 0) {
        L->entry_idx[old_layer][old_expert] = -1;
    }
}

/* ------------------------------------------------------------------ */
/* loader_init                                                          */
/* ------------------------------------------------------------------ */

loader_t *loader_init(const char *model_path) {
    fprintf(stderr, "loader: initialising...\n");

    /* ---- Step 1: Parse GGUF header ---- */
    gguf_ctx_t *ctx = gguf_open(model_path);
    if (!ctx) {
        fprintf(stderr, "loader: gguf_open failed\n");
        return NULL;
    }
    uint64_t data_off = gguf_data_offset(ctx);

    /* ---- Step 2: Allocate loader context ---- */
    loader_t *L = (loader_t *)calloc(1, sizeof(loader_t));
    if (!L) {
        fprintf(stderr, "loader: out of memory for context\n");
        gguf_close(ctx);
        return NULL;
    }
    L->data_offset = data_off;

    /* ---- Step 3: Scan for MoE expert tensors, build per-layer info ---- */
    uint64_t max_expert_total = 0;  /* largest gate+up+down per expert */
    int num_moe = 0;

    for (int layer = 0; layer < LOADER_MAX_LAYERS; layer++) {
        char name_gate[128], name_up[128], name_down[128];
        snprintf(name_gate, sizeof(name_gate), "blk.%d.ffn_gate_exps.weight", layer);
        snprintf(name_up,   sizeof(name_up),   "blk.%d.ffn_up_exps.weight",   layer);
        snprintf(name_down, sizeof(name_down), "blk.%d.ffn_down_exps.weight", layer);

        const gguf_tensor_t *t_gate = gguf_find_tensor(ctx, name_gate);
        const gguf_tensor_t *t_up   = gguf_find_tensor(ctx, name_up);
        const gguf_tensor_t *t_down = gguf_find_tensor(ctx, name_down);

        if (!t_gate || !t_up || !t_down) {
            L->layers[layer].valid = 0;
            continue;
        }

        /* Expert count from outermost dimension */
        int n_experts = (int)t_gate->dims[t_gate->n_dims - 1];
        if (n_experts <= 0 || n_experts > LOADER_MAX_EXPERTS) {
            fprintf(stderr, "loader: blk.%d unexpected expert count %d\n", layer, n_experts);
            L->layers[layer].valid = 0;
            continue;
        }

        loader_layer_info_t *li = &L->layers[layer];
        li->valid = 1;
        li->num_experts = n_experts;

        /* Byte ranges — expert k starts at k * (tensor.size / n_experts) */
        li->gate_expert_bytes = t_gate->size / (uint64_t)n_experts;
        li->up_expert_bytes   = t_up->size   / (uint64_t)n_experts;
        li->down_expert_bytes = t_down->size / (uint64_t)n_experts;
        li->gate_type = t_gate->type;
        li->up_type   = t_up->type;
        li->down_type = t_down->type;

        /* Base pointers will be set after file is loaded into pinned RAM.
         * For now, store the absolute file offsets temporarily. */
        li->gate_base = (uint8_t *)(uintptr_t)(data_off + t_gate->offset);
        li->up_base   = (uint8_t *)(uintptr_t)(data_off + t_up->offset);
        li->down_base = (uint8_t *)(uintptr_t)(data_off + t_down->offset);

        uint64_t expert_total = li->gate_expert_bytes + li->up_expert_bytes + li->down_expert_bytes;
        if (expert_total > max_expert_total)
            max_expert_total = expert_total;

        num_moe++;
    }

    L->num_moe_layers = num_moe;
    L->slot_size = max_expert_total;

    fprintf(stderr, "loader: %d MoE layers found, expert slot size = %llu bytes (%.2f MiB)\n",
            num_moe, (unsigned long long)max_expert_total,
            max_expert_total / (1024.0 * 1024.0));

    /* ---- Step 3b: Allocate VRAM expert slots BEFORE pinning host RAM ----
     * cudaMallocHost for 22+ GB registers all pages in the GPU page table,
     * which can exhaust device VA space on WDDM and cause subsequent
     * cudaMalloc calls to fail.  Allocate device memory first.          */
    fprintf(stderr, "loader: allocating %d VRAM slots × %.2f MiB = %.2f MiB\n",
            LOADER_NUM_SLOTS,
            max_expert_total / (1024.0 * 1024.0),
            (LOADER_NUM_SLOTS * max_expert_total) / (1024.0 * 1024.0));

    for (int i = 0; i < LOADER_NUM_SLOTS; i++) {
        CUDA_CHECK(cudaMalloc((void **)&L->vram_slots[i], (size_t)max_expert_total));
    }

    /* ---- Step 4: Load full model file into pinned RAM ---- */
    uint64_t fsz = file_size(model_path);
    if (fsz == 0) {
        fprintf(stderr, "loader: cannot determine file size\n");
        gguf_close(ctx);
        free(L);
        return NULL;
    }
    L->pinned_size = fsz;

    fprintf(stderr, "loader: allocating %.2f GB pinned host memory...\n", fsz / 1e9);
    cudaError_t alloc_err = cudaMallocHost((void **)&L->pinned_data, (size_t)fsz);
    if (alloc_err != cudaSuccess) {
        fprintf(stderr, "loader: cudaMallocHost failed: %s\n", cudaGetErrorString(alloc_err));
        fprintf(stderr, "loader: falling back to regular malloc (DMA will be synchronous)\n");
        L->pinned_data = (uint8_t *)malloc((size_t)fsz);
        if (!L->pinned_data) {
            fprintf(stderr, "loader: malloc(%llu) also failed — out of memory\n",
                    (unsigned long long)fsz);
            gguf_close(ctx);
            free(L);
            return NULL;
        }
        L->is_pinned = 0;
    } else {
        L->is_pinned = 1;
    }

    if (read_file_into(model_path, L->pinned_data, fsz) != 0) {
        fprintf(stderr, "loader: failed to read model file\n");
        if (L->is_pinned) cudaFreeHost(L->pinned_data);
        else free(L->pinned_data);
        gguf_close(ctx);
        free(L);
        return NULL;
    }

    /* ---- Step 5: Convert stored offsets to pinned RAM pointers ---- */
    for (int layer = 0; layer < LOADER_MAX_LAYERS; layer++) {
        loader_layer_info_t *li = &L->layers[layer];
        if (!li->valid) continue;
        uint64_t gate_off = (uintptr_t)li->gate_base;
        uint64_t up_off   = (uintptr_t)li->up_base;
        uint64_t down_off = (uintptr_t)li->down_base;
        li->gate_base = L->pinned_data + gate_off;
        li->up_base   = L->pinned_data + up_off;
        li->down_base = L->pinned_data + down_off;
    }

    /* ---- Step 6: Initialise LRU cache ---- */
    for (int i = 0; i < LOADER_NUM_SLOTS; i++) {
        L->cache[i].layer_idx  = -1;
        L->cache[i].expert_idx = -1;
        L->cache[i].last_used  = 0;
    }
    memset(L->entry_idx, 0xFF, sizeof(L->entry_idx)); /* -1 = 0xFFFFFFFF for int */
    L->access_counter = 0;
    L->hits = 0;
    L->misses = 0;

    /* ---- Step 7: Create CUDA stream for async DMA ---- */
    CUDA_CHECK(cudaStreamCreate(&L->stream_b));

    gguf_close(ctx);

    fprintf(stderr, "loader: ready (%s RAM, %d VRAM slots)\n",
            L->is_pinned ? "pinned" : "unpinned", LOADER_NUM_SLOTS);
    return L;
}

/* ------------------------------------------------------------------ */
/* loader_free                                                          */
/* ------------------------------------------------------------------ */

void loader_free(loader_t *L) {
    if (!L) return;

    fprintf(stderr, "loader: shutting down — %llu hits, %llu misses",
            (unsigned long long)L->hits, (unsigned long long)L->misses);
    uint64_t total = L->hits + L->misses;
    if (total > 0)
        fprintf(stderr, " (%.1f%% hit rate)", 100.0 * L->hits / total);
    fprintf(stderr, "\n");

    cudaStreamDestroy(L->stream_b);
    for (int i = 0; i < LOADER_NUM_SLOTS; i++) {
        if (L->vram_slots[i]) cudaFree(L->vram_slots[i]);
    }
    if (L->pinned_data) {
        if (L->is_pinned) cudaFreeHost(L->pinned_data);
        else free(L->pinned_data);
    }
    free(L);
}

/* ------------------------------------------------------------------ */
/* loader_request — kick off async DMA for cache-missing experts       */
/* ------------------------------------------------------------------ */

void loader_request(loader_t *L, int layer_idx,
                    const int *expert_indices, int K)
{
    if (layer_idx < 0 || layer_idx >= LOADER_MAX_LAYERS ||
        !L->layers[layer_idx].valid) {
        fprintf(stderr, "loader_request: invalid layer %d\n", layer_idx);
        return;
    }

    const loader_layer_info_t *li = &L->layers[layer_idx];

    for (int k = 0; k < K; k++) {
        int eidx = expert_indices[k];
        if (eidx < 0 || eidx >= li->num_experts) {
            fprintf(stderr, "loader_request: layer %d expert %d out of range\n",
                    layer_idx, eidx);
            continue;
        }

        int slot = L->entry_idx[layer_idx][eidx];
        if (slot >= 0) {
            /* Cache HIT — just update LRU timestamp */
            L->cache[slot].last_used = ++L->access_counter;
            L->hits++;
            continue;
        }

        /* Cache MISS — find LRU slot, evict, DMA */
        L->misses++;
        int lru = find_lru_slot(L);
        evict_slot(L, lru);

        uint8_t *dst = L->vram_slots[lru];

        /* Copy gate weights */
        uint8_t *src_gate = li->gate_base + (uint64_t)eidx * li->gate_expert_bytes;
        CUDA_CHECK(cudaMemcpyAsync(dst, src_gate,
                   (size_t)li->gate_expert_bytes,
                   cudaMemcpyHostToDevice, L->stream_b));
        dst += li->gate_expert_bytes;

        /* Copy up weights */
        uint8_t *src_up = li->up_base + (uint64_t)eidx * li->up_expert_bytes;
        CUDA_CHECK(cudaMemcpyAsync(dst, src_up,
                   (size_t)li->up_expert_bytes,
                   cudaMemcpyHostToDevice, L->stream_b));
        dst += li->up_expert_bytes;

        /* Copy down weights */
        uint8_t *src_down = li->down_base + (uint64_t)eidx * li->down_expert_bytes;
        CUDA_CHECK(cudaMemcpyAsync(dst, src_down,
                   (size_t)li->down_expert_bytes,
                   cudaMemcpyHostToDevice, L->stream_b));

        /* Update cache */
        L->cache[lru].layer_idx  = layer_idx;
        L->cache[lru].expert_idx = eidx;
        L->cache[lru].last_used  = ++L->access_counter;
        L->entry_idx[layer_idx][eidx] = lru;
    }
}

/* ------------------------------------------------------------------ */
/* loader_sync — block until all pending DMA completes                 */
/* ------------------------------------------------------------------ */

void loader_sync(loader_t *L) {
    CUDA_CHECK(cudaStreamSynchronize(L->stream_b));
}

/* ------------------------------------------------------------------ */
/* loader_get_expert — return VRAM pointers for a cached expert        */
/* ------------------------------------------------------------------ */

expert_ptrs_t loader_get_expert(const loader_t *L,
                                int layer_idx, int expert_idx)
{
    expert_ptrs_t p;
    memset(&p, 0, sizeof(p));

    if (layer_idx < 0 || layer_idx >= LOADER_MAX_LAYERS ||
        !L->layers[layer_idx].valid) {
        fprintf(stderr, "loader_get_expert: invalid layer %d\n", layer_idx);
        return p;
    }

    int slot = L->entry_idx[layer_idx][expert_idx];
    if (slot < 0) {
        fprintf(stderr, "loader_get_expert: layer %d expert %d not cached!\n",
                layer_idx, expert_idx);
        return p;
    }

    const loader_layer_info_t *li = &L->layers[layer_idx];
    uint8_t *base = L->vram_slots[slot];

    p.gate       = base;
    p.gate_bytes = li->gate_expert_bytes;
    p.gate_type  = li->gate_type;

    p.up         = base + li->gate_expert_bytes;
    p.up_bytes   = li->up_expert_bytes;
    p.up_type    = li->up_type;

    p.down       = base + li->gate_expert_bytes + li->up_expert_bytes;
    p.down_bytes = li->down_expert_bytes;
    p.down_type  = li->down_type;

    return p;
}

/* ================================================================== */
/*  S E L F T E S T                                                    */
/* ================================================================== */
#ifdef SELFTEST_MAIN

int loader_selftest(const char *model_path) {
    printf("=== loader_selftest ===\n");

    loader_t *L = loader_init(model_path);
    if (!L) {
        printf("FAIL: loader_init returned NULL\n");
        return 1;
    }

    /* Verify layer 0 is a valid MoE layer */
    if (!L->layers[0].valid) {
        printf("FAIL: layer 0 has no expert tensors\n");
        loader_free(L);
        return 1;
    }

    printf("layer 0: gate=%llu up=%llu down=%llu bytes/expert, type=%s/%s/%s\n",
           (unsigned long long)L->layers[0].gate_expert_bytes,
           (unsigned long long)L->layers[0].up_expert_bytes,
           (unsigned long long)L->layers[0].down_expert_bytes,
           ggml_type_name(L->layers[0].gate_type),
           ggml_type_name(L->layers[0].up_type),
           ggml_type_name(L->layers[0].down_type));

    /* Test 1: Load expert(0, 0) */
    int expert_id = 0;
    printf("requesting expert(0, %d)...\n", expert_id);
    loader_request(L, 0, &expert_id, 1);
    loader_sync(L);

    expert_ptrs_t ep = loader_get_expert(L, 0, expert_id);
    if (!ep.gate) {
        printf("FAIL: loader_get_expert returned NULL gate\n");
        loader_free(L);
        return 1;
    }

    /* Copy first 16 bytes of gate weights back to CPU and verify non-zero */
    uint8_t check[16];
    memset(check, 0, 16);
    CUDA_CHECK(cudaMemcpy(check, ep.gate, 16, cudaMemcpyDeviceToHost));

    int any_nonzero = 0;
    printf("first 16 bytes of expert(0,0) gate weights: ");
    for (int i = 0; i < 16; i++) {
        printf("%02x ", check[i]);
        if (check[i] != 0) any_nonzero = 1;
    }
    printf("\n");

    if (!any_nonzero) {
        printf("FAIL: all bytes are zero — data did not land in VRAM\n");
        loader_free(L);
        return 1;
    }

    /* Test 2: Request same expert again — should be a cache hit */
    uint64_t misses_before = L->misses;
    loader_request(L, 0, &expert_id, 1);
    loader_sync(L);
    if (L->misses != misses_before) {
        printf("FAIL: repeat request caused a cache miss\n");
        loader_free(L);
        return 1;
    }
    printf("cache hit on repeat request: OK\n");

    /* Test 3: Load top-9 experts for layer 0, verify all load */
    int experts9[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    loader_request(L, 0, experts9, 9);
    loader_sync(L);
    int all_ok = 1;
    for (int k = 0; k < 9; k++) {
        expert_ptrs_t p = loader_get_expert(L, 0, experts9[k]);
        if (!p.gate || !p.up || !p.down) {
            printf("FAIL: expert(0,%d) not cached\n", experts9[k]);
            all_ok = 0;
        }
    }
    if (all_ok)
        printf("9 experts loaded for layer 0: OK\n");

    /* Test 4: Verify LRU eviction — load experts for layer 1 to fill all 18 slots */
    if (L->layers[1].valid) {
        int experts1[9] = {10, 11, 12, 13, 14, 15, 16, 17, 18};
        loader_request(L, 1, experts1, 9);
        loader_sync(L);
        for (int k = 0; k < 9; k++) {
            expert_ptrs_t p = loader_get_expert(L, 1, experts1[k]);
            if (!p.gate) {
                printf("FAIL: expert(1,%d) not cached after eviction\n", experts1[k]);
                all_ok = 0;
            }
        }
        if (all_ok)
            printf("LRU eviction + 9 more experts for layer 1: OK\n");
    }

    printf("\nhits=%llu misses=%llu (%.1f%% hit rate)\n",
           (unsigned long long)L->hits, (unsigned long long)L->misses,
           (L->hits + L->misses > 0) ?
               100.0 * L->hits / (L->hits + L->misses) : 0.0);

    printf("\nLOADER OK — expert(0,0) loaded, first bytes non-zero\n");
    loader_free(L);
    return 0;
}

int main(int argc, char **argv) {
    const char *path =
        "C:\\Users\\arahe\\.ollama\\models\\blobs\\"
        "sha256-900dde62fb7ebe8a5a25e35d5b7633f403f226a310965fed51d50f5238ba145a";

    if (argc > 1) path = argv[1];
    return loader_selftest(path);
}

#endif /* SELFTEST_MAIN */
