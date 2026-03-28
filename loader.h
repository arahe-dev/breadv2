#ifndef LOADER_H
#define LOADER_H

/* loader.h — RAM→VRAM expert weight streaming for BREAD.
 *
 * Loads the full model file into host memory. On Windows/WDDM, pinning
 * an entire 20+ GB GGUF via cudaMallocHost can exhaust GPU-visible VA
 * space even when VRAM itself is mostly free, so the loader keeps the
 * whole blob in normal RAM for stability.
 *
 * Expert weights are cached in VRAM slots with LRU eviction.
 * The inference loop calls:
 *   loader_request()  — kick off async DMA for cache-missing experts
 *   loader_sync()     — block until all DMA is done
 *   loader_get_expert() — get VRAM pointers for a cached expert
 */

#include <stdint.h>
#include <cuda_runtime.h>
#include "gguf.h"

/* ------------------------------------------------------------------ */
/* Limits                                                               */
/* ------------------------------------------------------------------ */

#define LOADER_MAX_LAYERS   64    /* max transformer blocks */
#define LOADER_MAX_EXPERTS  256   /* max experts per MoE layer */
#define LOADER_NUM_SLOTS    18    /* 2 × 9 double-buffered VRAM slots */
#define LOADER_ACTIVE_K     9     /* top-K active experts per token */

/* ------------------------------------------------------------------ */
/* Per-layer expert tensor info                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t *gate_base;           /* pinned RAM ptr to gate_exps tensor start */
    uint8_t *up_base;             /* pinned RAM ptr to up_exps tensor start */
    uint8_t *down_base;           /* pinned RAM ptr to down_exps tensor start */
    uint64_t gate_expert_bytes;   /* bytes per expert in gate tensor */
    uint64_t up_expert_bytes;     /* bytes per expert in up tensor */
    uint64_t down_expert_bytes;   /* bytes per expert in down tensor */
    uint32_t gate_type;           /* ggml_type_t for gate weights */
    uint32_t up_type;             /* ggml_type_t for up weights */
    uint32_t down_type;           /* ggml_type_t for down weights */
    int      num_experts;         /* experts in this layer (from dims[2]) */
    int      valid;               /* 1 if this layer has expert tensors */
} loader_layer_info_t;

/* ------------------------------------------------------------------ */
/* LRU cache entry                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    int      layer_idx;           /* -1 = unused slot */
    int      expert_idx;
    uint64_t last_used;           /* monotonic counter for LRU ordering */
} loader_cache_entry_t;

/* ------------------------------------------------------------------ */
/* Expert pointers returned by loader_get_expert                       */
/* ------------------------------------------------------------------ */

typedef struct {
    void    *gate;                /* VRAM pointer to gate weights */
    void    *up;                  /* VRAM pointer to up weights */
    void    *down;                /* VRAM pointer to down weights */
    uint64_t gate_bytes;
    uint64_t up_bytes;
    uint64_t down_bytes;
    uint32_t gate_type;           /* ggml_type_t */
    uint32_t up_type;
    uint32_t down_type;
} expert_ptrs_t;

/* ------------------------------------------------------------------ */
/* Non-expert weight cache — per-layer VRAM pointers                   */
/* ------------------------------------------------------------------ */

typedef struct {
    /* Common tensors — all 40 layers */
    void *attn_norm_w;          /* F32  — blk.N.attn_norm.weight           */
    void *post_attn_norm_w;     /* F32  — blk.N.post_attention_norm.weight */
    void *ffn_gate_shexp_w;     /* Q4_K — blk.N.ffn_gate_shexp.weight      */
    void *ffn_up_shexp_w;       /* Q4_K — blk.N.ffn_up_shexp.weight        */
    void *ffn_down_shexp_w;     /* Q6_K — blk.N.ffn_down_shexp.weight      */

    /* Full-attention only (NULL for SSM layers) */
    void *attn_q_w;             /* Q4_K — blk.N.attn_q.weight              */
    void *attn_k_w;             /* Q4_K — blk.N.attn_k.weight              */
    void *attn_v_w;             /* Q6_K — blk.N.attn_v.weight              */
    void *attn_output_w;        /* Q4_K — blk.N.attn_output.weight         */

    /* SSM only (NULL for full-attention layers) */
    void *attn_qkv_w;           /* Q4_K — blk.N.attn_qkv.weight            */
    void *attn_gate_w;          /* Q4_K — blk.N.attn_gate.weight           */
    void *ssm_alpha_w;          /* Q4_K — blk.N.ssm_alpha.weight           */
    void *ssm_beta_w;           /* Q4_K — blk.N.ssm_beta.weight            */
    void *ssm_out_w;            /* Q4_K — blk.N.ssm_out.weight             */
} wc_layer_t;

typedef struct {
    wc_layer_t layers[40];      /* one entry per transformer block */
    uint64_t   total_bytes;     /* total VRAM consumed — for reporting */
} weight_cache_t;

/* ------------------------------------------------------------------ */
/* Loader context                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    /* Host memory holding the full model file */
    uint8_t *pinned_data;
    uint64_t pinned_size;         /* file size in bytes */
    int      is_pinned;           /* always 0 for now; reserved for future staging */

    /* GGUF metadata */
    uint64_t data_offset;         /* absolute byte offset of GGUF data section */
    int      num_moe_layers;      /* count of layers with expert tensors */
    loader_layer_info_t layers[LOADER_MAX_LAYERS];

    /* VRAM expert slots — each holds gate+up+down contiguously */
    uint8_t *vram_slots[LOADER_NUM_SLOTS];
    uint64_t slot_size;           /* bytes per slot (max gate+up+down) */

    /* LRU cache */
    loader_cache_entry_t cache[LOADER_NUM_SLOTS];
    int      entry_idx[LOADER_MAX_LAYERS][LOADER_MAX_EXPERTS]; /* slot or -1 */
    uint64_t access_counter;
    uint64_t hits;
    uint64_t misses;

    /* CUDA stream for async DMA */
    cudaStream_t stream_b;
} loader_t;

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

#ifdef __cplusplus
extern "C" {
#endif

/* Initialise loader: parse GGUF, load model into pinned RAM, allocate
 * VRAM slots, build LRU cache. Returns NULL on failure. */
loader_t *loader_init(const char *model_path);

/* Shut down: free pinned RAM, VRAM slots, CUDA stream. */
void loader_free(loader_t *L);

/* Kick off async DMA for K experts. Cache hits skip DMA.
 * Returns immediately — call loader_sync() before using results. */
void loader_request(loader_t *L, int layer_idx,
                    const int *expert_indices, int K);

/* Block until all pending DMA from loader_request has completed. */
void loader_sync(loader_t *L);

/* Get VRAM pointers for a cached expert. Must call after loader_sync().
 * Returns zeroed struct if expert is not cached (logic error). */
expert_ptrs_t loader_get_expert(const loader_t *L,
                                int layer_idx, int expert_idx);

/* Run a self-test: load the model, request expert(0,0), verify non-zero
 * data landed in VRAM. Returns 0 on success, 1 on failure. */
int loader_selftest(const char *model_path);

/* Upload all non-expert weights to VRAM once. Call after loader_init().
 * is_full_attn_fn: pass bread_layer_is_full_attention — avoids pulling
 * bread.h into loader.h, keeping the boundary clean. Returns NULL on failure. */
weight_cache_t *weight_cache_init(const loader_t *L, const gguf_ctx_t *g,
                                  int num_layers,
                                  int (*is_full_attn_fn)(int layer_idx));

/* Free all VRAM allocations in the weight cache and the struct itself. */
void weight_cache_free(weight_cache_t *wc);

#ifdef __cplusplus
}
#endif

#endif /* LOADER_H */
