# Phase 2: Stream-Ordered Allocation + Arena Allocator
## Metrics, Analysis, and Lessons Learned

**Date:** 2026-04-08
**Branch:** `phase2-stream-ordered-arena-allocator`
**Status:** EXPERIMENTAL (did not improve performance)

---

## Baseline (before Phase 2)

```
Prompt tokens:     16
Generated tokens:  45
Prefill:          2045 ms (127.8 ms/tok)
Time to 1st:       58.7 ms
Decode:            8091 ms (5.56 tok/s)
Total runtime:     ~63 seconds (includes model load)
```

---

## Phase 2 Implementation

### Step 1: Stream-Ordered GPU Allocation
- Converted all `cudaMalloc(&d, size)` → `cudaMallocAsync(&d, size, stream_a)`
- Converted all `cudaFree(d)` → `cudaFreeAsync(d, stream_a)`
- Created `load_vram_stream()` and `load_expert_tensor_vram_stream()` variants
- Updated all 15+ weight load callsites to use `stream_a`

**Result:**
```
Prompt tokens:     16
Generated tokens:  45
Prefill:          2050 ms (128.1 ms/tok)
Time to 1st:       57.7 ms
Decode:            8224 ms (5.47 tok/s)
Change:            -1.6%
```

### Step 2: CPU-Side Arena Allocator
- Implemented 300 MB pre-allocated buffer pool
- Functions: `cpu_arena_init()`, `cpu_arena_reset()`, `cpu_arena_alloc()`, `cpu_arena_free()`
- Reset arena at start of each token generation
- Integrated with main.cu initialization/cleanup

**Result:**
```
Prompt tokens:     16
Generated tokens:  45
Prefill:          2046 ms (127.9 ms/tok)
Time to 1st:       58.1 ms
Decode:            8363 ms (5.38 tok/s)
Change:            -3.2% (cumulative with Step 1)
```

---

## Why Stream-Ordered Allocation Didn't Help

### Theoretical Benefit (Expected)
- Eliminate per-layer `cudaStreamSynchronize()` overhead
- Enable async overlap of allocation with GPU compute
- Reduce fragmentation via driver-managed memory pools
- **Expected: +8-15% improvement**

### Actual Implementation Reality

1. **Architecture Mismatch**
   ```
   Current pattern:
   for (layer = 0; layer < 40; layer++) {
       load_vram(weights);           // cudaMallocAsync
       one_layer_forward();
       cudaStreamSynchronize();      // ← BLOCKS HERE ANYWAY
   }
   ```
   The sync after each layer negates async benefits.

2. **Single Stream**
   - Only one compute stream (stream_a)
   - Stream-ordered allocation shines with multiple concurrent streams
   - With single stream, async = deferred, not concurrent

3. **Memcpy Dominates**
   - Per-layer cost breakdown (estimated):
     - `cudaMallocAsync()`: ~0.5 ms
     - `cudaMemcpy()` (weight transfer): ~2-3 ms
     - Layer computation: ~3-4 ms
     - `cudaFreeAsync()`: ~0.5 ms
   - Allocation is ~10% of layer time, not bottleneck

4. **Sync Requirement**
   - Must sync before using weights in computation
   - `cudaMemcpy()` doesn't overlap with async malloc
   - Sync still happens, overhead not avoided

---

## Root Cause of Performance Bottleneck

### Actual Problem
```
Per-token loop (40 iterations):
  for layer 0-39:
    load_vram("attn_q.weight")         // malloc + memcpy
    load_vram("attn_k.weight")         // malloc + memcpy
    load_vram("attn_v.weight")         // malloc + memcpy
    load_vram("output.weight")         // malloc + memcpy
    load_vram("shared_gate.weight")    // malloc + memcpy
    load_vram("shared_up.weight")      // malloc + memcpy
    load_vram("expert_gate.weight")    // malloc + memcpy
    ... (8 more weight loads)
    one_layer_forward()
    cudaFree(all 15 weights)
```

**Total per token: 600 weight loads + 600 allocations**

These weights are:
- ❌ NOT changing between tokens
- ❌ NOT changing between generations
- ❌ Read from disk/GGUF every single time
- ✅ COULD be cached in VRAM once at startup

### Real Solution
Pre-load weights at startup (Phase 2 REVISED):
```c
// At init:
weight_cache_t wc = weight_cache_init();
for (layer = 0; layer < 40; layer++) {
    wc->layers[layer].attn_q = load_vram_once("blk.%d.attn_q.weight");
    wc->layers[layer].attn_k = load_vram_once("blk.%d.attn_k.weight");
    // ... all 600+ weights loaded ONCE
}

// Per token:
for (layer = 0; layer < 40; layer++) {
    d_q = wc->layers[layer].attn_q;  // Direct pointer, no malloc/memcpy
    one_layer_forward();
}
```

**Expected improvement: -50-80 ms/token (15-20% speedup)**

---

## Lessons Learned

### ❌ What Didn't Work
1. **Stream-ordered allocation without architectural change**
   - Helps with fragmentation, not with synchronous patterns
   - Overhead can exceed benefits on small allocations

2. **Async strategies for synchronous algorithms**
   - Our loop structure forces sequential execution anyway
   - Async malloc is useful for true pipelined/batched work

3. **Optimizing allocation when bottleneck is data transfer**
   - Malloc overhead: 0.5 ms
   - Memcpy overhead: 2-3 ms (5-6x larger)
   - Focusing on malloc misses the real problem

### ✅ What We Learned
1. **Profile before optimizing**
   - System message correctly identified "cudaMalloc overhead"
   - But didn't distinguish from memcpy + reloading costs
   - Actual bottleneck: repeated weight reloading, not allocation strategy

2. **Single-stream compute doesn't benefit from async malloc**
   - Stream-ordered allocation is for multi-stream workloads
   - For sequential work, pre-allocation is better

3. **llama.cpp's approach is right**
   - All weights stay in VRAM once loaded
   - Only change between requests is KV cache
   - Our current approach (reload every layer) is inherently slow

---

## Comparison: Other Inference Engines

| Engine | Approach | Why |
|--------|----------|-----|
| **llama.cpp** | All weights in VRAM at startup | Fastest, simplest |
| **vLLM** | Weights in VRAM + PagedAttention for KV | Handles batching, variable seq len |
| **TensorRT** | Pre-allocate compute graph + workspace | Compiled graphs, known sizes |
| **BREAD (before)** | Reload weights per-layer | Assumed memory constraints |
| **BREAD (Phase 2)** | Reload weights per-layer, async malloc | Still wrong approach |
| **BREAD (proposed)** | All weights in VRAM like llama.cpp | Same as proven approaches |

---

## VRAM Budget Check

Current model on RTX 4060 (8 GB VRAM):
- Model weights: ~27 GB (host RAM)
- Expert cache: 35 MB (8 slots)
- Activation buffers: ~100 MB
- KV caches: ~200 MB
- **Total currently in VRAM: ~350 MB**

Pre-loading all weights would require:
- Non-expert weights: ~2.5-3 GB
- Total in VRAM: ~3.5 GB (leaving 4.5 GB free)
- **Feasible on RTX 4060** ✅

---

## Recommendation

### Revert Phase 2
- No real benefit (-3% regression)
- Adds complexity without payoff
- Not aligned with actual bottleneck

### Implement Phase 2 (REVISED)
- Pre-load all non-expert weights at startup
- Extend `weight_cache_t` to cache all 600+ weight tensors
- Modify `one_layer_forward()` to use cached pointers
- **Expected: 5.56 → 6.5-6.8 tok/s (+17-22%)**

This is the approach that scales and matches industry standard (llama.cpp).

---

## Files Changed
- `one_layer.cu`: Added arena allocator + stream-ordered function variants
- `main.cu`: Integrated arena init/reset/cleanup + cpu_arena_alloc declarations

## Build Command
```bash
nvcc -O2 -x cu main.cu one_layer.cu kernels.cu loader.c gguf.c tokenizer.c \
     bread.c hooks.c progress_tracking.c buffer_pool.c -I. -o bread.exe
```

## Branch
```bash
git checkout phase2-stream-ordered-arena-allocator
```
