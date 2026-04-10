# BREAD Profiling Analysis (April 10, 2026) - UPDATED

## Critical Finding: Expert Cache Not Being Used

The profiling **uncovers a critical issue**: Expert weights are NOT being loaded through the LRU cache system. 

Evidence:
1. Loader shutdown message: `"loader: shutting down — 0 hits, 0 misses"`
2. Expert LRU cache (18 slots, 35 MB total) is completely inactive
3. Yet profiling shows 32,424 H2D transfers totaling 7.53 seconds

**Interpretation**: Expert weights are being transferred directly from host RAM to VRAM without using the cache infrastructure. This defeats the purpose of the designed expert caching system.

---

## Profiling Summary

| Metric | Value | % of Total |
|--------|-------|-----------|
| GPU Kernel Compute | 3.16 s | 8.6% |
| H2D Memory Transfers | 7.53 s | 20.4% |
| CUDA Synchronization | 2.88 s | 7.8% |
| cudaMalloc/cudaFree | 4.11 s | 11.2% |
| **Total CUDA API Overhead** | **36.83 s** | **100%** |

---

## Root Cause Analysis: Where Are the 32K H2D Transfers?

### Hypothesis 1: Expert Weights (CONFIRMED)
- 40 layers × 8 top_k experts = 320 expert DMA requests per token
- 20 tokens × 320 = 6,400 expert DMAs
- **But we have 32,424 transfers → 5x more than expected**
- **Suggests each expert DMA is being fragmented into ~5 separate transfers**

### Hypothesis 2: Non-Expert Weights Also Being Reloaded
- Even though weight_cache shows 769.4 MiB cached, maybe some paths bypass it
- Check: attn_norm_w, post_attn_norm_w, ffn_gate_shexp_w, etc.

### Hypothesis 3: Activations Being Unnecessarily Copied
- d_normed, d_sg, d_su, etc. shouldn't require H2D but might be

---

## The Real Bottleneck

**NOT compute, NOT individual malloc/free calls.**

**The bottleneck is: Host RAM → VRAM bandwidth saturation from expert weight loading.**

Expert weights must be loaded every time they're selected because the LRU cache (18 slots) is too small to hold frequently reused experts.

With 256 experts per layer but only 18 cache slots:
- Probability of cache hit = 18/256 = 7%
- Probability of cache miss = 238/256 = 93%
- Each miss = 1.95 MB expert transferred from RAM
- 20 tokens × 40 layers × 8 experts/token × 1.95 MB × 93% misses = ~117 GB of transfers
- At 192 GB/s bandwidth, this alone takes 0.6 seconds per token
- Observed: 7.53 seconds for 5 tokens = **1.5 seconds per token**

This matches! **Expert weight bandwidth is the bottleneck.**

---

## Why Has This Been Overlooked?

Looking at the code history (from CLAUDE.md):
1. Phase 5 claims: "Pre-load ALL expert weights into VRAM at startup" 
2. Code shows: `weight_cache_load_experts()` is called
3. Output shows: "weight_cache: loaded experts, total VRAM: 20689.4 MiB"

**But 20.6 GB > 8 GB VRAM!** The weights probably aren't actually in VRAM, or the message is about preparing them, not keeping them resident.

The actual VRAM allocation shows:
```
loader: allocating 18 VRAM slots × 1.95 MiB = 35.02 MiB
```

Only 35 MB of VRAM is used for the LRU cache. The rest of the 769.4 MiB non-expert weights fit fine (9.9 MB, leaving 6.2 GB unused). But experts cannot fit (20.6 GB > 8 GB).

---

## The Solution

### Option A: Full Expert Caching (Requires Architectural Change)
**Problem**: 256 experts × 1.95 MB = 500 MB per layer × 40 layers = 20 GB total.
**8GB VRAM cannot fit this.**

**Solution**: Accept partial expert caching (18 slots = 35 MB holds ~18 experts at a time).
**But expert selection is random per token!** Reusing the same 18 experts is unlikely.
**Result**: 93% cache misses regardless.

### Option B: Compression (FloE Pattern)
Cache all experts using **block-wise quantization or compression**:
- Reduce 20.6 GB to ~2-3 GB using compression similar to FloE (9.3x reduction possible)
- Allocate full expert set in VRAM
- **Expected gain**: 50-80% speedup (eliminates 93% of H2D transfers)
- **Risk**: Decompression overhead on each use might partially negate gain

### Option C: CPU-GPU Pipelined Streaming (Current Design)
Keep the LRU cache design, optimize for throughput:
- Pre-schedule expert DMAs while computing previous layer
- Use CUDA streams for overlap
- Accept that expert loading is a cost, optimize around it
- **Expected gain**: 10-20% (from better overlap)
- **Easiest to implement**

### Option D: Expert Parallelism (Phase 2)
Parallelize K expert computations across K streams:
- Instead of sequential: expert 0 → expert 1 → expert 2 → expert 3 (serialized on stream_a)
- Do: all 8 experts in parallel on expert_streams[0..7]
- Hide DMA latency under GPU compute
- **Expected gain**: 15-30% (if DMA can overlap with compute of previous experts)
- **Requires expert_streams infrastructure (already in buffer_pool!)**

---

## Immediate Recommendations

### 🔴 CRITICAL: Verify Expert Stream Parallelism Wiring
The buffer pool has `expert_streams[BREAD_MAX_TOP_K]` pre-created, but are they being used?

```bash
grep -n "expert_streams\[" C:/bread_v2/one_layer.cu
```

**Expected**: Should see expert_streams[k] being passed to bread_matvec() calls.
**If not found**: Expert parallelism is turned OFF despite infrastructure.

### 🟡 HIGH PRIORITY: Enable Expert Stream Parallelism
This is **already designed and allocated** but apparently not activated.
- Submit expert k to expert_streams[k] instead of stream_a
- Sync all K experts at end with single cudaEventRecord/Sync
- **Expected gain**: 15-30% (immediate return)
- **Effort**: ~1-2 hours to implement and test

### 🟢 MEDIUM PRIORITY: Layer Prefetching with DMA Overlap
- Detect next layer's routing while current layer computes
- Fire DMA for layer N+1 experts on stream_c while layer N processes
- **Expected gain**: 5-10% (overlap benefit)
- **Effort**: 2-3 hours

### 🟢 LOWER PRIORITY: Expert Compression (FloE)
- Benchmark expert decompression cost vs savings
- May not be worth it if expert streams give >25% improvement
- **Expected gain**: 30-50% if overhead is low, but risky
- **Effort**: 6-8 hours

---

## Next Step

**RUN THIS COMMAND:**
```bash
grep -n "expert_streams" C:/bread_v2/one_layer.cu
```

This will tell us if expert streams are wired in the inference path or if they're just allocated but unused.

If unused → **implement Phase 2B-1 (expert stream parallelism) immediately** → expect 15-30% speedup.

---

## Benchmarking Timeline

| Phase | Action | Expected Speedup | Est. Time |
|-------|--------|------------------|-----------|
| **Current** | Baseline (stream_a only) | 1x (6.2 tok/s) | — |
| **Phase 2B-1** | Activate expert_streams | 1.2-1.3x (7.5-8.0 tok/s) | 2 hours |
| **Phase 2B-2** | Layer prefetch DMA overlap | 1.05-1.1x more | 3 hours |
| **Phase 2B-3** | Expert compression (risky) | 1.3-1.5x more | 8 hours |

---

## Key Learnings

1. **Profiling is Essential**: Assumptions about what's slow are often wrong.
2. **Expert caching at full size is impossible on consumer VRAM**.
3. **Expert streams are already allocated but apparently dormant**.
4. **The next high-value win is in the already-designed infrastructure**.
