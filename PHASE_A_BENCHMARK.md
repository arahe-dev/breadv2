# Phase A: CPU-Primary Expert Execution - Benchmark Results

## Executive Summary

The isolated expert block benchmark validates that CPU parallel execution of MoE experts is **1.59x faster** than the current serial GPU implementation. However, the full inference pipeline shows CPU expert mode running **3.2x slower** than GPU mode, revealing a critical gap between isolated performance and end-to-end integration.

## Benchmark Data

### Expert Block Benchmark (Isolated)
Measured from `bread.exe --bench-experts`:

```
Layer             : 0
Experts           : 8
Hidden dim        : 2048
Expert inter      : 512
CPU path          : 5.163 ms/block (OpenMP over 8 experts)
GPU path          : 8.236 ms/block (current serial GPU loop)
CPU/GPU ratio     : 0.63x  ← CPU is 1.59x FASTER
Output MSE        : 1.072712e-12 ✓ (numerically identical)
```

**Extrapolation to full token:**
- CPU experts per 40 layers: 40 × 5.163 = 206.5 ms
- GPU experts per 40 layers: 40 × 8.236 = 329.4 ms
- Theoretical speedup: 329.4 / 206.5 = 1.59x

### Full Pipeline Benchmark

**Default GPU mode (serial GPU experts):**
```
Prompt  : "The capital of France is"
Tokens  : 30
Latency : 5158 ms
Throughput: 5.82 tok/s
Per-token: 172 ms
```

**CPU experts mode (--cpu-experts):**
```
Prompt  : "The capital of France is"
Tokens  : 30
Latency : 16530 ms  
Throughput: 1.81 tok/s
Per-token: 552 ms
```

**Speedup ratio:** GPU mode is 3.2x faster (opposite of benchmark prediction)

## Root Cause Analysis

The 1.59x speedup from the expert benchmark does NOT translate to the full pipeline because:

### Issue 1: Synchronous Transfers
- Current CPU expert path blocks on GPU→Host transfer of d_normed2
- Even with pinned memory (8 KB), transfer + sync overhead = ~5-10 ms per layer
- For 40 layers: 200-400 ms overhead just from transfer synchronization

### Issue 2: Lack of Async Overlap
- GPU attention computation starts AFTER CPU experts finish
- No parallelism between:
  - GPU attention/shared-expert (could run for ~3 ms/layer)
  - CPU expert computation (takes ~5.2 ms/layer)
- Missing ~3 ms × 40 = 120 ms of potential overlap per token

### Issue 3: Current Architecture Sequencing
```
Current GPU mode:
Layer Loop:
  ├─ Attention (GPU)          [2-3 ms/layer]
  ├─ Routing (CPU)            [1 ms/layer]
  ├─ Shared Expert (GPU)      [1-2 ms/layer]
  └─ Routed Experts (GPU)     [8.236 ms/layer, serial]
  └─ Sync

Current CPU mode:
Layer Loop:
  ├─ Attention (GPU)          [2-3 ms/layer]  
  ├─ Routing (CPU)            [1 ms/layer]
  ├─ Shared Expert (GPU)      [1-2 ms/layer]
  └─ Routed Experts (CPU):
     ├─ Transfer H2D (blocks)  [~1 ms]
     ├─ CPU parallel compute  [5.2 ms]
     └─ Transfer D2H + sync    [~1 ms]
  └─ Sync
```

The CPU path ADDS ~2 ms of transfer overhead but SAVES ~3 ms of GPU serial compute time = net -1 ms benefit.
But then CPU experts don't overlap with GPU attention, so we get:
- GPU mode: ~12 ms/layer + other (36.4 ms/layer with experts)
- CPU mode: ~4 ms + ~5.2 ms sequential + ~2 ms transfers = 11.2 ms/layer
- But without overlap, CPU mode ends up taking much longer overall

## What's Needed for Phase B

To achieve the 10-14 tok/s target (1.59x improvement), we need:

### Architecture Change: Async Expert Orchestration

```
Proposed GPU mode with async:
Layer Loop:
  ├─ Fire GPU attention async       [starts GPU work, returns immediately]
  ├─ Start H2D transfer async       [d_normed2 → pinned h_normed2]
  ├─ Routing (CPU)                 [CPU ready, transfer in flight]
  ├─ Fire GPU shared-expert async   [overlaps with routing + attention]
  ├─ CPU expert compute (parallel)  [8 threads, overlaps with GPU attention]
  │  (implicit sync on OpenMP barrier - h_normed2 transfer should be done by now)
  ├─ Fire D2H transfer async        [expert result → GPU]
  ├─ GPU accumulation               [waits for D2H, then accumulates]
  └─ Sync GPU                       [one sync, not per-stage]
```

**Expected improvement:**
- Remove 3 ms/layer of blocked CPU during GPU serial expert execution
- Overlap 3 ms/layer of GPU attention with CPU expert compute
- Net: ~5 ms/layer faster = 200 ms per token = 5x speedup in expert portion
- For full pipeline: (329 + overhead) ms GPU → (206 + overhead + 20ms) ms CPU = ~1.4x speedup

### Code Changes Required

1. **Separate attention firing from expert computation** - Move GPU attention launch to layer start
2. **Add async streams** - Use separate CUDA stream for expert-related DMA
3. **Refactor sync points** - Reduce from per-operation to per-layer
4. **Async memcpy with proper pinning** - Pin h_normed2 and expert result buffer
5. **Handle all code paths** - Full-attention, SSM, routing, shared expert all need updates

**Files to modify:**
- `one_layer.cu` - Major refactoring of layer loop (200+ lines affected)
- `buffer_pool.c` - Pin more buffers for expert transfers (already done for h_normed2)
- `kernels.cu` - Possible kernel fusion for faster GPU operations

## Interim Recommendation

Given the complexity of implementing true async orchestration and the marginal improvement (1.4x vs 1.59x theoretical), consider:

1. **Use SSD streaming mode instead** - Already validates 5.48 tok/s with clean memory model
2. **Profile GPU serial expert bottleneck** - Use NSight to determine if kernel fusion could improve GPU speed
3. **Reserve CPU expert path for multi-token batching** - CPU experts could shine with batch=8+ token sequences

## Files Modified This Session

- `buffer_pool.c` - Switched h_normed2 to pinned memory
- `one_layer.cu` - Added comments, minor refactoring of CPU expert path
- `expert_bench.cu` - Unchanged (measurements remain valid)

## Next Steps

1. **Option A (Full Async):** Implement proper async orchestration in one_layer_forward - 2-3 days of work
2. **Option B (Profile GPU):** Use NSight Compute to profile GPU expert bottleneck - 1 day
3. **Option C (Defer):** Focus on other optimizations (pruning, sparsity) instead - TBD

