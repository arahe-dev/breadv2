# Async GPU-CPU Expert Orchestration Refactor Plan

## Goal
Implement true async overlap where GPU attention and CPU expert computation run in parallel, reducing per-layer time from ~12 ms to ~5-6 ms.

## Architecture

### Current (Sequential) Flow
```
Layer N:
  ├─ Attention (GPU) [2-3 ms]
  ├─ Sync (blocks CPU)
  ├─ Routing (CPU) [1 ms]
  ├─ Shared Expert (GPU) [1-2 ms]
  ├─ CPU experts [5.2 ms]
  └─ Sync
Total: ~12 ms (sequential)
```

### New (Async) Flow
```
Layer N:
  ├─ Fire GPU attention async on stream_a [returns immediately]
  ├─ Start H2D transfer on stream_b (d_normed2) [async]
  ├─ CPU routing [1 ms, runs while GPU+DMA work]
  ├─ (wait for H2D to complete implicitly at OpenMP barrier)
  ├─ Fire GPU shared expert async on stream_a [parallel with next step]
  ├─ CPU experts parallel [5.2 ms, overlaps with shared expert ~1.5 ms]
  ├─ Start D2H transfer on stream_b (expert result) [async, ~1 ms]
  ├─ GPU accumulation waits for D2H [1 ms, overlaps with D2H]
  └─ Sync stream_a (wait for attention + accumulation)
Total: ~5.2 ms (critical path is CPU experts, GPU hidden)
```

## Key Operations

### Operation 1: Async GPU Attention Firing
**Current code (one_layer.cu ~1548):**
```cuda
bread_matvec(d_qw, d_normed, d_q, ...);
bread_matvec(d_kw, d_normed, d_k, ...);
bread_matvec(d_vw, d_normed, d_v, ...);
CUDA_CHECK(cudaStreamSynchronize(stream_a));  // ← BLOCKS HERE
```

**New code:**
```cuda
bread_matvec(d_qw, d_normed, d_q, ...);  // on stream_a, no sync
bread_matvec(d_kw, d_normed, d_k, ...);
bread_matvec(d_vw, d_normed, d_v, ...);
// NO SYNC - GPU will fetch Q/K/V when CPU needs them
// CPU proceeds to routing/experts in parallel
```

### Operation 2: Async H2D Transfer of d_normed2
**New code (in CPU expert path):**
```cuda
// Fire async copy to pinned h_normed2
CUDA_CHECK(cudaMemcpyAsync(h_normed2, d_normed2, H * sizeof(half),
                           cudaMemcpyDeviceToHost, stream_b));

// OpenMP parallel loop - implicit barrier synchronization
// ensures h_normed2 is ready by the time threads read it
#pragma omp parallel for num_threads(8)
for (int k = 0; k < cfg->top_k; k++) {
    // h_normed2 is guaranteed to be ready here
    cpu_tensor_matvec(...h_normed2...);
}

// After parallel region, implicit sync ensures D2H can fire
```

### Operation 3: Fire GPU Shared Expert Async
**New code (replacing old shared expert sync + compute):**
```cuda
// No sync before shared expert - fire it on stream_a
// while CPU experts are running
bread_matvec(d_sd_w, d_sg, d_sh_out, ..., stream_a);
scale_accum<<<...>>>(d_hidden, d_sh_out, shared_weight, ..., stream_a);
// Don't sync - GPU handles ordering on its own stream
```

### Operation 4: Async D2H Transfer of Expert Result
**New code (after CPU expert computation):**
```cuda
// Convert result to half
for (int i = 0; i < H; i++) h_hidden_half[i] = __float2half(h_cpu_expert_delta[i]);

// Fire async copy - D2H on stream_b
CUDA_CHECK(cudaMemcpyAsync(d_eo, h_hidden_half, H * sizeof(half),
                           cudaMemcpyHostToDevice, stream_b));

// GPU accumulation waits for D2H transfer (on same stream_b)
scale_accum<<<...>>>(d_hidden, d_eo, 1.0f, ..., stream_b);

// Don't sync yet - will sync once at end
```

### Operation 5: Single Sync Point at Layer End
**Current code (line ~2040):**
```cuda
CUDA_CHECK(cudaStreamSynchronize(stream_a));  // Per-stage sync
if (...) loader_sync(L);
```

**New code:**
```cuda
// Sync both streams to ensure all work is complete
CUDA_CHECK(cudaStreamSynchronize(stream_a));  // GPU attention + shared expert
CUDA_CHECK(cudaStreamSynchronize(stream_b));  // CPU expert transfers
```

## Implementation Steps

### Step 1: Add stream variable for transfers (stream_b)
- Declare `stream_b` in one_layer.cu as static
- Create once in first call
- Use for all async transfers

### Step 2: Refactor attention path (lines ~1548-1572)
- Remove `cudaStreamSynchronize(stream_a)` before CPU attention code
- Let GPU attention run async
- CPU attention code (Q/K/V on CPU) can now run in parallel

### Step 3: Refactor CPU expert path (lines ~1952-1983)
- Change `vram_half_to_cpu_float()` to `cudaMemcpyAsync()` on stream_b
- Keep OpenMP parallel loop (implicit barrier sync ensures h_normed2 is ready)
- Change final `cudaMemcpy()` to `cudaMemcpyAsync()` on stream_b
- Don't sync after expert accumulation

### Step 4: Refactor shared expert path (lines ~1889-1915)
- Remove sync before shared expert
- Fire shared expert on stream_a (same stream as attention)
- No intermediate syncs

### Step 5: Update final sync (line ~2040)
- Keep single sync at layer end for both streams

## Files to Modify
1. `one_layer.cu` - Main refactor (lines ~1500-2080)
   - Add stream_b initialization
   - Remove intermediate syncs
   - Make transfers async
   - Update final sync

2. No changes needed to:
   - `buffer_pool.c` - h_normed2 already pinned ✓
   - `kernels.cu` - kernels unchanged
   - `bread.h/bread.c` - config unchanged

## Testing Strategy

1. **Correctness test:**
   ```bash
   ./bread.exe --cpu-experts --prompt "The capital of France is" --tokens 5
   # Should output: "The capital of France is **Paris**."
   ```

2. **Timing test:**
   ```bash
   ./bread.exe --cpu-experts --prompt "The capital of France is" --tokens 30
   # Current: 1.81 tok/s (552 ms/token)
   # Target: 4-5 tok/s (200-250 ms/token) with proper async
   ```

3. **GPU mode baseline (unchanged):**
   ```bash
   ./bread.exe --prompt "The capital of France is" --tokens 30
   # Should still be ~5.82 tok/s (verify no regression)
   ```

## Expected Results

If async orchestration works:
- Per-layer time: 12 ms → 5-6 ms (CPU experts on critical path, GPU hidden)
- Token time: 552 ms → 200-240 ms
- Throughput: 1.81 tok/s → **4.2-5.0 tok/s**

Not quite GPU's 5.82 tok/s, but a massive improvement. The gap would be from:
- CPU experts being slower than GPU experts in absolute time (5.2 vs 3.2 ms base)
- This is fundamental - can't fix without faster CPU or smaller model

## Risk Assessment

**Low risk:**
- Async transfers don't affect computation correctness
- GPU streams are independent by default
- OpenMP barrier naturally synchronizes with async transfers

**Medium risk:**
- Stream ordering on different hardware might differ
- WDDM (Windows) might reorder operations unexpectedly
- Need careful testing

**Mitigation:**
- Test on GPU between each change
- Run correctness checks frequently
- If something breaks, can revert individual changes

## Estimated Timeline

- Step 1 (stream setup): 15 minutes
- Step 2 (attention refactor): 30 minutes  
- Step 3 (CPU expert refactor): 1 hour
- Step 4 (shared expert refactor): 30 minutes
- Step 5 (sync update): 15 minutes
- Testing & debugging: 1-2 hours
- **Total: 3-4 hours**

