# SSM SIMD Optimization — March 2026

## Summary

Implemented AVX2 vectorization of the CPU-bound SSM recurrence loop (`cpu_delta_net_autoregressive_step`), replacing scalar operations with SIMD instructions.

**Expected Performance Gain:** 5-8x speedup on SSM state updates (8 floats per vector cycle with 128-element key_dim)

## Problem Statement

SSM (Gated Delta Network) recurrence in BREAD is CPU-bound:
- 30 SSM layers per token × 0.5–1 ms each = **15–30 ms/token overhead**
- All computations are sequential per position (can't parallelize state evolution)
- Current implementation: scalar scalar loops over key_dim (128 elements)

This is the largest remaining bottleneck in BREAD after phases 5–6 optimizations.

## Solution: AVX2 Vectorization

**Target Instructions:**
- `_mm256_loadu_ps()` — Load 8 floats
- `_mm256_storeu_ps()` — Store 8 floats
- `_mm256_mul_ps()` — Element-wise multiply (8 floats)
- `_mm256_fmadd_ps()` — Fused multiply-add (critical for state updates)
- `_mm256_set1_ps()` — Broadcast scalar to 8 floats

**Horizontal Sum Helper:**
```c
static inline float m256_hsum(__m256 v) {
    // Permute/shuffle to sum all 8 elements into a single float
    // Uses 4 shuffles for 8→4→2→1 reduction
}
```

## Changes Made

### File 1: `layer_ops.cu` (lines 336–365)

**Original:** 4 separate scalar loops:
```c
// Phase 1: decay loop (line 345)
for (int ki = 0; ki < key_dim; ki++) row[ki] *= decay;

// Phase 2: dot product (line 350)
for (int ki = 0; ki < key_dim; ki++) sk += row[ki] * k[ki];

// Phase 3: state update (line 357)
for (int ki = 0; ki < key_dim; ki++) row[ki] += k[ki] * d;

// Phase 4: output (line 362)
for (int ki = 0; ki < key_dim; ki++) acc += row[ki] * q[ki];
```

**New:** 4 SIMD-vectorized loops (with scalar fallback):
```c
#ifdef __AVX2__
// Phase 1: vectorized decay (8 elements per iteration)
for (int ki = 0; ki < key_dim; ki += 8) {
    __m256 state_v = _mm256_loadu_ps(row + ki);
    state_v = _mm256_mul_ps(state_v, decay_v);
    _mm256_storeu_ps(row + ki, state_v);
}

// Phase 2: vectorized dot product with FMA
__m256 sk_v = _mm256_setzero_ps();
for (int ki = 0; ki < key_dim; ki += 8) {
    __m256 row_v = _mm256_loadu_ps(row + ki);
    __m256 k_v = _mm256_loadu_ps(k + ki);
    sk_v = _mm256_fmadd_ps(row_v, k_v, sk_v);  // multiply-add
}
float sk = m256_hsum(sk_v);  // horizontal sum

// Phase 3: vectorized FMA state update
__m256 d_v = _mm256_set1_ps(d);  // broadcast delta
for (int ki = 0; ki < key_dim; ki += 8) {
    __m256 state_v = _mm256_loadu_ps(row + ki);
    __m256 k_v = _mm256_loadu_ps(k + ki);
    state_v = _mm256_fmadd_ps(k_v, d_v, state_v);
    _mm256_storeu_ps(row + ki, state_v);
}

// Phase 4: vectorized output dot product
__m256 acc_v = _mm256_setzero_ps();
for (int ki = 0; ki < key_dim; ki += 8) {
    __m256 row_v = _mm256_loadu_ps(row + ki);
    __m256 q_v = _mm256_loadu_ps(q + ki);
    acc_v = _mm256_fmadd_ps(row_v, q_v, acc_v);
}
out[vi] = m256_hsum(acc_v);
#else
// Scalar fallback for systems without AVX2
// ...original code...
#endif
```

### File 2: `one_layer.cu` (lines 530–577)

**Same changes** applied to the duplicate `cpu_delta_net_autoregressive_step()` function (static version in one_layer.cu).

**Note:** Both files have the function because layer_ops.cu is for general utilities, one_layer.cu has a static copy for layer-specific use.

## Vectorization Details

### Loop Structure

```
Outer loop (value_dim, 128 iterations):
    vi = 0, 1, 2, ..., 127

    Inner SIMD loop (key_dim, 128/8 = 16 vector iterations):
        ki = 0, 8, 16, 24, ..., 120
        Process 8 floats per iteration
```

**Original instruction count:** 128 × 128 = 16,384 scalar operations
**Vectorized count:** 128 × (128/8) = 128 × 16 = 2,048 vector operations
**Theoretical speedup:** 8x (ignoring memory bandwidth, reduction overhead)

### Practical Speedup Estimate

- **Memory bandwidth:** 128 × 128 × 4 bytes = 64 KB per layer, easily cached (L1: 32 KB per core)
- **FMA instruction latency:** 4 cycles on modern CPUs
- **Vector loop overhead:** 16 iterations vs 128 iterations = ~8x fewer branch/loop overheads
- **Horizontal sum (m256_hsum):** 4 shuffle operations per dot product
- **Expected practical gain:** **5-8x** on SSM loops

## Compiler Integration

### Requirements

- **Compiler:** MSVC (via nvcc on Windows) or clang/GCC with `-mavx2` flag
- **CPU:** Any x86-64 CPU from 2013+ (Haswell era)
- **Runtime Detection:** None (AVX2 assumed available; could add CPUID check if needed)

### Build Flags

**Current:** No explicit flag needed (MSVC enables AVX2 by default on x86-64)

**Optional:** Add to build script:
```bash
nvcc -x cu ... -DAVX2 ...  # Force AVX2 (not needed, it's default)
```

### Fallback

If AVX2 is unavailable at compile time (e.g., very old CPU), the `#else` branch provides scalar fallback code—identical to the original implementation.

## Testing & Verification

### Build Status
✅ **Successful.** Both `layer_ops.cu` and `one_layer.cu` compile without errors.

### Correctness Testing Needed

1. **Functional test:** Run BREAD with SIMD vs scalar on same prompt, compare outputs
   ```bash
   ./bread.exe --prompt "Hello" --tokens 50
   # Should produce same output as before (SIMD is mathematically equivalent)
   ```

2. **Accuracy test:** Compare against llama.cpp reference per-layer hidden states

3. **Regression test:** Ensure full inference still works with tool-calling in AGENCY

### Performance Benchmarking

Run timing script:
```powershell
# Baseline (old scalar)
$before = (Measure-Command { .\bread.exe --prompt "2+2" --tokens 20 }).TotalMilliseconds

# With SIMD (new)
$after = (Measure-Command { .\bread.exe --prompt "2+2" --tokens 20 }).TotalMilliseconds

# Speedup = $before / $after
```

**Expected result:** ~2-3x faster on SSM-heavy layers (layers with SSM branch active)

## Architecture Notes

### Limitations

- **Assumes key_dim % 8 == 0:** Works for key_dim = 128 (Qwen3.5). Would need cleanup loop for other dims.
- **Assumes alignment:** `loadu/storeu` handle unaligned loads; could use `load/store` if guaranteed aligned.
- **No prefetch:** Could add `_mm_prefetch()` to hide memory latency for next iteration.

### Potential Further Optimizations

1. **BLAS integration:** Use OpenBLAS/MKL for dot products (phases 2 & 4)
   - Might be overkill for key_dim=128 (not memory-bound)
   - Worth trying if SIMD alone doesn't hit expected gains

2. **Loop fusion:** Combine phases 1+3 (decay + state update)
   ```c
   // Combined: state[i][k] = decay*state[i][k] + k[k]*delta[i]
   state_v = _mm256_fmadd_ps(k_v, d_v, _mm256_mul_ps(state_v, decay_v));
   ```

3. **Interleave with GPU:** Move SSM to GPU kernels to eliminate CPU↔GPU sync overhead
   - Bigger effort but 2-3x potential gain

4. **OpenMP parallelization:** Parallelize outer `vi` loop across cores
   - key_dim=128, each iteration is ~100-200 CPU cycles
   - Good parallelization candidate with shared state arrays

## Files Modified

1. **C:\bread_v2\layer_ops.cu** (lines 336–410)
   - Added `m256_hsum()` helper
   - Wrapped `cpu_delta_net_autoregressive_step()` with `#ifdef __AVX2__`
   - 4 SIMD loops (phases 1–4)
   - Scalar fallback in `#else`

2. **C:\bread_v2\one_layer.cu** (lines 530–637)
   - Same changes to static version
   - Added `m256_hsum_local()` helper (local scope to avoid conflicts)

## Integration with AGENCY

AGENCY CLI will automatically benefit:
1. SSM layers execute faster
2. Overall token latency decreases (15-30 ms saved per token if SSM is bottleneck)
3. No changes needed to AGENCY code—it calls BREAD via subprocess

## Next Steps

1. **Test correctness:** Run same prompt with old/new binary, verify identical output
2. **Benchmark:** Measure actual speedup on typical prompts (20-50 tokens)
3. **Profile:** Use timing to confirm SSM is now <2ms per layer (was 0.5-1ms each)
4. **If speedup < 2x:** Consider loop fusion or BLAS integration
5. **If speedup > 5x:** Successful; move to OpenMP or GPU acceleration

## Summary of Approach

| Phase | Technique | Estimated Gain | Complexity |
|-------|-----------|-----------------|------------|
| 1 | SIMD (this) | 5-8x on SSM | LOW ✅ |
| 2 | Loop fusion | 1.2x extra | LOW |
| 3 | BLAS for dot products | 1.5x extra | MEDIUM |
| 4 | OpenMP parallelization | 2-4x (4-8 cores) | MEDIUM |
| 5 | Move SSM to GPU | 2-3x total | HIGH |

---

**Status:** ✅ Implemented and compiled successfully. Ready for testing.
