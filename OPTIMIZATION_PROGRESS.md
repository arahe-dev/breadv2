# BREAD Performance Optimization Progress

**Date:** 2026-04-08
**Target:** 10+ tok/s (from current 5.73-6.62 tok/s)

---

## Baseline Measurements

| Test | Throughput | Latency | Notes |
|------|-----------|---------|-------|
| **BREAD GPU (current)** | 5.73-6.62 tok/s | 151-175 ms/token | 40 layers × ~4 ms each |
| **llama.cpp CPU** | 14.7 tok/s | 68 ms/token | Reference |
| **Gap** | 2.2-2.6x slower | 2.2-2.6x slower | Target: close to llama.cpp |

---

## Optimization Checklist

### Phase 1: Weight Caching ✅ DONE
- Status: Implemented (Phase 5, CLAUDE.md)
- Gain: +12.6% (measured)
- VRAM used: +1GB
- Code change: ~100 lines in loader.c
- Expected result: 5.73 → 6.62 tok/s

**Current status:** Already in production code

---

### Phase 2: Pre-allocated Buffers (IN PROGRESS)
- Status: Not implemented
- Estimated gain: +5-10%
- Target: Replace malloc/free with ring buffer
- Effort: 1-2 hours
- Files to modify: `one_layer.cu`

**Problem:**
```c
// Current (per token, 40 layers)
for (int layer = 0; layer < 40; layer++) {
    h_attn_out = malloc(cfg->attn_out_dim * sizeof(float));  // 40 allocations
    h_ssm_qrep = malloc(...);  // 40 allocations
    // ... 15+ more malloc calls

    // ... use buffers ...

    free(h_attn_out);  // 40 frees
    free(h_ssm_qrep);  // 40 frees
    // ... 15+ more free calls
}
```

**Solution:**
```c
// Proposed
static float buffer[TOTAL_SIZE];  // Pre-allocate once
static int offset = 0;

for (int layer = 0; layer < 40; layer++) {
    h_attn_out = buffer + offset;
    offset += cfg->attn_out_dim;

    h_ssm_qrep = buffer + offset;
    offset += (cfg->ssm_num_v_heads * cfg->ssm_head_dim);

    // ... use buffers ...
    // No free needed
}
if (offset > MAX) offset = 0;  // Wrap around
```

**Expected result:** 6.62 → 7.0-7.3 tok/s

---

### Phase 3: Kernel Fusion (TODO)
- Status: Not implemented
- Estimated gain: +10-15%
- Target: Fuse dequant + multiply in Q4_K kernel
- Effort: 2-3 hours
- Files to modify: `kernels.cu`

**Problem:**
```cuda
// Current (separate operations)
half x = dequant_q4k_elem(weights[i]);
sum += x * input[i];
```

**Solution (flash-moe style):**
```cuda
// Fused FMA
for (int i = 0; i < N; i += 8) {
    __m256 nibbles = load_q4k_8_nibbles(weights[i]);
    __m256 scale = load_q4k_scale(weights[i]);
    __m256 x = load_x_8(input[i]);
    sum = _mm256_fmadd_ps(_mm256_mul_ps(nibbles, scale), x, sum);
}
```

**Expected result:** 7.3 → 8.0-8.5 tok/s

---

### Phase 4: Queue Kernels (TODO)
- Status: Not implemented
- Estimated gain: +30-50%
- Target: Eliminate per-layer sync, queue all layers
- Effort: 4-6 hours (architectural)
- Files to modify: `main.cu`, `one_layer.cu`

**Problem:**
```cuda
// Current (blocking after each kernel)
for (int layer = 0; layer < 40; layer++) {
    bread_matvec(...);
    CUDA_CHECK(cudaStreamSynchronize(stream_a));  // STALL

    cpu_softmax(...);  // CPU waits for GPU
}
```

**Solution:**
```cuda
// Queue all kernels, sync once
for (int layer = 0; layer < 40; layer++) {
    bread_matvec(...);  // Queue only
    // DON'T sync
}
CUDA_CHECK(cudaStreamSynchronize(stream_a));  // Sync ONCE

// But need to restructure for routing/gating...
```

**Architectural challenge:** CPU needs layer output for:
- Router input (routing for MoE)
- Gate computation
- Activation function selection

**Possible solution:** Pre-compute routing for all layers before inference loop.

**Expected result:** 8.5 → 11-14 tok/s (would match/exceed llama.cpp)

---

## Testing Plan

After each optimization:
1. Compile with `-O2`
2. Run benchmark: `bread.exe --server --tokens 200 < test.txt`
3. Measure: tokens/sec (from final output)
4. Compare: against baseline
5. Profile: identify remaining bottlenecks

---

## Success Criteria

| Target | Improvement | Timeline |
|--------|------------|----------|
| 6.62 tok/s | Baseline (Phase 1 done) | NOW |
| 7.0-7.3 tok/s | +5-10% (Phase 2) | 2-3 hrs |
| 8.0-8.5 tok/s | +10-15% (Phase 3) | 2-3 hrs |
| 11-14 tok/s | +30-50% (Phase 4) | 4-6 hrs |

---

## Implementation Status

- ✅ Phase 1: Weight caching (DONE, ~6.62 tok/s)
- ⏳ Phase 2: Buffer pre-allocation (NEXT)
- ⏳ Phase 3: Kernel fusion (AFTER PHASE 2)
- ⏳ Phase 4: Queue kernels (HARDEST, LAST)

---

## Notes

- Phase 1 was already implemented (from Phase 5, CLAUDE.md)
- Phase 2-3 are straightforward optimizations
- Phase 4 is the big win but requires architecture changes
- Even without Phase 4, reaching 8-8.5 tok/s would be excellent

---

**Started:** 2026-04-08
**Last Updated:** 2026-04-08
**Status:** Planning phase 2 implementation
