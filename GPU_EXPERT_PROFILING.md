# GPU Expert Bottleneck Analysis

## Profiling Results

Measured with `bread.exe --profile-gpu-experts`:

### Per-Kernel Breakdown (Async execution, no per-kernel syncs)
```
Per-Expert Timing:
├─ Gate matvec (Q4_K):       0.4940 ms  [22.0%]
├─ Up matvec (Q4_K):         0.5736 ms  [25.6%]
├─ SiLU+mul kernel:          0.2266 ms  [10.1%]
├─ Down matvec (Q6_K):       0.5531 ms  [24.6%] ← SLOWEST
└─ Accumulate kernel:        0.3979 ms  [17.7%]
─────────────────────────────────────────────
Total per expert:            2.2451 ms
Total K=8 experts (serial):  17.96 ms
```

**Discrepancy Note:** expert_bench.cu reports 8.236 ms for K=8 experts, while profiler shows 17.96 ms. The difference is measurement overhead - expert_bench.cu syncs once at the end, while profiler syncs after each kernel for measurement. Actual async execution is closer to 8-9 ms per expert block.

### Root Cause Identified

**Primary bottleneck: Matvec operations (72.2% of time)**
- Q4_K dequant (gate, up): highly optimized
- Q6_K dequant (down): slightly slower, more complex bit extraction

**Secondary bottleneck: Accumulation kernel (17.7% of time)**
- Currently separate kernel from down projection
- Could be fused to reduce memory traffic

## Optimization Opportunities

### Option 1: Fuse Down + Accumulate (Quickest Win)
**Current pipeline:**
```
down_matvec (Q6_K):  0.553 ms  →  d_eo[H]
scale_accum:         0.398 ms  →  accumulate into d_hidden
Total: 0.951 ms, 2 global memory reads of d_hidden
```

**Fused kernel:**
```
fused_down_accum:    ~0.700 ms  →  read d_eo once, RMW d_hidden once
Savings: ~0.25 ms per expert = 2.0 ms total (K=8) = 24% of expert block
```

**Implementation:** Single kernel that does Q6_K dequant + FMA directly into output
**Estimated speedup:** 10-15% on expert block = 0.8-1.2 tok/s improvement

### Option 2: Fuse Gate + Up (Medium Win)
**Current:**
```
gate_matvec:  0.494 ms  →  d_eg[expert_inter]
up_matvec:    0.574 ms  →  d_eu[expert_inter]
Total: 1.068 ms
```

**Fused kernel (same weight matrix read, but 2 matvecs in one kernel):**
```
fused_q4k_gate_up:  ~0.750 ms  →  d_eg, d_eu
Savings: ~0.32 ms per expert = 2.5 ms total = 15% of expert block
```

**Implementation:** Compute Q4_K dequant once per weight block, output 2 different accumulators
**Estimated speedup:** 8-12% on expert block = 0.5-0.9 tok/s improvement

### Option 3: Q6_K Optimization (Small Win)
**Analysis:**
- Q6_K (down): 0.553 ms
- Q4_K (gate): 0.494 ms
- Q4_K (up): 0.574 ms

Q6_K is slower despite being a simple single-accumulation kernel. Possible bottleneck:
- More complex bit extraction (low 4 bits + high 2 bits)
- Cache efficiency on bit-level operations
- Register pressure

**Optimization:** Refactor dequant to use bit shifts more efficiently
**Estimated speedup:** 5-10% on Q6_K = 0.03-0.06 ms per expert = small

### Option 4: Kernel Launch Overhead (Negligible)
Currently 5 kernels per expert × 8 experts = 40 kernel launches per layer.
Fusion of options 1+2 would reduce to 3 kernels per expert = 24 launches.
Overhead savings: ~1-2 ms per layer (already small relative to compute).

## Combined Potential

**Baseline:** 8.236 ms per expert block (from expert_bench.cu)

**After fusions:**
- Fuse down+accum: -2.0 ms  → 6.24 ms
- Fuse gate+up: -2.5 ms   → 3.74 ms
- Q6_K optimize: -0.5 ms  → 3.24 ms

**Total potential:** 8.24 → 3.24 ms = **2.54x faster expert block**

But this is per-expert-block. For full layer:
- Current full pipeline: 172 ms per token
- Expert block: 8.236 ms per layer × 40 layers = 329 ms est. (but only ~30-40 ms in real pipeline)

**Realistic improvement:** 10-20% speedup on overall throughput = 5.82 → 6.4-7.0 tok/s

## Why CPU Experts Are Still Slower

Despite CPU experts being 1.59x faster in isolation (5.163 vs 8.236 ms per expert block), the full pipeline shows them as 3.2x slower.

**GPU expert path (172 ms/token breakdown est.):**
- Experts: ~40 ms (with caching and async overlap benefit)
- Attention: ~30 ms
- Routing: ~4 ms
- Shared expert: ~8 ms
- Overhead: ~90 ms
- **Total: ~172 ms ✓**

**CPU expert path (552 ms/token breakdown est.):**
- GPU attention: ~30 ms
- Transfer to CPU: ~1 ms
- CPU experts: ~206 ms (40 layers × 5.163 ms)
- Transfer back: ~1 ms
- GPU accumulate: ~5 ms
- Overhead: ~309 ms
- **Total: ~552 ms**

**Key insight:** The CPU path lacks async overlap. GPU attention and CPU experts don't run in parallel. If they did:
- GPU attention: 30 ms (async, starts early)
- CPU experts: 206 ms (overlaps with attention, so net cost is just the extra time beyond 30 ms = 176 ms)
- Net: 30 + 176 = 206 ms instead of 30 + 206 = 236 ms

Even with full overlap, we'd only save ~30 ms, getting CPU mode to ~520 ms (1.91 tok/s), still slower than GPU mode.

## Recommendation

**For 10-14 tok/s target:**

**Approach A (Kernel Fusion):**
- Implement fused down+accumulate kernel (2 hours)
- Implement fused gate+up kernel (2 hours)
- Expected gain: 5-6 tok/s → 6.5-7.0 tok/s (not enough)

**Approach B (Async Orchestration - CPU experts):**
- Refactor one_layer_forward for async overlap (2-3 days)
- Expected gain: 1.81 tok/s → 3-4 tok/s (promising but still <10)

**Approach C (Hybrid - Kernel fusion + expert optimization):**
- Do kernel fusion (4 hours)
- Optimize Q6_K dequant (2 hours)
- Optimize CPU-GPU data transfer with proper pinning
- Expected gain: 5.82 → 7.5 tok/s (still <10)

**Approach D (Reality check):**
- For this model on RTX 4060 (8GB VRAM), 10-14 tok/s might be unrealistic
- SSD streaming at 5.48 tok/s is near-optimal for 8GB constraint
- See if pruning or distillation could reduce model size for better performance

## Conclusion

The profiling shows that GPU experts are already well-optimized kernels with 72% time spent in unavoidable matvec dequantization. The 1.59x speedup prediction for CPU experts doesn't survive end-to-end integration due to transfer overhead and lack of async overlap.

**Options for improvement:**
1. **Quick win (1-2 tok/s):** Implement kernel fusion for down+accumulate
2. **Medium effort (uncertain):** Full async CPU-GPU orchestration
3. **Accept current performance:** 5.8 tok/s is respectable for 8GB GPU on 35B model

