# 3-Paper Analysis Report: Novel MoE & Attention Optimization

---

## Paper #1: FloE — On-the-Fly MoE Inference on Memory-constrained GPU

**Status:** ICML 2025 (accepted), arxiv 2505.05950
**Relevance to BREAD:** ⭐⭐⭐⭐ (highest potential)

### Core Contribution

FloE solves the expert offloading bottleneck by compressing expert parameters **asymmetrically**:
- **Gate & Down projections:** Contextual sparsification (remove weights tied to low-magnitude activations)
- **Up projection:** Ultra-low-bit quantization (INT2)
- This exploits the fact that different expert sub-modules have different quantization sensitivity

### Key Numbers

| Metric | Result |
|--------|--------|
| Parameter compression | 9.3× per expert |
| Memory footprint reduction | 8.5× |
| Speedup vs DeepSpeed-MII | 48.7× (RTX 3090) |
| Quality loss | 4.4–7.6% |
| VRAM needed | 11 GB (vs. 80+ GB baseline) |

### How It Works

1. **Inter-expert predictor:** Predicts which experts activate in next layer (uses hidden state similarity, >0.95 cosine with prev layer)
2. **Intra-expert predictor:** Estimates which parameter weights in up-projection matter (no extra memory)
3. **Hybrid DMA:** Decompress on-the-fly during matmul, overlap with GPU compute

### Applicability to BREAD (RTX 4060, 8 GB VRAM)

**High applicability, moderate integration complexity**

- **Best case:** 2–4× expert throughput improvement (Mixtral numbers don't directly scale to Qwen3.5, but principle is sound)
- **Effort:** 40–60 hours (implement contextual sparsification analysis + inter-expert predictor + int2 dequant kernel)
- **Trade-off:** 5–8% quality loss vs. 2–4 tok/s gain

**Why this matters for BREAD:**
- BREAD's expert DMA overhead dominates (160 ms per layer × 40 layers). Reducing expert size by 8–9× would compress this significantly
- RTX 4060's 8 GB VRAM is the hard constraint. FloE's 11 GB example (for Mixtral-8×7B) suggests 8 GB is tight but feasible for Qwen3.5-35B's experts
- Contextual sparsification aligns with MoE principle: not all expert weights matter for every token

**Limitations:**
- Evaluated on Mixtral only; generalization to Qwen3.5's gating/routing is unproven
- Inter-expert predictor has ~12% miss rate (wrong expert predicted → CPU stall)
- Performance degradation grows on complex reasoning (MMLU scores dropped)
- Sparse kernels are GPU-specific; unclear how to port to int2 dequant on CUDA

**Honest risk:** This is the most novel and high-upside option, but it requires implementing two new components (contextual sparsification analysis + learned predictor) that don't exist in BREAD yet. Estimated time to production quality: 3–4 weeks.

---

## Paper #2: MoE-Gen — High-Throughput MoE Inference on a Single GPU with Module-Based Batching

**Status:** arxiv 2503.09716 (March 2025)
**Relevance to BREAD:** ⭐⭐ (low, batch-size-specific)

### Core Contribution

MoE-Gen replaces traditional **model-based batching** (unified batch size across all layers) with **module-based batching** (different batch sizes for attention vs. expert modules).

The key insight: At batch=1, both attention and experts have terrible GPU utilization. But if you accumulate multiple prompts and dispatch larger batches to experts, GPU utilization skyrockets.

### Key Numbers

| Scenario | Speedup vs. baseline |
|----------|---------------------|
| 8 GPU cluster → 1 GPU (236B model) | 8–31× |
| Decoding (16–31×) | **Not applicable to token generation at scale** |
| Models enabled | 236B–671B on consumer GPU |

### How It Works

1. **Accumulation:** Hold incoming requests in host memory queue
2. **Dynamic dispatch:** Fire large expert batches to GPU (e.g., 256 tokens → 128 attention microbatches, then 256-token expert batch)
3. **KV offloading:** Move attention KV cache to host RAM, free GPU for experts
4. **DAG scheduling:** Overlap CPU→GPU transfers with GPU compute

### Applicability to BREAD (Single-user, batch=1)

**Low applicability**

- **Batch=1 case:** MoE-Gen explicitly shows "comparative advantage diminishes at tiny batch sizes"
- **Single-user constraint:** BREAD is interactive (one prompt, one stream). Accumulation doesn't apply.
- **Throughput ceiling:** MoE-Gen's 8–31× is for batches of 8–256. At batch=1, it defaults to on-demand expert loading (same as current BREAD)

**Why it's not useful here:**
- The 31× speedup comes from amortizing expert DMA + GPU idle time across 256 tokens
- BREAD already processes one token at a time. There's nothing to accumulate
- If you modified BREAD to support multi-user batching, this would become relevant (future work: BUTTER serving layer)

**Verdict:** Architecturally sound, but orthogonal to current single-user BREAD. Revisit when building BUTTER (multi-user orchestration layer).

---

## Paper #3: FlashAttention-3 — Fast and Accurate Attention with Asynchrony and Low-precision

**Status:** arxiv 2407.08608 (July 2024, follow-up to FlashAttention-2 from July 2023)
**Relevance to BREAD:** ⭐ (low for batch=1, token generation)

### Core Contribution

Three improvements over FlashAttention-2:

1. **Warp-specialization asynchrony:** Split warps into producers (data movers) and consumers (compute), eliminating data movement bottleneck
2. **GEMM-softmax pipelining:** Overlap softmax (3.9 TFLOPS on H100) with matmul (989 TFLOPS) to hide softmax latency
3. **FP8 quantization:** Block-wise quantization reduces quantization error 2.6× vs. per-tensor scaling

### Key Numbers (H100 GPU, very large batches / sequences)

| Scenario | Improvement |
|----------|-------------|
| FP16 forward | 1.5–2.0× vs. FlashAttention-2 |
| FP16 backward | 1.5–1.75× vs. FlashAttention-2 |
| FP8 throughput | 1.2 PFLOPs/s |
| Sequence length sweet spot | 2K–4K tokens |

### Applicability to BREAD (Batch=1, token generation)

**Very low applicability**

- **Batch=1 token generation:** Uses <1% GPU utilization (sequence length = 1, query length = 1)
- **Paper acknowledgment:** "Optimizing for LLM inference remains future work"
- **FP8 limitation:** Lacks persistent kernel design for small-sequence causal masking (exactly BREAD's scenario)
- **GPU saturation curve:** FlashAttention-3 reaches peak efficiency at seq_len 2K+, batch 32+; BREAD runs at seq_len 1, batch 1

**Why it doesn't help:**
- The warp-specialization asynchrony is a fix for batched training/long-sequence inference
- At batch=1, there are not enough tokens to fill warps efficiently
- BREAD's attention is already memory-efficient in absolute terms (KV cache is only 200 MB)

**Honest assessment:** FlashAttention-3 is a brilliant paper for large-batch inference and training. It is not a lever for token-generation latency on single-device consumer hardware.

---

## Summary Table: 3 Papers Ranked by Applicability to BREAD

| Paper | Problem Solved | Applicability | Implementation Cost | Expected Gain | Confidence |
|-------|----------------|----------------|---------------------|--------------|------------|
| **FloE** | Expert DMA overhead via compression | ⭐⭐⭐⭐ High | 40–60 hrs | 2–4 tok/s | Medium |
| **MoE-Gen** | GPU utilization at large batch sizes | ⭐⭐ Low | — (skip) | N/A | N/A |
| **FlashAttention-3** | Training/large-batch inference speed | ⭐ Very Low | — (skip) | ~0 tok/s | N/A |

---

## Recommendation

**Do NOT pursue MoE-Gen or FlashAttention-3 for BREAD.** They solve different problems (multi-batch inference, training). They contribute zero to single-user, single-token-at-a-time architecture.

**FloE is worth the investment** — but only after verifying the top 3 (Phase 6 FMA + SSM AVX2 + CPU expert exec) are complete and working. Here's why:

1. **Risk ordering:** Finish low-risk, high-payoff incremental work first (18 hours → 2–4 tok/s)
2. **Then gamble:** If you hit 8–10 tok/s and want 12–14 tok/s, FloE's compression + sparse prediction is the right next move (40–60 hours → 2–4 more tok/s)
3. **Fallback:** If FloE doesn't generalize to Qwen3.5, you still have a working 8–10 tok/s engine

**Timeline suggestion:**
- **Week 1–2:** Phase 6 FMA (verify), SSM AVX2, CPU expert exec → 8–10 tok/s
- **Week 3–4:** FloE prototype (contextual sparsification + inter-expert predictor) → 12–16 tok/s
- **Parallel (optional):** Prepare SRIRACHA speculative decoding to 3–4× effective throughput

---

## References

- [FloE: On-the-Fly MoE Inference on Memory-constrained GPU (arxiv 2505.05950)](https://arxiv.org/abs/2505.05950)
- [MoE-Gen: High-Throughput MoE Inference on a Single GPU (arxiv 2503.09716)](https://arxiv.org/html/2503.09716v1)
- [FlashAttention-3: Fast and Accurate Attention (arxiv 2407.08608)](https://arxiv.org/html/2407.08608v1)
