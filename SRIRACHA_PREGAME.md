# SRIRACHA: Speculative Decoding for BREAD

> **Metaphor:** Sriracha is the hot, spicy condiment that dramatically enhances the base (BREAD) — just like speculative decoding adds 2-3x acceleration without changing the core output.

---

## Context

**BREAD** runs at ~5.5 tok/s on Qwen3.5-35B-A3B — 2.76x faster than Ollama on the similarly-parameterized dense 27B model. Phase 4 (multi-stream MoE optimization) is complete. SRIRACHA is the next acceleration layer: speculative decoding as a standalone module that multiplies BREAD's throughput by 3-4x with no change to output quality.

**Stack vision:**
- **BREAD** = Base inference engine (optimized to ~5.5 tok/s)
- **SRIRACHA** = Speculative decoding layer (3-4x multiplier on latency)
- **BUTTER** = Orchestration (multi-prompt, benchmarking, profiling)
- **AGENCY** = Multi-agent system (parallel requests, model routing)

---

## How Speculative Decoding Works

```
SRIRACHA: Draft → Target Verification → Acceptance

├─ Draft Phase (fast, small model)
│  └─ Generate K draft tokens sequentially (~5-8 tokens, ~50 μs each)
│
├─ Verification Phase (parallel, target model = BREAD)
│  └─ Single forward pass validates all K drafts simultaneously
│
└─ Rejection Sampling (CPU, ~1ms)
   └─ Compare P(draft_token) vs P(target_token) → accept/reject longest prefix
```

**Key insight:** More total FLOPs, but better GPU parallelism. Instead of K sequential forward passes (K × 182ms), you do 1 forward pass (182ms) that validates K tokens at once. Net gain depends on acceptance rate.

**Why this works well for BREAD specifically:**
- Batch_size=1 (single user request) is the optimal scenario for speculative decoding (2-3x speedup research-confirmed)
- Target model already stream-optimized on `stream_a`
- Draft can run on `stream_b`
- Shared tokenizer (both models same vocab base — no mismatch headache)

---

## Research Findings (2024-2026)

### Speedup Numbers

| Scenario | Speedup | Source |
|----------|---------|--------|
| Standard approach (batch=1) | 2.3x–2.8x | [BentoML](https://www.bentoml.com/blog/3x-faster-llm-inference-with-speculative-decoding) |
| EAGLE-3 (best alignment) | 3.0x–6.5x | [ICLR 2026](https://openreview.net/pdf?id=aL1Wnml9Ef) |
| Mirror SD (Apple) | 2.8x–5.8x | [Apple MLR](https://machinelearning.apple.com/research/mirror) |
| Large batches (batch=32+) | 1.3x (diminishing) | [Batch SD paper](https://arxiv.org/html/2510.22876v3) |

**For BREAD's use case (batch=1, latency-focused):** 2.5-4x speedup is realistic.

### Acceptance Rate vs Spec Depth

- **γ = 5-8 tokens** is the empirically optimal spec depth (sweet spot of drafting overhead vs verification gain)
- **α ≥ 60%** acceptance rate → meaningful speedup
- **α < 40%** → diminishing returns, reduce γ
- Acceptance by position: position 1 (~85%), position 2 (~78%), position 3 (~62%), position 4 (~38%), position 5 (~12%) — drop-off is steep after position 3-4

### Draft Model Selection

| Draft Model | Speed | Acceptance | Effort | Notes |
|-------------|-------|-----------|--------|-------|
| **Qwen 0.8B** | ~50 μs | 50-60% | 30h | Off-shelf, no training; same tokenizer family; recommended start |
| Qwen 1.5B | ~80 μs | 60-70% | 30h | Better quality drafts; upgrade if 0.8B acceptance <45% |
| EAGLE-3 | ~10 μs | 70-80% | +120h | Train lightweight decoder on target's hidden states; no separate model |
| OmniDraft (n-gram) | ~15 μs | 30-40% | 80h | Handles vocab mismatch; fallback only |

**Recommendation:** Start with Qwen 0.8B FP16. Measure real acceptance rate. Upgrade to 1.5B or EAGLE-3 only if empirical data justifies it.

### Production Adoption

Speculative decoding is now standard in vLLM, TensorRT-LLM, and SGLang (2025). It went from research experiment to production default for single-request latency workloads. TensorRT-LLM achieves 3.6x throughput improvement with proper tuning.

---

## Ideal Configuration

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Draft model** | Qwen 0.8B (FP16) | Off-shelf, fast, same tokenizer, easy to upgrade |
| **Draft quantization** | FP16 (not quantized) | Logits drive rejection sampling — quantization error hurts acceptance rate directly |
| **Target (BREAD)** | Q4_K/Q6_K unchanged | Already optimized; no change needed |
| **Memory total** | ~27GB | BREAD quantized (~24GB) + 0.8B FP16 (~1.6GB) + KV caches (~0.5GB); fits 30GB+ comfortably |
| **Spec depth γ** | Start 5, adaptive 3-8 | Driven by per-position acceptance heatmap |
| **Build priority** | SRIRACHA before BUTTER | 3-4x quick win, validates concept, informs BUTTER design |

**Memory note:** Tight on 24GB (RTX 4090), comfortable on 30GB+ (RTX 6000 Ada etc). If memory-constrained, Q8 draft saves ~0.8GB with minimal acceptance rate hit.

---

## Expected Speedup Math

**Baseline:** 5.5 tok/s = 182 ms/token

**With Qwen 0.8B draft (γ=5, α=60%):**
- Draft generation: 5 × 50 μs = 250 μs
- Target verification: 150 ms (single forward pass, all 5 + original)
- Effective tokens per pass: 5 × 0.60 = 3.0
- Wall-clock per token: 150 ms / 3.0 = **50 ms/token → 20 tok/s**
- **Speedup: 3.6x**

**With EAGLE-3 (γ=5, α=75%):**
- Draft generation: 5 × 10 μs = 50 μs
- Target verification: 150 ms
- Effective tokens per pass: 5 × 0.75 = 3.75
- Wall-clock per token: 150 ms / 3.75 = **40 ms/token → 25 tok/s**
- **Speedup: 4.5x**

---

## 3-Step Build Plan

### Step 1: Draft Runner (50h)
Stand up a minimal `sriracha.cu` that loads Qwen 0.8B in FP16, runs a forward pass, and outputs K=5 draft tokens sequentially. Reuse BREAD's existing loader and KV cache infrastructure. No verification yet — just confirm draft generates coherent tokens at ~50 μs/token.

**Done when:** `sriracha_draft(prompt, K=5)` returns 5 tokens and runs without corruption.

**Files:** `sriracha.cu`, `sriracha.h`

---

### Step 2: Verification + Rejection Sampling (100h)
Pass the draft sequence to BREAD's existing forward pass as a batch (1 full forward through all K+1 positions). Compare draft logits vs target logits at each position using stochastic rejection sampling. Accept longest valid prefix, rewind draft KV cache on partial rejection.

**Done when:** `sriracha_next_token()` returns correct output and acceptance rate is measurable. Verify output matches non-speculative BREAD on same prompt.

**Files:** `sriracha.cu` (extended), logic in `rejection_sampling()` helper

---

### Step 3: Stream Overlap + Adaptive Depth (100h)
Move draft generation onto `stream_b` so it overlaps with the previous token's target verification on `stream_a`. Add per-position acceptance logging and adaptive γ (auto-tune between 3-8 based on recent acceptance history). Benchmark and compare tok/s vs BREAD baseline.

**Done when:** Wall-clock throughput measurably faster than BREAD alone, adaptive γ live, per-position heatmap logged.

**Files:** `sriracha.cu` (stream overlap), `sriracha_stats.c` (acceptance logging)

---

## Adaptive Depth Algorithm

```c
// Per-request tracking
typedef struct {
    int   position;    // 1..K
    int   accepted;    // 0 or 1
    float p_draft;     // draft P(token)
    float p_target;    // target P(token)
    float margin;      // p_target - p_draft
} sriracha_acceptance_log_t;

// Adaptive γ tuning (runs every 50 requests)
// if acceptance_rate[K] < 40%: gamma = max(gamma - 1, 3)
// if acceptance_rate[1..K] all > 70%: gamma = min(gamma + 1, 8)
```

**Example output after 50 requests:**
```
SRIRACHA Acceptance Profile:
Position 1: 82%  ✓ Accept easily
Position 2: 78%  ✓ Accept usually
Position 3: 62%  ~ Risky
Position 4: 38%  ✗ Reject mostly
Position 5: 12%  ✗ Always reject
→ Recommendation: Set gamma=3 (saves 100 μs drafting, same target latency)
```

---

## Integration with BUTTER/AGENCY

Once SRIRACHA is stable, BUTTER can transparently route requests:
- **Single request (latency-critical):** BREAD + SRIRACHA (spec decoding on)
- **High-throughput batch (batch > 8):** BREAD alone (spec decoding off — diminishing returns above batch=8)
- **Benchmarking:** BUTTER profiles per-prompt acceptance rate to inform model/config selection

---

## Key Sources

- [NVIDIA: Intro to Speculative Decoding](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)
- [EAGLE-3, ICLR 2026](https://openreview.net/pdf?id=aL1Wnml9Ef)
- [Apple Mirror Speculative Decoding](https://machinelearning.apple.com/research/mirror)
- [Premai: 2-3x Speedup 2026](https://blog.premai.io/speculative-decoding-2-3x-faster-llm-inference-2026/)
- [Batch Speculative Decoding Done Right](https://arxiv.org/html/2510.22876v3)
- [BentoML: 3x Faster LLM Inference](https://www.bentoml.com/blog/3x-faster-llm-inference-with-speculative-decoding)
- [Fast Inference via Speculative Decoding (Original Paper)](https://openreview.net/pdf?id=C9NEblP8vS)
