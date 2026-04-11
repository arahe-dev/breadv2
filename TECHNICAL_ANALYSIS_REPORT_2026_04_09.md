# BREAD Technical Analysis Report

Date: 2026-04-09  
Scope: BREAD custom CUDA/C inference engine for Qwen3.5-35B-A3B on RTX 4060 Laptop (8 GB VRAM)

## Executive Summary

BREAD is now a correct, end-to-end custom MoE inference engine with working tokenizer, layer execution, quantized matvec kernels, MoE routing, full-attention, and SSM/GatedDeltaNet. Its current limitation is not correctness but execution efficiency. The engine is materially slower than mature runtimes because the hot path still spends too much time:

- synchronizing streams
- copying activations and logits between GPU and CPU
- running routing and recurrent math on the host
- executing sequential prefill and decode with a largely imperative schedule

The most important architectural observation is that BREAD already contains two strong performance ideas that are not fully realized in the current path:

1. a preallocated buffer-pool subsystem
2. an MoE-oriented expert streaming/prefetch design

However, the current inference path only partially uses them. The live path initializes the buffer pool but still relies on separate static allocations inside `one_layer_forward()`, and it appears to preload all experts into VRAM-facing allocations instead of relying on the original slot/LRU streaming design.

The highest-confidence optimization path is therefore:

1. remove host/device round-trips and sync stalls
2. wire the live path into the existing buffer pool
3. move routing and sampling onto the GPU
4. decide whether BREAD is truly a streaming/offload engine or a full-cache engine
5. only then consider deeper quantization/kernel redesign

## 1. BREAD Current State

### 1.1 Architecture Summary

BREAD’s runtime configuration is metadata-driven in [bread.c](C:/bread_v2/bread.c#L136) and [bread.c](C:/bread_v2/bread.c#L207). It parses:

- layer count
- embedding length
- attention head counts
- key/value dimensions
- full-attention interval
- expert counts and `top_k`
- RoPE dimensions and sections
- SSM dimensions and flags

Layer typing is derived from model metadata in [bread.c](C:/bread_v2/bread.c#L315) and [bread.c](C:/bread_v2/bread.c#L322), not hardcoded layer assumptions.

The main generation loop lives in [main.cu](C:/bread_v2/main.cu#L364). Prompt formatting is model-aware in [main.cu](C:/bread_v2/main.cu#L249), tokenization is handled in [tokenizer.c](C:/bread_v2/tokenizer.c#L927), and the loop calls [one_layer.cu](C:/bread_v2/one_layer.cu#L854) once per layer per token for both prefill and decode.

The engine structure is:

1. Load model into host RAM in [loader.c](C:/bread_v2/loader.c#L244)
2. Build runtime config from GGUF metadata in [bread.c](C:/bread_v2/bread.c#L136)
3. Load tokenizer metadata in [tokenizer.c](C:/bread_v2/tokenizer.c#L333)
4. Upload non-expert weights through the weight cache in [loader.c](C:/bread_v2/loader.c#L533)
5. Execute each layer through `one_layer_forward()`
6. Compute logits in [main.cu](C:/bread_v2/main.cu#L291)
7. Sample next token on CPU in [main.cu](C:/bread_v2/main.cu#L318)

### 1.2 Verified Working Components

Current docs indicate correctness is solved. The repo’s current source of truth is [CLAUDE.md](C:/bread_v2/CLAUDE.md#L211), [CLAUDE.md](C:/bread_v2/CLAUDE.md#L450), and [README.md](C:/bread_v2/README.md#L8), which state that:

- factual and math outputs are correct
- the tokenizer path is working
- the engine generates coherent chat output
- the fp16 subnormal bug affecting `attn_v.weight` was fixed

Strong working components in the current code:

- Q4_K and Q6_K CUDA matvec kernels in [kernels.cu](C:/bread_v2/kernels.cu#L150), [kernels.cu](C:/bread_v2/kernels.cu#L198), and [kernels.cu](C:/bread_v2/kernels.cu#L289)
- Qwen-aware tokenizer load and encode path in [tokenizer.c](C:/bread_v2/tokenizer.c#L333), [tokenizer.c](C:/bread_v2/tokenizer.c#L711), [tokenizer.c](C:/bread_v2/tokenizer.c#L767), and [tokenizer.c](C:/bread_v2/tokenizer.c#L927)
- Attention path with RoPE, GQA, gating, and KV cache in [one_layer.cu](C:/bread_v2/one_layer.cu#L650), [one_layer.cu](C:/bread_v2/one_layer.cu#L692), and [one_layer.cu](C:/bread_v2/one_layer.cu#L710)
- SSM/GatedDeltaNet path in [one_layer.cu](C:/bread_v2/one_layer.cu#L469), [one_layer.cu](C:/bread_v2/one_layer.cu#L513), and [one_layer.cu](C:/bread_v2/one_layer.cu#L530)
- MoE routing and shared-expert path in [one_layer.cu](C:/bread_v2/one_layer.cu#L797) and [one_layer.cu](C:/bread_v2/one_layer.cu#L1665)
- Metadata-driven model config in [bread.c](C:/bread_v2/bread.c#L136)

### 1.3 Current Performance and Bottlenecks

The repo’s current performance baseline is documented at roughly `5.5–6.6 tok/s` in [README.md](C:/bread_v2/README.md#L8), [CLAUDE.md](C:/bread_v2/CLAUDE.md#L611), and [PERFORMANCE_INVESTIGATION_DETAILED.md](C:/bread_v2/PERFORMANCE_INVESTIGATION_DETAILED.md).

The primary bottlenecks visible in the source are:

#### A. Frequent synchronization

The production layer path has many explicit stream syncs:

- [one_layer.cu](C:/bread_v2/one_layer.cu#L1333)
- [one_layer.cu](C:/bread_v2/one_layer.cu#L1365)
- [one_layer.cu](C:/bread_v2/one_layer.cu#L1475)
- [one_layer.cu](C:/bread_v2/one_layer.cu#L1591)
- [one_layer.cu](C:/bread_v2/one_layer.cu#L1682)
- [one_layer.cu](C:/bread_v2/one_layer.cu#L1756)
- [one_layer.cu](C:/bread_v2/one_layer.cu#L1795)

`compute_logits()` also ends with a device-wide sync in [main.cu](C:/bread_v2/main.cu#L312).

#### B. Host/device round-trips in the hot path

BREAD repeatedly copies projection outputs back to CPU:

- attention Q/K/V in [one_layer.cu](C:/bread_v2/one_layer.cu#L1381)
- SSM QKV/Z/alpha/beta in [one_layer.cu](C:/bread_v2/one_layer.cu#L1484)
- router input extraction in [one_layer.cu](C:/bread_v2/one_layer.cu#L821)
- logits for sampling in [main.cu](C:/bread_v2/main.cu#L318)

This is likely the single biggest structural reason BREAD underperforms `llama.cpp`.

#### C. CPU routing and CPU recurrent math

Routing is CPU-side in [one_layer.cu](C:/bread_v2/one_layer.cu#L797). The SSM recurrent step is also host-side, despite GPU-side projections.

#### D. Sequential prefill

Prompt prefill is still token-by-token in [main.cu](C:/bread_v2/main.cu#L590), forfeiting the usual batched-prompt gains.

#### E. Performance subsystems not fully integrated

BREAD initializes a preallocated buffer pool in [main.cu](C:/bread_v2/main.cu#L453), and the pool itself is implemented in [buffer_pool.c](C:/bread_v2/buffer_pool.c#L33), but the live `one_layer_forward()` path still uses separate static allocations in [one_layer.cu](C:/bread_v2/one_layer.cu#L933).

This means the repo contains a meaningful optimization subsystem that is not yet actually driving the hot path.

## 2. Comparison with llama.cpp

### 2.1 Tokenizer and Prompt Handling

`llama.cpp` handles tokenization through a central vocab/tokenizer pipeline with:

- special-token cache
- `tokenizer_st_partition`
- model-specific BPE pretokenization sessions

This lives in upstream `llama-vocab.cpp`.

BREAD now mirrors this more closely than earlier versions:

- token metadata load: [tokenizer.c](C:/bread_v2/tokenizer.c#L333)
- special-token scanning: [tokenizer.c](C:/bread_v2/tokenizer.c#L711)
- Qwen35 pretokenization: [tokenizer.c](C:/bread_v2/tokenizer.c#L767)
- top-level encode: [tokenizer.c](C:/bread_v2/tokenizer.c#L927)

This is an area where BREAD has substantially caught up semantically.

### 2.2 Layer Semantics

At the semantic level, BREAD is closer to `llama.cpp` than the older flowchart docs imply.

`llama.cpp`’s Qwen35MoE path uses:

- `ggml_rope_multi`
- `ggml_ssm_conv`
- `ggml_silu`
- `ggml_repeat_4d`
- `build_delta_net(...)`
- `build_moe_ffn(...)`

These correspond well to BREAD’s current imperative implementations:

- RoPE helper in [one_layer.cu](C:/bread_v2/one_layer.cu#L650)
- SSM conv and state update in [one_layer.cu](C:/bread_v2/one_layer.cu#L469), [one_layer.cu](C:/bread_v2/one_layer.cu#L513), [one_layer.cu](C:/bread_v2/one_layer.cu#L530)
- MoE routing and expert execution in [one_layer.cu](C:/bread_v2/one_layer.cu#L797) and [one_layer.cu](C:/bread_v2/one_layer.cu#L1717)

So the main difference is no longer “BREAD has missing model semantics.” The main difference is execution strategy.

### 2.3 Execution Model Differences

The biggest advantages `llama.cpp` has are architectural:

#### A. Graph-oriented execution

`llama.cpp` builds backend graphs over prompt/decode batches. BREAD executes an imperative per-layer schedule with explicit host control.

#### B. Fewer CPU/GPU seams

In `llama.cpp`, tokenization and model semantics are centrally coordinated and backend execution stays within the graph/backend model. BREAD runs projections on GPU, then repeatedly copies activations to CPU for recurrent math and routing.

#### C. Better batching story

`llama.cpp`’s architecture is naturally friendlier to batched prefill and backend-level optimization. BREAD’s prefill path is still sequential.

#### D. Cleaner synchronization

`llama.cpp` relies on backend scheduling rather than explicit `cudaStreamSynchronize()` calls distributed across the inner loop.

### 2.4 Where BREAD Is Novel or Better

BREAD has genuinely interesting systems ideas:

#### A. Explicit MoE orchestration surface

The slot-based expert loading and prefetch APIs in [loader.c](C:/bread_v2/loader.c#L325) and [loader.c](C:/bread_v2/loader.c#L393) make expert movement a first-class scheduling concept.

#### B. Debuggable correctness-first path

The existence of a minimal mode alongside the optimized path is a real engineering strength.

#### C. Full control over quant kernels

The custom kernels in [kernels.cu](C:/bread_v2/kernels.cu#L150) and [kernels.cu](C:/bread_v2/kernels.cu#L198) make BREAD a better platform for low-level decode optimization experiments than a black-box backend.

### 2.5 Major Architectural Drift in Current BREAD

The current code reveals a meaningful contradiction:

- BREAD’s conceptual design is expert streaming and orchestration
- but the current execution path preloads all experts through [loader.c](C:/bread_v2/loader.c#L628)
- and uses those cached pointers directly in [one_layer.cu](C:/bread_v2/one_layer.cu#L1732)

This means the current runtime is not really testing the architecture BREAD is supposed to be about.

That is important because it affects both performance and identity.

## 3. Research-Driven Improvements

This section focuses on reliable or plausibly valuable techniques from 2024–2025 literature, filtered for:

- 8 GB VRAM
- correctness preservation
- BREAD’s actual bottlenecks

### 3.1 Top Recommendations

#### 1. Device-side orchestration and kernel fusion

Rating: ✅ Reliable and applicable  
Expected gain: `15–30%`  
Effort: Medium

The Korch paper shows that execution order, fusion/fission choices, and orchestration can yield large gains without changing model semantics. BREAD’s hot path is an especially good candidate because it is broken into many tiny operations with explicit syncs.

Apply to BREAD:

- reduce sync boundaries in [one_layer.cu](C:/bread_v2/one_layer.cu#L1333) through [one_layer.cu](C:/bread_v2/one_layer.cu#L1795)
- fuse tiny elementwise stages where possible
- defer synchronization until data is truly consumed on host or another stream

Why it fits:

- directly targets BREAD’s biggest current bottleneck
- exact semantics preserved
- no model conversion required

Source:

- [Korch](https://arxiv.org/abs/2406.09465)

#### 2. Actually use the existing buffer pool

Rating: ✅ Reliable and applicable  
Expected gain: `8–15%`  
Effort: Low

This is the clearest local engineering win. The buffer pool is already implemented in [buffer_pool.c](C:/bread_v2/buffer_pool.c#L33), but the layer path still uses separate long-lived static allocations in [one_layer.cu](C:/bread_v2/one_layer.cu#L933).

Apply to BREAD:

- replace the `static` device scratch in `one_layer_forward()` with `bread_buffer_pool_get()`
- reuse the pool’s expert streams/events
- make the live path depend on the same allocation strategy the repo already claims to use

Why it fits:

- minimal correctness risk
- aligns implementation with documented optimization intent

#### 3. GPU-side routing and top-k

Rating: ✅ Reliable and applicable  
Expected gain: `5–12%`  
Effort: Medium

Routing over 256 experts is small enough that a device kernel can do it exactly and cheaply. Right now BREAD copies hidden state to CPU just to compute router logits and top-k in [one_layer.cu](C:/bread_v2/one_layer.cu#L797).

Apply to BREAD:

- replace `route_layer()` with a CUDA kernel or short kernel chain
- keep top-k expert indices and normalized weights on device
- eliminate D2H routing copies

Code areas:

- [one_layer.cu](C:/bread_v2/one_layer.cu#L797)
- [one_layer.cu](C:/bread_v2/one_layer.cu#L1678)

#### 4. GPU-side sampling

Rating: ✅ Reliable and applicable  
Expected gain: `3–8%`  
Effort: Low

The current decode loop copies the full vocab logits back and argmaxes on CPU in [main.cu](C:/bread_v2/main.cu#L318).

Apply to BREAD:

- device argmax kernel
- optional exact top-p / top-k sampler on GPU
- return only selected token ID

Code areas:

- [main.cu](C:/bread_v2/main.cu#L291)
- [main.cu](C:/bread_v2/main.cu#L318)

#### 5. Restore true expert streaming/prefetch if full-caching is not stable

Rating: ✅ Reliable if measured first  
Expected gain: variable, but high strategic value  
Effort: Medium to High

FloE is relevant because it focuses on memory-constrained MoE inference where expert movement is a first-class cost. BREAD’s original design clearly points in that direction, but the current path appears to preload all experts through [loader.c](C:/bread_v2/loader.c#L628).

Apply to BREAD:

- first measure whether current expert preloading is causing hidden memory pressure or paging
- if yes, revert to the slot/LRU expert path and optimize it
- overlap routing, prefetch, and execution using the existing loader streams

Code areas:

- [loader.c](C:/bread_v2/loader.c#L325)
- [loader.c](C:/bread_v2/loader.c#L393)
- [loader.c](C:/bread_v2/loader.c#L628)
- [one_layer.cu](C:/bread_v2/one_layer.cu#L1799)

Source:

- [FloE](https://arxiv.org/abs/2505.05950)

### 3.2 Promising Medium/Long-Term Techniques

#### 6. Requantize for a more efficient decode kernel target

Rating: ⚠️ Promising, major effort  
Expected gain: potentially `1.5–2x` in the long run  
Effort: High

QQQ shows strong W4A8 results with efficient kernels. BREAD currently targets GGUF Q4_K/Q6_K weight-only GEMV. A future BREAD v3 could benefit from quantization designed around decode kernels rather than inherited file formats.

Apply to BREAD:

- new weight format and conversion path
- new kernels replacing or complementing [kernels.cu](C:/bread_v2/kernels.cu#L150) and [kernels.cu](C:/bread_v2/kernels.cu#L198)

Source:

- [QQQ](https://arxiv.org/abs/2406.09904)

#### 7. MoE-specific extreme expert compression

Rating: ⚠️ Promising, model-format dependent  
Expected gain: strategic, not immediate  
Effort: High

MiLo and FloE are compelling if BREAD’s future identity is “custom converted MoE runtime for memory-tier hardware.” They are not immediate GGUF drop-in wins.

Sources:

- [MiLo](https://arxiv.org/abs/2504.02658)
- [FloE](https://arxiv.org/abs/2505.05950)

#### 8. KV-cache quantization and dynamic memory systems

Rating: ⚠️ Good later, not first priority  
Expected gain: more memory and serving scale than immediate decode speed  
Effort: Medium to High

KIVI and vAttention matter more once BREAD is pushing longer contexts or multiple requests.

Sources:

- [KIVI](https://arxiv.org/abs/2402.02750)
- [vAttention](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/vattention_arxiv24.pdf)

### 3.3 Not Recommended Right Now

#### 9. FlashAttention-3 style work

Rating: ⚠️ Good ideas, limited near-term payoff here

FlashAttention-3 is impressive, but BREAD is on an RTX 4060 Laptop, not Hopper, and only 10 of 40 layers are full-attention. This is not where the next large win is likely to come from.

Source:

- [FlashAttention-3 PDF](https://papers.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf)

#### 10. Speculative decoding

Rating: ❌ Not recommended

Per project constraints, low-acceptance speculative decoding is off-limits. LayerSkip-style self-speculation is exact after verification, but it requires training-time changes and is not aligned with current BREAD priorities.

Source:

- [LayerSkip](https://arxiv.org/abs/2404.16710)

## 4. Recommended Action Plan

### Phase 1: Quick Wins

Time: `1–2 days`  
Effort: `<5%`  
Expected gain: `>5%`, likely `10%+`

1. Replace static scratch/state ownership in [one_layer.cu](C:/bread_v2/one_layer.cu#L933) with the existing pool in [buffer_pool.c](C:/bread_v2/buffer_pool.c#L33)
2. Remove easy sync barriers and convert them to event-based dependencies
3. Move logits argmax sampling off CPU
4. Replace device-wide sync in [main.cu](C:/bread_v2/main.cu#L312) with the narrowest possible stream-scoped dependency

### Phase 2: Medium Effort

Time: `3–5 days`  
Effort: `10–20%`  
Expected gain: `10–20%`

1. Move expert routing and top-k fully to GPU
2. Keep SSM recurrent math on device to remove D2H activation copies
3. Use the pool’s expert streams/events for real parallel expert execution
4. Rework prefill to use a more batched path instead of strictly sequential tokens

### Phase 3: Foundational Changes

Time: `1+ week`  
Effort: Major rewrite  
Potential upside: `2–3x` only if executed well

1. Decide whether BREAD is:
   - a full expert-cache runtime, or
   - a true memory-tier expert-streaming runtime
2. If streaming/offload remains the thesis:
   - restore the slot/LRU/prefetch architecture as the real execution path
   - optimize it rather than bypassing it
3. Rebuild the inner loop around CUDA Graphs or a more graph-like execution schedule
4. Consider a new quantization target if decode throughput becomes the main frontier

## 5. Overall Assessment

BREAD is now significantly better than a hobby inference engine because it:

- runs a hard MoE model correctly
- owns its tokenizer, kernels, and layer execution path
- has a real systems thesis around expert orchestration

What keeps it below top runtimes today is not correctness but execution discipline. The current code still looks like a correct engine that has not yet been fully re-architected around its own optimization ideas.

That is good news. It means the next gains are highly actionable:

- fewer syncs
- fewer host/device copies
- actual use of the buffer pool
- device-side routing and sampling
- a clear decision about expert residency strategy

If those are done well, BREAD can become a genuinely competitive specialized MoE runtime on consumer hardware.

## Sources

### Local repo sources

- [CLAUDE.md](C:/bread_v2/CLAUDE.md)
- [README.md](C:/bread_v2/README.md)
- [PERFORMANCE_INVESTIGATION_DETAILED.md](C:/bread_v2/PERFORMANCE_INVESTIGATION_DETAILED.md)
- [OPTIMIZATION_PROGRESS.md](C:/bread_v2/OPTIMIZATION_PROGRESS.md)
- [PROJECT_MATURITY_ASSESSMENT.md](C:/bread_v2/PROJECT_MATURITY_ASSESSMENT.md)
- [main.cu](C:/bread_v2/main.cu)
- [one_layer.cu](C:/bread_v2/one_layer.cu)
- [kernels.cu](C:/bread_v2/kernels.cu)
- [loader.c](C:/bread_v2/loader.c)
- [bread.c](C:/bread_v2/bread.c)
- [tokenizer.c](C:/bread_v2/tokenizer.c)
- [buffer_pool.c](C:/bread_v2/buffer_pool.c)

### External primary sources

- [llama.cpp `llama-vocab.cpp`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-vocab.cpp)
- [llama.cpp `qwen35moe.cpp`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/qwen35moe.cpp)
- [llama.cpp `delta-net-base.cpp`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/delta-net-base.cpp)
- [Korch](https://arxiv.org/abs/2406.09465)
- [QQQ](https://arxiv.org/abs/2406.09904)
- [FloE](https://arxiv.org/abs/2505.05950)
- [MoE-Gen](https://arxiv.org/abs/2503.09716)
- [MiLo](https://arxiv.org/abs/2504.02658)
- [KIVI](https://arxiv.org/abs/2402.02750)
- [vAttention](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/vattention_arxiv24.pdf)
- [FlashAttention-3 PDF](https://papers.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf)
- [LayerSkip](https://arxiv.org/abs/2404.16710)
