# BREAD — CLAUDE.md

## What is BREAD?

BREAD is a from-scratch CUDA inference engine for large MoE (Mixture-of-Experts) models on consumer hardware. It is inspired by danveloper's flash-moe project (which ran Qwen3.5-397B on a MacBook Pro using Apple's "LLM in a Flash" paper), but targets NVIDIA CUDA instead of Apple Metal.

The core idea: most inference engines make you decide upfront how many layers go on GPU. BREAD profiles your actual hardware bandwidth at startup and makes those decisions dynamically, then streams expert weights from the appropriate memory tier (VRAM → RAM → SSD) using async CUDA operations so the GPU is never idle waiting for weights.

## Hardware Context

- GPU: NVIDIA RTX 4060 Laptop (8GB VRAM, ~136 GB/s internal bandwidth)
- CPU: Intel i7-13650HX
- RAM: 48GB DDR5 @ 4800 MT/s (~38 GB/s bandwidth)
- SSD: ~3 GB/s sustained sequential read (PCIe Gen 4 NVMe)
- OS: Windows 11, CUDA 13.2, VS BuildTools 2026 (MSVC 14.50)

## First Benchmark Target

**Qwen3.5-35B-A3B** (qwen35moe architecture)
- Quantization: Q4_K_M
- Size on disk: ~22GB
- Location: `C:\Users\arahe\.ollama\models\blobs\sha256-900dde62fb7ebe8a5a25e35d5b7633f403f226a310965fed51d50f5238ba145a`
- This is a raw GGUF file (no extension, readable directly)
- Parameters: 36B total, ~3B active per token
- Architecture: 256 experts per MoE layer, top-9 routing, hybrid GatedDeltaNet + full attention

## Memory Hierarchy for 35B

The full 22GB model fits in RAM. SSD streaming is not needed for this model.
- VRAM (8GB): attention layers, routing matrices, norms, active expert slots
- RAM (48GB): full model resident, experts streamed to VRAM on demand
- SSD: reserved for future 100B+ models that exceed RAM capacity

## Long-term Target

Run a 3-digit billion parameter model (e.g. Qwen3.5-397B-A17B) on this hardware by streaming experts from SSD, same as danveloper but on CUDA.

## Project Structure

```
C:\bread_v2\
├─ CLAUDE.md                        — this file (architecture overview)
├─
├─ === CORE INFERENCE ENGINE ===
├─ main.cu                          — entry point, inference loop (40 layers + embed/output)
├─ bread.h / bread.c                — model config parser (reads GGUF metadata)
├─ loader.h / loader.c              — expert weight streaming, LRU cache, async DMA
├─ kernels.cu                       — CUDA kernels (matvec, norm, etc.)
├─ gguf.h / gguf.c                  — GGUF file parser (metadata only, no weight loading)
├─
├─ === DEQUANTIZATION ===
├─ dequant_q4k_cpu.c                — CPU reference decompression (Q4_K and Q6_K)
│  (CUDA matvec kernels in kernels.cu dispatch to these for correctness)
├─
├─ === TOKENIZATION ===
├─ tokenizer.c / tokenizer.h        — Qwen3.5 BPE tokenizer (248K vocab)
├─
├─ === BUILD SYSTEM ===
├─ build.bat                        — top-level build wrapper (sets up MSVC env)
├─ build_main.bat                   — link bread executable
├─ build_kernels.bat                — compile kernels.cu
├─ build_loader.bat                 — compile loader.c
├─ build_dequant.bat                — compile dequant_q4k_cpu.c
├─ build_tok.bat                    — compile tokenizer.c
├─
├─ === DIAGNOSTICS (standalone) ===
├─ bread_info.c                     — dumps all 1959 tensors, types, shapes, offsets
├─ config_reader.c                  — extracts GGUF metadata, prints config
├─
├─ === REFERENCE IMPLEMENTATIONS ===
├─ infer_ref.m                      — MATLAB reference (SSM forward, dequant, etc.)
├─
├─ === SCRATCH ===
├─ hck_tmp.c                        — temp test code (not part of build)
└─
```

**Key insight:** The engine is split into 3 phases:
1. **Startup:** topology_probe() → gguf_open() → loader_init() → allocate VRAM slots
2. **Per-token inference:** for layer in 0..39: one_layer_forward() with prefetch overlap
3. **Output:** apply_norm() → compute_logits() → greedy_sample()

All malloc/free happens at startup; hot loop uses pre-allocated buffers + expert cache.

## Build Environment

Build via the convenience wrapper (sets up MSVC environment automatically):
```cmd
cd C:\bread_v2
build.bat
```

Or with nmake directly (requires vcvars64 already called):
```cmd
nmake
```

nvcc + cl.exe paths:
- MSVC:  C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64
- CUDA:  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin  (nvcc 13.2, on PATH)

## Architecture — How BREAD Works

### Startup sequence
1. `topology_probe()` — benchmarks RAM→VRAM and SSD→RAM bandwidth with actual memcpy calls, stores results in `topo_t`
2. `gguf_open()` — parses the GGUF blob, builds a tensor index (name → offset, size, type) without loading any weights
3. `scheduler_init()` — using topo results + tensor sizes, decides which tensors are pinned in VRAM, which stay in RAM, which stream from SSD
4. Allocate pinned (page-locked) RAM buffers for expert weights so cudaMemcpyAsync can DMA directly without CPU involvement

### Per-token inference loop (mirrors danveloper's CMD1/CMD2/CMD3 pattern)
```
For each transformer layer:
  CUDA Stream A: run attention projections on whatever is already in VRAM
  CUDA Stream B: cudaMemcpyAsync — prefetch next layer's active experts RAM→VRAM
  CPU: compute routing (softmax + topK) to know WHICH experts are needed
  cudaStreamSynchronize — ensure experts are loaded
  CUDA Stream A: run expert forward pass + combine + residual
```

The key: Stream B prefetch overlaps with Stream A compute. GPU never stalls waiting for weights.

### Qwen3.5-35B-A3B specifics (confirmed from bread_info + topology)
- 1959 tensors total, 120 MoE expert tensors (40 MoE layers × 3 tensors: gate/up/down)
- 256 experts per MoE layer, top-9 activated per token
- Hybrid architecture: layers 0-2 are GatedDeltaNet/SSM (ssm_* tensors), layers 3+ add full attention
  - SSM layers have: ssm_a, ssm_alpha, ssm_beta, ssm_conv1d, ssm_dt, ssm_norm, ssm_out
  - Also has vision encoder (v.blk.*, v.merger.*, v.patch_embed.*) — multimodal model
- Q4_K_M quantization breakdown:
  - Q4_K: 1115 tensors = 12.67 GiB  (expert weights + most non-expert)
  - Q6_K:   93 tensors =  8.63 GiB  (attention V projections, ffn_down_exps)
  - F16:   207 tensors =  0.83 GiB  (vision encoder, some embeddings)
  - F32:   544 tensors =  0.09 GiB  (norms, biases, small tensors)
- Non-expert weights: 2.77 GiB → pin permanently in VRAM (leaves 4.17 GiB spare)
- Expert weights: 19.45 GiB across 40 layers → stream RAM→VRAM per layer
  - Per-layer budget: 498 MiB (gate 144 MiB Q4_K + up 144 MiB Q4_K + down 210 MiB Q6_K)
  - Double-buffer (2 layers): 996 MiB
  - Estimated load time: 42.70 ms @ 12.23 GB/s RAM→VRAM

## What BREAD is NOT

- Not built on llama.cpp, vLLM, ollama, or any existing inference framework
- Not using Python (pure C + CUDA)
- Not a general-purpose framework — it is purpose-built for MoE streaming inference
- Not doing model training or fine-tuning

## Key References

- danveloper/flash-moe: https://github.com/danveloper/flash-moe (Metal implementation, architecture reference)
- Alexintosh/flash-moe feature/runtime-model-config branch (runtime config reference)
- Apple "LLM in a Flash" paper (the theoretical basis for expert streaming)
- Qwen3.5-35B-A3B model card: https://huggingface.co/Qwen/Qwen3.5-35B-A3B

## Architecture Deep Dive — Qwen3.5-35B-A3B

### Layer Composition (40 total)
- **Full-attention layers (4 layers):** Indices 3, 7, 11, 15, 19, 23, 27, 31, 35, 39
  - GQA with 16 query heads, 2 KV heads, 256 dims per head
  - Partial RoPE: 128 dims rotated per head (partial rotation)
  - Requires: Q/K/V projection, output projection, attention scoring, KV cache

- **SSM/GatedDeltaNet layers (28 layers):** All others
  - Selective State Space Model with gating (hybrid transformer-SSM)
  - Recurrent state across tokens (not stateless like attention)
  - Requires: ssm_conv1d, ssm_dt projection, selective scan, persistent state cache
  - Reference: infer_ref.m (MATLAB implementation)

### Quantization Breakdown (22 GB total on disk)
- **Q4_K (4-bit, symmetric):** 1115 tensors = 12.67 GiB
  - Expert gate/up weights, most attention, SSM weights
  - Best compression for large tensors

- **Q6_K (6-bit, symmetric):** 93 tensors = 8.63 GiB
  - FFN down weights, attention V projections (more sensitive to quantization)
  - Trade-off between compression and accuracy

- **F16 (half-precision float):** 207 tensors = 0.83 GiB
  - Vision encoder (multimodal)
  - Some embedding layers

- **F32 (single-precision float):** 544 tensors = 0.09 GiB
  - Norms, biases, small scalar tensors

### Multimodal Support
- **Vision encoder included** in GGUF (v.blk.*, v.merger.*, v.patch_embed.*)
  - Currently ignored in bread.exe (text-only inference)
  - Can be extended for image input (patches → vision embedding → token stream)

### Memory Layout at Inference
```
VRAM (8 GB total):
├─ Non-expert weights:      ~2.77 GB (pinned, never evicted)
│  ├─ Attention matrices (Q/K/V/O): 1.2 GB
│  ├─ Shared expert gate/up/down: 1.47 GB
│  └─ RMSNorm + embeddings: 0.08 GB
├─ Expert double-buffer:    ~1.0 GB (18 slots × 56 MB per expert layer)
├─ KV cache (future):       ~2.6 GB (when implemented)
└─ Scratch/temp:            ~1.7 GB (atom buffers, reductions, etc.)

Host RAM (48 GB total):
├─ Full GGUF blob:          22 GB (pinned for DMA, not paged)
└─ KV cache (future SSM):   variable (recurrent state across generation)

SSD (reserved for future):
   Not used for 35B (fits in RAM); would stream experts for 100B+ models
```

## Coding Conventions

- **Language:** C99 for .c files, CUDA C for .cu files
- **Scope:** No external dependencies beyond CUDA SDK and C standard library
- **Instrumentation:** Every hardware operation (memcpy, file I/O, kernel) wrapped in timer calls for profiling
- **Naming:** Tensor names must match GGUF conventions exactly (e.g., `blk.0.ffn_gate_exps.weight`)
- **Error handling:** Every CUDA call checked with CUDA_CHECK macro; file ops with explicit checks; fail loud and early
- **Type consistency:** Use uint8_t* for raw tensors, float for computations; avoid implicit casts
- **Comments:** Mark TODO/FIXME clearly; explain *why* not just *what* for non-obvious code

## Current Status (as of 2026-03-24)

### Infrastructure ✓ Complete
- [x] CUDA environment verified (CUDA 13.2, cl.exe, nvcc smoke test passed)
- [x] Model located and confirmed (22GB GGUF blob @ C:\Users\arahe\.ollama\models\blobs\sha256-900dde62fb7ebe8a5a25e35d5b7633f403f226a310965fed51d50f5238ba145a)
- [x] Hardware bandwidth measured (topology.exe):
      - RAM→VRAM: 12.23 GB/s (PCIe 4.0 x8 laptop, flat across 64MB–1GB)
      - SSD→RAM:   1.50 GB/s (warm-cache run; cold boot ~3 GB/s)

### Core Components ✓ Complete
- [x] **gguf.c/gguf.h** — GGUF file parser
      - 1959 tensors parsed in 0.025s, 120 MoE expert tensors confirmed
      - bread_info.exe dumps full tensor table with name/type/shape/offset/size
      - No weights loaded at parse time, pure metadata index

- [x] **dequant_q4k_cpu.c** — CPU reference dequantization
      - dequant_q4k_block_cpu: Q4_K reference implementation
      - dequant_q6k_block_cpu: Q6_K reference implementation
      - Bit logic verified against ggml-quants.c, selftest PASS

- [x] **kernels.cu** — CUDA compute kernels
      - dequant_q4k_matvec: Q4_K matrix-vector multiply on GPU
      - dequant_q6k_matvec: Q6_K matrix-vector multiply on GPU
      - 256 threads/block with warp reduction
      - Selftest: Q4_K max_err=0.000000, Q6_K max_err=0.000000 — PASS
      - Output allocation/free currently per-call (bottleneck for inference loop)

- [x] **bread.h/bread.c** — Model config parser
      - bread_model_config_t struct with all QWEN3.5 architecture constants
      - Config reader: parses GGUF kv metadata at startup
      - All dimensions auto-derived from GGUF, no hardcoding

- [x] **loader.c/loader.h** — Expert weight streaming and caching
      - Full 22GB model loaded into pinned host RAM at startup
      - LRU cache: 18 VRAM slots (double-buffered 9 expert slots) for active experts
      - Per-expert footprint: 1.95 MiB (gate 576KB Q4_K + up 576KB Q4_K + down 840KB Q6_K)
      - Async cudaMemcpyAsync for prefetch overlap
      - Expert lookup: O(1) via 2D index array [layer][expert]
      - Verified: expert(0,0) loaded, first bytes non-zero, cache hit working, LRU eviction OK
      - Estimated load time: ~1.5ms per layer (9 active experts, warm cache zero-miss case)
      - Windows WDDM workaround: allocate VRAM slots BEFORE cudaMallocHost to avoid GPU VA exhaustion

- [x] **main.cu** — Inference pipeline
      - embed_token: CPU Q4K row dequant → copy to VRAM
      - one_layer_forward: × 40 MoE+attention transformer blocks
      - apply_output_norm: RMSNorm with output_norm.weight
      - compute_logits: bread_matvec Q6_K on output.weight (397.9 MB Q6_K)
      - greedy_sample: argmax on CPU
      - Full pipeline verified end-to-end
      - First token generated (output is garbage due to stubbed SSM + no KV cache — expected)
      - BOS token is -1 for Qwen3.5 (no explicit BOS); prompt encoded directly

### Performance Baseline
- **Benchmark:** bread.exe on Qwen3.5-35B-A3B
      - Throughput: 6.17 tok/s
      - TTFT: 9.7 ms
      - Total time: 8.1 s / 50 tokens
      - Identified bottleneck: one_layer_forward() cudaMalloc/Free per call (~30 allocs/layer)
      - At 6 tok/s × 40 layers × 30 allocs/layer = ~7200 cudaMalloc calls/sec
      - Expert cache hit rate: 0% (routing from garbage hidden state = uncorrelated experts)

### Tokenizer & Vocabulary
- [x] Tokenizer integrated (tokenizer.c: Qwen3.5 BPE with 248,320 vocab)
- [x] Prompt encoding validated: encodes input text correctly
- [x] Output sampling: greedy argmax works

### Diagnostics & Tooling
Available standalone binaries (build via build_*.bat scripts):
- **bread_info.exe** — GGUF tensor inspector: lists all 1959 tensors with types/shapes/offsets
- **config_reader.exe** — Extracts GGUF metadata, prints model config
- **dequant_q4k_cpu.exe** — Runs CPU dequant selftest
- **topology.exe** — Benchmarks hardware bandwidth, prints scheduler decisions (if implemented)

## Roadmap — Performance & Correctness

### Immediate Wins (Performance)

**1. VRAM weight caching** [PRIORITY 1 — biggest speedup]
   - **Problem:** one_layer_forward() does 30 cudaMalloc/cudaFree calls per layer call
     - At 6 tok/s × 40 layers = ~7200 allocs/sec; malloc dominates runtime
   - **Solution:** Pre-load all non-expert weights in VRAM at startup
     - Attention matrices (Q/K/V/O proj): ~1.2 GB
     - RMSNorm weights + embeddings: ~0.08 GB
     - Shared expert weights: ~1.47 GB
     - Total: ~2.77 GB (fits in 8GB VRAM, leaving 4.23GB for expert double-buffer + scratch)
   - **Implementation:** Modify one_layer_forward() to accept VRAM weight pointers instead of loading per-call
   - **Expected speedup:** 10–50× (eliminate malloc overhead)
   - **Files to modify:** one_layer.cu, loader.c/loader.h, main.cu

**2. RoPE for full-attention layers** [PRIORITY 2 — required for coherence]
   - **Problem:** Full-attn layers (3, 7, 11, ...) currently ignore position, softmax ≈1 for all tokens
   - **Solution:** Implement positional rotation for Q/K before dot product
     - Rope base: 1e7, dims per head: 128/256 (partial rope)
     - Qwen uses separate rotary dims for key/query
   - **Expected effect:** Multi-token prompts become coherent
   - **Files to modify:** kernels.cu (add rope_kernel), one_layer.cu (integrate into attn flow)

**3. GatedDeltaNet/SSM layers** [PRIORITY 3 — correctness]
   - **Problem:** Layers 0,1,2,4,5,6,8,9,... (28/40 layers) are SSM, currently stubbed as zero
   - **Solution:** Implement selective state space model forward pass
     - See infer_ref.m for MATLAB reference (7152 lines)
     - Requires: ssm_conv1d, ssm_dt projection, selective scan with delta dt
     - Persistent recurrent state across tokens (similar to KV cache)
   - **Files to implement:** ssm_kernels.cu, one_layer.cu SSM branch
   - **Estimated effort:** Medium (reference available)

**4. KV cache for full-attention** [PRIORITY 4 — enables proper generation]
   - **Problem:** No past K/V, so attention over full prompt every token
   - **Solution:** Allocate host-side KV cache (8192 tokens × 2 kv_heads × 256 dims)
     - ~256 MB per layer × 10 full-attn layers = ~2.6 GB total
     - Prefetch K/V to VRAM per-token (overlap with expert loading)
   - **Files to modify:** loader.h (add kv cache structures), main.cu (prefetch/manage KV)

### Secondary Improvements
- **Hybrid quantization awareness:** Some tensors are Q6_K, F16, F32—optimize dispatch per tensor type
- **Batch support:** Currently processes 1 token at a time; extend to batch_size > 1
- **Prompt caching:** Store computed KV across multiple generations (interactive use)
- **Benchmark vs ollama:** Run with same prompts to validate correctness

### Build & Debug

**Quick build (uses build_main.bat):**
```bash
cd C:\bread_v2
./build_main.bat
# or full rebuild:
./build.bat
```

**Full build from scratch (bash):**
```bash
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin:$PATH"
export INCLUDE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717/include;C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/ucrt;C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/um;C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/shared;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/include"
export LIB="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717/lib/x64;C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/ucrt/x64;C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/um/x64"
nvcc -O2 -x cu main.cu kernels.cu loader.c gguf.c bread.c -I. -o bread.exe
```

**Diagnostics:**
```bash
# List all tensors and verify quantization mix:
bread_info.exe C:\Users\arahe\.ollama\models\blobs\sha256-900dde62fb7ebe8a5a25e35d5b7633f403f226a310965fed51d50f5238ba145a

# Check hardware bandwidth and scheduler decisions:
topology.exe

# Verify dequant CPU code:
dequant_q4k_cpu.exe

# Test inference:
bread.exe --prompt "Hello, world" --tokens 10
```

### Known Issues & Workarounds
- **Output is garbage:** SSM layers stubbed, no KV cache — expected until priorities 2–4 done
- **0% expert cache hit:** Routing decisions are all garbage (hidden state uninitialized) — will improve after KV cache
- **VRAM VA exhaustion on Windows WDDM:** Allocate VRAM slots before cudaMallocHost (currently in loader.c)
- **Ollama conflict:** Both bread.exe and ollama compete for VRAM; stop ollama before testing bread.exe

---

## Optimization Tips & Debugging

### Performance Analysis
1. **Identify bottleneck:** Run bread.exe with perf counters
   - Currently: ~162 ms/token, 6.17 tok/s
   - Profile breakdown: cudaMalloc overhead likely >70% of hot loop time
   - Use Windows Performance Analyzer or NVIDIA profiler (nsys profile bread.exe)

2. **Verify expert cache hit rate:**
   - Add counters in loader_get_expert() / loader_request()
   - Print (hits / (hits+misses)) after generation
   - 0% is expected now (garbage routing); should approach 50%+ after KV cache

3. **DMA overlap validation:**
   - Add cudaEventRecord between loader_request() and kernel launch
   - Measure: is GPU busy while Stream B DMA is pending?
   - Goal: Overlap Stream A compute with Stream B memcpy

### Numerical Debugging
1. **Verify dequant correctness:**
   - Compare Q4_K output (bread.exe) with CPU reference (dequant_q4k_cpu.exe)
   - Max error should be near quantization noise (~0.01 for 4-bit)

2. **Check attention softmax stability:**
   - Verify max(attn_scores) < ~80 (log-domain stability)
   - Check softmax sums to ~1.0 (numerical stability)
   - If either fails, suspect RoPE frequency base or dimension mismatch

3. **Validate layer outputs:**
   - Insert assertions in one_layer_forward() to check intermediate ranges
   - Example: residual output should have similar magnitude to input
   - Sanity check: first layer embedding L2 norm should be ~sqrt(hidden_dim)

### Common Pitfalls
- **Tensor byte-order:** GGUF uses little-endian; ensure loader.c respects this
- **Quantization dispatch:** Always check tensor type before selecting kernel (Q4_K vs Q6_K vs F16 vs F32)
- **CUDA synchronization:** Missing cudaStreamSynchronize() before reading results is undefined behavior
- **Memory alignment:** Some CUDA operations require 128-byte alignment; double-check malloc calls
- **Integer overflow:** GGUF offsets are uint64_t; avoid truncation to int32_t

### Testing Strategy
```bash
# 1. Unit test each component
dequant_q4k_cpu.exe                    # CPU dequant reference
bread_info.exe <model>                 # Verify tensor parsing
config_reader.exe <model>              # Check config extraction

# 2. Integration test (current)
bread.exe --prompt "Hello" --tokens 5  # Full pipeline with visible output

# 3. Correctness validation (when KV cache + RoPE done)
# Compare against ollama: ollama run qwen3.5:35b-a3b "Hello"
# Side-by-side output comparison, logits distribution analysis

# 4. Performance profiling
nsys profile bread.exe --prompt "..." --tokens 10
# Check: GPU utilization, memory BW, kernel occupancy, stall reasons
```

### Environment Validation Checklist
- [ ] CUDA 13.2+ installed, nvcc on PATH
- [ ] MSVC 14.50 (VS BuildTools 2026), cl.exe on PATH
- [ ] Model file accessible and readable (check file size: 22 GB exactly)
- [ ] VRAM available: >8GB (RTX 4060 has 8GB, exactly sufficient)
- [ ] RAM available: >48GB for pinned buffer + system overhead (24GB minimum)
- [ ] SSD space: >30GB for future model variants
- [ ] Windows Update current (WDDM GPU drivers up to date)
