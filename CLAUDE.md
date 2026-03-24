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
  CLAUDE.md              — this file
  gguf.h / gguf.c        — GGUF file parser (reads tensor metadata, no weights loaded)
  topology.h / topology.c — hardware bandwidth prober (runs at startup)
  dequant_q4k_cpu.c      — CPU reference dequant for Q4_K and Q6_K (verified)
  kernels.cu             — CUDA kernels: dequant matvec, RMSNorm, SwiGLU, RoPE, MoE combine
  bread.h                — shared types, model config struct, constants
  loader.c               — async RAM→VRAM streaming via cudaMemcpyAsync
  main.c                 — inference loop + scheduler
  Makefile               — nmake build (cl.exe + nvcc)
  build.bat              — convenience build wrapper (calls vcvars64 + nmake targets)
```

Standalone diagnostic binaries (not part of inference engine):
```
  bread_info.exe   — dumps all 1959 tensors from GGUF blob
  topology.exe     — runs bandwidth probes, prints scheduler decisions
```

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

## Coding Conventions

- C99 for all .c files
- CUDA kernels in .cu files only
- No external dependencies beyond CUDA runtime and standard C library
- Every function that touches hardware (memcpy, file read, kernel launch) must be benchmarkable — wrap in timer calls
- Tensor names follow GGUF conventions exactly (e.g. `blk.0.ffn_gate_exps.weight`)
- Error handling: every CUDA call checked with a macro, every file op checked, fail loud and early

## Current Status

- [x] CUDA environment verified (CUDA 13.2, cl.exe, nvcc smoke test passed)
- [x] Model located and confirmed (22GB GGUF blob)
- [x] Hardware bandwidth measured — actual numbers from topology.exe:
        RAM→VRAM: 12.23 GB/s (PCIe 4.0 x8 laptop, flat across 64MB–1GB)
        SSD→RAM:   1.50 GB/s (warm-cache run; cold boot ~3 GB/s)
- [x] gguf.c — 1959 tensors parsed in 0.025s, 120 MoE expert tensors confirmed
        bread_info.exe <blob> prints full tensor table with name/type/shape/offset/size
- [x] topology.c — bandwidth probed, scheduler decisions derived automatically from
        measured BW + GGUF tensor sizes (no hardcoded values)
- [x] dequant_q4k_block_cpu — CPU reference, bit logic from ggml-quants.c, selftest PASS
- [x] dequant_q6k_block_cpu — CPU reference, bit logic from ggml-quants.c, selftest PASS
        Both functions live in: dequant_q4k_cpu.c
- [x] kernels.cu — CUDA matvec kernels for Q4_K and Q6_K, 256 threads/block reduction
        dequant_q4k_matvec / dequant_q6k_matvec, bread_matvec dispatcher
        selftest: Q4_K max_err=0.000000, Q6_K max_err=0.000000 — PASS
- [x] loader.c — pinned RAM load of full 22GB model, async cudaMemcpyAsync
        expert streaming, LRU cache (18 VRAM slots, 35MB total VRAM, O(1) lookup)
        Per-expert: 1.95 MiB (gate 576KB Q4_K + up 576KB Q4_K + down 840KB Q6_K)
        Verified: expert(0,0) loaded, first bytes non-zero, cache hit working, LRU eviction OK
        Estimated load: ~1.5ms per layer (9 active experts, warm cache zero-miss case)
        Fallback to malloc if cudaMallocHost unavailable (DMA still works, synchronous)
- [x] main.cu — inference loop: embed → 40 layers → output_norm → lm_head → greedy sample
        embed_token: CPU Q4K row dequant → VRAM (or F16 memcpy)
        compute_logits: bread_matvec Q6_K on output.weight (397.9 MB)
        output.weight is Q6_K (not Q4_K); lm_head pre-loaded in VRAM
        loader.c fix: VRAM slots now allocated BEFORE cudaMallocHost to avoid
          WDDM GPU VA exhaustion when pinning 23+ GB of host memory
- [x] First token generated — pipeline verified end-to-end
        Output is garbage (SSM layers stubbed, no KV cache) — expected at this stage
        BOS token is -1 for Qwen3.5 (no explicit BOS); prompt encoded directly
- [x] Benchmark (bread.exe vs. ollama baseline)
        BREAD  —  6.17 tok/s  |  9.7 ms TTFT  |  8.1 s / 50 tokens
        bottleneck: one_layer_forward() cudaMalloc/Free per call (~30 allocs/layer)
        0% expert cache hit rate (routing from garbage state = uncorrelated experts)
        ollama — run manually to compare: ollama run qwen3.5:35b-a3b "Hello"

## Next Session

**Starting point:** Full inference pipeline is working end-to-end (bread.exe).
All 4 components verified: tokenizer, one_layer_forward, generation loop, benchmark.

**Current bottleneck:** one_layer_forward() does cudaMalloc/cudaFree for every
attention + norm weight every layer call. At ~30 allocs/layer × 40 layers × 6 tok/s,
that's ~7200 cudaMalloc calls/second — most of the 162 ms/tok budget.

**Next tasks (in priority order):**

1. **Cache non-expert weights in VRAM at startup** (biggest speedup)
   - Attention Q/K/V/O weight matrices, RMSNorm weights, shared expert weights
   - ~2.77 GiB total (fits in VRAM with room for expert slots)
   - Modify one_layer_forward() signature to accept pre-loaded weight pointers
   - Expected: 10-50× speedup (eliminate per-call cudaMalloc overhead)

2. **RoPE for full-attention layers** (correctness)
   - Currently full-attn layers use GQA-expanded V directly (pos=0, softmax=1)
   - Proper RoPE: rotate Q/K by position before QK dot product
   - Required for coherent output beyond single-token prompts

3. **GatedDeltaNet (SSM) implementation** (correctness)
   - Currently stubbed as zero attention contribution
   - Layers 0,1,2,4,5,6,8,9,10,... (28 out of 40) are SSM layers
   - See infer_ref.m for reference implementation

4. **KV cache** (enables multi-token prompts and coherent generation)
   - Full attention layers need past K/V for proper contextual attention
   - SSM layers have persistent recurrent state

**Build command (bash with MSVC on PATH):**
```bash
MSVC_BASE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717"
KIT_BASE="C:/Program Files (x86)/Windows Kits/10"
KIT_VER="10.0.26100.0"
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin:$PATH"
export INCLUDE="$MSVC_BASE/include;$KIT_BASE/Include/$KIT_VER/ucrt;$KIT_BASE/Include/$KIT_VER/um;$KIT_BASE/Include/$KIT_VER/shared;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/include"
export LIB="$MSVC_BASE/lib/x64;$KIT_BASE/Lib/$KIT_VER/ucrt/x64;$KIT_BASE/Lib/$KIT_VER/um/x64"
nvcc -O2 -x cu main.cu one_layer.cu kernels.cu loader.c gguf.c tokenizer.c -I. -o bread.exe
```

**Known issues:**
- Output is incoherent (SSM stubs + no KV cache) — expected until tasks 2-4 done
- 0% expert cache hit rate (routing from garbage hidden state)
- VRAM slot allocation must happen before cudaMallocHost (fixed in loader.c)
- Stop ollama before running bread.exe (both compete for VRAM)
