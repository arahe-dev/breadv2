# Claude Code Feature Adoption Plan for BREAD v2

## Overview

This document outlines architectural patterns and features from the Claude Code source that can be adopted to improve BREAD v2's inference pipeline, debugging, and performance optimization.

**Source**: Claude Code source at `C:\Users\arahe\Downloads\claude-code-main\claude-code-main\src`

---

## High-Priority Adoptions

### 1. Task/Kernel State Management System
**Claude Reference**: `src/Task.ts`, `src/state/AppState.tsx`

**Current BREAD Gap**: No structured tracking of inference pipeline stages; linear execution without observability.

**What to Adopt**:
- State machine for kernel operations: `pending → running → completed/failed/killed`
- Task ID generation with type prefixes (similar to Claude's `b`, `a`, `r` prefixes)
- Task context with abort signals for cancellation

**BREAD Application**:
- Track per-layer execution state
- Monitor expert cache loading from different memory tiers
- Track VRAM→RAM→SSD streaming operations
- Enable early termination on error detection

**Implementation Priority**: ⭐⭐⭐ HIGH
**Estimated Effort**: 2-3 days (create kernel task types and state machine)

**Files to Create**:
```
bread_v2/
├── kernel_tasks.h        # Task type definitions and enums
├── kernel_tasks.c        # Task creation, state transition logic
└── task_tracking.h       # Global task context and callbacks
```

**Example Task Types**:
```c
typedef enum {
  TASK_TOKEN_EMBEDDING,   // t
  TASK_LAYER_FORWARD,     // l
  TASK_EXPERT_LOAD,       // e
  TASK_KV_CACHE_OPS,      // k
  TASK_OUTPUT_NORM,       // o
} KernelTaskType;

typedef enum {
  TASK_PENDING,
  TASK_RUNNING,
  TASK_COMPLETED,
  TASK_FAILED,
  TASK_KILLED
} KernelTaskStatus;
```

---

### 2. Structured Error Classification & Categorization
**Claude Reference**: `src/services/api/errors.ts`, `categorizeRetryableAPIError()`

**Current BREAD Gap**: Kernel failures treated uniformly; no distinction between fixable vs fundamental bugs.

**What to Adopt**:
- Classify errors as retryable, fatal, or resource constraints
- Error categorization logic for different failure modes
- Structured error reporting with context

**BREAD Application**:
- **Retryable Errors**:
  - OOM errors → retry with smaller expert cache or lower batch size
  - Memory fragmentation → retry after cleanup
  - Temporary quantization errors → fallback to full precision

- **Fatal Errors**:
  - Tensor shape mismatches → indicates Qwen3.5 semantic bug
  - Invalid model config → model loading issue
  - Corrupted GGUF tensors → data integrity failure

- **Resource Constraint Errors**:
  - VRAM insufficient → trigger memory tier fallback
  - Expert not found → corrupt expert cache

**Implementation Priority**: ⭐⭐⭐ HIGH
**Estimated Effort**: 1-2 days (classification logic + error handlers)

**Files to Create**:
```
bread_v2/
├── error_classification.h   # Error type enums and classification
├── error_classification.c   # Categorization logic
└── error_recovery.h         # Retry strategies per error type
```

**Example Structure**:
```c
typedef enum {
  ERROR_RETRYABLE,           // Can retry safely
  ERROR_FATAL,               // Cannot recover
  ERROR_RESOURCE_CONSTRAINT, // Resource issue, try fallback
} ErrorCategory;

typedef enum {
  ERR_OOM,
  ERR_VRAM_FRAGMENTED,
  ERR_TENSOR_SHAPE_MISMATCH,
  ERR_QUANTIZATION_OVERFLOW,
  ERR_KV_CACHE_CORRUPTION,
  ERR_EXPERT_NOT_FOUND,
  ERR_MODEL_CONFIG_INVALID,
} KernelErrorType;

ErrorCategory classify_kernel_error(KernelErrorType err);
```

---

### 3. Progress Tracking with Detailed Context
**Claude Reference**: `src/types/tools.ts`, `BashProgress`, `AgentToolProgress`, etc.

**Current BREAD Gap**: No intermediate progress reporting; binary success/failure feedback.

**What to Adopt**:
- Per-stage progress reporting
- Context tracking at each step
- Progress callback system

**BREAD Application**:
Track detailed progress for debugging "output quality is still wrong":

```
Token Embedding
├── Input tokenization
├── Token embedding lookup
└── Embedding transform

Layer Forward (×139 layers)
├── Layer type detection (Attention vs SSM)
├── Attention path or SSM path
├── Norm + activation
└── Post-processing

Expert Routing
├── Router logits computation
├── Expert selection
├── Expert cache lookup/load
├── Expert computation
└── Expert combine

Output Generation
├── Final layer norm
├── LM head projection
├── Logit processing
└── Token sampling
```

**Implementation Priority**: ⭐⭐⭐ HIGH
**Estimated Effort**: 2-3 days (progress struct + callback registration)

**Files to Create**:
```
bread_v2/
├── progress_tracking.h   # Progress types and callbacks
└── progress_tracking.c   # Progress state management
```

---

### 4. Hook System for Kernel Profiling & Validation
**Claude Reference**: `src/utils/hooks/` system, plugin architecture

**Current BREAD Gap**: No instrumentation points; hard to debug which layer causes output divergence.

**What to Adopt**:
- Pre/post operation hooks
- Extensible hook registration system
- Hook activation/deactivation at runtime

**BREAD Application**:
```c
// Hook registration examples
register_hook(HOOK_BEFORE_LAYER_FORWARD, validate_layer_input);
register_hook(HOOK_AFTER_LAYER_FORWARD, check_output_nan_inf);
register_hook(HOOK_BEFORE_EXPERT_LOAD, log_expert_memory);
register_hook(HOOK_BEFORE_KV_WRITE, validate_kv_cache_shape);
register_hook(HOOK_AFTER_ATTENTION, compare_with_reference);
```

**Key Hooks**:
- `HOOK_BEFORE_LAYER_FORWARD` → validate input shapes and values
- `HOOK_AFTER_LAYER_FORWARD` → compare with reference implementation
- `HOOK_BEFORE_EXPERT_LOAD` → log which experts are loaded
- `HOOK_AFTER_EXPERT_LOAD` → validate expert tensor integrity
- `HOOK_BEFORE_KV_WRITE` → verify KV cache shape
- `HOOK_AFTER_KV_READ` → check for NaN/Inf in attention scores

**Implementation Priority**: ⭐⭐⭐ HIGH
**Estimated Effort**: 2-3 days (hook registry + invocation points)

**Files to Create**:
```
bread_v2/
├── hooks.h           # Hook type definitions
├── hooks.c           # Hook registry and invocation
├── hook_validators.c # Built-in validation hooks (NaN checks, etc.)
└── hook_debuggers.c  # Built-in debugging hooks (logging, comparisons)
```

---

### 5. Configuration Management System
**Claude Reference**: `src/utils/config.ts`, `settings.json` + `settings.local.json` pattern

**Current BREAD Gap**: Hardcoded flags (`--minimal`, `--no-ssm`); difficult to experiment with variants.

**What to Adopt**:
- JSON configuration file support
- Environment variable overrides
- Local overrides for development
- Runtime config reloading

**BREAD Application**:

**bread_config.json** (default):
```json
{
  "model": {
    "name": "qwen3.5-35b",
    "gguf_path": "${HOME}/.ollama/models/blobs/sha256-...",
    "quantization": "q4_k"
  },
  "memory": {
    "vram_limit_mb": 8000,
    "expert_cache_size_mb": 4000,
    "kv_cache_mode": "host_side"
  },
  "inference": {
    "batch_size": 1,
    "enable_ssm_path": true,
    "enable_attention_path": true,
    "expert_scheduling": "async_orchestration"
  },
  "debugging": {
    "minimal_mode": false,
    "enable_profiling_hooks": true,
    "enable_correctness_checks": true,
    "log_level": "info"
  },
  "optimization": {
    "enable_simd": true,
    "enable_tensor_fusion": true,
    "kernel_variant": "default"
  }
}
```

**bread_config.local.json** (dev overrides):
```json
{
  "debugging": {
    "minimal_mode": true,
    "enable_profiling_hooks": true,
    "log_level": "debug"
  }
}
```

**Implementation Priority**: ⭐⭐ MEDIUM
**Estimated Effort**: 1-2 days (JSON parsing + config merging)

**Files to Create**:
```
bread_v2/
├── config.h        # Config struct definitions
├── config.c        # JSON parsing and merging logic
├── bread_config.json
└── bread_config.local.json (gitignored)
```

---

### 6. Cache Management for Kernel Metadata
**Claude Reference**: `src/utils/fileStateCache.ts`, caching patterns

**Current BREAD Gap**: Recomputing layer types, expert metadata, quantization info on every pass.

**What to Adopt**:
- Lazy-loaded cache for expensive computations
- Cache invalidation strategies
- Cache memory limits

**BREAD Application**:
```c
// Cache what's expensive to compute
CachedLayerMetadata layer_cache;      // Layer type (Attention vs SSM)
CachedExpertMetadata expert_cache;    // Which experts loaded, where
CachedQuantInfo quant_cache;          // Quantization boundaries
CachedTensorShapes shape_cache;       // Expected tensor shapes
```

**What to Cache**:
- Layer type decisions (computed from config during load)
- Expert quantization boundaries
- KV cache shape expectations
- Router thresholds per layer
- Normalization stats per layer

**Implementation Priority**: ⭐⭐ MEDIUM
**Estimated Effort**: 1-2 days (LRU cache + invalidation logic)

---

### 7. Cost/Resource Usage Tracking
**Claude Reference**: `src/cost-tracker.ts`, `getTotalCost()`, `getModelUsage()`

**Current BREAD Gap**: No tracking of inference efficiency metrics.

**What to Adopt**:
- Track resource consumption per inference
- Accumulate metrics over session
- Report efficiency metrics

**BREAD Application**:
```c
typedef struct {
  float peak_vram_mb;
  float avg_vram_mb;
  float peak_ram_mb;
  uint64_t bytes_vram_to_ram;       // Streaming volume
  uint64_t bytes_ram_to_vram;
  float inference_time_ms;
  float kernel_time_ms;
  float memory_transfer_time_ms;
  uint32_t tokens_generated;
  float tokens_per_second;
  float correctness_score;          // vs reference
} InferenceMetrics;
```

**Implementation Priority**: ⭐⭐ MEDIUM
**Estimated Effort**: 1-2 days (metric collection + reporting)

---

### 8. Permissioning/Capability System for CUDA/Memory Resources
**Claude Reference**: `src/types/permissions.ts`, permission mode system

**Current BREAD Gap**: No control over resource allocation; all or nothing.

**What to Adopt**:
- Resource capability declarations
- Runtime resource constraint enforcement
- Resource upgrade/fallback strategies

**BREAD Application**:
```c
typedef enum {
  CAP_VRAM_8GB,
  CAP_VRAM_16GB,
  CAP_RAM_STREAMING,
  CAP_SSD_STREAMING,
  CAP_ASYNC_EXPERTS,
  CAP_TENSOR_FUSION,
  CAP_SIMD_OPTIMIZATION,
} CapabilityFlag;
```

**Implementation Priority**: ⭐ LOW
**Estimated Effort**: 2-3 days (capability checking + fallbacks)

---

### 9. Plugin/Extension Architecture
**Claude Reference**: `src/tools/` plugin loading, dynamic plugin system

**Current BREAD Gap**: Kernel implementations hardcoded; difficult to swap implementations.

**What to Adopt**:
- Plugin discovery and loading
- Plugin interface standardization
- Runtime plugin selection

**BREAD Application**:
- Load different matvec kernels (Q4_K, Q6_K, FP32)
- Swap expert scheduling strategies
- Load custom validation kernels
- Load different attention implementations

**Example Plugin Types**:
```
plugins/kernels/
├── matvec_q4k.cu
├── matvec_q6k.cu
└── matvec_fp32.cu

plugins/expert_schedulers/
├── async_orchestration.c
└── eager_loading.c

plugins/validators/
├── layer_output_check.c
└── kv_cache_validator.c
```

**Implementation Priority**: ⭐ LOW
**Estimated Effort**: 3-4 days (plugin registry + loading system)

---

### 10. Session/History & Inference Tracing
**Claude Reference**: `src/assistant/sessionHistory.ts`, transcript recording

**Current BREAD Gap**: No way to review past inferences or compare variants.

**What to Adopt**:
- Session recording and replay
- Inference transcript with intermediate states
- Comparison tooling

**BREAD Application**:
```json
{
  "session_id": "sess_abc123",
  "timestamp": "2026-03-31T02:07:00Z",
  "config": { /* bread_config.json */ },
  "inferences": [
    {
      "id": "inf_001",
      "prompt": "...",
      "tokens_requested": 100,
      "tokens_generated": 45,
      "layers": [
        {
          "layer_id": 0,
          "layer_type": "attention",
          "input_shape": [1, 128, 3584],
          "output_shape": [1, 128, 3584],
          "timing_ms": 12.5,
          "expert_load_time_ms": 0
        },
        ...
      ],
      "final_output": [...],
      "reference_output": [...],
      "correctness_score": 0.87
    }
  ]
}
```

**Implementation Priority**: ⭐⭐ MEDIUM
**Estimated Effort**: 2-3 days (JSON transcript generation + comparison tools)

---

## Implementation Roadmap

### Phase 1: Debugging & Diagnostics (1-2 weeks)
Focus on isolating the "output quality is still wrong" bug.

1. **Task State Management** (2-3 days)
   - per-layer task tracking
   - Enable observation of where divergence occurs

2. **Error Classification** (1-2 days)
   - Distinguish kernel bugs from resource issues

3. **Progress Tracking** (2-3 days)
   - Monitor inference pipeline stage-by-stage

4. **Hook System** (2-3 days)
   - Add validation hooks at critical points
   - Enable side-by-side comparison with reference

**Goal**: Pinpoint exact layer/operation causing output divergence

---

### Phase 2: Performance Optimization (2-3 weeks)
Focus on "performance is still poor" issue.

1. **Cost/Resource Tracking** (1-2 days)
   - Measure VRAM usage, memory transfers, kernel times

2. **Configuration System** (1-2 days)
   - Make tuning parameters configurable

3. **Cache Optimization** (1-2 days)
   - Avoid recomputing expensive metadata

4. **Profiling Hooks** (already in Phase 1)
   - Use to identify bottlenecks

**Goal**: Identify performance bottleneck (VRAM transfers? kernel latency? expert loading?)

---

### Phase 3: Robustness & Experimentation (2-3 weeks)
Focus on production readiness and variant testing.

1. **Session/History Tracking** (2-3 days)
   - Enable comparison across kernel variants

2. **Plugin Architecture** (3-4 days)
   - Swap kernel implementations for comparison
   - Test different expert scheduling strategies

3. **Capability System** (2-3 days)
   - Runtime resource constraint handling
   - Graceful degradation

**Goal**: Enable rapid experimentation with kernel variants

---

## Quick Wins (Do First)

These provide immediate value with minimal effort:

### ✅ Add Kernel Execution Logging
**Effort**: 30 minutes
**Impact**: See which layer causes divergence
```c
fprintf(stderr, "[LAYER %d] input_shape=[%d,%d,%d] output_shape=[%d,%d,%d] time=%fms\n",
        layer_id, ...);
```

### ✅ Add Per-Layer NaN/Inf Checking
**Effort**: 1 hour
**Impact**: Detect numerical instability early
```c
bool has_nan_inf(const float *data, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (isnan(data[i]) || isinf(data[i])) return true;
  }
  return false;
}
```

### ✅ Add Timing Instrumentation
**Effort**: 1-2 hours
**Impact**: Identify performance bottleneck
```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
// kernel
cudaEventRecord(stop);
float ms = compute_elapsed_ms(start, stop);
fprintf(stderr, "[TIMING] layer_%d: %.2f ms\n", id, ms);
```

### ✅ Add Simple Config File Support
**Effort**: 2-3 hours
**Impact**: Make experimentation easier
- Parse JSON with minimal library (or hand-write simple parser)
- Support `--config bread_config.json` flag
- Override via env vars

---

## Files Affected by Adoption

### New Files to Create
```
bread_v2/
├── kernel_tasks.h / kernel_tasks.c
├── error_classification.h / error_classification.c
├── error_recovery.h / error_recovery.c
├── progress_tracking.h / progress_tracking.c
├── hooks.h / hooks.c
├── hook_validators.c
├── hook_debuggers.c
├── config.h / config.c
├── bread_config.json
├── bread_config.local.json
├── metrics.h / metrics.c
├── trace.h / trace.c
└── docs/ADOPTION_PROGRESS.md
```

### Files to Modify
```
bread_v2/
├── main.cu          (add config loading, metric reporting)
├── one_layer.cu     (add hooks, progress tracking, task state)
├── loader.c         (add error classification, caching)
├── bread.c          (add config merging)
└── Makefile         (add new source files)
```

---

## Success Criteria

✅ **Phase 1 Complete**: Can identify exact layer causing output divergence
✅ **Phase 2 Complete**: Know whether bottleneck is VRAM, compute, or memory transfers
✅ **Phase 3 Complete**: Can test 5+ kernel variants without manual code changes

---

## Notes

- All patterns are **optional** — adopt what helps most with debugging first
- Start with **Phase 1 (Task + Hook system)** to solve the output quality issue
- Use **Phase 2 (Timing + Cost tracking)** to solve the performance issue
- **Phase 3 (Plugins + Sessions)** enables long-term experimentation

---

## References

- Claude Code source: `C:\Users\arahe\Downloads\claude-code-main\claude-code-main\src`
- Key files:
  - `src/Task.ts` - task management
  - `src/services/api/errors.ts` - error classification
  - `src/types/tools.ts` - progress tracking
  - `src/utils/hooks/` - hook system
  - `src/utils/config.ts` - configuration
  - `src/cost-tracker.ts` - resource tracking
  - `src/assistant/sessionHistory.ts` - history/tracing
