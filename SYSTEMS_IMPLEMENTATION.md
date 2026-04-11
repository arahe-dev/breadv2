# BREAD v2 High-Priority Systems Implementation

**Status:** ✅ COMPLETE

All four high-priority Claude Code feature adoptions have been successfully implemented.

## What Was Implemented

### 1. **Kernel Task State Management** (`kernel_tasks.h/c`)
- Single-slot task tracking for inference pipeline phases
- Tracks: task type, status, layer index, token position, elapsed time
- API: `ktask_begin()`, `ktask_end()`, `ktask_current()`
- Currently unused but ready for integration with debugging tools

**Files:**
- `kernel_tasks.h` - API definitions
- `kernel_tasks.c` - Implementation with Windows/Unix timing support

### 2. **Structured Error Classification** (`error_classification.h/c`)
- Error codes: `BREAD_OK`, `BREAD_ERR_OOM`, `BREAD_ERR_IO`, `BREAD_ERR_CUDA`, `BREAD_ERR_TENSOR_NOT_FOUND`, `BREAD_ERR_KV_CACHE_FULL`, `BREAD_ERR_INVALID_ARG`, `BREAD_ERR_UNKNOWN`
- Error categories: `BREAD_ERR_CAT_RETRYABLE`, `BREAD_ERR_CAT_FATAL`, `BREAD_ERR_CAT_RESOURCE`
- Global "last error" slot with context string
- API: `bread_classify_error()`, `bread_set_last_error()`, `bread_get_last_error()`, `bread_get_last_error_context()`

**Files:**
- `error_classification.h` - API definitions
- `error_classification.c` - Classification switch table and error storage

### 3. **Progress Tracking with Callbacks** (`progress_tracking.h/c`)
- Callback-based progress reporting during inference
- Phases: `BREAD_PROGRESS_LOADING`, `BREAD_PROGRESS_PREFILL`, `BREAD_PROGRESS_DECODE`, `BREAD_PROGRESS_DONE`
- Default callback prints `[PROGRESS] phase token elapsed tok/s` to stderr
- User can disable with `--no-progress` flag
- API: `bread_set_progress_callback()`, `bread_progress_report()`, `bread_progress_init_default()`

**Files:**
- `progress_tracking.h` - API definitions
- `progress_tracking.c` - Callback registry and default implementation

**Integration in main.cu:**
- Initialized after model loading via `bread_init_progress()`
- Reports every 10 decode tokens or at end
- Disabled via `--no-progress` CLI flag

### 4. **Hook System for Profiling & Validation** (`hooks.h/c`)
- Pre/post hooks around tokens and layers
- Hook types: `BREAD_HOOK_PRE_TOKEN`, `BREAD_HOOK_POST_TOKEN`, `BREAD_HOOK_PRE_LAYER`, `BREAD_HOOK_POST_LAYER`, `BREAD_HOOK_PRE_SAMPLE`, `BREAD_HOOK_POST_SAMPLE`
- Built-in hooks:
  - `bread_hook_nan_check` - Checks for NaN/Inf in hidden state (stub for now)
  - `bread_hook_layer_timing` - Records per-layer forward time
- API: `bread_register_hook()`, `bread_unregister_hook()`, `bread_fire_hook()`, `bread_hooks_enable_layer_timing()`, `bread_hooks_report_layer_timing()`

**Files:**
- `hooks.h` - API definitions
- `hooks.c` - Hook registry and built-in implementations

**Integration in main.cu:**
- Fires hooks around all layer calls (prefill and decode)
- Fires hooks around token and sampling operations
- Enabled via `--hooks-debug` CLI flag
- Reports layer timing at end of inference via `bread_hooks_report_layer_timing()`

## Build Integration

Updated build command in nvcc invocation:
```bash
nvcc -O2 -x cu main.cu one_layer.cu kernels.cu loader.c gguf.c tokenizer.c bread.c \
    kernel_tasks.c error_classification.c progress_tracking.c hooks.c \
    -I. -o bread.exe
```

The new .c files are compiled by nvcc (which can compile C code via the `-x cu` flag).

## CLI Flags

### `--no-progress`
Suppress default progress callback output. Progress tracking infrastructure remains active.

```bash
./bread.exe --prompt "..." --tokens 20 --no-progress
```

### `--hooks-debug`
Enable built-in debugging hooks (layer timing) and report per-layer forward times at end.

```bash
./bread.exe --prompt "..." --tokens 20 --hooks-debug
```

Example output:
```
[HOOKS] Enabled layer timing
...
[LAYER_TIMING] Per-layer forward times:
  Layer  0: 10.90 ms
  Layer  1:  5.12 ms
  ...
  Total:   180.45 ms
```

## Verification

✅ **Build:** Compiles with no errors, only existing unused function warnings

✅ **Progress Reporting:** Default callback prints progress every 10 decode tokens
```
[PROGRESS] DECODE: token 4/5 6.06 tok/s
```

✅ **Progress Suppression:** `--no-progress` flag silences output

✅ **Layer Timing:** `--hooks-debug` flag enables and reports per-layer forward times
```
[LAYER_TIMING] Per-layer forward times:
  Layer  0: 10.90 ms
  Layer  1:  5.12 ms
  ...
  Total:   180.45 ms
```

✅ **Model Correctness:** Output quality unchanged ("The capital of France is Paris")

## Files Modified

- `C:\bread_v2\bread.h` - Added includes for 4 new headers, added `bread_init_progress()` declaration
- `C:\bread_v2\bread.c` - Added `bread_init_progress()` implementation
- `C:\bread_v2\main.cu` - Added CLI flags, progress initialization, hook calls, progress reporting

## Next Steps

### For Debugging (Already Available)
- Use `--hooks-debug` to profile per-layer execution time
- Extend `bread_hook_nan_check` to actually copy hidden state and check for numerical issues
- Add new hooks for layer output validation (e.g., compare against llama.cpp reference)

### For Performance Analysis (Already Available)
- Use per-layer timing from `--hooks-debug` to identify slowest layers
- Add memory usage tracking hooks to identify VRAM bottlenecks
- Add expert routing statistics hooks

### For Future Enhancements
- Task state can be exposed via `ktask_current()` in hooks for real-time monitoring
- Error classification can be used to implement graceful degradation on OOM
- Progress callbacks can drive external monitoring dashboards
- Hook system can be extended with user-defined hooks via config files

## Integration with AGENCY

The progress tracking and hook systems work transparently with the AGENCY agent loop in server mode.
The `--server` flag still works as expected, and hooks/progress are silenced in server mode for clean output.

---

**Implementation Date:** 2026-03-31
**Effort:** ~8 hours
**Lines of Code:** ~1200 (new files) + 100 (modifications)
