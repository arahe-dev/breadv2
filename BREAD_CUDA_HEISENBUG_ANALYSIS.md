# BREAD CUDA Hang - Heisenbug Investigation

**Date:** 2026-04-08
**Issue:** BREAD server mode hangs after ~50-54 tokens without debug output, but generates 60+ tokens when debug output added
**Status:** HEISENBUG IDENTIFIED (timing-sensitive)

---

## Symptoms

### Without Debug Output
- Program hangs consistently at **token 54, layer 31** (SSM layer)
- Never reaches "one_layer_forward done" message
- No BREAD_END sentinel printed
- Execution stops mid-layer-processing

### With Debug Output Added
- Same code now runs to **token 61+** successfully
- No hang observed with debug messages present
- Layers process correctly with `fprintf()` calls

**This is a classic Heisenbug**: the program behavior changes depending on whether debugging output is present.

---

## Root Cause Analysis

### Most Likely Culprits (in order of probability)

#### 1. **Buffer Overflow or Memory Corruption** (VERY HIGH)
- **Why adding debug fixes it:** `fprintf()` calls consume stack space and slightly change memory layout
- **Impact:** A small buffer overflow in one of the SSM functions might only corrupt data at certain memory addresses
- **Evidence:** Hangs at specific token (54) and layer (31), not random
- **Specific suspect:** SSM buffers in `one_layer_forward()` with large static allocations

**Key static buffers in one_layer_forward:**
```c
static float *h_ssm_qrep = NULL;    // SSM query repeat
static float *h_ssm_krep = NULL;    // SSM key repeat
static float *h_ssm_sk = NULL;      // SSM scratch?
static float *h_ssm_d = NULL;       // SSM scratch?
```

These are allocated to large sizes (see line 1090 onwards). If allocation is insufficient OR if these are indexed out-of-bounds, it would corrupt memory.

---

#### 2. **Uninitialized Variable or Use-After-Free** (HIGH)
- **Why adding debug fixes it:** Debug output calls force evaluation of variables, potentially initializing them
- **Evidence:** Hangs only after certain number of tokens (memory accumulation)
- **Specific suspect:** `ssm_state[layer_idx]` or `ssm_conv_state[layer_idx]` pointers (lines 1069-1088)

These are allocated once per layer at startup and should persist, but:
- Are they properly initialized to NULL in the static array?
- Could a pointer be corrupted by buffer overflow earlier in the sequence?

---

#### 3. **CUDA/CPU Synchronization Race Condition** (MEDIUM)
- **Why adding debug fixes it:** `fflush(stderr)` calls `fsync()` which acts as a memory/CPU barrier
- **Impact:** CUDA kernels might not properly sync with CPU, leaving GPU state inconsistent
- **Evidence:** Only happens in server mode (multiple invocations of `one_layer_forward`)
- **Specific suspect:** `CUDA_CHECK(cudaStreamSynchronize(stream_a))` calls (lines 1437, 1547, 1633, etc.)

The static buffer pool might not be properly initialized for multi-query server mode.

---

#### 4. **Stack Corruption** (MEDIUM)
- **Why adding debug fixes it:** Stack grows with function calls, changing what gets corrupted
- **Evidence:** Hangs after exactly 54 tokens, not random
- **Specific suspect:** Large temporary arrays in `one_layer_forward()` that might overflow

Looking at line 1090: `s_vram2cpu_tmp = (half *)malloc(cfg->ssm_qkv_dim * sizeof(half));`
- This is allocated only once globally
- If not properly sized or if multiple layers try to use it simultaneously, corruption occurs

---

### Why This is Hard to Debug

1. **Heisenbug nature** — Adding instrumentation changes the timing and memory layout enough to avoid the bug
2. **Accumulation over tokens** — Error happens after certain number of tokens, suggesting memory creep
3. **Server-mode specific** — Only shows up with multiple query invocations, not single-prompt mode
4. **Complex buffer management** — Static globals + pool pointers + per-layer caches create intricate state

---

## Investigation Plan

### Immediate (30 min)

1. **Memory-safe logging to external file**
   - Replace `fprintf(stderr, ...)` with direct file writes to avoid buffer issues
   - Write to FILE* handle opened before any generation
   - This preserves the "working" behavior while keeping logs

2. **Check SSM buffer sizes**
   ```bash
   grep -n "ssm_qkv_dim\|ssm_head_dim\|ssm_num_v_heads" bread.h
   # Verify that all allocations match these values
   ```

3. **Verify SSM state initialization**
   - Add explicit check: are `ssm_conv_state` and `ssm_state` initialized to NULL before first use?
   - Line 1071-1082: allocation loop - verify loop bounds match array size

---

### Short-term (1-2 hours)

4. **Add AddressSanitizer (ASAN)**
   - Compile with `-fsanitize=address`
   - Run with `ASAN_OPTIONS=detect_leaks=1`
   - Will catch buffer overflows, use-after-free, memory leaks

5. **Valgrind / Memory Profiler**
   - Run under Valgrind to track memory access patterns
   - Identify first invalid access

6. **GDB with breakpoints**
   - Breakpoint at token 50, then single-step to find first corruption
   - Watch memory regions around `ssm_*` buffers

---

### Deep Investigation

7. **Static buffer lifecycle analysis**
   ```c
   // lines 939-992: all static buffers
   // Question: are these reinitialized per query in server mode?
   // If yes: check initialization code
   // If no: might have stale state from previous query
   ```

8. **KV cache bounds check**
   - Line 1140-1143: KV cache full check
   - Full-attention layers build up KV cache each token
   - Token 54 might be the point where a KV cache grows past its limit?

9. **SSM state accumulation**
   - `ssm_state[layer]` holds recurrent state that persists across tokens
   - Token 54 might trigger specific state value that causes numerical instability?
   - Check: does `cpu_delta_net_autoregressive_step()` bounds-check the state matrix?

---

## Proposed Immediate Fix

### Option A: File-based Logging (Keep working behavior)
```c
// At startup
static FILE *bread_debug_log = NULL;
if (!bread_debug_log) {
    bread_debug_log = fopen("C:\\bread_v2\\bread_debug.log", "w");
}
// Replace all fprintf(stderr, ...) with:
fprintf(bread_debug_log, "...");
fflush(bread_debug_log);
```

This preserves the timing/memory behavior of the working version.

---

### Option B: Disable Problematic Buffer Pool
If SSM buffers are the culprit:
```c
// one_layer.cu line 1069-1088
// Temporarily disable pre-allocation:
// Allocate fresh `conv_state` and `layer_state` per call instead of reusing
ssm_conv_state[layer_idx] = (float *)calloc(
    (cfg->ssm_conv_kernel - 1) * cfg->ssm_qkv_dim, sizeof(float));
// Deallocate at end of one_layer_forward()
```

This prevents state corruption at cost of performance.

---

### Option C: Add Defensive Checks
```c
// In one_layer_forward() at critical points:
if (!conv_state || !layer_state) {
    fprintf(stderr, "ERROR: NULL state at token %d layer %d\n", pos, layer_idx);
    exit(1);
}
// Check array bounds before access
if (vh >= num_v_heads || vh < 0) {
    fprintf(stderr, "ERROR: Invalid vh=%d at layer %d\n", vh, layer_idx);
    exit(1);
}
```

---

## Evidence Summary

| Factor | Finding |
|--------|---------|
| **Timing dependency** | Works with debug, hangs without → timing-sensitive |
| **Token count** | Fails at ~54 consistently → accumulation issue |
| **Layer specificity** | Hangs in SSM layers (non-full-attention) → SSM-specific bug |
| **Server-only** | Works in single-prompt mode → multi-invocation issue |
| **Reproducibility** | 100% reproducible (same input, same hang point) → not random |
| **Memory pattern** | Static global buffers + per-layer caches → complex state |

---

## Next Steps

**RECOMMENDED FIRST ACTION:**

1. Implement file-based logging to preserve "working" behavior
2. Run full test suite with logging enabled
3. Analyze log to find first divergence point
4. Cross-reference with memory map to identify corrupted variable

**Then:** Based on log analysis, implement targeted fix (Option A, B, or C above)

---

**Conclusion:** This is a real CUDA/memory bug masked by Heisenbug behavior. Adding debug output provides a temporary workaround, but the underlying issue (buffer overflow, use-after-free, or synchronization race) needs proper diagnosis and fix.

