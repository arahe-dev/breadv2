# AGENCY Handoff Context — 2026-04-06

## Current Status: ⏳ PARTIALLY WORKING

AGENCY CLI loads and connects to BREAD, but responses are incomplete/malformed.

---

## What Works ✅

- **BREAD server mode** — loads model, stays resident, accepts multiple prompts
- **AGENCY startup** — loads BREAD, prints "BREAD ready"
- **First query** — gets response but with formatting issues
- **Pipe management** — child process kept alive across queries (fixed via `_child` in BreadServer struct)
- **Regex fix** — tool call extraction works (hermes.rs regex corrected)

---

## What Doesn't Work ❌

**Second and subsequent queries: responses are incomplete or cut off**

Example:
```
>>> hello
Thinking Process:
1. **Analyze the Request:**
   ...
2. **Determine...:**
   ...
3.
>>>
```

Response ends abruptly mid-sentence at "3." instead of completing.

---

## Root Causes Identified

### 1. **Incomplete Response Collection**
- BREAD outputs thinking process + answer
- AGENCY's `bread.rs` reads until `BREAD_END` sentinel
- But response appears truncated

**Likely cause:**
- `BREAD_END` is being sent before BREAD finishes generating
- Or stdout buffer is getting flushed incomplete
- Or the prompt format is causing BREAD to generate incomplete responses

### 2. **Stateless History**
- Changed to remove conversation history (each query independent)
- This was to fix pipe-closing errors on query 3+
- But system prompt is now too minimal to guide BREAD properly

### 3. **System Prompt Issues**
- Current: `"You are a helpful assistant. Provide clear, direct answers to questions."`
- Too vague — BREAD outputs markdown "Thinking Process:" instead of using `<think>` tags
- Qwen3.5 expects more structured guidance

### 4. **Prompt Format Problem**
- AGENCY builds: `<|im_start|>system\n...\n<|im_end|>\n<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
- This format works for direct BREAD calls
- But might be causing issues in AGENCY context

---

## Files Modified (Last Session)

| File | Changes |
|------|---------|
| `main.cu` | Added `--server` mode, silenced debug output in server mode |
| `agency/src/bread.rs` | Fixed child process lifetime, added debug output (now removed) |
| `agency/src/hermes.rs` | Fixed regex, added `clean_response()` to strip `<think>` tags, simplified system prompt |
| `agency/src/repl.rs` | Removed conversation history (stateless) |

---

## Current Code State

### bread.rs: BreadServer struct
```rust
pub struct BreadServer {
    _child: Child,  // ← CRITICAL: keeps process alive
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}
```

✅ Child process is now kept alive
❌ But responses still incomplete

### hermes.rs: System Prompt
```rust
r#"You are a helpful assistant. Provide clear, direct answers to questions."#
```

❌ Too vague — causes BREAD to generate markdown thinking instead of using XML tags

### repl.rs: History Handling
```rust
let minimal_history = vec![current_user_msg.clone()];
let prompt = hermes::build_prompt(&minimal_history, &system);
```

✅ Prevents pipe closure on query 3+
❌ Loses context, BREAD confused

---

## Testing Evidence

### Direct BREAD Server Test (Working)
```bash
cat > /tmp/test.txt << 'EOF'
hello

what is 2+2?

EOF

./bread.exe --server --tokens 20 < /tmp/test.txt
```

**Result:** ✅ Both prompts succeed, BREAD_READY and BREAD_END flow correctly

### AGENCY Test (Broken)
```
>>> hello
[Incomplete response with "Thinking Process:" header]

>>> (no prompt shown, user input lost?)
```

**Result:** ❌ Response truncated, input not accepted for second query

---

## Hypothesis: The Real Problem

1. BREAD generates complete response with thinking blocks
2. AGENCY reads until `BREAD_END` but response is malformed
3. Either:
   - **Hypothesis A:** `clean_response()` is not cleaning the "Thinking Process:" markdown (only cleans `<think>` tags)
   - **Hypothesis B:** BREAD is outputting to BREAD_END early, cutting response short
   - **Hypothesis C:** The prompt format causes BREAD to output incomplete generations

---

## Recommended Next Steps

### Immediate (Debug First)
1. **Test BREAD directly with exact AGENCY prompt**
   ```bash
   # Capture what AGENCY sends to BREAD
   # Send it directly to bread.exe --server
   # Compare output
   ```

2. **Add response length logging**
   - In `bread.rs::generate()`, log how many bytes collected before BREAD_END
   - Check if response is actually truncated or just displayed wrong

3. **Check BREAD_END timing**
   - Is BREAD printing BREAD_END too early?
   - Test with fixed token count vs greedy generation

### Short-term (If debugging shows BREAD issue)
4. **Check for early exit in main.cu server loop**
   - Verify `do { ... } while (server_mode)` doesn't exit prematurely
   - Check `if (pos == 0) break` condition

5. **Verify token generation**
   - Maybe `--tokens 100` is being hit mid-response
   - Test with `--tokens 500`

### Medium-term (Fix AGENCY)
6. **Improve system prompt** to guide Qwen3.5 better
   - Add example format
   - Be more explicit about response style
   - Test with: `"You are a helpful AI. Answer concisely. Do not show your reasoning."`

7. **Re-enable limited history**
   - Keep last 1-2 exchanges instead of full history
   - Use sliding window to prevent prompt bloat

---

## Build Commands

```bash
# Build BREAD with --server support
export PATH="$PATH:/c/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64"
cd /c/bread_v2
nvcc -O2 -x cu main.cu one_layer.cu kernels.cu loader.c gguf.c tokenizer.c bread.c progress_tracking.c buffer_pool.c hooks.c layer_ops.cu -I. -o bread.exe

# Build AGENCY
cd /c/bread_v2/agency
cargo build --release
# Binary: ./target/release/agency.exe
```

---

## Key Files to Review

- `C:\bread_v2\main.cu` — server loop (lines 524-721)
- `C:\bread_v2\agency\src\bread.rs` — response collection logic (lines 52-83)
- `C:\bread_v2\agency\src\hermes.rs` — prompt building (lines 107-119)
- `C:\bread_v2\agency\src\repl.rs` — conversation flow (lines 60-132)

---

## Session Summary

- ✅ Fixed regex tool extraction bug
- ✅ Fixed child process lifecycle (was being dropped)
- ✅ Implemented server mode in BREAD
- ✅ Created minimal AGENCY CLI with stateless queries
- ❌ Responses are incomplete/truncated
- ❌ Second prompt handling unclear (might not even run?)

**Status:** Proof of concept works, but incomplete. Need to debug response collection or BREAD behavior.

---

**Generated:** 2026-04-06
**Next investigator:** Please start with the "Debug First" section above.
