# AGENCY Prompt Inference Investigation
**Date:** 2026-04-08
**Status:** Complete suspect list compiled

---

## Problem Statement

**Symptom:** AGENCY CLI second and subsequent queries produce truncated/incomplete responses.

Example:
```
>>> hello
[Incomplete response with "Thinking Process:" header and numbering]
Response ends at "3." instead of completing

>>> [second prompt not even shown, input lost?]
```

---

## Complete Data Flow Analysis

### AGENCY Side (Rust)
1. **repl.rs:31-34** — User input added to history
2. **repl.rs:70** — `build_prompt(history)` called
3. **hermes.rs:32-42** — Prompt formatted in Qwen3.5 chat template:
   ```
   <|im_start|>system\n{system}\n<|im_end|>\n
   <|im_start|>user\n{user}\n<|im_end|>\n
   <|im_start|>assistant\n
   ```
4. **bread.rs:57** — Sends: `write!(self.stdin, "{}\n\n", prompt)?`
5. **bread.rs:64-79** — Reads response until `BREAD_END` sentinel

### BREAD Side (C/CUDA)
1. **main.cu:526-550** — `server_mode` stdin reading loop:
   - Reads chars until double newline (`\n\n`)
   - Removes trailing two newlines (line 543: `pos -= 2`)
   - Stores in 8192-byte buffer

2. **main.cu:562** — `format_prompt_for_model()` called:
   - If prompt starts with `<|im_start|>`: use as-is
   - Otherwise: wraps in user/assistant tags

3. **main.cu:559-579** — Tokenizes prompt

4. **main.cu:590-620** — Prefill loop: processes all prompt tokens through 40 layers

5. **main.cu:652-688** — Autoregressive generation loop:
   - Generates up to `max_tokens` tokens
   - Prints each token to stdout
   - Continues while `next_tok != eos`

6. **main.cu:717** — Prints `BREAD_END\n` sentinel when done

---

## All Possible Suspects (Categorized)

### **CATEGORY A: Prompt Formatting Issues**

#### **A1: System Prompt Dominates Token Budget** ⭐ HIGH PRIORITY
**Location:** hermes.rs:28
**Code:**
```rust
pub fn system_prompt() -> String {
    "You are a helpful assistant. Answer the user directly and briefly.".to_string()
}
```
**Issue:** ALTHOUGH this is now minimal, AGENCY's history accumulation (repl.rs:31-127) may store verbose assistant responses that get re-added to history, pushing the real user query off the attention window.

**Impact:** Model responds to old context instead of new query

**Test:** Check what actually gets sent to BREAD with `/history` command

---

#### **A2: History Accumulation Bug** ⭐ CRITICAL
**Location:** repl.rs:86-89, 123-126
**Code:**
```rust
history.push(Message {
    role: "assistant".to_string(),
    content: cleaned_response,
});
```
**Issue:** After each query, the FULL response (including tool responses, thinking blocks) is added to history. On second query, history now contains:
```
[Query 1, Response 1 (large), Query 2]
```
The truncation of Response 1 in the `cleaned_response` still contains partial/malformed content.

**Example:** If Response 1 is truncated at "3.", the history stores an incomplete/grammatically incorrect message that confuses the model on Query 2.

**Impact:** Second query inherits malformed context → model generates garbage

**Test:** Print history with `/history` command before second query

---

#### **A3: Clean Response Truncation** ⭐ HIGH PRIORITY
**Location:** hermes.rs:56-83
**Code:**
```rust
pub fn clean_response(response: &str) -> String {
    let think_re = Regex::new(r"(?s)<think>.*?</think>").unwrap();
    let mut cleaned = think_re.replace_all(response, "").to_string();

    if let Some(idx) = cleaned.find("<tool_call>") {
        cleaned.truncate(idx);  // ← TRUNCATES AT TOOL_CALL
    }
    // ... skip "Thinking Process:" lines ...
}
```
**Issue:** If BREAD generates incomplete output (e.g., starts a `<tool_call>` but doesn't finish it), the truncate() call silently drops content. This incomplete response is then stored in history.

**Impact:** History now contains truncated text → confuses model on next query

**Test:** Check raw response from BREAD before cleaning vs after

---

#### **A4: Qwen3.5 Chat Template Mismatch**
**Location:** hermes.rs:40
**Code:**
```rust
prompt.push_str("<|im_start|>assistant\n");
```
**Issue:** Prompt ends with open `<|im_start|>assistant\n` tag. If the tokenizer has implicit token limits or if there's trailing whitespace handling, the model might see this as incomplete turn structure.

**Additional issue:** build_prompt() adds the assistant tag BEFORE the model generates, but if BREAD is in "reasoning mode," it might not properly recognize this as the generation slot.

**Impact:** Model may generate thinking blocks or meta-commentary instead of direct answers

**Test:** Compare token sequences from first vs second query in BREAD's tokenizer output

---

### **CATEGORY B: BREAD Server Mode Issues**

#### **B1: stdin Buffer Not Cleared Between Iterations** ⭐ MEDIUM PRIORITY
**Location:** main.cu:516, 524-552
**Code:**
```c
char stdin_buf[8192];
// ...
do {
    if (server_mode) {
        int pos = 0;  // ← ONLY THIS IS RESET
        int empty_lines = 0;
        while (pos < (int)sizeof(stdin_buf) - 1) {
            // ... read until double newline ...
        }
    }
    // buf used, but what about bytes after pos?
} while (server_mode);
```
**Issue:** `stdin_buf` is reused across iterations. If the second prompt is shorter than the first, bytes from prompt[old_len:] contain garbage from the first prompt. Although `pos` is reset, the buffer has junk data.

**Example:**
- First prompt: `<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n...[very long]...\n<|im_end|>\n<|im_start|>assistant\n` (2000 bytes)
- Second prompt: `<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n...[short query]...\n<|im_end|>\n<|im_start|>assistant\n` (500 bytes)
- Result: stdin_buf[500:2000] contains leftover data from first prompt

**Impact:** Tokenizer may see garbage tokens after the intended prompt → corrupts generation

**Test:** Add debug output to BREAD to print exact bytes read from stdin, including length

---

#### **B2: Double Newline Detection Logic Off-by-One** ⭐ MEDIUM PRIORITY
**Location:** main.cu:539-549
**Code:**
```c
if (c == '\n') {
    empty_lines++;
    if (empty_lines == 2) {
        pos -= 2;
        stdin_buf[pos] = '\0';
        break;
    }
} else if (c != '\n') {
    empty_lines = 0;
}
```
**Issue:** The counter increments BEFORE storing the character (line 536: `stdin_buf[pos++] = (char)c;`). So:
- Read first `\n`: pos increments, empty_lines = 1
- Read second `\n`: pos increments AGAIN, empty_lines = 2
- Then: `pos -= 2` removes the LAST two increments

But this means we're removing TWO bytes from the buffer. If the prompt only has ONE byte after the second newline (which shouldn't happen but...), we might be reading incorrect data.

Actually, re-reading the code, the order is:
1. `stdin_buf[pos++] = (char)c;` — store char, increment pos
2. `if (c == '\n') { empty_lines++; ...}`

So when we break, pos has ALREADY been incremented for both newlines. Doing `pos -= 2` removes them. This looks correct, but there's a subtle issue:

**If second newline is the last char in the buffer (pos == sizeof(stdin_buf) - 1):**
- We store the newline
- empty_lines becomes 2
- We do `pos -= 2`, making pos = sizeof(stdin_buf) - 3
- We set `stdin_buf[pos] = '\0'`

This is fine. But what if there's a race condition between checking `pos < sizeof(stdin_buf) - 1` and incrementing?

**Impact:** Rare edge case, but possible buffer overread on exact buffer boundary

**Test:** Send prompts of exact lengths (8190, 8191, 8192 bytes) and check for crashes

---

#### **B3: BREAD_END Sentinel Timing** ⭐ LOW PRIORITY
**Location:** main.cu:717
**Code:**
```c
} else {
    /* Server mode: print sentinel to signal end of generation */
    printf("BREAD_END\n");
    fflush(stdout);
}
```
**Issue:** `BREAD_END` is printed inside the `if (!server_mode) { ... } else { ... }` block, which is CORRECT. But what if `printf()` or `fflush()` fails?

**Impact:** AGENCY's `read_line()` loop (bread.rs:66) never sees `BREAD_END`, gets EOF, and terminates early. But this would be visible as a crash/error.

**Test:** Add error checking around fflush()

---

### **CATEGORY C: AGENCY Response Collection Issues**

#### **C1: read_line() Behavior on Partial Responses** ⭐ HIGH PRIORITY
**Location:** bread.rs:64-79
**Code:**
```rust
loop {
    line.clear();
    match self.stdout.read_line(&mut line) {
        Ok(0) => break,  // EOF
        Ok(_n) => {
            let trimmed = line.trim_end();
            if trimmed == "BREAD_END" {
                break;
            }
            if !trimmed.is_empty() {
                response.push_str(&line);
            }
        }
        Err(e) => { /* error */ }
    }
}
```
**Issue:** `read_line()` BLOCKS on a read. If BREAD's stdout pipe is closed unexpectedly (e.g., crash, timeout), `read_line()` returns `Ok(0)` (EOF), and the loop breaks WITHOUT seeing `BREAD_END`.

But more subtle: if BREAD is generating slowly, `read_line()` might return partial lines. The code collects them until `BREAD_END` is seen. **BUT**, if BREAD crashes mid-generation, the last line may be incomplete and never terminated with `BREAD_END`.

**Impact:** Response collection stops early, truncated output stored in history

**Test:** Check if BREAD process is still alive after first query (use `ps` or WinAPI)

---

#### **C2: Empty Line Filtering Logic** ⭐ MEDIUM PRIORITY
**Location:** bread.rs:77
**Code:**
```rust
if !trimmed.is_empty() {
    response.push_str(&line);
}
```
**Issue:** The code SKIPS empty lines (`trimmed.is_empty()`). But what if BREAD generates legitimate empty lines as part of the response (e.g., paragraph breaks)?

Example: BREAD outputs
```
Hello there!

This is paragraph 2.

BREAD_END
```

AGENCY will collect:
```
Hello there!\nThis is paragraph 2.\n
```

The paragraph breaks are lost, creating a malformed response that gets added to history.

**Impact:** Response looks wrong, confuses model, incomplete context on next query

**Test:** Send a prompt expecting multi-paragraph output and check if paragraphs are joined

---

#### **C3: Pipe Flush Ordering** ⭐ MEDIUM PRIORITY
**Location:** bread.rs:57-58
**Code:**
```rust
write!(self.stdin, "{}\n\n", prompt)?;
self.stdin.flush()?;
```
**Issue:** After writing, `flush()` is called. But if the child process (BREAD) is blocked in `fgetc()` and the OS pipe buffer is full, the flush might block. Depending on the buffer size and BREAD's generation speed, there could be a deadlock:

- AGENCY writes prompt to stdin
- stdin pipe is full, write blocks
- BREAD tries to read from stdin, but stdin is not flushed yet
- Deadlock

Although Rust's `BufWriter` should handle this, it's worth checking.

**Impact:** Hangs on first or subsequent queries

**Test:** Add timeout to AGENCY's read_line() to detect hangs

---

### **CATEGORY D: Model Semantic Issues**

#### **D1: Token Limit Premature Termination** ⭐ MEDIUM PRIORITY
**Location:** main.cu:652
**Code:**
```c
while (n_gen < max_tokens && next_tok != eos) {
    // ... generate token ...
    n_gen++;
}
```
**Issue:** Generation stops at `max_tokens`. If BREAD is set to `--tokens 256` but the model wants to generate 300 tokens for a complete response, it stops at 256.

But wait, AGENCY passes `config.max_tokens` to BREAD (main.rs:19, default 256). So for a multi-turn conversation, the model might hit the token limit mid-response before generating `BREAD_END` equivalent.

But BREAD prints `BREAD_END` in the `} else` block (line 715) which is ALWAYS executed, even if max_tokens is reached. So that's not it.

**Issue:** The model may hit max_tokens and be forced to generate an EOS token, cutting off the response mid-sentence.

**Example:** `--tokens 256` but response needs 300 tokens → stops at 256 with truncated output

**Impact:** Incomplete response stored in history → confuses model

**Test:** Increase `--tokens` to 512 or 1024 and see if responses complete

---

#### **D2: Prompt Confusion — System Instructions Treated as Conversation** ⭐ CRITICAL
**Location:** hermes.rs:28-40
**Code:**
```rust
pub fn system_prompt() -> String {
    "You are a helpful assistant. Answer the user directly and briefly.".to_string()
}

pub fn build_prompt(history: &[Message], system: &str) -> String {
    let mut prompt = format!("<|im_start|>system\n{}\n<|im_end|>\n", system);
    // ... add history messages ...
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}
```
**Issue:** The prompt structure is CORRECT (system, then conversation). BUT, Qwen3.5 might expect the system prompt to be VERY short or to contain specific keywords.

The tokenizer_pre for Qwen35 is set (main.cu:255), so format_prompt_for_model should preserve the chat format. But if there's a mismatch between AGENCY's format and what the model expects, the model might hallucinate thinking blocks instead of direct answers.

From the handoff document:
> **Symptom:** Responses like "Do not output any content other than the answer." = model answering system instructions.

This suggests the model is treating the system instruction literally as part of its response, not as a system directive.

**Impact:** Model generates instructions/meta-commentary instead of answers, response format wrong, incomplete

**Test:** Compare AGENCY's prompt with output from ollama/llama.cpp's native format

---

#### **D3: Chat Format Implicit Tokens** ⭐ MEDIUM PRIORITY
**Location:** tokenizer_encode() call (main.cu:566-569)
**Code:**
```c
if (bos >= 0) {
    token_buf[0] = bos;
    n_prompt = 1 + tokenizer_encode(tok, model_prompt, token_buf + 1, 4095);
} else {
    n_prompt = tokenizer_encode(tok, model_prompt, token_buf, 4096);
}
```
**Issue:** The tokenizer may add implicit tokens for the Qwen3.5 chat format (e.g., special role tokens before `<|im_start|>`). If AGENCY's prompt building and BREAD's tokenizer don't align, the token sequence is wrong.

**Example:**
- AGENCY expects: `<|im_start|>system[special_token]You are...[special_token]<|im_end|>[special_token]<|im_start|>user...`
- BREAD tokenizes: `[BOS]<|im_start|>system[no_special]You are...[no_special]<|im_end|>[no_special]<|im_start|>user...`

Mismatch → corrupted token sequence → garbage output

**Impact:** Model receives wrong token sequence → generates corrupted output

**Test:** Use bread_info.exe to inspect token sequence, compare with tokenizer.encode() directly

---

### **CATEGORY E: Subtle State/Race Conditions**

#### **E1: CUDA Stream Not Synchronized Between Queries** ⭐ LOW PRIORITY
**Location:** main.cu:613, 663
**Code:**
```c
CUDA_CHECK(cudaDeviceSynchronize());
```
**Issue:** `cudaDeviceSynchronize()` is called at the end of prefill and token loop. But what if a kernel from the FIRST query is still running when the SECOND query starts?

In theory, the cleanup should wait, but if there's an issue with the stream ordering...

**Impact:** VRAM state from first query corrupts second query

**Test:** Add explicit `cudaDeviceSynchronize()` before/after BREAD_END

---

#### **E2: Host Memory Reuse Issues** ⭐ LOW PRIORITY
**Location:** main.cu:504-508
**Code:**
```c
half *h_emb_row = (half *)malloc(cfg->hidden_dim * sizeof(half));
half *h_logits  = (half *)malloc((size_t)cfg->vocab_size * sizeof(half));
```
**Issue:** These buffers are allocated ONCE at startup and reused across all queries. If there's uninitialized memory or if a previous query left garbage in these buffers, the next query might use corrupted data.

Although `cudaMemcpy()` should overwrite them fully, uninitialized host memory could theoretically cause issues.

**Impact:** Corrupted embeddings/logits on second query

**Test:** Memset buffers to zero explicitly between queries (if server mode is enabled)

---

### **CATEGORY F: Rust-specific Issues**

#### **F1: BufReader Line Buffering Edge Case** ⭐ MEDIUM PRIORITY
**Location:** bread.rs:9, 29
**Code:**
```rust
stdout: BufReader::new(...)
```
**Issue:** `BufReader` has internal buffering. If BREAD writes incomplete lines (e.g., without newline), `read_line()` will BLOCK waiting for the newline.

**Example:**
```
BREAD outputs: "Hello World" (no newline)
AGENCY calls: read_line()
Result: BLOCKS indefinitely, waiting for \n
```

Although `printf()` in BREAD typically writes complete lines, if there's a flush issue...

**Impact:** AGENCY hangs on read_line()

**Test:** Check if BREAD outputs all lines with newlines, especially token outputs

---

#### **F2: String Encoding Issues** ⭐ LOW PRIORITY
**Location:** bread.rs:78, hermes.rs:56
**Code:**
```rust
response.push_str(&line);
```
**Issue:** If BREAD outputs non-UTF8 bytes (e.g., from corrupted CUDA memory), Rust's push_str() will panic or silently drop invalid UTF8.

**Impact:** Response collection crashes or silently truncates

**Test:** Check for panic output in AGENCY

---

### **CATEGORY G: Environmental/System Issues**

#### **G1: Windows WDDM GPU Sharing Issue** ⭐ MEDIUM PRIORITY
**Context:** Windows 11, RTX 4060 Laptop
**Issue:** WDDM (Windows Display Driver Model) has timeout limits on GPU kernels. If the GPU hangs, WDDM resets it, and BREAD crashes.

On second query, if the model is more complex (longer history), the kernels take longer, and WDDM might timeout.

**Impact:** BREAD crashes mid-generation, AGENCY sees EOF instead of BREAD_END

**Test:** Check Windows Event Viewer for GPU timeout events

---

#### **G2: File Descriptor Limit / Pipe Closure** ⭐ LOW PRIORITY
**Location:** Process spawning (bread.rs:16-23)
**Issue:** If BREAD crashes or exits unexpectedly, the stdin/stdout pipes may close. On second query attempt, `write!()` might fail.

**Impact:** AGENCY detects error and terminates

**Test:** Check error handling in AGENCY's write/read calls

---

---

## Summary Table: Priority & Likelihood

| Suspect | Category | Priority | Likelihood | Root Cause? |
|---------|----------|----------|-----------|-------------|
| **A2: History accumulation** | Prompt | CRITICAL | HIGH | ✅ LIKELY |
| **D2: Prompt confusion** | Semantic | CRITICAL | HIGH | ✅ LIKELY |
| **B1: Buffer not cleared** | BREAD | MEDIUM | MEDIUM | ? |
| **A3: Clean response truncation** | Prompt | HIGH | MEDIUM | ✅ POSSIBLE |
| **C1: read_line() EOF** | AGENCY | HIGH | MEDIUM | ✅ POSSIBLE |
| **D1: Token limit** | Semantic | MEDIUM | MEDIUM | ✅ POSSIBLE |
| **B2: Double newline logic** | BREAD | MEDIUM | LOW | ? |
| **E1: CUDA sync** | State | LOW | LOW | ? |
| **A1: System prompt** | Prompt | HIGH | LOW | ? |
| **G1: WDDM timeout** | System | MEDIUM | MEDIUM | ? |

---

## Recommended Investigation Order

### **Phase 1: Quick Wins (< 5 min each)**
1. Check if BREAD process is still alive after first query → diagnose C1/G1
2. Add debug output: print exact stdin bytes read → diagnose B1
3. Print history with `/history` → diagnose A2
4. Increase `--tokens` to 512 → diagnose D1
5. Compare AGENCY's prompt format with llama.cpp → diagnose D2

### **Phase 2: Detailed Testing (10-20 min)**
6. Send multi-paragraph prompt → diagnose C2
7. Log raw BREAD response before cleaning → diagnose A3
8. Enable tokenizer debug in BREAD → diagnose D3
9. Test stdin buffer edge cases (exact 8192 bytes) → diagnose B1/B2

### **Phase 3: Advanced (if needed)**
10. Add CUDA sync logging → diagnose E1
11. Monitor WDDM in Event Viewer → diagnose G1
12. Add BufReader timeout → diagnose F1

---

## Most Likely Root Cause Chain

**Hypothesis:**
1. First query generates TRUNCATED response (due to D1 token limit OR D2 prompt confusion)
2. `clean_response()` stores incomplete text in history (A3)
3. Second query includes malformed history (A2)
4. Model confused by corrupted context, generates garbage or incomplete
5. Response truncated AGAIN → cycle repeats

**Fix Order:**
1. Increase token limit to 512+ (D1)
2. Verify prompt format matches llama.cpp native format (D2)
3. Fix history storage to validate complete responses (A2)
4. Add buffer clearing between queries (B1)

---

## Test Commands

```bash
# Test 1: Direct BREAD (should work fine)
cat > /tmp/test.txt << 'EOF'
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is 2+2?
<|im_end|>
<|im_start|>assistant

EOF

./bread.exe --server --tokens 512 < /tmp/test.txt

# Test 2: Two queries in sequence
cat > /tmp/test2.txt << 'EOF'
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is 2+2?
<|im_end|>
<|im_start|>assistant

<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is 3+3?
<|im_end|>
<|im_start|>assistant

EOF

./bread.exe --server --tokens 512 < /tmp/test2.txt
# Should see two BREAD_END sentinels

# Test 3: AGENCY with debugging
AGENCY_DEBUG=1 ./agency.exe --bread ./bread.exe --tokens 512
>>> What is 2+2?
/history
>>> What is 3+3?
```

---

**Generated:** 2026-04-08
**Status:** Ready for investigative implementation
