# AGENCY Loop Investigation Report

**Date:** 2026-03-31
**Status:** Functional but has critical bugs

## Overview

AGENCY is a Rust-based interactive CLI that wraps BREAD (via `--server` mode) with Hermes tool-calling support. The architecture is clean and well-organized, but there are several bugs that prevent it from working reliably.

## Architecture

```
main.rs
  ├─ BreadServer (bread.rs)
  │   ├─ Spawns bread.exe --server process
  │   ├─ Waits for BREAD_READY signal
  │   └─ Manages stdin/stdout pipes
  │
  ├─ REPL loop (repl.rs)
  │   ├─ Reads user input (rustyline)
  │   ├─ Calls generate_with_tools()
  │   └─ Handles slash commands
  │
  ├─ Tool-calling loop (repl.rs)
  │   ├─ Builds prompt (hermes.rs)
  │   ├─ Sends to BREAD via BreadServer
  │   ├─ Extracts tool calls (hermes.rs)
  │   ├─ Executes tools (tools.rs)
  │   └─ Loops up to max_iterations times
  │
  ├─ Hermes format (hermes.rs)
  │   ├─ System prompt with tool definitions
  │   ├─ Build Qwen3.5 chat format
  │   ├─ Extract and parse tool calls
  │   └─ Format tool responses
  │
  └─ Tools (tools.rs)
      ├─ read_file() — Read file, limit 8 KB
      ├─ list_dir() — List directory
      ├─ shell() — Run shell commands, limit 4 KB output
      └─ write_file() — Write to file
```

## Critical Issues

### 1. **CRITICAL: Tool Call Extraction Regex Bug** ⚠️

**Location:** `hermes.rs`, line 123

```rust
let re = Regex::new(r#"<tool_call>\s*(\{[^}]*\})\s*</tool_call>"#).unwrap();
```

**Problem:** The regex `\{[^}]*\}` matches only a SINGLE level of braces. Nested JSON will fail to parse.

**Example failure:**
```
<tool_call>
{"name": "read_file", "arguments": {"path": "/dir/file.txt"}}
</tool_call>
```

The regex will match only:
```json
{"name": "read_file", "arguments": {"path"
```

This stops at the first `}` which closes the `"path"` value, not the whole object.

**Impact:** Tool calls with any nested objects (which is all of them) will fail to parse.

**Test Case:**
```rust
#[test]
fn test_nested_objects() {
    let text = r#"<tool_call>
{"name": "read_file", "arguments": {"path": "/file.txt"}}
</tool_call>"#;
    let calls = hermes::extract_tool_calls(text);
    assert_eq!(calls.len(), 1);  // THIS WILL FAIL
    assert_eq!(calls[0].name, "read_file");
}
```

**Fix:** Use a proper JSON parser or a recursive regex. Since serde_json is already imported:

```rust
pub fn extract_tool_calls(text: &str) -> Vec<ToolCall> {
    let re = Regex::new(r#"<tool_call>\s*(\{.*?\})\s*</tool_call>"#).unwrap();
    re.captures_iter(text)
        .filter_map(|cap| {
            cap.get(1).and_then(|m| ToolCall::parse(m.as_str()))
        })
        .collect()
}
```

**Better fix (robust):**
```rust
pub fn extract_tool_calls(text: &str) -> Vec<ToolCall> {
    let re = Regex::new(r#"<tool_call>\s*\n([\s\S]*?)\n</tool_call>"#).unwrap();
    re.captures_iter(text)
        .filter_map(|cap| {
            cap.get(1).and_then(|m| ToolCall::parse(m.as_str()))
        })
        .collect()
}
```

---

### 2. **HIGH: BreadServer Output Parsing Fragility**

**Location:** `bread.rs`, lines 74-90

```rust
// Skip BREAD's header noise until we see "--- Generated output ---"
if !started {
    if trimmed == "--- Generated output ---" {
        started = true;
        in_generation = true;
    }
    continue;
}
```

**Problem:** In server mode, BREAD doesn't print "--- Generated output ---" banner. The parsing will wait forever for this marker.

**Test:**
```bash
./bread.exe --server --tokens 10 --no-progress
<prompt>

```

Output:
```
<|im_start|>user
...
<|im_start|>assistant
<think>
...
</think>

Token token token
BREAD_END
```

No "--- Generated output ---" banner appears. The AGENCY BreadServer will skip all tokens and return empty string.

**Impact:** AGENCY will show empty responses from BREAD even though BREAD is generating correctly.

**Fix:** Remove dependency on banner marker:

```rust
let mut in_generation = true;  // Start collecting immediately

loop {
    line.clear();
    let n = self.stdout.read_line(&mut line)?;
    if n == 0 { break; }

    let trimmed = line.trim_end();

    // Stop at BREAD_END sentinel
    if trimmed == "BREAD_END" { break; }

    // Collect all non-empty lines
    if !trimmed.is_empty() {
        response.push_str(&line);
    }
}
```

---

### 3. **MEDIUM: No Conversation History Truncation**

**Location:** `repl.rs`, line 11

```rust
let mut history: Vec<Message> = Vec::new();
```

**Problem:** Conversation history grows unbounded. With max_tokens=256, a long conversation will eventually exceed BREAD's token limit and fail.

**Example:**
```
User:    "Read this file" (10 tokens)
Tool:    (8KB file content, ~2000 tokens)
Response: (256 tokens)
User:    "Read another file"
Tool:    (8KB file, ~2000 tokens)
Response: (256 tokens)
...
Total prompt becomes: system + user + tool + response + user + tool + response + ...
Eventually >> max_tokens
```

**Impact:** BREAD will fail with "too many tokens" after several tool interactions.

**Fix:** Implement sliding window:

```rust
fn maybe_truncate_history(history: &mut Vec<Message>, max_tokens: u32) {
    // Keep removing oldest non-system messages until size is reasonable
    while estimate_tokens(history) > max_tokens / 2 {
        if history.len() > 2 {
            history.remove(1); // Keep system, remove oldest user/assistant
        }
    }
}

fn estimate_tokens(history: &[Message]) -> u32 {
    history.iter().map(|m| m.content.len() / 4).sum::<u32>() + 100 // rough estimate
}
```

---

### 4. **MEDIUM: stderr Suppression Loses Debugging Info**

**Location:** `bread.rs`, line 20

```rust
.stderr(Stdio::null())  // Suppress debug output
```

**Problem:** Suppresses all stderr from BREAD, including our new progress tracking and hook reports.

**Impact:**
- User can't see why BREAD is slow (no layer timing with `--hooks-debug`)
- Progress messages are lost
- BREAD errors during model loading are hidden

**Example:** Running AGENCY with `--hooks-debug` shows nothing:
```bash
./agency.exe --bread ./bread.exe --tokens 256
>>> Hello
→ Tool: read_file ✓
[No layer timing output because stderr is null]
```

**Fix:** Redirect stderr to stderr instead of null (but filter BREAD_READY):

```rust
// Option 1: Show all stderr
.stderr(Stdio::piped())

// Option 2: More sophisticated - filter stderr for important messages
// (left as exercise, but would require reading stderr in separate thread)
```

---

### 5. **LOW: No Validation of bread.exe Path**

**Location:** `main.rs`, lines 30-33

```rust
"--bread" => {
    if let Some(path) = args.next() {
        config.bread_exe = PathBuf::from(path);
    }
}
```

**Problem:** No check that the path exists or is executable.

**Impact:** If user provides wrong path, error occurs during `BreadServer::start()` with a confusing message.

**Fix:**
```rust
"--bread" => {
    if let Some(path) = args.next() {
        let pb = PathBuf::from(&path);
        if !pb.exists() {
            eprintln!("Error: bread.exe not found at: {}", path);
            return Ok(());
        }
        config.bread_exe = pb;
    }
}
```

---

### 6. **LOW: Silent Token Parsing Failure**

**Location:** `main.rs`, line 37

```rust
config.max_tokens = tokens.parse().unwrap_or(256);
```

**Problem:** If user passes invalid token count, it silently defaults to 256 instead of warning.

**Fix:**
```rust
config.max_tokens = match tokens.parse::<u32>() {
    Ok(n) => n,
    Err(_) => {
        eprintln!("Warning: invalid token count '{}', using default 256", tokens);
        256
    }
};
```

---

## Severity Summary

| Issue | Severity | Impact | File |
|-------|----------|--------|------|
| Tool call regex | **CRITICAL** | All tool calls fail | hermes.rs |
| BreadServer parsing | **HIGH** | Empty responses from BREAD | bread.rs |
| History truncation | **MEDIUM** | Token overflow after ~5 tool calls | repl.rs |
| stderr suppression | **MEDIUM** | No debugging output visible | bread.rs |
| No exe validation | **LOW** | Confusing error messages | main.rs |
| Silent token parse | **LOW** | Unexpected behavior | main.rs |

---

## Current Usability

**Status: Broken**

The tool call regex bug means AGENCY cannot successfully execute any tool calls. Even if BREAD generates correct Hermes format output with tool calls, the extraction will fail.

**Testing:**
```bash
cd C:\bread_v2\agency
cargo build --release
./target/release/agency.exe --bread ..\bread.exe --tokens 256
>>> Read main.cu and tell me what it does
# Expected: BREAD calls read_file("main.cu"), gets response, analyzes
# Actual: Tool call is not extracted due to regex bug, BREAD loops asking for clarification
```

---

## Recommended Fixes (Priority Order)

### Fix 1: Tool Call Regex (CRITICAL) — 30 minutes

File: `hermes.rs`, line 123

**Before:**
```rust
let re = Regex::new(r#"<tool_call>\s*(\{[^}]*\})\s*</tool_call>"#).unwrap();
```

**After:**
```rust
let re = Regex::new(r#"<tool_call>\s*\n([\s\S]*?)\n</tool_call>"#).unwrap();
```

Test case already exists in `hermes.rs` at line 147, but the nested object test fails:
```rust
#[test]
fn test_nested_json_in_tool_call() {
    let text = r#"Some response
<tool_call>
{"name": "read_file", "arguments": {"path": "main.cu"}}
</tool_call>
More text"#;
    let calls = extract_tool_calls(text);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "read_file");
    assert_eq!(calls[0].arguments["arguments"]["path"].as_str(), Some("main.cu"));
}
```

### Fix 2: BreadServer Output Parsing (HIGH) — 15 minutes

File: `bread.rs`, lines 74-90

Remove dependency on "--- Generated output ---" banner.

### Fix 3: History Truncation (MEDIUM) — 45 minutes

File: `repl.rs`, add truncation logic before prompt building.

### Fix 4: stderr Handling (MEDIUM) — 60 minutes

File: `bread.rs`, line 20

Options:
- Simple: Pipe stderr to parent stderr
- Better: Read stderr in separate thread, filter/display selectively

### Fix 5: Validation (LOW) — 15 minutes

File: `main.rs`, lines 30-40

Add path existence check.

---

## Integration Points with New Monitoring Systems

AGENCY will benefit from the new progress tracking and hook systems:

✅ With Fix 2 (output parsing), `--no-progress` stderr suppression becomes safer
✅ With Fix 4 (stderr handling), AGENCY can show layer timing from `--hooks-debug`
✅ New error classification can improve tool failure handling

**Example improved flow:**
```bash
./agency.exe --bread "./bread.exe --hooks-debug" --tokens 256
>>> Read main.cu
→ Tool: read_file ✓
[BREAD output with layer timing, no progress chatter]
[Response with file analysis]
```

---

## Testing Checklist

- [ ] Tool call extraction with nested JSON objects
- [ ] BREAD output parsing without banner marker
- [ ] Multiple tool calls in sequence
- [ ] Long conversation (10+ exchanges) to test history truncation
- [ ] Invalid --bread path gives helpful error
- [ ] Invalid --tokens value is rejected with warning
- [ ] stderr is visible when BREAD runs with --hooks-debug
- [ ] Tool call loop terminates correctly (max_iterations or no tools)
- [ ] Slash commands (/clear, /history, /depth) work correctly
- [ ] Ctrl-C handling works
- [ ] File write_file tool doesn't allow writing outside current directory (security)

---

## Conclusion

AGENCY is architecturally sound and well-written, but **cannot function until the tool call extraction regex is fixed**. This is a one-line fix with ~90% of AGENCY being blocked by it.

After the regex fix, the output parsing fragility should be addressed before release.

The history truncation issue will only surface after extended use, so it can be a follow-up fix.
