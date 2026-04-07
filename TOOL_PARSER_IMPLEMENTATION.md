# Tool Call Parser - Implementation Complete ✅

**Date:** 2026-04-08
**Status:** FULLY TESTED AND WORKING

---

## What Was Implemented

### 1. **ToolCall Struct**
```rust
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}
```

**Purpose:** Represents a parsed tool call from BREAD output
**Example:**
```rust
ToolCall {
    name: "read_file".to_string(),
    arguments: {"path": "bread.h"},
}
```

---

### 2. **Core Parser Functions**

#### `extract_tool_call(text: &str) -> Option<ToolCall>`
Extracts a **single tool call** from text.

```rust
let text = r#"I'll read that file.
<tool_call>
{"name": "read_file", "arguments": {"path": "bread.h"}}
</tool_call>"#;

let tool = extract_tool_call(text)?;
assert_eq!(tool.name, "read_file");
```

#### `extract_all_tool_calls(text: &str) -> Vec<ToolCall>`
Extracts **multiple tool calls** from text (handles chains).

```rust
let text = r#"
<tool_call>{"name": "read_file", "arguments": {"path": "file1.txt"}}</tool_call>
Some text.
<tool_call>{"name": "shell", "arguments": {"command": "ls"}}</tool_call>
"#;

let calls = extract_all_tool_calls(text);
assert_eq!(calls.len(), 2);
assert_eq!(calls[0].name, "read_file");
assert_eq!(calls[1].name, "shell");
```

---

### 3. **Helper Functions**

#### `remove_tool_calls_from_text(text: &mut String)`
Removes `<tool_call>` blocks from text for clean display.

```rust
let mut text = "Before <tool_call>{...}</tool_call> after".to_string();
remove_tool_calls_from_text(&mut text);
// Result: "Before  after"
```

#### `has_tool_calls(text: &str) -> bool`
Quick check if text contains any tool calls.

```rust
assert!(has_tool_calls("<tool_call>{}</tool_call>"));
assert!(!has_tool_calls("no tools here"));
```

#### `format_tool_result(tool_name: &str, result: &str) -> String`
Formats a tool result for feeding back to BREAD.

```rust
let result = format_tool_result("read_file", "file contents");
// Returns:
// <tool_result>
// {
//   "name": "read_file",
//   "content": "file contents"
// }
// </tool_result>
```

---

## Test Coverage

✅ **All 7 tests passing:**

| Test | Purpose | Status |
|------|---------|--------|
| `test_extract_tool_call` | Extract single tool call | ✅ PASS |
| `test_extract_all_tool_calls` | Extract multiple tool calls | ✅ PASS |
| `test_remove_tool_calls` | Clean tool calls from text | ✅ PASS |
| `test_has_tool_calls` | Detect presence of tool calls | ✅ PASS |
| `test_format_tool_result` | Format result for BREAD | ✅ PASS |
| `test_build_prompt` (hermes) | Hermes prompt building | ✅ PASS |
| `test_extract_tool_calls` (hermes) | Hermes tool extraction | ✅ PASS |

---

## Usage Example

### Scenario: BREAD generates a response with tool calls

```
BREAD Output:
I'll help you with that. Let me read the file first.

<tool_call>
{"name": "read_file", "arguments": {"path": "bread.h"}}
</tool_call>

Now let me search for something in it...

<tool_call>
{"name": "shell", "arguments": {"command": "grep -n 'bread_config' bread.h"}}
</tool_call>
```

### Parsing workflow:

```rust
// 1. Capture BREAD output
let bread_output = "...the text above...";

// 2. Extract all tool calls
let tool_calls = extract_all_tool_calls(bread_output);
// Result: [
//   ToolCall { name: "read_file", arguments: {...} },
//   ToolCall { name: "shell", arguments: {...} }
// ]

// 3. Remove tool calls from display
let mut display_text = bread_output.to_string();
remove_tool_calls_from_text(&mut display_text);
// Shows user the readable response without XML blocks

// 4. Execute each tool
for tool_call in &tool_calls {
    let result = execute_tool(&tool_call.name, &tool_call.arguments)?;
    let formatted = format_tool_result(&tool_call.name, &result);
    println!("🔧 {} → {}", tool_call.name, result);

    // Save for next prompt to BREAD
    tool_results.push(formatted);
}

// 5. Continue with results in next prompt
let continuation_prompt = build_continuation_with_results(tool_results);
```

---

## Integration Points (Next Steps)

### **Step 2:** Modify `bread.rs`
Update `BreadServer::generate()` to return tool calls alongside text:

```rust
pub fn generate_with_tools(&mut self, prompt: &str)
    -> Result<(String, Vec<ToolCall>)>
{
    // ... read BREAD output ...
    // Use extract_all_tool_calls() on the output
    // Separate text from tool calls
    // Return both
}
```

### **Step 3:** Modify `repl.rs`
Implement the agentic loop:

```rust
pub fn run_agentic_loop(&mut self, user_input: &str) -> Result<String> {
    loop {
        let prompt = self.build_prompt_with_tools();
        let (text, tool_calls) = self.bread.generate_with_tools(&prompt)?;

        println!("{}", text);

        if tool_calls.is_empty() {
            break;  // No more tools, done
        }

        for tool in &tool_calls {
            let result = tools::execute_tool(&tool.name, &tool.arguments)?;
            // ... add to history ...
        }
    }
    Ok(final_response)
}
```

---

## Technical Details

### Regex Pattern Used
```regex
<tool_call>\s*(\{.*?\})\s*</tool_call>
```

**Explanation:**
- `<tool_call>` - Literal opening tag
- `\s*` - Optional whitespace
- `(\{.*?\})` - Captures JSON (non-greedy)
- `</tool_call>` - Closing tag

### Error Handling
- **Invalid JSON:** Skipped silently (returns None/empty vector)
- **Missing fields:** Tool name must be present, arguments default to `{}`
- **Multiple calls:** All extracted and returned as vector

### Performance
- **Regex compilation:** Cached (compile once, use many times)
- **Parsing:** Linear O(n) scan
- **JSON parsing:** Only for captured blocks (not entire response)

---

## Files Modified

| File | Changes |
|------|---------|
| `agency/src/tools.rs` | Added ToolCall struct, 6 functions, 5 unit tests |
| `agency/Cargo.toml` | No changes (regex already in deps) |

---

## Next Phase

**Phase 2: Response Handler**
- Modify `bread.rs` to parse BREAD output in real-time
- Return tuple of (text, tool_calls)
- Stream output to user while parsing

**Estimated effort:** 1-2 hours

---

## Verification Command

Run tests anytime:
```bash
cd /c/bread_v2/agency
cargo test 2>&1 | grep "test result"
```

Expected output:
```
test result: ok. 7 passed; 0 failed
```

---

**Status:** ✅ COMPLETE AND TESTED
**Next Action:** Implement Phase 2 (Response Handler in bread.rs)
