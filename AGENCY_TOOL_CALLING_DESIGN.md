# AGENCY Tool Calling - Implementation Design

**Inspired by:** Claude API, Hermes XML format, OpenAI function calling

---

## Current Architecture (Broken)

```
User Input
   ↓
AGENCY builds prompt → BREAD subprocess
   ↓ (via stdin)
BREAD generates tokens → stdout
   ↓
AGENCY reads response (ignores any <tool_call> blocks)
   ↓
Response shown to user
```

**Problem:** BREAD has no way to request tools or receive results.

---

## New Architecture (Proper Tool Calling)

```
User Input
   ↓
build_prompt_with_tools() → Format: Hermes + tool definitions + history
   ↓
Send to BREAD stdin
   ↓
Parse BREAD output in real-time:
   ├─ Regular text → buffer for display
   └─ <tool_call>...</tool_call> → STOP, execute tool
   ↓
Execute tool in Rust (read_file, shell, etc.)
   ↓
build_continuation_prompt() → Add <tool_result> block + ask to continue
   ↓
Send continuation to BREAD stdin
   ↓
BREAD generates more (may call more tools or finish)
   ↓
Repeat until: no more <tool_call> blocks
   ↓
Final response shown to user + stored in history
```

---

## Key Design Decisions (From Claude Precedent)

### 1. **Prompt Format** (Already using this via Hermes)

```
<|im_start|>system
You are a helpful assistant with access to tools.

# Tools

You have access to the following tools:

<tool_definitions>
{
  "name": "read_file",
  "description": "Read the contents of a file",
  "input_schema": {
    "type": "object",
    "properties": {
      "path": {"type": "string", "description": "File path to read"}
    },
    "required": ["path"]
  }
}
...more tools...
</tool_definitions>

When you need to use a tool, output it in this XML format:
<tool_call>
{"name": "read_file", "arguments": {"path": "bread.h"}}
</tool_call>

The tool result will be provided in the next message.
<|im_end|>

<|im_start|>user
Read bread.h and tell me what the main config struct is
<|im_end|>

<|im_start|>assistant
I'll read the bread.h file for you.

<tool_call>
{"name": "read_file", "arguments": {"path": "bread.h"}}
</tool_call>
<|im_end|>

<|im_start|>user
<tool_result>
{"name": "read_file", "content": "typedef struct { ... } bread_config_t;"}
</tool_result>
<|im_end|>

<|im_start|>assistant
The main config struct is `bread_config_t`. It contains...
<|im_end|>
```

### 2. **Real-time Parsing** (Critical for UX)

Don't wait for full response. Parse as BREAD generates:

```rust
// In bread.rs, while reading from BREAD stdout:
let mut buffer = String::new();
for token in bread_stdout {
    buffer.push_str(&token);

    // Check for complete tool_call blocks
    if let Some(tool_call) = extract_tool_call(&buffer) {
        // Found a tool call!
        // 1. Display what we have so far
        // 2. Execute the tool
        // 3. Continue generation with result
        // 4. Clear buffer and keep going
    } else {
        // No complete tool call yet, display token to user
        print!("{}", token);
        stdout().flush();
    }
}
```

### 3. **Tool Loop** (Inspired by Claude's agentic loop)

```rust
loop {
    // 1. Build prompt with history + tool definitions
    let prompt = build_prompt_with_tools(&history, &tools);

    // 2. Send to BREAD and get response
    let response = bread_server.generate(&prompt);

    // 3. Parse for tool calls
    let (text, tool_calls) = parse_response_and_tools(&response);

    // 4. Display text
    println!("{}", text);

    // 5. If no tool calls, we're done
    if tool_calls.is_empty() {
        history.add(Role::Assistant, text);
        break;
    }

    // 6. Execute tools and collect results
    let mut tool_results = Vec::new();
    for tool_call in tool_calls {
        let result = execute_tool(&tool_call);
        tool_results.push((tool_call.name, result));
    }

    // 7. Add assistant message + tool results to history
    history.add(Role::Assistant, text);
    for (name, result) in tool_results {
        history.add(Role::ToolResult, format!("<tool_result>{}</tool_result>", result));
    }

    // 8. Loop: prompt BREAD again with history + results
}
```

---

## Implementation Steps

### Step 1: Add Tool Call Parser

**File:** `agency/src/tools.rs`

```rust
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

pub fn extract_tool_call(text: &str) -> Option<ToolCall> {
    // Use regex to find <tool_call>...</tool_call>
    // Parse JSON inside
    // Return structured ToolCall

    let re = regex::Regex::new(
        r#"<tool_call>\s*({.*?})\s*</tool_call>"#
    ).unwrap();

    if let Some(cap) = re.captures(text) {
        let json_str = &cap[1];
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
            return Some(ToolCall {
                name: val["name"].as_str()?.to_string(),
                arguments: val["arguments"].clone(),
            });
        }
    }
    None
}
```

### Step 2: Modify Response Parser

**File:** `agency/src/bread.rs`

Change `generate()` to return both text and any tool calls found:

```rust
pub fn generate_with_tools(&mut self, prompt: &str)
    -> (String, Vec<ToolCall>)
{
    // Send prompt to BREAD
    self.process.stdin.write_all(prompt.as_bytes())?;
    self.process.stdin.write_all(b"\n\n")?;

    let mut text = String::new();
    let mut tool_calls = Vec::new();

    // Read response line by line
    while let Some(line) = read_line_from_bread() {
        if line == "BREAD_END" {
            break;
        }

        text.push_str(&line);
        text.push('\n');

        // Check for tool calls as we read
        if let Some(tool_call) = extract_tool_call(&text) {
            tool_calls.push(tool_call);
            // Remove the tool_call from display text
            remove_tool_call_from_text(&mut text);
        }
    }

    (text, tool_calls)
}
```

### Step 3: Modify REPL Loop

**File:** `agency/src/repl.rs`

Replace the simple `generate_with_tools()` with the agentic loop:

```rust
pub fn run_agentic_loop(&mut self, user_input: &str) -> String {
    self.conversation.push(Turn {
        role: Role::User,
        content: user_input.to_string(),
    });

    let mut final_response = String::new();
    let max_iterations = self.max_depth;
    let mut iteration = 0;

    loop {
        iteration += 1;
        if iteration > max_iterations {
            break;
        }

        // Build prompt with full history + tool definitions
        let prompt = self.build_prompt_with_tools();

        // Generate with real-time tool call parsing
        let (text, tool_calls) = self.bread.generate_with_tools(&prompt)?;

        // Display response
        println!("{}", text);
        final_response.push_str(&text);

        // No tool calls = we're done
        if tool_calls.is_empty() {
            self.conversation.push(Turn {
                role: Role::Assistant,
                content: text,
            });
            break;
        }

        // Execute tools
        let mut tool_results = Vec::new();
        for tool_call in tool_calls {
            println!("  🔧 Executing: {}", tool_call.name);
            let result = self.execute_tool(&tool_call)?;
            tool_results.push((tool_call, result));
        }

        // Add to history: assistant message + tool results
        self.conversation.push(Turn {
            role: Role::Assistant,
            content: text,
        });

        for (tool_call, result) in tool_results {
            self.conversation.push(Turn {
                role: Role::ToolResult,
                content: format!(
                    "<tool_result>\n{{\n  \"name\": \"{}\",\n  \"content\": {}\n}}\n</tool_result>",
                    tool_call.name,
                    serde_json::to_string(&result)?
                ),
            });
        }

        // Loop continues: next iteration builds prompt with history + results
    }

    Ok(final_response)
}
```

### Step 4: Update Hermes Prompt Builder

**File:** `agency/src/hermes.rs`

```rust
pub fn build_prompt_with_tools(
    history: &[Turn],
    available_tools: &[Tool]
) -> String {
    let mut prompt = String::new();

    // System message with tool definitions
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str("You are a helpful assistant with access to tools.\n\n");
    prompt.push_str("# Available Tools\n\n");

    for tool in available_tools {
        prompt.push_str(&format!(
            "{{\n  \"name\": \"{}\",\n  \"description\": \"{}\",\n  \"input_schema\": {}\n}}\n\n",
            tool.name, tool.description,
            serde_json::to_string(&tool.input_schema)?
        ));
    }

    prompt.push_str("When you need to use a tool, output:\n");
    prompt.push_str("<tool_call>\n");
    prompt.push_str("{\"name\": \"tool_name\", \"arguments\": {...}}\n");
    prompt.push_str("</tool_call>\n");
    prompt.push_str("<|im_end|>\n\n");

    // Conversation history
    for turn in history {
        match turn.role {
            Role::User => prompt.push_str("<|im_start|>user\n"),
            Role::Assistant => prompt.push_str("<|im_start|>assistant\n"),
            Role::ToolResult => {
                // Tool results go in assistant context
                prompt.push_str(&turn.content);
                prompt.push_str("\n");
                continue;
            }
            _ => continue,
        }
        prompt.push_str(&turn.content);
        prompt.push_str("\n<|im_end|>\n\n");
    }

    // Start new assistant turn
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}
```

---

## Data Structures

```rust
#[derive(Debug, Clone)]
pub enum Role {
    User,
    Assistant,
    ToolResult,
}

#[derive(Debug, Clone)]
pub struct Turn {
    pub role: Role,
    pub content: String,
}

#[derive(Debug)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}
```

---

## Error Handling

**Handle edge cases:**

1. **Tool execution fails** → Return error message to BREAD
```rust
<tool_result>
{"name": "read_file", "error": "File not found: /nonexistent/path"}
</tool_result>
```

2. **Invalid tool call** → Ask BREAD to try again
```
Your tool call was invalid. Please check the format and try again.
```

3. **Tool timeout** → Interrupt and return timeout message
```
Tool execution timed out after 30 seconds.
```

4. **Max iterations reached** → Return what we have so far
```
Reached maximum tool call depth. Here's what I found so far...
```

---

## Testing

Test cases to verify:

1. **Single tool call** → User asks for file, BREAD calls read_file, returns result
2. **Chain of tool calls** → BREAD reads file → calls shell → reads another file
3. **No tool calls** → Regular conversation without tools
4. **Tool error handling** → BREAD gracefully handles tool failures
5. **Streaming** → Tokens display in real-time while parsing for tool calls

---

## Expected Outcome

After implementation:

```
>>> Read bread.h and tell me the main config struct

🔧 Executing: read_file
  (read bread.h successfully)

The main config struct is `bread_config_t`. It contains:
- Hidden dimension
- Layer count
- Vocab size
- And many other fields...

>>> What parameters does it have?

(Uses previous context, no tool call needed)

It has parameters like:
- h (hidden dimension): 8192
- num_layers: 40
- vocab_size: 248320
...
```

---

## Effort Estimate

- **Step 1-2:** Tool parser & response handler → 1 hour
- **Step 3:** REPL loop refactor → 1-2 hours
- **Step 4:** Prompt builder update → 30 min
- **Testing & debugging:** 1-2 hours
- **Edge cases & error handling:** 1 hour

**Total: 5-7 hours for fully working tool calling**

---

## Why This Works

✅ **Real-time parsing** → User sees output as it happens
✅ **Stateful conversation** → BREAD has full context of previous tool results
✅ **Hermes format** → Already what BREAD expects
✅ **Agentic loop** → BREAD can call multiple tools and reason over results
✅ **Claude-like UX** → Familiar to users of Claude API

This is essentially **bringing Claude's tool-use protocol to AGENCY with BREAD as the inference engine**.
