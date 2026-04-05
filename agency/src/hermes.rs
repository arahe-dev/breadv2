use regex::Regex;
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,  // "user", "assistant"
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    pub fn parse(json_str: &str) -> Option<Self> {
        let value: Value = serde_json::from_str(json_str).ok()?;
        Some(ToolCall {
            name: value.get("name")?.as_str()?.to_string(),
            arguments: value.get("arguments")?.clone(),
        })
    }
}

/// System prompt with tool definitions and explicit response style guidance.
pub fn system_prompt() -> String {
    r#"You are a helpful assistant.
Answer the user's request directly.
Do not output chain-of-thought or "Thinking Process".

Only if you need a tool, reply with exactly one block:
<tool_call>
{"name":"read_file|list_dir|shell|write_file","arguments":{...}}
</tool_call>

Tool results will be sent back inside:
<tool_response>...</tool_response>"#
        .to_string()
}

/// Build Qwen3 chat format prompt from conversation history
pub fn build_prompt(history: &[Message], system: &str) -> String {
    let mut prompt = format!("<|im_start|>system\n{}\n<|im_end|>\n", system);

    for msg in history {
        prompt.push_str(&format!("<|im_start|>{}\n{}\n<|im_end|>\n", msg.role, msg.content));
    }

    // Open assistant response without pre-seeding hidden reasoning tags.
    prompt.push_str("<|im_start|>assistant\n");

    prompt
}

/// Extract all <tool_call> blocks from text
pub fn extract_tool_calls(text: &str) -> Vec<ToolCall> {
    let re = Regex::new(r#"<tool_call>\s*\n([\s\S]*?)\n</tool_call>"#).unwrap();
    re.captures_iter(text)
        .filter_map(|cap| {
            cap.get(1).and_then(|m| ToolCall::parse(m.as_str()))
        })
        .collect()
}

/// Strip <think>...</think> tags from response for storage in history
pub fn clean_response(response: &str) -> String {
    let think_re = Regex::new(r"(?s)<think>.*?</think>").unwrap();
    let mut cleaned = think_re.replace_all(response, "").to_string();

    if let Some(idx) = cleaned.find("<tool_call>") {
        cleaned.truncate(idx);
    }

    let mut lines = Vec::new();
    let mut skipping_reasoning = false;

    for line in cleaned.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Thinking Process:") {
            skipping_reasoning = true;
            continue;
        }
        if skipping_reasoning {
            if trimmed.is_empty() {
                skipping_reasoning = false;
            }
            continue;
        }
        lines.push(line);
    }

    lines.join("\n").trim().to_string()
}

/// Format a tool response for inclusion in conversation
pub fn format_tool_response(tool_name: &str, content: &str) -> String {
    format!(
        "<tool_response>\n{}\n</tool_response>",
        json!({
            "name": tool_name,
            "content": content
        })
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_tool_calls() {
        let text = r#"Some text here
<tool_call>
{"name": "read_file", "arguments": {"path": "main.cu"}}
</tool_call>
More text"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
    }

    #[test]
    fn test_build_prompt() {
        let history = vec![
            Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "Hi there!".to_string(),
            },
        ];
        let prompt = build_prompt(&history, "You are helpful.");
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("Hello"));
    }
}
