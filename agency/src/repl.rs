use crate::bread::BreadServer;
use crate::hermes::{self, Message};
use crate::tools;
use rustyline::DefaultEditor;
use std::io::Write;
use colored::*;
use anyhow::Result;

pub fn run_repl(server: &mut BreadServer) -> Result<()> {
    let mut editor = DefaultEditor::new()?;
    let mut history: Vec<Message> = Vec::new();
    let mut max_tool_iterations = 5;

    loop {
        let readline = editor.readline(">>> ");
        match readline {
            Ok(line) => {
                if line.trim().is_empty() {
                    continue;
                }

                // Handle slash commands
                if line.starts_with('/') {
                    if !handle_slash_command(&line, &mut history, &mut max_tool_iterations) {
                        break;
                    }
                    continue;
                }

                // Add user message to history
                history.push(Message {
                    role: "user".to_string(),
                    content: line.clone(),
                });

                // Generate response (with tool loop)
                if let Err(e) = generate_with_tools(server, &mut history, max_tool_iterations, server.max_tokens) {
                    eprintln!("{}", format!("Error: {}", e).red());
                    history.pop(); // Remove failed user message
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("{}", "^C".yellow());
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("{}", "\nGoodbye!".green());
                break;
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }

    Ok(())
}

fn generate_with_tools(
    server: &mut BreadServer,
    history: &mut Vec<Message>,
    max_iterations: usize,
    max_tokens: u32,
) -> Result<()> {
    let system = hermes::system_prompt();
    truncate_history_if_needed(history, &system, max_tokens as usize);

    for iteration in 0..max_iterations {
        let prompt = hermes::build_prompt(history, &system);

        // Get response from BREAD
        let response = server.generate(&prompt)?;
        let cleaned_response = hermes::clean_response(&response);

        // Stream output to user
        print!("{}", cleaned_response.green());
        std::io::stdout().flush()?;

        // Check for tool calls
        let tool_calls = hermes::extract_tool_calls(&response);

        if tool_calls.is_empty() {
            // No tools called; we're done
            println!(); // Newline after response
            history.push(Message {
                role: "assistant".to_string(),
                content: cleaned_response,
            });
            break;
        }

        // Execute tools and collect responses
        let mut tool_responses = String::new();

        for tool_call in tool_calls {
            print!(
                "\n{}  {}{}",
                "→ Tool:".cyan(),
                tool_call.name.yellow(),
                " ".normal()
            );
            std::io::stdout().flush()?;

            match tools::execute_tool(&tool_call.name, &tool_call.arguments) {
                Ok(result) => {
                    println!("{}", " ✓".green());
                    tool_responses.push_str(&hermes::format_tool_response(&tool_call.name, &result));
                    tool_responses.push('\n');
                }
                Err(e) => {
                    println!("{}", " ✗".red());
                    tool_responses.push_str(&hermes::format_tool_response(
                        &tool_call.name,
                        &format!("Error: {}", e),
                    ));
                    tool_responses.push('\n');
                }
            }
        }

        // Add response with tool calls to history (clean response to remove <think> tags)
        history.push(Message {
            role: "assistant".to_string(),
            content: format!("{}\n{}", cleaned_response, tool_responses),
        });

        truncate_history_if_needed(history, &system, max_tokens as usize);

        if iteration < max_iterations - 1 {
            print!("\n{} Continuing...\n", "→".cyan());
        }
    }

    Ok(())
}

fn estimate_prompt_tokens(history: &[Message], system: &str) -> usize {
    let system_tokens = system.len() / 4;
    let history_tokens: usize = history.iter().map(|m| m.content.len() / 4 + 16).sum();
    system_tokens + history_tokens + 128
}

fn truncate_history_if_needed(history: &mut Vec<Message>, system: &str, max_tokens: usize) {
    let target_budget = (max_tokens * 3).max(512);

    while history.len() > 2 && estimate_prompt_tokens(history, system) > target_budget {
        history.remove(0);
        if !history.is_empty() {
            history.remove(0);
        }
    }
}

fn handle_slash_command(line: &str, history: &mut Vec<Message>, max_tool_iterations: &mut usize) -> bool {
    let parts: Vec<&str> = line.split_whitespace().collect();

    match parts.get(0).map(|s| *s) {
        Some("/quit") | Some("/exit") => return false,
        Some("/clear") => {
            history.clear();
            println!("{}", "✓ History cleared".green());
        }
        Some("/help") => {
            println!("{}", "AGENCY Commands:".cyan());
            println!("  /quit, /exit      Exit the REPL");
            println!("  /clear            Clear conversation history");
            println!("  /help             Show this message");
            println!("  /tools            List available tools");
            println!("  /depth N          Set max tool loop depth (default: 5)");
            println!("  /history          Show conversation history");
        }
        Some("/tools") => {
            println!("{}", "Available Tools:".cyan());
            println!("  • read_file      Read a file's contents");
            println!("  • list_dir       List directory contents");
            println!("  • shell          Run shell commands");
            println!("  • write_file     Write to a file");
        }
        Some("/depth") => {
            if let Some(depth_str) = parts.get(1) {
                if let Ok(depth) = depth_str.parse::<usize>() {
                    *max_tool_iterations = depth;
                    println!("{} Tool loop depth set to {}", "✓".green(), depth);
                }
            }
        }
        Some("/history") => {
            if history.is_empty() {
                println!("{}", "(empty)".dimmed());
            } else {
                for (i, msg) in history.iter().enumerate() {
                    let role_colored = if msg.role == "user" {
                        msg.role.cyan()
                    } else {
                        msg.role.green()
                    };
                    println!("{} {}: {}", i + 1, role_colored, &msg.content[..msg.content.len().min(60)]);
                }
            }
        }
        _ => {
            println!("{}", "Unknown command. Type /help for list.".yellow());
        }
    }

    true
}
