mod bread;
mod hermes;
mod tools;
mod repl;

use std::path::PathBuf;
use anyhow::Result;

#[derive(Debug)]
struct Config {
    bread_exe: PathBuf,
    max_tokens: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bread_exe: PathBuf::from(".\\bread.exe"),
            max_tokens: 256,
        }
    }
}

fn main() -> Result<()> {
    let mut config = Config::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bread" => {
                if let Some(path) = args.next() {
                    let pb = PathBuf::from(&path);
                    if !pb.exists() {
                        eprintln!("Error: bread.exe not found at: {}", path);
                        return Ok(());
                    }
                    if !pb.is_file() {
                        eprintln!("Error: {} is not a file", path);
                        return Ok(());
                    }
                    config.bread_exe = pb;
                }
            }
            "--tokens" => {
                if let Some(tokens) = args.next() {
                    config.max_tokens = match tokens.parse::<u32>() {
                        Ok(n) if n > 0 => n,
                        _ => {
                            eprintln!("Warning: invalid token count '{}', using default 256", tokens);
                            256
                        }
                    };
                }
            }
            "--help" => {
                println!("BREAD AGENCY — Hermes Agent CLI");
                println!("Usage: agency [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --bread PATH       Path to bread.exe (default: .\\bread.exe)");
                println!("  --tokens N         Max tokens per response (default: 256)");
                println!("  --help             Show this message");
                return Ok(());
            }
            _ => {}
        }
    }

    // Start BREAD server
    println!("🍞 Starting BREAD inference engine...");
    let mut server = bread::BreadServer::start(&config.bread_exe, config.max_tokens)?;
    println!("✓ BREAD ready");
    println!();

    // Run interactive REPL
    println!("🤖 AGENCY Agent (type /help for commands)\n");
    repl::run_repl(&mut server)?;

    Ok(())
}
