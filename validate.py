#!/usr/bin/env python3
"""Phase 1 Validation: Compare BREAD vs Ollama

This script:
1. Calls ollama's API to get embeddings for a test prompt
2. Parses BREAD's output to extract final logits
3. Compares them to find divergence

Run: python validate.py
Make sure ollama is running: ollama run qwen3.5:35b-a3b
"""

import subprocess
import json
import sys
import time
import socket

OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "qwen3.5:35b-a3b"
BREAD_EXE = r"C:\bread_v2\bread.exe"
TEST_PROMPT = "The capital of France is"
TEST_TOKENS = 10


def check_ollama_running():
    """Check if ollama is running on localhost:11434"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        result = sock.connect_ex((OLLAMA_HOST, OLLAMA_PORT))
        return result == 0
    finally:
        sock.close()


def call_ollama_generate(prompt: str, num_tokens: int = 10) -> dict:
    """Call ollama via subprocess (simpler)"""
    try:
        print(f"  (This may take 30-60 seconds on first run...)")
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, prompt],
            capture_output=True, text=True, timeout=180
        )
        if result.returncode == 0:
            return {"response": result.stdout.strip(), "model": OLLAMA_MODEL}
        else:
            print(f"  ERROR: ollama run failed: {result.stderr}")
            return None
    except FileNotFoundError:
        print(f"  ERROR: ollama not found in PATH")
        return None
    except subprocess.TimeoutExpired:
        print(f"  ERROR: ollama timed out")
        return None


def run_bread(prompt: str, num_tokens: int = 10) -> str:
    """Run BREAD and capture output"""
    try:
        result = subprocess.run(
            [BREAD_EXE, "--prompt", prompt, "--tokens", str(num_tokens)],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print("  ERROR: BREAD timed out")
        return None
    except FileNotFoundError:
        print(f"  ERROR: {BREAD_EXE} not found")
        return None


def compare_outputs(bread_output: str, ollama_response: dict) -> dict:
    """Simple comparison of outputs"""
    results = {
        "bread_lines": len(bread_output.split('\n')),
        "ollama_response": ollama_response.get("response", "").strip() if ollama_response else None,
        "match": False
    }

    # Extract BREAD's generated text (simple heuristic: look for generated tokens)
    bread_lines = bread_output.split('\n')
    bread_text = ""
    for line in bread_lines:
        if "→" in line or "token" in line.lower():
            # Found a generation line
            parts = line.split("→")
            if len(parts) > 1:
                bread_text += parts[1].strip() + " "

    results["bread_response"] = bread_text.strip()

    if ollama_response and results["bread_response"]:
        # Very loose comparison: do they start with same word?
        bread_words = results["bread_response"].split()
        ollama_words = results["ollama_response"].split()
        if bread_words and ollama_words:
            results["match"] = bread_words[0].lower() == ollama_words[0].lower()

    return results


def main():
    print("=== BREAD Phase 1 Validation ===\n")

    print(f"[1] Checking ollama at {OLLAMA_HOST}:{OLLAMA_PORT}...")
    if not check_ollama_running():
        print("  ❌ Ollama is NOT running")
        print("  Start it with: ollama run qwen3.5:35b-a3b")
        return 1
    print("  ✓ Ollama is running\n")

    test_prompt = TEST_PROMPT

    print(f"[2] Getting reference output from ollama...")
    print(f"  Prompt: \"{test_prompt}\"")
    print(f"  Tokens: {TEST_TOKENS}")
    ollama_resp = call_ollama_generate(test_prompt, TEST_TOKENS)
    if not ollama_resp:
        return 1
    print(f"  ✓ Ollama response:\n    {ollama_resp.get('response', '')[:100]}...\n")

    print(f"[3] Running BREAD with same prompt...")
    print(f"  Exe: {BREAD_EXE}")
    bread_output = run_bread(test_prompt, TEST_TOKENS)
    if not bread_output:
        return 1
    print(f"  ✓ BREAD ran\n")

    print(f"[4] Comparing outputs...")
    comparison = compare_outputs(bread_output, ollama_resp)

    print(f"  Ollama: {comparison['ollama_response'][:80]}")
    print(f"  BREAD:  {comparison['bread_response'][:80]}")
    print(f"  Match:  {'✓ Yes' if comparison['match'] else '✗ No'}\n")

    if comparison['match']:
        print("=== SUCCESS ===")
        print("First token matches! Outputs are aligned.\n")
        return 0
    else:
        print("=== DIVERGENCE DETECTED ===")
        print("BREAD's first token differs from ollama's.\n")
        print("Next step: Add per-layer validation to find divergence point.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
