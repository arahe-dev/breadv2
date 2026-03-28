#!/usr/bin/env python3
"""Phase 2: Detailed Validation with Tokenization & Layer RMS

This script:
A) Verifies tokenization matches BREAD vs ollama (token IDs)
B) Dumps per-layer RMS norms to find divergence point

Run: python validate_detailed.py
"""

import subprocess
import json
import sys
import re
from pathlib import Path

BREAD_EXE = r"C:\bread_v2\bread.exe"
BREAD_MINIMAL_EXE = r"C:\bread_v2\bread.exe"  # Same exe, run with --minimal
TEST_PROMPT = "The capital of France is"
TEST_TOKENS = 10

# ================================================================
# Part B: Tokenization Verification
# ================================================================

def tokenize_with_ollama_cli(prompt: str) -> list:
    """Get token IDs using ollama run command

    This is a workaround: we run with verbose output and try to extract
    token information. Ideally we'd use a dedicated tokenizer API.
    """
    print("  Note: ollama CLI doesn't expose raw token IDs easily.")
    print("  Fallback: verifying tokenizer via BREAD's output\n")
    return None


def tokenize_with_bread(prompt: str) -> list:
    """Extract token IDs from BREAD's output

    BREAD outputs: "[n tokens: [123 456 789 ...]"
    """
    try:
        result = subprocess.run(
            [BREAD_EXE, "--prompt", prompt, "--tokens", "1"],
            capture_output=True, text=True, timeout=60
        )
        output = result.stdout + result.stderr

        # Look for pattern like "39 tokens: [27 91 316 62 2388 91 29 846 ...]"
        # The [ may appear on same line or next line
        match = re.search(r'(\d+)\s+tokens?:\s*\[([^\]]*)\]', output, re.DOTALL)
        if match:
            n_tokens = int(match.group(1))
            token_str = match.group(2)
            tokens = []
            # Parse tokens, skipping "..." and empty values
            for part in token_str.split():
                part = part.strip()
                if part and part != "...":
                    try:
                        tokens.append(int(part))
                    except ValueError:
                        pass
            if tokens:
                return tokens
        else:
            print(f"  ERROR: Could not extract tokens from BREAD output")
            print(f"  Output:\n{output[:500]}")
            return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def verify_tokenization(prompt: str) -> bool:
    """Verify tokenization is consistent"""
    print("=== Part B: Tokenization Verification ===\n")
    print(f"Prompt: \"{prompt}\"\n")

    print("[B1] Getting tokens from BREAD...")
    bread_tokens = tokenize_with_bread(prompt)
    if not bread_tokens:
        print("  Failed to get tokens from BREAD")
        return False

    print(f"  ✓ BREAD tokenized to {len(bread_tokens)} tokens:")
    print(f"    {bread_tokens[:15]}", end="")
    if len(bread_tokens) > 15:
        print(f"\n    ... ({len(bread_tokens)-15} more)")
    else:
        print()

    print("\n[B2] Verifying consistency")
    print("  ✓ Tokenization extracted from BREAD")
    print("  Note: Without direct ollama tokenizer API, can't compare directly")
    print("  → Workaround: Compare final outputs")
    print("  → If tokens differ, generated text will differ significantly\n")

    return True


# ================================================================
# Part A: Per-Layer RMS Dumping
# ================================================================

def parse_bread_output_for_rms(output: str) -> dict:
    """Extract RMS information from BREAD output if available

    Looks for lines like "Layer 0: rms=0.1234" (from stderr with --debug flag)
    """
    lines = output.split('\n')
    rms_data = {}

    # Look for lines like "Layer 0: rms=0.1234" or "layer 0: rms=0.1234"
    for line in lines:
        match = re.search(r'[Ll]ayer\s+(\d+):\s+rms\s*=\s*([\d.e+-]+)', line)
        if match:
            layer_idx = int(match.group(1))
            rms_val = float(match.group(2))
            rms_data[layer_idx] = rms_val

    return rms_data


def run_bread_with_layer_dump(prompt: str, mode: str = "normal") -> dict:
    """Run BREAD and attempt to extract per-layer RMS

    Note: This requires BREAD to have debug output enabled.
    For now, we run it and indicate what should be added.
    """
    print(f"\n=== Part A: Per-Layer RMS Dump ({mode} mode) ===\n")

    args = [BREAD_EXE, "--prompt", prompt, "--tokens", "5"]
    if mode == "minimal":
        args.append("--minimal")

    print(f"Running: {' '.join(args)}\n")

    try:
        result = subprocess.run(
            args,
            capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr

        # Try to parse RMS data
        rms_data = parse_bread_output_for_rms(output)

        if rms_data:
            print("  ✓ Found per-layer RMS norms:")
            for layer, rms in sorted(rms_data.items()):
                print(f"    Layer {layer:2d}: RMS = {rms:.6f}")
            return {"rms": rms_data, "output": output}
        else:
            print("  ⚠ No per-layer RMS found in output")
            print("  (BREAD needs to be modified to output this)")
            print("\n  Output preview (first 1000 chars):")
            print("  " + output[:1000].replace('\n', '\n  '))
            return {"rms": {}, "output": output}

    except subprocess.TimeoutExpired:
        print("  ERROR: BREAD timed out")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def compare_rms_between_modes(normal_rms: dict, minimal_rms: dict) -> dict:
    """Compare RMS norms between normal and minimal modes

    If they diverge at the same layer, it's in BREAD's core logic.
    If they match, the issue is in expert caching or quantization.
    """
    print("\n=== Comparing Normal vs Minimal Mode ===\n")

    if not normal_rms or not minimal_rms:
        print("  Insufficient data to compare")
        return {}

    all_layers = set(normal_rms.keys()) | set(minimal_rms.keys())

    for layer in sorted(all_layers):
        n = normal_rms.get(layer, None)
        m = minimal_rms.get(layer, None)

        if n is not None and m is not None:
            diff = abs(n - m) / (abs(n) + 1e-8)
            status = "✓" if diff < 0.01 else "✗"
            print(f"  Layer {layer:2d}: normal={n:.6f}, minimal={m:.6f}, diff={diff:.2%} {status}")
        elif n is not None:
            print(f"  Layer {layer:2d}: only in normal mode")
        else:
            print(f"  Layer {layer:2d}: only in minimal mode")

    return {}


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 60)
    print("Phase 2: Detailed Validation")
    print("A) Tokenization Verification")
    print("B) Per-Layer RMS Dumping")
    print("=" * 60)
    print()

    # Part B: Tokenization
    if not verify_tokenization(TEST_PROMPT):
        print("Tokenization verification failed\n")
        # Don't exit; continue with part A

    # Part A: Layer-by-layer RMS
    print("\n" + "=" * 60)

    normal_result = run_bread_with_layer_dump(TEST_PROMPT, mode="normal")
    if not normal_result:
        print("\nFailed to run BREAD in normal mode")
        return 1

    print("\n" + "-" * 60)

    minimal_result = run_bread_with_layer_dump(TEST_PROMPT, mode="minimal")
    if not minimal_result:
        print("\nFailed to run BREAD in minimal mode")
        return 1

    # Compare modes
    normal_rms = normal_result.get("rms", {})
    minimal_rms = minimal_result.get("rms", {})
    compare_rms_between_modes(normal_rms, minimal_rms)

    # Final diagnosis
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)

    if normal_rms and minimal_rms:
        print("✓ Per-layer RMS data available")
        print("  Use this to identify the first layer where BREAD diverges")
    else:
        print("⚠ Per-layer RMS not available in current BREAD build")
        print("\nTo enable RMS dumping, modify BREAD to:")
        print("  1. Add --debug flag")
        print("  2. After each layer, compute & print RMS norm")
        print("  3. Format: 'Layer 0: rms=0.123456'")
        print("\nSee: main.cu apply_output_norm() as reference")

    print("\nNext steps:")
    print("  1. Add per-layer RMS output to BREAD")
    print("  2. Compare normal vs minimal mode RMS")
    print("  3. Focus debugging on first diverging layer")

    return 0


if __name__ == "__main__":
    sys.exit(main())
