import re

# Extract timing from loader output
tests = {
    50: "bench_corrected_50.log",
    100: "bench_corrected_100.log", 
    200: "bench_corrected_200.log"
}

for tokens, filename in tests.items():
    try:
        with open(filename) as f:
            content = f.read()
        
        # Extract model load time
        match = re.search(r'loader: read 23\.87 GB in ([\d.]+) s', content)
        if match:
            load_time = float(match.group(1))
            
            # Count actual tokens generated (rough estimate based on output)
            output = content.split("BREAD_READY")[1].split("BREAD_END")[0] if "BREAD_READY" in content else ""
            word_count = len(output.split())
            
            # Estimate tokens (rough: 1 word ≈ 1-1.3 tokens)
            est_tokens = int(word_count * 1.2)
            
            print(f"\n=== Test: {tokens} tokens ===")
            print(f"Model load time: {load_time:.1f}s")
            print(f"Estimated output tokens: {est_tokens}")
            print(f"Output words: {word_count}")
    except FileNotFoundError:
        print(f"File not found: {filename}")

