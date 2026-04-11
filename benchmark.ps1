# BREAD Performance Baseline Benchmark
# Usage: powershell -ExecutionPolicy Bypass -File benchmark.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BREAD Performance Baseline Benchmark" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

cd C:\bread_v2

function Benchmark-Tokens {
    param(
        [int]$Tokens
    )

    $outputFile = "bench_${Tokens}.log"

    Write-Host "[TEST] Running with $Tokens tokens..." -ForegroundColor Yellow

    # Create test input
    $testInput = @"
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is 2+2?
<|im_end|>
<|im_start|>assistant
"@

    # Measure time
    $startTime = Get-Date
    $output = $testInput | & .\bread.exe --server --tokens $Tokens 2>&1 | Tee-Object -FilePath $outputFile
    $endTime = Get-Date

    $elapsed = ($endTime - $startTime).TotalSeconds
    $throughput = [math]::Round($Tokens / $elapsed, 2)

    # Check for BREAD_END
    if ($output | Select-String "BREAD_END" -Quiet) {
        Write-Host "  ✓ Generation completed" -ForegroundColor Green
        Write-Host "  Time: ${elapsed:F2}s"
        Write-Host "  Throughput: $throughput tok/s" -ForegroundColor Cyan
    } else {
        Write-Host "  ✗ BREAD_END not found - may have timed out" -ForegroundColor Red
    }

    Write-Host "  Output saved to: $outputFile"
    Write-Host ""
}

# Run benchmarks
Benchmark-Tokens 50
Benchmark-Tokens 100
Benchmark-Tokens 200

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Benchmark complete!" -ForegroundColor Cyan
Write-Host "Output files:" -ForegroundColor Cyan
Write-Host "  bench_50.log"
Write-Host "  bench_100.log"
Write-Host "  bench_200.log"
Write-Host "========================================" -ForegroundColor Cyan
