# PowerShell script to build BREAD with async GPU-CPU orchestration
$VsPath = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
$vcvarsAll = "$VsPath\VC\Auxiliary\Build\vcvarsall.bat"

# Set environment and build with OpenMP support
cmd /c """$vcvarsAll"" x64 && nvcc -O2 -x cu main.cu one_layer.cu kernels.cu loader.c gguf.c tokenizer.c bread.c layer_ops.cu buffer_pool.c hooks.c progress_tracking.c expert_bench.cu expert_profile.cu -I. -Xcompiler /openmp -o bread.exe"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful" -ForegroundColor Green
    .\bread.exe --prompt "The capital of France is" --tokens 5
} else {
    Write-Host "Build failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
