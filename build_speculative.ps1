# build_speculative.ps1 — Build SRIRACHA Step 2 speculative decoding driver
$VsPath = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
$vcvarsAll = "$VsPath\VC\Auxiliary\Build\vcvarsall.bat"

# one_layer.cu provides require_tensor/tensor_ram — no sriracha_stubs.c needed
cmd /c """$vcvarsAll"" x64 && nvcc -O2 -x cu main_speculative.cu one_layer.cu layer_ops.cu kernels.cu sriracha.cu loader.c gguf.c tokenizer.c bread.c -I. -o speculative.exe 2>&1"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful -> speculative.exe" -ForegroundColor Green
} else {
    Write-Host "Build failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
