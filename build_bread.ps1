# PowerShell script to build BREAD
$VsPath = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
$vcvarsAll = "$VsPath\VC\Auxiliary\Build\vcvarsall.bat"

# Set environment and build
cmd /c """$vcvarsAll"" x64 && nvcc -O2 -x cu main.cu one_layer.cu layer_ops.cu kernels.cu loader.c gguf.c tokenizer.c bread.c -I. -o bread.exe"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful" -ForegroundColor Green
} else {
    Write-Host "Build failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
