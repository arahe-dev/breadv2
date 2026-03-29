# build_sriracha.ps1 — Build SRIRACHA Step 1 test driver
$VsPath = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
$vcvarsAll = "$VsPath\VC\Auxiliary\Build\vcvarsall.bat"

cmd /c """$vcvarsAll"" x64 && nvcc -O2 -x cu main_sriracha.cu sriracha.cu layer_ops.cu kernels.cu sriracha_stubs.c loader.c gguf.c tokenizer.c bread.c -I. -o sriracha.exe 2>&1"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful -> sriracha.exe" -ForegroundColor Green
} else {
    Write-Host "Build failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
