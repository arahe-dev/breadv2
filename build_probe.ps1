$VsPath = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
$vcvarsAll = "$VsPath\VC\Auxiliary\Build\vcvarsall.bat"
cmd /c """$vcvarsAll"" x64 && nvcc -O2 -x cu probe_tensors.cu gguf.c -I. -o probe_tensors.exe 2>&1"
Write-Host "Exit: $LASTEXITCODE"
