@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1
cd /d C:\bread_v2

echo === Building kernels_test.exe ===
nvcc -DSELFTEST_MAIN -O2 -Xcompiler "/nologo /W3" kernels.cu -o kernels_test.exe
if errorlevel 1 ( echo FAILED: kernels.cu compile & exit /b 1 )

echo === Running kernels_selftest ===
kernels_test.exe
