@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1
cd /d C:\bread_v2
cl /nologo /W3 /O2 dequant_q4k_cpu.c /Fe:dequant_q4k_cpu.exe
if errorlevel 1 ( echo COMPILE FAILED & exit /b 1 )
echo.
dequant_q4k_cpu.exe
