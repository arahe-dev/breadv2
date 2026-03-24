@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1
cd /d C:\bread_v2

echo === Building tokenizer_test.exe ===
cl.exe /DSELFTEST_MAIN /O2 /nologo /W3 tokenizer.c /Fe:tokenizer_test.exe
if errorlevel 1 ( echo FAILED: compile & exit /b 1 )

echo === Running tokenizer_selftest ===
tokenizer_test.exe
