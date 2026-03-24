# Makefile for BREAD — nmake (MSVC) build
# Usage: nmake
# Requires cl.exe on PATH (VS Build Tools)

CC     = cl
CFLAGS = /nologo /W3 /O2

all: bread_info.exe

bread_info.exe: bread_info.obj gguf.obj
	$(CC) $(CFLAGS) /Febread_info.exe bread_info.obj gguf.obj

bread_info.obj: bread_info.c gguf.h
	$(CC) $(CFLAGS) /c /Fobread_info.obj bread_info.c

gguf.obj: gguf.c gguf.h
	$(CC) $(CFLAGS) /c /Fogguf.obj gguf.c

clean:
	-@del /Q bread_info.exe 2>NUL
	-@del /Q bread_info.obj 2>NUL
	-@del /Q gguf.obj 2>NUL
