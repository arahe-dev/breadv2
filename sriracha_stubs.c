/* sriracha_stubs.c — provides symbols defined in one_layer.cu that
 * layer_ops.cu references but are not needed by sriracha.cu directly. */
#ifdef _WIN32
#  define _CRT_SECURE_NO_WARNINGS
#endif
#include <stdio.h>
#include <stdlib.h>
#include "gguf.h"
#include "loader.h"

const gguf_tensor_t *require_tensor(const gguf_ctx_t *g, const char *name)
{
    const gguf_tensor_t *t = gguf_find_tensor(g, name);
    if (!t) { fprintf(stderr, "require_tensor: '%s' not found\n", name); exit(1); }
    return t;
}

uint8_t *tensor_ram(const loader_t *L, const gguf_ctx_t *g, const char *name)
{
    const gguf_tensor_t *t = require_tensor(g, name);
    return L->pinned_data + L->data_offset + t->offset;
}
