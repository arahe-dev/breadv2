#ifndef GGUF_H
#define GGUF_H

/* gguf.h — GGUF binary format types and parser API
 * C99, no external dependencies.
 */

#include <stdint.h>
#include <stddef.h>

/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */

#define GGUF_MAGIC             0x46554747u   /* "GGUF" little-endian */
#define GGUF_DEFAULT_ALIGNMENT 32u
#define GGUF_MAX_DIMS          4

/* ------------------------------------------------------------------ */
/* ggml tensor quantisation type IDs                                   */
/* ------------------------------------------------------------------ */

typedef enum {
    GGML_TYPE_F32      = 0,
    GGML_TYPE_F16      = 1,
    GGML_TYPE_Q4_0     = 2,
    GGML_TYPE_Q4_1     = 3,
    /* 4, 5 reserved */
    GGML_TYPE_Q5_0     = 6,
    GGML_TYPE_Q5_1     = 7,
    GGML_TYPE_Q8_0     = 8,
    GGML_TYPE_Q8_1     = 9,
    GGML_TYPE_Q2_K     = 10,
    GGML_TYPE_Q3_K     = 11,
    GGML_TYPE_Q4_K     = 12,
    GGML_TYPE_Q5_K     = 13,
    GGML_TYPE_Q6_K     = 14,
    GGML_TYPE_Q8_K     = 15,
    GGML_TYPE_IQ2_XXS  = 16,
    GGML_TYPE_IQ2_XS   = 17,
    GGML_TYPE_IQ3_XXS  = 18,
    GGML_TYPE_IQ1_S    = 19,
    GGML_TYPE_IQ4_NL   = 20,
    GGML_TYPE_IQ3_S    = 21,
    GGML_TYPE_IQ2_S    = 22,
    GGML_TYPE_IQ4_XS   = 23,
    GGML_TYPE_I8       = 24,
    GGML_TYPE_I16      = 25,
    GGML_TYPE_I32      = 26,
    GGML_TYPE_I64      = 27,
    GGML_TYPE_F64      = 28,
    GGML_TYPE_IQ1_M    = 29,
    GGML_TYPE_BF16     = 30,
    GGML_TYPE_COUNT
} ggml_type_t;

/* ------------------------------------------------------------------ */
/* GGUF metadata value types                                           */
/* ------------------------------------------------------------------ */

typedef enum {
    GGUF_VAL_UINT8   = 0,
    GGUF_VAL_INT8    = 1,
    GGUF_VAL_UINT16  = 2,
    GGUF_VAL_INT16   = 3,
    GGUF_VAL_UINT32  = 4,
    GGUF_VAL_INT32   = 5,
    GGUF_VAL_FLOAT32 = 6,
    GGUF_VAL_BOOL    = 7,
    GGUF_VAL_STRING  = 8,
    GGUF_VAL_ARRAY   = 9,
    GGUF_VAL_UINT64  = 10,
    GGUF_VAL_INT64   = 11,
    GGUF_VAL_FLOAT64 = 12
} gguf_val_type_t;

/* ------------------------------------------------------------------ */
/* Tensor descriptor (filled in by gguf_open, never modified after)   */
/* ------------------------------------------------------------------ */

typedef struct {
    char     *name;                     /* heap-alloc'd, null-terminated */
    uint32_t  type;                     /* ggml_type_t */
    uint32_t  n_dims;
    uint64_t  dims[GGUF_MAX_DIMS];      /* dims[0] = innermost (columns) */
    uint64_t  offset;                   /* bytes from start of data section */
    uint64_t  size;                     /* bytes of tensor data */
    uint64_t  n_elems;                  /* total element count */
} gguf_tensor_t;

/* Opaque context */
typedef struct gguf_ctx gguf_ctx_t;

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

#ifdef __cplusplus
extern "C" {
#endif

/* Open a GGUF file and parse the header + tensor index.
 * Does NOT load weight data into memory.
 * Returns NULL on error (message printed to stderr). */
gguf_ctx_t *gguf_open(const char *path);

/* Close the file and free all resources. */
void gguf_close(gguf_ctx_t *ctx);

/* Header accessors */
uint32_t gguf_version(const gguf_ctx_t *ctx);
uint64_t gguf_num_tensors(const gguf_ctx_t *ctx);

/* Absolute byte offset in the file where tensor data begins. */
uint64_t gguf_data_offset(const gguf_ctx_t *ctx);

/* Get tensor by index [0, n_tensors). Returns NULL if out of range. */
const gguf_tensor_t *gguf_tensor_by_idx(const gguf_ctx_t *ctx, uint64_t idx);

/* Find tensor by exact name. Returns NULL if not found. */
const gguf_tensor_t *gguf_find_tensor(const gguf_ctx_t *ctx, const char *name);

/* Human-readable name for a ggml type ID (e.g. "Q4_K", "F32"). */
const char *ggml_type_name(uint32_t type);

/* Byte size of a tensor with n_elems elements of the given ggml type.
 * Returns 0 for unknown or unsupported types. */
uint64_t ggml_tensor_nbytes(uint64_t n_elems, uint32_t type);

#ifdef __cplusplus
}
#endif

#endif /* GGUF_H */
