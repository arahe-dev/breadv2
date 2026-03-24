/* gguf.c — GGUF binary format parser
 *
 * Parses the header and tensor index of a GGUF model file without loading
 * any weight data.  Handles GGUF versions 2 and 3.
 *
 * C99, no external dependencies beyond the standard library.
 */

#ifdef _WIN32
#  define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "gguf.h"

/* ------------------------------------------------------------------ */
/* Portable 64-bit fseek / ftell                                       */
/* On Windows, fseek/ftell are limited to 32-bit offsets.             */
/* ------------------------------------------------------------------ */

#ifdef _WIN32
#  define fseek64(f, o, w)  _fseeki64((f), (__int64)(o), (w))
#  define ftell64(f)        ((int64_t)_ftelli64(f))
#else
#  include <sys/types.h>
#  define fseek64(f, o, w)  fseeko((f), (off_t)(o), (w))
#  define ftell64(f)        ((int64_t)ftello(f))
#endif

/* ------------------------------------------------------------------ */
/* Internal context                                                     */
/* ------------------------------------------------------------------ */

struct gguf_ctx {
    FILE           *fp;
    uint32_t        version;
    uint64_t        n_kv;
    uint64_t        n_tensors;
    uint32_t        alignment;
    uint64_t        data_off;    /* absolute byte offset of data section */
    gguf_tensor_t  *tensors;
};

/* ------------------------------------------------------------------ */
/* ggml type size table                                                 */
/*                                                                      */
/* Each entry: { type_id, elements_per_block, bytes_per_block }        */
/*                                                                      */
/* Block sizes from ggml-quants.h:                                     */
/*   QK4_0=32  QK8_0=32  QK_K=256                                     */
/* ------------------------------------------------------------------ */

typedef struct { uint32_t type; uint32_t block_size; uint32_t type_size; } type_info_t;

static const type_info_t g_types[] = {
    /* Plain float */
    { GGML_TYPE_F32,      1,    4   },
    { GGML_TYPE_F16,      1,    2   },
    { GGML_TYPE_BF16,     1,    2   },
    { GGML_TYPE_F64,      1,    8   },
    /* Integer */
    { GGML_TYPE_I8,       1,    1   },
    { GGML_TYPE_I16,      1,    2   },
    { GGML_TYPE_I32,      1,    4   },
    { GGML_TYPE_I64,      1,    8   },
    /* Legacy k=32 quants */
    { GGML_TYPE_Q4_0,     32,   18  },  /* d(2) + qs[16]          */
    { GGML_TYPE_Q4_1,     32,   20  },  /* d(2) + m(2) + qs[16]   */
    { GGML_TYPE_Q5_0,     32,   22  },  /* d(2) + qh(4) + qs[16]  */
    { GGML_TYPE_Q5_1,     32,   24  },  /* d(2)+m(2)+qh(4)+qs[16] */
    { GGML_TYPE_Q8_0,     32,   34  },  /* d(2) + qs[32]          */
    { GGML_TYPE_Q8_1,     32,   36  },  /* d(2)+s(2)+qs[32]       */
    /* K-quants (QK_K=256) */
    { GGML_TYPE_Q2_K,     256,  84  },  /* qs[64]+scales[16]+d(2)+dmin(2) */
    { GGML_TYPE_Q3_K,     256,  110 },  /* hmask[32]+qs[64]+scales[12]+d(2) */
    { GGML_TYPE_Q4_K,     256,  144 },  /* d(2)+dmin(2)+scales[12]+qs[128] */
    { GGML_TYPE_Q5_K,     256,  176 },  /* d(2)+dmin(2)+scales[12]+qh[32]+qs[128] */
    { GGML_TYPE_Q6_K,     256,  210 },  /* ql[128]+qh[64]+scales[16]+d(2) */
    { GGML_TYPE_Q8_K,     256,  292 },  /* d(4)+qs[256]+bsums[32] */
    /* IQ quants */
    { GGML_TYPE_IQ2_XXS,  256,  66  },
    { GGML_TYPE_IQ2_XS,   256,  74  },
    { GGML_TYPE_IQ3_XXS,  256,  98  },
    { GGML_TYPE_IQ1_S,    256,  50  },
    { GGML_TYPE_IQ4_NL,   32,   18  },
    { GGML_TYPE_IQ3_S,    256,  110 },
    { GGML_TYPE_IQ2_S,    256,  84  },
    { GGML_TYPE_IQ4_XS,   256,  136 },
    { GGML_TYPE_IQ1_M,    256,  56  },
};

#define N_TYPES (sizeof(g_types) / sizeof(g_types[0]))

static const type_info_t *lookup_type(uint32_t type) {
    for (size_t i = 0; i < N_TYPES; i++)
        if (g_types[i].type == type) return &g_types[i];
    return NULL;
}

const char *ggml_type_name(uint32_t type) {
    switch ((ggml_type_t)type) {
        case GGML_TYPE_F32:     return "F32";
        case GGML_TYPE_F16:     return "F16";
        case GGML_TYPE_BF16:    return "BF16";
        case GGML_TYPE_F64:     return "F64";
        case GGML_TYPE_I8:      return "I8";
        case GGML_TYPE_I16:     return "I16";
        case GGML_TYPE_I32:     return "I32";
        case GGML_TYPE_I64:     return "I64";
        case GGML_TYPE_Q4_0:    return "Q4_0";
        case GGML_TYPE_Q4_1:    return "Q4_1";
        case GGML_TYPE_Q5_0:    return "Q5_0";
        case GGML_TYPE_Q5_1:    return "Q5_1";
        case GGML_TYPE_Q8_0:    return "Q8_0";
        case GGML_TYPE_Q8_1:    return "Q8_1";
        case GGML_TYPE_Q2_K:    return "Q2_K";
        case GGML_TYPE_Q3_K:    return "Q3_K";
        case GGML_TYPE_Q4_K:    return "Q4_K";
        case GGML_TYPE_Q5_K:    return "Q5_K";
        case GGML_TYPE_Q6_K:    return "Q6_K";
        case GGML_TYPE_Q8_K:    return "Q8_K";
        case GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
        case GGML_TYPE_IQ2_XS:  return "IQ2_XS";
        case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
        case GGML_TYPE_IQ1_S:   return "IQ1_S";
        case GGML_TYPE_IQ4_NL:  return "IQ4_NL";
        case GGML_TYPE_IQ3_S:   return "IQ3_S";
        case GGML_TYPE_IQ2_S:   return "IQ2_S";
        case GGML_TYPE_IQ4_XS:  return "IQ4_XS";
        case GGML_TYPE_IQ1_M:   return "IQ1_M";
        default:                return "UNKNOWN";
    }
}

uint64_t ggml_tensor_nbytes(uint64_t n_elems, uint32_t type) {
    const type_info_t *ti = lookup_type(type);
    if (!ti || ti->block_size == 0) return 0;
    return (n_elems / ti->block_size) * ti->type_size;
}

/* ------------------------------------------------------------------ */
/* Low-level read helpers — call exit(1) on short read                */
/* ------------------------------------------------------------------ */

static void fatal(const char *msg) {
    fprintf(stderr, "gguf error: %s\n", msg);
    exit(1);
}

static void xread(FILE *fp, void *buf, size_t n) {
    if (fread(buf, 1, n, fp) != n)
        fatal("unexpected end of file");
}

static uint8_t  r_u8 (FILE *fp) { uint8_t  v; xread(fp, &v, 1); return v; }
static int8_t   r_i8 (FILE *fp) { int8_t   v; xread(fp, &v, 1); return v; }
static uint16_t r_u16(FILE *fp) { uint16_t v; xread(fp, &v, 2); return v; }
static int16_t  r_i16(FILE *fp) { int16_t  v; xread(fp, &v, 2); return v; }
static uint32_t r_u32(FILE *fp) { uint32_t v; xread(fp, &v, 4); return v; }
static int32_t  r_i32(FILE *fp) { int32_t  v; xread(fp, &v, 4); return v; }
static uint64_t r_u64(FILE *fp) { uint64_t v; xread(fp, &v, 8); return v; }
static int64_t  r_i64(FILE *fp) { int64_t  v; xread(fp, &v, 8); return v; }
static float    r_f32(FILE *fp) { float    v; xread(fp, &v, 4); return v; }
static double   r_f64(FILE *fp) { double   v; xread(fp, &v, 8); return v; }

/* Read a GGUF string: u64 length followed by raw UTF-8 bytes.
 * Returns a heap-allocated null-terminated C string. */
static char *read_str(FILE *fp) {
    uint64_t len = r_u64(fp);
    char *s = (char *)malloc(len + 1);
    if (!s) fatal("out of memory allocating string");
    xread(fp, s, (size_t)len);
    s[len] = '\0';
    return s;
}

/* Skip a GGUF string without allocating. */
static void skip_str(FILE *fp) {
    uint64_t len = r_u64(fp);
    if (len > 0 && fseek64(fp, (int64_t)len, SEEK_CUR) != 0)
        fatal("fseek failed while skipping string");
}

/* ------------------------------------------------------------------ */
/* Recursive value skipper                                             */
/* ------------------------------------------------------------------ */

static void skip_val(FILE *fp, uint32_t vtype) {
    switch (vtype) {
        case GGUF_VAL_UINT8:
        case GGUF_VAL_INT8:
        case GGUF_VAL_BOOL:    (void)r_u8(fp);  return;
        case GGUF_VAL_UINT16:
        case GGUF_VAL_INT16:   (void)r_u16(fp); return;
        case GGUF_VAL_UINT32:
        case GGUF_VAL_INT32:
        case GGUF_VAL_FLOAT32: (void)r_u32(fp); return;
        case GGUF_VAL_UINT64:
        case GGUF_VAL_INT64:
        case GGUF_VAL_FLOAT64: (void)r_u64(fp); return;
        case GGUF_VAL_STRING:  skip_str(fp);     return;

        case GGUF_VAL_ARRAY: {
            uint32_t elem_type = r_u32(fp);
            uint64_t count     = r_u64(fp);

            /* For fixed-size element types, skip the whole block in one seek. */
            size_t elem_size = 0;
            switch (elem_type) {
                case GGUF_VAL_UINT8:
                case GGUF_VAL_INT8:
                case GGUF_VAL_BOOL:    elem_size = 1; break;
                case GGUF_VAL_UINT16:
                case GGUF_VAL_INT16:   elem_size = 2; break;
                case GGUF_VAL_UINT32:
                case GGUF_VAL_INT32:
                case GGUF_VAL_FLOAT32: elem_size = 4; break;
                case GGUF_VAL_UINT64:
                case GGUF_VAL_INT64:
                case GGUF_VAL_FLOAT64: elem_size = 8; break;
                default:               elem_size = 0; break;
            }

            if (elem_size > 0) {
                if (fseek64(fp, (int64_t)(count * elem_size), SEEK_CUR) != 0)
                    fatal("fseek failed while skipping fixed-size array");
            } else {
                /* String or nested array: must visit each element */
                for (uint64_t i = 0; i < count; i++)
                    skip_val(fp, elem_type);
            }
            return;
        }

        default:
            fprintf(stderr, "gguf error: unknown metadata value type %u\n", vtype);
            fatal("unknown metadata value type");
    }
}

/* ------------------------------------------------------------------ */
/* Metadata KV pass                                                     */
/* Skips all KV pairs; captures general.alignment if present.         */
/* ------------------------------------------------------------------ */

static uint32_t parse_metadata(FILE *fp, uint64_t n_kv) {
    uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;

    for (uint64_t i = 0; i < n_kv; i++) {
        char    *key = read_str(fp);
        uint32_t vt  = r_u32(fp);

        int want = (strcmp(key, "general.alignment") == 0);
        free(key);

        if (want && vt == GGUF_VAL_UINT32) {
            alignment = r_u32(fp);
        } else {
            skip_val(fp, vt);
        }
    }

    return alignment;
}

/* ------------------------------------------------------------------ */
/* Tensor info pass                                                     */
/* ------------------------------------------------------------------ */

static void parse_tensors(FILE *fp, uint64_t n, gguf_tensor_t *tensors) {
    for (uint64_t i = 0; i < n; i++) {
        gguf_tensor_t *t = &tensors[i];

        t->name   = read_str(fp);
        t->n_dims = r_u32(fp);

        if (t->n_dims > GGUF_MAX_DIMS) {
            fprintf(stderr, "gguf error: tensor '%s' has %u dims (max %d)\n",
                    t->name, t->n_dims, GGUF_MAX_DIMS);
            fatal("tensor dimension count exceeds GGUF_MAX_DIMS");
        }

        t->n_elems = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->dims[d]  = r_u64(fp);  /* GGUF v2+ uses uint64 per dim */
            t->n_elems *= t->dims[d];
        }
        for (uint32_t d = t->n_dims; d < GGUF_MAX_DIMS; d++)
            t->dims[d] = 0;

        t->type   = r_u32(fp);
        t->offset = r_u64(fp);  /* relative to start of data section */
        t->size   = ggml_tensor_nbytes(t->n_elems, t->type);
    }
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

gguf_ctx_t *gguf_open(const char *path) {
    clock_t t0 = clock();

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "gguf: cannot open '%s'\n", path);
        return NULL;
    }

    /* Large stdio buffer reduces seek overhead when skipping token vocab arrays */
    setvbuf(fp, NULL, _IOFBF, 1 << 20);  /* 1 MiB */

    /* --- GGUF header --- */
    uint32_t magic = r_u32(fp);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "gguf: bad magic 0x%08x (expected 0x%08x)\n",
                magic, GGUF_MAGIC);
        fclose(fp);
        return NULL;
    }

    uint32_t version = r_u32(fp);
    if (version < 2 || version > 3) {
        fprintf(stderr, "gguf: unsupported version %u (supported: 2, 3)\n", version);
        fclose(fp);
        return NULL;
    }

    uint64_t n_tensors = r_u64(fp);
    uint64_t n_kv      = r_u64(fp);

    fprintf(stderr, "gguf: version=%u  tensors=%llu  kv_pairs=%llu\n",
            version,
            (unsigned long long)n_tensors,
            (unsigned long long)n_kv);

    /* --- Metadata KV pairs --- */
    uint32_t alignment = parse_metadata(fp, n_kv);
    fprintf(stderr, "gguf: alignment=%u bytes\n", alignment);

    /* --- Tensor info records --- */
    gguf_tensor_t *tensors = (gguf_tensor_t *)calloc(n_tensors, sizeof(gguf_tensor_t));
    if (!tensors) {
        fclose(fp);
        fatal("out of memory for tensor array");
    }

    parse_tensors(fp, n_tensors, tensors);

    /* Data section starts at the next alignment boundary after tensor infos */
    int64_t after_infos = ftell64(fp);
    if (after_infos < 0) {
        fclose(fp);
        fatal("ftell64 failed");
    }
    uint64_t data_off = ((uint64_t)after_infos + alignment - 1)
                        / alignment * alignment;

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    fprintf(stderr, "gguf: header parsed in %.3f s  —  data section at 0x%llx\n",
            elapsed, (unsigned long long)data_off);

    /* --- Build context --- */
    gguf_ctx_t *ctx = (gguf_ctx_t *)malloc(sizeof(gguf_ctx_t));
    if (!ctx) {
        fclose(fp);
        fatal("out of memory for context");
    }

    ctx->fp        = fp;
    ctx->version   = version;
    ctx->n_kv      = n_kv;
    ctx->n_tensors = n_tensors;
    ctx->alignment = alignment;
    ctx->data_off  = data_off;
    ctx->tensors   = tensors;

    return ctx;
}

void gguf_close(gguf_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->fp) fclose(ctx->fp);
    if (ctx->tensors) {
        for (uint64_t i = 0; i < ctx->n_tensors; i++)
            free(ctx->tensors[i].name);
        free(ctx->tensors);
    }
    free(ctx);
}

uint32_t gguf_version(const gguf_ctx_t *ctx)    { return ctx->version;   }
uint64_t gguf_num_tensors(const gguf_ctx_t *ctx) { return ctx->n_tensors; }
uint64_t gguf_data_offset(const gguf_ctx_t *ctx) { return ctx->data_off;  }

const gguf_tensor_t *gguf_tensor_by_idx(const gguf_ctx_t *ctx, uint64_t idx) {
    if (idx >= ctx->n_tensors) return NULL;
    return &ctx->tensors[idx];
}

const gguf_tensor_t *gguf_find_tensor(const gguf_ctx_t *ctx, const char *name) {
    for (uint64_t i = 0; i < ctx->n_tensors; i++)
        if (strcmp(ctx->tensors[i].name, name) == 0)
            return &ctx->tensors[i];
    return NULL;
}
