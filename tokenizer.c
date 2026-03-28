/* tokenizer.c — BPE tokenizer for GPT-2 byte-level BPE (Qwen3.5).
 *
 * Reads tokenizer.ggml.tokens and tokenizer.ggml.merges from the GGUF
 * metadata section.  Implements the standard GPT-2 bytes_to_unicode
 * mapping so that every byte value has a unique printable representation.
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
#include <limits.h>
#include <ctype.h>

#include "tokenizer.h"

/* ------------------------------------------------------------------ */
/* Portable 64-bit fseek / ftell                                       */
/* ------------------------------------------------------------------ */

#ifdef _WIN32
#  define fseek64(f,o,w)  _fseeki64((f), (__int64)(o), (w))
#  define ftell64(f)      ((int64_t)_ftelli64(f))
#else
#  include <sys/types.h>
#  define fseek64(f,o,w)  fseeko((f), (off_t)(o), (w))
#  define ftell64(f)      ((int64_t)ftello(f))
#endif

/* ------------------------------------------------------------------ */
/* GGUF read helpers                                                    */
/* ------------------------------------------------------------------ */

static void tok_fatal(const char *msg) {
    fprintf(stderr, "tokenizer: %s\n", msg);
    exit(1);
}

static void xread(FILE *fp, void *buf, size_t n) {
    if (fread(buf, 1, n, fp) != n) tok_fatal("unexpected end of file");
}

static uint8_t  r_u8 (FILE *fp) { uint8_t  v; xread(fp,&v,1); return v; }
static int8_t   r_i8 (FILE *fp) { int8_t   v; xread(fp,&v,1); return v; }
static uint16_t r_u16(FILE *fp) { uint16_t v; xread(fp,&v,2); return v; }
static int16_t  r_i16(FILE *fp) { int16_t  v; xread(fp,&v,2); return v; }
static uint32_t r_u32(FILE *fp) { uint32_t v; xread(fp,&v,4); return v; }
static int32_t  r_i32(FILE *fp) { int32_t  v; xread(fp,&v,4); return v; }
static uint64_t r_u64(FILE *fp) { uint64_t v; xread(fp,&v,8); return v; }
static int64_t  r_i64(FILE *fp) { int64_t  v; xread(fp,&v,8); return v; }
static float    r_f32(FILE *fp) { float    v; xread(fp,&v,4); return v; }
static double   r_f64(FILE *fp) { double   v; xread(fp,&v,8); return v; }

static char *read_str(FILE *fp) {
    uint64_t len = r_u64(fp);
    char *s = (char *)malloc(len + 1);
    if (!s) tok_fatal("out of memory reading string");
    xread(fp, s, (size_t)len);
    s[len] = '\0';
    return s;
}

static void skip_str(FILE *fp) {
    uint64_t len = r_u64(fp);
    if (len > 0 && fseek64(fp, (int64_t)len, SEEK_CUR) != 0)
        tok_fatal("fseek failed skipping string");
}

/* Skip one GGUF value (recursive for ARRAY). */
static void skip_val(FILE *fp, uint32_t vt) {
    switch (vt) {
        case  0: case  1: case  7: (void)r_u8 (fp); return;
        case  2: case  3:          (void)r_u16(fp); return;
        case  4: case  5: case  6: (void)r_u32(fp); return;
        case 10: case 11: case 12: (void)r_u64(fp); return;
        case  8: skip_str(fp); return;
        case  9: {
            uint32_t et  = r_u32(fp);
            uint64_t cnt = r_u64(fp);
            size_t   esz = 0;
            switch (et) {
                case  0: case  1: case  7: esz=1; break;
                case  2: case  3:          esz=2; break;
                case  4: case  5: case  6: esz=4; break;
                case 10: case 11: case 12: esz=8; break;
                default: esz=0; break;
            }
            if (esz > 0) {
                fseek64(fp, (int64_t)(cnt * esz), SEEK_CUR);
            } else {
                for (uint64_t i = 0; i < cnt; i++) skip_val(fp, et);
            }
            return;
        }
        default:
            fprintf(stderr, "tokenizer: unknown vtype %u\n", vt);
            tok_fatal("unknown metadata value type");
    }
}

/* Read an array of STRING elements (caller must have already read
 * the ARRAY header fields et and cnt). */
static char **read_str_array(FILE *fp, uint64_t cnt) {
    char **arr = (char **)malloc(cnt * sizeof(char *));
    if (!arr) tok_fatal("OOM allocating string array");
    for (uint64_t i = 0; i < cnt; i++)
        arr[i] = read_str(fp);
    return arr;
}

/* Skip cnt elements of a fixed-size type et (already consumed from file). */
static void skip_fixed_array(FILE *fp, uint32_t et, uint64_t cnt) {
    size_t esz = 0;
    switch (et) {
        case  0: case  1: case  7: esz=1; break;
        case  2: case  3:          esz=2; break;
        case  4: case  5: case  6: esz=4; break;
        case 10: case 11: case 12: esz=8; break;
        default: esz=0; break;
    }
    if (esz > 0) {
        fseek64(fp, (int64_t)(cnt * esz), SEEK_CUR);
    } else {
        for (uint64_t i = 0; i < cnt; i++) skip_val(fp, et);
    }
}

/* ------------------------------------------------------------------ */
/* GPT-2 bytes_to_unicode mapping                                      */
/*                                                                      */
/* Bytes 33-126 (! to ~), 161-172 (¡ to ¬), 174-255 (® to ÿ)        */
/* map to themselves.                                                  */
/*                                                                      */
/* The remaining 68 bytes (0-32, 127-160, 173) map in sorted order    */
/* to U+0100, U+0101, ..., U+0143.  Space (0x20) → U+0120 (Ġ).      */
/* ------------------------------------------------------------------ */

static void build_missing(uint8_t *missing) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int direct = (b >= 33 && b <= 126) ||
                     (b >= 161 && b <= 172) ||
                     (b >= 174 && b <= 255);
        if (!direct) missing[n++] = (uint8_t)b;
    }
    /* n == 68 */
}

static int is_direct_byte(int b) {
    return (b >= 33 && b <= 126) ||
           (b >= 161 && b <= 172) ||
           (b >= 174 && b <= 255);
}

/* Map a raw byte to its GPT-2 Unicode codepoint. */
static uint32_t byte_to_cp(uint8_t b, const uint8_t *missing) {
    if (is_direct_byte((int)b)) return (uint32_t)b;
    for (int i = 0; i < 68; i++)
        if (missing[i] == b) return 0x100u + (uint32_t)i;
    return (uint32_t)b; /* unreachable */
}

/* Map a GPT-2 Unicode codepoint back to a raw byte.
 * Returns -1 if the codepoint is not a valid byte encoding. */
static int cp_to_byte(uint32_t cp, const uint8_t *missing) {
    if (is_direct_byte((int)cp)) return (int)cp;
    if (cp >= 0x100u && cp < 0x100u + 68u)
        return (int)missing[cp - 0x100u];
    return -1;
}

/* Encode a Unicode codepoint as UTF-8 into buf (no null terminator).
 * Returns number of bytes written (1-3 for codepoints up to U+FFFF). */
static int cp_to_utf8(uint32_t cp, char *buf) {
    if (cp < 0x80u) {
        buf[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800u) {
        buf[0] = (char)(0xC0u | (cp >> 6));
        buf[1] = (char)(0x80u | (cp & 0x3Fu));
        return 2;
    }
    buf[0] = (char)(0xE0u | (cp >> 12));
    buf[1] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
    buf[2] = (char)(0x80u | (cp & 0x3Fu));
    return 3;
}

/* Decode one UTF-8 codepoint from *s; advance *s past the bytes read. */
static uint32_t utf8_to_cp(const char **s) {
    const unsigned char *p = (const unsigned char *)*s;
    uint32_t cp;
    if (p[0] < 0x80u) {
        cp = p[0]; *s += 1;
    } else if ((p[0] & 0xE0u) == 0xC0u) {
        cp = ((uint32_t)(p[0] & 0x1Fu) << 6) | (uint32_t)(p[1] & 0x3Fu);
        *s += 2;
    } else if ((p[0] & 0xF0u) == 0xE0u) {
        cp = ((uint32_t)(p[0] & 0x0Fu) << 12) |
             ((uint32_t)(p[1] & 0x3Fu) <<  6) |
              (uint32_t)(p[2] & 0x3Fu);
        *s += 3;
    } else {
        cp = ((uint32_t)(p[0] & 0x07u) << 18) |
             ((uint32_t)(p[1] & 0x3Fu) << 12) |
             ((uint32_t)(p[2] & 0x3Fu) <<  6) |
              (uint32_t)(p[3] & 0x3Fu);
        *s += 4;
    }
    return cp;
}

/* ------------------------------------------------------------------ */
/* Hash map: string key → int32 value                                  */
/* Open-addressing with FNV-1a hash and linear probing.               */
/* ------------------------------------------------------------------ */

#define HM_EMPTY INT32_MIN

typedef struct {
    char    **keys;
    int32_t  *vals;
    uint32_t  cap;
    uint32_t  used;
} hmap_t;

static hmap_t *hm_new(uint32_t cap) {
    hmap_t *m = (hmap_t *)malloc(sizeof(hmap_t));
    if (!m) tok_fatal("OOM allocating hash map");
    m->cap  = cap;
    m->used = 0;
    m->keys = (char **)calloc(cap, sizeof(char *));
    m->vals = (int32_t *)malloc(cap * sizeof(int32_t));
    if (!m->keys || !m->vals) tok_fatal("OOM allocating hash map arrays");
    for (uint32_t i = 0; i < cap; i++) m->vals[i] = HM_EMPTY;
    return m;
}

static uint32_t hm_hash(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) h = (h ^ (uint8_t)*s) * 16777619u;
    return h;
}

static void hm_put(hmap_t *m, const char *key, int32_t val) {
    uint32_t h = hm_hash(key) % m->cap;
    while (m->keys[h]) {
        if (strcmp(m->keys[h], key) == 0) { m->vals[h] = val; return; }
        h = (h + 1) % m->cap;
    }
    m->keys[h] = strdup(key);
    m->vals[h] = val;
    m->used++;
}

static int32_t hm_get(const hmap_t *m, const char *key) {
    uint32_t h = hm_hash(key) % m->cap;
    while (m->keys[h]) {
        if (strcmp(m->keys[h], key) == 0) return m->vals[h];
        h = (h + 1) % m->cap;
    }
    return HM_EMPTY;
}

static void hm_free(hmap_t *m) {
    for (uint32_t i = 0; i < m->cap; i++)
        if (m->keys[i]) free(m->keys[i]);
    free(m->keys);
    free(m->vals);
    free(m);
}

/* ------------------------------------------------------------------ */
/* Tokenizer struct                                                     */
/* ------------------------------------------------------------------ */

struct tokenizer_t {
    char    **id_to_tok;    /* id → token string (GPT-2 encoded), vocab_size entries */
    int32_t   vocab_size;
    int32_t   bos_id;
    int32_t   eos_id;
    char     *pre_name;
    int32_t  *token_types;
    hmap_t   *tok_to_id;   /* token string → id */
    hmap_t   *merge_rank;  /* "A B" → merge rank (array index) */
    int32_t   num_merges;
    char    **special_texts;
    int32_t  *special_ids;
    int32_t   num_specials;
    int32_t   max_special_len;
    uint8_t   missing[68]; /* GPT-2 extra byte table */
};

static int looks_like_special_token(const char *s);
static int match_special_token(const tokenizer_t *tok, const uint8_t *data, int n,
                               int32_t *out_id);

enum {
    TOK_TYPE_UNDEFINED    = 0,
    TOK_TYPE_NORMAL       = 1,
    TOK_TYPE_UNKNOWN      = 2,
    TOK_TYPE_CONTROL      = 3,
    TOK_TYPE_USER_DEFINED = 4,
    TOK_TYPE_UNUSED       = 5,
    TOK_TYPE_BYTE         = 6,
};

static int32_t *read_i32_array(FILE *fp, uint32_t et, uint64_t cnt) {
    int32_t *arr = (int32_t *)malloc((size_t)cnt * sizeof(int32_t));
    if (!arr) tok_fatal("OOM allocating int32 array");
    if (et == 4) {
        for (uint64_t i = 0; i < cnt; i++) arr[i] = (int32_t)r_u32(fp);
    } else if (et == 5) {
        for (uint64_t i = 0; i < cnt; i++) arr[i] = r_i32(fp);
    } else {
        free(arr);
        return NULL;
    }
    return arr;
}

/* ------------------------------------------------------------------ */
/* Load tokenizer from GGUF file                                       */
/* ------------------------------------------------------------------ */

tokenizer_t *tokenizer_load(const char *model_path) {
    FILE *fp = fopen(model_path, "rb");
    if (!fp) {
        fprintf(stderr, "tokenizer: cannot open '%s'\n", model_path);
        return NULL;
    }

    /* --- GGUF header --- */
    uint32_t magic = r_u32(fp);
    if (magic != 0x46554747u) {
        fprintf(stderr, "tokenizer: not a GGUF file (magic=%08X)\n", magic);
        fclose(fp);
        return NULL;
    }
    uint32_t version   = r_u32(fp);
    uint64_t n_tensors = r_u64(fp);
    uint64_t n_kv      = r_u64(fp);
    (void)version; (void)n_tensors;

    tokenizer_t *tok = (tokenizer_t *)calloc(1, sizeof(tokenizer_t));
    if (!tok) tok_fatal("OOM allocating tokenizer");
    tok->bos_id = -1;
    tok->eos_id = -1;
    build_missing(tok->missing);

    char   **tokens   = NULL;
    int32_t  n_tokens = 0;
    char   **merges   = NULL;
    int32_t  n_merges = 0;
    int32_t *token_types = NULL;

    /* --- Scan metadata KV pairs --- */
    for (uint64_t ki = 0; ki < n_kv; ki++) {
        char    *key   = read_str(fp);
        uint32_t vtype = r_u32(fp);
        int      handled = 0;

        /* tokenizer.ggml.tokens — ARRAY(STRING) */
        if (strcmp(key, "tokenizer.ggml.tokens") == 0 && vtype == 9) {
            uint32_t et  = r_u32(fp);
            uint64_t cnt = r_u64(fp);
            if (et == 8) {
                n_tokens = (int32_t)cnt;
                tokens   = read_str_array(fp, cnt);
            } else {
                skip_fixed_array(fp, et, cnt);
            }
            handled = 1;

        /* tokenizer.ggml.merges — ARRAY(STRING) */
        } else if (strcmp(key, "tokenizer.ggml.merges") == 0 && vtype == 9) {
            uint32_t et  = r_u32(fp);
            uint64_t cnt = r_u64(fp);
            if (et == 8) {
                n_merges = (int32_t)cnt;
                merges   = read_str_array(fp, cnt);
            } else {
                skip_fixed_array(fp, et, cnt);
            }
            handled = 1;

        /* tokenizer.ggml.token_type — ARRAY(INT32) */
        } else if (strcmp(key, "tokenizer.ggml.token_type") == 0 && vtype == 9) {
            uint32_t et  = r_u32(fp);
            uint64_t cnt = r_u64(fp);
            token_types = read_i32_array(fp, et, cnt);
            if (!token_types) {
                skip_fixed_array(fp, et, cnt);
            }
            handled = 1;

        /* tokenizer.ggml.scores — ARRAY(FLOAT32), skip */
        } else if (strcmp(key, "tokenizer.ggml.scores") == 0 && vtype == 9) {
            uint32_t et  = r_u32(fp);
            uint64_t cnt = r_u64(fp);
            skip_fixed_array(fp, et, cnt);
            handled = 1;

        /* BOS token ID */
        } else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) {
            if      (vtype == 4) tok->bos_id = (int32_t)r_u32(fp);
            else if (vtype == 5) tok->bos_id = r_i32(fp);
            else                 skip_val(fp, vtype);
            handled = 1;

        /* EOS token ID */
        } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
            if      (vtype == 4) tok->eos_id = (int32_t)r_u32(fp);
            else if (vtype == 5) tok->eos_id = r_i32(fp);
            else                 skip_val(fp, vtype);
            handled = 1;
        } else if (strcmp(key, "tokenizer.ggml.pre") == 0) {
            if (vtype == 8) tok->pre_name = read_str(fp);
            else            skip_val(fp, vtype);
            handled = 1;
        }

        if (!handled) skip_val(fp, vtype);
        free(key);
    }

    fclose(fp);

    if (!tokens || n_tokens == 0) {
        fprintf(stderr, "tokenizer: tokenizer.ggml.tokens not found in %s\n", model_path);
        free(tok);
        return NULL;
    }

    /* --- Build vocab tables --- */
    tok->vocab_size = n_tokens;
    tok->id_to_tok  = tokens;
    tok->token_types = token_types;
    tok->tok_to_id  = hm_new((uint32_t)n_tokens * 2 + 3);
    for (int32_t i = 0; i < n_tokens; i++)
        hm_put(tok->tok_to_id, tokens[i], i);

    tok->special_texts = (char **)malloc((size_t)n_tokens * sizeof(char *));
    tok->special_ids   = (int32_t *)malloc((size_t)n_tokens * sizeof(int32_t));
    if (!tok->special_texts || !tok->special_ids) tok_fatal("OOM allocating special token tables");
    tok->num_specials = 0;
    tok->max_special_len = 0;
    for (int32_t i = 0; i < n_tokens; i++) {
        int is_special = 0;
        if (tok->token_types) {
            const int32_t tt = tok->token_types[i];
            is_special = (tt == TOK_TYPE_CONTROL ||
                          tt == TOK_TYPE_USER_DEFINED ||
                          tt == TOK_TYPE_UNKNOWN);
        } else {
            is_special = looks_like_special_token(tokens[i]);
        }
        if (is_special) {
            tok->special_texts[tok->num_specials] = tokens[i];
            tok->special_ids[tok->num_specials] = i;
            tok->num_specials++;
            {
                int len = (int)strlen(tokens[i]);
                if (len > tok->max_special_len) tok->max_special_len = len;
            }
        }
    }

    /* --- Build merge rank table --- */
    tok->num_merges = n_merges;
    if (merges && n_merges > 0) {
        tok->merge_rank = hm_new((uint32_t)n_merges * 2 + 3);
        for (int32_t i = 0; i < n_merges; i++) {
            hm_put(tok->merge_rank, merges[i], i);
            free(merges[i]);
        }
        free(merges);
    }

    fprintf(stderr, "tokenizer: vocab=%d merges=%d bos=%d eos=%d specials=%d\n",
            tok->vocab_size, tok->num_merges, tok->bos_id, tok->eos_id, tok->num_specials);
    return tok;
}

void tokenizer_free(tokenizer_t *tok) {
    if (!tok) return;
    for (int32_t i = 0; i < tok->vocab_size; i++)
        free(tok->id_to_tok[i]);
    free(tok->id_to_tok);
    if (tok->tok_to_id)   hm_free(tok->tok_to_id);
    if (tok->merge_rank)  hm_free(tok->merge_rank);
    free(tok->special_texts);
    free(tok->special_ids);
    free(tok->token_types);
    if (tok->pre_name)    free(tok->pre_name);
    free(tok);
}

int32_t tokenizer_bos(const tokenizer_t *tok)        { return tok->bos_id; }
int32_t tokenizer_eos(const tokenizer_t *tok)        { return tok->eos_id; }
int32_t tokenizer_vocab_size(const tokenizer_t *tok) { return tok->vocab_size; }
const char *tokenizer_pre(const tokenizer_t *tok)    { return tok->pre_name ? tok->pre_name : ""; }

/* ------------------------------------------------------------------ */
/* BPE encode                                                           */
/* ------------------------------------------------------------------ */

/* Convert n raw bytes to an array of GPT-2-encoded single-char strings.
 * Each output string is heap-allocated (1-3 UTF-8 bytes + null term).
 * Caller owns the returned array and its strings. */
static char **bytes_to_pieces(const uint8_t *data, int n, const uint8_t *missing) {
    char **out = (char **)malloc(n * sizeof(char *));
    if (!out) tok_fatal("OOM in bytes_to_pieces");
    for (int i = 0; i < n; i++) {
        uint32_t cp = byte_to_cp(data[i], missing);
        char buf[4];
        int  len = cp_to_utf8(cp, buf);
        buf[len] = '\0';
        out[i] = strdup(buf);
    }
    return out;
}

/* Apply BPE merges to the symbol array s[0..n-1] in-place.
 * After this function:
 *   - Every string in s[0..n-1] has been either freed (consumed in a merge)
 *     or replaced with a merged allocation.
 *   - *n_sym is updated to the final symbol count.
 * Writes the resulting token IDs to out[]. Returns count written. */
static int bpe_merge_and_lookup(const tokenizer_t *tok,
                                char **s, int *n_sym,
                                int32_t *out, int max_out) {
    int n = *n_sym;

    /* If no merge table, each piece is a separate token. */
    if (!tok->merge_rank) {
        int written = 0;
        for (int i = 0; i < n; i++) {
            if (written < max_out) {
                int32_t id = hm_get(tok->tok_to_id, s[i]);
                out[written++] = (id != HM_EMPTY) ? id : 0;
            }
            free(s[i]);
        }
        *n_sym = written;
        return written;
    }

    /* Repeatedly find and apply the lowest-rank merge. O(n^2) per word —
     * fast enough for inference-time per-token text (short strings). */
    /* Pair key buffer: max piece len is bounded by vocab, use 1024 bytes. */
    char pair_buf[1024];

    while (n > 1) {
        int    best_rank = INT_MAX;
        int    best_i    = -1;

        for (int i = 0; i < n - 1; i++) {
            size_t l1 = strlen(s[i]);
            size_t l2 = strlen(s[i + 1]);
            if (l1 + 1 + l2 + 1 > sizeof(pair_buf)) continue;
            memcpy(pair_buf, s[i], l1);
            pair_buf[l1] = ' ';
            memcpy(pair_buf + l1 + 1, s[i + 1], l2);
            pair_buf[l1 + 1 + l2] = '\0';

            int32_t rank = hm_get(tok->merge_rank, pair_buf);
            if (rank != HM_EMPTY && rank < best_rank) {
                best_rank = rank;
                best_i    = i;
            }
        }

        if (best_i < 0) break; /* no more applicable merges */

        /* Merge s[best_i] and s[best_i+1] into a new heap string. */
        size_t l1 = strlen(s[best_i]);
        size_t l2 = strlen(s[best_i + 1]);
        char  *merged = (char *)malloc(l1 + l2 + 1);
        if (!merged) tok_fatal("OOM in BPE merge");
        memcpy(merged, s[best_i], l1);
        memcpy(merged + l1, s[best_i + 1], l2);
        merged[l1 + l2] = '\0';

        free(s[best_i]);
        free(s[best_i + 1]);
        s[best_i] = merged;

        /* Shift elements left to fill the gap at best_i+1. */
        for (int i = best_i + 1; i < n - 1; i++)
            s[i] = s[i + 1];
        n--;
    }

    /* Look up each surviving symbol in the vocabulary. */
    int written = 0;
    for (int i = 0; i < n; i++) {
        if (written < max_out) {
            int32_t id = hm_get(tok->tok_to_id, s[i]);
            if (id == HM_EMPTY)
                fprintf(stderr, "tokenizer: unknown piece '%s'\n", s[i]);
            out[written++] = (id != HM_EMPTY) ? id : 0;
        }
        free(s[i]);
    }

    *n_sym = written;
    return written;
}

static int is_ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f';
}

static int is_ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static int is_ascii_digit(uint8_t c) {
    return (c >= '0' && c <= '9');
}

static int utf8_peek_cp(const uint8_t *data, int n, uint32_t *out_cp) {
    if (n <= 0) return 0;
    if (data[0] < 0x80u) {
        *out_cp = data[0];
        return 1;
    }
    if ((data[0] & 0xE0u) == 0xC0u && n >= 2) {
        *out_cp = ((uint32_t)(data[0] & 0x1Fu) << 6) |
                  (uint32_t)(data[1] & 0x3Fu);
        return 2;
    }
    if ((data[0] & 0xF0u) == 0xE0u && n >= 3) {
        *out_cp = ((uint32_t)(data[0] & 0x0Fu) << 12) |
                  ((uint32_t)(data[1] & 0x3Fu) << 6) |
                  (uint32_t)(data[2] & 0x3Fu);
        return 3;
    }
    if ((data[0] & 0xF8u) == 0xF0u && n >= 4) {
        *out_cp = ((uint32_t)(data[0] & 0x07u) << 18) |
                  ((uint32_t)(data[1] & 0x3Fu) << 12) |
                  ((uint32_t)(data[2] & 0x3Fu) << 6) |
                  (uint32_t)(data[3] & 0x3Fu);
        return 4;
    }
    *out_cp = data[0];
    return 1;
}

static int cp_is_newline(uint32_t cp) {
    return cp == '\r' || cp == '\n';
}

static int cp_is_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\v' || cp == '\f' || cp_is_newline(cp);
}

static int cp_is_digit(uint32_t cp) {
    return cp <= 0x7Fu && is_ascii_digit((uint8_t)cp);
}

static int cp_is_letter_mark(uint32_t cp) {
    if (cp <= 0x7Fu) return is_ascii_alpha((uint8_t)cp);
    /* Qwen35 treats non-ASCII letters/marks as part of word spans.
     * We conservatively classify non-ASCII non-space non-digit codepoints
     * as letter-ish here so multilingual text stays grouped instead of
     * degenerating into byte-wise spans. */
    return !cp_is_space(cp) && !cp_is_digit(cp);
}

static int cp_is_qwen35_punct(uint32_t cp) {
    return !cp_is_space(cp) && !cp_is_letter_mark(cp) && !cp_is_digit(cp);
}

static int looks_like_special_token(const char *s) {
    size_t n = strlen(s);
    if (n < 3) return 0;
    if (s[0] != '<' || s[n - 1] != '>') return 0;
    for (size_t i = 1; i + 1 < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x20 || c > 0x7E) return 0;
    }
    return 1;
}

static int match_special_token(const tokenizer_t *tok, const uint8_t *data, int n,
                               int32_t *out_id) {
    int best_len = 0;
    int32_t best_id = -1;
    for (int i = 0; i < tok->num_specials; i++) {
        const char *sp = tok->special_texts[i];
        int len = (int)strlen(sp);
        if (len <= best_len || len > n) continue;
        if (memcmp(data, sp, (size_t)len) == 0) {
            best_len = len;
            best_id = tok->special_ids[i];
        }
    }
    if (best_len > 0 && out_id) *out_id = best_id;
    return best_len;
}

static int find_next_special_token(const tokenizer_t *tok, const uint8_t *data, int n,
                                   int *out_off, int32_t *out_id, int *out_len) {
    int best_off = -1;
    int best_len = 0;
    int32_t best_id = -1;
    for (int off = 0; off < n; off++) {
        int32_t id = -1;
        int len = match_special_token(tok, data + off, n - off, &id);
        if (len <= 0) continue;
        if (best_off < 0 || off < best_off || (off == best_off && len > best_len)) {
            best_off = off;
            best_len = len;
            best_id = id;
            if (best_off == 0 && best_len == tok->max_special_len) break;
        }
    }
    if (best_off < 0) return 0;
    if (out_off) *out_off = best_off;
    if (out_id)  *out_id  = best_id;
    if (out_len) *out_len = best_len;
    return 1;
}

static int match_ascii_contraction_ci(const uint8_t *data, int n, int i) {
    static const char *suffixes[] = { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };
    if (i >= n || data[i] != '\'') return 0;
    for (int k = 0; k < (int)(sizeof(suffixes) / sizeof(suffixes[0])); k++) {
        int len = (int)strlen(suffixes[k]);
        int ok = 1;
        if (i + len > n) continue;
        for (int j = 0; j < len; j++) {
            unsigned char a = data[i + j];
            unsigned char b = (unsigned char)suffixes[k][j];
            if (tolower(a) != tolower(b)) {
                ok = 0;
                break;
            }
        }
        if (ok) return len;
    }
    return 0;
}

static int encode_piece_bytes(const tokenizer_t *tok, const uint8_t *data, int n,
                              int32_t *out, int max_out) {
    char **pieces;
    int n_sym;
    int total;
    if (n <= 0 || max_out <= 0) return 0;
    pieces = bytes_to_pieces(data, n, tok->missing);
    n_sym  = n;
    total  = bpe_merge_and_lookup(tok, pieces, &n_sym, out, max_out);
    free(pieces);
    return total;
}

static int encode_qwen35_raw(const tokenizer_t *tok, const uint8_t *data, int n,
                             int32_t *out, int max_out) {
    int total = 0;
    int i = 0;

    while (i < n && total < max_out) {
        const int start = i;
        uint32_t cp0 = 0, cp1 = 0;
        int len0 = utf8_peek_cp(data + i, n - i, &cp0);
        int len1 = 0;
        int c_len = 0;

        if ((c_len = match_ascii_contraction_ci(data, n, i)) > 0) {
            i += c_len;
        } else if (cp_is_letter_mark(cp0)) {
            i += len0;
            while (i < n) {
                int len = utf8_peek_cp(data + i, n - i, &cp1);
                if (!cp_is_letter_mark(cp1)) break;
                i += len;
            }
        } else if (cp_is_digit(cp0)) {
            i += len0; /* Qwen35 uses \p{N}, one digit/codepoint at a time */
        } else if (cp_is_newline(cp0)) {
            i += len0;
            while (i < n) {
                int len = utf8_peek_cp(data + i, n - i, &cp1);
                if (!cp_is_newline(cp1)) break;
                i += len;
            }
        } else if (cp_is_space(cp0)) {
            int j = i;
            while (j < n) {
                int len = utf8_peek_cp(data + j, n - j, &cp1);
                if (!cp_is_space(cp1) || cp_is_newline(cp1)) break;
                j += len;
            }
            len1 = utf8_peek_cp(data + j, n - j, &cp1);
            if (j < n && cp_is_newline(cp1)) {
                i = j + len1;
                while (i < n) {
                    int len = utf8_peek_cp(data + i, n - i, &cp1);
                    if (!cp_is_newline(cp1)) break;
                    i += len;
                }
            } else if (cp0 == ' ' && j == i + 1 && j < n && cp_is_letter_mark(cp1)) {
                i = j + len1;
                while (i < n) {
                    int len = utf8_peek_cp(data + i, n - i, &cp1);
                    if (!cp_is_letter_mark(cp1)) break;
                    i += len;
                }
            } else if (cp0 == ' ' && j == i + 1 && j < n && cp_is_qwen35_punct(cp1)) {
                i = j + len1;
                while (i < n) {
                    int len = utf8_peek_cp(data + i, n - i, &cp1);
                    if (!cp_is_qwen35_punct(cp1)) break;
                    i += len;
                }
                while (i < n) {
                    int len = utf8_peek_cp(data + i, n - i, &cp1);
                    if (!cp_is_newline(cp1)) break;
                    i += len;
                }
            } else {
                i = j;
            }
        } else {
            len1 = utf8_peek_cp(data + i + len0, n - (i + len0), &cp1);
            if (!cp_is_newline(cp0) && !cp_is_letter_mark(cp0) && !cp_is_digit(cp0) &&
                len1 > 0 && cp_is_letter_mark(cp1)) {
                i += len0 + len1;
                while (i < n) {
                    int len = utf8_peek_cp(data + i, n - i, &cp1);
                    if (!cp_is_letter_mark(cp1)) break;
                    i += len;
                }
            } else {
                i += len0;
                while (i < n) {
                    int len = utf8_peek_cp(data + i, n - i, &cp1);
                    if (!cp_is_qwen35_punct(cp1)) break;
                    i += len;
                }
                while (i < n) {
                    int len = utf8_peek_cp(data + i, n - i, &cp1);
                    if (!cp_is_newline(cp1)) break;
                    i += len;
                }
            }
        }

        if (i > start) {
            total += encode_piece_bytes(tok, data + start, i - start, out + total, max_out - total);
        } else {
            i += len0 > 0 ? len0 : 1;
        }
    }

    return total;
}

static int encode_basic_raw(const tokenizer_t *tok, const uint8_t *data, int n,
                            int32_t *out, int max_out) {
    int i = 0;
    int total = 0;

    while (i < n && total < max_out) {
        int start = i;
        int len = 0;
        int c_len;

        if (is_ascii_space(data[i])) {
            if (i + 1 < n && !is_ascii_space(data[i + 1])) {
                i++;
                c_len = match_ascii_contraction_ci(data, n, i);
                if (c_len > 0) {
                    i += c_len;
                } else if (is_ascii_alpha(data[i])) {
                    while (i < n && is_ascii_alpha(data[i])) i++;
                } else if (is_ascii_digit(data[i])) {
                    while (i < n && is_ascii_digit(data[i])) i++;
                } else {
                    while (i < n && !is_ascii_space(data[i]) &&
                           !is_ascii_alpha(data[i]) && !is_ascii_digit(data[i]) &&
                           match_ascii_contraction_ci(data, n, i) == 0) {
                        i++;
                    }
                }
            } else {
                while (i < n && is_ascii_space(data[i])) i++;
            }
        } else {
            c_len = match_ascii_contraction_ci(data, n, i);
            if (c_len > 0) {
                i += c_len;
            } else if (is_ascii_alpha(data[i])) {
                while (i < n && is_ascii_alpha(data[i])) i++;
            } else if (is_ascii_digit(data[i])) {
                while (i < n && is_ascii_digit(data[i])) i++;
            } else {
                while (i < n && !is_ascii_space(data[i]) &&
                       !is_ascii_alpha(data[i]) && !is_ascii_digit(data[i]) &&
                       match_ascii_contraction_ci(data, n, i) == 0) {
                    i++;
                }
            }
        }

        len = i - start;
        if (len > 0) {
            total += encode_piece_bytes(tok, data + start, len, out + total, max_out - total);
        } else {
            i++;
        }
    }

    return total;
}

int tokenizer_encode(const tokenizer_t *tok, const char *text,
                     int32_t *out, int max_out) {
    int n = (int)strlen(text);
    int pos = 0;
    int total = 0;
    const uint8_t *data = (const uint8_t *)text;

    if (n <= 0 || max_out <= 0) return 0;

    while (pos < n && total < max_out) {
        int off = 0, len = 0;
        int32_t special_id = -1;
        if (find_next_special_token(tok, data + pos, n - pos, &off, &special_id, &len)) {
            if (off > 0) {
                if (tok->pre_name && strcmp(tok->pre_name, "qwen35") == 0) {
                    total += encode_qwen35_raw(tok, data + pos, off, out + total, max_out - total);
                } else {
                    total += encode_basic_raw(tok, data + pos, off, out + total, max_out - total);
                }
            }
            if (total < max_out) out[total++] = special_id;
            pos += off + len;
        } else {
            if (tok->pre_name && strcmp(tok->pre_name, "qwen35") == 0) {
                total += encode_qwen35_raw(tok, data + pos, n - pos, out + total, max_out - total);
            } else {
                total += encode_basic_raw(tok, data + pos, n - pos, out + total, max_out - total);
            }
            break;
        }
    }

    return total;
}

/* ------------------------------------------------------------------ */
/* Decode                                                               */
/* ------------------------------------------------------------------ */

char *tokenizer_decode(const tokenizer_t *tok, const int32_t *ids, int n) {
    /* Upper bound: each token can decode to at most ~50 bytes. */
    char *out = (char *)malloc((size_t)n * 64 + 1);
    if (!out) tok_fatal("OOM in tokenizer_decode");
    int pos = 0;

    for (int i = 0; i < n; i++) {
        int32_t id = ids[i];
        if (id < 0 || id >= tok->vocab_size) continue;
        const char *piece = tok->id_to_tok[id];

        /* Decode each GPT-2 Unicode codepoint back to the original byte. */
        const char *p = piece;
        while (*p) {
            uint32_t cp = utf8_to_cp(&p);
            int      b  = cp_to_byte(cp, tok->missing);
            if (b >= 0)
                out[pos++] = (char)(uint8_t)b;
        }
    }
    out[pos] = '\0';
    return out;
}

/* ------------------------------------------------------------------ */
/* Self-test                                                            */
/* ------------------------------------------------------------------ */

#ifdef SELFTEST_MAIN

int tokenizer_selftest(const char *model_path) {
    printf("=== Tokenizer self-test ===\n");
    printf("Loading %s\n", model_path);

    tokenizer_t *tok = tokenizer_load(model_path);
    if (!tok) { printf("FAILED: tokenizer_load returned NULL\n"); return 1; }

    printf("vocab=%d  merges=%d  bos=%d  eos=%d\n",
           tok->vocab_size, tok->num_merges, tok->bos_id, tok->eos_id);

    /* ---- encode "Hello" ---- */
    const char *text = "Hello";
    int32_t ids[64];
    int n = tokenizer_encode(tok, text, ids, 64);

    printf("encode(\"%s\") -> %d token(s): [", text, n);
    for (int i = 0; i < n; i++) printf("%s%d", i ? ", " : "", ids[i]);
    printf("]\n");

    /* Print token strings */
    printf("token strings: [");
    for (int i = 0; i < n; i++) {
        if (ids[i] >= 0 && ids[i] < tok->vocab_size) {
            /* Decode GPT-2 piece → show printable */
            char *dec = tokenizer_decode(tok, &ids[i], 1);
            printf("%s\"%s\"", i ? ", " : "", dec);
            free(dec);
        }
    }
    printf("]\n");

    /* ---- decode back ---- */
    char *decoded = tokenizer_decode(tok, ids, n);
    printf("decode([...]) -> \"%s\"\n", decoded);

    int ok = (strcmp(decoded, text) == 0);
    printf("Round-trip: %s\n", ok ? "PASS" : "FAIL");

    free(decoded);
    tokenizer_free(tok);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] :
        "C:\\Users\\arahe\\.ollama\\models\\blobs\\"
        "sha256-900dde62fb7ebe8a5a25e35d5b7633f403f226a310965fed51d50f5238ba145a";
    return tokenizer_selftest(path);
}

#endif /* SELFTEST_MAIN */
