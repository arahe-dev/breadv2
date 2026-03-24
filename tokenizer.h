#ifndef TOKENIZER_H
#define TOKENIZER_H

/* tokenizer.h — BPE tokenizer for GPT-2 byte-level BPE (Qwen3.5).
 *
 * Reads vocabulary and merge rules directly from GGUF metadata.
 * Implements encode(text → ids) and decode(ids → text).
 * C99, no external dependencies.
 */

#include <stdint.h>

/* Opaque tokenizer context */
typedef struct tokenizer_t tokenizer_t;

/* Load vocabulary and merge rules from a GGUF model file.
 * Reads only the metadata section — does not load weight data.
 * Returns NULL on failure. */
tokenizer_t *tokenizer_load(const char *model_path);

/* Free all resources. */
void tokenizer_free(tokenizer_t *tok);

/* Encode UTF-8 text to token IDs using BPE.
 * out[] must be pre-allocated with at least max_out entries.
 * Returns the number of tokens written. */
int tokenizer_encode(const tokenizer_t *tok, const char *text,
                     int32_t *out, int max_out);

/* Decode token IDs to UTF-8 text.
 * Returns a heap-allocated null-terminated string; caller must free(). */
char *tokenizer_decode(const tokenizer_t *tok,
                       const int32_t *ids, int n);

/* BOS and EOS token IDs (from GGUF metadata). */
int32_t tokenizer_bos(const tokenizer_t *tok);
int32_t tokenizer_eos(const tokenizer_t *tok);

/* Vocabulary size. */
int32_t tokenizer_vocab_size(const tokenizer_t *tok);

/* Run self-test: encode "Hello", print IDs, decode back, verify round-trip.
 * Returns 0 on success, 1 on failure. */
int tokenizer_selftest(const char *model_path);

#endif /* TOKENIZER_H */
