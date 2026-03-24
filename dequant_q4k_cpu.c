/* dequant_q4k_cpu.c
 *
 * CPU reference implementations of Q4_K and Q6_K block dequantisation.
 * Bit-manipulation logic copied directly from:
 *   llama.cpp/ggml/src/ggml-quants.c
 *     — dequantize_row_q4_K
 *     — dequantize_row_q6_K
 *
 * Q4_K block layout (144 bytes, 256 elements):
 *   [0..1]    fp16  d       — super-block scale
 *   [2..3]    fp16  dmin    — super-block min
 *   [4..15]   u8[12]        — 8 pairs of 6-bit sub-scale/sub-min (packed)
 *   [16..143] u8[128]       — 256 4-bit quantised values, two per byte
 *
 * Q6_K block layout (210 bytes, 256 elements):
 *   [0..127]   u8[128] ql   — low 4 bits of each 6-bit value
 *   [128..191] u8[64]  qh   — high 2 bits (4 values packed per byte)
 *   [192..207] i8[16]  sc   — per-group int8 scales
 *   [208..209] fp16    d    — super-block scale
 *
 * Compile (MSVC, no CUDA):
 *   cl /nologo /W3 /O2 dequant_q4k_cpu.c /Fe:dequant_q4k_cpu.exe
 */

#ifdef _WIN32
#  define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* fp16 -> fp32 (no external deps)                                     */
/* Handles: zero, normal, inf/nan. Subnormals flushed to zero         */
/* (never appear in model weights).                                    */
/* ------------------------------------------------------------------ */
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign     = (uint32_t)(h >> 15) << 31;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x03FF;
    uint32_t bits;

    if (exponent == 0) {
        bits = sign;                                     /* ±0 / subnormal → 0 */
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);   /* inf / NaN */
    } else {
        /* normal: fp32_exp = fp16_exp - 15 + 127 = fp16_exp + 112 */
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }

    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* ------------------------------------------------------------------ */
/* get_scale_min_k4 — copied verbatim from ggml-quants.c              */
/*                                                                      */
/* Extracts one (sub-scale, sub-min) pair from the 12-byte packed     */
/* scales field of a Q4_K block.                                       */
/*                                                                      */
/* The 12 bytes encode 8 pairs of 6-bit values:                       */
/*   indices 0..3  → stored in lower 6 bits of scales[0..3] (d)      */
/*                   and lower 6 bits of scales[4..7] (m)             */
/*   indices 4..7  → upper 2 bits overflow into scales[8..11],        */
/*                   packed two-per-byte as low/high nibble            */
/* ------------------------------------------------------------------ */
static void get_scale_min_k4(int j, const uint8_t *scales,
                              uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = scales[j]     & 63;
        *m = scales[j + 4] & 63;
    } else {
        *d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m = (scales[j + 4] >>   4) | ((scales[j - 0] >> 6) << 4);
    }
}

/* ------------------------------------------------------------------ */
/* dequant_q4k_block_cpu                                               */
/*                                                                      */
/* Dequantise one 144-byte Q4_K block into 256 float32 values.        */
/* Formula (from ggml-quants.c dequantize_row_q4_K):                  */
/*   out[l]      = d * sc1 * (qs[l] & 0xF) - dmin * m1               */
/*   out[32 + l] = d * sc2 * (qs[l] >> 4)  - dmin * m2               */
/* for each group of 32 bytes (four groups of 64 elements total).     */
/* ------------------------------------------------------------------ */
void dequant_q4k_block_cpu(const uint8_t *block, float *out) {
    /* Parse header */
    uint16_t d_raw, dmin_raw;
    memcpy(&d_raw,    block + 0, 2);
    memcpy(&dmin_raw, block + 2, 2);
    const float d    = fp16_to_fp32(d_raw);
    const float dmin = fp16_to_fp32(dmin_raw);

    const uint8_t *scales = block + 4;    /* 12 bytes */
    const uint8_t *qs     = block + 16;   /* 128 bytes */

    /* --- dequantize_row_q4_K inner loop, copied from ggml-quants.c --- */
    int is = 0;
    float *y = out;
    const uint8_t *q = qs;

    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;

        get_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = dmin * m;

        get_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = dmin * m;

        for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;

        q  += 32;
        is += 2;
    }
}

/* ------------------------------------------------------------------ */
/* dequant_q6k_block_cpu                                               */
/*                                                                      */
/* Dequantise one 210-byte Q6_K block into 256 float32 values.        */
/* Algorithm (from ggml-quants.c dequantize_row_q6_K):                */
/*                                                                      */
/* Two outer passes of 128 elements each (n = 0, 128).                */
/* Inner loop l = 0..31 reconstructs 4 signed 6-bit values per l:    */
/*   q1: low nibble of ql[l],    bits 0-1 of qh[l]                    */
/*   q2: low nibble of ql[l+32], bits 2-3 of qh[l]                    */
/*   q3: high nibble of ql[l],   bits 4-5 of qh[l]                    */
/*   q4: high nibble of ql[l+32],bits 6-7 of qh[l]                    */
/* Each raw 6-bit value is offset by -32 → signed range -32..31.      */
/* Scattered to y[l+0], y[l+32], y[l+64], y[l+96].                   */
/* Formula: y = d * sc[group] * q                                      */
/* ------------------------------------------------------------------ */
void dequant_q6k_block_cpu(const uint8_t *block, float *out) {
    uint16_t d_raw;
    memcpy(&d_raw, block + 208, 2);
    const float d = fp16_to_fp32(d_raw);

    const uint8_t *ql = block + 0;            /* 128 bytes: low nibbles */
    const uint8_t *qh = block + 128;          /* 64 bytes:  high 2-bit pairs */
    const int8_t  *sc = (const int8_t *)(block + 192);  /* 16 int8 scales */

    float *y = out;

    /* --- dequantize_row_q6_K inner loops, copied from ggml-quants.c --- */
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;

            const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            y[l +  0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

/* ------------------------------------------------------------------ */
/* main — construct blocks with known values, run both functions      */
/* ------------------------------------------------------------------ */
int main(void) {
    uint8_t block[144];
    memset(block, 0, sizeof(block));

    /* Set d = 1.0 (fp16 = 0x3C00), dmin = 0.0 (fp16 = 0x0000) */
    uint16_t d_fp16    = 0x3C00;   /* 1.0 */
    uint16_t dmin_fp16 = 0x0000;   /* 0.0 */
    memcpy(block + 0, &d_fp16,    2);
    memcpy(block + 2, &dmin_fp16, 2);

    /*
     * Set scales so every sub-scale sc = 1, sub-min m = 0.
     *
     * get_scale_min_k4 for indices 0..3 (j < 4):
     *   sc = scales[j] & 63  → set scales[0..3] = 1
     *   m  = scales[j+4] & 63 → set scales[4..7] = 0
     *
     * get_scale_min_k4 for indices 4..7 (j >= 4):
     *   sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
     *      = (1 & 0x0F) | 0 = 1           → set scales[8..11] = 1
     *   m  = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
     *      = 0 | 0 = 0                     ✓ (scales[4..7] = 0)
     *
     * With d=1.0, dmin=0.0, sc=1, m=0:
     *   out[l]      = 1.0 * 1 * (qs[l] & 0xF) - 0 = low nibble
     *   out[32 + l] = 1.0 * 1 * (qs[l]  >> 4) - 0 = high nibble
     */
    uint8_t *scales = block + 4;
    scales[0] = 1; scales[1] = 1; scales[2] = 1; scales[3] = 1;
    scales[4] = 0; scales[5] = 0; scales[6] = 0; scales[7] = 0;
    scales[8] = 1; scales[9] = 1; scales[10] = 1; scales[11] = 1;

    /*
     * Fill qs with 0x59: low nibble = 9, high nibble = 5.
     * Expected outputs:
     *   elements   0..31  → 9.0f  (low nibble, group 0)
     *   elements  32..63  → 5.0f  (high nibble, group 0)
     *   elements  64..95  → 9.0f  (low nibble, group 1)
     *   ... repeating
     */
    memset(block + 16, 0x59, 128);

    float out[256];
    dequant_q4k_block_cpu(block, out);

    printf("--- Q4_K ---\n");
    printf("first 8 outputs (expect 9.0 each):\n");
    for (int i = 0; i < 8; i++)
        printf("  out[%d] = %.4f\n", i, out[i]);

    printf("first 8 of high-nibble group (expect 5.0 each):\n");
    for (int i = 32; i < 40; i++)
        printf("  out[%d] = %.4f\n", i, out[i]);

    int q4k_pass = 1;
    for (int i = 0;  i < 32; i++) if (out[i] != 9.0f) { q4k_pass = 0; break; }
    for (int i = 32; i < 64; i++) if (out[i] != 5.0f) { q4k_pass = 0; break; }
    printf("Q4_K: %s\n\n", q4k_pass ? "PASS" : "FAIL");

    /* ----------------------------------------------------------------
     * Q6_K test
     *
     * Target: d=1.0, all scales=1, all 6-bit raw values = 40.
     *
     * 6-bit value = 40 = 0b101000:
     *   low 4 bits  = 8  (0b1000)
     *   high 2 bits = 2  (0b10)
     *
     * q1 needs: ql[l+0]  low  nibble = 8, qh[l] bits 0-1 = 2
     * q2 needs: ql[l+32] low  nibble = 8, qh[l] bits 2-3 = 2
     * q3 needs: ql[l+0]  high nibble = 8, qh[l] bits 4-5 = 2
     * q4 needs: ql[l+32] high nibble = 8, qh[l] bits 6-7 = 2
     *
     * → ql[all] = 0x88  (low nibble = 8, high nibble = 8)
     * → qh[all] = 0xAA  (bits 0-1=10, 2-3=10, 4-5=10, 6-7=10 = 0b10101010)
     *
     * Verification: q1 = (0x88 & 0xF) | ((0xAA>>0 & 3) << 4) - 32
     *                  = 8 | (2<<4) - 32 = 40 - 32 = 8
     * output = d * sc * q = 1.0 * 1 * 8 = 8.0  (all 256 elements)
     * ---------------------------------------------------------------- */
    uint8_t blk6[210];
    memset(blk6, 0, sizeof(blk6));

    uint16_t d6_fp16 = 0x3C00;             /* 1.0 */
    memcpy(blk6 + 208, &d6_fp16, 2);

    memset(blk6 + 0,   0x88, 128);         /* ql: all bytes = 0x88 */
    memset(blk6 + 128, 0xAA, 64);          /* qh: all bytes = 0xAA */
    memset(blk6 + 192, 0x01, 16);          /* scales: all int8 = 1  */

    float out6[256];
    dequant_q6k_block_cpu(blk6, out6);

    printf("--- Q6_K ---\n");
    printf("first 8 outputs (expect 8.0 each):\n");
    for (int i = 0; i < 8; i++)
        printf("  out[%d] = %.4f\n", i, out6[i]);

    printf("spot-checks at 32, 64, 96, 128, 160, 192, 224 (expect 8.0):\n");
    int spots[] = {32, 64, 96, 128, 160, 192, 224};
    for (int i = 0; i < 7; i++)
        printf("  out[%d] = %.4f\n", spots[i], out6[spots[i]]);

    int q6k_pass = 1;
    for (int i = 0; i < 256; i++)
        if (out6[i] != 8.0f) { q6k_pass = 0; break; }
    printf("Q6_K: %s\n\n", q6k_pass ? "PASS" : "FAIL");

    int all_pass = q4k_pass && q6k_pass;
    printf("Overall: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
