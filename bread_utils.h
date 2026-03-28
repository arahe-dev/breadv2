#ifndef BREAD_UTILS_H
#define BREAD_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/* CUDA error check — fail loud and early                             */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call) do {                                           \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d — %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));            \
        exit(1);                                                        \
    }                                                                   \
} while (0)

/* ------------------------------------------------------------------ */
/* Host fp16 → float32 conversion (correct subnormal handling)         */
/* ------------------------------------------------------------------ */

static inline float bread_h2f(uint16_t h)
{
    uint32_t sign     = (uint32_t)(h >> 15) << 31;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x03FF;
    uint32_t bits;

    if (exponent == 0) {
        /* zero or subnormal fp16: value = ±mantissa × 2^(-24) */
        float val = (float)mantissa * (1.0f / 16777216.0f);
        return (h >> 15) ? -val : val;
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }

    float f;
    memcpy(&f, &bits, 4);
    return f;
}

#endif /* BREAD_UTILS_H */
