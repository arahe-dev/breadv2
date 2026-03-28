/* kernels.cu — BREAD CUDA dequant matvec kernels
 *
 * Q4_K block layout (144 bytes, 256 elements):
 *   [0..1]    fp16  d       — super-block scale
 *   [2..3]    fp16  dmin    — super-block min
 *   [4..15]   u8[12]        — 8 pairs of 6-bit (sub-scale, sub-min)
 *   [16..143] u8[128]       — 256 4-bit quantised values, two per byte
 *
 * Q6_K block layout (210 bytes, 256 elements):
 *   [0..127]   u8[128] ql   — low 4 bits of each 6-bit value
 *   [128..191] u8[64]  qh   — high 2 bits (4 values packed per byte)
 *   [192..207] i8[16]  sc   — per-group int8 scales
 *   [208..209] fp16    d    — super-block scale
 *
 * Kernel layout: one CUDA block per output row, 256 threads per block.
 * Thread tid handles element tid within each 256-element quantised block.
 *
 * Build (standalone selftest):
 *   nvcc -DSELFTEST_MAIN -O2 kernels.cu -o kernels_test.exe
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Error-checking macro — fail loud and early                          */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call) do {                                          \
    cudaError_t _err = (call);                                         \
    if (_err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s:%d — %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(_err));         \
        exit(1);                                                       \
    }                                                                  \
} while (0)

/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */
#define Q4K_BLOCK_BYTES  144
#define Q6K_BLOCK_BYTES  210
#define BLOCK_ELEMS      256   /* elements per quantised block (both types) */
#define THREADS_PER_ROW  256   /* one thread per element */

#define QTYPE_Q4_K  12
#define QTYPE_Q6_K  14

/* ------------------------------------------------------------------ */
/* __device__: dequantise element e (0..255) from one Q4_K block      */
/*                                                                      */
/* Math mirrors dequant_q4k_block_cpu in dequant_q4k_cpu.c:           */
/*   element e → group = e/64, sub = (e%64)/32, pos = e%32            */
/*   is = group*2 + sub  (sub-scale index 0..7)                        */
/*   qs_byte = qs[group*32 + pos]                                      */
/*   nibble  = sub==0 ? (qs_byte & 0xF) : (qs_byte >> 4)             */
/*   out     = d * sc * nibble  −  dmin * m                            */
/* ------------------------------------------------------------------ */
static __device__ float dequant_q4k_elem(const uint8_t * __restrict__ blk,
                                          int e)
{
    /* Read fp16 header — byte-by-byte to avoid alignment issues */
    uint16_t d_raw    = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    uint16_t dmin_raw = (uint16_t)blk[2] | ((uint16_t)blk[3] << 8);
    float d    = __half2float(__ushort_as_half(d_raw));
    float dmin = __half2float(__ushort_as_half(dmin_raw));

    const uint8_t *scales = blk + 4;    /* 12 bytes */
    const uint8_t *qs     = blk + 16;   /* 128 bytes */

    int group = e / 64;
    int sub   = (e % 64) / 32;   /* 0 = low nibble, 1 = high nibble */
    int pos   = e % 32;
    int is    = group * 2 + sub; /* 0..7 */

    /* get_scale_min_k4(is, scales, &sc, &m) — verbatim from ggml-quants.c */
    uint8_t sc, m;
    if (is < 4) {
        sc = scales[is]     & 63;
        m  = scales[is + 4] & 63;
    } else {
        sc = (scales[is + 4] & 0x0F) | ((scales[is - 4] >> 6) << 4);
        m  = (scales[is + 4] >>   4) | ((scales[is    ] >> 6) << 4);
    }

    uint8_t qs_byte = qs[group * 32 + pos];
    int nibble = (sub == 0) ? (qs_byte & 0xF) : (qs_byte >> 4);

    return d * (float)sc * (float)nibble - dmin * (float)m;
}

/* ------------------------------------------------------------------ */
/* __device__: dequantise element e (0..255) from one Q6_K block      */
/*                                                                      */
/* Math mirrors dequant_q6k_block_cpu in dequant_q4k_cpu.c:           */
/*   pass = e/128, e_in = e%128, sub = e_in/32, l = e_in%32           */
/*   is   = l/16  (scale sub-index within pass, 0 or 1)               */
/*   ql_base = pass*64, qh_base = pass*32, sc_base = pass*8           */
/*   sub 0: ql[ql_base+l]    low nibble, qh bits [1:0]               */
/*   sub 1: ql[ql_base+l+32] low nibble, qh bits [3:2]               */
/*   sub 2: ql[ql_base+l]    high nibble, qh bits [5:4]              */
/*   sub 3: ql[ql_base+l+32] high nibble, qh bits [7:6]              */
/*   q = (nibble | (qh_bits << 4)) − 32                               */
/*   out = d * sc[sc_base + is + sub*2] * q                           */
/* ------------------------------------------------------------------ */
static __device__ float dequant_q6k_elem(const uint8_t * __restrict__ blk,
                                          int e)
{
    uint16_t d_raw = (uint16_t)blk[208] | ((uint16_t)blk[209] << 8);
    float d = __half2float(__ushort_as_half(d_raw));

    const uint8_t *ql = blk + 0;
    const uint8_t *qh = blk + 128;
    const int8_t  *sc = (const int8_t *)(blk + 192);

    int pass   = e / 128;
    int e_in   = e % 128;
    int sub    = e_in / 32;   /* 0..3 */
    int l      = e_in % 32;
    int is     = l / 16;      /* 0 or 1 within pass */

    int ql_base = pass * 64;
    int qh_base = pass * 32;
    int sc_base = pass * 8;

    int ql_idx, qh_shift, use_high_nibble;
    if      (sub == 0) { ql_idx = ql_base + l;      qh_shift = 0; use_high_nibble = 0; }
    else if (sub == 1) { ql_idx = ql_base + l + 32; qh_shift = 2; use_high_nibble = 0; }
    else if (sub == 2) { ql_idx = ql_base + l;      qh_shift = 4; use_high_nibble = 1; }
    else               { ql_idx = ql_base + l + 32; qh_shift = 6; use_high_nibble = 1; }

    int nibble  = use_high_nibble ? (ql[ql_idx] >> 4) : (ql[ql_idx] & 0xF);
    int qh_bits = (qh[qh_base + l] >> qh_shift) & 3;
    int8_t q    = (int8_t)(nibble | (qh_bits << 4)) - 32;

    int sc_idx = sc_base + is + sub * 2;
    return d * (float)sc[sc_idx] * (float)q;
}

/* ------------------------------------------------------------------ */
/* Warp-level sum reduction (full mask)                                */
/* ------------------------------------------------------------------ */
static __device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xFFFFFFFF, v, offset);
    return v;
}

/* ------------------------------------------------------------------ */
/* Q4_K matvec kernel                                                   */
/*                                                                      */
/* y[row] = sum_col( dequant(W[row, col]) * x[col] )                  */
/* Grid: (rows,)   Block: (256,)                                       */
/* Shared: 256 floats for x-tile + 8 floats for warp reduction        */
/* ------------------------------------------------------------------ */
__global__ void dequant_q4k_matvec(
    const uint8_t * __restrict__ w,
    const half    * __restrict__ x,
    half          * __restrict__ y,
    int rows, int cols)
{
    __shared__ float sx[BLOCK_ELEMS];
    __shared__ float warp_buf[8];           /* one slot per warp (8 warps) */

    int row    = (int)blockIdx.x;
    if (row >= rows) return;

    int tid    = (int)threadIdx.x;
    int warp   = tid >> 5;
    int lane   = tid & 31;
    int nblocks = cols / BLOCK_ELEMS;       /* Q4_K blocks per row */

    float sum = 0.0f;

    for (int b = 0; b < nblocks; b++) {
        /* Cooperatively load 256 x-values into shared memory */
        sx[tid] = __half2float(x[b * BLOCK_ELEMS + tid]);
        __syncthreads();

        /* Each thread dequantises its element and accumulates */
        const uint8_t *blk =
            w + ((size_t)row * nblocks + b) * Q4K_BLOCK_BYTES;
        sum += dequant_q4k_elem(blk, tid) * sx[tid];

        __syncthreads();
    }

    /* Reduce within each warp */
    sum = warp_reduce_sum(sum);
    if (lane == 0) warp_buf[warp] = sum;
    __syncthreads();

    /* Reduce across warps — first warp finishes the job */
    if (warp == 0) {
        float v = (lane < 8) ? warp_buf[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) y[row] = __float2half(v);
    }
}

/* ------------------------------------------------------------------ */
/* Q6_K matvec kernel  (same structure as Q4_K)                       */
/* ------------------------------------------------------------------ */
__global__ void dequant_q6k_matvec(
    const uint8_t * __restrict__ w,
    const half    * __restrict__ x,
    half          * __restrict__ y,
    int rows, int cols)
{
    __shared__ float sx[BLOCK_ELEMS];
    __shared__ float warp_buf[8];

    int row    = (int)blockIdx.x;
    if (row >= rows) return;

    int tid    = (int)threadIdx.x;
    int warp   = tid >> 5;
    int lane   = tid & 31;
    int nblocks = cols / BLOCK_ELEMS;

    float sum = 0.0f;

    for (int b = 0; b < nblocks; b++) {
        sx[tid] = __half2float(x[b * BLOCK_ELEMS + tid]);
        __syncthreads();

        const uint8_t *blk =
            w + ((size_t)row * nblocks + b) * Q6K_BLOCK_BYTES;
        sum += dequant_q6k_elem(blk, tid) * sx[tid];

        __syncthreads();
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) warp_buf[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < 8) ? warp_buf[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) y[row] = __float2half(v);
    }
}

/* ------------------------------------------------------------------ */
/* Host dispatcher                                                      */
/* ------------------------------------------------------------------ */
void bread_matvec(void *w, half *x, half *y, int rows, int cols, int qtype)
{
    dim3 grid(rows);
    dim3 block(THREADS_PER_ROW);

    if (qtype == QTYPE_Q4_K) {
        dequant_q4k_matvec<<<grid, block>>>(
            (const uint8_t *)w, x, y, rows, cols);
    } else if (qtype == QTYPE_Q6_K) {
        dequant_q6k_matvec<<<grid, block>>>(
            (const uint8_t *)w, x, y, rows, cols);
    } else {
        fprintf(stderr, "bread_matvec: unknown qtype %d\n", qtype);
        exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}

/* ================================================================== */
/*  S E L F T E S T                                                    */
/* ================================================================== */
#ifdef SELFTEST_MAIN

/* ------------------------------------------------------------------ */
/* CPU reference: fp16 bytes → float32                                 */
/* ------------------------------------------------------------------ */
static float fp16_to_fp32_cpu(uint16_t h) {
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
    float f; memcpy(&f, &bits, 4); return f;
}

/* ------------------------------------------------------------------ */
/* CPU reference get_scale_min_k4 (verbatim from ggml-quants.c)       */
/* ------------------------------------------------------------------ */
static void get_scale_min_k4_cpu(int j, const uint8_t *scales,
                                  uint8_t *d_out, uint8_t *m_out)
{
    if (j < 4) {
        *d_out = scales[j]     & 63;
        *m_out = scales[j + 4] & 63;
    } else {
        *d_out = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m_out = (scales[j + 4] >>   4) | ((scales[j    ] >> 6) << 4);
    }
}

/* ------------------------------------------------------------------ */
/* CPU reference: dequant one Q4_K block → float32[256]               */
/* ------------------------------------------------------------------ */
static void cpu_dequant_q4k(const uint8_t *block, float *out)
{
    uint16_t d_raw, dmin_raw;
    memcpy(&d_raw,    block + 0, 2);
    memcpy(&dmin_raw, block + 2, 2);
    float d    = fp16_to_fp32_cpu(d_raw);
    float dmin = fp16_to_fp32_cpu(dmin_raw);

    const uint8_t *scales = block + 4;
    const uint8_t *qs     = block + 16;

    int is = 0;
    float *y = out;
    const uint8_t *q = qs;

    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4_cpu(is + 0, scales, &sc, &m);
        float d1 = d * sc, m1 = dmin * m;
        get_scale_min_k4_cpu(is + 1, scales, &sc, &m);
        float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * (float)(q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * (float)(q[l] >>   4) - m2;
        q  += 32;
        is += 2;
    }
}

/* ------------------------------------------------------------------ */
/* CPU reference: dequant one Q6_K block → float32[256]               */
/* ------------------------------------------------------------------ */
static void cpu_dequant_q6k(const uint8_t *block, float *out)
{
    uint16_t d_raw;
    memcpy(&d_raw, block + 208, 2);
    float d = fp16_to_fp32_cpu(d_raw);

    const uint8_t *ql = block + 0;
    const uint8_t *qh = block + 128;
    const int8_t  *sc = (const int8_t *)(block + 192);

    float *y = out;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l +  0] = d * (float)sc[is + 0] * (float)q1;
            y[l + 32] = d * (float)sc[is + 2] * (float)q2;
            y[l + 64] = d * (float)sc[is + 4] * (float)q3;
            y[l + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        y  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

/* ------------------------------------------------------------------ */
/* CPU naive dot product (float32)                                     */
/* ------------------------------------------------------------------ */
static float cpu_dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* ------------------------------------------------------------------ */
/* Build one Q4_K test block                                           */
/*   d = 1.0, dmin = 0.0, all sub-scales = 1, all sub-mins = 0       */
/*   qs = 0x59  →  low nibble = 9, high nibble = 5                    */
/* Expected dequant per element:                                       */
/*   elements 0-31: 9.0, 32-63: 5.0, 64-95: 9.0, ..., 224-255: 5.0  */
/* ------------------------------------------------------------------ */
static void make_q4k_test_block(uint8_t *blk)
{
    memset(blk, 0, Q4K_BLOCK_BYTES);
    uint16_t d_fp16    = 0x3C00; /* 1.0 */
    uint16_t dmin_fp16 = 0x0000; /* 0.0 */
    memcpy(blk + 0, &d_fp16,    2);
    memcpy(blk + 2, &dmin_fp16, 2);

    /* scales[0..3]=1 → sc=1 for is=0..3
     * scales[4..7]=0 → m=0 for is=0..3
     * scales[8..11]=1 → sc=1 for is=4..7 (the (is+4) nibble = 1)
     * All m=0 because scales[4..7]=0 and upper bits of scales[0..3]=0 */
    uint8_t *s = blk + 4;
    s[0]=1; s[1]=1; s[2]=1; s[3]=1;
    s[4]=0; s[5]=0; s[6]=0; s[7]=0;
    s[8]=1; s[9]=1; s[10]=1; s[11]=1;

    /* qs = 0x59: low nibble=9, high nibble=5 */
    memset(blk + 16, 0x59, 128);
}

/* ------------------------------------------------------------------ */
/* Build one Q6_K test block                                           */
/*   d = 1.0, all scales = 1, all 6-bit values = 40 → q = 8          */
/*   ql = 0x88, qh = 0xAA                                             */
/* Expected: all 256 outputs = 8.0                                     */
/* ------------------------------------------------------------------ */
static void make_q6k_test_block(uint8_t *blk)
{
    memset(blk, 0, Q6K_BLOCK_BYTES);
    uint16_t d_fp16 = 0x3C00; /* 1.0 */
    memcpy(blk + 208, &d_fp16, 2);
    memset(blk + 0,   0x88, 128); /* ql */
    memset(blk + 128, 0xAA, 64);  /* qh */
    memset(blk + 192, 0x01, 16);  /* sc: all int8 = 1 */
}

/* ------------------------------------------------------------------ */
/* kernels_selftest — returns 0 on PASS, 1 on FAIL                    */
/* ------------------------------------------------------------------ */
int kernels_selftest(void)
{
    printf("=== kernels_selftest ===\n");

    /* Test dimensions: ROWS rows, COLS cols (one quant block per row) */
    const int ROWS = 8;
    const int COLS = 256; /* = one block per row */
    int pass = 1;

    /* ---- Q4_K test ---- */
    {
        printf("Q4_K: %d rows × %d cols ... ", ROWS, COLS);

        /* Build weight matrix on CPU */
        size_t w_bytes = (size_t)ROWS * Q4K_BLOCK_BYTES;
        uint8_t *w_cpu = (uint8_t *)malloc(w_bytes);
        for (int r = 0; r < ROWS; r++)
            make_q4k_test_block(w_cpu + (size_t)r * Q4K_BLOCK_BYTES);

        /* Input vector x = all 1.0 (fp16) */
        half *x_cpu = (half *)malloc(COLS * sizeof(half));
        for (int i = 0; i < COLS; i++)
            x_cpu[i] = __float2half(1.0f);

        /* CPU reference output */
        float ref_dequant[256];
        float ref_y[ROWS];
        float x_f32[COLS];
        for (int i = 0; i < COLS; i++) x_f32[i] = 1.0f;
        for (int r = 0; r < ROWS; r++) {
            cpu_dequant_q4k(w_cpu + (size_t)r * Q4K_BLOCK_BYTES, ref_dequant);
            ref_y[r] = cpu_dot(ref_dequant, x_f32, COLS);
        }

        /* Allocate GPU buffers */
        uint8_t *w_gpu; half *x_gpu, *y_gpu;
        CUDA_CHECK(cudaMalloc(&w_gpu, w_bytes));
        CUDA_CHECK(cudaMalloc(&x_gpu, COLS * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&y_gpu, ROWS * sizeof(half)));

        CUDA_CHECK(cudaMemcpy(w_gpu, w_cpu, w_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(x_gpu, x_cpu, COLS * sizeof(half), cudaMemcpyHostToDevice));

        /* Run kernel */
        bread_matvec(w_gpu, x_gpu, y_gpu, ROWS, COLS, QTYPE_Q4_K);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Copy result back */
        half y_cpu[ROWS];
        CUDA_CHECK(cudaMemcpy(y_cpu, y_gpu, ROWS * sizeof(half), cudaMemcpyDeviceToHost));

        /* Compare */
        float max_err = 0.0f;
        for (int r = 0; r < ROWS; r++) {
            float gpu_val = __half2float(y_cpu[r]);
            float err = fabsf(gpu_val - ref_y[r]);
            if (err > max_err) max_err = err;
        }

        printf("ref_y[0]=%.2f  gpu_y[0]=%.2f  max_err=%.6f  %s\n",
               ref_y[0], __half2float(y_cpu[0]), max_err,
               max_err < 0.05f ? "PASS" : "FAIL");

        if (max_err >= 0.05f) pass = 0;

        cudaFree(w_gpu); cudaFree(x_gpu); cudaFree(y_gpu);
        free(w_cpu); free(x_cpu);
    }

    /* ---- Q6_K test ---- */
    {
        printf("Q6_K: %d rows × %d cols ... ", ROWS, COLS);

        size_t w_bytes = (size_t)ROWS * Q6K_BLOCK_BYTES;
        uint8_t *w_cpu = (uint8_t *)malloc(w_bytes);
        for (int r = 0; r < ROWS; r++)
            make_q6k_test_block(w_cpu + (size_t)r * Q6K_BLOCK_BYTES);

        half *x_cpu = (half *)malloc(COLS * sizeof(half));
        for (int i = 0; i < COLS; i++)
            x_cpu[i] = __float2half(1.0f);

        float ref_dequant[256];
        float ref_y[ROWS];
        float x_f32[COLS];
        for (int i = 0; i < COLS; i++) x_f32[i] = 1.0f;
        for (int r = 0; r < ROWS; r++) {
            cpu_dequant_q6k(w_cpu + (size_t)r * Q6K_BLOCK_BYTES, ref_dequant);
            ref_y[r] = cpu_dot(ref_dequant, x_f32, COLS);
        }

        uint8_t *w_gpu; half *x_gpu, *y_gpu;
        CUDA_CHECK(cudaMalloc(&w_gpu, w_bytes));
        CUDA_CHECK(cudaMalloc(&x_gpu, COLS * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&y_gpu, ROWS * sizeof(half)));

        CUDA_CHECK(cudaMemcpy(w_gpu, w_cpu, w_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(x_gpu, x_cpu, COLS * sizeof(half), cudaMemcpyHostToDevice));

        bread_matvec(w_gpu, x_gpu, y_gpu, ROWS, COLS, QTYPE_Q6_K);
        CUDA_CHECK(cudaDeviceSynchronize());

        half y_cpu[ROWS];
        CUDA_CHECK(cudaMemcpy(y_cpu, y_gpu, ROWS * sizeof(half), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (int r = 0; r < ROWS; r++) {
            float gpu_val = __half2float(y_cpu[r]);
            float err = fabsf(gpu_val - ref_y[r]);
            if (err > max_err) max_err = err;
        }

        printf("ref_y[0]=%.2f  gpu_y[0]=%.2f  max_err=%.6f  %s\n",
               ref_y[0], __half2float(y_cpu[0]), max_err,
               max_err < 0.05f ? "PASS" : "FAIL");

        if (max_err >= 0.05f) pass = 0;

        cudaFree(w_gpu); cudaFree(x_gpu); cudaFree(y_gpu);
        free(w_cpu); free(x_cpu);
    }

    printf("\nkernels_selftest: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main(void) {
    return kernels_selftest();
}

#endif /* SELFTEST_MAIN */
