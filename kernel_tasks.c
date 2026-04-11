#include "kernel_tasks.h"
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
static double now_ms_windows(void) {
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER now;
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart * 1000.0;
}
#define now_ms() now_ms_windows()
#else
#include <sys/time.h>
static double now_ms_unix(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}
#define now_ms() now_ms_unix()
#endif

static KernelTask g_current_task = {KTASK_NONE, KTASK_STATUS_PENDING, -1, -1, 0.0, 0.0, {0}};

void ktask_begin(KernelTaskType type, int layer_idx, int token_pos) {
    memset(&g_current_task, 0, sizeof(KernelTask));
    g_current_task.type = type;
    g_current_task.status = KTASK_STATUS_RUNNING;
    g_current_task.layer_idx = layer_idx;
    g_current_task.token_pos = token_pos;
    g_current_task.start_ms = now_ms();
}

void ktask_end(KernelTaskStatus status) {
    g_current_task.status = status;
    g_current_task.elapsed_ms = now_ms() - g_current_task.start_ms;
}

const KernelTask *ktask_current(void) {
    return &g_current_task;
}

const char *ktask_type_name(KernelTaskType type) {
    switch (type) {
        case KTASK_NONE: return "NONE";
        case KTASK_TOKEN_EMBEDDING: return "TOKEN_EMBEDDING";
        case KTASK_LAYER_FORWARD: return "LAYER_FORWARD";
        case KTASK_EXPERT_LOAD: return "EXPERT_LOAD";
        case KTASK_KV_CACHE_OPS: return "KV_CACHE_OPS";
        case KTASK_OUTPUT_NORM: return "OUTPUT_NORM";
        case KTASK_SAMPLING: return "SAMPLING";
        default: return "UNKNOWN";
    }
}

const char *ktask_status_name(KernelTaskStatus status) {
    switch (status) {
        case KTASK_STATUS_PENDING: return "PENDING";
        case KTASK_STATUS_RUNNING: return "RUNNING";
        case KTASK_STATUS_COMPLETED: return "COMPLETED";
        case KTASK_STATUS_FAILED: return "FAILED";
        default: return "UNKNOWN";
    }
}
