#include "hooks.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

typedef struct {
    bread_hook_fn fn;
    void *userdata;
} HookEntry;

static HookEntry g_hooks[BREAD_HOOK_COUNT] = {0};

/* Built-in hook: NaN/Inf checker */
static float *g_nan_check_buffer = NULL;
static size_t g_nan_check_buffer_size = 0;

static void bread_hook_nan_check_impl(const bread_hook_event_t *e, void *userdata) {
    (void)userdata;
    if (!e->d_hidden) {
        return;
    }

    /* Only check on POST_LAYER hook to reduce overhead */
    if (e->type != BREAD_HOOK_POST_LAYER) {
        return;
    }

    /* For now, just report that we would check. Full implementation would:
       1. Allocate host buffer if needed
       2. Copy d_hidden to host
       3. Scan for NaN/Inf
       4. Report if found
    */
    fprintf(stderr, "[HOOK_NAN_CHECK] Layer %d, token %d: (disabled for demo)\n",
            e->layer_idx, e->token_pos);
}

/* Built-in hook: Layer timing */
#define MAX_LAYER_TIMING 40
static double g_layer_times[MAX_LAYER_TIMING] = {0};
static int g_layer_timing_enabled = 0;
static double g_last_layer_time_start = 0.0;

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

static void bread_hook_layer_timing_impl(const bread_hook_event_t *e, void *userdata) {
    (void)userdata;

    if (e->type == BREAD_HOOK_PRE_LAYER) {
        g_last_layer_time_start = now_ms();
    } else if (e->type == BREAD_HOOK_POST_LAYER && e->layer_idx >= 0 && e->layer_idx < MAX_LAYER_TIMING) {
        double elapsed = now_ms() - g_last_layer_time_start;
        g_layer_times[e->layer_idx] = elapsed;
    }
}

void bread_register_hook(bread_hook_type_t type, bread_hook_fn fn, void *userdata) {
    if (type < BREAD_HOOK_COUNT) {
        g_hooks[type].fn = fn;
        g_hooks[type].userdata = userdata;
    }
}

void bread_unregister_hook(bread_hook_type_t type) {
    if (type < BREAD_HOOK_COUNT) {
        g_hooks[type].fn = NULL;
        g_hooks[type].userdata = NULL;
    }
}

void bread_fire_hook(bread_hook_type_t type, int token_pos, int layer_idx,
                     const void *d_hidden, double elapsed_ms) {
    if (type >= BREAD_HOOK_COUNT) {
        return;
    }

    HookEntry *entry = &g_hooks[type];
    if (!entry->fn) {
        return;
    }

    bread_hook_event_t event;
    event.type = type;
    event.token_pos = token_pos;
    event.layer_idx = layer_idx;
    event.d_hidden = d_hidden;
    event.elapsed_ms = elapsed_ms;

    (*entry->fn)(&event, entry->userdata);
}

void bread_hooks_enable_nan_check(void) {
    bread_register_hook(BREAD_HOOK_POST_LAYER, bread_hook_nan_check_impl, NULL);
    fprintf(stderr, "[HOOKS] Enabled NaN/Inf checker\n");
}

void bread_hooks_enable_layer_timing(void) {
    memset(g_layer_times, 0, sizeof(g_layer_times));
    g_layer_timing_enabled = 1;
    bread_register_hook(BREAD_HOOK_PRE_LAYER, bread_hook_layer_timing_impl, NULL);
    bread_register_hook(BREAD_HOOK_POST_LAYER, bread_hook_layer_timing_impl, NULL);
    fprintf(stderr, "[HOOKS] Enabled layer timing\n");
}

void bread_hooks_disable_all(void) {
    for (int i = 0; i < BREAD_HOOK_COUNT; i++) {
        bread_unregister_hook((bread_hook_type_t)i);
    }
    g_layer_timing_enabled = 0;
    fprintf(stderr, "[HOOKS] All hooks disabled\n");
}

void bread_hooks_report_layer_timing(void) {
    if (!g_layer_timing_enabled) {
        return;
    }

    fprintf(stderr, "\n[LAYER_TIMING] Per-layer forward times:\n");
    double total = 0.0;
    for (int i = 0; i < MAX_LAYER_TIMING; i++) {
        if (g_layer_times[i] > 0.0) {
            fprintf(stderr, "  Layer %2d: %.2f ms\n", i, g_layer_times[i]);
            total += g_layer_times[i];
        }
    }
    fprintf(stderr, "  Total:     %.2f ms\n", total);
}

const char *bread_hook_type_name(bread_hook_type_t type) {
    switch (type) {
        case BREAD_HOOK_PRE_TOKEN: return "PRE_TOKEN";
        case BREAD_HOOK_POST_TOKEN: return "POST_TOKEN";
        case BREAD_HOOK_PRE_LAYER: return "PRE_LAYER";
        case BREAD_HOOK_POST_LAYER: return "POST_LAYER";
        case BREAD_HOOK_PRE_SAMPLE: return "PRE_SAMPLE";
        case BREAD_HOOK_POST_SAMPLE: return "POST_SAMPLE";
        default: return "UNKNOWN";
    }
}
