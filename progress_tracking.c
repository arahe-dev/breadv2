#include "progress_tracking.h"
#include <stdio.h>

static bread_progress_fn g_progress_fn = NULL;
static void *g_progress_userdata = NULL;

/* Default progress callback */
static void bread_progress_default_callback(const bread_progress_t *p, void *userdata) {
    (void)userdata;
    const char *phase_str = bread_progress_phase_name(p->phase);
    fprintf(stderr, "[PROGRESS] %s: token %d/%d %.2f tok/s\n",
            phase_str, p->token_idx, p->total_tokens, p->tok_per_s);
}

void bread_set_progress_callback(bread_progress_fn fn, void *userdata) {
    if (fn) {
        g_progress_fn = fn;
        g_progress_userdata = userdata;
    } else {
        g_progress_fn = NULL;
        g_progress_userdata = NULL;
    }
}

void bread_progress_report(bread_progress_phase_t phase, int token_idx,
                           int total_tokens, double elapsed_ms,
                           double tok_per_s, int layer_idx) {
    if (!g_progress_fn) {
        return; /* callback disabled */
    }

    bread_progress_t p;
    p.phase = phase;
    p.token_idx = token_idx;
    p.total_tokens = total_tokens;
    p.elapsed_ms = elapsed_ms;
    p.tok_per_s = tok_per_s;
    p.layer_idx = layer_idx;

    (*g_progress_fn)(&p, g_progress_userdata);
}

const char *bread_progress_phase_name(bread_progress_phase_t phase) {
    switch (phase) {
        case BREAD_PROGRESS_LOADING: return "LOADING";
        case BREAD_PROGRESS_PREFILL: return "PREFILL";
        case BREAD_PROGRESS_DECODE: return "DECODE";
        case BREAD_PROGRESS_DONE: return "DONE";
        default: return "UNKNOWN";
    }
}

void bread_progress_init_default(void) {
    bread_set_progress_callback(bread_progress_default_callback, NULL);
}
