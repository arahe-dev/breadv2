#ifndef PROGRESS_TRACKING_H
#define PROGRESS_TRACKING_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BREAD_PROGRESS_LOADING = 0,
    BREAD_PROGRESS_PREFILL,
    BREAD_PROGRESS_DECODE,
    BREAD_PROGRESS_DONE,
} bread_progress_phase_t;

typedef struct {
    bread_progress_phase_t phase;
    int    token_idx;        /* current token index */
    int    total_tokens;     /* total tokens expected */
    double elapsed_ms;       /* time elapsed for this phase */
    double tok_per_s;        /* tokens per second */
    int    layer_idx;        /* -1 if not layer-specific */
} bread_progress_t;

typedef void (*bread_progress_fn)(const bread_progress_t *p, void *userdata);

/* Set progress callback (NULL to disable) */
void bread_set_progress_callback(bread_progress_fn fn, void *userdata);

/* Report progress */
void bread_progress_report(bread_progress_phase_t phase, int token_idx,
                           int total_tokens, double elapsed_ms,
                           double tok_per_s, int layer_idx);

/* Get phase name */
const char *bread_progress_phase_name(bread_progress_phase_t phase);

/* Initialize default progress callback */
void bread_progress_init_default(void);

#ifdef __cplusplus
}
#endif

#endif /* PROGRESS_TRACKING_H */
