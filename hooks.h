#ifndef HOOKS_H
#define HOOKS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BREAD_HOOK_PRE_TOKEN = 0,
    BREAD_HOOK_POST_TOKEN,
    BREAD_HOOK_PRE_LAYER,
    BREAD_HOOK_POST_LAYER,
    BREAD_HOOK_PRE_SAMPLE,
    BREAD_HOOK_POST_SAMPLE,
    BREAD_HOOK_COUNT
} bread_hook_type_t;

typedef struct {
    bread_hook_type_t type;
    int               token_pos;
    int               layer_idx;   /* -1 if N/A */
    const void       *d_hidden;    /* device pointer (half*) */
    double            elapsed_ms;  /* cumulative time for this token */
} bread_hook_event_t;

typedef void (*bread_hook_fn)(const bread_hook_event_t *e, void *userdata);

/* Register a hook function */
void bread_register_hook(bread_hook_type_t type, bread_hook_fn fn, void *userdata);

/* Unregister a hook */
void bread_unregister_hook(bread_hook_type_t type);

/* Fire a hook (calls the registered function if any) */
void bread_fire_hook(bread_hook_type_t type, int token_pos, int layer_idx,
                     const void *d_hidden, double elapsed_ms);

/* Enable built-in NaN/Inf checker hook */
void bread_hooks_enable_nan_check(void);

/* Enable built-in layer timing hook */
void bread_hooks_enable_layer_timing(void);

/* Disable built-in hooks */
void bread_hooks_disable_all(void);

/* Get hook type name */
const char *bread_hook_type_name(bread_hook_type_t type);

/* Report layer timing after inference (only if enabled) */
void bread_hooks_report_layer_timing(void);

#ifdef __cplusplus
}
#endif

#endif /* HOOKS_H */
