#ifndef KERNEL_TASKS_H
#define KERNEL_TASKS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    KTASK_NONE = 0,
    KTASK_TOKEN_EMBEDDING,
    KTASK_LAYER_FORWARD,
    KTASK_EXPERT_LOAD,
    KTASK_KV_CACHE_OPS,
    KTASK_OUTPUT_NORM,
    KTASK_SAMPLING,
} KernelTaskType;

typedef enum {
    KTASK_STATUS_PENDING,
    KTASK_STATUS_RUNNING,
    KTASK_STATUS_COMPLETED,
    KTASK_STATUS_FAILED,
} KernelTaskStatus;

typedef struct {
    KernelTaskType  type;
    KernelTaskStatus status;
    int             layer_idx;   /* -1 if N/A */
    int             token_pos;   /* -1 if N/A */
    double          start_ms;
    double          elapsed_ms;
    char            label[64];
} KernelTask;

/* Begin a new task (replaces current task) */
void ktask_begin(KernelTaskType type, int layer_idx, int token_pos);

/* End current task with status */
void ktask_end(KernelTaskStatus status);

/* Get current task (read-only) */
const KernelTask *ktask_current(void);

/* Get task type name */
const char *ktask_type_name(KernelTaskType type);

/* Get status name */
const char *ktask_status_name(KernelTaskStatus status);

#ifdef __cplusplus
}
#endif

#endif /* KERNEL_TASKS_H */
