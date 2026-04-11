#ifndef ERROR_CLASSIFICATION_H
#define ERROR_CLASSIFICATION_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BREAD_OK = 0,
    BREAD_ERR_OOM,
    BREAD_ERR_IO,
    BREAD_ERR_CUDA,
    BREAD_ERR_TENSOR_NOT_FOUND,
    BREAD_ERR_KV_CACHE_FULL,
    BREAD_ERR_INVALID_ARG,
    BREAD_ERR_UNKNOWN,
} bread_err_t;

typedef enum {
    BREAD_ERR_CAT_NONE = 0,
    BREAD_ERR_CAT_RETRYABLE,
    BREAD_ERR_CAT_FATAL,
    BREAD_ERR_CAT_RESOURCE,
} bread_err_category_t;

/* Classify an error into a category */
bread_err_category_t bread_classify_error(bread_err_t err);

/* Set the last error with context */
void bread_set_last_error(bread_err_t err, const char *context);

/* Get the last error */
bread_err_t bread_get_last_error(void);

/* Get the context of the last error (human-readable message) */
const char *bread_get_last_error_context(void);

/* Clear the last error */
void bread_clear_error(void);

/* Get string name of error code */
const char *bread_err_name(bread_err_t err);

/* Get string name of error category */
const char *bread_err_category_name(bread_err_category_t cat);

#ifdef __cplusplus
}
#endif

#endif /* ERROR_CLASSIFICATION_H */
