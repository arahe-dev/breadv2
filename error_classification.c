#include "error_classification.h"
#include <string.h>

static bread_err_t g_last_error = BREAD_OK;
static char g_error_context[256] = {0};

bread_err_category_t bread_classify_error(bread_err_t err) {
    switch (err) {
        case BREAD_OK:
            return BREAD_ERR_CAT_NONE;

        case BREAD_ERR_OOM:
        case BREAD_ERR_KV_CACHE_FULL:
            return BREAD_ERR_CAT_RESOURCE;

        case BREAD_ERR_CUDA:
        case BREAD_ERR_TENSOR_NOT_FOUND:
        case BREAD_ERR_INVALID_ARG:
            return BREAD_ERR_CAT_FATAL;

        case BREAD_ERR_IO:
            return BREAD_ERR_CAT_RETRYABLE;

        case BREAD_ERR_UNKNOWN:
        default:
            return BREAD_ERR_CAT_FATAL;
    }
}

void bread_set_last_error(bread_err_t err, const char *context) {
    g_last_error = err;
    if (context) {
        strncpy(g_error_context, context, sizeof(g_error_context) - 1);
        g_error_context[sizeof(g_error_context) - 1] = '\0';
    } else {
        g_error_context[0] = '\0';
    }
}

bread_err_t bread_get_last_error(void) {
    return g_last_error;
}

const char *bread_get_last_error_context(void) {
    return g_error_context;
}

void bread_clear_error(void) {
    g_last_error = BREAD_OK;
    g_error_context[0] = '\0';
}

const char *bread_err_name(bread_err_t err) {
    switch (err) {
        case BREAD_OK: return "OK";
        case BREAD_ERR_OOM: return "OUT_OF_MEMORY";
        case BREAD_ERR_IO: return "IO_ERROR";
        case BREAD_ERR_CUDA: return "CUDA_ERROR";
        case BREAD_ERR_TENSOR_NOT_FOUND: return "TENSOR_NOT_FOUND";
        case BREAD_ERR_KV_CACHE_FULL: return "KV_CACHE_FULL";
        case BREAD_ERR_INVALID_ARG: return "INVALID_ARGUMENT";
        case BREAD_ERR_UNKNOWN: return "UNKNOWN";
        default: return "UNDEFINED";
    }
}

const char *bread_err_category_name(bread_err_category_t cat) {
    switch (cat) {
        case BREAD_ERR_CAT_NONE: return "NONE";
        case BREAD_ERR_CAT_RETRYABLE: return "RETRYABLE";
        case BREAD_ERR_CAT_FATAL: return "FATAL";
        case BREAD_ERR_CAT_RESOURCE: return "RESOURCE_CONSTRAINT";
        default: return "UNKNOWN";
    }
}
