#ifndef LAYA_H
#define LAYA_H

/*
 * Public C ABI for the Laya GB10 inference engine.
 *
 * This is the Laya port of the o1.c runtime: the status codes, dtype helpers
 * and error-string contract are copied unchanged from _reference/o1.c
 * (include/hidream.h). All functions return laya_status and set an error
 * string retrievable via laya_last_error(). Nothing here silently ignores
 * errors.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Status codes                                                        */
/* ------------------------------------------------------------------ */

typedef enum {
    LAYA_OK = 0,
    LAYA_ERR_IO = 1,        /* file read/stat failure */
    LAYA_ERR_PARSE = 2,     /* JSON/text parse failure */
    LAYA_ERR_PROFILE = 3,   /* unknown/invalid profile */
    LAYA_ERR_MANIFEST = 4,  /* tensor manifest structural problem */
    LAYA_ERR_MISMATCH = 5,  /* validation mismatch (counts, shapes, ...) */
    LAYA_ERR_MISSING = 6,   /* required tensor absent */
    LAYA_ERR_OOM = 7,       /* allocation failure */
    LAYA_ERR_RUNTIME = 8,   /* runtime/backend failure (e.g. cuDNN SDPA) */
    LAYA_ERR_UNSUPPORTED = 9, /* feature/target not supported (fails closed) */
} laya_status;

const char *laya_last_error(void);

/* Sets the shared error string (used by sibling modules). */
void laya_set_error(const char *fmt, ...);

/* ------------------------------------------------------------------ */
/* Dtypes                                                              */
/* ------------------------------------------------------------------ */

typedef enum {
    LAYA_DTYPE_F32 = 0,
    LAYA_DTYPE_BF16 = 1,
    LAYA_DTYPE_F16 = 2,
    LAYA_DTYPE_F64 = 3,
    LAYA_DTYPE_I64 = 4,
    LAYA_DTYPE_I32 = 5,
    LAYA_DTYPE_UNKNOWN = -1,
} laya_dtype;

laya_dtype laya_dtype_from_string(const char *s); /* accepts torch.* aliases */
const char *laya_dtype_to_string(laya_dtype d);

int64_t laya_numel(const int64_t *shape, int rank);

#ifdef __cplusplus
}
#endif

#endif /* LAYA_H */
