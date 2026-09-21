#ifndef LAYA_GGUF_H
#define LAYA_GGUF_H

/*
 * Minimal GGUF v3 reader for the o1.c materialized weight pack.
 *
 * Scope: parse the header (magic/version/tensor_count/KV count), the tensor
 * info table (name/shape/type/offset), and stream tensor payloads. No
 * ggml/llama.cpp dependency. The pack is produced by tools/hidream_convert.py
 * with general.alignment = 256 and BF16 tensors in production order.
 *
 * GGUF layout (v3, little-endian):
 *   [u32 magic "GGUF"][u32 version][u64 tensor_count][u64 kv_count]
 *   [kv pairs...][tensor infos...][pad to alignment][tensor data...]
 */

#include <stddef.h>
#include <stdint.h>

#include "laya.h"

/* ggml_type values we accept. */
#define LAYA_GGML_TYPE_F32 0
#define LAYA_GGML_TYPE_F16 1
#define LAYA_GGML_TYPE_BF16 30

/* Human readable name for a ggml_type, for error messages. */
const char *laya_ggml_type_name(uint32_t type);

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t dims[8];
    uint32_t type;        /* ggml_type */
    uint64_t offset;      /* relative to tensor_data */
    uint64_t nbytes;      /* payload bytes for this tensor */
} laya_gguf_tensor;

/*
 * Captured metadata entry. Only keys with a `laya.` / `modernbert.` prefix are
 * retained (everything else is skipped, exactly like o1.c). Scalars land in
 * i64/f64/str; arrays keep their element type in `elem_type` and their values
 * in vals/strs, so the model layer can read the whole config without a JSON
 * sidecar and without re-parsing at every startup.
 */
typedef struct {
    char *key;
    uint32_t type;        /* GGUF metadata type */
    uint32_t elem_type;   /* array element type (type == 9) */
    char *str;
    int64_t i64;
    double f64;
    uint64_t n;
    double *vals;         /* numeric arrays */
    char **strs;          /* string arrays */
} laya_gguf_kv;

typedef struct {
    char *path;
    int64_t n_tensors;
    laya_gguf_tensor *tensors;  /* in file order (production order) */
    uint64_t alignment;
    uint64_t tensor_data_off; /* absolute file offset of tensor_data */
    uint64_t payload_bytes;
    char *arch;               /* general.architecture */
    char *profile;            /* hidream.profile */
    char *revision;           /* hidream.revision */
    char *dtype;              /* hidream.dtype */
    int64_t num_layers;       /* hidream.num_layers */

    laya_gguf_kv *kvs;          /* retained laya.* / modernbert.* metadata */
    int64_t n_kvs;
} laya_gguf_file;

/* Parses the GGUF header + tensor table. No payload I/O. */
laya_status laya_gguf_open(const char *path, laya_gguf_file *out);
void laya_gguf_close(laya_gguf_file *f);

/* Reads one tensor's payload into dst (must be >= nbytes). */
laya_status laya_gguf_read_tensor(const laya_gguf_file *f, const laya_gguf_tensor *t,
                              void *dst);

/* Metadata lookup over the retained laya_gguf_kv table. */
const laya_gguf_kv *laya_gguf_find_kv(const laya_gguf_file *f, const char *key);

/* Scalar accessors; return the fallback when the key is absent or the type
 * does not match (so callers need no type checks for optional keys). */
int64_t laya_gguf_kv_int(const laya_gguf_file *f, const char *key, int64_t fallback);
double laya_gguf_kv_double(const laya_gguf_file *f, const char *key, double fallback);
const char *laya_gguf_kv_str(const laya_gguf_file *f, const char *key);

/* Last error string set by this module. */
const char *laya_gguf_last_error(void);

#endif /* LAYA_GGUF_H */