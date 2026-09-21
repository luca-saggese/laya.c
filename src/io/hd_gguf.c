#define _POSIX_C_SOURCE 200809L
#include "hd_gguf.h"
#include "laya_timing.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

static char g_gguf_error[512] = "";

static void gguf_err(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_gguf_error, sizeof(g_gguf_error), fmt, ap);
    va_end(ap);
}

const char *laya_gguf_last_error(void) { return g_gguf_error; }

/* ---- little-endian readers ---- */

static uint64_t rd_u64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (i * 8);
    return v;
}

static uint32_t rd_u32(const uint8_t *p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) v |= (uint32_t)p[i] << (i * 8);
    return v;
}

/* ---- cursor over a memory buffer ---- */

typedef struct {
    const uint8_t *base;
    size_t len;
    size_t pos;
    int need_more; /* set when a bounds check failed (buffer too short) */
} gguf_buf;

/* Reads a GGUF string (u64 len + bytes). Returns 0 on success, -1 on
 * bounds failure (need_more set) or allocation failure. */
static int buf_str(gguf_buf *b, char **out) {
    if (b->pos + 8 > b->len) { b->need_more = 1; return -1; }
    uint64_t n = rd_u64(b->base + b->pos);
    b->pos += 8;
    if (n > (uint64_t)1 << 20) return -1;
    if (b->pos + n > b->len) { b->need_more = 1; return -1; }
    char *s = malloc((size_t)n + 1);
    if (!s) return -1;
    memcpy(s, b->base + b->pos, (size_t)n);
    s[n] = '\0';
    b->pos += (size_t)n;
    *out = s;
    return 0;
}

/* Skips a metadata value of the given type. */
static int buf_skip_value(gguf_buf *b, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: b->pos += 1; return 0;   /* uint8/int8/bool */
        case 2: case 3: b->pos += 2; return 0;           /* uint16/int16 */
        case 4: case 5: case 6: b->pos += 4; return 0;   /* uint32/int32/f32 */
        case 8: { /* string */
            char *tmp = NULL;
            int rc = buf_str(b, &tmp);
            free(tmp);
            return rc;
        }
        case 9: { /* array */
            if (b->pos + 12 > b->len) { b->need_more = 1; return -1; }
            uint32_t atype = rd_u32(b->base + b->pos);
            uint64_t n = rd_u64(b->base + b->pos + 4);
            b->pos += 12;
            for (uint64_t i = 0; i < n; i++) {
                if (buf_skip_value(b, atype) != 0) return -1;
            }
            return 0;
        }
        case 10: case 11: case 12: b->pos += 8; return 0; /* uint64/int64/f64 */
        default: return -1;
    }
}

/* Dry-run: advances the cursor over the KV section and the tensor info
 * table without allocating. Returns 0 when the whole header fits in the
 * buffer, -1 when more data is needed (need_more set). */
static int gguf_dry_parse(gguf_buf *b, uint64_t n_kv, uint64_t n_tensors) {
    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = NULL;
        if (buf_str(b, &key) != 0) return -1;
        free(key);
        if (b->pos + 4 > b->len) { b->need_more = 1; return -1; }
        uint32_t vtype = rd_u32(b->base + b->pos);
        b->pos += 4;
        if (buf_skip_value(b, vtype) != 0) return -1;
    }
    for (uint64_t i = 0; i < n_tensors; i++) {
        char *name = NULL;
        if (buf_str(b, &name) != 0) return -1;
        free(name);
        if (b->pos + 4 > b->len) { b->need_more = 1; return -1; }
        uint32_t nd = rd_u32(b->base + b->pos);
        b->pos += 4;
        if (nd > 8) return -2;
        if (b->pos + (size_t)nd * 8 > b->len) { b->need_more = 1; return -1; }
        b->pos += (size_t)nd * 8;
        if (b->pos + 4 + 8 > b->len) { b->need_more = 1; return -1; }
        b->pos += 4 + 8; /* type + offset */
    }
    return 0;
}

/* Appends one retained KV (laya.* / modernbert.* only). */
static int kv_push(laya_gguf_file *out, laya_gguf_kv *kv) {
    laya_gguf_kv *ns = realloc(out->kvs, (size_t)(out->n_kvs + 1) * sizeof(*ns));
    if (!ns) return -1;
    out->kvs = ns;
    out->kvs[out->n_kvs++] = *kv;
    return 0;
}

/* Reads a metadata value of a known type into kv. Returns 0 on success. */
static int buf_read_value(gguf_buf *b, uint32_t type, laya_gguf_kv *kv) {
    kv->type = type;
    switch (type) {
        case 0: case 7: { /* uint8 / bool */
            if (b->pos + 1 > b->len) { b->need_more = 1; return -1; }
            kv->i64 = b->base[b->pos];
            b->pos += 1;
            return 0;
        }
        case 1: { /* int8 */
            if (b->pos + 1 > b->len) { b->need_more = 1; return -1; }
            kv->i64 = (int8_t)b->base[b->pos];
            b->pos += 1;
            return 0;
        }
        case 2: { /* uint16 */
            if (b->pos + 2 > b->len) { b->need_more = 1; return -1; }
            kv->i64 = (uint16_t)(b->base[b->pos] | (b->base[b->pos + 1] << 8));
            b->pos += 2;
            return 0;
        }
        case 3: { /* int16 */
            if (b->pos + 2 > b->len) { b->need_more = 1; return -1; }
            kv->i64 = (int16_t)(b->base[b->pos] | (b->base[b->pos + 1] << 8));
            b->pos += 2;
            return 0;
        }
        case 4: case 5: { /* uint32 / int32 */
            if (b->pos + 4 > b->len) { b->need_more = 1; return -1; }
            uint32_t v = rd_u32(b->base + b->pos);
            kv->i64 = (type == 4) ? (int64_t)v : (int64_t)(int32_t)v;
            b->pos += 4;
            return 0;
        }
        case 6: { /* f32 */
            if (b->pos + 4 > b->len) { b->need_more = 1; return -1; }
            uint32_t v = rd_u32(b->base + b->pos);
            float f;
            memcpy(&f, &v, 4);
            kv->f64 = f;
            b->pos += 4;
            return 0;
        }
        case 8: { /* string */
            char *s = NULL;
            if (buf_str(b, &s) != 0) return -1;
            kv->str = s;
            return 0;
        }
        case 10: case 11: { /* uint64 / int64 */
            if (b->pos + 8 > b->len) { b->need_more = 1; return -1; }
            kv->i64 = (int64_t)rd_u64(b->base + b->pos);
            b->pos += 8;
            return 0;
        }
        case 12: { /* f64 */
            if (b->pos + 8 > b->len) { b->need_more = 1; return -1; }
            uint64_t v = rd_u64(b->base + b->pos);
            memcpy(&kv->f64, &v, 8);
            b->pos += 8;
            return 0;
        }
        case 9: { /* array */
            if (b->pos + 12 > b->len) { b->need_more = 1; return -1; }
            uint32_t atype = rd_u32(b->base + b->pos);
            uint64_t n = rd_u64(b->base + b->pos + 4);
            b->pos += 12;
            kv->elem_type = atype;
            kv->n = n;
            if (atype == 8) {
                kv->strs = calloc((size_t)n + 1, sizeof(char *));
                if (!kv->strs) return -1;
                for (uint64_t i = 0; i < n; i++) {
                    if (buf_str(b, &kv->strs[i]) != 0) return -1;
                }
                return 0;
            }
            kv->vals = calloc((size_t)n + 1, sizeof(double));
            if (!kv->vals) return -1;
            for (uint64_t i = 0; i < n; i++) {
                laya_gguf_kv tmp;
                memset(&tmp, 0, sizeof(tmp));
                if (buf_read_value(b, atype, &tmp) != 0) return -1;
                kv->vals[i] = (atype == 6 || atype == 12) ? tmp.f64 : (double)tmp.i64;
                free(tmp.str);
            }
            return 0;
        }
        default: return -1;
    }
}

static int is_retained_key(const char *key) {
    return strncmp(key, "laya.", 5) == 0 || strncmp(key, "modernbert.", 11) == 0;
}

/* Real parse: populates out from the (complete) header buffer. */
static int gguf_real_parse(gguf_buf *b, uint64_t n_kv, uint64_t n_tensors,
                           laya_gguf_file *out) {
    out->alignment = 32; /* GGUF default */
    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = NULL;
        if (buf_str(b, &key) != 0) return -1;
        if (b->pos + 4 > b->len) { free(key); b->need_more = 1; return -1; }
        uint32_t vtype = rd_u32(b->base + b->pos);
        b->pos += 4;
        if (strcmp(key, "general.alignment") == 0 && vtype == 4) {
            if (b->pos + 4 > b->len) { free(key); b->need_more = 1; return -1; }
            out->alignment = rd_u32(b->base + b->pos);
            b->pos += 4;
        } else if (strcmp(key, "general.architecture") == 0 && vtype == 8) {
            char *v = NULL;
            if (buf_str(b, &v) != 0) { free(key); return -1; }
            out->arch = v;
        } else if (strcmp(key, "hidream.profile") == 0 && vtype == 8) {
            char *v = NULL;
            if (buf_str(b, &v) != 0) { free(key); return -1; }
            out->profile = v;
        } else if (strcmp(key, "hidream.revision") == 0 && vtype == 8) {
            char *v = NULL;
            if (buf_str(b, &v) != 0) { free(key); return -1; }
            out->revision = v;
        } else if (strcmp(key, "hidream.dtype") == 0 && vtype == 8) {
            char *v = NULL;
            if (buf_str(b, &v) != 0) { free(key); return -1; }
            out->dtype = v;
        } else if (strcmp(key, "hidream.num_layers") == 0 && vtype == 8) {
            char *v = NULL;
            if (buf_str(b, &v) != 0) { free(key); return -1; }
            out->num_layers = atoll(v ? v : "0");
            free(v);
        } else if (is_retained_key(key)) {
            laya_gguf_kv kv;
            memset(&kv, 0, sizeof(kv));
            kv.key = key;
            if (buf_read_value(b, vtype, &kv) != 0) {
                free(kv.key); free(kv.str); free(kv.vals);
                if (kv.strs) {
                    for (uint64_t k = 0; k < kv.n; k++) free(kv.strs[k]);
                    free(kv.strs);
                }
                return -1;
            }
            if (kv_push(out, &kv) != 0) { free(key); return -1; }
            continue; /* key ownership moved into the KV */
        } else {
            if (buf_skip_value(b, vtype) != 0) { free(key); return -1; }
        }
        free(key);
    }

    out->tensors = calloc((size_t)n_tensors, sizeof(laya_gguf_tensor));
    if (!out->tensors) return -1;
    out->n_tensors = (int64_t)n_tensors;

    for (uint64_t i = 0; i < n_tensors; i++) {
        laya_gguf_tensor *t = &out->tensors[i];
        if (buf_str(b, &t->name) != 0) return -1;
        if (b->pos + 4 > b->len) { b->need_more = 1; return -1; }
        t->n_dims = rd_u32(b->base + b->pos);
        b->pos += 4;
        if (t->n_dims > 8) return -2;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (b->pos + 8 > b->len) { b->need_more = 1; return -1; }
            t->dims[d] = rd_u64(b->base + b->pos);
            b->pos += 8;
        }
        if (b->pos + 4 > b->len) { b->need_more = 1; return -1; }
        t->type = rd_u32(b->base + b->pos);
        b->pos += 4;
        if (b->pos + 8 > b->len) { b->need_more = 1; return -1; }
        t->offset = rd_u64(b->base + b->pos);
        b->pos += 8;

        uint64_t nbytes = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) nbytes *= t->dims[d];
        switch (t->type) {
            case LAYA_GGML_TYPE_F32: nbytes *= 4; break;
            case LAYA_GGML_TYPE_F16: nbytes *= 2; break;
            case LAYA_GGML_TYPE_BF16: nbytes *= 2; break;
            default: return -2;
        }
        t->nbytes = nbytes;
    }
    return 0;
}

laya_status laya_gguf_open(const char *path, laya_gguf_file *out) {
    LAYA_TIMING_BEGIN("GGUF_PARSE");
    memset(out, 0, sizeof(*out));
    out->path = strdup(path);
    if (!out->path) { gguf_err("oom path"); return LAYA_ERR_OOM; }

    FILE *f = fopen(path, "rb");
    if (!f) { gguf_err("cannot open %s", path); return LAYA_ERR_IO; }

    uint8_t hdr[24];
    if (fread(hdr, 1, 24, f) != 24) {
        fclose(f); gguf_err("short header %s", path); return LAYA_ERR_IO;
    }
    if (rd_u32(hdr) != 0x46554747u) {
        fclose(f); gguf_err("%s: bad GGUF magic", path); return LAYA_ERR_PARSE;
    }
    uint32_t version = rd_u32(hdr + 4);
    if (version != 3) {
        fclose(f); gguf_err("%s: unsupported GGUF version %u", path, version);
        return LAYA_ERR_PARSE;
    }
    uint64_t n_tensors = rd_u64(hdr + 8);
    uint64_t n_kv = rd_u64(hdr + 16);
    if (n_tensors == 0 || n_tensors > (uint64_t)1 << 20) {
        fclose(f); gguf_err("%s: implausible tensor count %llu", path,
                            (unsigned long long)n_tensors);
        return LAYA_ERR_PARSE;
    }

    /* Read the header region (KV + tensor infos) into a growing buffer.
     * First do a dry parse to learn when the header is complete, then a
     * real parse. The header is small (a few MB for 759 tensors). */
    size_t cap = 1 << 20;
    uint8_t *buf = malloc(cap);
    if (!buf) { fclose(f); gguf_err("oom header"); return LAYA_ERR_OOM; }
    size_t used = 0;

    laya_status st = LAYA_OK;
    int header_done = 0;
    while (!header_done) {
        if (used == cap) {
            cap *= 2;
            uint8_t *nb = realloc(buf, cap);
            if (!nb) { free(buf); fclose(f); gguf_err("oom header"); return LAYA_ERR_OOM; }
            buf = nb;
        }
        size_t rd = fread(buf + used, 1, cap - used, f);
        used += rd;
        if (rd == 0 && used == 0) { free(buf); fclose(f); gguf_err("empty header"); return LAYA_ERR_IO; }

        gguf_buf b = {buf, used, 0, 0};
        int rc = gguf_dry_parse(&b, n_kv, n_tensors);
        if (rc == 0) {
            header_done = 1;
        } else if (rc == -2) {
            free(buf); fclose(f); gguf_err("%s: malformed header", path);
            return LAYA_ERR_PARSE;
        } else if (b.need_more) {
            continue; /* grow and retry */
        } else {
            free(buf); fclose(f); gguf_err("%s: malformed header", path);
            return LAYA_ERR_PARSE;
        }
    }

    /* Real parse on the complete buffer. */
    gguf_buf b = {buf, used, 0, 0};
    int rc = gguf_real_parse(&b, n_kv, n_tensors, out);
    if (rc != 0) {
        laya_gguf_close(out);
        free(buf); fclose(f);
        gguf_err("%s: header parse failed", path);
        return LAYA_ERR_PARSE;
    }

    /* tensor_data starts after the header, padded to alignment. The file
     * pointer is ahead of the real header end (we read in 1 MiB blocks), so
     * compute the offset from the parse cursor instead of ftell. */
    uint64_t align = out->alignment ? out->alignment : 32;
    out->tensor_data_off = (24 + b.pos + align - 1) & ~(align - 1);

    /* payload bytes = max tensor offset + size, aligned. */
    uint64_t payload = 0;
    for (int64_t i = 0; i < out->n_tensors; i++) {
        uint64_t end_off = out->tensors[i].offset + out->tensors[i].nbytes;
        if (end_off > payload) payload = end_off;
    }
    out->payload_bytes = (payload + align - 1) & ~(align - 1);

    free(buf);
    fclose(f);
    LAYA_TIMING_END("GGUF_PARSE");
    return LAYA_OK;
}

void laya_gguf_close(laya_gguf_file *f) {
    if (!f) return;
    free(f->path);
    free(f->arch);
    free(f->profile);
    free(f->revision);
    free(f->dtype);
    for (int64_t i = 0; i < f->n_tensors; i++) free(f->tensors[i].name);
    free(f->tensors);
    for (int64_t i = 0; i < f->n_kvs; i++) {
        laya_gguf_kv *kv = &f->kvs[i];
        free(kv->key);
        free(kv->str);
        free(kv->vals);
        if (kv->strs) {
            for (uint64_t k = 0; k < kv->n; k++) free(kv->strs[k]);
            free(kv->strs);
        }
    }
    free(f->kvs);
    memset(f, 0, sizeof(*f));
}

const laya_gguf_kv *laya_gguf_find_kv(const laya_gguf_file *f, const char *key) {
    if (!f || !f->kvs) return NULL;
    for (int64_t i = 0; i < f->n_kvs; i++) {
        if (strcmp(f->kvs[i].key, key) == 0) return &f->kvs[i];
    }
    return NULL;
}

int64_t laya_gguf_kv_int(const laya_gguf_file *f, const char *key, int64_t fallback) {
    const laya_gguf_kv *kv = laya_gguf_find_kv(f, key);
    if (!kv) return fallback;
    switch (kv->type) {
        case 0: case 1: case 2: case 3: case 4: case 5:
        case 7: case 10: case 11:
            return kv->i64;
        case 6: case 12:
            return (int64_t)kv->f64;
        default:
            return fallback;
    }
}

double laya_gguf_kv_double(const laya_gguf_file *f, const char *key, double fallback) {
    const laya_gguf_kv *kv = laya_gguf_find_kv(f, key);
    if (!kv) return fallback;
    switch (kv->type) {
        case 0: case 1: case 2: case 3: case 4: case 5:
        case 7: case 10: case 11:
            return (double)kv->i64;
        case 6: case 12:
            return kv->f64;
        default:
            return fallback;
    }
}

const char *laya_gguf_kv_str(const laya_gguf_file *f, const char *key) {
    const laya_gguf_kv *kv = laya_gguf_find_kv(f, key);
    if (!kv || kv->type != 8) return NULL;
    return kv->str;
}


laya_status laya_gguf_read_tensor(const laya_gguf_file *f, const laya_gguf_tensor *t,
                              void *dst) {
    FILE *fp = fopen(f->path, "rb");
    if (!fp) { gguf_err("cannot open %s", f->path); return LAYA_ERR_IO; }
    if (fseeko(fp, (off_t)(f->tensor_data_off + t->offset), SEEK_SET) != 0) {
        fclose(fp); gguf_err("seek %s", f->path); return LAYA_ERR_IO;
    }
    size_t rd = fread(dst, 1, (size_t)t->nbytes, fp);
    fclose(fp);
    if (rd != (size_t)t->nbytes) {
        gguf_err("short read %s (%zu/%llu)", t->name, rd,
                 (unsigned long long)t->nbytes);
        return LAYA_ERR_IO;
    }
    return LAYA_OK;
}
