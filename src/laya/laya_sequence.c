#define _POSIX_C_SOURCE 200809L

/*
 * Direct C translation of the sequence-construction half of
 * _reference/laya/laya/common.py:
 *
 *   serialize_state, render_criterion, render_options, build_sequence
 *   plus Agent._to_internal from laya/agent.py.
 *
 * The control flow mirrors the Python line by line on purpose: the goal is
 * that `diff`ing the two files stays meaningful while parity is established.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "laya_sequence.h"
#include "laya_tokenizer.h"

/* ------------------------------------------------------------------ */
/* small helpers                                                       */
/* ------------------------------------------------------------------ */

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *r = malloc(n);
    if (r) memcpy(r, s, n);
    return r;
}

/* Python str.replace(old, new) for a single needle. */
static char *replace_all(const char *src, const char *needle, const char *with) {
    size_t nl = strlen(needle), wl = strlen(with);
    if (nl == 0) return xstrdup(src);
    size_t cap = strlen(src) + 16, len = 0;
    char *out = malloc(cap);
    if (!out) return NULL;
    const char *p = src;
    for (;;) {
        const char *hit = strstr(p, needle);
        size_t chunk = hit ? (size_t)(hit - p) : strlen(p);
        if (len + chunk + wl + 1 > cap) {
            cap = (len + chunk + wl + 1) * 2;
            char *grown = realloc(out, cap);
            if (!grown) {
                free(out);
                return NULL;
            }
            out = grown;
        }
        memcpy(out + len, p, chunk);
        len += chunk;
        if (!hit) break;
        memcpy(out + len, with, wl);
        len += wl;
        p = hit + nl;
    }
    out[len] = '\0';
    return out;
}

typedef struct {
    char *buf;
    size_t len, cap;
} sbuf;

static int sb_put(sbuf *b, const char *s, size_t n) {
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (cap < b->len + n + 1) cap *= 2;
        char *grown = realloc(b->buf, cap);
        if (!grown) return 0;
        b->buf = grown;
        b->cap = cap;
    }
    memcpy(b->buf + b->len, s, n);
    b->len += n;
    b->buf[b->len] = '\0';
    return 1;
}

static int sb_puts(sbuf *b, const char *s) { return sb_put(b, s, strlen(s)); }

static int sb_putc(sbuf *b, char c) { return sb_put(b, &c, 1); }

/* ------------------------------------------------------------------ */
/* JSON encoding                                                       */
/* ------------------------------------------------------------------ */

/* json.dumps string escaping with ensure_ascii=False. */
static int json_dump_string(sbuf *b, const char *s) {
    if (!sb_putc(b, '"')) return 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned c = *p;
        switch (c) {
        case '"':  if (!sb_puts(b, "\\\"")) return 0; break;
        case '\\': if (!sb_puts(b, "\\\\")) return 0; break;
        case '\b': if (!sb_puts(b, "\\b")) return 0; break;
        case '\f': if (!sb_puts(b, "\\f")) return 0; break;
        case '\n': if (!sb_puts(b, "\\n")) return 0; break;
        case '\r': if (!sb_puts(b, "\\r")) return 0; break;
        case '\t': if (!sb_puts(b, "\\t")) return 0; break;
        default:
            if (c < 0x20) {
                char tmp[8];
                snprintf(tmp, sizeof tmp, "\\u%04x", c);
                if (!sb_puts(b, tmp)) return 0;
            } else if (!sb_putc(b, (char)c)) {
                return 0;
            }
        }
    }
    return sb_putc(b, '"');
}

/* Python repr(float): shortest representation that round-trips. */
static void json_dump_double(sbuf *b, double v) {
    char tmp[64];
    if (isnan(v)) {
        sb_puts(b, "NaN");
        return;
    }
    if (isinf(v)) {
        sb_puts(b, v < 0 ? "-Infinity" : "Infinity");
        return;
    }
    for (int prec = 1; prec <= 17; prec++) {
        snprintf(tmp, sizeof tmp, "%.*g", prec, v);
        if (strtod(tmp, NULL) == v) break;
    }
    if (!strpbrk(tmp, ".eEnN")) {
        size_t n = strlen(tmp);
        snprintf(tmp + n, sizeof(tmp) - n, ".0");
    }
    sb_puts(b, tmp);
}

/* json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str) */
static int json_dump_value(sbuf *b, const hd_json *v) {
    if (!v) return sb_puts(b, "null");
    switch (v->type) {
    case HD_JSON_NULL:   return sb_puts(b, "null");
    case HD_JSON_BOOL:   return sb_puts(b, v->u.boolean ? "true" : "false");
    case HD_JSON_INT: {
        char tmp[32];
        snprintf(tmp, sizeof tmp, "%lld", (long long)v->u.integer);
        return sb_puts(b, tmp);
    }
    case HD_JSON_DOUBLE: json_dump_double(b, v->u.number); return 1;
    case HD_JSON_STRING: return json_dump_string(b, v->u.string ? v->u.string : "");
    case HD_JSON_ARRAY: {
        if (!sb_putc(b, '[')) return 0;
        for (size_t i = 0; i < v->u.array.count; i++) {
            if (i && !sb_puts(b, ", ")) return 0;
            if (!json_dump_value(b, v->u.array.items[i])) return 0;
        }
        return sb_putc(b, ']');
    }
    case HD_JSON_OBJECT: {
        if (!sb_putc(b, '{')) return 0;
        for (size_t i = 0; i < v->u.object.count; i++) {
            if (i && !sb_puts(b, ", ")) return 0;
            if (!json_dump_string(b, v->u.object.keys[i])) return 0;
            if (!sb_puts(b, ": ")) return 0;
            if (!json_dump_value(b, v->u.object.values[i])) return 0;
        }
        return sb_putc(b, '}');
    }
    }
    return sb_puts(b, "null");
}

/* ------------------------------------------------------------------ */
/* render_criterion / serialize_state / render_options                 */
/* ------------------------------------------------------------------ */

char *laya_render_criterion(const hd_json *value) {
    if (value && value->type == HD_JSON_STRING) return xstrdup(value->u.string);
    sbuf b = {0};
    if (!json_dump_value(&b, value)) {
        free(b.buf);
        return NULL;
    }
    return b.buf;
}

/* serialize_state: strings pass through, containers become JSON text. */
static char *serialize_state(const hd_json *state) {
    if (state && state->type == HD_JSON_STRING) return xstrdup(state->u.string);
    sbuf b = {0};
    if (!json_dump_value(&b, state)) {
        free(b.buf);
        return NULL;
    }
    return b.buf;
}

int laya_qtype_of(const char *type) {
    if (!type) return -1;
    if (strcmp(type, "choice") == 0) return LAYA_QTYPE_CHOICE;
    if (strcmp(type, "score") == 0) return LAYA_QTYPE_SCORE;
    if (strcmp(type, "noul") == 0) return LAYA_QTYPE_NOUL;
    return -1;
}

static int options_push(laya_options *o, char *text) {
    char **grown = realloc(o->opts, (o->n_opts + 1) * sizeof(char *));
    if (!grown) return 0;
    o->opts = grown;
    o->opts[o->n_opts++] = text;
    return 1;
}

void laya_render_options(const hd_json *q, laya_options *out) {
    memset(out, 0, sizeof *out);
    const hd_json *crit = hd_json_get(q, "crit");
    const char *t = hd_json_string(hd_json_get(q, "t"));

    if (t && strcmp(t, "choice") == 0) {
        /* [k if v is None or v == "" else "%s: %s" % (k, render_criterion(v))] */
        if (crit && crit->type == HD_JSON_OBJECT) {
            for (size_t i = 0; i < crit->u.object.count; i++) {
                const char *k = crit->u.object.keys[i];
                const hd_json *v = crit->u.object.values[i];
                int empty = !v || v->type == HD_JSON_NULL ||
                            (v->type == HD_JSON_STRING && v->u.string[0] == '\0');
                char *text;
                if (empty) {
                    text = xstrdup(k);
                } else {
                    char *rendered = laya_render_criterion(v);
                    size_t need = strlen(k) + strlen(rendered) + 3;
                    text = malloc(need);
                    if (text) snprintf(text, need, "%s: %s", k, rendered);
                    free(rendered);
                }
                if (!text || !options_push(out, text)) {
                    free(text);
                    return;
                }
            }
        }
        return;
    }

    if (t && strcmp(t, "score") == 0) {
        /* ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)] */
        if (crit && crit->type == HD_JSON_ARRAY) {
            for (size_t i = 0; i < crit->u.array.count; i++) {
                char *rendered = laya_render_criterion(crit->u.array.items[i]);
                size_t need = strlen(rendered) + 24;
                char *text = malloc(need);
                if (text) snprintf(text, need, "level %zu: %s", i, rendered);
                free(rendered);
                if (!text || !options_push(out, text)) {
                    free(text);
                    return;
                }
            }
        }
        return;
    }

    /* noul (and anything else): always [false, true] */
    {
        const hd_json *fc = crit && crit->type == HD_JSON_OBJECT ? hd_json_get(crit, "false") : NULL;
        const hd_json *tc = crit && crit->type == HD_JSON_OBJECT ? hd_json_get(crit, "true") : NULL;
        int f_empty = !fc || fc->type == HD_JSON_NULL ||
                      (fc->type == HD_JSON_STRING && fc->u.string[0] == '\0');
        int t_empty = !tc || tc->type == HD_JSON_NULL ||
                      (tc->type == HD_JSON_STRING && tc->u.string[0] == '\0');
        char *f = f_empty ? xstrdup("no, the statement does not hold") : laya_render_criterion(fc);
        char *tv = t_empty ? xstrdup("yes, the statement holds") : laya_render_criterion(tc);
        size_t nf = strlen(f) + 8, nt = strlen(tv) + 8;
        char *ftext = malloc(nf), *ttext = malloc(nt);
        if (ftext) snprintf(ftext, nf, "false: %s", f);
        if (ttext) snprintf(ttext, nt, "true: %s", tv);
        free(f);
        free(tv);
        if (!ftext || !ttext || !options_push(out, ftext) || !options_push(out, ttext)) {
            free(ftext);
            free(ttext);
            return;
        }
    }
}

void laya_options_free(laya_options *o) {
    if (!o) return;
    for (size_t i = 0; i < o->n_opts; i++) free(o->opts[i]);
    free(o->opts);
    if (o->opt_ids) {
        for (size_t i = 0; i < o->n_opts; i++) free(o->opt_ids[i]);
        free(o->opt_ids);
    }
    free(o->opt_len);
    memset(o, 0, sizeof *o);
}

/* ------------------------------------------------------------------ */
/* build_sequence                                                      */
/* ------------------------------------------------------------------ */

laya_status laya_build_sequence(const hd_json *state, const hd_json *q,
                                int max_len, int head_max_len,
                                int **out_ids, size_t *out_count,
                                int **out_markers, size_t *out_n_markers) {
    *out_ids = NULL;
    *out_count = 0;
    *out_markers = NULL;
    *out_n_markers = 0;

    const char *t = hd_json_string(hd_json_get(q, "t"));
    const char *ins_raw = hd_json_string(hd_json_get(q, "ins"));
    if (!t || !ins_raw) {
        laya_set_error("build_sequence: question missing 't'/'ins'");
        return LAYA_ERR_PARSE;
    }

    const char *mask_tok = "[MASK]";
    laya_status status = LAYA_OK;
    laya_options opts;
    laya_render_options(q, &opts);
    if (opts.n_opts == 0) {
        laya_options_free(&opts);
        laya_set_error("build_sequence: no options rendered");
        return LAYA_ERR_PARSE;
    }

    /* ins = str(q["ins"]).replace(mask_tok, " ") */
    char *ins = replace_all(ins_raw, mask_tok, " ");
    /* head text = "%s question: %s" % (q["t"], ins) */
    size_t head_need = strlen(t) + strlen(ins) + 16;
    char *head_text = malloc(head_need);
    if (head_text) snprintf(head_text, head_need, "%s question: %s", t, ins);
    free(ins);
    if (!head_text) {
        laya_options_free(&opts);
        laya_set_error("build_sequence: oom (head text)");
        return LAYA_ERR_OOM;
    }

    int *head_ids = NULL;
    size_t head_len = 0;
    if (laya_tokenizer_encode(head_text, &head_ids, &head_len) != LAYA_OK) {
        free(head_text);
        laya_options_free(&opts);
        return LAYA_ERR_PARSE;
    }
    free(head_text);

    /* opt_ids[i] = [mask] + tok(" " + opts[i].replace(mask_tok, " "))[:48] */
    opts.opt_ids = calloc(opts.n_opts, sizeof(int *));
    opts.opt_len = calloc(opts.n_opts, sizeof(size_t));
    if (!opts.opt_ids || !opts.opt_len) {
        status = LAYA_ERR_OOM;
        goto out;
    }
    for (size_t i = 0; i < opts.n_opts; i++) {
        char *clean = replace_all(opts.opts[i], mask_tok, " ");
        size_t need = strlen(clean) + 2;
        char *text = malloc(need);
        if (text) snprintf(text, need, " %s", clean);
        free(clean);
        int *body = NULL;
        size_t body_len = 0;
        if (!text || laya_tokenizer_encode(text, &body, &body_len) != LAYA_OK) {
            free(text);
            status = LAYA_ERR_PARSE;
            goto out;
        }
        free(text);
        if (body_len > 48) body_len = 48;
        int *row = malloc((body_len + 1) * sizeof(int));
        if (!row) {
            free(body);
            status = LAYA_ERR_OOM;
            goto out;
        }
        row[0] = LAYA_TOK_MASK;
        memcpy(row + 1, body, body_len * sizeof(int));
        free(body);
        opts.opt_ids[i] = row;
        opts.opt_len[i] = body_len + 1;
    }

    /* opt_budget = head_max_len - sum(len(o) for o in opt_ids) */
    long total = 0;
    for (size_t i = 0; i < opts.n_opts; i++) total += (long)opts.opt_len[i];
    long opt_budget = (long)head_max_len - total;
    if (opt_budget < 16) {
        long per = (head_max_len - 16) / (opts.n_opts > 0 ? (long)opts.n_opts : 1);
        if (per < 4) per = 4;
        total = 0;
        for (size_t i = 0; i < opts.n_opts; i++) {
            if ((long)opts.opt_len[i] > per) opts.opt_len[i] = (size_t)per;
            total += (long)opts.opt_len[i];
        }
        opt_budget = (long)head_max_len - total;
    }

    /* head_ids = head_ids[: max(8, opt_budget)] */
    long keep = opt_budget > 8 ? opt_budget : 8;
    if (keep < (long)head_len) head_len = (size_t)keep;

    /* ids = [cls] + head_ids + [sep] */
    size_t cap = head_len + 2 + 8;
    for (size_t i = 0; i < opts.n_opts; i++) cap += opts.opt_len[i] + 1;
    int *ids = malloc(cap * sizeof(int));
    int *markers = malloc((opts.n_opts + 1) * sizeof(int));
    if (!ids || !markers) {
        free(ids);
        free(markers);
        status = LAYA_ERR_OOM;
        goto out;
    }
    size_t n = 0;
    ids[n++] = LAYA_TOK_CLS;
    memcpy(ids + n, head_ids, head_len * sizeof(int));
    n += head_len;
    ids[n++] = LAYA_TOK_SEP;

    size_t n_markers = 0;
    for (size_t i = 0; i < opts.n_opts; i++) {
        markers[n_markers++] = (int)n;
        memcpy(ids + n, opts.opt_ids[i], opts.opt_len[i] * sizeof(int));
        n += opts.opt_len[i];
    }
    ids[n++] = LAYA_TOK_SEP;

    /* room = max(0, max_len - len(ids) - 1) */
    long room = (long)max_len - (long)n - 1;
    if (room < 0) room = 0;

    char *state_text = serialize_state(state);
    if (!state_text) {
        free(ids);
        free(markers);
        status = LAYA_ERR_OOM;
        goto out;
    }
    char *state_clean = replace_all(state_text, mask_tok, " ");
    free(state_text);
    int *st = NULL;
    size_t st_len = 0;
    if (!state_clean || laya_tokenizer_encode(state_clean, &st, &st_len) != LAYA_OK) {
        free(state_clean);
        free(ids);
        free(markers);
        status = LAYA_ERR_PARSE;
        goto out;
    }
    free(state_clean);
    if (st_len > (size_t)room) st_len = (size_t)room;

    /* ids = ids + st + [sep] */
    if (n + st_len + 1 > cap) {
        int *grown = realloc(ids, (n + st_len + 1) * sizeof(int));
        if (!grown) {
            free(st);
            free(ids);
            free(markers);
            status = LAYA_ERR_OOM;
            goto out;
        }
        ids = grown;
    }
    memcpy(ids + n, st, st_len * sizeof(int));
    n += st_len;
    free(st);
    ids[n++] = LAYA_TOK_SEP;

    /* ids[:max_len], markers filtered < max_len */
    if ((long)n > (long)max_len) n = (size_t)max_len;
    size_t m = 0;
    for (size_t i = 0; i < n_markers; i++)
        if (markers[i] < max_len) markers[m++] = markers[i];

    free(head_ids);
    laya_options_free(&opts);
    *out_ids = ids;
    *out_count = n;
    *out_markers = markers;
    *out_n_markers = m;
    return LAYA_OK;

out:
    free(head_ids);
    laya_options_free(&opts);
    return status;
}

/* ------------------------------------------------------------------ */
/* Agent._to_internal                                                  */
/* ------------------------------------------------------------------ */

static hd_json *json_new(hd_json_type type) {
    hd_json *v = calloc(1, sizeof *v);
    if (v) v->type = type;
    return v;
}

static int json_object_add(hd_json *obj, const char *key, hd_json *value) {
    hd_json_object *o = &obj->u.object;
    if (o->count + 1 > o->cap) {
        size_t cap = o->cap ? o->cap * 2 : 8;
        char **k = realloc(o->keys, cap * sizeof(char *));
        if (!k) return 0;
        o->keys = k;
        hd_json **v = realloc(o->values, cap * sizeof(hd_json *));
        if (!v) return 0;
        o->values = v;
        o->cap = cap;
    }
    o->keys[o->count] = xstrdup(key);
    o->values[o->count] = value;
    o->count++;
    return 1;
}

hd_json *laya_question_internal(const hd_json *qdef) {
    const char *type = hd_json_string(hd_json_get(qdef, "type"));
    if (!type) return NULL;

    hd_json *out = json_new(HD_JSON_OBJECT);
    if (!out) return NULL;
    if (!json_object_add(out, "t", json_new(HD_JSON_STRING))) {
        hd_json_free(out);
        return NULL;
    }
    free(out->u.object.values[0]->u.string);
    out->u.object.values[0]->u.string = xstrdup(type);

    /* criteria: for choice a list becomes {c: None} */
    const hd_json *crit = hd_json_get(qdef, "criteria");
    hd_json *crit_out = NULL;
    if (crit && crit->type == HD_JSON_ARRAY && strcmp(type, "choice") == 0) {
        crit_out = json_new(HD_JSON_OBJECT);
        for (size_t i = 0; crit_out && i < crit->u.array.count; i++) {
            const hd_json *item = crit->u.array.items[i];
            json_object_add(crit_out, hd_json_string(item) ? item->u.string : "", json_new(HD_JSON_NULL));
        }
    } else if (crit && crit->type == HD_JSON_ARRAY) {
        crit_out = json_new(HD_JSON_ARRAY);
        for (size_t i = 0; crit_out && i < crit->u.array.count; i++) {
            hd_json *item = json_new(HD_JSON_STRING);
            const char *s = hd_json_string(crit->u.array.items[i]);
            item->u.string = xstrdup(s ? s : "");
            hd_json_array *a = &crit_out->u.array;
            if (a->count + 1 > a->cap) {
                size_t cap = a->cap ? a->cap * 2 : 8;
                hd_json **grown = realloc(a->items, cap * sizeof(hd_json *));
                if (!grown) {
                    hd_json_free(item);
                    break;
                }
                a->items = grown;
                a->cap = cap;
            }
            a->items[a->count++] = item;
        }
    } else if (crit && crit->type == HD_JSON_OBJECT) {
        crit_out = json_new(HD_JSON_OBJECT);
        for (size_t i = 0; crit_out && i < crit->u.object.count; i++) {
            hd_json *copy = json_new(HD_JSON_STRING);
            const char *s = hd_json_string(crit->u.object.values[i]);
            copy->u.string = xstrdup(s ? s : "");
            json_object_add(crit_out, crit->u.object.keys[i], copy);
        }
    }

    /* ins: non-strings are json.dumps'd */
    const hd_json *ins = hd_json_get(qdef, "instructions");
    hd_json *ins_out = json_new(HD_JSON_STRING);
    if (ins && ins->type == HD_JSON_STRING) {
        ins_out->u.string = xstrdup(ins->u.string);
    } else {
        char *dumped = laya_render_criterion(ins);
        ins_out->u.string = dumped ? dumped : xstrdup("null");
    }

    if (!json_object_add(out, "ins", ins_out) ||
        !json_object_add(out, "crit", crit_out ? crit_out : json_new(HD_JSON_NULL))) {
        hd_json_free(out);
        return NULL;
    }
    return out;
}

/* ------------------------------------------------------------------ */
/* fixture check                                                       */
/* ------------------------------------------------------------------ */

static void print_ids(const int *ids, size_t n) {
    printf("[");
    for (size_t i = 0; i < n; i++) printf("%s%d", i ? ", " : "", ids[i]);
    printf("]");
}

static int print_one(const char *name, const char *qid, const hd_json *state,
                     const hd_json *qdef, int max_len, int head_max_len) {
    hd_json *q = laya_question_internal(qdef);
    if (!q) {
        fprintf(stderr, "error: bad question %s\n", qid);
        return 1;
    }
    int *ids = NULL, *markers = NULL;
    size_t n = 0, nm = 0;
    laya_status st = laya_build_sequence(state, q, max_len, head_max_len, &ids, &n, &markers, &nm);
    if (st != LAYA_OK) {
        fprintf(stderr, "error: %s: %s\n", qid, laya_last_error());
        hd_json_free(q);
        return 1;
    }
    printf("  {\"name\": \"%s\", \"question_id\": \"%s\", \"input_ids\": ", name, qid);
    print_ids(ids, n);
    printf(", \"marker_pos\": ");
    print_ids(markers, nm);
    printf("}\n");
    free(ids);
    free(markers);
    hd_json_free(q);
    return 0;
}

int build_sequence_fixture(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = malloc((size_t)sz + 1);
    if (!text || fread(text, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f);
        free(text);
        fprintf(stderr, "error: short read\n");
        return 1;
    }
    fclose(f);
    text[sz] = '\0';

    const char *err = NULL;
    hd_json *root = hd_json_parse(text, &err);
    free(text);
    if (!root) {
        fprintf(stderr, "error: bad corpus: %s\n", err ? err : "parse");
        return 1;
    }

    /* accept either a single request object or a list of them */
    size_t n_req = root->type == HD_JSON_ARRAY ? root->u.array.count : 1;
    printf("{\"items\": [\n");
    int first = 1, rc = 0;
    for (size_t r = 0; r < n_req; r++) {
        const hd_json *req = root->type == HD_JSON_ARRAY ? root->u.array.items[r] : root;
        const char *name = hd_json_string(hd_json_get(req, "name"));
        const hd_json *state = hd_json_get(req, "state");
        const hd_json *questions = hd_json_get(req, "questions");
        if (!questions || questions->type != HD_JSON_OBJECT) continue;
        for (size_t i = 0; i < questions->u.object.count; i++) {
            if (!first) printf(",\n");
            first = 0;
            rc |= print_one(name ? name : "", questions->u.object.keys[i], state,
                            questions->u.object.values[i], 512, 192);
        }
    }
    printf("\n]}\n");
    hd_json_free(root);
    return rc;
}
