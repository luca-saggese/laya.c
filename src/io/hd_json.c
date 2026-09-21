#include "hd_json.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *p;
    const char *end;
    const char *err;
    int depth;
} parser_t;

static hd_json *parse_value(parser_t *ps);

static void set_err(parser_t *ps, const char *msg) {
    if (!ps->err) ps->err = msg;
}

static void skip_ws(parser_t *ps) {
    while (ps->p < ps->end && isspace((unsigned char)*ps->p)) ps->p++;
}

static hd_json *json_new(hd_json_type t) {
    hd_json *v = calloc(1, sizeof(hd_json));
    if (v) v->type = t;
    return v;
}

static int parse_string(parser_t *ps, char **out) {
    if (ps->p >= ps->end || *ps->p != '"') return 0;
    ps->p++;
    size_t cap = 32, len = 0;
    char *buf = malloc(cap);
    if (!buf) { set_err(ps, "oom"); return 0; }
    while (ps->p < ps->end) {
        char c = *ps->p++;
        if (c == '"') {
            buf[len] = '\0';
            *out = buf;
            return 1;
        }
        if ((unsigned char)c < 0x20) { free(buf); set_err(ps, "control char in string"); return 0; }
        if (c == '\\') {
            if (ps->p >= ps->end) { free(buf); set_err(ps, "bad escape"); return 0; }
            char e = *ps->p++;
            switch (e) {
                case '"': c = '"'; break;
                case '\\': c = '\\'; break;
                case '/': c = '/'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                case 'u': {
                    if (ps->end - ps->p < 4) { free(buf); set_err(ps, "bad \\u"); return 0; }
                    unsigned code = 0;
                    for (int i = 0; i < 4; i++) {
                        char h = *ps->p++;
                        code <<= 4;
                        if (h >= '0' && h <= '9') code |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') code |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') code |= (unsigned)(h - 'A' + 10);
                        else { free(buf); set_err(ps, "bad hex"); return 0; }
                    }
                    /* encode as UTF-8 (BMP only; adequate for our inputs) */
                    if (code < 0x80) {
                        c = (char)code;
                    } else if (code < 0x800) {
                        if (len + 2 >= cap) { cap *= 2; char *n = realloc(buf, cap); if (!n) { free(buf); set_err(ps, "oom"); return 0; } buf = n; }
                        buf[len++] = (char)(0xC0 | (code >> 6));
                        c = (char)(0x80 | (code & 0x3F));
                    } else {
                        if (len + 3 >= cap) { cap *= 2; char *n = realloc(buf, cap); if (!n) { free(buf); set_err(ps, "oom"); return 0; } buf = n; }
                        buf[len++] = (char)(0xE0 | (code >> 12));
                        buf[len++] = (char)(0x80 | ((code >> 6) & 0x3F));
                        c = (char)(0x80 | (code & 0x3F));
                    }
                    break;
                }
                default: free(buf); set_err(ps, "bad escape char"); return 0;
            }
        }
        if (len + 1 >= cap) { cap *= 2; char *n = realloc(buf, cap); if (!n) { free(buf); set_err(ps, "oom"); return 0; } buf = n; }
        buf[len++] = c;
    }
    free(buf);
    set_err(ps, "unterminated string");
    return 0;
}

static hd_json *parse_array(parser_t *ps) {
    hd_json *v = json_new(HD_JSON_ARRAY);
    if (!v) { set_err(ps, "oom"); return NULL; }
    ps->p++; /* '[' */
    skip_ws(ps);
    if (ps->p < ps->end && *ps->p == ']') { ps->p++; return v; }
    for (;;) {
        skip_ws(ps);
        hd_json *item = parse_value(ps);
        if (!item) { hd_json_free(v); return NULL; }
        if (v->u.array.count == v->u.array.cap) {
            size_t ncap = v->u.array.cap ? v->u.array.cap * 2 : 8;
            hd_json **n = realloc(v->u.array.items, ncap * sizeof(*n));
            if (!n) { set_err(ps, "oom"); hd_json_free(v); return NULL; }
            v->u.array.items = n;
            v->u.array.cap = ncap;
        }
        v->u.array.items[v->u.array.count++] = item;
        skip_ws(ps);
        if (ps->p >= ps->end) { hd_json_free(v); set_err(ps, "unterminated array"); return NULL; }
        char c = *ps->p++;
        if (c == ']') return v;
        if (c != ',') { hd_json_free(v); set_err(ps, "expected , or ]"); return NULL; }
    }
}

static hd_json *parse_object(parser_t *ps) {
    hd_json *v = json_new(HD_JSON_OBJECT);
    if (!v) { set_err(ps, "oom"); return NULL; }
    ps->p++; /* '{' */
    skip_ws(ps);
    if (ps->p < ps->end && *ps->p == '}') { ps->p++; return v; }
    for (;;) {
        skip_ws(ps);
        char *key = NULL;
        if (!parse_string(ps, &key)) { hd_json_free(v); return NULL; }
        skip_ws(ps);
        if (ps->p >= ps->end || *ps->p != ':') { free(key); hd_json_free(v); set_err(ps, "expected :"); return NULL; }
        ps->p++;
        skip_ws(ps);
        hd_json *val = parse_value(ps);
        if (!val) { free(key); hd_json_free(v); return NULL; }
        if (v->u.object.count == v->u.object.cap) {
            size_t ncap = v->u.object.cap ? v->u.object.cap * 2 : 8;
            char **nk = realloc(v->u.object.keys, ncap * sizeof(*nk));
            hd_json **nv = realloc(v->u.object.values, ncap * sizeof(*nv));
            if (!nk || !nv) { free(key); set_err(ps, "oom"); hd_json_free(v); return NULL; }
            v->u.object.keys = nk;
            v->u.object.values = nv;
            v->u.object.cap = ncap;
        }
        v->u.object.keys[v->u.object.count] = key;
        v->u.object.values[v->u.object.count] = val;
        v->u.object.count++;
        skip_ws(ps);
        if (ps->p >= ps->end) { hd_json_free(v); set_err(ps, "unterminated object"); return NULL; }
        char c = *ps->p++;
        if (c == '}') return v;
        if (c != ',') { hd_json_free(v); set_err(ps, "expected , or }"); return NULL; }
    }
}

static hd_json *parse_number(parser_t *ps) {
    const char *start = ps->p;
    if (ps->p < ps->end && (*ps->p == '-' || *ps->p == '+')) ps->p++;
    while (ps->p < ps->end && isdigit((unsigned char)*ps->p)) ps->p++;
    int is_float = 0;
    if (ps->p < ps->end && *ps->p == '.') { is_float = 1; ps->p++; while (ps->p < ps->end && isdigit((unsigned char)*ps->p)) ps->p++; }
    if (ps->p < ps->end && (*ps->p == 'e' || *ps->p == 'E')) {
        is_float = 1; ps->p++;
        if (ps->p < ps->end && (*ps->p == '-' || *ps->p == '+')) ps->p++;
        while (ps->p < ps->end && isdigit((unsigned char)*ps->p)) ps->p++;
    }
    size_t len = (size_t)(ps->p - start);
    char *tmp = malloc(len + 1);
    if (!tmp) { set_err(ps, "oom"); return NULL; }
    memcpy(tmp, start, len);
    tmp[len] = '\0';
    hd_json *v = json_new(is_float ? HD_JSON_DOUBLE : HD_JSON_INT);
    if (!v) { free(tmp); set_err(ps, "oom"); return NULL; }
    if (is_float) v->u.number = strtod(tmp, NULL);
    else v->u.integer = strtoll(tmp, NULL, 10);
    free(tmp);
    return v;
}

static hd_json *parse_value(parser_t *ps) {
    if (ps->depth > 512) { set_err(ps, "nesting too deep"); return NULL; }
    skip_ws(ps);
    if (ps->p >= ps->end) { set_err(ps, "unexpected EOF"); return NULL; }
    char c = *ps->p;
    if (c == '{') { ps->depth++; hd_json *v = parse_object(ps); ps->depth--; return v; }
    if (c == '[') { ps->depth++; hd_json *v = parse_array(ps); ps->depth--; return v; }
    if (c == '"') {
        hd_json *v = json_new(HD_JSON_STRING);
        if (!v) { set_err(ps, "oom"); return NULL; }
        if (!parse_string(ps, &v->u.string)) { free(v); return NULL; }
        return v;
    }
    if (c == '-' || c == '+' || isdigit((unsigned char)c)) return parse_number(ps);
    if (ps->end - ps->p >= 4 && strncmp(ps->p, "true", 4) == 0) {
        ps->p += 4; hd_json *v = json_new(HD_JSON_BOOL); if (v) v->u.boolean = 1; return v;
    }
    if (ps->end - ps->p >= 5 && strncmp(ps->p, "false", 5) == 0) {
        ps->p += 5; hd_json *v = json_new(HD_JSON_BOOL); if (v) v->u.boolean = 0; return v;
    }
    if (ps->end - ps->p >= 4 && strncmp(ps->p, "null", 4) == 0) {
        ps->p += 4; return json_new(HD_JSON_NULL);
    }
    set_err(ps, "unexpected character");
    return NULL;
}

hd_json *hd_json_parse(const char *text, const char **err) {
    parser_t ps = { text, text + strlen(text), NULL, 0 };
    hd_json *v = parse_value(&ps);
    if (!v) {
        if (err) *err = ps.err ? ps.err : "parse error";
        return NULL;
    }
    skip_ws(&ps);
    if (ps.p != ps.end) {
        hd_json_free(v);
        if (err) *err = "trailing content after JSON";
        return NULL;
    }
    if (err) *err = NULL;
    return v;
}

void hd_json_free(hd_json *v) {
    if (!v) return;
    switch (v->type) {
        case HD_JSON_STRING: free(v->u.string); break;
        case HD_JSON_ARRAY:
            for (size_t i = 0; i < v->u.array.count; i++) hd_json_free(v->u.array.items[i]);
            free(v->u.array.items);
            break;
        case HD_JSON_OBJECT:
            for (size_t i = 0; i < v->u.object.count; i++) {
                free(v->u.object.keys[i]);
                hd_json_free(v->u.object.values[i]);
            }
            free(v->u.object.keys);
            free(v->u.object.values);
            break;
        default: break;
    }
    free(v);
}

const hd_json *hd_json_get(const hd_json *obj, const char *key) {
    if (!obj || obj->type != HD_JSON_OBJECT) return NULL;
    for (size_t i = 0; i < obj->u.object.count; i++) {
        if (strcmp(obj->u.object.keys[i], key) == 0) return obj->u.object.values[i];
    }
    return NULL;
}

const char *hd_json_string(const hd_json *v) {
    return (v && v->type == HD_JSON_STRING) ? v->u.string : NULL;
}

int64_t hd_json_int(const hd_json *v, int64_t dflt) {
    if (!v) return dflt;
    if (v->type == HD_JSON_INT) return v->u.integer;
    if (v->type == HD_JSON_DOUBLE) return (int64_t)v->u.number;
    return dflt;
}

double hd_json_double(const hd_json *v, double dflt) {
    if (!v) return dflt;
    if (v->type == HD_JSON_DOUBLE) return v->u.number;
    if (v->type == HD_JSON_INT) return (double)v->u.integer;
    return dflt;
}

int hd_json_bool(const hd_json *v, int dflt) {
    return (v && v->type == HD_JSON_BOOL) ? v->u.boolean : dflt;
}

size_t hd_json_array_len(const hd_json *v) {
    return (v && v->type == HD_JSON_ARRAY) ? v->u.array.count : 0;
}

const hd_json *hd_json_array_at(const hd_json *v, size_t i) {
    if (!v || v->type != HD_JSON_ARRAY || i >= v->u.array.count) return NULL;
    return v->u.array.items[i];
}
