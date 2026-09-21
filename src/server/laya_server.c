/*
 * laya-server — resident TypeSafe System One (Jev-compatible) HTTP server.
 *
 * The HTTP transport, socket handling, thread/queue discipline, JSON
 * helpers and the resident-model lifecycle are PORTED unchanged from
 * _reference/o1.c/src/server/o1_server.c (which itself ports the ds4
 * reference server, MIT License, Copyright (c) 2026 Salvatore Sanfilippo).
 * All HiDream/image/multipart/LoRA logic has been removed.
 *
 * Architecture:
 *
 *   client connection thread                single resident GPU worker
 *   -------------------------               -------------------------
 *   parse/validate HTTP + JSON              dequeue job
 *   build System One questions              ONE batched Laya forward
 *   enqueue job, wait                       serialize Jev answers
 *   send response                           signal client
 *
 * The resident Laya model owns the CUDA device, the weights and the large
 * workspaces. Exactly one thread may touch the GPU, so all inference is
 * serialized through a bounded queue while HTTP connections stay
 * concurrent. One HTTP request with N questions is ONE forward.
 */

#define _GNU_SOURCE

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "hd_json.h"
#include "laya.h"
#include "laya_infer.h"
#include "laya_sequence.h"
#include "model.h"

/* ------------------------------------------------------------------ */
/* Process globals                                                     */
/* ------------------------------------------------------------------ */

static volatile sig_atomic_t g_stop_requested = 0;
static volatile sig_atomic_t g_listen_fd = -1;

#define LAYA_SERVER_IO_TIMEOUT_SEC 10
#define LAYA_SERVER_SEND_STALL_TIMEOUT_MS 2000
#define LAYA_SERVER_MAX_HEADER_BYTES (64u * 1024u)

static void stop_signal_handler(int sig) {
    (void)sig;
    if (g_stop_requested) _exit(130);
    g_stop_requested = 1;
    if (g_listen_fd >= 0) {
        int fd = (int)g_listen_fd;
        g_listen_fd = -1;
        close(fd);
    }
}

/* ------------------------------------------------------------------ */
/* Buffers and allocation helpers                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} buf;

static void die(const char *msg) {
    fprintf(stderr, "laya-server: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static void buf_reserve(buf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) die("buffer overflow");
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < need) {
        if (cap > SIZE_MAX / 2) { cap = need; break; }
        cap *= 2;
    }
    b->ptr = xrealloc(b->ptr, cap);
    b->cap = cap;
}

static void buf_append(buf *b, const void *p, size_t n) {
    if (n == 0) return;
    buf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void buf_putc(buf *b, char c) { buf_append(b, &c, 1); }

static void buf_puts(buf *b, const char *s) { buf_append(b, s, strlen(s)); }

static void buf_printf(buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) die("vsnprintf failed");
    buf_reserve(b, (size_t)n);
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

static char *buf_take(buf *b) {
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static void buf_free(buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

/* ------------------------------------------------------------------ */
/* Minimal JSON reader (ported from ds4-server.c)                      */
/* ------------------------------------------------------------------ */

static void json_ws(const char **p) {
    while (**p && isspace((unsigned char)**p)) (*p)++;
}

static bool json_lit(const char **p, const char *lit) {
    size_t n = strlen(lit);
    if (strncmp(*p, lit, n) != 0) return false;
    *p += n;
    return true;
}

static int json_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static void utf8_put(buf *b, uint32_t cp) {
    if (cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) cp = 0xfffd;
    if (cp <= 0x7f) {
        buf_putc(b, (char)cp);
    } else if (cp <= 0x7ff) {
        buf_putc(b, (char)(0xc0 | (cp >> 6)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        buf_putc(b, (char)(0xe0 | (cp >> 12)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else {
        buf_putc(b, (char)(0xf0 | (cp >> 18)));
        buf_putc(b, (char)(0x80 | ((cp >> 12) & 0x3f)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    }
}

static bool json_u16(const char **p, uint32_t *out) {
    if ((*p)[0] != '\\' || (*p)[1] != 'u') return false;
    uint32_t cp = 0;
    for (int i = 0; i < 4; i++) {
        int h = json_hex((*p)[2 + i]);
        if (h < 0) return false;
        cp = (cp << 4) | (uint32_t)h;
    }
    *p += 6;
    *out = cp;
    return true;
}

static bool json_string(const char **p, char **out) {
    /* Always define *out: several callers reparse in place with
     * `free(x); json_string(&p, &x)`, and a stale freed pointer would be a
     * double free. */
    *out = NULL;
    json_ws(p);
    if (**p != '"') return false;
    (*p)++;
    buf b = {0};
    while (**p && **p != '"') {
        unsigned char c = (unsigned char)*(*p)++;
        if (c != '\\') { buf_putc(&b, (char)c); continue; }
        c = (unsigned char)*(*p)++;
        switch (c) {
        case '"': buf_putc(&b, '"'); break;
        case '\\': buf_putc(&b, '\\'); break;
        case '/': buf_putc(&b, '/'); break;
        case 'b': buf_putc(&b, '\b'); break;
        case 'f': buf_putc(&b, '\f'); break;
        case 'n': buf_putc(&b, '\n'); break;
        case 'r': buf_putc(&b, '\r'); break;
        case 't': buf_putc(&b, '\t'); break;
        case 'u': {
            *p -= 2;
            uint32_t cp = 0, lo = 0;
            if (!json_u16(p, &cp)) goto fail;
            if (cp >= 0xd800 && cp <= 0xdbff) {
                const char *low_start = *p;
                if (json_u16(p, &lo) && lo >= 0xdc00 && lo <= 0xdfff) {
                    cp = 0x10000u + ((cp - 0xd800u) << 10) + (lo - 0xdc00u);
                } else {
                    *p = low_start;
                    cp = 0xfffd;
                }
            }
            utf8_put(&b, cp);
            break;
        }
        default: goto fail;
        }
    }
    if (**p != '"') goto fail;
    (*p)++;
    *out = buf_take(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}

static bool json_number(const char **p, double *out) {
    json_ws(p);
    char *end = NULL;
    double v = strtod(*p, &end);
    if (end == *p) return false;
    *p = end;
    *out = v;
    return true;
}

/* Ignored fields still nest, so skipping is recursive with an explicit
 * ceiling: without it a useless field like {"x":[[[...]]]} can exhaust the
 * C stack before the request is rejected. */
#define JSON_MAX_NESTING 256

static void json_escape(buf *b, const char *s) {
    buf_putc(b, '"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') {
            buf_putc(b, '\\');
            buf_putc(b, (char)c);
        } else if (c == '\n') {
            buf_puts(b, "\\n");
        } else if (c == '\r') {
            buf_puts(b, "\\r");
        } else if (c == '\t') {
            buf_puts(b, "\\t");
        } else if (c < 0x20) {
            buf_printf(b, "\\u%04x", (unsigned)c);
        } else {
            buf_putc(b, (char)c);
        }
    }
    buf_putc(b, '"');
}

/* ------------------------------------------------------------------ */
/* HTTP transport (ported from ds4-server.c)                           */
/* ------------------------------------------------------------------ */

static long long wall_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static bool send_all(int fd, const void *p, size_t n) {
    const char *s = p;
    long long deadline = wall_ms() + LAYA_SERVER_SEND_STALL_TIMEOUT_MS;
    while (n) {
        if (g_stop_requested) return false;
        ssize_t w = send(fd, s, n, 0);
        if (w < 0 && errno == EINTR) continue;
        if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            long long remaining = deadline - wall_ms();
            if (remaining <= 0) return false;
            struct pollfd pfd = {.fd = fd, .events = POLLOUT};
            int timeout = remaining > 50 ? 50 : (int)remaining;
            int rc;
            do {
                rc = poll(&pfd, 1, timeout);
            } while (rc < 0 && errno == EINTR);
            if (rc < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)))
                return false;
            continue;
        }
        if (w <= 0) return false;
        s += w;
        n -= (size_t)w;
        deadline = wall_ms() + LAYA_SERVER_SEND_STALL_TIMEOUT_MS;
    }
    return true;
}

static void append_cors_headers(buf *h) {
    buf_puts(h,
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: *\r\n");
}

static const char *http_reason(int code) {
    switch (code) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 409: return "Conflict";
    case 413: return "Payload Too Large";
    case 415: return "Unsupported Media Type";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default: return "Error";
    }
}

static bool http_response(int fd, bool enable_cors, int code, const char *type,
                          const char *body) {
    const size_t body_len = body ? strlen(body) : 0;
    buf h = {0};
    buf_printf(&h,
        "HTTP/1.1 %d %s\r\n"
        "Content-Length: %zu\r\n",
        code, http_reason(code), body_len);
    if (type && type[0]) {
        buf_puts(&h, "Content-Type: ");
        buf_puts(&h, type);
        buf_puts(&h, "\r\n");
    }
    if (enable_cors) append_cors_headers(&h);
    buf_puts(&h, "Connection: close\r\n\r\n");
    bool ok = send_all(fd, h.ptr, h.len);
    if (ok && body_len) ok = send_all(fd, body, body_len);
    buf_free(&h);
    return ok;
}

/* Error body: a small {"error":"..."} object. The spec intentionally does
 * not reuse the OpenAI envelope, and the exact prose is not contractual. */
static bool http_error(int fd, bool enable_cors, int code, const char *msg) {
    buf b = {0};
    buf_puts(&b, "{\"error\":");
    json_escape(&b, msg ? msg : "error");
    buf_puts(&b, "}");
    bool ok = http_response(fd, enable_cors, code, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

typedef struct {
    char method[8];
    char path[256];
    char *body;
    size_t body_len;
    char *content_type;
    char *authorization;
    char *request_id;    /* x-typesafe-request-id, echoed back when present */
} http_request;

static void http_request_free(http_request *r) {
    free(r->body);
    free(r->content_type);
    free(r->authorization);
    free(r->request_id);
    memset(r, 0, sizeof(*r));
}

static ssize_t header_end(const char *p, size_t n) {
    for (size_t i = 3; i < n; i++) {
        if (p[i - 3] == '\r' && p[i - 2] == '\n' && p[i - 1] == '\r' &&
            p[i] == '\n')
            return (ssize_t)(i + 1);
    }
    for (size_t i = 1; i < n; i++) {
        if (p[i - 1] == '\n' && p[i] == '\n') return (ssize_t)(i + 1);
    }
    return -1;
}

/* Returns the Content-Length value, or -1 when the header is absent or
 * malformed. A malformed value must never be treated as 0: that would let a
 * body be silently ignored (or, worse, mis-framed). */
static long content_length(const char *h, size_t n) {
    const char *p = h, *end = h + n;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len >= 15 && strncasecmp(line, "Content-Length:", 15) == 0) {
            const char *v = line + 15;
            while (v < line + len && isspace((unsigned char)*v)) v++;
            if (v >= line + len) return -1;
            char *vend = NULL;
            long val = strtol(v, &vend, 10);
            if (vend == v || val < 0) return -1;
            return val;
        }
        if (p < end) p++;
    }
    return -1;
}

static char *header_value(const char *h, size_t n, const char *name) {
    size_t name_len = strlen(name);
    const char *p = h, *end = h + n;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len > name_len && strncasecmp(line, name, name_len) == 0 &&
            line[name_len] == ':') {
            const char *v = line + name_len + 1;
            const char *vend = line + len;
            while (v < vend && isspace((unsigned char)*v)) v++;
            while (vend > v && isspace((unsigned char)vend[-1])) vend--;
            return xstrndup(v, (size_t)(vend - v));
        }
        if (p < end) p++;
    }
    return NULL;
}

static bool read_http_request(int fd, http_request *r, size_t max_body,
                              bool *too_large) {
    buf b = {0};
    ssize_t hend = -1;

    while (hend < 0 && b.len < LAYA_SERVER_MAX_HEADER_BYTES) {
        char tmp[4096];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        buf_append(&b, tmp, (size_t)n);
        hend = header_end(b.ptr, b.len);
    }
    if (hend < 0) goto fail;

    char line[512];
    size_t i = 0;
    while (i < b.len && b.ptr[i] != '\n' && i + 1 < sizeof(line)) {
        line[i] = b.ptr[i];
        i++;
    }
    line[i] = '\0';
    if (sscanf(line, "%7s %255s", r->method, r->path) != 2) goto fail;
    char *q = strchr(r->path, '?');
    if (q) *q = '\0';

    r->content_type = header_value(b.ptr, (size_t)hend, "Content-Type");
    r->authorization = header_value(b.ptr, (size_t)hend, "Authorization");
    r->request_id = header_value(b.ptr, (size_t)hend, "x-typesafe-request-id");

    long clen = content_length(b.ptr, (size_t)hend);
    if (clen < 0) clen = 0;
    if ((size_t)clen > max_body) {
        /* Drain (bounded) so the client can read the 413 response instead of
         * seeing a connection reset while it is still writing the body. */
        if (too_large) *too_large = true;
        size_t remaining = (size_t)clen;
        while (remaining > 0) {
            char tmp[65536];
            size_t want = remaining < sizeof(tmp) ? remaining : sizeof(tmp);
            ssize_t n = recv(fd, tmp, want, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            remaining -= (size_t)n;
        }
        goto fail;
    }
    while (b.len < (size_t)hend + (size_t)clen) {
        char tmp[8192];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        buf_append(&b, tmp, (size_t)n);
    }

    r->body_len = (size_t)clen;
    r->body = xmalloc(r->body_len + 1);
    memcpy(r->body, b.ptr + hend, r->body_len);
    r->body[r->body_len] = '\0';
    buf_free(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}

static void configure_client_socket(int fd) {
    struct timeval tv;
    tv.tv_sec = LAYA_SERVER_IO_TIMEOUT_SEC;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

static int listen_on(const char *host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    if (!strcmp(host, "localhost")) host = "127.0.0.1";
    if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) {
        close(fd);
        errno = EINVAL;
        return -1;
    }
    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 128) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}
/* ------------------------------------------------------------------ */
/* Server configuration and the resident model                         */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *model_path;      /* GGUF file */
    const char *served_model_name;
    const char *model_alias;     /* optional extra accepted model id */
    int device_id;
    const char *host;
    int port;
    bool cors;
    size_t max_body;
    int queue_depth;
    const char *api_key;
} server_config;

static server_config g_cfg;

/* The resident Laya model. It owns the CUDA device, the weights and every
 * large workspace, so exactly ONE thread may touch it: all inference is
 * serialized through the worker queue below. */
static laya_model g_model;
static bool g_model_ready = false;

/* Number of inference requests served; reported on shutdown so the residency
 * gate (spec 32.2) can be checked: one model load, many requests. */
static int g_requests_served = 0;

static const char *served_model_id(void) {
    return g_cfg.served_model_name ? g_cfg.served_model_name : "laya";
}

static bool model_id_matches(const char *wanted) {
    if (!wanted || !wanted[0]) return true; /* omitted: use the served model */
    if (!strcmp(wanted, served_model_id())) return true;
    if (g_cfg.model_alias && !strcmp(wanted, g_cfg.model_alias)) return true;
    return false;
}

/* ------------------------------------------------------------------ */
/* System One request parsing                                          */
/* ------------------------------------------------------------------ */

/* The questions are handed to the runtime as the caller wrote them, and the
 * runtime renders the prompt itself (laya_build_sequence), so the server
 * never re-implements prompt building: it only validates the wire schema
 * and re-serializes the normalized questions back into a runtime request.
 *
 * Rejections follow the spec:
 *   400  malformed JSON body
 *   422  structurally valid JSON with an invalid question schema
 */

static void json_emit_value(buf *b, const hd_json *v);

typedef struct {
    size_t count;
    const char **keys;   /* caller-controlled ids, returned unchanged */
    const hd_json **values;
} sysone_questions;

static bool reject_question(buf *err, const char *qid, const char *why) {
    buf_printf(err, "question '%s': %s", qid ? qid : "?", why);
    return false;
}

/* `state` may be a string, object or array; numbers/booleans/null are not a
 * valid conversation state (spec 3). */
static bool validate_state(const hd_json *state, buf *err) {
    if (!state) {
        buf_puts(err, "missing 'state'");
        return false;
    }
    if (state->type != HD_JSON_STRING && state->type != HD_JSON_OBJECT &&
        state->type != HD_JSON_ARRAY) {
        buf_puts(err, "'state' must be a string, object or array");
        return false;
    }
    return true;
}

static bool validate_choice(const hd_json *qdef, buf *err, const char *qid) {
    const hd_json *crit = hd_json_get(qdef, "criteria");
    if (!crit) return reject_question(err, qid, "choice requires 'criteria'");
    if (crit->type != HD_JSON_OBJECT)
        return reject_question(err, qid, "choice 'criteria' must be an object");
    const size_t n = crit->u.object.count;
    if (n < 2)
        return reject_question(err, qid, "choice needs at least 2 criteria");
    if (n > 255)
        return reject_question(err, qid, "choice supports at most 255 criteria");
    for (size_t i = 0; i < n; i++) {
        if (!crit->u.object.keys[i] || !crit->u.object.keys[i][0])
            return reject_question(err, qid, "choice criteria keys must be non-empty");
        const hd_json *v = crit->u.object.values[i];
        if (!v || v->type != HD_JSON_STRING)
            return reject_question(err, qid,
                                   "choice criteria values must be strings");
    }
    return true;
}

static bool validate_score(const hd_json *qdef, buf *err, const char *qid) {
    const hd_json *crit = hd_json_get(qdef, "criteria");
    if (!crit) return reject_question(err, qid, "score requires 'criteria'");
    if (crit->type != HD_JSON_ARRAY)
        return reject_question(err, qid, "score 'criteria' must be an array");
    const size_t n = crit->u.array.count;
    if (n < 2)
        return reject_question(err, qid, "score needs at least 2 criteria");
    if (n > 10)
        return reject_question(err, qid, "score supports at most 10 criteria");
    for (size_t i = 0; i < n; i++) {
        const hd_json *v = crit->u.array.items[i];
        if (!v || v->type != HD_JSON_STRING)
            return reject_question(err, qid,
                                   "score criteria must be strings");
    }
    return true;
}

static bool validate_noul(const hd_json *qdef, buf *err, const char *qid) {
    const hd_json *crit = hd_json_get(qdef, "criteria");
    if (!crit) return true; /* optional */
    if (crit->type != HD_JSON_OBJECT)
        return reject_question(err, qid, "noul 'criteria' must be an object");
    for (size_t i = 0; i < crit->u.object.count; i++) {
        const char *k = crit->u.object.keys[i];
        const hd_json *v = crit->u.object.values[i];
        if (!k || (strcmp(k, "true") != 0 && strcmp(k, "false") != 0))
            return reject_question(
                err, qid, "noul criteria keys must be \"true\" or \"false\"");
        if (!v || v->type != HD_JSON_STRING)
            return reject_question(err, qid,
                                   "noul criteria values must be strings");
    }
    return true;
}

static bool parse_systemone(const char *body, sysone_questions *out,
                            const hd_json **out_state, bool *out_model_ok,
                            int *out_status, buf *err) {
    out->count = 0;
    out->keys = NULL;
    out->values = NULL;
    *out_model_ok = true;
    *out_status = 422;

    const char *jerr = NULL;
    hd_json *root = hd_json_parse(body, &jerr);
    if (!root) {
        buf_printf(err, "malformed JSON request body: %s",
                   jerr ? jerr : "parse error");
        *out_status = 400;
        return false;
    }

    const hd_json *model = hd_json_get(root, "model");
    if (model && model->type != HD_JSON_STRING)
        *out_model_ok = false;
    else if (model && !model_id_matches(model->u.string))
        *out_model_ok = false;

    const hd_json *state = hd_json_get(root, "state");
    if (!validate_state(state, err)) {
        hd_json_free(root);
        return false;
    }

    const hd_json *questions = hd_json_get(root, "questions");
    if (!questions) {
        buf_puts(err, "missing 'questions'");
        hd_json_free(root);
        return false;
    }
    if (questions->type != HD_JSON_OBJECT) {
        buf_puts(err, "'questions' must be an object");
        hd_json_free(root);
        return false;
    }
    const size_t B = questions->u.object.count;
    if (B == 0) {
        buf_puts(err, "'questions' must contain at least one question");
        hd_json_free(root);
        return false;
    }
    if (B > LAYA_INFER_MAX_Q) {
        buf_printf(err, "at most %d questions per request", LAYA_INFER_MAX_Q);
        hd_json_free(root);
        return false;
    }
    for (size_t b = 0; b < B; b++) {
        const char *qid = questions->u.object.keys[b];
        const hd_json *qdef = questions->u.object.values[b];
        if (!qid || !qid[0]) {
            reject_question(err, qid, "question id must be non-empty");
            hd_json_free(root);
            return false;
        }
        if (!qdef || qdef->type != HD_JSON_OBJECT) {
            reject_question(err, qid, "question must be an object");
            hd_json_free(root);
            return false;
        }
        const hd_json *type = hd_json_get(qdef, "type");
        if (!type || type->type != HD_JSON_STRING) {
            reject_question(err, qid, "question requires a string 'type'");
            hd_json_free(root);
            return false;
        }
        bool ok;
        if (!strcmp(type->u.string, "choice"))
            ok = validate_choice(qdef, err, qid);
        else if (!strcmp(type->u.string, "score"))
            ok = validate_score(qdef, err, qid);
        else if (!strcmp(type->u.string, "noul"))
            ok = validate_noul(qdef, err, qid);
        else
            ok = reject_question(err, qid, "unsupported question type");
        if (!ok) {
            hd_json_free(root);
            return false;
        }
    }

    /* The root is kept alive: the questions are referenced, not copied, and
     * the worker renders prompt text straight out of them. */
    out->count = B;
    out->keys = (const char **)questions->u.object.keys;
    out->values = (const hd_json **)questions->u.object.values;
    *out_state = state;
    return true;
}

/* Re-serialize a parsed JSON value. The server only needs this to echo the
 * verbatim `state` and the caller's questions back into the runtime request
 * shape; numbers use a round-trip-exact representation. */
static void json_emit_value(buf *b, const hd_json *v) {
    if (!v) { buf_puts(b, "null"); return; }
    switch (v->type) {
    case HD_JSON_NULL:
        buf_puts(b, "null");
        break;
    case HD_JSON_BOOL:
        buf_puts(b, v->u.boolean ? "true" : "false");
        break;
    case HD_JSON_INT:
        buf_printf(b, "%lld", (long long)v->u.integer);
        break;
    case HD_JSON_DOUBLE: {
        char tmp[40];
        snprintf(tmp, sizeof(tmp), "%.17g", v->u.number);
        buf_puts(b, tmp);
        break;
    }
    case HD_JSON_STRING:
        json_escape(b, v->u.string ? v->u.string : "");
        break;
    case HD_JSON_ARRAY:
        buf_putc(b, '[');
        for (size_t i = 0; i < v->u.array.count; i++) {
            if (i) buf_putc(b, ',');
            json_emit_value(b, v->u.array.items[i]);
        }
        buf_putc(b, ']');
        break;
    case HD_JSON_OBJECT:
        buf_putc(b, '{');
        for (size_t i = 0; i < v->u.object.count; i++) {
            if (i) buf_putc(b, ',');
            json_escape(b, v->u.object.keys[i]);
            buf_putc(b, ':');
            json_emit_value(b, v->u.object.values[i]);
        }
        buf_putc(b, '}');
        break;
    default:
        buf_puts(b, "null");
        break;
    }
}

/* Serialize the validated request back into the exact shape the runtime
 * expects: {"state":<verbatim>, "questions":{...}}. Only the fields the
 * forward pass reads are emitted; unknown top-level fields are dropped. */
static char *serialize_runtime_request(const hd_json *state,
                                       const sysone_questions *q) {
    buf b = {0};
    buf_puts(&b, "{\"state\":");
    json_emit_value(&b, state);
    buf_puts(&b, ",\"questions\":{");
    for (size_t i = 0; i < q->count; i++) {
        if (i) buf_putc(&b, ',');
        json_escape(&b, q->keys[i]);
        buf_putc(&b, ':');
        json_emit_value(&b, q->values[i]);
    }
    buf_puts(&b, "}}");
    return buf_take(&b);
}

/* ------------------------------------------------------------------ */
/* Jobs                                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    char *request;        /* raw JSON body, parsed by the worker */
    sysone_questions questions;
    char *runtime_req;    /* normalized {state, questions} for the runtime */
    const char *req_id;   /* x-typesafe-request-id, echoed back */

    /* results, produced by the worker */
    int status;
    char *error_msg;
    char *body;

    long long queue_wait_ms;
    long long total_ms;

    int client_fd;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    bool done;
    bool cancelled;
} server_job;

static void server_job_free(server_job *j) {
    if (!j) return;
    free(j->request);
    free(j->runtime_req);
    free(j->error_msg);
    free(j->body);
    pthread_mutex_destroy(&j->mu);
    pthread_cond_destroy(&j->cv);
    free(j);
}

static server_job *server_job_new(int client_fd) {
    server_job *j = xmalloc(sizeof(*j));
    memset(j, 0, sizeof(*j));
    j->client_fd = client_fd;
    j->status = 200;
    pthread_mutex_init(&j->mu, NULL);
    pthread_cond_init(&j->cv, NULL);
    return j;
}

static void server_job_fail(server_job *j, int status, const char *msg) {
    j->status = status;
    free(j->error_msg);
    j->error_msg = xstrdup(msg ? msg : "error");
}

/* ------------------------------------------------------------------ */
/* Bounded job queue                                                   */
/* ------------------------------------------------------------------ */

static server_job **g_queue = NULL;
static int g_queue_cap = 0;
static int g_queue_head = 0;
static int g_queue_len = 0;
static bool g_queue_stopping = false;
static pthread_mutex_t g_queue_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_queue_cv = PTHREAD_COND_INITIALIZER;

static bool enqueue(server_job *j) {
    pthread_mutex_lock(&g_queue_mu);
    if (g_queue_stopping || g_queue_len >= g_queue_cap) {
        pthread_mutex_unlock(&g_queue_mu);
        return false;
    }
    g_queue[(g_queue_head + g_queue_len) % g_queue_cap] = j;
    g_queue_len++;
    pthread_cond_signal(&g_queue_cv);
    pthread_mutex_unlock(&g_queue_mu);
    return true;
}

static server_job *dequeue(void) {
    pthread_mutex_lock(&g_queue_mu);
    while (g_queue_len == 0 && !g_queue_stopping)
        pthread_cond_wait(&g_queue_cv, &g_queue_mu);
    if (g_queue_len == 0) {
        pthread_mutex_unlock(&g_queue_mu);
        return NULL;
    }
    server_job *j = g_queue[g_queue_head];
    g_queue_head = (g_queue_head + 1) % g_queue_cap;
    g_queue_len--;
    pthread_mutex_unlock(&g_queue_mu);
    return j;
}

static void queue_stop(void) {
    pthread_mutex_lock(&g_queue_mu);
    g_queue_stopping = true;
    pthread_cond_broadcast(&g_queue_cv);
    pthread_mutex_unlock(&g_queue_mu);
}

/* ------------------------------------------------------------------ */
/* Worker                                                              */
/* ------------------------------------------------------------------ */

static void job_run(server_job *j);

static void *worker_main(void *arg) {
    (void)arg;
    for (;;) {
        server_job *j = dequeue();
        if (!j) break;
        job_run(j);
        pthread_mutex_lock(&j->mu);
        j->done = true;
        pthread_cond_broadcast(&j->cv);
        pthread_mutex_unlock(&j->mu);
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Jev-compatible response serialization                               */
/* ------------------------------------------------------------------ */

static void emit_f32(buf *b, float v) {
    if (isnan(v) || isinf(v)) {
        /* JSON has no NaN/Infinity: those are a runtime failure upstream, so
         * emit null rather than producing a body no client can parse. */
        buf_puts(b, "null");
        return;
    }
    char tmp[40];
    snprintf(tmp, sizeof(tmp), "%.7g", (double)v);
    buf_puts(b, tmp);
}

/* choice:
 *   {type, choice, probabilities:{orig key -> p}, confidence}
 * score:
 *   {type, score, legend:{"0":label,...}, probabilities:{"0":p,...},
 *    confidence}
 * noul:
 *   {type, noul}   - P(true), no confidence, no action
 *
 * The answer shapes are exact (spec 5-7); no extra fields are ever added. */
static void emit_answers(buf *b, const laya_infer_result *res, int n,
                         const sysone_questions *q) {
    buf_puts(b, "{");
    for (int i = 0; i < n; i++) {
        const laya_infer_result *r = &res[i];
        if (i) buf_putc(b, ',');
        json_escape(b, q->keys[i]);
        buf_puts(b, ":{");
        if (r->qtype == LAYA_QTYPE_CHOICE) {
            buf_puts(b, "\"type\":\"choice\",\"choice\":");
            json_escape(b, (r->choice >= 0 && r->choice < r->n_options &&
                            r->option_keys[r->choice])
                               ? r->option_keys[r->choice]
                               : "");
            buf_puts(b, ",\"probabilities\":{");
            for (int k = 0; k < r->n_options; k++) {
                if (k) buf_putc(b, ',');
                json_escape(b, r->option_keys[k] ? r->option_keys[k] : "");
                buf_putc(b, ':');
                emit_f32(b, r->probs[k]);
            }
            buf_puts(b, "},\"confidence\":");
            emit_f32(b, r->confidence);
        } else if (r->qtype == LAYA_QTYPE_SCORE) {
            buf_puts(b, "\"type\":\"score\",\"score\":");
            emit_f32(b, r->score);
            /* The legend carries the caller's criteria verbatim, exactly as
             * Python emits `{str(i): c for i, c in enumerate(q["crit"])}`;
             * the rendered option text is what the prompt uses, not the
             * legend. */
            buf_puts(b, ",\"legend\":{");
            const hd_json *crit = hd_json_get(q->values[i], "criteria");
            for (int k = 0; k < r->n_options; k++) {
                if (k) buf_putc(b, ',');
                json_escape(b, r->option_keys[k] ? r->option_keys[k] : "");
                buf_putc(b, ':');
                if (crit && crit->type == HD_JSON_ARRAY &&
                    (size_t)k < crit->u.array.count)
                    json_emit_value(b, crit->u.array.items[k]);
                else
                    json_escape(b, r->option_labels[k] ? r->option_labels[k] : "");
            }
            buf_puts(b, "},\"probabilities\":{");
            for (int k = 0; k < r->n_options; k++) {
                if (k) buf_putc(b, ',');
                json_escape(b, r->option_keys[k] ? r->option_keys[k] : "");
                buf_putc(b, ':');
                emit_f32(b, r->probs[k]);
            }
            buf_puts(b, "},\"confidence\":");
            emit_f32(b, r->confidence);
        } else {
            buf_puts(b, "\"type\":\"noul\",\"noul\":");
            emit_f32(b, r->noul);
        }
        buf_putc(b, '}');
    }
    buf_putc(b, '}');
}

static char *systemone_body(const laya_infer_result *res, int n,
                            const sysone_questions *q, long input_tokens) {
    buf b = {0};
    buf_puts(&b, "{\"model\":");
    json_escape(&b, served_model_id());
    buf_puts(&b, ",\"answers\":");
    emit_answers(&b, res, n, q);
    buf_printf(&b, ",\"usage\":{\"input_tokens\":%ld,\"output_tokens\":0}}",
               input_tokens);
    return buf_take(&b);
}

/* ------------------------------------------------------------------ */
/* Worker                                                              */
/* ------------------------------------------------------------------ */

/* Worker-side failure helper: stores the message the client will receive. */
static void job_error(server_job *j, int status, const char *msg) {
    server_job_fail(j, status, msg);
}

static void job_run(server_job *j) {
    long long t0 = wall_ms();
    j->queue_wait_ms = t0 - j->total_ms; /* total_ms holds the enqueue stamp */
    g_requests_served++;

    /* The normalized request is parsed on the worker so the CUDA device is
     * only ever touched by this thread. */
    const char *jerr = NULL;
    hd_json *root = hd_json_parse(j->runtime_req, &jerr);
    if (!root) {
        job_error(j, 500, "internal request serialization failed");
        return;
    }

    laya_infer_result *res = NULL;
    int n = 0;
    long tokens = 0;
    laya_status st = laya_infer_run(&g_model, root, &res, &n, &tokens);
    if (st != LAYA_OK) {
        job_error(j, 500, laya_last_error());
        hd_json_free(root);
        return;
    }

    j->body = systemone_body(res, n, &j->questions, tokens);
    j->status = 200;

    laya_infer_results_free(res, n);
    hd_json_free(root);
    j->total_ms = wall_ms() - t0;
}

/* ------------------------------------------------------------------ */
/* Client handling                                                     */
/* ------------------------------------------------------------------ */

static void handle_request(int fd, http_request *r);

static bool client_socket_disconnected(int fd) {
    struct pollfd pfd = {.fd = fd, .events = POLLIN};
    int rc = poll(&pfd, 1, 0);
    if (rc < 0) return true;
    if (rc == 0) return false;
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) return true;
    if (pfd.revents & POLLIN) {
        char tmp[1];
        ssize_t n = recv(fd, tmp, 1, MSG_PEEK | MSG_DONTWAIT);
        if (n == 0) return true;
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
            errno != EINTR)
            return true;
    }
    return false;
}

/* Wait for the worker to finish the job, aborting early if the client goes
 * away. The job is never freed here: the worker may still be inside the
 * model forward, so the caller waits for `done` first. */
static void wait_for_job_or_disconnect(server_job *j) {
    pthread_mutex_lock(&j->mu);
    while (!j->done) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 100 * 1000 * 1000;
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&j->cv, &j->mu, &ts);
        if (!j->done && client_socket_disconnected(j->client_fd))
            j->cancelled = true;
    }
    pthread_mutex_unlock(&j->mu);
}

static void *client_main(void *arg) {
    int fd = (int)(intptr_t)arg;
    configure_client_socket(fd);

    http_request r;
    memset(&r, 0, sizeof(r));
    bool too_large = false;
    if (!read_http_request(fd, &r, g_cfg.max_body, &too_large)) {
        if (too_large)
            http_error(fd, g_cfg.cors, 413,
                       "request body exceeds the configured limit");
        else
            http_error(fd, g_cfg.cors, 400, "malformed HTTP request");
        close(fd);
        return NULL;
    }

    handle_request(fd, &r);

    http_request_free(&r);
    close(fd);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Endpoint handlers                                                   */
/* ------------------------------------------------------------------ */

static void send_models(int fd) {
    buf b = {0};
    buf_puts(&b, "{\"object\":\"list\",\"data\":[{\"id\":");
    json_escape(&b, served_model_id());
    buf_puts(&b, ",\"object\":\"model\",\"created\":0,\"owned_by\":\"local\"");
    if (g_cfg.model_alias) {
        buf_puts(&b, ",\"aliases\":[");
        json_escape(&b, g_cfg.model_alias);
        buf_puts(&b, "]");
    }
    buf_puts(&b, "}]}");
    http_response(fd, g_cfg.cors, 200, "application/json", b.ptr);
    buf_free(&b);
}

static void send_health(int fd) {
    buf b = {0};
    buf_puts(&b, "{\"status\":\"ok\",\"ready\":");
    buf_puts(&b, g_model_ready ? "true" : "false");
    buf_puts(&b, ",\"endpoints\":[\"/v1/systemone\",\"/v1/models\"]}");
    http_response(fd, g_cfg.cors, 200, "application/json", b.ptr);
    buf_free(&b);
}

static void dispatch_job(int fd, server_job *j, const char *req_id) {
    j->total_ms = wall_ms(); /* enqueue stamp; job_run converts it to wait */
    if (!enqueue(j)) {
        http_error(fd, g_cfg.cors, 429, "the queue is full, retry later");
        server_job_free(j);
        return;
    }
    wait_for_job_or_disconnect(j);
    if (j->cancelled) {
        server_job_free(j);
        return;
    }
    if (j->status == 200 && j->body) {
        buf h = {0};
        buf_printf(&h, "HTTP/1.1 200 OK\r\nContent-Length: %zu\r\n",
                   strlen(j->body));
        buf_puts(&h, "Content-Type: application/json\r\n");
        if (req_id) {
            buf_puts(&h, "x-typesafe-request-id: ");
            buf_puts(&h, req_id);
            buf_puts(&h, "\r\n");
        }
        if (g_cfg.cors) append_cors_headers(&h);
        buf_puts(&h, "Connection: close\r\n\r\n");
        bool ok = send_all(fd, h.ptr, h.len);
        if (ok) ok = send_all(fd, j->body, strlen(j->body));
        buf_free(&h);
    } else {
        http_error(fd, g_cfg.cors, j->status, j->error_msg);
    }
    server_job_free(j);
}

/* POST /v1/systemone — the only inference endpoint. One request with N
 * questions becomes ONE batched forward on the worker. */
static void handle_systemone(int fd, http_request *r) {
    if (r->content_type &&
        strncasecmp(r->content_type, "application/json", 16) != 0) {
        http_error(fd, g_cfg.cors, 415, "Content-Type must be application/json");
        return;
    }

    sysone_questions q;
    const hd_json *state = NULL;
    bool model_ok = true;
    int status = 422;
    buf err = {0};
    if (!parse_systemone(r->body, &q, &state, &model_ok, &status, &err)) {
        http_error(fd, g_cfg.cors, status, err.ptr ? err.ptr : "bad request");
        buf_free(&err);
        return;
    }
    buf_free(&err);
    if (!model_ok) {
        http_error(fd, g_cfg.cors, 404, "unknown model");
        return;
    }

    server_job *j = server_job_new(fd);
    j->request = xstrdup(r->body ? r->body : "");
    j->questions = q;
    j->runtime_req = serialize_runtime_request(state, &q);
    if (!j->runtime_req) {
        http_error(fd, g_cfg.cors, 500, "internal serialization error");
        server_job_free(j);
        return;
    }
    j->req_id = r->request_id;
    dispatch_job(fd, j, r->request_id);
}

static bool authorized(http_request *r) {
    if (!g_cfg.api_key) return true;
    const char *h = r->authorization;
    if (!h) return false;
    const char *prefix = "Bearer ";
    if (strncmp(h, prefix, strlen(prefix)) != 0) return false;
    const char *tok = h + strlen(prefix);
    const char *want = g_cfg.api_key;
    size_t n = strlen(tok), m = strlen(want);
    unsigned char diff = (unsigned char)(n ^ m);
    for (size_t i = 0; i < n && i < m; i++)
        diff |= (unsigned char)(tok[i] ^ want[i]);
    return diff == 0;
}

static void handle_request(int fd, http_request *r) {
    if (strcmp(r->method, "OPTIONS") == 0) {
        http_response(fd, g_cfg.cors, 204, NULL, NULL);
        return;
    }
    /* Health probes are intentionally unauthenticated: supervisors must be
     * able to observe liveness even when an API key is configured, and the
     * response exposes no request or model data. */
    if (strcmp(r->method, "GET") == 0 &&
        (!strcmp(r->path, "/health") || !strcmp(r->path, "/healthz"))) {
        send_health(fd);
        return;
    }
    if (!authorized(r)) {
        http_error(fd, g_cfg.cors, 401, "missing or invalid API key");
        return;
    }
    if (strcmp(r->method, "GET") == 0 && strcmp(r->path, "/v1/models") == 0) {
        send_models(fd);
        return;
    }
    if (strcmp(r->path, "/v1/systemone") == 0) {
        if (strcmp(r->method, "POST") != 0) {
            http_error(fd, g_cfg.cors, 405, "method not allowed");
            return;
        }
        handle_systemone(fd, r);
        return;
    }
    http_error(fd, g_cfg.cors, 404, "unknown endpoint");
}

/* ------------------------------------------------------------------ */
/* CLI                                                                 */
/* ------------------------------------------------------------------ */

static void usage(FILE *f) {
    fprintf(f,
        "laya-server - TypeSafe System One (Jev-compatible) API for Laya\n"
        "\n"
        "Usage: laya-server [options]\n"
        "\n"
        "  --model PATH            resident Laya GGUF (required)\n"
        "  --served-model-name ID  model id advertised by /v1/models\n"
        "  --model-alias ID        extra accepted model id\n"
        "  --device N              CUDA device index (default 0)\n"
        "  --host HOST             listen address (default 127.0.0.1)\n"
        "  --port PORT             listen port (default 8000)\n"
        "  --cors                  enable permissive CORS headers\n"
        "  --max-body-mb N         maximum request body size (default 16)\n"
        "  --queue-depth N         pending inference jobs (default 8)\n"
        "  --api-key KEY           require Authorization: Bearer KEY\n"
        "  -h, --help              show this help\n");
}

#ifndef LAYA_SERVER_TEST
int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *served_model_name = NULL;
    const char *model_alias = NULL;
    int device_id = 0;
    const char *host = "127.0.0.1";
    int port = 8000;
    bool cors = false;
    size_t max_body = 16u * 1024u * 1024u;
    int queue_depth = 8;
    const char *api_key = NULL;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(a, "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(a, "--served-model-name") && i + 1 < argc) {
            served_model_name = argv[++i];
        } else if (!strcmp(a, "--model-alias") && i + 1 < argc) {
            model_alias = argv[++i];
        } else if (!strcmp(a, "--device") && i + 1 < argc) {
            device_id = atoi(argv[++i]);
        } else if (!strcmp(a, "--host") && i + 1 < argc) {
            host = argv[++i];
        } else if (!strcmp(a, "--port") && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (!strcmp(a, "--cors")) {
            cors = true;
        } else if (!strcmp(a, "--max-body-mb") && i + 1 < argc) {
            long mb = atol(argv[++i]);
            if (mb <= 0) { fprintf(stderr, "invalid --max-body-mb\n"); return 2; }
            max_body = (size_t)mb * 1024u * 1024u;
        } else if (!strcmp(a, "--queue-depth") && i + 1 < argc) {
            queue_depth = atoi(argv[++i]);
        } else if (!strcmp(a, "--api-key") && i + 1 < argc) {
            api_key = argv[++i];
        } else {
            fprintf(stderr, "laya-server: unknown option '%s'\n", a);
            usage(stderr);
            return 2;
        }
    }

    if (!model_path) {
        fprintf(stderr, "laya-server: --model PATH is required\n");
        usage(stderr);
        return 2;
    }
    if (port <= 0 || port > 65535) {
        fprintf(stderr, "laya-server: invalid --port\n");
        return 2;
    }
    if (queue_depth <= 0) queue_depth = 1;

    memset(&g_cfg, 0, sizeof(g_cfg));
    g_cfg.model_path = model_path;
    g_cfg.served_model_name = served_model_name;
    g_cfg.model_alias = model_alias;
    g_cfg.device_id = device_id;
    g_cfg.host = host;
    g_cfg.port = port;
    g_cfg.cors = cors;
    g_cfg.max_body = max_body;
    g_cfg.queue_depth = queue_depth;
    g_cfg.api_key = api_key;

    g_queue_cap = queue_depth;
    g_queue = xmalloc((size_t)g_queue_cap * sizeof(*g_queue));

    signal(SIGINT, stop_signal_handler);
    signal(SIGTERM, stop_signal_handler);
    signal(SIGPIPE, SIG_IGN);

    /* The model is loaded BEFORE the socket is opened: a client must never
     * observe a listening port that cannot serve a request. No model load is
     * ever performed per request. */
    fprintf(stderr, "laya-server: loading resident model from %s ...\n",
            model_path);
    laya_status mst = laya_model_load(model_path, device_id, &g_model);
    if (mst != LAYA_OK) {
        fprintf(stderr, "laya-server: model load failed: %s\n",
                laya_model_last_error());
        return 1;
    }
    g_model_ready = true;

    int listen_fd = listen_on(host, port);
    if (listen_fd < 0) {
        fprintf(stderr, "laya-server: cannot listen on %s:%d: %s\n", host,
                port, strerror(errno));
        laya_model_free(&g_model);
        return 1;
    }
    g_listen_fd = listen_fd;

    pthread_t worker;
    if (pthread_create(&worker, NULL, worker_main, NULL) != 0) {
        fprintf(stderr, "laya-server: cannot start worker thread\n");
        close(listen_fd);
        laya_model_free(&g_model);
        return 1;
    }

    fprintf(stderr,
            "laya-server: listening on http://%s:%d (model %s, device %d)\n",
            host, port, served_model_id(), device_id);

    while (!g_stop_requested) {
        /* A signal may be delivered to any thread, so closing the listen fd
         * from the handler does not reliably wake a blocked accept(). Poll
         * with a short timeout so the stop flag is observed promptly. */
        struct pollfd pfd;
        pfd.fd = listen_fd;
        pfd.events = POLLIN;
        pfd.revents = 0;
        int pr = poll(&pfd, 1, 200);
        if (pr < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (pr == 0) continue;
        if (g_stop_requested) break;

        struct sockaddr_in sa;
        socklen_t slen = sizeof(sa);
        int cfd = accept(listen_fd, (struct sockaddr *)&sa, &slen);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            if (g_stop_requested) break;
            if (errno == EBADF || errno == EINVAL) break;
            continue;
        }
        pthread_t th;
        if (pthread_create(&th, NULL, client_main, (void *)(intptr_t)cfd) != 0) {
            close(cfd);
            continue;
        }
        pthread_detach(th);
    }

    fprintf(stderr, "laya-server: shutting down\n");
    if (g_listen_fd >= 0) {
        close((int)g_listen_fd);
        g_listen_fd = -1;
    }
    queue_stop();
    pthread_join(worker, NULL);
    /* Residency gate: exactly one model load for the whole process. */
    fprintf(stderr, "laya-server: lifecycle open=%d model_load=%d requests=%d\n",
            g_model_ready ? 1 : 0, 1, g_requests_served);
    laya_model_free(&g_model);
    g_model_ready = false;
    free(g_queue);
    return 0;
}
#endif /* LAYA_SERVER_TEST */
