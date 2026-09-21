/*
 * Native Laya tokenizer: mmBERT byte-level BPE.
 *
 * Structure and helper layout are adapted from o1.c src/model/tokenizer.c.
 * The differences from that file, all required by the Laya tokenizer, are:
 *
 *   - the pre-tokenizer regex is the transformers ByteLevel pattern
 *     ("'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"),
 *     matched left-to-right with the alternation order above;
 *   - the text is NFC-normalized first;
 *   - added tokens are extracted longest-first and carry lstrip/rstrip;
 *   - the vocabulary is mmBERT's, so the o1.c tables are replaced by the
 *     generated ones.
 */

#include "laya_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hd_json.h"
#include "laya_tokenizer_tables.h"

/* ------------------------------------------------------------------ */
/* Unicode helpers                                                     */
/* ------------------------------------------------------------------ */

static size_t utf8_decode(const char *s, unsigned *cp) {
    const unsigned char *p = (const unsigned char *)s;
    if (p[0] < 0x80) {
        *cp = p[0];
        return 1;
    }
    int n;
    unsigned v;
    if ((p[0] & 0xE0) == 0xC0) {
        n = 2;
        v = p[0] & 0x1F;
    } else if ((p[0] & 0xF0) == 0xE0) {
        n = 3;
        v = p[0] & 0x0F;
    } else if ((p[0] & 0xF8) == 0xF0) {
        n = 4;
        v = p[0] & 0x07;
    } else {
        *cp = 0xFFFDu;
        return 1;
    }
    for (int i = 1; i < n; i++) {
        if ((p[i] & 0xC0) != 0x80) {
            *cp = 0xFFFDu;
            return 1;
        }
        v = (v << 6) | (p[i] & 0x3F);
    }
    *cp = v;
    return (size_t)n;
}

static unsigned utf8_peek(const char *s) {
    unsigned cp;
    utf8_decode(s, &cp);
    return cp;
}

static size_t utf8_len(const char *s) {
    unsigned cp;
    return utf8_decode(s, &cp);
}

static size_t utf8_encode(unsigned cp, char *out) {    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static int cp_in_ranges(const unsigned *lo, const unsigned *hi, int n, unsigned cp) {
    int a = 0, b = n - 1;
    while (a <= b) {
        int mid = a + (b - a) / 2;
        if (cp < lo[mid])
            b = mid - 1;
        else if (cp > hi[mid])
            a = mid + 1;
        else
            return 1;
    }
    return 0;
}

static int cp_is_letter(unsigned cp) {
    return cp_in_ranges(laya_tok_letter_range_lo, laya_tok_letter_range_hi, laya_tok_letter_range_count, cp);
}

static int cp_is_number(unsigned cp) {
    return cp_in_ranges(laya_tok_number_range_lo, laya_tok_number_range_hi, laya_tok_number_range_count, cp);
}

static int cp_is_ws(unsigned cp) {
    for (int i = 0; i < laya_tok_ws_count; i++)
        if (laya_tok_ws_cp[i] == cp) return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* NFC normalization                                                   */
/* ------------------------------------------------------------------ */

static int ccc_of(unsigned cp) {
    int a = 0, b = laya_tok_ccc_count - 1;
    while (a <= b) {
        int mid = a + (b - a) / 2;
        if (cp < laya_tok_ccc_cp[mid])
            b = mid - 1;
        else if (cp > laya_tok_ccc_cp[mid])
            a = mid + 1;
        else
            return laya_tok_ccc_val[mid];
    }
    return 0;
}

static unsigned compose_pair(unsigned a, unsigned b) {
    int lo = 0, hi = laya_tok_compose_count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (a < laya_tok_compose_a[mid] || (a == laya_tok_compose_a[mid] && b < laya_tok_compose_b[mid]))
            hi = mid - 1;
        else if (a > laya_tok_compose_a[mid] || (a == laya_tok_compose_a[mid] && b > laya_tok_compose_b[mid]))
            lo = mid + 1;
        else
            return laya_tok_compose_c[mid];
    }
    return 0;
}

static void decompose_cp(unsigned cp, unsigned *out, size_t *n) {
    if (*n + 4 > 32) return;
    int lo = 0, hi = laya_tok_decomp_count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (cp < laya_tok_decomp_cp[mid])
            hi = mid - 1;
        else if (cp > laya_tok_decomp_cp[mid])
            lo = mid + 1;
        else {
            if (laya_tok_decomp_b[mid] == 0) {
                decompose_cp(laya_tok_decomp_a[mid], out, n);
            } else {
                decompose_cp(laya_tok_decomp_a[mid], out, n);
                decompose_cp(laya_tok_decomp_b[mid], out, n);
            }
            return;
        }
    }
    out[(*n)++] = cp;
}

static int canonical_order(unsigned *cps, size_t n) {
    for (size_t i = 1; i < n; i++) {
        unsigned c = cps[i];
        int cc = ccc_of(c);
        if (cc == 0) continue;
        size_t j = i;
        while (j > 0) {
            int pc = ccc_of(cps[j - 1]);
            if (pc == 0 || pc <= cc) break;
            cps[j] = cps[j - 1];
            j--;
        }
        cps[j] = c;
    }
    return 1;
}

static int nfc_normalize(const char *in, size_t in_len, char **out) {
    *out = NULL;
    unsigned *buf = malloc((in_len * 4 + 4) * sizeof(unsigned));
    unsigned *dec = malloc(32 * sizeof(unsigned));
    if (!buf || !dec) {
        free(buf);
        free(dec);
        laya_set_error("tokenizer oom (nfc)");
        return 0;
    }

    size_t n = 0, pos = 0;
    while (pos < in_len) {
        unsigned cp;
        pos += utf8_decode(in + pos, &cp);
        size_t dn = 0;
        decompose_cp(cp, dec, &dn);
        for (size_t k = 0; k < dn; k++) buf[n++] = dec[k];
    }
    canonical_order(buf, n);

    /* canonical composition (single pass, matching UAX#15 NFC) */
    size_t m = 0;
    size_t starter_pos = 0;
    unsigned starter = 0;
    int last_cc = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned c = buf[i];
        int cc = ccc_of(c);
        if (m > 0) {
            unsigned comp = compose_pair(starter, c);
            if (comp && (last_cc == 0 || last_cc < cc)) {
                buf[starter_pos] = comp;
                starter = comp;
                continue;
            }
        }
        if (cc == 0) {
            starter_pos = m;
            starter = c;
        }
        last_cc = cc;
        buf[m++] = c;
    }

    char *res = malloc(m * 4 + 1);
    if (!res) {
        free(buf);
        free(dec);
        laya_set_error("tokenizer oom (nfc out)");
        return 0;
    }
    size_t o = 0;
    for (size_t i = 0; i < m; i++) o += utf8_encode(buf[i], res + o);
    res[o] = '\0';
    free(buf);
    free(dec);
    *out = res;
    return 1;
}

/* ------------------------------------------------------------------ */
/* Added-token lookup                                                  */
/* ------------------------------------------------------------------ */

static int added_lookup(const char *content) {
    for (int i = 0; i < laya_tok_added_count; i++)
        if (strcmp(laya_tok_added_str[i], content) == 0) return laya_tok_added_id[i];
    return -1;
}

int laya_tokenizer_added_id(const char *content) {
    if (!content) return -1;
    return added_lookup(content);
}

/* Leftmost-longest added-token match at `at`. A token flagged lstrip has the
   effective pattern `\s*<content>`, so it also wins over (and swallows) the
   whitespace run that precedes it. Returns the id and sets *mlen. */
static int added_match(const char *text, size_t len, size_t at, size_t *mlen) {
    int best_id = -1;
    size_t best_len = 0;
    for (int i = 0; i < laya_tok_added_count; i++) {
        const char *s = laya_tok_added_str[i];
        size_t sl = strlen(s);
        size_t off = 0;
        if (laya_tok_added_lstrip[i]) {
            while (at + off < len) {
                unsigned cp;
                size_t adv = utf8_decode(text + at + off, &cp);
                if (!cp_is_ws(cp)) break;
                off += adv;
            }
        }
        if (off + sl > len - at) continue;
        if (off + sl <= best_len) continue;
        if (memcmp(text + at + off, s, sl) != 0) continue;
        best_len = off + sl;
        best_id = laya_tok_added_id[i];
    }
    if (best_id >= 0) {
        *mlen = best_len;
        return best_id;
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* Byte-level BPE                                                      */
/* ------------------------------------------------------------------ */

static int vocab_lookup(const char *piece) {
    int lo = 0, hi = laya_tok_vocab_count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int c = strcmp(laya_tok_vocab_str[mid], piece);
        if (c == 0) return laya_tok_vocab_id[mid];
        if (c < 0)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return -1;
}

static int merge_rank_of(const char *l, const char *r) {
    int lo = 0, hi = laya_tok_merge_count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int c = strcmp(laya_tok_merge_left[mid], l);
        if (c == 0) c = strcmp(laya_tok_merge_right[mid], r);
        if (c == 0) return laya_tok_merge_rank[mid];
        if (c < 0)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return -1;
}

typedef struct {
    size_t count;
    const char **str;
    unsigned char *owned;
} bpe_pieces;

static void pieces_clear(bpe_pieces *p) {
    for (size_t i = 0; i < p->count; i++)
        if (p->owned[i]) free((void *)p->str[i]);
    free(p->str);
    free(p->owned);
}

static int bpe_apply_merge(bpe_pieces *p) {
    int best_rank = -1;
    int best_at = -1;
    for (size_t i = 0; i + 1 < p->count; i++) {
        int r = merge_rank_of(p->str[i], p->str[i + 1]);
        if (r >= 0 && (best_rank < 0 || r < best_rank)) {
            best_rank = r;
            best_at = (int)i;
        }
    }
    if (best_rank < 0) return 0;

    const char *l = p->str[best_at], *r = p->str[best_at + 1];
    size_t nl = strlen(l), nr = strlen(r);
    char *merged = malloc(nl + nr + 1);
    if (!merged) return -1;
    memcpy(merged, l, nl);
    memcpy(merged + nl, r, nr);
    merged[nl + nr] = '\0';

    if (p->owned[best_at + 1]) free((void *)p->str[best_at + 1]);
    p->str[best_at] = merged;
    p->owned[best_at] = 1;
    for (size_t i = (size_t)best_at + 1; i + 1 < p->count; i++) {
        p->str[i] = p->str[i + 1];
        p->owned[i] = p->owned[i + 1];
    }
    p->count--;
    return 1;
}

static int bpe_encode(const char *text, size_t nbytes, int **ids_out, size_t *n_ids, size_t *cap) {
    if (nbytes == 0) return 1;
    bpe_pieces p;
    p.count = nbytes;
    p.str = malloc(nbytes * sizeof(char *));
    p.owned = malloc(nbytes);
    if (!p.str || !p.owned) {
        free(p.str);
        free(p.owned);
        laya_set_error("tokenizer oom (pieces)");
        return 0;
    }
    for (size_t i = 0; i < nbytes; i++) {
        p.str[i] = laya_tok_byte_piece[(unsigned char)text[i]];
        p.owned[i] = 0;
    }

    for (;;) {
        int r = bpe_apply_merge(&p);
        if (r < 0) {
            pieces_clear(&p);
            laya_set_error("tokenizer oom (merge)");
            return 0;
        }
        if (r == 0) break;
    }

    for (size_t i = 0; i < p.count; i++) {
        int vid = vocab_lookup(p.str[i]);
        if (vid < 0) {
            laya_set_error("tokenizer: piece '%s' not in vocab", p.str[i]);
            pieces_clear(&p);
            return 0;
        }
        if (*n_ids + 1 > *cap) {
            size_t ncap = *cap ? *cap * 2 : 16;
            int *ni = realloc(*ids_out, ncap * sizeof(int));
            if (!ni) {
                pieces_clear(&p);
                laya_set_error("tokenizer oom (ids)");
                return 0;
            }
            *ids_out = ni;
            *cap = ncap;
        }
        (*ids_out)[(*n_ids)++] = vid;
    }
    pieces_clear(&p);
    return 1;
}

/* ------------------------------------------------------------------ */
/* transformers ByteLevel pre-tokenizer                                */
/* ------------------------------------------------------------------ */

/*
 * Matches one pre-token starting at `pos` (bounded by `end`) and returns the
 * byte length, or 0. The alternation is evaluated in source order; the first
 * branch that matches wins, exactly like the regex engine.
 */
static size_t pretokenize(const char *text, size_t end, size_t pos) {
    size_t p = pos;
    unsigned cp;
    size_t adv;

    /* "'s|'t|'re|'ve|'m|'ll|'d" (the ByteLevel pattern is case-insensitive) */
    if (text[p] == '\'') {
        static const char *const APOS[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
        for (size_t i = 0; i < sizeof(APOS) / sizeof(APOS[0]); i++) {
            size_t l = strlen(APOS[i]);
            if (p + l > end) continue;
            if (text[p + 1] != APOS[i][1]) continue;
            if (l == 3 && text[p + 2] != APOS[i][2]) continue;
            return l;
        }
    }

    /* " ?\p{L}+" */
    adv = 0;
    if (text[p] == ' ') adv = 1;
    if (p + adv < end) {
        size_t q = p + adv;
        size_t n = utf8_decode(text + q, &cp);
        if (cp_is_letter(cp)) {
            size_t r = q + n;
            for (;;) {
                if (r >= end) break;
                unsigned c2;
                size_t n2 = utf8_decode(text + r, &c2);
                if (!cp_is_letter(c2)) break;
                r += n2;
            }
            return r - p;
        }
    }

    /* " ?\p{N}+" */
    adv = 0;
    if (text[p] == ' ') adv = 1;
    if (p + adv < end) {
        size_t q = p + adv;
        size_t n = utf8_decode(text + q, &cp);
        if (cp_is_number(cp)) {
            size_t r = q + n;
            for (;;) {
                if (r >= end) break;
                unsigned c2;
                size_t n2 = utf8_decode(text + r, &c2);
                if (!cp_is_number(c2)) break;
                r += n2;
            }
            return r - p;
        }
    }

    /* " ?[^\s\p{L}\p{N}]+" */
    adv = 0;
    if (text[p] == ' ') adv = 1;
    if (p + adv < end) {
        size_t q = p + adv;
        size_t n = utf8_decode(text + q, &cp);
        if (!cp_is_ws(cp) && !cp_is_letter(cp) && !cp_is_number(cp)) {
            size_t r = q + n;
            for (;;) {
                if (r >= end) break;
                unsigned c2;
                size_t n2 = utf8_decode(text + r, &c2);
                if (cp_is_ws(c2) || cp_is_letter(c2) || cp_is_number(c2)) break;
                r += n2;
            }
            return r - p;
        }
    }

    /* "\s+(?!\S)": greedy whitespace run, backtracking one character at a
       time, stopping at the first split whose remainder does not start with
       whitespace. This is what the regex engine does with a negative
       lookahead after a greedy quantifier. */
    {
        size_t r = p;
        while (r < end && cp_is_ws(utf8_peek(text + r))) r += utf8_len(text + r);
        while (r > p) {
            if (r >= end || cp_is_ws(utf8_peek(text + r))) return r - p;
            size_t s = p, last = p;
            while (s < r) {
                last = s;
                s += utf8_len(text + s);
            }
            r = last;
        }
    }

    /* "\s+" */
    {
        size_t r = p;
        while (r < end) {
            unsigned c2;
            size_t n2 = utf8_decode(text + r, &c2);
            if (!cp_is_ws(c2)) break;
            r += n2;
        }
        if (r > p) return r - p;
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* encode                                                              */
/* ------------------------------------------------------------------ */

laya_status laya_tokenizer_encode(const char *text, int **out_ids, size_t *out_count) {
    *out_ids = NULL;
    *out_count = 0;
    if (!text) {
        laya_set_error("tokenizer: null input");
        return LAYA_ERR_PARSE;
    }

    char *norm = NULL;
    if (!nfc_normalize(text, strlen(text), &norm)) return LAYA_ERR_OOM;
    size_t len = strlen(norm);

    int *ids = NULL;
    size_t n_ids = 0, cap = 0;
    laya_status status = LAYA_OK;
    size_t i = 0;

    while (i < len) {
        size_t mlen = 0;
        int mid = added_match(norm, len, i, &mlen);
        if (mid >= 0) {
            if (n_ids + 1 > cap) {
                size_t ncap = cap ? cap * 2 : 16;
                int *ni = realloc(ids, ncap * sizeof(int));
                if (!ni) {
                    status = LAYA_ERR_OOM;
                    break;
                }
                ids = ni;
                cap = ncap;
            }
            ids[n_ids++] = mid;
            i += mlen;
            continue;
        }

        /* Plain text run: stop where the next added token starts. A lstrip
           token absorbs the whitespace run before it, so the scan breaks at
           the beginning of that run and the run is never pre-tokenized. */
        size_t start = i;
        while (i < len) {
            size_t sl = 0;
            if (added_match(norm, len, i, &sl) >= 0) break;
            unsigned cp;
            i += utf8_decode(norm + i, &cp);
        }
        size_t end = i;

        size_t pos = start;
        while (pos < end) {
            size_t plen = pretokenize(norm, end, pos);
            if (plen == 0) {
                unsigned cp;
                plen = utf8_decode(norm + pos, &cp);
            }
            if (plen > end - pos) plen = end - pos;
            if (!bpe_encode(norm + pos, plen, &ids, &n_ids, &cap)) {
                status = LAYA_ERR_PARSE;
                goto out;
            }
            pos += plen;
        }
    }

out:
    free(norm);
    if (status != LAYA_OK) {
        free(ids);
        return status;
    }
    *out_ids = ids;
    *out_count = n_ids;
    return LAYA_OK;
}

/* ------------------------------------------------------------------ */
/* decode                                                              */
/* ------------------------------------------------------------------ */

laya_status laya_tokenizer_decode(const int *ids, size_t count, char **out) {
    *out = NULL;
    size_t cap = 64;
    char *buf = malloc(cap);
    if (!buf) {
        laya_set_error("tokenizer oom (decode)");
        return LAYA_ERR_OOM;
    }
    size_t n = 0;
    buf[0] = '\0';

    for (size_t i = 0; i < count; i++) {
        const char *piece = NULL;
        for (int k = 0; k < laya_tok_vocab_count; k++) {
            if (laya_tok_vocab_id[k] == ids[i]) {
                piece = laya_tok_vocab_str[k];
                break;
            }
        }
        if (!piece) {
            for (int k = 0; k < laya_tok_added_count; k++) {
                if (laya_tok_added_id[k] == ids[i]) {
                    piece = laya_tok_added_str[k];
                    break;
                }
            }
        }
        if (!piece) continue;

        const char *p = piece;
        while (*p) {
            unsigned cp;
            size_t adv = utf8_decode(p, &cp);
            unsigned char byte = 0;
            for (int b = 0; b < 256; b++) {
                unsigned mapped;
                utf8_decode(laya_tok_byte_piece[b], &mapped);
                if (mapped == cp) {
                    byte = (unsigned char)b;
                    break;
                }
            }
            if (n + 2 > cap) {
                cap *= 2;
                char *nb = realloc(buf, cap);
                if (!nb) {
                    free(buf);
                    laya_set_error("tokenizer oom (decode grow)");
                    return LAYA_ERR_OOM;
                }
                buf = nb;
            }
            buf[n++] = (char)byte;
            p += adv;
        }
    }
    buf[n] = '\0';
    *out = buf;
    return LAYA_OK;
}

void laya_tokenizer_free(int *ids) { free(ids); }

/* ------------------------------------------------------------------ */
/* fixture check                                                       */
/* ------------------------------------------------------------------ */

int tokenize_fixture(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = malloc((size_t)sz + 1);
    if (!text) {
        fclose(f);
        fprintf(stderr, "error: oom\n");
        return 1;
    }
    if (fread(text, 1, (size_t)sz, f) != (size_t)sz) {
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
    if (!root || root->type != HD_JSON_OBJECT) {
        fprintf(stderr, "error: bad fixture: %s\n", err ? err : "not an object");
        if (root) hd_json_free(root);
        return 1;
    }

    size_t pass = 0, fail = 0;
    for (size_t i = 0; i < root->u.object.count; i++) {
        const char *key = root->u.object.keys[i];
        const hd_json *want = root->u.object.values[i];
        int *ids = NULL;
        size_t count = 0;
        if (laya_tokenizer_encode(key, &ids, &count) != LAYA_OK) {
            fprintf(stderr, "FAIL %-40s error: %s\n", key, laya_last_error());
            fail++;
            continue;
        }
        size_t wn = hd_json_array_len(want);
        int ok = (wn == count);
        for (size_t k = 0; ok && k < wn; k++)
            ok = hd_json_int(hd_json_array_at(want, k), -1) == ids[k];
        if (ok) {
            pass++;
        } else {
            fail++;
            printf("FAIL %s\n  want [", key);
            for (size_t k = 0; k < wn; k++)
                printf("%s%lld", k ? ", " : "", (long long)hd_json_int(hd_json_array_at(want, k), -1));
            printf("]\n  got  [");
            for (size_t k = 0; k < count; k++) printf("%s%d", k ? ", " : "", ids[k]);
            printf("]\n");
        }
        laya_tokenizer_free(ids);
    }
    hd_json_free(root);
    printf("tokenizer fixture: %zu passed, %zu failed\n", pass, fail);
    return fail ? 1 : 0;
}
