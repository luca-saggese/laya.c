#ifndef LAYA_SEQUENCE_H
#define LAYA_SEQUENCE_H

/*
 * Laya sequence construction.
 *
 * This is a direct C translation of ``build_sequence`` /
 * ``render_options`` / ``render_criterion`` from the original Laya
 * ``laya/common.py``. It is intentionally a transliteration: same option
 * rendering, same ordering, same head/option/state truncation and the same
 * marker positions, so token arrays can be compared with the Python oracle
 * entry by entry.
 */

#include <stddef.h>

#include "laya.h"
#include "hd_json.h"

/* QTYPES from laya/common.py */
enum {
    LAYA_QTYPE_CHOICE = 0,
    LAYA_QTYPE_SCORE = 1,
    LAYA_QTYPE_NOUL = 2,
};

typedef struct {
    /* rendered option texts, in label-index order */
    char **opts;
    size_t n_opts;

    /* tokenized option rows: row i starts with the mask token */
    int **opt_ids;
    size_t *opt_len;
} laya_options;

/* ``render_criterion``: strings pass through, anything structured becomes
 * compact JSON with ", " / ": " separators. Caller frees. */
char *laya_render_criterion(const hd_json *value);

/* ``render_options``: choice -> "k: crit", score -> "level i: crit",
 * noul -> ["false: ...", "true: ..."]. Caller frees with
 * laya_options_free. */
void laya_render_options(const hd_json *q, laya_options *out);

void laya_options_free(laya_options *o);

/* ``build_sequence``. `q` is the internal question dict produced by
 * laya_question_internal (keys "t", "ins", "crit"). On success *out_ids is
 * malloc'd and *out_markers holds the [MASK] positions. */
laya_status laya_build_sequence(const hd_json *state, const hd_json *q,
                                int max_len, int head_max_len,
                                int **out_ids, size_t *out_count,
                                int **out_markers, size_t *out_n_markers);

/* Agent._to_internal: {"type","instructions","criteria"} -> {"t","ins","crit"}.
 * Caller frees the returned object with hd_json_free. */
hd_json *laya_question_internal(const hd_json *qdef);

/* qtype index for a "type" string, or -1. */
int laya_qtype_of(const char *type);

/* Parses a request file ({"name","state","questions"}) and prints the ids
 * and marker positions for every question, for oracle comparison. */
int build_sequence_fixture(const char *path);

#endif /* LAYA_SEQUENCE_H */
