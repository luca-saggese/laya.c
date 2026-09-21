#ifndef LAYA_INFER_H
#define LAYA_INFER_H

/*
 * Batched Laya decision inference entry point.
 *
 * One parsed System One request ({"state", "questions"}) becomes
 *   - one Laya sequence per question,
 *   - ONE batched native forward,
 *   - one typed result per question.
 *
 * This is the shared runtime core used by the `--infer` CLI harness and by
 * the HTTP server; neither of them is allowed to build its own forward.
 */

#include "hd_json.h"
#include "laya.h"
#include "model.h"

/* Upper bound on questions per request (wire level). */
#define LAYA_INFER_MAX_Q 100

typedef struct {
    int qtype;              /* LAYA_QTYPE_* */

    int n_options;
    char **option_keys;     /* choice: criteria keys; score: "0".."N"; noul: false/true */
    char **option_labels;   /* rendered criterion / level text, may be NULL */

    int choice;             /* qtype == choice */
    float score;            /* qtype == score  */
    float noul;             /* qtype == noul: P(true) */
    float confidence;
    float act_probability;

    float *probs;           /* [n_options] calibrated */
    float *raw_logits;      /* [n_options] uncalibrated */
} laya_infer_result;

/* Runs all questions of `root` in one batched forward. On success
 * *out_res is a malloc'd array of *out_n results (free with
 * laya_infer_results_free) and *out_input_tokens receives the total number
 * of real (unpadded) tokens actually processed. */
laya_status laya_infer_run(const laya_model *model, const hd_json *root,
                           laya_infer_result **out_res, int *out_n,
                           long *out_input_tokens);

void laya_infer_results_free(laya_infer_result *res, int n);

/* Full `--infer` harness for a request file. */
int infer_request(const laya_model *model, const char *request_path,
                  const char *dump_spec, const char *out_path);

#endif /* LAYA_INFER_H */
