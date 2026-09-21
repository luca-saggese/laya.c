/*
 * Laya native CLI.
 *
 * M2.1 scope: open the GGUF pack, load every weight into resident VRAM and
 * report the bindings (`--inspect`). No inference yet.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "laya.h"
#include "hd_cuda.h"
#include "model.h"
#include "modernbert.h"
#include "laya_tokenizer.h"
#include "laya_sequence.h"
#include "laya_timing.h"

/* `--encode` parity harness (laya_encode.c): runs the native encoder on one
 * request and dumps the hidden states the Python oracle can dump too. */
int encode_request(const laya_model *model, const char *request_path,
                   const char *dump_spec, const char *out_path);

/* `--infer` end-to-end harness (laya_infer.c): encoder + decision head. */
int infer_request(const laya_model *model, const char *request_path,
                  const char *dump_spec, const char *out_path);

static void usage(const char *argv0) {
    printf("usage: %s [options]\n", argv0);
    printf("\n");
    printf("  --model PATH            Laya GGUF pack (required)\n");
    printf("  --device N              CUDA device index (default: 0)\n");
    printf("  --inspect               load the model, print the resident summary, exit\n");
    printf("  --tokenize TEXT         encode TEXT and print the token ids\n");
    printf("  --tokenize-file PATH    encode every JSON string in PATH's \"texts\" array\n");
    printf("  --sequence-file PATH    build sequences for every question in PATH\n");
    printf("  --encode                run the ModernBERT encoder on --request and dump it\n");
    printf("  --infer                 full native inference (encoder + decision head)\n");
    printf("  --request PATH          one request object (JSON) for --encode\n");
    printf("  --dump SPEC             comma list: embeddings,layer:0,layer:1,final\n");
    printf("  --out PATH              write the --encode JSON here (default: stdout)\n");
    printf("  --timing                print the timing report on exit\n");
    printf("  -h, --help              show this help\n");
}

static int tokenize_text(const char *text) {
    int *ids = NULL;
    size_t count = 0;
    laya_status st = laya_tokenizer_encode(text, &ids, &count);
    if (st != LAYA_OK) {
        fprintf(stderr, "error: %s\n", laya_last_error());
        return 1;
    }
    printf("[");
    for (size_t i = 0; i < count; i++) printf("%s%d", i ? ", " : "", ids[i]);
    printf("]\n");
    laya_tokenizer_free(ids);
    return 0;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *tokenize_file = NULL;
    const char *sequence_file = NULL;
    const char *tokenize_text_arg = NULL;
    const char *request_path = NULL;
    const char *dump_spec = NULL;
    const char *out_path = NULL;
    int device_id = 0;
    int inspect = 0;
    int timing = 0;
    int encode = 0;
    int infer = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--inspect") == 0) {
            inspect = 1;
        } else if (strcmp(argv[i], "--tokenize") == 0 && i + 1 < argc) {
            tokenize_text_arg = argv[++i];
        } else if (strcmp(argv[i], "--tokenize-file") == 0 && i + 1 < argc) {
            tokenize_file = argv[++i];
        } else if (strcmp(argv[i], "--sequence-file") == 0 && i + 1 < argc) {
            sequence_file = argv[++i];
        } else if (strcmp(argv[i], "--encode") == 0) {
            encode = 1;
        } else if (strcmp(argv[i], "--infer") == 0) {
            infer = 1;
        } else if (strcmp(argv[i], "--request") == 0 && i + 1 < argc) {
            request_path = argv[++i];
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_spec = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--timing") == 0) {
            timing = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "%s: unknown argument '%s'\n", argv[0], argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    if (tokenize_text_arg) return tokenize_text(tokenize_text_arg);
    if (tokenize_file) return tokenize_fixture(tokenize_file);
    if (sequence_file) return build_sequence_fixture(sequence_file);

    if (!model_path) {
        fprintf(stderr, "%s: --model is required\n", argv[0]);
        usage(argv[0]);
        return 2;
    }
    if (!inspect && !encode && !infer) {
        fprintf(stderr, "%s: nothing to do (use --inspect, --encode or --infer)\n", argv[0]);
        return 2;
    }

    laya_model model;
    laya_status st = laya_model_load(model_path, device_id, &model);
    if (st != LAYA_OK) {
        fprintf(stderr, "error: %s\n", laya_model_last_error());
        return 1;
    }

    if (encode) {
        int rc = encode_request(&model, request_path, dump_spec, out_path);
        laya_model_free(&model);
        return rc;
    }

    if (infer) {
        int rc = infer_request(&model, request_path, dump_spec, out_path);
        laya_model_free(&model);
        return rc;
    }

    laya_model_report(&model);
    printf("status:           loaded and resident\n");

    laya_model_free(&model);

#ifdef LAYA_DEBUG_TIMING
    if (timing) laya_timing_report(NULL);
#else
    (void)timing;
#endif
    return 0;
}
