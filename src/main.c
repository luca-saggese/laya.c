/*
 * Laya native CLI.
 *
 * M2.1 scope: open the GGUF pack, load every weight into resident VRAM and
 * report the bindings (`--inspect`). No inference yet.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "laya.h"
#include "model.h"
#include "laya_timing.h"

static void usage(const char *argv0) {
    printf("usage: %s [options]\n", argv0);
    printf("\n");
    printf("  --model PATH            Laya GGUF pack (required)\n");
    printf("  --device N              CUDA device index (default: 0)\n");
    printf("  --inspect               load the model, print the resident summary, exit\n");
    printf("  --timing                print the timing report on exit\n");
    printf("  -h, --help              show this help\n");
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    int device_id = 0;
    int inspect = 0;
    int timing = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--inspect") == 0) {
            inspect = 1;
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

    if (!model_path) {
        fprintf(stderr, "%s: --model is required\n", argv[0]);
        usage(argv[0]);
        return 2;
    }
    if (!inspect) {
        fprintf(stderr, "%s: nothing to do (use --inspect)\n", argv[0]);
        return 2;
    }

    laya_model model;
    laya_status st = laya_model_load(model_path, device_id, &model);
    if (st != LAYA_OK) {
        fprintf(stderr, "error: %s\n", laya_model_last_error());
        return 1;
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
