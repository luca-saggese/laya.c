#define _POSIX_C_SOURCE 200809L
#include "model.h"
#include "laya_timing.h"
#include "hd_cuda.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include <cuda_runtime.h>

/* Loader tuning copied from o1.c (hd_weights_to_device_gguf). */
#define LAYA_LOADER_STAGE_SLOTS 4
#define LAYA_LOADER_STAGE_BYTES (128u * 1024u * 1024u) /* 128 MiB per slot */
#define LAYA_GEMM_WORKSPACE (64u << 20)

static char g_error[512] = "";

const char *laya_last_error(void) { return g_error; }

void laya_set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
}

static void set_err(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
}

const char *laya_model_last_error(void) { return g_error; }

/* ------------------------------------------------------------------ */
/* dtype helpers (copied from o1.c src/model/model.c)                  */
/* ------------------------------------------------------------------ */

laya_dtype laya_dtype_from_string(const char *s) {
    if (!s) return LAYA_DTYPE_UNKNOWN;
    if (strstr(s, "bfloat16") || strstr(s, "bf16")) return LAYA_DTYPE_BF16;
    if (strstr(s, "float16") || strstr(s, "fp16") || strstr(s, "half"))
        return LAYA_DTYPE_F16;
    if (strstr(s, "float32") || strstr(s, "fp32")) return LAYA_DTYPE_F32;
    if (strstr(s, "float64") || strstr(s, "fp64")) return LAYA_DTYPE_F64;
    if (strstr(s, "int64")) return LAYA_DTYPE_I64;
    if (strstr(s, "int32")) return LAYA_DTYPE_I32;
    return LAYA_DTYPE_UNKNOWN;
}

const char *laya_dtype_to_string(laya_dtype d) {
    switch (d) {
        case LAYA_DTYPE_F32: return "float32";
        case LAYA_DTYPE_BF16: return "bfloat16";
        case LAYA_DTYPE_F16: return "float16";
        case LAYA_DTYPE_F64: return "float64";
        case LAYA_DTYPE_I64: return "int64";
        case LAYA_DTYPE_I32: return "int32";
        default: return "unknown";
    }
}

int64_t laya_numel(const int64_t *shape, int rank) {
    int64_t n = 1;
    for (int i = 0; i < rank; i++) n *= shape[i];
    return n;
}

/* ------------------------------------------------------------------ */
/* Config from GGUF metadata                                           */
/* ------------------------------------------------------------------ */

static void load_config(const laya_gguf_file *gf, laya_config *c) {
    memset(c, 0, sizeof(*c));
    c->hidden_size = (int)laya_gguf_kv_int(gf, "modernbert.hidden_size", 0);
    c->n_layers = (int)laya_gguf_kv_int(gf, "modernbert.num_hidden_layers", 0);
    c->n_heads = (int)laya_gguf_kv_int(gf, "modernbert.num_attention_heads", 0);
    c->intermediate_size = (int)laya_gguf_kv_int(gf, "modernbert.intermediate_size", 0);
    c->max_position_embeddings =
        (int)laya_gguf_kv_int(gf, "modernbert.max_position_embeddings", 8192);
    c->vocab_size = (int)laya_gguf_kv_int(gf, "modernbert.vocab_size", 0);
    c->local_attention = (int)laya_gguf_kv_int(gf, "modernbert.local_attention", 128);
    c->global_attn_every_n_layers =
        (int)laya_gguf_kv_int(gf, "modernbert.global_attn_every_n_layers", 3);
    c->layer_norm_eps = (float)laya_gguf_kv_double(gf, "modernbert.layer_norm_eps", 1e-5);
    c->global_rope_theta = (float)laya_gguf_kv_double(gf, "modernbert.global_rope_theta", 160000.0);
    c->local_rope_theta = (float)laya_gguf_kv_double(gf, "modernbert.local_rope_theta", 10000.0);
    c->cls_token_id = (int)laya_gguf_kv_int(gf, "modernbert.cls_token_id", 50281);
    c->sep_token_id = (int)laya_gguf_kv_int(gf, "modernbert.sep_token_id", 50282);
    c->mask_token_id = (int)laya_gguf_kv_int(gf, "modernbert.mask_token_id", 50284);
    c->pad_token_id = (int)laya_gguf_kv_int(gf, "modernbert.pad_token_id", 50283);

    c->max_len = (int)laya_gguf_kv_int(gf, "laya.max_len", 512);
    c->head_max_len = (int)laya_gguf_kv_int(gf, "laya.head_max_len", 192);
    c->n_head_layers = (int)laya_gguf_kv_int(gf, "laya.head_layers", 2);
    c->n_act = (int)laya_gguf_kv_int(gf, "laya.n_act", 2);
    c->head_dim = c->n_heads ? c->hidden_size / c->n_heads : 0;
    c->sliding_window = c->local_attention / 2;

    const laya_gguf_kv *t = laya_gguf_find_kv(gf, "laya.temperature");
    if (t && t->vals) {
        for (uint64_t i = 0; i < t->n && i < 3; i++) c->temperature[i] = (float)t->vals[i];
    }
    const laya_gguf_kv *b = laya_gguf_find_kv(gf, "laya.temperature_by_options");
    if (b && b->vals) {
        for (uint64_t i = 0; i < b->n && i < LAYA_TEMP_BUCKETS; i++)
            c->temperature_by_options[i] = (float)b->vals[i];
    }
    const laya_gguf_kv *g = laya_gguf_find_kv(gf, "laya.layer_is_global");
    if (g && g->vals) {
        for (uint64_t i = 0; i < g->n && i < LAYA_MAX_LAYERS; i++)
            c->layer_is_global[i] = (uint8_t)(g->vals[i] != 0.0);
    } else {
        for (int i = 0; i < c->n_layers && i < LAYA_MAX_LAYERS; i++)
            c->layer_is_global[i] =
                (uint8_t)(c->global_attn_every_n_layers > 0 &&
                          (i % c->global_attn_every_n_layers) == 0);
    }
}

/* ------------------------------------------------------------------ */
/* Tensor binding                                                      */
/* ------------------------------------------------------------------ */

static const laya_gguf_tensor *find_tensor(const laya_gguf_file *gf, const char *name) {
    for (int64_t i = 0; i < gf->n_tensors; i++) {
        if (strcmp(gf->tensors[i].name, name) == 0) return &gf->tensors[i];
    }
    return NULL;
}

/* Resolves a tensor to a resident device pointer. Every required tensor is
 * checked, so a missing name fails the load instead of crashing later. */
static laya_status bind_tensor(const laya_gguf_file *gf, const void *arena,
                               const char *name, laya_tensor *out) {
    const laya_gguf_tensor *t = find_tensor(gf, name);
    if (!t) {
        set_err("missing tensor %s", name);
        return LAYA_ERR_MISSING;
    }
    if (t->type != LAYA_GGML_TYPE_BF16 && t->type != LAYA_GGML_TYPE_F32) {
        set_err("tensor %s: unsupported type %u", name, t->type);
        return LAYA_ERR_UNSUPPORTED;
    }
    out->ptr = (uint8_t *)arena + t->offset;
    out->nbytes = (int64_t)t->nbytes;
    snprintf(out->name, sizeof(out->name), "%s", name);
    return LAYA_OK;
}


/* ------------------------------------------------------------------ */
/* Resident load (copied from o1.c hd_weights_to_device_gguf)          */
/* ------------------------------------------------------------------ */

static laya_status device_info(int device_id, laya_cuda_runtime *rt) {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) { set_err("cudaGetDeviceCount: %s", cudaGetErrorString(e)); return LAYA_ERR_IO; }
    if (device_id < 0 || device_id >= count) {
        set_err("device %d out of range (visible=%d)", device_id, count);
        return LAYA_ERR_PROFILE;
    }
    struct cudaDeviceProp prop;
    e = cudaGetDeviceProperties(&prop, device_id);
    if (e != cudaSuccess) { set_err("cudaGetDeviceProperties: %s", cudaGetErrorString(e)); return LAYA_ERR_IO; }
    rt->device_id = device_id;
    snprintf(rt->device_name, sizeof(rt->device_name), "%s", prop.name);
    rt->compute_major = prop.major;
    rt->compute_minor = prop.minor;
    rt->device_mem_bytes = (int64_t)prop.totalGlobalMem;
    return LAYA_OK;
}

static laya_status upload_payload(const char *gguf_path, const laya_gguf_file *gf,
                                  laya_cuda_runtime *rt) {
    uint8_t *stage[LAYA_LOADER_STAGE_SLOTS] = {0};
    cudaEvent_t ev[LAYA_LOADER_STAGE_SLOTS] = {0};
    laya_status st = LAYA_OK;
    cudaError_t e;

    for (int i = 0; i < LAYA_LOADER_STAGE_SLOTS; i++) {
        e = cudaMallocHost((void **)&stage[i], LAYA_LOADER_STAGE_BYTES);
        if (e != cudaSuccess) {
            set_err("cudaMallocHost stage %d: %s", i, cudaGetErrorString(e));
            st = LAYA_ERR_OOM;
            goto done;
        }
        cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming);
    }
    e = cudaStreamCreateWithFlags(&rt->upload_stream, cudaStreamNonBlocking);
    if (e != cudaSuccess) {
        set_err("cudaStreamCreate: %s", cudaGetErrorString(e));
        st = LAYA_ERR_OOM;
        goto done;
    }

    int fd = open(gguf_path, O_RDONLY);
    if (fd < 0) { set_err("open %s: %s", gguf_path, strerror(errno)); st = LAYA_ERR_IO; goto done; }

    LAYA_TIMING_BEGIN("FILE_READ");
    uint64_t payload = gf->payload_bytes;
    uint64_t off = 0;
    int slot = 0;
    while (off < payload) {
        size_t n = (size_t)((payload - off) < LAYA_LOADER_STAGE_BYTES
                                ? (payload - off) : LAYA_LOADER_STAGE_BYTES);
        if (off >= (uint64_t)LAYA_LOADER_STAGE_BYTES * LAYA_LOADER_STAGE_SLOTS)
            cudaEventSynchronize(ev[slot]);

        size_t got = 0;
        while (got < n) {
            ssize_t rd = pread(fd, stage[slot] + got, n - got,
                               (off_t)(gf->tensor_data_off + off + got));
            if (rd <= 0) {
                set_err("short read gguf (%zu/%zu)", got, n);
                close(fd);
                st = LAYA_ERR_IO;
                goto done;
            }
            got += (size_t)rd;
        }
        e = cudaMemcpyAsync((uint8_t *)rt->arena_ptr + off, stage[slot], n,
                            cudaMemcpyHostToDevice, rt->upload_stream);
        if (e != cudaSuccess) {
            set_err("cudaMemcpyAsync: %s", cudaGetErrorString(e));
            close(fd);
            st = LAYA_ERR_IO;
            goto done;
        }
        cudaEventRecord(ev[slot], rt->upload_stream);
        off += n;
        slot = (slot + 1) % LAYA_LOADER_STAGE_SLOTS;
    }
    LAYA_TIMING_END("FILE_READ");
    close(fd);

    e = cudaStreamSynchronize(rt->upload_stream);
    if (e != cudaSuccess) { set_err("upload stream sync: %s", cudaGetErrorString(e)); st = LAYA_ERR_IO; }

done:
    for (int i = 0; i < LAYA_LOADER_STAGE_SLOTS; i++) {
        if (ev[i]) cudaEventDestroy(ev[i]);
        if (stage[i]) cudaFreeHost(stage[i]);
    }
    return st;
}

/* ------------------------------------------------------------------ */
/* Load / free                                                         */
/* ------------------------------------------------------------------ */

laya_status laya_model_load(const char *gguf_path, int device_id, laya_model *out) {
    memset(out, 0, sizeof(*out));
    LAYA_TIMING_BEGIN("MODEL_LOAD");

    laya_status st = device_info(device_id, &out->cuda);
    if (st != LAYA_OK) return st;

    cudaError_t e = cudaSetDevice(device_id);
    if (e != cudaSuccess) { set_err("cudaSetDevice: %s", cudaGetErrorString(e)); return LAYA_ERR_IO; }

    st = laya_gguf_open(gguf_path, &out->gguf);
    if (st != LAYA_OK) { set_err("gguf: %s", laya_gguf_last_error()); return (laya_status)st; }

    const char *arch = out->gguf.arch ? out->gguf.arch : "";
    if (strcmp(arch, "laya") != 0) {
        set_err("%s: unexpected architecture '%s' (expected 'laya')", gguf_path, arch);
        laya_gguf_close(&out->gguf);
        return LAYA_ERR_PARSE;
    }

    load_config(&out->gguf, &out->cfg);
    const laya_config *c = &out->cfg;
    if (c->hidden_size <= 0 || c->n_layers <= 0 || c->n_heads <= 0 ||
        c->n_layers > LAYA_MAX_LAYERS || c->n_head_layers > LAYA_MAX_HEAD_LAYERS) {
        set_err("%s: implausible config (d=%d layers=%d heads=%d head_layers=%d)",
                gguf_path, c->hidden_size, c->n_layers, c->n_heads, c->n_head_layers);
        laya_gguf_close(&out->gguf);
        return LAYA_ERR_PARSE;
    }

    /* ---- one aligned CUDA arena (single cudaMalloc, as in o1.c) ---- */
    LAYA_TIMING_BEGIN("CUDA_ALLOC");
    e = cudaMalloc(&out->cuda.arena_ptr, (size_t)out->gguf.payload_bytes);
    LAYA_TIMING_END("CUDA_ALLOC");
    if (e != cudaSuccess) {
        set_err("cudaMalloc arena %llu bytes: %s",
                (unsigned long long)out->gguf.payload_bytes, cudaGetErrorString(e));
        laya_gguf_close(&out->gguf);
        return LAYA_ERR_OOM;
    }
    out->cuda.arena_bytes = (int64_t)out->gguf.payload_bytes;

    st = upload_payload(gguf_path, &out->gguf, &out->cuda);
    if (st != LAYA_OK) { laya_model_free(out); return st; }

    /* ---- bind native pointers ---- */
    const void *arena = out->cuda.arena_ptr;
    char name[160];

    st = bind_tensor(&out->gguf, arena, "encoder.embeddings.tok_embeddings.weight",
                     &out->tok_embeddings);
    if (st != LAYA_OK) goto fail;
    st = bind_tensor(&out->gguf, arena, "encoder.embeddings.norm.weight", &out->emb_norm);
    if (st != LAYA_OK) goto fail;
    st = bind_tensor(&out->gguf, arena, "encoder.final_norm.weight", &out->final_norm);
    if (st != LAYA_OK) goto fail;

    for (int i = 0; i < c->n_layers; i++) {
        laya_layer_weights *lw = &out->layers[i];
        lw->is_global = c->layer_is_global[i] ? 1 : 0;
        if (i == 0) {
            /* layer 0's attn_norm is nn.Identity(): no weight in the checkpoint. */
            memset(&lw->attn_norm, 0, sizeof(lw->attn_norm));
        } else {
            snprintf(name, sizeof(name), "encoder.layers.%02d.attn_norm.weight", i);
            st = bind_tensor(&out->gguf, arena, name, &lw->attn_norm);
            if (st != LAYA_OK) goto fail;
        }
        snprintf(name, sizeof(name), "encoder.layers.%02d.attn.Wqkv.weight", i);
        st = bind_tensor(&out->gguf, arena, name, &lw->Wqkv);
        if (st != LAYA_OK) goto fail;
        snprintf(name, sizeof(name), "encoder.layers.%02d.attn.Wo.weight", i);
        st = bind_tensor(&out->gguf, arena, name, &lw->Wo);
        if (st != LAYA_OK) goto fail;
        snprintf(name, sizeof(name), "encoder.layers.%02d.mlp_norm.weight", i);
        st = bind_tensor(&out->gguf, arena, name, &lw->mlp_norm);
        if (st != LAYA_OK) goto fail;
        snprintf(name, sizeof(name), "encoder.layers.%02d.mlp.Wi.weight", i);
        st = bind_tensor(&out->gguf, arena, name, &lw->Wi);
        if (st != LAYA_OK) goto fail;
        snprintf(name, sizeof(name), "encoder.layers.%02d.mlp.Wo.weight", i);
        st = bind_tensor(&out->gguf, arena, name, &lw->mlp_Wo);
        if (st != LAYA_OK) goto fail;
    }

    st = bind_tensor(&out->gguf, arena, "decision.type_emb.weight", &out->type_emb);
    if (st != LAYA_OK) goto fail;

    for (int i = 0; i < c->n_head_layers; i++) {
        laya_head_layer_weights *hw = &out->head[i];
        snprintf(name, sizeof(name), "decision.head.%02d.norm1.weight", i);
        st = bind_tensor(&out->gguf, arena, name, &hw->norm1_w);
        if (st != LAYA_OK) goto fail;
        snprintf(name, sizeof(name), "decision.head.%02d.norm1.bias", i);
        st = bind_tensor(&out->gguf, arena, name, &hw->norm1_b);
        if (st != LAYA_OK) goto fail;
        snprintf(name, sizeof(name), "decision.head.%02d.linear1.weight", i);
        st = bind_tensor(&out->gguf, arena, name, &hw->linear1_w);
        if (st != LAYA_OK) goto fail;
        snprintf(name, sizeof(name), "decision.head.%02d.linear1.bias", i);
        st = bind_tensor(&out->gguf, arena, name, &hw->linear1_b);
        if (st != LAYA_OK) goto fail;
        snprintf(name, sizeof(name), "decision.head.%02d.norm2.weight", i);
        st = bind_tensor(&out->gguf, arena, name, &hw->norm2_w);
        if (st != LAYA_OK) goto fail;
        snprintf(name, sizeof(name), "decision.head.%02d.norm2.bias", i);
        st = bind_tensor(&out->gguf, arena, name, &hw->norm2_b);
        if (st != LAYA_OK) goto fail;
        snprintf(name, sizeof(name), "decision.head.%02d.linear2.weight", i);
        st = bind_tensor(&out->gguf, arena, name, &hw->linear2_w);
        if (st != LAYA_OK) goto fail;
        snprintf(name, sizeof(name), "decision.head.%02d.linear2.bias", i);
        st = bind_tensor(&out->gguf, arena, name, &hw->linear2_b);
        if (st != LAYA_OK) goto fail;
        snprintf(name, sizeof(name), "decision.head.%02d.self_attn.in_proj_weight", i);
        st = bind_tensor(&out->gguf, arena, name, &hw->in_proj_w);
        if (st != LAYA_OK) goto fail;
        snprintf(name, sizeof(name), "decision.head.%02d.self_attn.in_proj_bias", i);
        st = bind_tensor(&out->gguf, arena, name, &hw->in_proj_b);
        if (st != LAYA_OK) goto fail;
        snprintf(name, sizeof(name), "decision.head.%02d.self_attn.out_proj.weight", i);
        st = bind_tensor(&out->gguf, arena, name, &hw->out_proj_w);
        if (st != LAYA_OK) goto fail;
        snprintf(name, sizeof(name), "decision.head.%02d.self_attn.out_proj.bias", i);
        st = bind_tensor(&out->gguf, arena, name, &hw->out_proj_b);
        if (st != LAYA_OK) goto fail;
    }

    st = bind_tensor(&out->gguf, arena, "decision.scorer.0.weight", &out->decision.scorer0_w);
    if (st != LAYA_OK) goto fail;
    st = bind_tensor(&out->gguf, arena, "decision.scorer.0.bias", &out->decision.scorer0_b);
    if (st != LAYA_OK) goto fail;
    st = bind_tensor(&out->gguf, arena, "decision.scorer.1.weight", &out->decision.scorer1_w);
    if (st != LAYA_OK) goto fail;
    st = bind_tensor(&out->gguf, arena, "decision.scorer.1.bias", &out->decision.scorer1_b);
    if (st != LAYA_OK) goto fail;
    st = bind_tensor(&out->gguf, arena, "decision.scorer.3.weight", &out->decision.scorer3_w);
    if (st != LAYA_OK) goto fail;
    st = bind_tensor(&out->gguf, arena, "decision.scorer.3.bias", &out->decision.scorer3_b);
    if (st != LAYA_OK) goto fail;
    st = bind_tensor(&out->gguf, arena, "decision.act_head.0.weight", &out->decision.act0_w);
    if (st != LAYA_OK) goto fail;
    st = bind_tensor(&out->gguf, arena, "decision.act_head.0.bias", &out->decision.act0_b);
    if (st != LAYA_OK) goto fail;
    st = bind_tensor(&out->gguf, arena, "decision.act_head.2.weight", &out->decision.act2_w);
    if (st != LAYA_OK) goto fail;
    st = bind_tensor(&out->gguf, arena, "decision.act_head.2.bias", &out->decision.act2_b);
    if (st != LAYA_OK) goto fail;

    out->cuda.n_allocs = out->gguf.n_tensors;
    out->cuda.device_bytes = (int64_t)out->gguf.payload_bytes;

    out->cuda.gemm = laya_gemm_runtime_init(device_id, LAYA_GEMM_WORKSPACE);
    if (!out->cuda.gemm) {
        set_err("gemm runtime init failed: %s", laya_cuda_last_error());
        goto fail;
    }

    LAYA_TIMING_END("MODEL_LOAD");
    return LAYA_OK;

fail:
    laya_model_free(out);
    return st;
}

void laya_model_report(const laya_model *m) {
    const laya_config *c = &m->cfg;
    printf("model:            %s\n", m->gguf.path ? m->gguf.path : "(none)");
    printf("architecture:     %s (variant %s)\n",
           m->gguf.arch ? m->gguf.arch : "?",
           laya_gguf_kv_str(&m->gguf, "laya.variant") ? laya_gguf_kv_str(&m->gguf, "laya.variant") : "?");
    printf("dtype:            %s\n",
           laya_gguf_kv_str(&m->gguf, "laya.dtype") ? laya_gguf_kv_str(&m->gguf, "laya.dtype") : "?");
    printf("device:           %s (sm_%d%d), %.1f GiB total\n", m->cuda.device_name,
           m->cuda.compute_major, m->cuda.compute_minor,
           (double)m->cuda.device_mem_bytes / 1073741824.0);
    printf("tensors:          %lld in gguf, %lld resident\n",
           (long long)m->gguf.n_tensors, (long long)m->cuda.n_allocs);
    printf("resident bytes:   %.1f MB (arena %.1f MB)\n",
           (double)m->cuda.device_bytes / 1e6, (double)m->cuda.arena_bytes / 1e6);
    printf("encoder:          d=%d layers=%d heads=%d head_dim=%d inter=%d vocab=%d\n",
           c->hidden_size, c->n_layers, c->n_heads, c->head_dim,
           c->intermediate_size, c->vocab_size);
    printf("attention:        local=%d (window=%d), global every %d layers\n",
           c->local_attention, c->sliding_window, c->global_attn_every_n_layers);
    printf("rope theta:       global=%.1f local=%.1f\n",
           (double)c->global_rope_theta, (double)c->local_rope_theta);
    printf("special ids:      cls=%d sep=%d mask=%d pad=%d\n",
           c->cls_token_id, c->sep_token_id, c->mask_token_id, c->pad_token_id);
    printf("decision head:    %d layers, n_act=%d, max_len=%d, head_max_len=%d\n",
           c->n_head_layers, c->n_act, c->max_len, c->head_max_len);
    printf("temperature:      [%.6f %.6f %.6f]\n",
           (double)c->temperature[0], (double)c->temperature[1], (double)c->temperature[2]);
    int n_global = 0;
    for (int i = 0; i < c->n_layers; i++) n_global += c->layer_is_global[i] ? 1 : 0;
    printf("layer types:      %d global, %d sliding\n", n_global, c->n_layers - n_global);
    printf("metadata keys:    %lld retained\n", (long long)m->gguf.n_kvs);
}

void laya_model_free(laya_model *m) {
    if (!m) return;
    if (m->cuda.gemm) laya_gemm_runtime_destroy(m->cuda.gemm);
    if (m->cuda.upload_stream) cudaStreamDestroy(m->cuda.upload_stream);
    if (m->cuda.arena_ptr) cudaFree(m->cuda.arena_ptr);
    laya_gguf_close(&m->gguf);
    memset(&m->cuda, 0, sizeof(m->cuda));
}
