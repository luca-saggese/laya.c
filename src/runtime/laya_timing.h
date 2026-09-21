/*
 * M2 pre-baseline timing instrumentation (compile-time gated).
 *
 * With -DLAYA_DEBUG_TIMING the LAYA_TIMING_* macros record wall-clock (CPU) and
 * CUDA-event (GPU) region timings into a process-wide accumulator. Without
 * the flag every macro compiles to a no-op, so production builds contain
 * zero timing code and zero timing output.
 *
 * Region kinds:
 *   LAYA_TIMING_BEGIN/END        CPU wall-clock region (clock_gettime)
 *   LAYA_TIMING_BEGIN_GPU/END_GPU  GPU region (CUDA events, no per-region sync)
 *   LAYA_TIMING_ADD              accumulate an externally measured value (s)
 *
 * laya_timing_report(path) synchronizes the device, folds all pending GPU
 * event pairs into the accumulators, prints a compact summary to stdout and
 * writes a machine-readable JSON dump to `path` (NULL to skip the file).
 */
#ifndef LAYA_TIMING_H
#define LAYA_TIMING_H

#ifdef LAYA_DEBUG_TIMING

void laya_timing_reset(void);
void laya_timing_begin_cpu(const char *name);
void laya_timing_end_cpu(const char *name);
void laya_timing_begin_gpu(const char *name);
void laya_timing_end_gpu(const char *name);
void laya_timing_add(const char *name, double seconds);
void laya_timing_add_gpu(const char *name, double seconds);
void laya_timing_counter_add(const char *name, double value);
void laya_timing_counter_set(const char *name, double value);
double laya_timing_region_seconds(const char *name);
void laya_timing_report(const char *json_path);

#define LAYA_TIMING_BEGIN(name) laya_timing_begin_cpu(name)
#define LAYA_TIMING_END(name) laya_timing_end_cpu(name)
#define LAYA_TIMING_BEGIN_GPU(name) laya_timing_begin_gpu(name)
#define LAYA_TIMING_END_GPU(name) laya_timing_end_gpu(name)
#define LAYA_TIMING_ADD(name, value) laya_timing_add(name, value)
#define LAYA_TIMING_ADD_GPU(name, value) laya_timing_add_gpu(name, value)
#define LAYA_TIMING_COUNTER_ADD(name, value) laya_timing_counter_add(name, value)
#define LAYA_TIMING_COUNTER_SET(name, value) laya_timing_counter_set(name, value)

/*
 * Fine-grained decoder-block sub-stage timing. Gated separately from
 * LAYA_DEBUG_TIMING because it records ~20 CUDA event pairs per block
 * invocation (36 blocks x 28 steps = ~20k pairs), which is acceptable for
 * the B1-B4 microbenchmarks but adds measurable host overhead to a full
 * end-to-end run. Enable with -DLAYA_DEBUG_BLOCK_TIMING (implies
 * -DLAYA_DEBUG_TIMING).
 */
#ifdef LAYA_DEBUG_BLOCK_TIMING
#define LAYA_BTIMING_BEGIN_GPU(name) laya_timing_begin_gpu(name)
#define LAYA_BTIMING_END_GPU(name) laya_timing_end_gpu(name)
#else
#define LAYA_BTIMING_BEGIN_GPU(name) ((void)0)
#define LAYA_BTIMING_END_GPU(name) ((void)0)
#endif

#else /* !LAYA_DEBUG_TIMING */

#define LAYA_TIMING_BEGIN(name) ((void)0)
#define LAYA_TIMING_END(name) ((void)0)
#define LAYA_TIMING_BEGIN_GPU(name) ((void)0)
#define LAYA_TIMING_END_GPU(name) ((void)0)
#define LAYA_TIMING_ADD(name, value) ((void)0)
#define LAYA_TIMING_ADD_GPU(name, value) ((void)0)
#define LAYA_TIMING_COUNTER_ADD(name, value) ((void)0)
#define LAYA_TIMING_COUNTER_SET(name, value) ((void)0)
#define LAYA_BTIMING_BEGIN_GPU(name) ((void)0)
#define LAYA_BTIMING_END_GPU(name) ((void)0)

#endif /* LAYA_DEBUG_TIMING */

#endif /* LAYA_TIMING_H */