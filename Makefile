# Laya native C/CUDA inference engine (DGX Spark / GB10)
#
# Build structure copied from _reference/o1.c: same CFLAGS/CPPFLAGS layout,
# same CUDA object directory, same link line. Only the source lists and the
# binary name are Laya-specific.

CC      ?= gcc
NVCC    ?= nvcc
CFLAGS  ?= -O2 -g -std=c11 -Wall -Wextra
CPPFLAGS += -Iinclude -Isrc/io -Isrc/laya -Isrc/cuda -Isrc/runtime -Isrc/tokenizer

CUDA_HOME  ?= /usr/local/cuda
CUDA_CPPFLAGS := -I$(CUDA_HOME)/include
CUDA_LDFLAGS  := -L$(CUDA_HOME)/lib64 -lcudart
CUBLAS_LDFLAGS := -lcublas -lcublasLt

# cuDNN runtime comes from the nvidia-cudnn pip package (only the .so is
# linked; Python is never used at runtime). Override CUDNN_HOME for a native
# cuDNN install.
CUDNN_HOME ?= /home/lvx/.local/lib/python3.12/site-packages/nvidia/cudnn
CUDNN_CPPFLAGS := -I$(CUDNN_HOME)/include
CUDNN_LDFLAGS  := -L$(CUDNN_HOME)/lib -lcudnn

TARGET_ARCH ?= sm_121

CORE_SRCS := src/io/hd_json.c src/io/hd_gguf.c src/laya/model.c src/laya/laya_sequence.c src/tokenizer/laya_tokenizer.c
SRCS      := src/main.c $(CORE_SRCS) src/runtime/laya_timing.c
OBJS      := $(SRCS:.c=.o)

BIN      := build/laya
CUBIN    := build/obj/cuda
CUDA_OBJS := $(CUBIN)/support.o $(CUBIN)/hd_gemm.o

.PHONY: all laya-spark clean

all: $(BIN)

# Primary target: NVIDIA DGX Spark / GB10 (CUDA sm_121).
laya-spark: $(BIN)

$(BIN): $(OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CUDA_CPPFLAGS) -MMD -MP -c -o $@ $<

$(CUBIN)/%.o: src/cuda/%.cu
	@mkdir -p $(CUBIN)
	$(NVCC) -arch=$(TARGET_ARCH) -O2 -std=c++17 $(CPPFLAGS) $(CUDA_CPPFLAGS) -MMD -MP -c -o $@ $<

clean:
	rm -rf build
	find src tests -name '*.o' -delete

-include $(SRCS:.c=.d)
-include $(wildcard $(CUBIN)/*.d)
