# csrc/Makefile — build the standalone CUDA MMSE estimator.
# ARCH: sm_75 (T4) | sm_80 (A100) | sm_89 (RTX 4090)
ARCH ?= sm_89
NVCC ?= nvcc
FLAGS = -O3 -std=c++14 -arch=$(ARCH) -lineinfo
LIBS  = -lcublas -lcusolver

mmse_est: mmse_channel_est.cu
	$(NVCC) $(FLAGS) $< -o $@ $(LIBS)

# Compile-only (no GPU needed) — used by CI to prove the CUDA builds.
compile-check: mmse_channel_est.cu
	$(NVCC) $(FLAGS) -c mmse_channel_est.cu -o /tmp/mmse_est.o

clean:
	rm -f mmse_est /tmp/mmse_est.o

.PHONY: compile-check clean
