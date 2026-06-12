# Example results

Run:
```bash
python benchmarks/run_end_to_end.py
python benchmarks/benchmark_cpu_gpu.py
```

Qualitative expectations:
- LS channel estimate works.
- MMSE has finite output and often lower denoised channel MSE.
- GPU RawKernel output matches CPU output.
- Speedup depends on GPU, transfer overhead, and whether data is already GPU-resident.
