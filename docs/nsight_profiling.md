# Nsight profiling guide

## Nsight Systems
```bash
nsys profile -o results/nsys_complex_scale python benchmarks/benchmark_cpu_gpu.py
```

## Nsight Compute
```bash
ncu --set full -o results/ncu_complex_scale python benchmarks/benchmark_cpu_gpu.py
```

Look for achieved occupancy, memory throughput, global load/store efficiency, warp execution efficiency, kernel launch overhead, and synchronization points.

Next optimization: fuse pilot LS estimation, interpolation, equalization and hard demod into fewer kernels to reduce global memory traffic.
