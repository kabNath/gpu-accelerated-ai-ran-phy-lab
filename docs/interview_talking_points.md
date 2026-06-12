# Interview talking points

## 30-second pitch
I built a compact AI-RAN PHY lab showing OFDM transmission, LS/MMSE channel estimation, OLLA link adaptation, a PPO research scaffold, and a CUDA RawKernel benchmark. The project bridges classical wireless PHY and GPU-accelerated AI-RAN research.

## Why MMSE over LS?
LS is unbiased but noisy. MMSE uses channel correlation and noise variance to project the estimate toward the likely channel subspace.

## Why OLLA baseline?
OLLA is a strong industrial baseline. Any DRL method should be compared against it honestly.

## Why CUDA kernel?
CuPy demonstrates acceleration quickly, but NVIDIA roles value kernel launch overhead, memory bandwidth, occupancy, and profiling literacy.
