"""Real GPU PHY kernels for channel estimation.

Replaces the placeholder `complex_scale_kernel` with kernels that do actual
channel-estimation work:

  * build_corr_kernel : power-delay profile -> frequency correlation matrix R
                        (compute-bound: one sincos per tap per matrix entry)
  * ls_divide_kernel  : elementwise complex LS division Y ./ X

The dense solve and matmul that turn R into the MMSE estimate are delegated to
CuPy's cuSolver/cuBLAS backends (`cp.linalg.solve`, `@`). The GPU MMSE here is
numerically identical to `airan_phy_lab.estimators.mmse_channel_estimate`
(verified to < 1e-4 relative error in tests/test_cuda_mmse.py), so the CPU
result is a real ground truth for the GPU path.
"""
from __future__ import annotations
import numpy as np

# --- Real CUDA C++ source, compiled at runtime by CuPy RawModule -------------
_KERNEL_SRC = r'''
extern "C" __global__
void build_corr_kernel(float2* R, const float* pdp, int n, int n_taps)
{
    // One thread per (row, col) entry of the n x n correlation matrix.
    // R[r,c] = sum_l pdp[l] * exp(-j 2pi (r-c) l / n)
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= n || c >= n) return;
    float dk = (float)(r - c);
    float re = 0.f, im = 0.f;
    const float TWO_PI = 6.283185307179586f;
    for (int l = 0; l < n_taps; ++l) {
        float theta = -TWO_PI * dk * (float)l / (float)n;
        float s, co; __sincosf(theta, &s, &co);
        re += pdp[l] * co;
        im += pdp[l] * s;
    }
    R[r * n + c] = make_float2(re, im);   // row-major (C order)
}

extern "C" __global__
void ls_divide_kernel(const float2* Y, const float2* X, float2* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float2 y = Y[i], x = X[i];
    float den = x.x * x.x + x.y * x.y + 1e-12f;
    out[i] = make_float2((y.x * x.x + y.y * x.y) / den,
                         (y.y * x.x - y.x * x.y) / den);
}
'''

_module = None


def cupy_available() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def _get_module():
    global _module
    import cupy as cp
    if _module is None:
        _module = cp.RawModule(code=_KERNEL_SRC, options=('--std=c++11',))
    return _module


def _pdp(n_taps: int, decay: float):
    p = np.exp(-decay * np.arange(n_taps)).astype(np.float32)
    return p / p.sum()


def frequency_correlation_matrix_gpu(n_subcarriers: int, n_taps: int = 8, decay: float = 0.5):
    """Build R on the GPU with the custom kernel. Returns a CuPy (n, n) complex64."""
    import cupy as cp
    R = cp.empty((n_subcarriers, n_subcarriers), dtype=cp.complex64)
    pdp = cp.asarray(_pdp(n_taps, decay))
    k = _get_module().get_function('build_corr_kernel')
    block = (16, 16)
    grid = ((n_subcarriers + 15) // 16, (n_subcarriers + 15) // 16)
    k(grid, block, (R, pdp, np.int32(n_subcarriers), np.int32(n_taps)))
    return R


def mmse_channel_estimate_gpu(h_ls, noise_variance: float, n_taps: int = 8, decay: float = 0.5):
    """GPU MMSE estimate; mirrors estimators.mmse_channel_estimate exactly.

    Uses the custom build_corr kernel for R, then a cuSolver-backed Hermitian
    solve (R @ A^{-1} h_ls) instead of forming the full Wiener matrix.
    """
    import cupy as cp
    h = cp.asarray(h_ls, dtype=cp.complex64)
    n = h.size
    R = frequency_correlation_matrix_gpu(n, n_taps, decay)
    A = R + cp.float32(noise_variance) * cp.eye(n, dtype=cp.complex64)
    h_mmse = R @ cp.linalg.solve(A, h)          # = (R A^{-1}) h_ls
    return cp.asnumpy(h_mmse).astype(np.complex64)


def ls_pilot_divide_gpu(y_pilots, x_pilots):
    """Elementwise LS division on the GPU via the custom kernel."""
    import cupy as cp
    Y = cp.asarray(y_pilots, dtype=cp.complex64).ravel()
    X = cp.asarray(x_pilots, dtype=cp.complex64).ravel()
    out = cp.empty_like(Y)
    k = _get_module().get_function('ls_divide_kernel')
    block = (256,)
    grid = ((Y.size + 255) // 256,)
    k(grid, block, (Y, X, out, np.int32(Y.size)))
    return cp.asnumpy(out)
