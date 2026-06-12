import time, numpy as np
from airan_phy_lab.cuda.raw_kernels import cupy_available, complex_scale_gpu

def main():
    n=2_000_000; rng=np.random.default_rng(0); x=(rng.normal(size=n)+1j*rng.normal(size=n)).astype(np.complex64); scale=0.5
    t0=time.perf_counter(); y_cpu=scale*x; t_cpu=time.perf_counter()-t0; print(f'CPU NumPy scale: {t_cpu*1e3:.2f} ms')
    if not cupy_available(): print('CuPy not installed. Skipping GPU benchmark.'); return
    import cupy as cp
    y_gpu=complex_scale_gpu(x,scale); cp.cuda.Stream.null.synchronize()
    t0=time.perf_counter(); y_gpu=complex_scale_gpu(x,scale); cp.cuda.Stream.null.synchronize(); t_gpu=time.perf_counter()-t0
    print(f'GPU CUDA RawKernel scale: {t_gpu*1e3:.2f} ms'); print(f'Speedup: {t_cpu/t_gpu:.2f}x'); print(f'Max error: {np.max(np.abs(y_cpu-cp.asnumpy(y_gpu))):.3e}')
if __name__=='__main__': main()
