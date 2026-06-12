"""CPU vs GPU benchmark on the REAL channel-estimation workload.

Replaces the previous scalar-scaling demo (multiply-by-0.5), which measured
memory bandwidth, not PHY. The MMSE estimator is dominated by the N x N
Hermitian solve + matmul, which is where the GPU advantage is real and grows
with the number of subcarriers.

    python benchmarks/benchmark_cpu_gpu.py            # CPU + GPU if available
    python benchmarks/benchmark_cpu_gpu.py --sizes 256 512 1024

Numbers print to stdout; paste them into results/ rather than hard-coding any
speedup you cannot reproduce.
"""
import argparse, time
import numpy as np
from airan_phy_lab.estimators import mmse_channel_estimate
from airan_phy_lab.cuda.mmse_kernels import cupy_available, mmse_channel_estimate_gpu

NV = 0.1


def time_call(fn, reps):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter(); fn(); best = min(best, (time.perf_counter() - t) * 1e3)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    have_gpu = cupy_available()
    if not have_gpu:
        print("CuPy not installed - CPU-only run.\n")
    print(f"{'N':>6} | {'CPU (ms)':>9} | {'GPU (ms)':>9} | {'speedup':>7} | {'rel.err':>9}")
    print("-" * 54)
    for n in args.sizes:
        rng = np.random.default_rng(0)
        h_ls = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
        cpu = time_call(lambda: mmse_channel_estimate(h_ls, NV), args.reps)
        if have_gpu:
            import cupy as cp
            mmse_channel_estimate_gpu(h_ls, NV)            # warmup (JIT compile kernel)
            cp.cuda.Stream.null.synchronize()
            gpu = time_call(lambda: (mmse_channel_estimate_gpu(h_ls, NV),
                                     cp.cuda.Stream.null.synchronize()), args.reps)
            ref = mmse_channel_estimate(h_ls, NV)
            got = mmse_channel_estimate_gpu(h_ls, NV)
            rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
            print(f"{n:>6} | {cpu:>9.3f} | {gpu:>9.3f} | {cpu/gpu:>6.2f}x | {rel:>9.2e}")
        else:
            print(f"{n:>6} | {cpu:>9.3f} | {'-':>9} | {'-':>7} | {'-':>9}")


if __name__ == "__main__":
    main()
