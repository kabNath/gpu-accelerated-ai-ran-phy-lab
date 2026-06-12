"""GPU correctness tests. All self-skip when no GPU / no built binary, so the
suite stays green on a CPU-only laptop and meaningful on a workstation."""
import os, subprocess
import numpy as np
import pytest

from airan_phy_lab.estimators import mmse_channel_estimate
from airan_phy_lab.cuda.mmse_kernels import cupy_available, mmse_channel_estimate_gpu

CUDA_BIN = os.path.join(os.path.dirname(__file__), "..", "csrc", "mmse_est")


@pytest.mark.skipif(not cupy_available(), reason="CuPy/GPU not available")
def test_gpu_mmse_matches_cpu():
    for n in (64, 256, 512):
        rng = np.random.default_rng(n)
        h_ls = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
        ref = mmse_channel_estimate(h_ls, 0.1)
        got = mmse_channel_estimate_gpu(h_ls, 0.1)
        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        assert rel < 1e-3, f"n={n}: GPU vs CPU rel err {rel:.2e}"


@pytest.mark.skipif(not (cupy_available() and os.path.exists(CUDA_BIN)),
                    reason="GPU or csrc/mmse_est not available")
def test_cuda_binary_matches_cpu(tmp_path):
    n, B = 64, 8
    rng = np.random.default_rng(7)
    h_ls = (rng.normal(size=(n, B)) + 1j * rng.normal(size=(n, B))).astype(np.complex64)
    in_f, out_f = tmp_path / "in.bin", tmp_path / "out.bin"
    h_ls.flatten(order="F").tofile(in_f)                  # column-major n x B
    subprocess.run([CUDA_BIN, "--n", str(n), "--B", str(B), "--nv", "0.1",
                    "--in", str(in_f), "--out", str(out_f)], check=True)
    cuda = np.fromfile(out_f, dtype=np.complex64).reshape(n, B, order="F")
    ref = np.stack([mmse_channel_estimate(h_ls[:, b], 0.1) for b in range(B)], axis=1)
    rel = np.linalg.norm(cuda - ref) / np.linalg.norm(ref)
    assert rel < 1e-3, f"CUDA binary vs CPU rel err {rel:.2e}"
