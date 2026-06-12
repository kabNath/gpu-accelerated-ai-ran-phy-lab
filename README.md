# OFDM PHY + AI-RAN Lab

[![tests](https://github.com/kabNath/gpu-accelerated-ai-ran-phy-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/kabNath/gpu-accelerated-ai-ran-phy-lab/actions/workflows/tests.yml)

A compact, honest wireless-PHY research stack for AI-RAN / 6G work:

- **GPU-accelerated MMSE channel estimation** with hand-written CUDA kernels +
  cuSolver/cuBLAS — **up to 133× faster than NumPy** on an RTX 4090, verified
  correct against a CPU reference (~1e-4 relative error).
- **A 5G link-level BLER simulator** (NVIDIA Sionna): OFDM + 5G LDPC over a
  3GPP TR38.901 TDL channel, swept over MCS × SNR.
- **Link adaptation** (OLLA + a PPO agent) that consumes the **measured Sionna
  BLER curves** instead of a hand-tuned analytic model.

The goal isn't to replace a production 5G NR stack — it's to show end-to-end
engineering judgment: real GPU kernels, a verifiable CPU ground truth, a
standards-based link, and honest limitations.

---

## 1. GPU-accelerated MMSE channel estimation

`H_mmse = R (R + σ²I)⁻¹ H_ls`, with `R` built from a power-delay profile.

The pipeline never forms the Wiener matrix explicitly: it builds `R` with a
**custom CUDA kernel**, solves the Hermitian system with **cuSolver** (`Cpotrf`/
`Cpotrs`), and applies `R` with **cuBLAS** (`Cgemm`). Re-implementing a dense
Cholesky by hand would be slower and less numerically robust than the vendor
libraries — knowing *when not* to write a kernel is part of the point.

### Measured on RTX 4090 (sm_89, CUDA 12.8, driver 596.36)

| N (subcarriers) | CPU (NumPy) | GPU | **Speedup** | rel. error vs CPU |
|---:|---:|---:|---:|---:|
| 256  | 13.9 ms | 0.61 ms | **22.8×** | 1.6e-4 |
| 512  | 88.9 ms | 1.40 ms | **63.4×** | 1.5e-4 |
| 1024 | 313 ms  | 2.35 ms | **133.6×** | 3.4e-4 |

Standalone CUDA C++ (`csrc/mmse_est`): **2.16 ms/batch** at N=1024, B=4096
(≈1900 OFDM symbols/ms). The speedup grows with N because the O(N³) solve
dominates — exactly where the GPU pays off. Numbers are reproduced by
`benchmarks/benchmark_cpu_gpu.py`; correctness is enforced in
`tests/test_cuda_mmse.py` (GPU vs CPU and the compiled binary vs CPU).

```bash
make -C csrc ARCH=sm_89                  # sm_80 A100, sm_75 T4
csrc/mmse_est --bench --n 1024 --B 4096
python benchmarks/benchmark_cpu_gpu.py --sizes 256 512 1024
```

---

## 2. Sionna 5G BLER link

A SISO OFDM link built with **NVIDIA Sionna (PHY 1.x)**:

- 5G LDPC encoder/decoder, QAM mapping (QPSK→256QAM)
- OFDM resource grid (128 subcarriers, 14 symbols, 30 kHz SCS, Kronecker pilots)
- 3GPP TR38.901 **TDL-A** channel (100 ns delay spread, 3 m/s)
- LS channel estimation + LMMSE equalization + APP demapping

`benchmarks/run_sionna_bler.py` sweeps the **block-error rate over MCS × SNR**
with `sim_ber` and exports `results/sionna_bler_curves.json`.

```bash
pip install -r requirements-sionna.txt   # Sionna PHY 1.x + TensorFlow (GPU)
python benchmarks/run_sionna_bler.py     # writes results/sionna_bler_curves.json
```

Standards detail handled honestly: the 5G LDPC code supports coderates in
`[1/5, 8/9]` and ≤ 8448 information bits per codeword. MCS below 1/5 are clamped
to 1/5, and the highest-order MCS (256QAM) is capped at 8448 info bits
(effective rate ≈ 0.69 instead of 0.75) rather than segmented into multiple
code blocks.

---

## 3. Link adaptation driven by the Sionna curves

`airan_phy_lab/bler_table.py` loads the swept curves and exposes
`bler(snr_db, mcs)` — a drop-in for the old `synthetic_bler` — plus
`best_mcs(snr_db, target_bler)`. The PPO environment and the OLLA controller use
this table, so the agent learns against **measured 5G-link BLER**. If the curves
file is absent, the table transparently falls back to the analytic model
(`load_or_synthetic(...).source` reports `"sionna"` vs `"synthetic"`).

```bash
python -c "from airan_phy_lab.bler_table import load_or_synthetic as l; print(l().source)"
python benchmarks/train_ppo_demo.py
```

---

## Repository layout

```
airan_phy_lab/
  ofdm.py, modulation.py, channel.py, estimators.py   # classical PHY + LS/MMSE
  cuda/mmse_kernels.py                                 # CuPy RawKernels (real PHY work)
  sionna_link.py                                       # 5G TDL OFDM + LDPC BLER link
  bler_table.py                                        # Sionna curves -> OLLA/PPO bridge
  link_adaptation.py, ppo_link_adaptation.py           # OLLA + PPO
csrc/mmse_channel_est.cu                               # standalone CUDA C++ (cuSolver/cuBLAS)
benchmarks/                                            # CPU-vs-GPU, Sionna sweep, PPO demo
tests/                                                 # CPU + GPU correctness (GPU self-skips)
results/                                               # measured benchmarks + BLER curves
.github/workflows/tests.yml                            # CI: CPU tests + CUDA compile-check
```

## Setup notes

- GPU work (CUDA build, CuPy, TensorFlow/Sionna) runs on Linux or **WSL2**
  (TensorFlow has no native-Windows GPU support since 2.11). Keep Sionna in its
  own venv to avoid a numpy conflict with CuPy.
- CI runs CPU tests on every push and **compile-checks the CUDA** (no GPU needed
  on the runner); GPU tests self-skip when no GPU is present.

## Limitations

SISO link, single LDPC codeword (no code-block segmentation), TDL (not full
ray-traced) channel, and a simplified MCS table. These are the natural next
extensions, not hidden assumptions.

## License

MIT.
