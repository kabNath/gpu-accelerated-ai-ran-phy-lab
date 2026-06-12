# Design notes & review guide

A short, honest map of *what* this repo claims, *why* each design choice was
made, and *how* a reviewer can verify each claim in a few minutes.

## What this repo is

A compact, reproducible OFDM PHY research stack: a real CUDA C++ MMSE channel
estimator, a Sionna-based 5G BLER link, and an OLLA/PPO link-adaptation layer
that consumes the measured BLER curves. It is **not** a production 5G NR stack;
it is a demonstration of end-to-end engineering judgment.

## Design choices (and the trade-offs)

- **CuPy *and* CUDA C++.** CuPy gives a fast, readable reference and easy
  batched experimentation; the hand-written `.cu` (cuSolver Cholesky + cuBLAS
  GEMM) is where the real GPU work and the 133× live. Keeping both makes the
  speedup attributable rather than magic.
- **cuSolver/cuBLAS instead of a hand-rolled Cholesky.** A dense Hermitian solve
  by hand would be slower and less numerically robust than the vendor library.
  Knowing *when not* to write a kernel is part of the point — the deeper kernel
  study (naive → tiled → cuBLAS, with Nsight) lives in the companion repo
  [cuda-phy-channel-estimation](https://github.com/kabNath/cuda-phy-channel-estimation).
- **Sionna for the link.** Rather than a hand-tuned analytic BLER model, the
  curves come from a real 5G LDPC + OFDM + TR38.901 TDL simulation, so the
  link-adaptation layer is driven by standards-based block-error behaviour.
- **OLLA as the industrial baseline; PPO as a research scaffold.** OLLA is the
  deployed outer-loop method and the honest yardstick. The PPO path is wired to
  the same BLER environment; a *reproducible comparison result* (delayed/noisy
  CQI, throughput plots) is the next roadmap item, not a finished claim.

## How each claim is verified

| Claim | How to check it |
|---|---|
| 133× CUDA speedup | `benchmarks/benchmark_cpu_gpu.py` + table in README; figure in `results/figures/` |
| GPU == CPU math | `tests/test_cuda_mmse.py` (rel. err ~1e-4 vs CPU reference) |
| MMSE ≈ 9 dB gain | `benchmarks/mse_vs_snr.py` matches `10·log₁₀(N/L)` |
| Sionna link is real | `benchmarks/run_sionna_bler.py` regenerates `results/sionna_bler_curves.json` |
| CI builds the CUDA | `.github/workflows/tests.yml` compile-checks the `.cu` (no GPU) |

## Limitations (explicit, not hidden)

- SISO link, single LDPC codeword (no code-block segmentation), TDL (not
  ray-traced) channel, simplified MCS table.
- 256QAM is capped at 8448 info bits (5G LDPC max) rather than segmented.
- Benchmarks are on an RTX 4090; A100/H100 numbers are *expected* to scale, not
  measured here.
- The committed link-adaptation result uses a model-free contextual bandit; the stateful PPO agent
  (`ppo_link_adaptation.py`) is the next extension on the roadmap.
- Nsight Compute live counters require enabling GPU performance counters; the
  kernel-level profiling methodology is documented in the companion repo.

## Five-minute reproduce

```bash
pip install -r requirements.txt && pip install -e .
pytest -v                                  # CPU tests + GPU tests self-skip
python benchmarks/run_end_to_end.py        # OFDM RX + LS/MMSE + OLLA
python benchmarks/make_figures.py          # regenerate the figures
```
