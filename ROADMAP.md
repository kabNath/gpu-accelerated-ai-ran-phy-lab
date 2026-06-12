# Roadmap

Honest status of the project. The aim is *verifiable*, not *bigger*: every item
moves to "done" only when there is a command, an output, and a check for it.

## Done

- Real CUDA C++ MMSE channel estimation (custom kernel + cuSolver/cuBLAS),
  **133× vs NumPy** on an RTX 4090, validated against a CPU reference (~1e-4).
- Sionna 5G link (OFDM + 5G LDPC + TR38.901 TDL), BLER swept over MCS × SNR,
  curves committed to `results/sionna_bler_curves.json`.
- OLLA + PPO link-adaptation layer wired to the measured BLER curves.
- CI: CPU tests on every push + CUDA compile-check (no GPU on the runner).
- Figures: CPU/GPU speedup, numerical accuracy, Sionna BLER curves.
- Companion kernel-optimization deep-dive (naive → tiled → cuBLAS + Nsight
  methodology): [cuda-phy-channel-estimation](https://github.com/kabNath/cuda-phy-channel-estimation).

## Next

- **PPO vs OLLA vs greedy vs random** on the Sionna curves, under
  **delayed / noisy CQI** and non-stationary SNR — with throughput plots and a
  results JSON. This is the headline AI-RAN result; it needs careful tuning on
  the real curves rather than a quick pass, which is why it is staged here.
- More detailed GPU correctness tests (edge SNRs, larger N).
- Nsight Compute reports for the channel-estimation kernels (counters must be
  enabled; see the companion repo's `profiling/`).

## Stretch

- End-to-end GPU OFDM receiver benchmark (FFT → LS → MMSE → equalize → demap),
  CPU vs GPU.
- A100 / H100 benchmark points.
- Comb-type pilots with 2-D Wiener interpolation; MIMO; doubly-selective
  (Kalman) tracking.
- Docker GPU image.

## Non-goals

- A production 5G NR stack or an O-RAN xApp. This is a research/portfolio stack
  whose value is reproducibility and honest scope.
