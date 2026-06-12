# Roadmap

Honest status of the project. The aim is *verifiable*, not *bigger*: every item
moves to "done" only when there is a command, an output, and a check for it.

## Done

- Real CUDA C++ MMSE channel estimation (custom kernel + cuSolver/cuBLAS),
  **133× vs NumPy** on an RTX 4090, validated against a CPU reference (~1e-4).
- Sionna 5G link (OFDM + 5G LDPC + TR38.901 TDL), BLER swept over MCS × SNR,
  curves committed to `results/sionna_bler_curves.json`.
- Link-adaptation **comparison** on the measured curves — random / greedy /
  OLLA / **model-free contextual bandit** under delayed + noisy CQI, with
  throughput and BLER (`experiments/la_compare.py`, figure in `results/figures/`).
  The model-free agent matches the tuned/model-based baselines from ACK/NACK
  feedback alone, at the lowest BLER.
- CI: CPU tests on every push + CUDA compile-check (no GPU on the runner).
- Figures: CPU/GPU speedup, numerical accuracy, Sionna BLER curves, LA throughput.
- Companion kernel-optimization deep-dive (naive → tiled → cuBLAS + Nsight
  methodology): [cuda-phy-channel-estimation](https://github.com/kabNath/cuda-phy-channel-estimation).

## Next

- **Stateful PPO** vs the model-free bandit / OLLA / greedy under delayed/noisy
  CQI. The committed comparison uses a *model-free contextual bandit*; a stateful
  PPO agent exploiting temporal structure (HARQ value, fading prediction) is the
  next extension — that is where a learner can move *past* matching the baselines.
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
