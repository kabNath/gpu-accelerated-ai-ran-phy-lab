# GPU-Accelerated AI-RAN PHY Lab

**Portfolio-grade AI-RAN / 6G PHY research repository for NVIDIA-style roles.**

This repository demonstrates a compact wireless PHY research stack:

- Complete OFDM transmitter / receiver
- QPSK modulation and demodulation
- LS and Linear-MMSE channel estimation
- OLLA link adaptation baseline
- PPO link-adaptation research scaffold
- Optional NVIDIA Sionna integration point
- Simple CUDA/C++ kernel through CuPy RawKernel
- CPU vs GPU benchmark harness
- Docker + GitHub Actions CI
- Reproducible example results

The goal is not to replace a production 5G NR stack. The goal is to show strong engineering judgment for AI-RAN roles: clear math, runnable code, baselines, tests, GPU awareness, and honest limitations.

---

## Quick start: CPU

```bash
git clone https://github.com/YOUR_NAME/gpu-accelerated-ai-ran-phy-lab.git
cd gpu-accelerated-ai-ran-phy-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -v
python benchmarks/run_end_to_end.py
```

## GPU quick start

```bash
pip install -r requirements-gpu.txt
pip install -e .
python benchmarks/benchmark_cpu_gpu.py
```

The GPU benchmark uses a simple CUDA C++ kernel:

```cpp
extern "C" __global__
void complex_scale_kernel(const float2* x, float2* y, float scale, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        y[i].x = scale * x[i].x;
        y[i].y = scale * x[i].y;
    }
}
```

## Docker

CPU:

```bash
docker build -t airan-phy-lab:cpu .
docker run --rm airan-phy-lab:cpu
```

GPU:

```bash
docker build -f Dockerfile.gpu -t airan-phy-lab:gpu .
docker run --rm --gpus all airan-phy-lab:gpu
```

## Optional Sionna integration

Sionna is a GPU-accelerated link-level simulation library for next-generation wireless research. This repository includes `airan_phy_lab/sionna_backend.py` as a version-guarded integration point. The default repo runs without Sionna. For a full NVIDIA-facing extension, complete:

- Resource grid
- OFDM modulation/demodulation
- 3GPP TDL channel
- LDPC encoder/decoder
- BLER sweep over MCS/SNR
- Export comparable BLER curves to the OLLA/PPO environment

Install example:

```bash
pip install sionna tensorflow
python -c "from airan_phy_lab.sionna_backend import sionna_available; print(sionna_available())"
```

## Mathematical model

Received OFDM frequency-domain samples:

```text
Y[k] = H[k] X[k] + N[k]
```

LS estimate on pilots:

```text
H_LS[k] = Y[k] / X[k]
```

Linear-MMSE estimate:

```text
H_MMSE = R_HH (R_HH + sigma^2 I)^(-1) H_LS
```

## Recruiter summary

This repo demonstrates practical OFDM PHY implementation, classical estimator baselines, GPU acceleration awareness, AI-RAN link adaptation framing, clean Python engineering, CUDA kernel literacy, and reproducible benchmarking mindset.


## Limitations

This is not a full 5G NR stack. It intentionally simplifies single-antenna channel, QPSK default modulation, simplified MCS table, simplified BLER model, optional Sionna execution, and no production O-RAN xApp integration yet.
