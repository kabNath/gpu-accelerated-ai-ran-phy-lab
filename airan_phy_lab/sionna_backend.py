"""Sionna environment checks + a real link smoke test.

Previously this file only checked availability and returned a TODO. It now runs
an actual SISO TDL OFDM + 5G-LDPC transmission for one batch and reports the
measured BER/BLER, so "Sionna integration" refers to code that runs rather than
a placeholder.
"""
from __future__ import annotations


def sionna_available() -> bool:
    try:
        import sionna.phy  # noqa: F401
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


def require_sionna():
    if not sionna_available():
        raise ImportError("Sionna PHY/TensorFlow not available. "
                          "Install with `pip install -r requirements-sionna.txt` "
                          "in a CUDA environment.")


def describe_sionna_environment() -> dict:
    require_sionna()
    import sionna.phy as sp
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    return {"sionna_version": getattr(sp, "__version__", "unknown"),
            "tensorflow_version": tf.__version__,
            "num_gpus": len(gpus),
            "gpus": [str(g) for g in gpus]}


def run_sionna_smoke_test(snr_db: float = 10.0, batch_size: int = 16) -> dict:
    """Run one real coded OFDM transmission over a TDL channel and report BER/BLER."""
    require_sionna()
    import numpy as np
    from .sionna_link import _build_model

    model = _build_model(num_bits_per_symbol=2, coderate=0.5, perfect_csi=False,
                         tdl_model="A", delay_spread=100e-9, speed=3.0,
                         carrier_frequency=3.5e9, fft_size=128, num_ofdm_symbols=14)
    b, b_hat = model(batch_size, float(snr_db))
    b = np.asarray(b).reshape(batch_size, -1)
    b_hat = np.asarray(b_hat).reshape(batch_size, -1)
    ber = float(np.mean(b != b_hat))
    bler = float(np.mean(np.any(b != b_hat, axis=1)))
    return {"status": "ok", "snr_db": snr_db, "ber": ber, "bler": bler,
            "environment": describe_sionna_environment()}
