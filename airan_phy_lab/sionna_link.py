"""Real 5G-style link-level BLER simulation with NVIDIA Sionna.

Builds a SISO OFDM link over a 3GPP TR38.901 TDL channel with 5G LDPC coding,
QAM modulation, LS channel estimation and LMMSE equalization, then sweeps the
block-error rate (BLER) over MCS x SNR. The resulting curves are exported and
consumed by the OLLA / PPO link-adaptation environment (see bler_table.py),
which previously used a hand-tuned logistic BLER model.

API target: **Sionna PHY 1.x** (`sionna.phy.*`). Validated against the official
Sionna 1.x component API (ResourceGrid / LDPC5G / TDL / OFDMChannel /
LSChannelEstimator / LMMSEEqualizer / sim_ber). Requires a CUDA GPU and
TensorFlow; install with `pip install -r requirements-sionna.txt`.

The MCS table is taken from airan_phy_lab.link_adaptation.MCS_TABLE so the BLER
curves are indexed by the same MCS the controller selects.

Run the sweep with benchmarks/run_sionna_bler.py (not on import).
"""
from __future__ import annotations
import json
import numpy as np

from .link_adaptation import MCS_TABLE

_MOD_TO_BPS = {"QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8}


def sionna_available() -> bool:
    try:
        import sionna.phy  # noqa: F401
        return True
    except Exception:
        return False


def _build_model(num_bits_per_symbol: int, coderate: float, perfect_csi: bool,
                 tdl_model: str, delay_spread: float, speed: float,
                 carrier_frequency: float, fft_size: int, num_ofdm_symbols: int):
    """Construct a SISO TDL link as a Sionna Block. Imported lazily."""
    import tensorflow as tf
    from sionna.phy import Block
    from sionna.phy.mimo import StreamManagement
    from sionna.phy.ofdm import (ResourceGrid, ResourceGridMapper,
                                 LSChannelEstimator, LMMSEEqualizer,
                                 RemoveNulledSubcarriers)
    from sionna.phy.channel.tr38901 import TDL
    from sionna.phy.channel import OFDMChannel
    from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
    from sionna.phy.mapping import Mapper, Demapper, BinarySource

    class Link(Block):
        def __init__(self):
            super().__init__()
            self._perfect_csi = perfect_csi
            self._rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                                    fft_size=fft_size,
                                    subcarrier_spacing=30e3,
                                    num_tx=1, num_streams_per_tx=1,
                                    cyclic_prefix_length=20,
                                    pilot_pattern="kronecker",
                                    pilot_ofdm_symbol_indices=[2, 11])
            self._sm = StreamManagement(np.array([[1]]), 1)
            self._n = int(self._rg.num_data_symbols * num_bits_per_symbol)
            self._k = int(self._n * coderate)
            self._bps = num_bits_per_symbol

            self._binary_source = BinarySource()
            self._encoder = LDPC5GEncoder(self._k, self._n)
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
            self._mapper = Mapper("qam", num_bits_per_symbol)
            self._rg_mapper = ResourceGridMapper(self._rg)
            self._demapper = Demapper("app", "qam", num_bits_per_symbol)

            tdl = TDL(model=tdl_model, delay_spread=delay_spread,
                      carrier_frequency=carrier_frequency,
                      min_speed=speed, max_speed=speed)
            self._ofdm_channel = OFDMChannel(tdl, self._rg, add_awgn=True,
                                             normalize_channel=True,
                                             return_channel=True)
            self._remove_nulled = RemoveNulledSubcarriers(self._rg)
            self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
            self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)

        @property
        def info_bits(self):
            return self._k

        @tf.function(jit_compile=False)
        def call(self, batch_size, snr_db):
            # Index BLER by channel SNR (Es/No); constellation energy is unit-normalized.
            no = tf.pow(10.0, -snr_db / 10.0)
            b = self._binary_source([batch_size, 1, 1, self._k])
            c = self._encoder(b)
            x = self._mapper(c)
            x_rg = self._rg_mapper(x)
            y, h = self._ofdm_channel(x_rg, no)
            if self._perfect_csi:
                h_hat, err_var = self._remove_nulled(h), tf.cast(0.0, y.dtype.real_dtype)
            else:
                h_hat, err_var = self._ls_est(y, no)
            x_hat, no_eff = self._lmmse_equ(y, h_hat, err_var, no)
            llr = self._demapper(x_hat, no_eff)
            b_hat = self._decoder(llr)
            return b, b_hat

    return Link()


def run_bler_sweep(snr_dbs=None, mcs_table=MCS_TABLE, perfect_csi=False,
                   tdl_model="A", delay_spread=100e-9, speed=3.0,
                   carrier_frequency=3.5e9, fft_size=128, num_ofdm_symbols=14,
                   batch_size=128, max_mc_iter=100, num_target_block_errors=200,
                   out_path="results/sionna_bler_curves.json", seed=42):
    """Sweep BLER over MCS x SNR and export curves to JSON.

    Returns the dict that is also written to `out_path`:
        {"snr_dbs": [...], "curves": {mcs_index: [bler at each snr]}, "meta": {...}}
    """
    if not sionna_available():
        raise RuntimeError("Sionna PHY not available. pip install -r requirements-sionna.txt")
    import sionna.phy
    from sionna.phy.utils import sim_ber
    sionna.phy.config.seed = seed

    if snr_dbs is None:
        snr_dbs = list(np.arange(-6.0, 30.0001, 2.0))
    snr_arr = np.asarray(snr_dbs, dtype=np.float32)

    curves = {}
    for row in mcs_table:
        bps = _MOD_TO_BPS[row["mod"]]
        rate = float(row["rate"])
        model = _build_model(bps, rate, perfect_csi, tdl_model, delay_spread,
                             speed, carrier_frequency, fft_size, num_ofdm_symbols)
        # sim_ber treats the x-axis as a scalar passed to model(batch_size, x);
        # here that scalar is the channel SNR (Es/No) in dB.
        _, bler = sim_ber(model, snr_arr, batch_size=batch_size,
                          max_mc_iter=max_mc_iter,
                          num_target_block_errors=num_target_block_errors,
                          soft_estimates=False, early_stop=True, verbose=False)
        curves[int(row["mcs"])] = [float(x) for x in np.asarray(bler)]
        print(f"MCS {row['mcs']:2d} ({row['mod']:>6} r={rate:.2f}) done")

    out = {"snr_dbs": [float(x) for x in snr_arr],
           "curves": curves,
           "meta": {"tdl_model": tdl_model, "delay_spread": delay_spread,
                    "speed": speed, "carrier_frequency": carrier_frequency,
                    "perfect_csi": perfect_csi, "fft_size": fft_size,
                    "num_ofdm_symbols": num_ofdm_symbols}}
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    return out
