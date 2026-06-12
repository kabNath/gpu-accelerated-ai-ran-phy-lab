"""Bridge between Sionna BLER curves and the link-adaptation controllers.

`BLERTable` loads the curves exported by sionna_link.run_bler_sweep and exposes
`bler(snr_db, mcs)` with the SAME signature as
airan_phy_lab.link_adaptation.synthetic_bler, so the OLLA / PPO environment can
swap the hand-tuned logistic model for measured 5G-link curves by changing one
argument.

If no curves file exists yet, `load_or_synthetic` falls back to the analytic
model so the repo (and CI) keeps running — the source of the BLER is always
explicit via `.source`.
"""
from __future__ import annotations
import json
import numpy as np

from .link_adaptation import MCS_TABLE, synthetic_bler


class BLERTable:
    source = "sionna"

    def __init__(self, snr_dbs, curves):
        self._snr = np.asarray(snr_dbs, dtype=float)
        # curves: {mcs_index(int): [bler aligned to snr_dbs]}
        self._curves = {int(k): np.asarray(v, dtype=float) for k, v in curves.items()}

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["snr_dbs"], d["curves"])

    def bler(self, snr_db, mcs):
        """BLER for (snr_db, mcs). Drop-in for synthetic_bler(snr_db, mcs).

        Linear interpolation in dB, clamped to [0, 1]; outside the swept SNR
        range it saturates to the nearest endpoint (1.0 below, curve min above).
        """
        mcs = int(mcs)
        if mcs not in self._curves:
            return synthetic_bler(snr_db, mcs)
        y = self._curves[mcs]
        return float(np.clip(np.interp(snr_db, self._snr, y), 0.0, 1.0))

    def best_mcs(self, snr_db, target_bler=0.1):
        """Highest-rate MCS whose interpolated BLER is <= target at this SNR."""
        best, best_rate = 0, -1.0
        for row in MCS_TABLE:
            m = int(row["mcs"])
            if self.bler(snr_db, m) <= target_bler and row["rate"] > best_rate:
                best, best_rate = m, row["rate"]
        return best


class _SyntheticTable:
    source = "synthetic"

    def bler(self, snr_db, mcs):
        return synthetic_bler(snr_db, mcs)

    def best_mcs(self, snr_db, target_bler=0.1):
        best, best_rate = 0, -1.0
        for row in MCS_TABLE:
            m = int(row["mcs"])
            if synthetic_bler(snr_db, m) <= target_bler and row["rate"] > best_rate:
                best, best_rate = m, row["rate"]
        return best


def load_or_synthetic(path="results/sionna_bler_curves.json"):
    """Return a BLERTable from real curves if present, else the synthetic model.

    Check `.source` to know which one you got ("sionna" vs "synthetic").
    """
    try:
        return BLERTable.from_json(path)
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return _SyntheticTable()


def make_bler_fn(path="results/sionna_bler_curves.json"):
    """Return a function f(snr_db, mcs) -> BLER, drop-in for synthetic_bler."""
    table = load_or_synthetic(path)
    return table.bler
