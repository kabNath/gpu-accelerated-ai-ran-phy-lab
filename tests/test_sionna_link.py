"""Tests for the Sionna link and the BLER bridge.

The BLERTable tests run anywhere (no Sionna). The link/sweep tests self-skip
without Sionna+GPU, so CI stays green on CPU runners.
"""
import numpy as np
import pytest

from airan_phy_lab.bler_table import BLERTable, load_or_synthetic, make_bler_fn
from airan_phy_lab.link_adaptation import MCS_TABLE, synthetic_bler
from airan_phy_lab.sionna_link import sionna_available


def _synthetic_curves():
    snr = list(np.arange(-6, 30.001, 2.0))
    curves = {int(r["mcs"]): [synthetic_bler(s, int(r["mcs"])) for s in snr] for r in MCS_TABLE}
    return snr, curves


def test_bler_table_interpolates_and_clamps():
    snr, curves = _synthetic_curves()
    t = BLERTable(snr, curves)
    assert 0.0 <= t.bler(5.0, 2) <= 1.0
    assert t.bler(-100.0, 3) == pytest.approx(1.0, abs=1e-3)   # far below -> bad
    assert t.bler(100.0, 0) <= 0.05                            # far above -> good


def test_bler_matches_synthetic_signature():
    # drop-in: same (snr_db, mcs) argument order as synthetic_bler
    fn = make_bler_fn("does_not_exist.json")   # falls back to synthetic
    for s in (-3.0, 0.0, 10.0, 20.0):
        for m in (0, 3, 7):
            assert fn(s, m) == pytest.approx(synthetic_bler(s, m), abs=1e-9)


def test_best_mcs_monotone_in_snr():
    snr, curves = _synthetic_curves()
    t = BLERTable(snr, curves)
    seq = [t.best_mcs(s, target_bler=0.1) for s in range(-5, 26, 5)]
    assert seq == sorted(seq)            # higher SNR -> >= MCS
    assert seq[-1] >= seq[0]


def test_load_or_synthetic_reports_source():
    assert load_or_synthetic("nope.json").source == "synthetic"


@pytest.mark.skipif(not sionna_available(), reason="Sionna/GPU not available")
def test_sionna_link_runs_one_batch():
    from airan_phy_lab.sionna_backend import run_sionna_smoke_test
    out = run_sionna_smoke_test(snr_db=15.0, batch_size=8)
    assert out["status"] == "ok"
    assert 0.0 <= out["bler"] <= 1.0


@pytest.mark.skipif(not sionna_available(), reason="Sionna/GPU not available")
def test_sionna_sweep_quick(tmp_path):
    from airan_phy_lab.sionna_link import run_bler_sweep
    out = run_bler_sweep(snr_dbs=[0.0, 10.0, 20.0],
                         mcs_table=MCS_TABLE[:2], batch_size=32,
                         max_mc_iter=5, num_target_block_errors=10,
                         out_path=str(tmp_path / "curves.json"))
    assert set(out["curves"]) == {0, 1}
    assert all(0.0 <= v <= 1.0 for vals in out["curves"].values() for v in vals)
