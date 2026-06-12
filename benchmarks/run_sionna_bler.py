#!/usr/bin/env python3
"""Run the Sionna BLER sweep and export curves for link adaptation.

Requires a CUDA GPU + Sionna PHY 1.x (`pip install -r requirements-sionna.txt`).

    python benchmarks/run_sionna_bler.py                       # full sweep
    python benchmarks/run_sionna_bler.py --quick               # coarse, fast
    python benchmarks/run_sionna_bler.py --perfect-csi
"""
import argparse
import numpy as np
from airan_phy_lab.sionna_link import run_bler_sweep, sionna_available


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="coarse SNR grid + fewer iterations (smoke test)")
    ap.add_argument("--perfect-csi", action="store_true")
    ap.add_argument("--tdl-model", default="A")
    ap.add_argument("--delay-spread", type=float, default=100e-9)
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--out", default="results/sionna_bler_curves.json")
    args = ap.parse_args()

    if not sionna_available():
        raise SystemExit("Sionna PHY not available. pip install -r requirements-sionna.txt "
                         "(needs a CUDA GPU + TensorFlow).")

    if args.quick:
        snr = list(np.arange(-6.0, 30.0001, 4.0))
        kw = dict(batch_size=64, max_mc_iter=20, num_target_block_errors=50)
    else:
        snr = list(np.arange(-6.0, 30.0001, 2.0))
        kw = dict(batch_size=128, max_mc_iter=100, num_target_block_errors=200)

    run_bler_sweep(snr_dbs=snr, perfect_csi=args.perfect_csi,
                   tdl_model=args.tdl_model, delay_spread=args.delay_spread,
                   speed=args.speed, out_path=args.out, **kw)


if __name__ == "__main__":
    main()
