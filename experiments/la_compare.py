#!/usr/bin/env python3
"""Link-adaptation comparison on the measured Sionna BLER curves.

Honest question: can a *model-free* learned policy (only ACK/NACK feedback)
recover the throughput of a *model-based* greedy and a hand-tuned OLLA, under
realistic delayed + noisy CQI?

Policies on a non-stationary SNR trace (CQI delayed `DELAY` slots, `NOISE` dB
estimation error):
    random  - uniform MCS (floor)
    greedy  - argmax expected throughput, USES the BLER curves at observed CQI
    olla    - outer-loop LA, USES the curves + a bounded SINR offset -> target BLER
    learned - contextual bandit over discretized observed CQI, MODEL-FREE
              (sees only its own ACK/NACK rewards, never the curves)

The TRUE SNR drives the actual block error; agents see only the delayed/noisy
estimate. Delivered throughput = mean spectral efficiency over decoded slots.
Self-contained (numpy + matplotlib).

    python experiments/la_compare.py
writes results/la_results.json and results/figures/la_throughput.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
RNG = np.random.default_rng(0)

BLER_TARGET = 0.10
DELAY, NOISE = 3, 2.0          # CQI feedback delay (slots), CQI noise (dB)
N_BINS = 24
SE = np.array([0.23, 0.38, 0.60, 0.88, 1.18, 1.48, 1.91, 2.41, 2.73, 3.32])


def load_bler():
    d = json.load(open(os.path.join(RES, "sionna_bler_curves.json")))
    snr = np.asarray(d["snr_dbs"], float)
    keys = sorted(d["curves"], key=lambda k: int(k))
    table = np.vstack([np.asarray(d["curves"][k], float) for k in keys])
    M = table.shape[0]
    se = SE[:M] if M <= len(SE) else np.linspace(0.23, 3.32, M)
    return snr, table, se


def bler_of(g, table, s, m):
    return float(np.interp(s, g, table[m], left=1.0, right=0.0))


def make_trace(T):
    a = np.empty(T); s = RNG.uniform(2, 18)
    for t in range(T):
        s += RNG.normal(0, 0.5)
        if RNG.random() < 0.01:
            s += RNG.normal(0, 6)
        a[t] = s = float(np.clip(s, -6, 30))
    return a


def obs(tr, t):
    return tr[max(0, t - DELAY)] + RNG.normal(0, NOISE)


def exp_tput(g, table, se, c):
    return np.array([se[m] * (1 - bler_of(g, table, c, m)) for m in range(len(se))])


EDGES = np.linspace(-6, 30, N_BINS + 1)
def cbin(c):
    return int(np.clip(np.digitize(c, EDGES) - 1, 0, N_BINS - 1))


def train_bandit(g, table, se, trace, iters=200000):
    M = len(se); Q = np.zeros((N_BINS, M)); eps = 0.2
    for t in range(iters):
        c = obs(trace, t % len(trace)); b = cbin(c)
        m = RNG.integers(M) if RNG.random() < eps else int(np.argmax(Q[b]))
        ack = 1 if RNG.random() > bler_of(g, table, trace[t % len(trace)], m) else 0
        Q[b, m] += 0.1 * (se[m] * ack - Q[b, m])
    return Q


def run(policy, g, table, se, trace, Q=None):
    M = len(se); deliv = nack = mcs = 0.0
    off = 0.0; up = BLER_TARGET / (1 - BLER_TARGET); dn = 1.0
    for t in range(len(trace)):
        c = obs(trace, t); b = cbin(c)
        if policy == "random":
            m = RNG.integers(M)
        elif policy == "greedy":
            m = int(np.argmax(exp_tput(g, table, se, c)))
        elif policy == "olla":
            ok = [mm for mm in range(M) if bler_of(g, table, c + off, mm) <= BLER_TARGET]
            m = max(ok) if ok else 0
        elif policy == "learned":
            m = int(np.argmax(Q[b]))
        ack = 1 if RNG.random() > bler_of(g, table, trace[t], m) else 0
        deliv += se[m] * ack; nack += 1 - ack; mcs += m
        if policy == "olla":
            off = float(np.clip(off + (up if ack else -dn), -8, 8))
    n = len(trace)
    return dict(tput=deliv / n, bler=nack / n, mean_mcs=mcs / n)


def main():
    g, table, se = load_bler()
    tr_train, tr_eval = make_trace(60000), make_trace(20000)
    Q = train_bandit(g, table, se, tr_train)
    stats = {p: run(p, g, table, se, tr_eval, Q) for p in
             ["random", "greedy", "olla", "learned"]}

    cfg = dict(cqi_delay_slots=DELAY, cqi_noise_db=NOISE, bler_target=BLER_TARGET,
               eval_slots=len(tr_eval), n_mcs=len(se))
    json.dump({"config": cfg, "policies": stats},
              open(os.path.join(RES, "la_results.json"), "w"), indent=2)

    order = ["random", "greedy", "olla", "learned"]
    lab = {"random": "Random", "greedy": "Greedy\n(model)",
           "olla": "OLLA\n(tuned)", "learned": "Learned\n(model-free)"}
    col = ["#bbb", "#4c72b0", "#dd8452", "#76b900"]
    vals = [stats[p]["tput"] for p in order]
    plt.figure(figsize=(7.2, 4.6), dpi=150)
    bars = plt.bar([lab[p] for p in order], vals, color=col)
    for bar, p in zip(bars, order):
        plt.text(bar.get_x() + bar.get_width() / 2, stats[p]["tput"] * 1.01,
                 f"{stats[p]['tput']:.3f}\n{stats[p]['bler']*100:.0f}% BLER",
                 ha="center", va="bottom", fontsize=9)
    plt.ylabel("delivered throughput (bits/symbol)")
    plt.title(f"Link adaptation, delayed (+{DELAY} slots) / noisy ({NOISE} dB) CQI")
    plt.grid(axis="y", alpha=0.3); plt.ylim(top=max(vals) * 1.25); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "la_throughput.png")); plt.close()

    print(f"{'policy':<22}{'throughput':>11}{'BLER':>9}{'mean MCS':>10}")
    for p in order:
        s = stats[p]
        print(f"{lab[p].replace(chr(10),' '):<22}{s['tput']:>11.4f}"
              f"{s['bler']*100:>8.1f}%{s['mean_mcs']:>10.2f}")
    print("\nwrote results/la_results.json + results/figures/la_throughput.png")


if __name__ == "__main__":
    main()
