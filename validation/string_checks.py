"""Immersed-boundary string in the lattice Boltzmann solver: one exact check and the streaming test.

    python validation/string_checks.py --only stokes        # about 20 s
    python validation/string_checks.py --only streaming     # about 15 min on 2 cores (3 amplitudes)

stokes     A flat string oscillates along itself, u = U cos(wt), in still fluid. Exact: a Stokes
           layer on each side, u = U exp(-s/d) cos(wt - s/d), s = distance from the string.
           The immersed boundary smears the string over a few lattice spacings, so it acts as a
           thin plate; the check fits its half-thickness h (s -> s - h) and expects h near 1.
streaming  The string carries a standing wave, w = A sin(kx) cos(wt), one wavelength in a periodic
           box with still walls far above and below. Markers move only up and down. The
           period-averaged (Eulerian mean) flow gives the steady streaming cells. It is compared
           with the small-amplitude theory in streaming_theory.py; three amplitudes test that the
           mean flow grows as A^2.

Writes validation/string_checks.json, string_stokes.png and string_streaming.png (grey scale),
and string_streaming_A*.npz with the mean fields.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from lbm import LBM  # noqa: E402
from lbm.ib import IBString  # noqa: E402


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                         "font.size": 16, "axes.titlelocation": "left", "axes.titlesize": 17,
                         "legend.frameon": False, "legend.fontsize": 13})
    return plt


# --------------------------------------------------------------------------------------------
def stokes(tau=1.0, period=1200, U=0.01, nx=8, ny=200, periods=5, samples=16):
    sim = LBM(nx, ny, tau)
    sim.solid[:, 0] = sim.solid[:, -1] = True
    y0 = ny / 2 + 0.25                                    # off the grid nodes on purpose
    X = np.arange(nx) + 0.5
    Y = np.full(nx, y0)
    ds = np.ones(nx)
    ib = IBString(nx)
    w = 2 * np.pi / period
    d = np.sqrt(2 * sim.nu / w)
    Vy = np.zeros(nx)

    def hook(t, s):
        ib.force(s, X, Y, np.full(nx, U * np.cos(w * (t + 0.5))), Vy, ds)

    sim.run(periods * period, hook)
    y = np.arange(ny)
    s = np.abs(y - y0)
    far = (~sim.solid[0]) & (s > 3) & (s < 6 * d)        # beyond the smeared string
    snaps = []
    for k in range(samples):
        sim.run(period // samples, hook)
        snaps.append((sim.t, sim.fields()[1][0].copy()))

    def exact(t, h):                                      # string of half-thickness h
        sh = np.maximum(s - h, 0.0)
        return U * np.exp(-sh / d) * np.cos(w * t - sh / d)

    def err(h):
        return max(np.abs(u - exact(t, h))[far].max() / U for t, u in snaps)

    hs = np.arange(0.0, 2.01, 0.05)
    errs = [err(h) for h in hs]
    h = float(hs[int(np.argmin(errs))])
    prof = [(u, exact(t, h)) for t, u in snaps[::4]]
    e_thin = err(0.0)
    e_fit = float(min(errs))
    return dict(err_thin=float(e_thin), err_fit=e_fit, half_thickness=h, ok=(e_fit < 0.03) and (0.5 <= h <= 1.5),
                delta=float(d), y=y - y0, prof=prof, U=U)


def stokes_figure(r, path):
    plt = style()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for j, (num, ex) in enumerate(r["prof"]):
        ax.plot(ex / r["U"], r["y"], color="0.6", lw=3.5,
                label=f"exact, half-thickness {r['half_thickness']:.2f}" if j == 0 else None)
        ax.plot(num / r["U"], r["y"], "o", color="k", ms=3, label="LBM + string" if j == 0 else None)
    ax.axhline(0, color="k", lw=0.8, ls=(0, (4, 3)))
    ax.set_ylim(-4 * r["delta"], 4 * r["delta"])
    ax.set_xlabel(r"$u_x / U$")
    ax.set_ylabel("distance from string")
    ax.set_title(f"Stokes layers on both sides, error {100 * r['err_fit']:.1f}%")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# --------------------------------------------------------------------------------------------
def streaming(A, lam=128, tau=1.0, delta=6.0, height=128, periods=None, avg_periods=4):
    """Standing wave on the string; returns the period-averaged flow."""
    nx, ny = lam, 2 * height + 2
    sim = LBM(nx, ny, tau)
    sim.solid[:, 0] = sim.solid[:, -1] = True
    w = 2 * sim.nu / delta ** 2                          # sets the Stokes layer thickness
    period = int(round(2 * np.pi / w))
    w = 2 * np.pi / period
    k = 2 * np.pi / lam
    n = lam                                               # one marker per lattice spacing
    X = np.arange(n) + 0.5
    y0 = ny / 2 - 0.5 + 0.25
    ds = np.ones(n)
    zero = np.zeros(n)
    ib = IBString(n)
    shape = np.sin(k * X)

    def hook(t, s):
        tm = t + 0.5
        Y = y0 + A * shape * np.cos(w * tm)
        ib.force(s, X, Y, zero, -A * w * shape * np.sin(w * tm), ds)

    if periods is None:                                   # long enough for the mean flow to settle
        periods = int(np.ceil(height ** 2 / sim.nu / period))
    sim.run(periods * period, hook)
    mu = np.zeros((nx, ny))
    mv = np.zeros((nx, ny))
    slip = 0.0
    for _ in range(avg_periods * period):
        sim.step(hook)
        _, ux, uy = sim.fields()
        mu += ux
        mv += uy
    mu /= avg_periods * period
    mv /= avg_periods * period
    Y = y0 + A * shape * np.cos(w * (sim.t - 0.5))
    slip = ib.slip(sim, X, Y, zero, -A * w * shape * np.sin(w * (sim.t - 0.5))) / (A * w)
    return dict(A=A, mu=mu, mv=mv, y0=y0, k=k, w=w, period=period, delta=np.sqrt(2 * sim.nu / w),
                nu=sim.nu, periods=periods, slip=slip, lam=lam)


def streaming_summary(r):
    return dict(A=r["A"], umax=float(np.abs(r["mu"]).max()), slip=float(r["slip"]), kd=float(r["k"] * r["delta"]),
                delta=float(r["delta"]), period=int(r["period"]), periods=int(r["periods"]))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="+", choices=["stokes", "streaming"])
    p.add_argument("--amps", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    a = p.parse_args()
    todo = a.only or ["stokes", "streaming"]
    out = {}
    path = os.path.join(HERE, "string_checks.json")
    if os.path.exists(path):
        out = json.load(open(path))
    if "stokes" in todo:
        t0 = time.perf_counter()
        r = stokes()
        out["stokes"] = dict(err_thin=r["err_thin"], err_fit=r["err_fit"], half_thickness=r["half_thickness"],
                             ok=bool(r["ok"]), delta=r["delta"],
                             seconds=round(time.perf_counter() - t0, 1))
        print("stokes", "PASS" if r["ok"] else "FAIL", out["stokes"], flush=True)
        stokes_figure(r, os.path.join(HERE, "string_stokes.png"))
    if "streaming" in todo:
        runs, summ = [], []
        for A in a.amps:
            t0 = time.perf_counter()
            r = streaming(A)
            s = streaming_summary(r)
            s["seconds"] = round(time.perf_counter() - t0, 1)
            print("streaming", s, flush=True)
            np.savez_compressed(os.path.join(HERE, f"string_streaming_A{A:g}.npz"),
                                **{k: v for k, v in r.items() if not isinstance(v, dict)})
            runs.append(r)
            summ.append(s)
        import streaming_theory as th
        rows, comps = th.summary()
        for s_, row in zip(summ, rows):
            s_.update(row)
        out["streaming"] = dict(runs=summ, ok=bool(all(row["err_u"] < 0.1 and row["err_v"] < 0.1 for row in rows)))
        th.figure(comps, os.path.join(HERE, "string_streaming.png"))
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
