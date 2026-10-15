"""Experiment 2: a permeable vibrating string with a source below it.

    python experiments/02_permeable_string/run.py               # three runs, about 15 min on 2 cores
    python experiments/02_permeable_string/run.py --k 1         # one run, k in units of omega
    python experiments/02_permeable_string/run.py --figure      # figure from saved runs

Setup (the flat version of the permeable shell):
  * Periodic box, one wavelength lambda = 128 wide. The string sits in the middle, 128 lattice
    spacings above the bottom and below the top.
  * Source: the bottom row injects fluid upward at speed q0 (moving-wall inlet). Outlet: the top
    row is held at fixed pressure. The fluid starts in uniform flow at q0.
  * The string carries a standing wave, w = A sin(kx) cos(wt); its points move only up and down.
  * Shell boundary condition (vibrating Navier-Stokes notes, eq:normal_bc and eq:flux): fluid at the
    string moves with the string plus a permeation flow q along the tilted normal,
        u = (0, dw/dt) + q n,   n = (-dw/dx, 1) / sqrt(1 + (dw/dx)^2),   q = q0 + k w,
    so u.n = dw/dt + q and the tangential velocity matches the string (no slip).
  * Runs: k = 0, +omega, -omega (k in units of omega, so k W0 matches the wall speed W0 omega).

Checks: the mean flux through any line across the box equals q0 lambda (the net-outflow result
Q_out = Q0: the vibration adds nothing to it).

Output: run_k*.npz (mean fields and two snapshots), results.json, permeable_string.png.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from lbm import LBM  # noqa: E402
from lbm.ib import IBString  # noqa: E402
from lbm.boundaries import outlet_top  # noqa: E402


def build(k_over_w, A=2.0, q0=0.005, lam=128, half=128, tau=1.0, delta=6.0):
    """Set up the box, source, outlet and string. Returns the solver, the string hook and the parameters."""
    nx, ny = lam, 2 * half + 2
    sim = LBM(nx, ny, tau)
    sim.solid[:, 0] = True                                # source row
    sim.uwy[:, 0] = q0
    sim.set_velocity(0.0, q0)
    sim.post = lambda s: outlet_top(s.f, 1.0)
    w = 2 * sim.nu / delta ** 2
    period = int(round(2 * np.pi / w))
    w = 2 * np.pi / period
    kw = 2 * np.pi / lam                                  # wavenumber
    kp = k_over_w * w                                     # permeation coefficient, q = q0 + kp * w
    X = np.arange(lam) + 0.5
    y0 = half + 0.75
    S, C = np.sin(kw * X), np.cos(kw * X)
    ib = IBString(lam)

    def state(tm):
        wd = A * S * np.cos(w * tm)                       # displacement
        wt = -A * w * S * np.sin(w * tm)                  # velocity
        wx = A * kw * C * np.cos(w * tm)                  # slope
        norm = np.sqrt(1 + wx * wx)
        nxv, nyv = -wx / norm, 1.0 / norm
        q = q0 + kp * wd
        return y0 + wd, q * nxv, wt + q * nyv, norm

    def hook(t, s):
        Y, Ux, Uy, ds = state(t + 0.5)
        ib.force(s, X, Y, Ux, Uy, ds)

    periods = int(np.ceil(half ** 2 / sim.nu / period))
    par = dict(k_over_w=k_over_w, A=A, q0=q0, lam=lam, half=half, y0=y0, w=w, kw=kw, kp=kp, nu=sim.nu,
               delta=np.sqrt(2 * sim.nu / w), period=period, periods=periods)
    return sim, hook, par


def run(k_over_w, avg_periods=4, **kw_args):
    sim, hook, par = build(k_over_w, **kw_args)
    nx, ny = sim.nx, sim.ny
    period, periods, y0, q0, lam = par["period"], par["periods"], par["y0"], par["q0"], par["lam"]
    sim.run(periods * period, hook)
    mu, mv = np.zeros((nx, ny)), np.zeros((nx, ny))
    snaps = {}
    n_avg = avg_periods * period
    for n in range(n_avg):
        sim.step(hook)
        _, ux, uy = sim.fields()
        mu += ux
        mv += uy
        ph = (sim.t % period)                             # field at time sim.t; w t = 0 or pi
        if n >= n_avg - period and ph in (0, period // 2):
            snaps["peak" if ph == 0 else "trough"] = (ux.copy(), uy.copy())
    mu /= n_avg
    mv /= n_avg
    rows = {d: int(round(y0 + d)) for d in (-60, -30, 30, 60)}
    flux = {d: float(mv[:, j].sum() / (q0 * lam)) for d, j in rows.items()}
    par = {key: val for key, val in par.items() if key != "half"}
    return dict(**par, mu=mu, mv=mv,
                peak_u=snaps["peak"][0], peak_v=snaps["peak"][1],
                trough_u=snaps["trough"][0], trough_v=snaps["trough"][1], flux=flux)


def jets(r):
    """Amplitudes of the mean flow pattern above the string, in units of A^2 w kw."""
    x = np.arange(r["lam"]) + 0.5
    kw, sc = r["kw"], r["A"] ** 2 * r["w"] * r["kw"]
    js = np.arange(int(np.floor(r["y0"])) + 1, int(r["y0"] + 8 * r["delta"]))
    y = js - r["y0"]
    u = np.array([2 * np.mean(r["mu"][:, j] * np.sin(2 * kw * x)) for j in js]) / sc
    v = np.array([2 * np.mean((r["mv"][:, j] - r["q0"]) * np.cos(2 * kw * x)) for j in js]) / sc
    return y, u, v


def figure(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                         "font.size": 15, "axes.titlelocation": "left", "axes.titlesize": 16,
                         "legend.frameon": False, "legend.fontsize": 12})
    runs = sorted(runs, key=lambda r: r["k_over_w"])
    pos = [r for r in runs if r["k_over_w"] > 0][0]
    lam = pos["lam"]
    x = (np.arange(lam) + 0.5) / lam
    xs = np.linspace(0, 1, 400)
    fig = plt.figure(figsize=(18, 11.5))

    def field(ax, r, U, V, title, sgn, mean=False):
        y = (np.arange(U.shape[1]) - r["y0"]) / lam
        win = np.abs(y) < 0.5
        Uw, Vw = U[:, win], V[:, win]
        if mean:
            Vw = Vw - r["q0"]                             # remove the uniform throughflow
        sp = np.hypot(Uw, Vw)
        ax.contourf(*np.meshgrid(x, y[win], indexing="ij"), sp / sp.max(), levels=np.linspace(0, 1, 11),
                    cmap="Greys", alpha=0.45)
        ax.streamplot(x, y[win], Uw.T, Vw.T, color="k", linewidth=0.8, density=1.3, arrowsize=0.9)
        if sgn is None:
            ax.fill_between(xs, -r["A"] / lam * np.abs(np.sin(2 * np.pi * xs)),
                            r["A"] / lam * np.abs(np.sin(2 * np.pi * xs)), color="0.35", lw=0)
        else:
            ax.plot(xs, sgn * r["A"] / lam * np.sin(2 * np.pi * xs), color="0.2", lw=2.4)
        for xn in (0, 0.5, 1):
            ax.plot(xn, 0, "o", color="k", ms=9, zorder=5, clip_on=False)
        for xa in (0.25, 0.75):
            ax.plot(xa, 0, "o", mfc="white", mec="k", ms=9, mew=1.5, zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x/\lambda$")
        ax.set_ylabel(r"$y/\lambda$")
        ax.set_title(title)

    w0 = 0.04
    field(fig.add_axes([w0, 0.54, 0.27, 0.40]), pos, pos["peak_u"], pos["peak_v"],
          r"(A) $k>0$, $\omega t = 0$", +1)
    field(fig.add_axes([w0 + 0.31, 0.54, 0.27, 0.40]), pos, pos["trough_u"], pos["trough_v"],
          r"(B) $k>0$, $\omega t = \pi$", -1)
    labels = {-1: "(C) Mean, $k<0$", 0: "(D) Mean, $k=0$", 1: "(E) Mean, $k>0$"}
    for n, r in enumerate(runs):
        field(fig.add_axes([w0 + 0.31 * n, 0.05, 0.27, 0.40]), r, r["mu"], r["mv"],
              labels[int(np.sign(r["k_over_w"]))] + " (throughflow removed)", None, mean=True)
    bx = fig.add_axes([0.72, 0.56, 0.25, 0.37])
    marks = {-1: "v", 0: "o", 1: "^"}
    for r in runs:
        y, u, v = jets(r)
        sgn = int(np.sign(r["k_over_w"]))
        bx.plot(v, y / r["delta"], marks[sgn] + "-", color="k", mfc="white", ms=5, lw=0.8,
                label=f"$k = {r['k_over_w']:+g}\\,\\omega$" if sgn else "$k = 0$")
    bx.axvline(0, color="k", lw=0.8)
    bx.set_ylim(0, 8)
    bx.set_xlabel(r"mean $\bar v - q_0$: amplitude of $\cos 2kx$ / $A^2\omega k$")
    bx.set_ylabel(r"$y/\delta$")
    bx.set_title("(F) Jets: + at Nodes, − at Antinodes")
    bx.legend(loc="upper right")
    fig.savefig(path, dpi=140)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--k", type=float, nargs="+", default=[0.0, 1.0, -1.0], help="k in units of omega")
    p.add_argument("--figure", action="store_true", help="only redraw the figure from saved runs")
    a = p.parse_args()
    res_path = os.path.join(HERE, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    if not a.figure:
        for k in a.k:
            t0 = time.perf_counter()
            r = run(k)
            np.savez_compressed(os.path.join(HERE, f"run_k{k:+g}.npz"),
                                **{key: val for key, val in r.items() if key != "flux"})
            y, u, v = jets(r)
            i = int(np.argmax(np.abs(v)))
            results[f"{k:+g}"] = dict(k_over_w=k, flux_over_q0_lambda=r["flux"], jet_v_peak=float(v[i]),
                                      jet_y_over_delta=float(y[i] / r["delta"]), period=r["period"],
                                      periods=r["periods"], delta=float(r["delta"]),
                                      seconds=round(time.perf_counter() - t0, 1))
            print(f"{k:+g}", results[f"{k:+g}"], flush=True)
            json.dump(results, open(res_path, "w"), indent=2)
    runs = []
    for f in glob.glob(os.path.join(HERE, "run_k*.npz")):
        d = dict(np.load(f))
        d = {key: (val.item() if val.ndim == 0 else val) for key, val in d.items()}
        runs.append(d)
    figure(runs, os.path.join(HERE, "permeable_string.png"))


if __name__ == "__main__":
    main()
