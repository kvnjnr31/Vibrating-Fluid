"""Tracer movie for experiment 2: fluid particles carried through the permeable vibrating string.

    python experiments/02_permeable_string/tracers.py              # runs k = -w and +w, then the movie
    python experiments/02_permeable_string/tracers.py --movie      # movie only, from saved tracks

Both cases start from the same setup as run.py and spin up for the same number of periods (the
spun-up state is saved in state_k*.npz, so a rerun skips it). Then tracers are seeded below the
string on a grid filling the window (64 x 64) and carried by the instantaneous LBM velocity (bilinear interpolation, midpoint
step, one step per lattice time step). A tracer that leaves the top of the window re-enters at
the bottom at the same x. Positions are stored 12 times per cycle.

Output: tracks_k*.npz, tracers.mp4 (k = -w left, k = +w right) and tracers_drift.png.
"""
import argparse
import os
import sys
import time

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from run import build  # noqa: E402


@njit(cache=True)
def _interp(ux, uy, x, y):
    nx, ny = ux.shape
    i0 = int(np.floor(x))
    j0 = int(np.floor(y))
    fx = x - i0
    fy = y - j0
    i0 %= nx
    i1 = (i0 + 1) % nx
    j0 = min(max(j0, 0), ny - 2)
    j1 = j0 + 1
    u = ((1 - fx) * (1 - fy) * ux[i0, j0] + fx * (1 - fy) * ux[i1, j0]
         + (1 - fx) * fy * ux[i0, j1] + fx * fy * ux[i1, j1])
    v = ((1 - fx) * (1 - fy) * uy[i0, j0] + fx * (1 - fy) * uy[i1, j0]
         + (1 - fx) * fy * uy[i0, j1] + fx * fy * uy[i1, j1])
    return u, v


@njit(cache=True)
def advect(ux, uy, px, py, nx, y_lo, y_hi, wrapped):
    for n in range(px.shape[0]):
        u, v = _interp(ux, uy, px[n], py[n])
        xm = px[n] + 0.5 * u
        ym = py[n] + 0.5 * v
        u, v = _interp(ux, uy, xm, ym)
        px[n] = (px[n] + u) % nx
        py[n] += v
        if py[n] > y_hi:
            py[n] -= (y_hi - y_lo)
            wrapped[n] = True


def track(k_over_w, cycles=30, per_cycle=12, cols=64, rows=64):
    sim, hook, par = build(k_over_w)
    lam, y0, period = par["lam"], par["y0"], par["period"]
    state = os.path.join(HERE, f"state_k{k_over_w:+g}.npz")
    if os.path.exists(state):
        d = np.load(state)
        sim.f[:] = d["f"]
        sim.t = int(d["t"])
    else:
        sim.run(par["periods"] * period, hook)
        np.savez_compressed(state, f=sim.f, t=sim.t)
    gx, gy = np.meshgrid((np.arange(cols) + 0.5) * lam / cols, y0 + lam * ((np.arange(rows) + 0.5) / rows - 0.5))
    px, py = gx.ravel().copy(), gy.ravel().copy()
    x_start = px.copy()
    y_lo, y_hi = y0 - 0.5 * lam, y0 + 0.5 * lam
    every = period // per_cycle
    n_frames = cycles * per_cycle
    X = np.zeros((n_frames + 1, px.size))
    Y = np.zeros_like(X)
    W = np.zeros((n_frames + 1, px.size), dtype=bool)
    T = np.zeros(n_frames + 1, dtype=np.int64)
    X[0], Y[0], T[0] = px, py, sim.t
    wrapped = np.zeros(px.size, dtype=np.bool_)
    for fr in range(1, n_frames + 1):
        for _ in range(every):
            sim.step(hook)
            _, ux, uy = sim.fields()
            advect(ux, uy, px, py, lam, y_lo, y_hi, wrapped)
        X[fr], Y[fr], W[fr], T[fr] = px, py, wrapped, sim.t
        wrapped[:] = False
    out = os.path.join(HERE, f"tracks_k{k_over_w:+g}.npz")
    np.savez_compressed(out, X=X.astype(np.float32), Y=Y.astype(np.float32), W=W, T=T, x_start=x_start,
                        y_start=gy.ravel().copy(), **{k: v for k, v in par.items()})
    return out


def load(k):
    d = dict(np.load(os.path.join(HERE, f"tracks_k{k:+g}.npz")))
    return {key: (val.item() if val.ndim == 0 else val) for key, val in d.items()}


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                         "font.size": 15, "axes.titlelocation": "left", "axes.titlesize": 17})
    return plt


def shades(d, stripes=8):
    """Horizontal stripes at the start, alternately dark and light, like bands of dye."""
    band = np.floor(((d["y_start"] - d["y0"]) / d["lam"] + 0.5) * stripes).astype(int)
    return np.where(band % 2 == 0, "0.1", "0.72")


def movie(path, fps=24):
    plt = style()
    from matplotlib.animation import FFMpegWriter
    runs = [load(-1), load(1)]
    lam = runs[0]["lam"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.6), gridspec_kw=dict(left=0.06, right=0.98, top=0.88,
                                                                       bottom=0.1, wspace=0.12))
    xs = np.linspace(0, 1, 400)
    art = []
    for ax, d, name in zip(axes, runs, (r"(A) $k<0$: jets at nodes", r"(B) $k>0$: jets at antinodes")):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x/\lambda$")
        ax.set_ylabel(r"$y/\lambda$")
        ax.set_title(name)
        dots = ax.scatter(d["X"][0] / lam, (d["Y"][0] - d["y0"]) / lam, s=4, c=shades(d), lw=0, zorder=3)
        string, = ax.plot([], [], color="k", lw=2.6, zorder=5)
        for xn in (0, 0.5, 1):
            ax.plot(xn, 0, "o", color="k", ms=9, zorder=6, clip_on=False)
        for xa in (0.25, 0.75):
            ax.plot(xa, 0, "o", mfc="white", mec="k", ms=9, mew=1.5, zorder=6)
        art.append((string, dots))
    stamp = fig.text(0.5, 0.95, "", ha="center", fontsize=17)
    fig.text(0.5, 0.025, "Fluid tracers start in flat horizontal stripes. Fluid enters from below and leaves at the top. "
             "Black dots: nodes, open dots: antinodes.", ha="center", fontsize=12, color="0.3")
    n_frames = runs[0]["X"].shape[0]
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "24"])
    with writer.saving(fig, path, dpi=110):
        for fr in range(n_frames):
            for d, (string, dots) in zip(runs, art):
                string.set_data(xs, d["A"] * np.sin(2 * np.pi * xs) * np.cos(d["w"] * d["T"][fr]) / lam)
                dots.set_offsets(np.c_[d["X"][fr] / lam, (d["Y"][fr] - d["y0"]) / lam])
            stamp.set_text(f"Permeable vibrating string with a source below: cycle {fr / 12:4.1f}")
            writer.grab_frame()


def drift_figure(path, per_cycle=12, nbins=32):
    """Mean upward speed against x in a band above and below the string: tracers (Lagrangian mean,
    displacement over whole cycles) against the period-averaged velocity at fixed points (Eulerian)."""
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax, k in zip(axes, (-1, 1)):
        d = load(k)
        e = dict(np.load(os.path.join(HERE, f"run_k{k:+g}.npz")))
        lam, q0, y0 = d["lam"], float(d["q0"]), float(d["y0"])
        X, Y, W = d["X"].astype(float), d["Y"].astype(float), d["W"]
        n = X.shape[0] - per_cycle
        dy = Y[per_cycle:] - Y[:n]
        wrapped = np.zeros_like(dy, dtype=bool)
        for s_ in range(1, per_cycle + 1):
            wrapped |= W[s_:s_ + n]
        x0 = (X[:n] / lam) % 1
        y0r = (Y[:n] - y0) / lam
        period = float(d["period"])
        bins = np.linspace(0, 1, nbins + 1)
        xc = 0.5 * (bins[1:] + bins[:-1])
        for band, mk, lab in (((0.1, 0.3), "o", "above"), ((-0.3, -0.1), "s", "below")):
            m = (~wrapped) & (y0r > band[0]) & (y0r < band[1])
            idx = np.clip(np.digitize(x0[m], bins) - 1, 0, nbins - 1)
            vL = np.array([dy[m][idx == b].mean() / period for b in range(nbins)])
            js = np.arange(int(y0 + band[0] * lam), int(y0 + band[1] * lam))
            vE = e["mv"][:, js].mean(axis=1)
            xg = (np.arange(lam) + 0.5) / lam
            ax.plot(xg, (vE - q0) / q0, "-", color="0.55" if lab == "above" else "0.8", lw=3,
                    label=f"fixed points, {lab}")
            ax.plot(xc, (vL - q0) / q0, mk, color="k", mfc="white", ms=6, label=f"tracers, {lab}")
        for xn in (0, 0.5, 1):
            ax.axvline(xn, color="k", lw=0.8, ls=(0, (4, 3)))
        for xa in (0.25, 0.75):
            ax.axvline(xa, color="0.6", lw=0.8, ls=(0, (1, 2)))
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xlabel(r"$x/\lambda$   (dashed: nodes, dotted: antinodes)")
        ax.set_title(("(A) $k<0$" if k < 0 else "(B) $k>0$") + r": upward speed, $0.1<|y|/\lambda<0.3$")
    axes[0].set_ylabel(r"mean $(v - q_0)/q_0$")
    axes[1].legend(frameon=False, fontsize=11, loc="lower center", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--movie", action="store_true", help="only render the movie from saved tracks")
    p.add_argument("--cycles", type=int, default=30)
    a = p.parse_args()
    if not a.movie:
        for k in (-1.0, 1.0):
            t0 = time.perf_counter()
            print("tracked", track(k, cycles=a.cycles), round(time.perf_counter() - t0), "s", flush=True)
    drift_figure(os.path.join(HERE, "tracers_drift.png"))
    movie(os.path.join(HERE, "tracers.mp4"))
    print("movie written", flush=True)


if __name__ == "__main__":
    main()
