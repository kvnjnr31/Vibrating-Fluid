"""Experiment 5: a square array of vibrating circular cells locked together.

    python experiments/05_locked_cells/cells.py              # both runs, about 5 min on 2 cores
    python experiments/05_locked_cells/cells.py --mode 4     # 8 antinodes instead of 4
    python experiments/05_locked_cells/cells.py --plot       # figure from saved runs

Each cell is a closed string, r = R + A cos(m (theta - theta_c)) cos(wt + phase_c), with fluid inside
and outside, no slip, no source, no permeation. Default m = 2: four antinodes, two bulging out while
the other two pull in, and four nodes between them.

The cells sit on a 4 x 4 square lattice in a periodic box, spacing D = 48, radius R = 20 (gap 8),
so neighbors nearly touch. Two lockings:
  aligned   every cell has the same orientation and phase: neighbors bulge toward each other
            at the same moment.
  geared    checkerboard: neighboring cells are half a cycle apart, so where one bulges the
            neighbor pulls in, like meshing gears.
Output: run_<locking>_m<m>.npz, locked_cells_m<m>.png.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from lbm import LBM  # noqa: E402
from lbm.ib import IBString  # noqa: E402


def build(locking, m=2, n=4, D=48, R=20.0, A=2.0, tau=1.0, delta=6.0):
    L = n * D
    sim = LBM(L, L, tau)
    w = 2 * sim.nu / delta ** 2
    period = int(round(2 * np.pi / w))
    w = 2 * np.pi / period
    nm = int(np.ceil(2 * np.pi * R))                      # about one marker per lattice spacing
    th = 2 * np.pi * np.arange(nm) / nm
    centers, phases = [], []
    for i in range(n):
        for j in range(n):
            centers.append(((i + 0.5) * D + 0.25, (j + 0.5) * D + 0.25))
            phases.append(0.0 if (locking == "aligned" or (i + j) % 2 == 0) else np.pi)
    centers = np.array(centers)
    phases = np.repeat(np.array(phases), nm)
    cx = np.repeat(centers[:, 0], nm)
    cy = np.repeat(centers[:, 1], nm)
    TH = np.tile(th, len(centers))
    ib = IBString(TH.size)
    shape = np.cos(m * TH)
    dshape = -m * np.sin(m * TH)
    dth = 2 * np.pi / nm

    def hook(t, s):
        tm = t + 0.5
        c, sn = np.cos(w * tm + phases), np.sin(w * tm + phases)
        r = R + A * shape * c
        rt = -A * w * shape * sn                          # radial speed of the wall
        X = (cx + r * np.cos(TH)) % L
        Y = (cy + r * np.sin(TH)) % L
        drdth = A * dshape * c
        ds = np.sqrt(r * r + drdth * drdth) * dth
        ib.force(s, X, Y, rt * np.cos(TH), rt * np.sin(TH), ds)

    par = dict(locking=locking, m=m, n=n, D=D, R=R, A=A, L=L, w=w, period=period, nu=sim.nu,
               delta=np.sqrt(2 * sim.nu / w), centers=centers, cell_phase=phases[::nm])
    return sim, hook, par


def run(locking, m, periods=40, avg_periods=4):
    sim, hook, par = build(locking, m)
    period = par["period"]
    sim.run((periods - avg_periods) * period, hook)
    mu = np.zeros((sim.nx, sim.ny))
    mv = np.zeros_like(mu)
    snaps = {}
    for _ in range(avg_periods * period):
        sim.step(hook)
        _, ux, uy = sim.fields()
        mu += ux
        mv += uy
        ph = sim.t % period
        if ph in (0, period // 4, period // 2):
            snaps[{0: "a", period // 4: "q", period // 2: "b"}[ph]] = (ux.copy(), uy.copy())
    n = avg_periods * period
    return dict(**par, mu=mu / n, mv=mv / n, a_u=snaps["a"][0], a_v=snaps["a"][1], q_u=snaps["q"][0],
                q_v=snaps["q"][1], b_u=snaps["b"][0], b_v=snaps["b"][1])


def load(locking, m):
    d = dict(np.load(os.path.join(HERE, f"run_{locking}_m{m}.npz")))
    return {k: (v.item() if v.ndim == 0 else v) for k, v in d.items()}


def figure(m, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                         "font.size": 15, "axes.titlelocation": "left", "axes.titlesize": 16})
    runs = [load("aligned", m), load("geared", m)]
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 12), gridspec_kw=dict(left=0.04, right=0.99, top=0.95,
                                                                          bottom=0.075, wspace=0.12, hspace=0.16))
    th = np.linspace(0, 2 * np.pi, 400)

    def draw(ax, d, U, V, title, wt=None, mean=False):
        L, D = d["L"], d["D"]
        g = np.arange(L) + 0.5
        sp = np.hypot(U, V)
        ax.contourf(g / D, g / D, (sp / sp.max()).T, levels=np.linspace(0, 1, 11), cmap="Greys", alpha=0.5)
        ax.streamplot(g / D, g / D, U.T, V.T, color="k", linewidth=0.7, density=2.2, arrowsize=0.8)
        for (x0, y0), ph in zip(d["centers"], d["cell_phase"]):
            if mean:
                ax.fill(np.r_[(x0 + (d["R"] + d["A"] * np.abs(np.cos(m * th))) * np.cos(th)) / D,
                              (x0 + (d["R"] - d["A"] * np.abs(np.cos(m * th))) * np.cos(th[::-1])) / D],
                        np.r_[(y0 + (d["R"] + d["A"] * np.abs(np.cos(m * th))) * np.sin(th)) / D,
                              (y0 + (d["R"] - d["A"] * np.abs(np.cos(m * th))) * np.sin(th[::-1])) / D],
                        color="0.35", lw=0)
            else:
                r = d["R"] + d["A"] * np.cos(m * th) * np.cos(wt + ph)
                ax.plot((x0 + r * np.cos(th)) / D, (y0 + r * np.sin(th)) / D, color="k", lw=2.0)
        ax.set_xlim(0, d["n"])
        ax.set_ylim(0, d["n"])
        ax.set_aspect("equal")
        ax.set_xticks(range(d["n"] + 1))
        ax.set_yticks(range(d["n"] + 1))
        ax.set_xlabel("$x / D$")
        ax.set_ylabel("$y / D$")
        ax.set_title(title)

    names = ("Aligned (all in phase)", "Geared (neighbors opposite)")
    L = iter("ABCDEF")
    def shapes(ax, d, title):
        for (x0, y0), ph in zip(d["centers"], d["cell_phase"]):
            for wt, ls, col in ((0.0, "-", "k"), (np.pi, (0, (4, 3)), "0.45")):
                r = d["R"] + d["A"] * 2.5 * np.cos(m * th) * np.cos(wt + ph)     # amplitude drawn 2.5x
                ax.plot((x0 + r * np.cos(th)) / d["D"], (y0 + r * np.sin(th)) / d["D"], color=col, lw=2.0, ls=ls)
            for k in range(2 * m):
                a = np.pi * k / m
                ax.plot((x0 + d["R"] * np.cos(a)) / d["D"], (y0 + d["R"] * np.sin(a)) / d["D"], "o", mfc="white",
                        mec="k", ms=5, zorder=5)
                b = a + np.pi / (2 * m)
                ax.plot((x0 + d["R"] * np.cos(b)) / d["D"], (y0 + d["R"] * np.sin(b)) / d["D"], "o", color="k",
                        ms=5, zorder=5)
        ax.set_xlim(0, d["n"])
        ax.set_ylim(0, d["n"])
        ax.set_aspect("equal")
        ax.set_xticks(range(d["n"] + 1))
        ax.set_yticks(range(d["n"] + 1))
        ax.set_xlabel("$x / D$")
        ax.set_ylabel("$y / D$")
        ax.set_title(title)

    for row, (d, name) in enumerate(zip(runs, names)):
        shapes(axes[row, 0], d, rf"({next(L)}) {name}: shapes, $\omega t = 0$ and $\pi$")
        draw(axes[row, 1], d, d["q_u"], d["q_v"], rf"({next(L)}) Flow at $\omega t = \pi/2$ (fastest wall)", wt=np.pi / 2)
        draw(axes[row, 2], d, d["mu"], d["mv"], rf"({next(L)}) Mean flow over a cycle", mean=True)
    fig.text(0.5, 0.008, "Column 1: solid and dashed outlines are the two extreme shapes (amplitude drawn 2.5 times); "
             "open dots antinodes, black dots nodes. Shading: speed, scaled to each panel.", ha="center", fontsize=12,
             color="0.3")
    fig.savefig(path, dpi=130)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, default=2)
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    if not a.plot:
        for locking in ("aligned", "geared"):
            t0 = time.perf_counter()
            np.savez_compressed(os.path.join(HERE, f"run_{locking}_m{a.mode}.npz"), **run(locking, a.mode))
            print(locking, round(time.perf_counter() - t0), "s", flush=True)
    figure(a.mode, os.path.join(HERE, f"locked_cells_m{a.mode}.png"))


if __name__ == "__main__":
    main()
