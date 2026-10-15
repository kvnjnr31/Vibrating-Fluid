"""Experiment 7: turning cells, from moving nodes (toward Maxwell's molecular vortices).

    python experiments/07_turning_cells/turning.py           # two runs and the movie, about 12 min on 2 cores
    python experiments/07_turning_cells/turning.py --plot    # figure and movie from saved runs

Each cell is the oval-mode cell of experiment 5, but its nodes move around it, as variable tension
moves the nodes of the string in the membrane paper:

    r = R + A cos(m (theta - theta_n(t))) cos(wt),    theta_n(t) = s * Omega_n * t,

with s = +1 (nodes move counterclockwise) or -1 (clockwise). The wall points still move only in and
out; only the pattern turns. Node speed Omega_n = 0.25 w / m: the nodes advance a quarter of the
node spacing each cycle. No slip, no permeation, no source, periodic box (as experiment 5).

Two runs:
  uniform   every cell's nodes turn counterclockwise.
  sheared   the left two columns turn counterclockwise, the right two clockwise.

Maxwell (1861, Part II, eq. 27): the idle particles between two vortices move at the mean of the two
vortex surface velocities, and a net current of particles appears where the vortex rotation varies
in space. Here the fluid in the gaps plays the idle particles. The script measures each cell's mean
swirl just outside its wall, and the mean flow along each vertical gap between columns.

Output: run_<case>.npz, results.json, turning_cells.png, turning_cells.mp4.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from lbm import LBM  # noqa: E402
from lbm.ib import IBString  # noqa: E402


@njit(cache=True)
def advect_periodic(ux, uy, px, py):
    nx, ny = ux.shape
    for n in range(px.shape[0]):
        x, y = px[n], py[n]
        u = 0.0
        v = 0.0
        for stage in range(2):
            i0 = int(np.floor(x))
            j0 = int(np.floor(y))
            fx, fy = x - i0, y - j0
            i0 %= nx
            j0 %= ny
            i1 = (i0 + 1) % nx
            j1 = (j0 + 1) % ny
            u = ((1 - fx) * (1 - fy) * ux[i0, j0] + fx * (1 - fy) * ux[i1, j0]
                 + (1 - fx) * fy * ux[i0, j1] + fx * fy * ux[i1, j1])
            v = ((1 - fx) * (1 - fy) * uy[i0, j0] + fx * (1 - fy) * uy[i1, j0]
                 + (1 - fx) * fy * uy[i0, j1] + fx * fy * uy[i1, j1])
            if stage == 0:
                x = px[n] + 0.5 * u
                y = py[n] + 0.5 * v
        px[n] = (px[n] + u) % nx
        py[n] = (py[n] + v) % ny


def build(case, m=2, n=4, D=48, R=20.0, A=2.0, node_speed=0.25, tau=1.0, delta=6.0):
    L = n * D
    sim = LBM(L, L, tau)
    w = 2 * sim.nu / delta ** 2
    period = int(round(2 * np.pi / w))
    w = 2 * np.pi / period
    Om = node_speed * w / m
    nm = int(np.ceil(2 * np.pi * R))
    th = 2 * np.pi * np.arange(nm) / nm
    centers, sense = [], []
    for i in range(n):
        for j in range(n):
            centers.append(((i + 0.5) * D + 0.25, (j + 0.5) * D + 0.25))
            sense.append(1.0 if (case == "uniform" or i < n // 2) else -1.0)
    centers, sense = np.array(centers), np.array(sense)
    s = np.repeat(sense, nm)
    cx, cy = np.repeat(centers[:, 0], nm), np.repeat(centers[:, 1], nm)
    TH = np.tile(th, len(centers))
    dth = 2 * np.pi / nm
    ib = IBString(TH.size)

    def wall(tm):
        ph = m * (TH - s * Om * tm)
        c, sn = np.cos(w * tm), np.sin(w * tm)
        r = R + A * np.cos(ph) * c
        rt = A * (m * s * Om * np.sin(ph) * c - w * np.cos(ph) * sn)
        drdth = -A * m * np.sin(ph) * c
        return r, rt, drdth

    def hook(t, sm):
        r, rt, drdth = wall(t + 0.5)
        X = (cx + r * np.cos(TH)) % L
        Y = (cy + r * np.sin(TH)) % L
        ib.force(sm, X, Y, rt * np.cos(TH), rt * np.sin(TH), np.sqrt(r * r + drdth * drdth) * dth)

    par = dict(case=case, m=m, n=n, D=D, R=R, A=A, L=L, w=w, Om=Om, node_speed=node_speed, period=period,
               nu=sim.nu, delta=np.sqrt(2 * sim.nu / w), centers=centers, sense=sense)
    return sim, hook, par


def run(case, spin=40, cycles=30, per_cycle=12, side=64):
    sim, hook, par = build(case)
    L, period = par["L"], par["period"]
    sim.run(spin * period, hook)
    g = (np.arange(side) + 0.5) * L / side
    gx, gy = np.meshgrid(g, g, indexing="ij")
    px, py = gx.ravel().copy(), gy.ravel().copy()
    every = period // per_cycle
    nf = cycles * per_cycle
    Xs = np.zeros((nf + 1, px.size), np.float32)
    Ys = np.zeros_like(Xs)
    T = np.zeros(nf + 1, np.int64)
    Xs[0], Ys[0], T[0] = px, py, sim.t
    mu, mv = np.zeros((L, L)), np.zeros((L, L))
    for fr in range(1, nf + 1):
        for _ in range(every):
            sim.step(hook)
            _, ux, uy = sim.fields()
            advect_periodic(ux, uy, px, py)
            mu += ux
            mv += uy
        Xs[fr], Ys[fr], T[fr] = px, py, sim.t
    k = nf * every
    out = dict(**par, mu=mu / k, mv=mv / k, X=Xs, Y=Ys, T=T, y_start=gy.ravel().copy(), x_start=gx.ravel().copy())
    return out


def measures(d):
    """Mean swirl just outside each cell, and mean flow along each vertical gap, in units of A w."""
    L, D, R = d["L"], d["D"], d["R"]
    Aw = d["A"] * d["w"]
    g = np.arange(L) + 0.0
    X, Y = np.meshgrid(g, g, indexing="ij")
    swirl = []
    for (x0, y0) in d["centers"]:
        dx = (X - x0 + L / 2) % L - L / 2
        dy = (Y - y0 + L / 2) % L - L / 2
        r = np.hypot(dx, dy)
        ring = np.abs(r - (R + 3.5)) < 1.0
        ut = (-d["mu"] * dy + d["mv"] * dx) / np.maximum(r, 1e-9)
        swirl.append(float(ut[ring].mean() / Aw))
    gaps = {}
    for i in range(d["n"]):
        xg = int(round(i * D + 0.25)) % L                 # gap line between column i-1 and column i
        gaps[f"x={xg}"] = float(d["mv"][xg, :].mean() / Aw)
    return dict(swirl=swirl, gap_flow=gaps)


def load(case):
    d = dict(np.load(os.path.join(HERE, f"run_{case}.npz")))
    return {k: (v.item() if v.ndim == 0 else v) for k, v in d.items()}


def outlines(d, t, scale=1.0):
    th = np.linspace(0, 2 * np.pi, 300)
    out = []
    for (x0, y0), s in zip(d["centers"], d["sense"]):
        r = d["R"] + scale * d["A"] * np.cos(d["m"] * (th - s * d["Om"] * t)) * np.cos(d["w"] * t)
        out.append(((x0 + r * np.cos(th)) / d["D"], (y0 + r * np.sin(th)) / d["D"]))
    return out


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                         "font.size": 15, "axes.titlelocation": "left", "axes.titlesize": 16})
    return plt


def figure(path):
    plt = style()
    runs = [load("uniform"), load("sheared")]
    fig = plt.figure(figsize=(18, 12.5))
    for row, (d, name) in enumerate(zip(runs, ("Uniform: all nodes turn counterclockwise",
                                               "Sheared: left columns counterclockwise, right clockwise"))):
        ax = fig.add_axes([0.04, 0.54 - 0.48 * row, 0.4, 0.4])
        L, D = d["L"], d["D"]
        g = (np.arange(L) + 0.5) / D
        Aw = d["A"] * d["w"]
        sp = np.hypot(d["mu"], d["mv"]) / Aw
        cf = ax.contourf(g, g, sp.T, levels=np.linspace(0, 0.05, 11), cmap="Greys", extend="max")
        ax.streamplot(g, g, d["mu"].T, d["mv"].T, color="k", linewidth=0.7, density=2.2, arrowsize=0.8)
        for (x0, y0), s in zip(d["centers"], d["sense"]):
            ax.add_patch(plt.Circle((x0 / D, y0 / D), d["R"] / D, fill=False, color="0.25", lw=1.4, ls=(0, (3, 2))))
            ax.annotate("", xy=((x0 + 0.7 * d["R"] * np.cos(s * 0.6)) / D, (y0 + 0.7 * d["R"] * np.sin(s * 0.6)) / D),
                        xytext=((x0 + 0.7 * d["R"]) / D, y0 / D),
                        arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.6, connectionstyle=f"arc3,rad={0.35 * s}"))
        ax.set_xlim(0, d["n"])
        ax.set_ylim(0, d["n"])
        ax.set_aspect("equal")
        ax.set_xlabel("$x / D$")
        ax.set_ylabel("$y / D$")
        ax.set_title(f"({'AB'[row]}) {name}")
        cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(r"mean speed / $A\omega$")
        mz = measures(d)
        bx = fig.add_axes([0.56, 0.54 - 0.48 * row, 0.18, 0.4])
        sw = np.array(mz["swirl"]).reshape(d["n"], d["n"])         # [column, row]
        lim = 45.0
        bx.imshow(sw.T * 1e3, origin="lower", cmap="Greys", vmin=-lim, vmax=lim, extent=(0, d["n"], 0, d["n"]))
        for i in range(d["n"]):
            for j in range(d["n"]):
                bx.text(i + 0.5, j + 0.5, f"{1e3 * sw[i, j]:+.1f}", ha="center", va="center", fontsize=11,
                        color="w" if sw[i, j] > 0 else "k")
        bx.set_xticks(range(d["n"] + 1))
        bx.set_yticks(range(d["n"] + 1))
        bx.set_xlabel("$x / D$")
        bx.set_title(f"({'CD'[row]}) Swirl per cell" + "\n" + r"$\bar u_\theta / A\omega \times 10^3$ (+ ccw)",
                     fontsize=14)
        cx = fig.add_axes([0.8, 0.54 - 0.48 * row, 0.17, 0.4])
        xs = [int(k.split("=")[1]) / D for k in mz["gap_flow"]]
        cx.bar(xs, 1e3 * np.array(list(mz["gap_flow"].values())), width=0.25, color="0.35")
        cx.axhline(0, color="k", lw=0.8)
        cx.set_xlim(-0.4, d["n"] - 0.6)
        cx.set_xticks(range(d["n"]))
        cx.set_xlabel("gap position, $x / D$")
        cx.set_ylabel(r"mean $\bar v$ along the gap / $A\omega \times 10^3$")
        cx.set_title(f"({'EF'[row]}) Net flow in each gap", fontsize=14)
    fig.text(0.5, 0.012, "Dashed circles: cells (mean shape); curved arrows: direction the nodes turn. "
             "Gaps at x/D = 0 and 4 are the same line (periodic box).", ha="center", fontsize=12, color="0.3")
    fig.savefig(path, dpi=130)


def movie(path, fps=24):
    plt = style()
    from matplotlib.animation import FFMpegWriter
    runs = [load("uniform"), load("sheared")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.2), gridspec_kw=dict(left=0.05, right=0.98, top=0.88,
                                                                       bottom=0.1, wspace=0.12))
    art = []
    for ax, d, name in zip(axes, runs, ("(A) Uniform: all cells turn counterclockwise",
                                        "(B) Sheared: left ccw, right cw")):
        D = d["D"]
        band = np.floor(d["y_start"] / d["L"] * 16).astype(int)
        dots = ax.scatter(d["X"][0] / D, d["Y"][0] / D, s=4, c=np.where(band % 2 == 0, "0.1", "0.72"), lw=0, zorder=3)
        lines = [ax.plot([], [], color="k", lw=1.8, zorder=5)[0] for _ in d["centers"]]
        nodes = ax.plot([], [], "o", color="k", ms=4, zorder=6)[0]
        ax.set_xlim(0, d["n"])
        ax.set_ylim(0, d["n"])
        ax.set_aspect("equal")
        ax.set_xlabel("$x / D$")
        ax.set_ylabel("$y / D$")
        ax.set_title(name, fontsize=15)
        art.append((dots, lines, nodes))
    stamp = fig.text(0.5, 0.95, "", ha="center", fontsize=17)
    fig.text(0.5, 0.02, "Each cell's nodes (black dots) move around it; wall points move only in and out. "
             "Tracers start in horizontal stripes. Vibration drawn 2.5 times larger.",
             ha="center", fontsize=11.5, color="0.3")
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "24"])
    with writer.saving(fig, path, dpi=100):
        for fr in range(runs[0]["X"].shape[0]):
            for d, (dots, lines, nodes) in zip(runs, art):
                t = d["T"][fr]
                dots.set_offsets(np.c_[d["X"][fr] / d["D"], d["Y"][fr] / d["D"]])
                for ln, (xs, ys) in zip(lines, outlines(d, t, 2.5)):
                    ln.set_data(xs, ys)
                nx_, ny_ = [], []
                for (x0, y0), s in zip(d["centers"], d["sense"]):
                    a = s * d["Om"] * t + np.pi / (2 * d["m"]) + np.arange(2 * d["m"]) * np.pi / d["m"]
                    nx_ += list((x0 + d["R"] * np.cos(a)) / d["D"])
                    ny_ += list((y0 + d["R"] * np.sin(a)) / d["D"])
                nodes.set_data(nx_, ny_)
            stamp.set_text(f"Cells turned by moving nodes: cycle {fr / 12:4.1f}")
            writer.grab_frame()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    res_path = os.path.join(HERE, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    if not a.plot:
        for case in ("uniform", "sheared"):
            t0 = time.perf_counter()
            out = run(case)
            np.savez_compressed(os.path.join(HERE, f"run_{case}.npz"), **out)
            results[case] = dict(**measures(out), seconds=round(time.perf_counter() - t0, 1))
            print(case, results[case], flush=True)
            json.dump(results, open(res_path, "w"), indent=2)
    figure(os.path.join(HERE, "turning_cells.png"))
    movie(os.path.join(HERE, "turning_cells.mp4"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
