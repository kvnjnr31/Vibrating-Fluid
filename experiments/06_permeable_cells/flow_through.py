"""Experiment 6: flow from the left through an array of permeable vibrating cells (tracer movie).

    python experiments/06_permeable_cells/flow_through.py           # both lockings, about 25 min on 2 cores
    python experiments/06_permeable_cells/flow_through.py --plot    # figure and movie from saved runs

Setup
  * The 4 x 4 array of experiment 5 (R = 20, spacing D = 48, oval mode m = 2, A = 2), with an open
    stretch of one spacing D on each side: box 6D x 4D, periodic top to bottom.
  * Fluid enters through the left edge at q0 = 0.0025 (moving-wall inlet) and leaves through the right
    edge at fixed pressure. It starts in uniform flow.
  * The cell walls are permeable as in the derivation: fluid crosses each wall along its normal at
    q = kappa (p_in - p_out), with the pressure read 3 spacings inside and outside the wall
    (relaxed over about 50 steps), on top of moving with the wall. kappa = 4, so a density step of
    about 0.002 across a wall drives q0 through it.
  * Two lockings, as in experiment 5: aligned (all in phase) and geared (neighbors opposite).
  * After 50 cycles of spin-up, 60 cycles with 3456 tracers that start in vertical stripes.

Output: run_<locking>.npz, results.json, permeable_cells.png, permeable_cells.mp4.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "02_permeable_string"))
from lbm import LBM  # noqa: E402
from lbm.ib import IBString, interpolate  # noqa: E402
from lbm.boundaries import outlet_right  # noqa: E402
from tracers import style  # noqa: E402
from numba import njit  # noqa: E402


@njit(cache=True)
def advect_x(ux, uy, px, py, x_lo, x_hi, wrapped):
    """Midpoint step with bilinear velocity; periodic in y; leaving on the right re-enters on the left."""
    nx, ny = ux.shape
    for n in range(px.shape[0]):
        x, y = px[n], py[n]
        for stage in range(2):
            i0 = int(np.floor(x))
            j0 = int(np.floor(y))
            fx, fy = x - i0, y - j0
            i0 = min(max(i0, 0), nx - 2)
            i1 = i0 + 1
            j0 %= ny
            j1 = (j0 + 1) % ny
            u = ((1 - fx) * (1 - fy) * ux[i0, j0] + fx * (1 - fy) * ux[i1, j0]
                 + (1 - fx) * fy * ux[i0, j1] + fx * fy * ux[i1, j1])
            v = ((1 - fx) * (1 - fy) * uy[i0, j0] + fx * (1 - fy) * uy[i1, j0]
                 + (1 - fx) * fy * uy[i0, j1] + fx * fy * uy[i1, j1])
            if stage == 0:
                x = px[n] + 0.5 * u
                y = py[n] + 0.5 * v
        px[n] += u
        py[n] = (py[n] + v) % ny
        if px[n] > x_hi:
            px[n] -= (x_hi - x_lo)
            wrapped[n] = True


def build(locking, m=2, n=4, D=48, R=20.0, A=2.0, q0=0.0025, kappa=4.0, tau=1.0, delta=6.0, side=3.0, relax=0.02):
    nx, ny = (n + 2) * D, n * D
    sim = LBM(nx, ny, tau)
    sim.solid[0, :] = True
    sim.uwx[0, :] = q0
    sim.set_velocity(q0, 0.0)
    sim.post = lambda s: outlet_right(s.f, 1.0)
    w = 2 * sim.nu / delta ** 2
    period = int(round(2 * np.pi / w))
    w = 2 * np.pi / period
    nm = int(np.ceil(2 * np.pi * R))
    th = 2 * np.pi * np.arange(nm) / nm
    centers, phases = [], []
    for i in range(n):
        for j in range(n):
            centers.append(((i + 1.5) * D + 0.25, (j + 0.5) * D + 0.25))
            phases.append(0.0 if (locking == "aligned" or (i + j) % 2 == 0) else np.pi)
    centers = np.array(centers)
    ph = np.repeat(np.array(phases), nm)
    cx, cy = np.repeat(centers[:, 0], nm), np.repeat(centers[:, 1], nm)
    TH = np.tile(th, len(centers))
    shape, dshape = np.cos(m * TH), -m * np.sin(m * TH)
    dth = 2 * np.pi / nm
    ib = IBString(TH.size)
    q = np.zeros(TH.size)
    rin, rout, dummy = np.zeros(TH.size), np.zeros(TH.size), np.zeros(TH.size)
    er = np.c_[np.cos(TH), np.sin(TH)]
    et = np.c_[-np.sin(TH), np.cos(TH)]

    def hook(t, s):
        tm = t + 0.5
        c, sn = np.cos(w * tm + ph), np.sin(w * tm + ph)
        r = R + A * shape * c
        rt = -A * w * shape * sn
        drdth = A * dshape * c
        X = cx + r * er[:, 0]
        Y = (cy + r * er[:, 1]) % ny
        nvec = r[:, None] * er - drdth[:, None] * et                 # outward normal of r(theta)
        nn = np.hypot(nvec[:, 0], nvec[:, 1])
        nxv, nyv = nvec[:, 0] / nn, nvec[:, 1] / nn
        rho, _, _ = s.fields()
        interpolate(rho, rho, X - side * nxv, (Y - side * nyv) % ny, rin, dummy)
        interpolate(rho, rho, X + side * nxv, (Y + side * nyv) % ny, rout, dummy)
        q[:] += relax * (kappa * (rin - rout) / 3.0 - q)
        ib.force(s, X, Y, rt * er[:, 0] + q * nxv, rt * er[:, 1] + q * nyv, nn * dth)

    par = dict(locking=locking, m=m, n=n, D=D, R=R, A=A, q0=q0, kappa=kappa, nx=nx, ny=ny, w=w, period=period,
               nu=sim.nu, delta=np.sqrt(2 * sim.nu / w), centers=centers, cell_phase=np.array(phases), nm=nm)
    return sim, hook, par, q


def run(locking, spin=50, cycles=60, per_cycle=12, cols=72, rows=48):
    sim, hook, par, q = build(locking)
    nx, ny, period = par["nx"], par["ny"], par["period"]
    sim.run(spin * period, hook)
    gx, gy = np.meshgrid((np.arange(cols) + 0.5) * (nx - 2) / cols + 1, (np.arange(rows) + 0.5) * ny / rows,
                         indexing="ij")
    px, py = gx.ravel().copy(), gy.ravel().copy()
    every = period // per_cycle
    nf = cycles * per_cycle
    Xs = np.zeros((nf + 1, px.size), np.float32)
    Ys = np.zeros_like(Xs)
    Ws = np.zeros((nf + 1, px.size), bool)
    T = np.zeros(nf + 1, np.int64)
    Xs[0], Ys[0], T[0] = px, py, sim.t
    wrapped = np.zeros(px.size, np.bool_)
    mu, mv = np.zeros((nx, ny)), np.zeros((nx, ny))
    qa, qabs = np.zeros_like(q), np.zeros_like(q)
    for fr in range(1, nf + 1):
        for _ in range(every):
            sim.step(hook)
            _, ux, uy = sim.fields()
            advect_x(ux, uy, px, py, 1.0, nx - 2.0, wrapped)
            mu += ux
            mv += uy
            qa += q
            qabs += np.abs(q)
        Xs[fr], Ys[fr], Ws[fr], T[fr] = px, py, wrapped, sim.t
        wrapped[:] = False
    k = nf * every
    out = dict(**par, mu=mu / k, mv=mv / k, q_mean=qa / k, q_abs=qabs / k, X=Xs, Y=Ys, W=Ws, T=T,
               x_start=gx.ravel().copy())
    # fraction of the incoming flow that passes through cell walls (one crossing in, one out)
    ds = 2 * np.pi * par["R"] / par["nm"]
    through = float(np.sum(np.clip(out["q_mean"], 0, None)) * ds)
    out["through_fraction"] = through / (par["q0"] * ny)
    flux = {lab: float(out["mu"][int(x), :].sum() / (par["q0"] * ny)) for lab, x in
            (("inlet side", par["D"] // 2), ("middle", nx // 2), ("outlet side", nx - par["D"] // 2))}
    return out, flux


def load(locking):
    d = dict(np.load(os.path.join(HERE, f"run_{locking}.npz")))
    return {k: (v.item() if v.ndim == 0 else v) for k, v in d.items()}


def cell_outlines(d, wt, scale=1.0):
    th = np.linspace(0, 2 * np.pi, 300)
    out = []
    for (x0, y0), ph in zip(d["centers"], d["cell_phase"]):
        r = d["R"] + scale * d["A"] * np.cos(d["m"] * th) * np.cos(wt + ph)
        out.append(((x0 + r * np.cos(th)) / d["D"], (y0 + r * np.sin(th)) / d["D"]))
    return out


def figure(path):
    plt = style()
    runs = [load("aligned"), load("geared")]
    fig = plt.figure(figsize=(18, 11))
    for row, (d, name) in enumerate(zip(runs, ("Aligned (all in phase)", "Geared (neighbors opposite)"))):
        ax = fig.add_axes([0.04, 0.55 - 0.48 * row, 0.56, 0.4])
        nx, ny, D = d["nx"], d["ny"], d["D"]
        gx, gy = (np.arange(nx) + 0.5) / D, (np.arange(ny) + 0.5) / D
        U, V = d["mu"], d["mv"]
        sp = np.hypot(U, V) / d["q0"]
        cf = ax.contourf(gx, gy, sp.T, levels=np.linspace(0, 2.5, 11), cmap="Greys", extend="max")
        ax.streamplot(gx, gy, U.T, V.T, color="k", linewidth=0.7, density=2.0, arrowsize=0.8)
        for xs, ys in cell_outlines(d, 0.0, 0.0):
            ax.plot(xs, ys, color="0.15", lw=1.6)
        ax.set_xlim(0, nx / D)
        ax.set_ylim(0, ny / D)
        ax.set_aspect("equal")
        ax.set_xlabel("$x / D$")
        ax.set_ylabel("$y / D$")
        ax.set_title(f"({'AB'[row]}) {name}: mean flow, " + f"{100 * d['through_fraction']:.0f}% passes through cell walls")
        if row == 0:
            cb = fig.colorbar(cf, ax=ax, fraction=0.025, pad=0.01)
            cb.set_label(r"mean speed / $q_0$")
        bx = fig.add_axes([0.7, 0.55 - 0.48 * row, 0.27, 0.4])
        nm = d["nm"]
        th = np.degrees(2 * np.pi * np.arange(nm) / nm)
        qm = d["q_mean"].reshape(-1, nm) / d["q0"]
        for k in range(qm.shape[0]):
            bx.plot(th, qm[k], color="0.75", lw=0.8)
        bx.plot(th, qm.mean(axis=0), color="k", lw=2.4, label="average of 16 cells")
        for a in range(0, 361, 90):
            bx.axvline(a, color="k", lw=0.7, ls=(0, (1, 2)))
        bx.axhline(0, color="0.4", lw=0.8)
        bx.set_xlim(0, 360)
        bx.set_xticks(range(0, 361, 45))
        bx.set_xlabel(r"angle around the cell, $\theta$ (deg); dotted: antinodes")
        bx.set_ylabel(r"mean flow out through the wall, $\bar q / q_0$")
        bx.set_title(f"({'CD'[row]}) Where fluid crosses the walls")
        bx.legend(frameon=False, fontsize=11)
    fig.text(0.5, 0.01, "Fluid enters on the left at q0 and leaves on the right. theta = 0 faces downstream, "
             "180 deg faces the incoming flow.", ha="center", fontsize=12, color="0.3")
    fig.savefig(path, dpi=130)


def movie(path, fps=24):
    plt = style()
    from matplotlib.animation import FFMpegWriter
    runs = [load("aligned"), load("geared")]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 13.6), gridspec_kw=dict(left=0.08, right=0.98, top=0.93,
                                                                           bottom=0.06, hspace=0.16))
    art = []
    for ax, d, name in zip(axes, runs, ("(A) Aligned (all in phase)", "(B) Geared (neighbors opposite)")):
        D = d["D"]
        stripe = np.floor((d["x_start"] - 1) / (d["nx"] - 2) * 12).astype(int)
        cols = np.where(stripe % 2 == 0, "0.1", "0.72")
        dots = ax.scatter(d["X"][0] / D, d["Y"][0] / D, s=5, c=cols, lw=0, zorder=3)
        lines = [ax.plot([], [], color="k", lw=1.8, zorder=5)[0] for _ in d["centers"]]
        ax.set_xlim(0, d["nx"] / D)
        ax.set_ylim(0, d["ny"] / D)
        ax.set_aspect("equal")
        ax.set_xlabel("$x / D$")
        ax.set_ylabel("$y / D$")
        ax.set_title(name)
        art.append((dots, lines))
    stamp = fig.text(0.5, 0.965, "", ha="center", fontsize=17)
    fig.text(0.5, 0.015, "Permeable vibrating cells; fluid enters on the left. Tracers start in vertical stripes.",
             ha="center", fontsize=12, color="0.3")
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "24"])
    with writer.saving(fig, path, dpi=100):
        for fr in range(runs[0]["X"].shape[0]):
            for d, (dots, lines) in zip(runs, art):
                dots.set_offsets(np.c_[d["X"][fr] / d["D"], d["Y"][fr] / d["D"]])
                wt = d["w"] * d["T"][fr]
                for ln, (xs, ys) in zip(lines, cell_outlines(d, wt)):
                    ln.set_data(xs, ys)
            stamp.set_text(f"Flow through permeable vibrating cells: cycle {fr / 12:4.1f}")
            writer.grab_frame()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    res_path = os.path.join(HERE, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    if not a.plot:
        for locking in ("aligned", "geared"):
            t0 = time.perf_counter()
            out, flux = run(locking)
            np.savez_compressed(os.path.join(HERE, f"run_{locking}.npz"), **out)
            results[locking] = dict(flux_over_q0_height=flux, through_fraction=out["through_fraction"],
                                    seconds=round(time.perf_counter() - t0, 1))
            print(locking, results[locking], flush=True)
            json.dump(results, open(res_path, "w"), indent=2)
    figure(os.path.join(HERE, "permeable_cells.png"))
    movie(os.path.join(HERE, "permeable_cells.mp4"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
