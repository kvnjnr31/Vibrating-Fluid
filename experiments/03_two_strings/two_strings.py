"""Experiment 3: two parallel permeable vibrating strings, in phase and in opposite phase.

    python experiments/03_two_strings/two_strings.py            # both runs, about 15 min on 2 cores
    python experiments/03_two_strings/two_strings.py --plot     # figure and movie from saved runs

Same box, source, outlet and string condition as experiment 2 (k = +omega, A = 2, q0 = 0.005),
with two strings a distance D = lambda / 4 apart, centered in the box, nodes lined up:

    lower string  w1 = A sin(kx) cos(wt)
    upper string  w2 = A sin(kx) cos(wt + phi),    phi = 0 (in phase) or pi (opposite phase)

Each run spins up, then runs 30 more cycles while it carries 4096 tracers and averages the flow.
Output: run_phi*.npz, results.json, two_strings.png, two_strings.mp4.
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
from lbm.ib import IBString  # noqa: E402
from lbm.boundaries import outlet_top  # noqa: E402
from tracers import advect, style  # noqa: E402


def build(phi, k_over_w=1.0, A=2.0, q0=0.005, lam=128, half=128, tau=1.0, delta=6.0, D=32):
    nx, ny = lam, 2 * half + 2
    sim = LBM(nx, ny, tau)
    sim.solid[:, 0] = True
    sim.uwy[:, 0] = q0
    sim.set_velocity(0.0, q0)
    sim.post = lambda s: outlet_top(s.f, 1.0)
    w = 2 * sim.nu / delta ** 2
    period = int(round(2 * np.pi / w))
    w = 2 * np.pi / period
    kw = 2 * np.pi / lam
    kp = k_over_w * w
    yc = half + 0.75
    ys = (yc - D / 2, yc + D / 2)
    phases = (0.0, phi)
    X1 = np.arange(lam) + 0.5
    X = np.concatenate([X1, X1])
    S, C = np.sin(kw * X1), np.cos(kw * X1)
    ib = IBString(2 * lam)

    def hook(t, s):
        tm = t + 0.5
        Ys, Ux, Uy, ds = [], [], [], []
        for y0, ph in zip(ys, phases):
            wd = A * S * np.cos(w * tm + ph)
            wt = -A * w * S * np.sin(w * tm + ph)
            wx = A * kw * C * np.cos(w * tm + ph)
            norm = np.sqrt(1 + wx * wx)
            q = q0 + kp * wd
            Ys.append(y0 + wd)
            Ux.append(-q * wx / norm)
            Uy.append(wt + q / norm)
            ds.append(norm)
        ib.force(s, X, np.concatenate(Ys), np.concatenate(Ux), np.concatenate(Uy), np.concatenate(ds))

    periods = int(np.ceil(half ** 2 / sim.nu / period))
    par = dict(phi=phi, k_over_w=k_over_w, A=A, q0=q0, lam=lam, y0=yc, ys=np.array(ys), D=D, w=w, kw=kw,
               kp=kp, nu=sim.nu, delta=np.sqrt(2 * sim.nu / w), period=period, periods=periods)
    return sim, hook, par


def run(phi, cycles=30, per_cycle=12, cols=64, rows=64):
    sim, hook, par = build(phi)
    lam, yc, period = par["lam"], par["y0"], par["period"]
    sim.run(par["periods"] * period, hook)
    gx, gy = np.meshgrid((np.arange(cols) + 0.5) * lam / cols, yc + lam * ((np.arange(rows) + 0.5) / rows - 0.5))
    px, py = gx.ravel().copy(), gy.ravel().copy()
    y_lo, y_hi = yc - 0.5 * lam, yc + 0.5 * lam
    every = period // per_cycle
    nf = cycles * per_cycle
    Xs = np.zeros((nf + 1, px.size), np.float32)
    Ys = np.zeros_like(Xs)
    Ws = np.zeros((nf + 1, px.size), bool)
    T = np.zeros(nf + 1, np.int64)
    Xs[0], Ys[0], T[0] = px, py, sim.t
    wrapped = np.zeros(px.size, np.bool_)
    mu = np.zeros((sim.nx, sim.ny))
    mv = np.zeros_like(mu)
    snaps = {}
    for fr in range(1, nf + 1):
        for _ in range(every):
            sim.step(hook)
            _, ux, uy = sim.fields()
            advect(ux, uy, px, py, lam, y_lo, y_hi, wrapped)
            mu += ux
            mv += uy
            ph = sim.t % period
            if fr > nf - per_cycle and ph in (0, period // 2):
                snaps["a" if ph == 0 else "b"] = (ux.copy(), uy.copy())
        Xs[fr], Ys[fr], Ws[fr], T[fr] = px, py, wrapped, sim.t
        wrapped[:] = False
    n_avg = nf * every
    mu /= n_avg
    mv /= n_avg
    flux = {d: float(mv[:, int(round(yc + d))].sum() / (par["q0"] * lam)) for d in (-60, 0, 60)}
    out = dict(**par, mu=mu, mv=mv, snap_a_u=snaps["a"][0], snap_a_v=snaps["a"][1], snap_b_u=snaps["b"][0],
               snap_b_v=snaps["b"][1], X=Xs, Y=Ys, W=Ws, T=T, y_start=gy.ravel().copy())
    return out, flux


def load(phi):
    d = dict(np.load(os.path.join(HERE, f"run_phi{int(round(np.degrees(phi)))}.npz")))
    return {k: (v.item() if v.ndim == 0 else v) for k, v in d.items()}


def strings(d, t):
    xs = np.linspace(0, 1, 400)
    return [(xs, (y0 - d["y0"] + d["A"] * np.sin(2 * np.pi * xs) * np.cos(d["w"] * t + ph)) / d["lam"])
            for y0, ph in zip(d["ys"], (0.0, d["phi"]))]


def field(ax, d, U, V, title, t=None, mean=False):
    lam = d["lam"]
    x = (np.arange(lam) + 0.5) / lam
    y = (np.arange(U.shape[1]) - d["y0"]) / lam
    win = np.abs(y) < 0.4
    Uw, Vw = U[:, win], V[:, win] - (d["q0"] if mean else 0.0)
    sp = np.hypot(Uw, Vw)
    ax.contourf(*np.meshgrid(x, y[win], indexing="ij"), sp / sp.max(), levels=np.linspace(0, 1, 11),
                cmap="Greys", alpha=0.45)
    ax.streamplot(x, y[win], Uw.T, Vw.T, color="k", linewidth=0.8, density=1.4, arrowsize=0.9)
    if mean:
        xs = np.linspace(0, 1, 400)
        for y0 in d["ys"]:
            env = d["A"] * np.abs(np.sin(2 * np.pi * xs)) / lam
            ax.fill_between(xs, (y0 - d["y0"]) / lam - env, (y0 - d["y0"]) / lam + env, color="0.3", lw=0)
    else:
        for xs, ys in strings(d, t):
            ax.plot(xs, ys, color="0.15", lw=2.4)
    for y0 in d["ys"]:
        yy = (y0 - d["y0"]) / lam
        ax.plot([0, 0.5, 1], [yy] * 3, "o", color="k", ms=7, zorder=5, clip_on=False)
        ax.plot([0.25, 0.75], [yy] * 2, "o", mfc="white", mec="k", ms=7, mew=1.3, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 0.4)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/\lambda$")
    ax.set_ylabel(r"$y/\lambda$")
    ax.set_title(title)


def figure(path):
    plt = style()
    runs = [load(0.0), load(np.pi)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12.4), gridspec_kw=dict(left=0.05, right=0.99, top=0.95,
                                                                        bottom=0.06, wspace=0.18, hspace=0.22))
    for row, (d, name) in enumerate(zip(runs, ("In phase", "Opposite phase"))):
        L = "ABC" if row == 0 else "DEF"
        field(axes[row, 0], d, d["snap_a_u"], d["snap_a_v"], rf"({L[0]}) {name}, $\omega t = 0$", t=0.0)
        field(axes[row, 1], d, d["snap_b_u"], d["snap_b_v"], rf"({L[1]}) {name}, $\omega t = \pi$", t=np.pi / d["w"])
        field(axes[row, 2], d, d["mu"], d["mv"], rf"({L[2]}) {name}, mean (throughflow removed)", mean=True)
    fig.savefig(path, dpi=130)


def movie(path, fps=24):
    plt = style()
    from matplotlib.animation import FFMpegWriter
    runs = [load(0.0), load(np.pi)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.6), gridspec_kw=dict(left=0.06, right=0.98, top=0.88,
                                                                       bottom=0.1, wspace=0.12))
    art = []
    for ax, d, name in zip(axes, runs, ("(A) In phase", "(B) Opposite phase")):
        lam = d["lam"]
        band = np.floor(((d["y_start"] - d["y0"]) / lam + 0.5) * 8).astype(int)
        cols = np.where(band % 2 == 0, "0.1", "0.72")
        dots = ax.scatter(d["X"][0] / lam, (d["Y"][0] - d["y0"]) / lam, s=4, c=cols, lw=0, zorder=3)
        lines = [ax.plot([], [], color="k", lw=2.4, zorder=5)[0] for _ in d["ys"]]
        for y0 in d["ys"]:
            yy = (y0 - d["y0"]) / lam
            ax.plot([0, 0.5, 1], [yy] * 3, "o", color="k", ms=8, zorder=6, clip_on=False)
            ax.plot([0.25, 0.75], [yy] * 2, "o", mfc="white", mec="k", ms=8, mew=1.4, zorder=6)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x/\lambda$")
        ax.set_ylabel(r"$y/\lambda$")
        ax.set_title(name)
        art.append((dots, lines))
    stamp = fig.text(0.5, 0.95, "", ha="center", fontsize=17)
    fig.text(0.5, 0.025, "Two permeable strings (k > 0), source below. Tracers start in flat stripes. "
             "Black dots: nodes, open dots: antinodes.", ha="center", fontsize=12, color="0.3")
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "24"])
    with writer.saving(fig, path, dpi=110):
        for fr in range(runs[0]["X"].shape[0]):
            for d, (dots, lines) in zip(runs, art):
                lam = d["lam"]
                dots.set_offsets(np.c_[d["X"][fr] / lam, (d["Y"][fr] - d["y0"]) / lam])
                for ln, (xs, ys) in zip(lines, strings(d, d["T"][fr])):
                    ln.set_data(xs, ys)
            stamp.set_text(f"Two vibrating strings: cycle {fr / 12:4.1f}")
            writer.grab_frame()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    res_path = os.path.join(HERE, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    if not a.plot:
        for phi in (0.0, np.pi):
            t0 = time.perf_counter()
            out, flux = run(phi)
            name = f"phi{int(round(np.degrees(phi)))}"
            np.savez_compressed(os.path.join(HERE, f"run_{name}.npz"), **out)
            results[name] = dict(flux_over_q0_lambda=flux, seconds=round(time.perf_counter() - t0, 1))
            print(name, results[name], flush=True)
            json.dump(results, open(res_path, "w"), indent=2)
    figure(os.path.join(HERE, "two_strings.png"))
    movie(os.path.join(HERE, "two_strings.mp4"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
