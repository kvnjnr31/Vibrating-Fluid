"""Experiment 4: a stack of permeable vibrating strings, toward a string at every height.

    python experiments/04_string_stack/stack.py --check        # A = 0: Darcy flow through the stack, 2 min
    python experiments/04_string_stack/stack.py                # stacks of 3, 5 and 9 strings, about 25 min
    python experiments/04_string_stack/stack.py --plot         # figure and movie from saved runs

Setup
  * Same box, source (q0 from below) and fixed-pressure outlet as experiments 2 and 3.
  * N strings, all in phase, w = A sin(kx) cos(wt), spread evenly through a band of height
    H = lambda/2 centered in the box: spacing h = H / (N - 1) = 32, 16, 8 for N = 3, 5, 9.
  * Permeation follows the notes' q = kappa (p_in - p_out): each string lets fluid through along its
    normal at q = kappa (p_below - p_above), with the pressure p = cs^2 rho read from the LBM a short
    distance on each side of the string. The value of q is relaxed toward kappa dp over about 50
    steps to keep the coupling smooth.
  * kappa = N q0 / dp_total, so the whole band always needs the same total pressure drop dp_total to
    pass q0. As N grows the stack approaches a smooth vibrating porous medium with fixed resistance:
    the "string at every height" limit.
  * Tangential velocity matches the string (no slip), as before.

Output: run_N*.npz, results.json, string_stack.png, string_stack.mp4 (densest stack).
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
from lbm.boundaries import outlet_top  # noqa: E402
from tracers import advect, style  # noqa: E402


def build(N, A=2.0, q0=0.005, lam=128, half=128, tau=1.0, delta=6.0, H=64, drho_total=0.01, side=3.0,
          relax=0.02):
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
    yc = half + 0.75
    ys = yc + (np.arange(N) - (N - 1) / 2) * (H / (N - 1))
    kappa = N * q0 / (drho_total / 3.0)                  # p = rho / 3
    X1 = np.arange(lam) + 0.5
    X = np.tile(X1, N)
    S, C = np.sin(kw * X), np.cos(kw * X)
    Y0 = np.repeat(ys, lam)
    ib = IBString(N * lam)
    q = np.full(N * lam, q0)
    rb, ra = np.zeros(N * lam), np.zeros(N * lam)
    dummy = np.zeros(N * lam)

    def hook(t, s):
        tm = t + 0.5
        wd = A * S * np.cos(w * tm)
        wt = -A * w * S * np.sin(w * tm)
        wx = A * kw * C * np.cos(w * tm)
        norm = np.sqrt(1 + wx * wx)
        nxv, nyv = -wx / norm, 1.0 / norm
        Y = Y0 + wd
        rho, _, _ = s.fields()
        interpolate(rho, rho, X - side * nxv, Y - side * nyv, rb, dummy)      # below the string
        interpolate(rho, rho, X + side * nxv, Y + side * nyv, ra, dummy)      # above the string
        q[:] += relax * (kappa * (rb - ra) / 3.0 - q)
        ib.force(s, X, Y, q * nxv, wt + q * nyv, norm)

    periods = int(np.ceil(half ** 2 / sim.nu / period))
    par = dict(N=N, A=A, q0=q0, lam=lam, y0=yc, ys=ys, H=H, h=H / (N - 1), kappa=kappa, w=w, kw=kw,
               nu=sim.nu, delta=np.sqrt(2 * sim.nu / w), period=period, periods=periods)
    return sim, hook, par, q


def check(N=5):
    """A = 0: plain Darcy flow. Each string should drop the density by drho_total / N and pass q0."""
    sim, hook, par, q = build(N, A=0.0)
    sim.run(40000, hook)
    rho, ux, uy = sim.fields()
    prof = rho.mean(axis=0)
    drop = prof[int(par["ys"][0] - 8)] - prof[int(par["ys"][-1] + 8)]
    return dict(N=N, q_mean_over_q0=float(q.mean() / par["q0"]), q_spread=float(np.ptp(q) / par["q0"]),
                drho_band=float(drop), drho_expected=0.01, uy_far_over_q0=float(uy[:, -20].mean() / par["q0"]))


def run(N, cycles=30, per_cycle=12, cols=64, rows=64, tracers=False):
    sim, hook, par, q = build(N)
    lam, yc, period = par["lam"], par["y0"], par["period"]
    sim.run(par["periods"] * period, hook)
    every = period // per_cycle
    nf = cycles * per_cycle
    if tracers:
        gx, gy = np.meshgrid((np.arange(cols) + 0.5) * lam / cols, yc + lam * ((np.arange(rows) + 0.5) / rows - 0.5))
        px, py = gx.ravel().copy(), gy.ravel().copy()
        Xs = np.zeros((nf + 1, px.size), np.float32)
        Ys = np.zeros_like(Xs)
        Ws = np.zeros((nf + 1, px.size), bool)
        T = np.zeros(nf + 1, np.int64)
        Xs[0], Ys[0], T[0] = px, py, sim.t
        wrapped = np.zeros(px.size, np.bool_)
    mu = np.zeros((sim.nx, sim.ny))
    mv = np.zeros_like(mu)
    qm = np.zeros_like(q)
    snaps = {}
    for fr in range(1, nf + 1):
        for _ in range(every):
            sim.step(hook)
            rho, ux, uy = sim.fields()
            if tracers:
                advect(ux, uy, px, py, lam, yc - 0.5 * lam, yc + 0.5 * lam, wrapped)
            mu += ux
            mv += uy
            qm += q
            ph = sim.t % period
            if fr > nf - per_cycle and ph in (0, period // 2):
                snaps["a" if ph == 0 else "b"] = (ux.copy(), uy.copy(), rho.copy())
        if tracers:
            Xs[fr], Ys[fr], Ws[fr], T[fr] = px, py, wrapped, sim.t
            wrapped[:] = False
    n_avg = nf * every
    mu /= n_avg
    mv /= n_avg
    qm /= n_avg
    flux = {d: float(mv[:, int(round(yc + d))].sum() / (par["q0"] * lam)) for d in (-60, 0, 60)}
    out = dict(**par, mu=mu, mv=mv, q_mean=qm, snap_a_u=snaps["a"][0], snap_a_v=snaps["a"][1],
               snap_a_rho=snaps["a"][2], snap_b_u=snaps["b"][0], snap_b_v=snaps["b"][1])
    if tracers:
        out.update(X=Xs, Y=Ys, W=Ws, T=T, y_start=gy.ravel().copy())
    return out, flux


def load(N):
    d = dict(np.load(os.path.join(HERE, f"run_N{N}.npz")))
    return {k: (v.item() if v.ndim == 0 else v) for k, v in d.items()}


def jets(d, above=18):
    """Amplitude of cos(2kx) in the mean upward flow, (v - q0)/q0, a fixed distance above the stack."""
    x = np.arange(d["lam"]) + 0.5
    j = int(round(d["ys"][-1] + above))
    return float(2 * np.mean((d["mv"][:, j] - d["q0"]) * np.cos(2 * d["kw"] * x)) / d["q0"])


def figure(Ns, path):
    plt = style()
    runs = [load(N) for N in Ns]
    fig = plt.figure(figsize=(18, 12.2))
    lam = runs[0]["lam"]
    x = (np.arange(lam) + 0.5) / lam
    xs = np.linspace(0, 1, 400)

    def draw(ax, d, U, V, title, t=None, mean=False):
        y = (np.arange(U.shape[1]) - d["y0"]) / lam
        win = np.abs(y) < 0.45
        Uw, Vw = U[:, win], V[:, win] - (d["q0"] if mean else 0.0)
        sp = np.hypot(Uw, Vw)
        ax.contourf(*np.meshgrid(x, y[win], indexing="ij"), sp / sp.max(), levels=np.linspace(0, 1, 11),
                    cmap="Greys", alpha=0.45)
        ax.streamplot(x, y[win], Uw.T, Vw.T, color="k", linewidth=0.75, density=1.4, arrowsize=0.8)
        for y0 in d["ys"]:
            yy = (y0 - d["y0"]) / lam
            if mean:
                env = d["A"] * np.abs(np.sin(2 * np.pi * xs)) / lam
                ax.fill_between(xs, yy - env, yy + env, color="0.3", lw=0)
            else:
                ax.plot(xs, yy + d["A"] * np.sin(2 * np.pi * xs) * np.cos(d["w"] * t) / lam, color="0.15", lw=1.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.45, 0.45)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x/\lambda$")
        ax.set_ylabel(r"$y/\lambda$")
        ax.set_title(title)

    L = iter("ABCDEFGH")
    for n, d in enumerate(runs):
        ax = fig.add_axes([0.04 + 0.245 * n, 0.53, 0.22, 0.42])
        draw(ax, d, d["mu"], d["mv"], f"({next(L)}) {d['N']} strings, mean", mean=True)
    dn = runs[-1]
    draw(fig.add_axes([0.04, 0.05, 0.22, 0.42]), dn, dn["snap_a_u"], dn["snap_a_v"],
         rf"({next(L)}) {dn['N']} strings, $\omega t = 0$", t=0.0)
    draw(fig.add_axes([0.285, 0.05, 0.22, 0.42]), dn, dn["snap_b_u"], dn["snap_b_v"],
         rf"({next(L)}) {dn['N']} strings, $\omega t = \pi$", t=np.pi / dn["w"])
    bx = fig.add_axes([0.58, 0.08, 0.17, 0.37])
    for d, mk in zip(runs, ("s", "^", "o")):
        y = np.arange(d["mv"].shape[1]) - d["y0"]
        xl = np.arange(lam) + 0.5                          # lattice units for the wavenumber
        amp = [2 * np.mean((d["mv"][:, j] - d["q0"]) * np.cos(2 * d["kw"] * xl)) / d["q0"] for j in range(len(y))]
        m = np.abs(y / lam) < 0.45
        bx.plot(np.array(amp)[m], y[m] / lam, mk + "-", color="k", mfc="white", ms=3.5, lw=0.7, markevery=4,
                label=f"N = {d['N']}")
    bx.axvline(0, color="k", lw=0.8)
    bx.axhspan(-0.25, 0.25, color="0.9", zorder=0)
    bx.set_ylim(-0.45, 0.45)
    bx.set_xlabel(r"$\cos 2kx$ amplitude of $(\bar v - q_0)/q_0$")
    bx.set_ylabel(r"$y/\lambda$")
    bx.set_title(f"({next(L)}) + nodes, − antinodes")
    bx.legend(frameon=False, fontsize=11, loc="lower right")
    cx = fig.add_axes([0.81, 0.08, 0.17, 0.37])
    for d, mk in zip(runs, ("s", "^", "o")):
        prof = d["snap_a_rho"].mean(axis=0)
        y = np.arange(prof.size) - d["y0"]
        m = np.abs(y / lam) < 0.45
        cx.plot((prof[m] - 1) * 1e3, y[m] / lam, color="k", lw=1.2 if mk == "o" else 0.8,
                ls={"s": ":", "^": "--", "o": "-"}[mk], label=f"N = {d['N']}")
    cx.axhspan(-0.25, 0.25, color="0.9", zorder=0)
    cx.set_ylim(-0.45, 0.45)
    cx.set_xlabel(r"$(\rho - 1) \times 10^3$, $x$-average")
    cx.set_title(f"({next(L)}) Pressure Through the Stack")
    cx.legend(frameon=False, fontsize=11)
    fig.savefig(path, dpi=130)


def movie(N, path, fps=24):
    plt = style()
    from matplotlib.animation import FFMpegWriter
    d = load(N)
    lam = d["lam"]
    xs = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(8.2, 8.4))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.9, bottom=0.1)
    band = np.floor(((d["y_start"] - d["y0"]) / lam + 0.5) * 8).astype(int)
    dots = ax.scatter(d["X"][0] / lam, (d["Y"][0] - d["y0"]) / lam, s=4, c=np.where(band % 2 == 0, "0.1", "0.72"),
                      lw=0, zorder=3)
    lines = [ax.plot([], [], color="k", lw=1.6, zorder=5)[0] for _ in d["ys"]]
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/\lambda$")
    ax.set_ylabel(r"$y/\lambda$")
    stamp = ax.set_title("")
    fig.text(0.5, 0.02, f"{N} permeable strings in phase, Darcy permeation, source below. Tracers start in flat stripes.",
             ha="center", fontsize=11, color="0.3")
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "24"])
    with writer.saving(fig, path, dpi=110):
        for fr in range(d["X"].shape[0]):
            dots.set_offsets(np.c_[d["X"][fr] / lam, (d["Y"][fr] - d["y0"]) / lam])
            for ln, y0 in zip(lines, d["ys"]):
                ln.set_data(xs, (y0 - d["y0"] + d["A"] * np.sin(2 * np.pi * xs) * np.cos(d["w"] * d["T"][fr])) / lam)
            stamp.set_text(f"Stack of {N} strings: cycle {fr / 12:4.1f}")
            writer.grab_frame()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--N", type=int, nargs="+", default=[3, 5, 9])
    a = p.parse_args()
    res_path = os.path.join(HERE, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    if a.check:
        r = check()
        results["check_A0"] = r
        print(r)
        json.dump(results, open(res_path, "w"), indent=2)
        return
    if not a.plot:
        for N in a.N:
            t0 = time.perf_counter()
            out, flux = run(N, tracers=(N == max(a.N)))
            np.savez_compressed(os.path.join(HERE, f"run_N{N}.npz"), **out)
            results[f"N{N}"] = dict(flux_over_q0_lambda=flux, jets_18_above=jets(out),
                                    q_mean_over_q0=float(out["q_mean"].mean() / out["q0"]),
                                    seconds=round(time.perf_counter() - t0, 1))
            print(N, results[f"N{N}"], flush=True)
            json.dump(results, open(res_path, "w"), indent=2)
    figure(a.N, os.path.join(HERE, "string_stack.png"))
    movie(max(a.N), os.path.join(HERE, "string_stack.mp4"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
