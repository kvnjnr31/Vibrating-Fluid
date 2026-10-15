"""Experiment 9: turning cells as Maxwell's molecular vortices -- swirl (magnetism) and gap currents.

    python experiments/09_maxwell_cells/maxwell.py            # 4 patterns + mode sweep, about 40 min on 2 cores
    python experiments/09_maxwell_cells/maxwell.py --plot     # figures and movie from saved runs

Changes from experiment 7 (to make the swirl visible and the gaps better resolved):
  * wider gaps: spacing D = 56, radius R = 20, gap 16 lattice spacings (8 at closest approach);
  * stronger drive: amplitude A = 4 and nodes moving at full speed, Omega_n = w / m;
  * longer runs (30 cycles spin-up + 120 cycles) and a stroboscopic movie: one frame per cycle,
    so the back-and-forth swing drops out and only the slow drift is left;
  * cells drawn at true size.

Cell wall:  r = R + A cos(m (theta - s Omega_n t)) cos(wt), s = +1 (nodes counterclockwise) or -1.

Spin patterns (s for the cell in column i, row j of a 4 x 4 lattice, periodic box):
  uniform   s = +1 everywhere
  sheared   s = +1 in the left two columns, -1 in the right two
  defect    s = +1 everywhere except one cell (column 1, row 1), which turns the other way
  checker   s = (-1)^(i + j)

Maxwell (1861, Part II): the vortex rotation is the magnetic force H; the idle particles between
vortices move at the mean of the neighboring surface velocities, and their net flow, the current,
is proportional to curl H. Here H is each cell's swirl and the current is the mean flow of the gap
fluid. In 2D, J = c curl(H z) means the stream function of the cell-averaged mean flow equals c H
plus a constant: the script tests that with one constant c for all four patterns.

Mode sweep: the sheared pattern for m = 2, 3, 4, 6 (same A and node speed), 44 cycles each.
Output: run_<pattern>.npz, mode_m<m>.npz, results.json, maxwell_fields.png, maxwell_law.png,
modes.png, maxwell_cells.mp4.
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
sys.path.insert(0, os.path.join(ROOT, "experiments", "07_turning_cells"))
from lbm import LBM  # noqa: E402
from lbm.ib import IBString  # noqa: E402
from turning import advect_periodic, style  # noqa: E402

PATTERNS = ("uniform", "sheared", "defect", "checker")


def senses(pattern, n):
    s = np.ones((n, n))
    if pattern == "sheared":
        s[n // 2:, :] = -1
    elif pattern == "defect":
        s[1, 1] = -1
    elif pattern == "checker":
        s = np.fromfunction(lambda i, j: (-1.0) ** (i + j), (n, n))
    return s


def build(pattern, m=2, n=4, D=56, R=20.0, A=4.0, node_speed=1.0, tau=1.0, delta=6.0):
    L = n * D
    sim = LBM(L, L, tau)
    w = 2 * sim.nu / delta ** 2
    period = int(round(2 * np.pi / w))
    w = 2 * np.pi / period
    Om = node_speed * w / m
    nm = int(np.ceil(2 * np.pi * (R + A)))
    th = 2 * np.pi * np.arange(nm) / nm
    S = senses(pattern, n)
    centers = np.array([((i + 0.5) * D + 0.25, (j + 0.5) * D + 0.25) for i in range(n) for j in range(n)])
    sense = np.array([S[i, j] for i in range(n) for j in range(n)])
    s = np.repeat(sense, nm)
    cx, cy = np.repeat(centers[:, 0], nm), np.repeat(centers[:, 1], nm)
    TH = np.tile(th, len(centers))
    dth = 2 * np.pi / nm
    ib = IBString(TH.size)

    def hook(t, sm):
        tm = t + 0.5
        ph = m * (TH - s * Om * tm)
        c, sn = np.cos(w * tm), np.sin(w * tm)
        r = R + A * np.cos(ph) * c
        rt = A * (m * s * Om * np.sin(ph) * c - w * np.cos(ph) * sn)
        drdth = -A * m * np.sin(ph) * c
        X = (cx + r * np.cos(TH)) % L
        Y = (cy + r * np.sin(TH)) % L
        ib.force(sm, X, Y, rt * np.cos(TH), rt * np.sin(TH), np.sqrt(r * r + drdth * drdth) * dth)

    par = dict(pattern=pattern, m=m, n=n, D=D, R=R, A=A, L=L, w=w, Om=Om, node_speed=node_speed, period=period,
               nu=sim.nu, delta=np.sqrt(2 * sim.nu / w), centers=centers, sense=sense)
    return sim, hook, par


def run(pattern, m=2, spin=30, cycles=120, avg_last=20, tracers=True, side=72):
    sim, hook, par = build(pattern, m)
    L, period = par["L"], par["period"]
    sim.run(spin * period, hook)
    if tracers:
        g = (np.arange(side) + 0.5) * L / side
        gx, gy = np.meshgrid(g, g, indexing="ij")
        px, py = gx.ravel().copy(), gy.ravel().copy()
        Xs = np.zeros((cycles + 1, px.size), np.float32)
        Ys = np.zeros_like(Xs)
        Xs[0], Ys[0] = px, py
    mu, mv = np.zeros((L, L)), np.zeros((L, L))
    for c in range(1, cycles + 1):
        for _ in range(period):
            sim.step(hook)
            if tracers or c > cycles - avg_last:
                _, ux, uy = sim.fields()
            if tracers:
                advect_periodic(ux, uy, px, py)
            if c > cycles - avg_last:
                mu += ux
                mv += uy
        if tracers:
            Xs[c], Ys[c] = px, py
    k = avg_last * period
    out = dict(**par, mu=mu / k, mv=mv / k, cycles=cycles)
    if tracers:
        out.update(X=Xs, Y=Ys, y_start=gy.ravel().copy())
    return out


def analyze(d):
    """Swirl per cell (H), mean vorticity, stream function, cell-averaged stream function and flow."""
    L, D, R, n = d["L"], d["D"], d["R"], d["n"]
    Aw = d["A"] * d["w"]
    mu, mv = d["mu"], d["mv"]
    k = 2 * np.pi * np.fft.fftfreq(L)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    U, V = np.fft.fft2(mu), np.fft.fft2(mv)
    vort = np.real(np.fft.ifft2(1j * KX * V - 1j * KY * U))
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0
    psi_h = np.fft.fft2(vort) / K2                       # u = d psi/dy, v = -d psi/dx  =>  lap psi = -vort
    psi_h[0, 0] = 0.0
    psi = np.real(np.fft.ifft2(psi_h))
    box = np.zeros((L, L))                                # average over one cell (D x D box), periodic
    h = D // 2
    box[:h + 1, :h + 1] = box[-h:, :h + 1] = box[:h + 1, -h:] = box[-h:, -h:] = 1.0
    box /= box.sum()
    B = np.fft.fft2(box)
    psi_c = np.real(np.fft.ifft2(np.fft.fft2(psi) * B))
    uc = np.real(np.fft.ifft2(U * B))
    vc = np.real(np.fft.ifft2(V * B))
    g = np.arange(L) + 0.0
    X, Y = np.meshgrid(g, g, indexing="ij")
    swirl, psi_cell = [], []
    for (x0, y0) in d["centers"]:
        dx = (X - x0 + L / 2) % L - L / 2
        dy = (Y - y0 + L / 2) % L - L / 2
        r = np.hypot(dx, dy)
        ring = np.abs(r - (R + d["A"] + 3)) < 1.0
        ut = (-mu * dy + mv * dx) / np.maximum(r, 1e-9)
        swirl.append(float(ut[ring].mean() / Aw))
        psi_cell.append(float(psi_c[int(round(x0)) % L, int(round(y0)) % L] / (Aw * D)))
    return dict(vort=vort, psi=psi, psi_c=psi_c, uc=uc, vc=vc, swirl=np.array(swirl),
                psi_cell=np.array(psi_cell) - np.mean(psi_cell))


def load(name):
    d = dict(np.load(os.path.join(HERE, f"{name}.npz")))
    return {k: (v.item() if v.ndim == 0 else v) for k, v in d.items()}


def outline(d, x0, y0, s, t):
    th = np.linspace(0, 2 * np.pi, 300)
    r = d["R"] + d["A"] * np.cos(d["m"] * (th - s * d["Om"] * t)) * np.cos(d["w"] * t)
    return (x0 + r * np.cos(th)) / d["D"], (y0 + r * np.sin(th)) / d["D"]


def draw_cells(ax, d, t=None):
    th = np.linspace(0, 2 * np.pi, 200)
    for (x0, y0), s in zip(d["centers"], d["sense"]):
        if t is None:
            ax.plot((x0 + d["R"] * np.cos(th)) / d["D"], (y0 + d["R"] * np.sin(th)) / d["D"], color="0.3", lw=1.0,
                    ls=(0, (3, 2)))
            ax.text(x0 / d["D"], y0 / d["D"], "ccw" if s > 0 else "cw", ha="center", va="center", fontsize=11,
                    color="0.15", bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="none", alpha=0.8))
        else:
            ax.plot(*outline(d, x0, y0, s, t), color="k", lw=1.6)


def fields_figure(path):
    plt = style()
    runs = [load(f"run_{p}") for p in PATTERNS]
    names = {"uniform": "Uniform spin", "sheared": "Sheared spin", "defect": "One reversed cell",
             "checker": "Checkerboard spin"}
    fig, axes = plt.subplots(2, 4, figsize=(21, 11.2), gridspec_kw=dict(left=0.04, right=0.99, top=0.93,
                                                                          bottom=0.08, wspace=0.18, hspace=0.3))
    for col, d in enumerate(runs):
        a = analyze(d)
        n, D, L = d["n"], d["D"], d["L"]
        g = (np.arange(L) + 0.5) / D
        Aw = d["A"] * d["w"]
        ax = axes[0, col]
        lim = np.percentile(np.abs(a["vort"]), 99) * D / Aw
        ax.pcolormesh(g, g, (a["vort"] * D / Aw).T, cmap="Greys", vmin=-lim, vmax=lim, shading="auto")
        ax.streamplot(g, g, d["mu"].T, d["mv"].T, color="k", linewidth=0.6, density=1.6, arrowsize=0.7)
        draw_cells(ax, d)
        ax.set_title(f"({'ABCD'[col]}) {names[d['pattern']]}: vortices" + "\n(mean vorticity; dark = ccw)",
                     fontsize=15)
        bx = axes[1, col]
        H = d["sense"].reshape(n, n) * np.abs(a["swirl"]).reshape(n, n)
        Hs = a["swirl"].reshape(n, n)
        bx.imshow(Hs.T, origin="lower", extent=(0, n, 0, n), cmap="Greys", vmin=-0.08, vmax=0.08, alpha=0.75)
        step = D // 4
        sl = slice(step // 2, L, step)
        Jx, Jy = a["uc"][sl, sl] / Aw, a["vc"][sl, sl] / Aw
        jmax = np.hypot(a["uc"], a["vc"]).max() / Aw
        bx.quiver(g[sl], g[sl], Jx.T, Jy.T, scale=jmax * 18, width=0.005, color="k")
        draw_cells(bx, d)
        bx.set_title(f"({'EFGH'[col]}) Swirl H (shade) and current J (arrows)" + f"\nlargest J = {jmax:.3f} Aω",
                     fontsize=15)
        for ax_ in (ax, bx):
            ax_.set_xlim(0, n)
            ax_.set_ylim(0, n)
            ax_.set_aspect("equal")
            ax_.set_xlabel("$x/D$")
            ax_.set_ylabel("$y/D$")
    fig.text(0.5, 0.012, "Top: mean flow and its vorticity; ccw/cw the way each cell's nodes turn. Bottom: each cell's swirl "
             "(dark = ccw) and the mean flow averaged over a D x D box (the current; arrows scaled to each panel).", ha="center",
             fontsize=12.5, color="0.3")
    fig.savefig(path, dpi=120)


def law_figure(path):
    """Test J = c curl H: the cell-averaged stream function should equal c H (+ const) in every pattern."""
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    marks = dict(uniform="s", sheared="o", defect="^", checker="D")
    allH, allP = [], []
    for p in PATTERNS:
        d = load(f"run_{p}")
        a = analyze(d)
        H = a["swirl"] - a["swirl"].mean()
        axes[0].plot(H, a["psi_cell"], marks[p], color="k", mfc="white", ms=8, label=p)
        allH += list(H)
        allP += list(a["psi_cell"])
    allH, allP = np.array(allH), np.array(allP)
    c = float(np.sum(allH * allP) / np.sum(allH * allH))
    resid = float(np.sqrt(np.mean((allP - c * allH) ** 2)) / np.sqrt(np.mean(allP ** 2)))
    hh = np.linspace(allH.min() * 1.1, allH.max() * 1.1, 10)
    axes[0].plot(hh, c * hh, color="0.55", lw=2.5, zorder=0, label=f"$\\psi_c = {c:.2f}\\,H$")
    axes[0].axhline(0, color="0.7", lw=0.8)
    axes[0].axvline(0, color="0.7", lw=0.8)
    axes[0].set_xlabel(r"cell swirl $H$ minus its average  [$A\omega$]")
    axes[0].set_ylabel(r"cell-averaged stream function $\psi_c$  [$A\omega D$]")
    axes[0].set_title(f"(A) Best single constant, misfit {100 * resid:.0f}%")
    axes[0].legend(frameon=False, fontsize=12)
    d = load("run_sheared")
    a = analyze(d)
    L, D, Aw, n = d["L"], d["D"], d["A"] * d["w"], d["n"]
    x = (np.arange(L) + 0.5) / D
    J = a["vc"].mean(axis=1) / Aw
    axes[1].plot(x, J, color="k", lw=2.2, label=r"current $J_y$: cell-averaged mean $v$")
    Hcol = [a["swirl"][i * n:(i + 1) * n].mean() for i in range(n)]
    xs = np.repeat(np.arange(n + 1), 2)[1:-1]
    axes[1].plot(xs, np.repeat(Hcol, 2), color="0.55", lw=2.5, ls=(0, (5, 3)), label="cell swirl $H$ (by column)")
    for xg in range(n + 1):
        axes[1].axvline(xg, color="0.8", lw=0.8)
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set_xlabel("$x/D$  (sheared: left two columns ccw, right two cw)")
    axes[1].set_ylabel(r"[$A\omega$]")
    axes[1].set_title("(B) Current Sheets Where the Spin Changes")
    axes[1].legend(frameon=False, fontsize=12, loc="lower center")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return dict(c=c, misfit=resid)


def modes_figure(ms, path):
    plt = style()
    fig = plt.figure(figsize=(20, 9.6))
    sw, cur = [], []
    for k, m in enumerate(ms):
        d = load(f"mode_m{m}")
        a = analyze(d)
        L, D, n, Aw = d["L"], d["D"], d["n"], d["A"] * d["w"]
        g = (np.arange(L) + 0.5) / D
        ax = fig.add_axes([0.03 + 0.18 * k, 0.43, 0.165, 0.5])
        lim = np.percentile(np.abs(a["vort"]), 99) * D / Aw
        ax.pcolormesh(g, g, (a["vort"] * D / Aw).T, cmap="Greys", vmin=-lim, vmax=lim, shading="auto")
        ax.streamplot(g, g, d["mu"].T, d["mv"].T, color="k", linewidth=0.5, density=1.4, arrowsize=0.6)
        for (x0, y0), s in zip(d["centers"], d["sense"]):
            ax.plot(*outline(d, x0, y0, s, 0.0), color="0.2", lw=1.0)
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect("equal")
        ax.set_xticks(range(n + 1))
        ax.set_yticks(range(n + 1))
        ax.set_title(f"({'ABCD'[k]}) $m = {m}$ ({2 * m} antinodes)", fontsize=15)
        sw.append(np.abs(a["swirl"]).mean())
        cur.append(np.abs(a["vc"].mean(axis=1)).max() / Aw)
    bx = fig.add_axes([0.06, 0.07, 0.38, 0.27])
    bx.plot(ms, sw, "o-", color="k", mfc="white", ms=8)
    bx.set_xlabel("mode $m$")
    bx.set_ylabel(r"cell swirl $|H|$ [$A\omega$]")
    bx.set_title("(E) Swirl per Cell", fontsize=15)
    cx = fig.add_axes([0.56, 0.07, 0.38, 0.27])
    cx.plot(ms, cur, "s-", color="k", mfc="white", ms=8)
    cx.set_xlabel("mode $m$")
    cx.set_ylabel(r"peak current $J_y$ [$A\omega$]")
    cx.set_title("(F) Current Between Opposite Spins", fontsize=15)
    for ax_ in (bx, cx):
        ax_.set_xticks(ms)
    fig.savefig(path, dpi=120)
    return dict(m=list(ms), swirl=[float(v) for v in sw], current=[float(v) for v in cur])


def movie(path, fps=12):
    plt = style()
    from matplotlib.animation import FFMpegWriter
    runs = [load(f"run_{p}") for p in PATTERNS]
    names = {"uniform": "(A) Uniform spin", "sheared": "(B) Sheared spin", "defect": "(C) One reversed cell",
             "checker": "(D) Checkerboard spin"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 13.2), gridspec_kw=dict(left=0.07, right=0.98, top=0.92,
                                                                          bottom=0.07, wspace=0.14, hspace=0.18))
    art = []
    for ax, d in zip(axes.ravel(), runs):
        D, n = d["D"], d["n"]
        band = np.floor(d["y_start"] / d["L"] * 16).astype(int)
        dots = ax.scatter(d["X"][0] / D, d["Y"][0] / D, s=3, c=np.where(band % 2 == 0, "0.1", "0.72"), lw=0, zorder=3)
        lines = []
        for (x0, y0), s in zip(d["centers"], d["sense"]):
            lines.append(ax.plot(*outline(d, x0, y0, s, 0.0), color="k", lw=1.4, zorder=5)[0])
            ax.text(x0 / D, y0 / D, "ccw" if s > 0 else "cw", ha="center", va="center", fontsize=11, color="0.2",
                    zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="none", alpha=0.8))
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect("equal")
        ax.set_xticks(range(n + 1))
        ax.set_yticks(range(n + 1))
        ax.set_title(names[d["pattern"]], fontsize=15)
        art.append((dots, lines))
    stamp = fig.text(0.5, 0.955, "", ha="center", fontsize=17)
    fig.text(0.5, 0.015, "One frame per cycle (the swing is hidden; only the drift shows). Cells at true size; "
             "ccw/cw: the way the nodes turn. Tracers start in horizontal stripes.", ha="center", fontsize=11.5,
             color="0.3")
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "23"])
    with writer.saving(fig, path, dpi=100):
        for fr in range(runs[0]["X"].shape[0]):
            for d, (dots, lines) in zip(runs, art):
                dots.set_offsets(np.c_[d["X"][fr] / d["D"], d["Y"][fr] / d["D"]])
                t = fr * d["period"]
                for ln, (x0, y0), s in zip(lines, d["centers"], d["sense"]):
                    ln.set_data(*outline(d, x0, y0, s, t))
            stamp.set_text(f"Turning cells (Maxwell's vortices), stroboscopic: cycle {fr}")
            writer.grab_frame()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plot", action="store_true")
    p.add_argument("--modes", type=int, nargs="+", default=[2, 3, 4, 6])
    a = p.parse_args()
    res_path = os.path.join(HERE, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    if not a.plot:
        for pat in PATTERNS:
            t0 = time.perf_counter()
            out = run(pat)
            np.savez_compressed(os.path.join(HERE, f"run_{pat}.npz"), **out)
            an = analyze(out)
            res[pat] = dict(swirl=an["swirl"].tolist(), seconds=round(time.perf_counter() - t0, 1))
            print(pat, round(float(np.abs(an["swirl"]).mean()), 4), res[pat]["seconds"], flush=True)
            json.dump(res, open(res_path, "w"), indent=2)
        for m in a.modes:
            t0 = time.perf_counter()
            out = run("sheared", m=m, spin=40, cycles=4, avg_last=4, tracers=False)
            np.savez_compressed(os.path.join(HERE, f"mode_m{m}.npz"), **out)
            print("mode", m, round(time.perf_counter() - t0, 1), flush=True)
    res["law"] = law_figure(os.path.join(HERE, "maxwell_law.png"))
    res["modes"] = modes_figure(a.modes, os.path.join(HERE, "modes.png"))
    json.dump(res, open(res_path, "w"), indent=2)
    fields_figure(os.path.join(HERE, "maxwell_fields.png"))
    movie(os.path.join(HERE, "maxwell_cells.mp4"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
