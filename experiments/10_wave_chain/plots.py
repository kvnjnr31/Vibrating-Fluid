"""Figures and movie for experiment 10 (wave chain).

    python experiments/10_wave_chain/plots.py

wave_chain.mp4   the last 12 cycles of each free run, in real time: cell shapes (true size) and fluid
                 speed, with the driven cell in the middle; below each, every cell's mode a now
                 (dots) and its swing over the last cycle (bars).
em_analog.png    the chain read as an electromagnetic medium: space-time plots of the wave,
                 attenuation (lossy conductor against gain medium), phase per cell (wavenumber),
                 and the two oval modes of each cell as the two polarizations of a wave.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CASES = ("passive", "active", "spin")
TITLE = {"passive": "Passive cells (route 1: thin Stokes layers, delta/R = 0.15)",
         "active": "Self-driven cells (route 2: each cell makes up its own loss)",
         "spin": "Spin drive (route 3: driven cell's nodes move around it; cells self-driven)"}


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                         "font.size": 14, "axes.titlelocation": "left", "axes.titlesize": 15})
    return plt


def load(case):
    d = dict(np.load(os.path.join(HERE, f"run_{case}.npz")))
    return {k: (v.item() if v.ndim == 0 else v) for k, v in d.items()}


def order(N):
    """Display order with the driven cell (0) in the middle; and distance of each cell from it."""
    idx = np.r_[np.arange(N // 2 + 1, N), np.arange(0, N // 2 + 1)]
    dist = np.minimum(np.arange(N), N - np.arange(N))
    return idx, dist


def response(d, periods_avg=4):
    t, w, T = d["t"], d["w"], d["period"]
    m = t > t[-1] - periods_avg * T
    e = np.exp(-1j * w * t[m])[:, None]
    return 2 * np.mean(d["a"][m] * e, axis=0), 2 * np.mean(d["b"][m] * e, axis=0)


def movie(path, fps=16):
    plt = style()
    from matplotlib.animation import FFMpegWriter
    runs = [load(c) for c in CASES if os.path.exists(os.path.join(HERE, f"run_{c}.npz"))]
    d0 = runs[0]
    N, D, R, Lx, Ly, dn = d0["N"], d0["D"], d0["R"], d0["Lx"], d0["Ly"], d0["down"]
    idx, _ = order(N)
    shift = Lx / 2 - d0["centers"][0, 0]                     # put cell 0 in the middle
    th = np.linspace(0, 2 * np.pi, 160)
    fig = plt.figure(figsize=(16, 3.9 * len(runs) + 0.8))
    art = []
    vmax = max(np.percentile(r["speed"].astype(float), 99.5) for r in runs)
    for k, d in enumerate(runs):
        top = 1 - (k * 3.9 + 0.5) / (3.9 * len(runs) + 0.8)
        h = 3.9 / (3.9 * len(runs) + 0.8)
        ax = fig.add_axes([0.05, top - 0.5 * h, 0.9, 0.42 * h])
        bx = fig.add_axes([0.05, top - 0.93 * h, 0.9, 0.3 * h])
        sp = np.roll(d["speed"][0].astype(float), int(round(shift / dn)), axis=0)
        im = ax.imshow(sp.T, origin="lower", extent=(0, Lx / D, 0, Ly / D), cmap="Greys", vmin=0, vmax=vmax,
                       interpolation="bilinear", aspect="equal")
        lines = [ax.plot([], [], color="k", lw=1.5)[0] for _ in range(N)]
        ax.set_xlim(0, Lx / D)
        ax.set_ylim(0, Ly / D)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f"({'ABC'[k]}) {TITLE[d['case']]}", fontsize=14)
        xs = ((d["centers"][:, 0] + shift) % Lx) / D
        bars = bx.vlines(xs, 0, 0, color="0.7", lw=10)
        dots = bx.plot(xs, np.zeros(N), "o", color="k", ms=6)[0]
        if d["case"] == "spin":
            dotsb = bx.plot(xs + 0.12, np.zeros(N), "s", color="0.45", ms=5)[0]
        else:
            dotsb = None
        bx.axhline(0, color="0.5", lw=0.8)
        bx.set_xlim(0, Lx / D)
        bx.set_ylim(-1.15, 1.15)
        bx.set_xticks(xs)
        bx.set_xticklabels([str(min(i, N - i)) for i in range(N)])
        bx.set_ylabel("$a / A$")
        if k == len(runs) - 1:
            bx.set_xlabel("cells from the driven cell (driven cell in the middle)" +
                          ("; spin panel: dots a, squares b" if any(r["case"] == "spin" for r in runs) else ""))
        art.append((im, lines, bars, dots, dotsb, xs))
    stamp = fig.text(0.5, 1 - 0.25 / (3.9 * len(runs) + 0.8), "", ha="center", fontsize=16)
    per = d0["period"]
    nf = min(r["fa"].shape[0] for r in runs)
    fpp = int(round(per / (d0["ft"][1] - d0["ft"][0]))) if nf > 1 else 24
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "24"])
    with writer.saving(fig, path, dpi=96):
        for fr in range(nf):
            for d, (im, lines, bars, dots, dotsb, xs) in zip(runs, art):
                im.set_data(np.roll(d["speed"][fr].astype(float), int(round(shift / dn)), axis=0).T)
                a, b = d["fa"][fr], d["fb"][fr]
                for i, ln in enumerate(lines):
                    r = R + a[i] * np.cos(2 * th) + b[i] * np.sin(2 * th)
                    x0 = (d["centers"][i, 0] + shift) % Lx
                    ln.set_data((x0 + r * np.cos(th)) / D, (d["centers"][i, 1] + r * np.sin(th)) / D)
                lo = max(0, fr - fpp)
                swing = np.abs(d["fa"][lo:fr + 1]).max(axis=0) / d["A"]
                bars.set_segments([[(x, -s), (x, s)] for x, s in zip(xs, swing)])
                dots.set_ydata(a / d["A"])
                if dotsb is not None:
                    dotsb.set_ydata(b / d["A"])
            stamp.set_text(f"Waves along a chain of vibrating cells (real time, shading = fluid speed): "
                           f"cycle {d0['ft'][fr] / per:5.2f}")
            writer.grab_frame()


def em_figure(path, ref_decay=4.4):
    plt = style()
    runs = {c: load(c) for c in CASES if os.path.exists(os.path.join(HERE, f"run_{c}.npz"))}
    d0 = next(iter(runs.values()))
    N, T, A = d0["N"], d0["period"], d0["A"]
    idx, dist = order(N)
    fig = plt.figure(figsize=(19, 12))
    # (A, B) space-time
    for k, c in enumerate([c for c in ("passive", "active") if c in runs]):
        d = runs[c]
        ax = fig.add_axes([0.05 + 0.31 * k, 0.56, 0.27, 0.38])
        tt = d["t"] / T
        m = tt > tt[-1] - 8
        ax.pcolormesh(np.arange(N) - (N - 1 - N // 2), tt[m], (d["a"][m][:, idx] / A), cmap="Greys", vmin=-1, vmax=1,
                      shading="nearest")
        ax.set_xlabel("cell, relative to the driven cell")
        ax.set_ylabel("time (cycles)")
        ax.set_title(f"({'AB'[k]}) {c.capitalize()}: mode a in space and time" +
                     ("\nlike a wave in a lossy conductor" if c == "passive" else "\nlike a wave in a gain medium"),
                     fontsize=14)
    # (C) attenuation and (D) phase
    cx = fig.add_axes([0.68, 0.56, 0.29, 0.38])
    dx = fig.add_axes([0.05, 0.07, 0.27, 0.38])
    marks = {"passive": "o", "active": "s", "spin": "^"}
    out = {}
    for c, d in runs.items():
        Xa, Xb = response(d)
        amp = np.abs(Xa) / A
        ph = np.angle(Xa / Xa[0])
        right = np.arange(0, N // 2 + 1)
        cx.semilogy(right, amp[right], marks[c] + "-", color="k", mfc="white", ms=7, label=f"{c}, mode a")
        if c == "spin":
            cx.semilogy(right, np.abs(Xb)[right] / A, "^--", color="0.45", mfc="white", ms=7, label="spin, mode b")
        phu = np.unwrap(ph[right])
        dx.plot(right, -phu / np.pi, marks[c] + "-", color="k", mfc="white", ms=7, label=c)
        lag = float(np.mean(np.diff(-phu[: N // 2])) / np.pi)
        dec = float(np.exp(-np.mean(np.diff(np.log(np.maximum(amp[right][: N // 2], 1e-9))))))
        out[c] = dict(lag_per_cell_over_pi=lag, amplitude_ratio_per_cell=dec,
                      wavelength_cells=float(2 / lag) if lag > 0 else None)
    rr = np.arange(0, N // 2 + 1)
    cx.semilogy(rr, 0.22 * ref_decay ** (-(rr - 1.0)) * (rr > 0) + (rr == 0), color="0.7", lw=2.5, zorder=0,
                label=f"experiment 8 (delta/R = 0.3): /{ref_decay} per cell")
    cx.set_ylim(1e-4, 2)
    cx.set_xlabel("cells from the driven cell")
    cx.set_ylabel("amplitude / A")
    cx.set_title("(C) Attenuation: skin depth of a conductor\nagainst a medium with gain", fontsize=14)
    cx.legend(frameon=False, fontsize=10.5, loc="lower left")
    dx.set_xlabel("cells from the driven cell")
    dx.set_ylabel(r"phase lag / $\pi$")
    dx.set_title("(D) Phase lag per cell = wavenumber k D\n(lag of 2 pi = one wavelength)", fontsize=14)
    dx.legend(frameon=False, fontsize=11)
    # (E) polarization ellipses for the spin run
    ex = fig.add_axes([0.38, 0.05, 0.59, 0.25])
    tx = fig.add_axes([0.38, 0.31, 0.59, 0.17])
    tx.axis("off")
    rows = [("Cell chain", "Electromagnetic analog"),
            ("oval modes a, b of a cell", "two polarization components of the field"),
            ("phase lag per cell", "wavenumber k (here about 0.6 pi per cell: wavelength about 3.4 cells)"),
            ("viscous loss in the fluid (passive)", "conductivity: the wave dies within a skin depth"),
            ("each cell's own energy supply (active)", "gain medium (as in a laser): loss made up, wave keeps going"),
            ("nodes turning (spin drive)", "circular polarization; S = helicity"),
            ("a and b modes coupled differently", "birefringence: circular turns elliptical and linear")]
    for i, (l, r) in enumerate(rows):
        y = 1 - i / len(rows)
        w_ = "bold" if i == 0 else "normal"
        tx.text(0.0, y, l, fontsize=12, va="top", weight=w_)
        tx.text(0.42, y, r, fontsize=12, va="top", weight=w_)
    if "spin" in runs:
        d = runs["spin"]
        m = d["t"] > d["t"][-1] - T
        for j, i in enumerate(idx):
            a, b = d["a"][m][:, i] / A, d["b"][m][:, i] / A
            ex.plot(j + 0.42 * a, 0.42 * b, color="k", lw=1.6)
            ex.plot(j + 0.42 * a[-1], 0.42 * b[-1], "o", color="k", ms=4)
            S = np.mean(a[:-1] * np.diff(b) - b[:-1] * np.diff(a)) / (d["w"] * (d["t"][1] - d["t"][0])) / 1.0
            ex.text(j, -0.62, f"S={S:+.2f}", ha="center", fontsize=10.5)
        ex.set_xlim(-0.6, N - 0.4)
        ex.set_ylim(-0.75, 0.55)
        ex.set_aspect("equal")
        ex.set_xticks(range(N))
        ex.set_xticklabels([str(j - (N - 1 - N // 2)) for j in range(N)])
        ex.set_yticks([])
        ex.set_xlabel("cell, relative to the driven cell")
        ex.set_title("(E) Polarization (spin run): each cell's path in the (a, b) plane over one cycle\n"
                     "circle = circular polarization (nodes turning); line = linear; S = spin (+1 ccw, -1 cw)",
                     fontsize=13)
    fig.savefig(path, dpi=120)
    return out


def main():
    res_path = os.path.join(HERE, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    res["em_analog"] = em_figure(os.path.join(HERE, "em_analog.png"))
    json.dump(res, open(res_path, "w"), indent=2)
    print(json.dumps(res["em_analog"], indent=1))
    movie(os.path.join(HERE, "wave_chain.mp4"))
    print("movie done")


if __name__ == "__main__":
    main()
