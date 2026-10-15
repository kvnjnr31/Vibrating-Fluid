"""Movie of experiment 9 that shows the vibrating shells as well as the drift.

    python experiments/09_maxwell_cells/movie_full.py           # clips (about 12 min) and movie
    python experiments/09_maxwell_cells/movie_full.py --movie   # movie only, from saved clips

Part 1, real time: after the same 30-cycle spin-up as maxwell.py, 6 cycles with 24 frames per
cycle. The shells vibrate and their nodes travel around them; tracers swing back and forth.
Part 2, one frame per cycle: the 120-cycle drift from run_<pattern>.npz (maxwell.py). The swing is
hidden here, so each shell is drawn at its stroboscopic shape with its full vibration range as a grey
band behind it.

Output: clip_<pattern>.npz, maxwell_cells_full.mp4.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from maxwell import PATTERNS, build, load, outline, advect_periodic, style  # noqa: E402

NAMES = {"uniform": "(A) Uniform spin", "sheared": "(B) Sheared spin", "defect": "(C) One reversed cell",
         "checker": "(D) Checkerboard spin"}


def clip(pattern, spin=30, cycles=6, per_cycle=24, side=72):
    sim, hook, par = build(pattern)
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
    for fr in range(1, nf + 1):
        for _ in range(every):
            sim.step(hook)
            _, ux, uy = sim.fields()
            advect_periodic(ux, uy, px, py)
        Xs[fr], Ys[fr], T[fr] = px, py, sim.t
    np.savez_compressed(os.path.join(HERE, f"clip_{pattern}.npz"), X=Xs, Y=Ys, T=T, y_start=gy.ravel().copy(),
                        per_cycle=per_cycle)


def band(d, x0, y0):
    """Outer and inner limits of the shell over a cycle (radius R +- A)."""
    th = np.linspace(0, 2 * np.pi, 200)
    ro, ri = d["R"] + d["A"], d["R"] - d["A"]
    return (np.r_[(x0 + ro * np.cos(th)), (x0 + ri * np.cos(th[::-1]))] / d["D"],
            np.r_[(y0 + ro * np.sin(th)), (y0 + ri * np.sin(th[::-1]))] / d["D"])


def movie(path, fps=24, strobe_repeat=2):
    plt = style()
    from matplotlib.animation import FFMpegWriter
    runs = {p: load(f"run_{p}") for p in PATTERNS}
    clips = {p: dict(np.load(os.path.join(HERE, f"clip_{p}.npz"))) for p in PATTERNS}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 13.4), gridspec_kw=dict(left=0.07, right=0.98, top=0.91,
                                                                          bottom=0.07, wspace=0.14, hspace=0.18))
    art = {}
    for ax, p in zip(axes.ravel(), PATTERNS):
        d = runs[p]
        D, n = d["D"], d["n"]
        bands = [ax.fill(*band(d, x0, y0), color="0.88", lw=0, zorder=1)[0] for (x0, y0) in d["centers"]]
        stripe = np.floor(clips[p]["y_start"] / d["L"] * 16).astype(int)
        dots = ax.scatter(clips[p]["X"][0] / D, clips[p]["Y"][0] / D, s=3, lw=0, zorder=3,
                          c=np.where(stripe % 2 == 0, "0.1", "0.7"))
        lines = [ax.plot([], [], color="k", lw=1.6, zorder=5)[0] for _ in d["centers"]]
        nodes = ax.plot([], [], "o", color="k", ms=3.5, zorder=6)[0]
        for (x0, y0), s in zip(d["centers"], d["sense"]):
            ax.text(x0 / D, y0 / D, "ccw" if s > 0 else "cw", ha="center", va="center", fontsize=10, color="0.25",
                    zorder=7, bbox=dict(boxstyle="round,pad=0.12", fc="w", ec="none", alpha=0.75))
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect("equal")
        ax.set_xticks(range(n + 1))
        ax.set_yticks(range(n + 1))
        ax.set_title(NAMES[p], fontsize=15)
        art[p] = (bands, dots, lines, nodes)
    stamp = fig.text(0.5, 0.955, "", ha="center", fontsize=16)
    note = fig.text(0.5, 0.015, "", ha="center", fontsize=11.5, color="0.3")

    def draw(p, X, Y, t, show_bands):
        d = runs[p]
        bands, dots, lines, nodes = art[p]
        for b in bands:
            b.set_visible(show_bands)
        dots.set_offsets(np.c_[X / d["D"], Y / d["D"]])
        nx_, ny_ = [], []
        for ln, (x0, y0), s in zip(lines, d["centers"], d["sense"]):
            ln.set_data(*outline(d, x0, y0, s, t))
            a = s * d["Om"] * t + np.pi / (2 * d["m"]) + np.arange(2 * d["m"]) * np.pi / d["m"]
            nx_ += list((x0 + d["R"] * np.cos(a)) / d["D"])
            ny_ += list((y0 + d["R"] * np.sin(a)) / d["D"])
        nodes.set_data(nx_, ny_)

    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "28"])
    with writer.saving(fig, path, dpi=100):
        pc = int(clips[PATTERNS[0]]["per_cycle"])
        for fr in range(clips[PATTERNS[0]]["X"].shape[0]):
            for p in PATTERNS:
                c = clips[p]
                draw(p, c["X"][fr], c["Y"][fr], c["T"][fr], False)
            stamp.set_text(f"Part 1, real time: shells vibrate, nodes (dots) travel around them — cycle {fr / pc:4.2f}")
            note.set_text("Cells at true size. Tracers swing back and forth each cycle. ccw/cw: the way the nodes turn.")
            writer.grab_frame()
        for fr in range(runs[PATTERNS[0]]["X"].shape[0]):
            for p in PATTERNS:
                d = runs[p]
                draw(p, d["X"][fr], d["Y"][fr], fr * d["period"], True)
            stamp.set_text(f"Part 2, one frame per cycle: only the drift shows — cycle {fr}")
            note.set_text("Grey band: the range each shell sweeps every cycle (still vibrating, hidden by the "
                          "one-frame-per-cycle view). Tracers restart in stripes.")
            for _ in range(strobe_repeat):
                writer.grab_frame()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--movie", action="store_true")
    a = p.parse_args()
    if not a.movie:
        for pat in PATTERNS:
            t0 = time.perf_counter()
            clip(pat)
            print("clip", pat, round(time.perf_counter() - t0), "s", flush=True)
    movie(os.path.join(HERE, "maxwell_cells_full.mp4"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
