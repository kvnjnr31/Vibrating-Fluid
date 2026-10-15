"""Static figure: where the oval lobes and nodes sit on the four-lobed tension pattern.

    python experiments/13_four_lobed_tension/lobes_figure.py
"""
import os

import numpy as np
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
DELTA = 0.24


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                         "font.size": 14, "axes.titlelocation": "left", "axes.titlesize": 15})
    return plt


def tension(th):
    return 1 + DELTA * np.cos(4 * th)


def shell(ax, mode, title):
    th = np.linspace(0, 2 * np.pi, 721)
    # ring shaded by tension: dark = high, light = low
    pts = np.c_[np.cos(th), np.sin(th)]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="Greys", linewidths=11, zorder=1)
    lc.set_array(tension(0.5 * (th[:-1] + th[1:])))
    lc.set_clim(1 - DELTA * 1.6, 1 + DELTA)
    ax.add_collection(lc)
    eta = np.cos(2 * th) if mode == "a" else np.sin(2 * th)
    r = 1 + 0.18 * eta
    ax.plot(r * np.cos(th), r * np.sin(th), color="k", lw=2.4, zorder=3)
    r2 = 1 - 0.18 * eta
    ax.plot(r2 * np.cos(th), r2 * np.sin(th), color="k", lw=1.2, ls="--", zorder=3)
    lob = np.arange(4) * np.pi / 2 + (0 if mode == "a" else np.pi / 4)
    nod = lob + np.pi / 4
    ax.plot((1 + 0.18) * np.cos(lob), (1 + 0.18) * np.sin(lob), "o", color="k", ms=10, zorder=4)
    ax.plot(np.cos(nod), np.sin(nod), "s", mfc="w", mec="k", mew=2, ms=9, zorder=4)
    for t in lob:
        ax.text(1.42 * np.cos(t), 1.42 * np.sin(t), "lobe", ha="center", va="center", fontsize=12)
    for t in nod:
        ax.text(1.45 * np.cos(t), 1.45 * np.sin(t), "node", ha="center", va="center", fontsize=12, color="0.35")
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.85, 1.85)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)


def main():
    plt = style()
    fig = plt.figure(figsize=(15, 11.5))
    deg = np.linspace(0, 360, 721)
    th = np.radians(deg)
    # (A) tension around the shell
    ax = fig.add_axes([0.07, 0.66, 0.40, 0.28])
    ax.plot(deg, tension(th), color="k", lw=2.4)
    hi = np.arange(0, 361, 90)
    lo = np.arange(45, 361, 90)
    ax.plot(hi, tension(np.radians(hi)), "v", color="k", ms=10, label="tension high (slope 0)")
    ax.plot(lo, tension(np.radians(lo)), "^", mfc="w", mec="k", mew=2, ms=10, label="tension low (slope 0)")
    for x in np.arange(0, 361, 45):
        ax.axvline(x, color="0.85", lw=1, zorder=0)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xlim(0, 360)
    ax.set_ylim(0.7, 1.35)
    ax.set_xlabel("angle around the shell (degrees)")
    ax.set_ylabel("tension / mean")
    ax.legend(loc="upper center", ncol=2, fontsize=11, frameon=False)
    ax.set_title("(A) Tension 1 + 0.24 cos 4$\\theta$: slope 0 every 45$^\\circ$")
    # (D) bending: where the wall bends, the tension counts
    ax = fig.add_axes([0.56, 0.66, 0.40, 0.28])
    ax.plot(deg, (2 * np.sin(2 * th)) ** 2, color="k", lw=2.4, label="oval a: bending")
    ax.plot(deg, (2 * np.cos(2 * th)) ** 2, color="0.5", lw=2.4, ls="--", label="oval b: bending")
    for x in hi:
        ax.axvline(x, color="k", lw=0.8, ls=":")
    for x in np.arange(0, 361, 45):
        ax.axvline(x, color="0.85", lw=1, zorder=0)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 5.2)
    ax.set_xlabel("angle around the shell (degrees)")
    ax.set_ylabel("wall slope squared")
    ax.legend(loc="upper center", ncol=2, fontsize=11, frameon=False)
    ax.set_title("(B) Each oval bends most at its nodes (dotted: tension highs)")
    shell(fig.add_axes([0.03, 0.04, 0.44, 0.47]), "a",
          "(C) Oval a: lobes on tension highs, nodes on lows\n      bends at low tension: softer, K(1 $-$ 0.2)")
    shell(fig.add_axes([0.52, 0.04, 0.44, 0.47]), "b",
          "(D) Oval b: lobes on tension lows, nodes on highs\n      bends at high tension: stiffer, K(1 + 0.2)")
    fig.text(0.5, 0.015, "Ring shading: tension (dark = high). Solid and dashed outlines: the oval half a cycle apart. "
             "Dots: lobes (antinodes). Squares: nodes. The 1.50 wave in experiment 13 is oval b.",
             ha="center", fontsize=11.5, color="0.3")
    fig.savefig(os.path.join(HERE, "lobes_and_tension.png"), dpi=120)


if __name__ == "__main__":
    main()
