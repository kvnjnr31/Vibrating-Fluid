"""Qualitative figure: flow through a permeable vibrating shell, seen in one slice.

    python shell_flux_figure.py                      # mode 4, writes shell_flux.png and .pdf
    python shell_flux_figure.py --mode 6 --amp 0.12 --out shell_flux_m6

The shell sits at r = R + w, with w = W cos(wt) Y and Y = cos(m theta) in the slice.

(A) Shell shape (solid) against the mean circle (dashed). Arrows give the shell
    normal n = r_hat - grad_s w, the direction in which fluid leaves the shell.
    At the antinodes the shell is flat and n points along the mean radius. At the
    nodal lines the slope is largest and n leans sideways; at one nodal line n is
    split into its part along the mean radius and its part along the sphere.
(B) Cycle-averaged sideways flux <J_s> ~ -grad_s(Y^2) for k > 0 (q = k w): it runs
    from each antinode toward the nearest nodal line, where the flow collects.

Self-contained (numpy and matplotlib only). Grey scale, in the style of the paper figures.
"""
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def paper_style():
    """Arial-type font, plain left-aligned titles, larger text, white background."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlesize": 22,
        "font.size": 20, "legend.fontsize": 15, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
    })


def shell(theta, R, A, m):
    """Point on the shell, unit normal, and the radial and sideways unit vectors at angle theta."""
    r = R + A * np.cos(m * theta)
    dr = -A * m * np.sin(m * theta)
    er = np.array([np.cos(theta), np.sin(theta)])
    et = np.array([-np.sin(theta), np.cos(theta)])
    n = r * er - dr * et                       # outward normal of the polar curve r(theta)
    return r * er, n / np.linalg.norm(n), er, et


def arrow(ax, start, vec, color, lw, scale=1.0):
    ax.arrow(*start, *vec, color=color, lw=lw, head_width=0.045 * scale, head_length=0.06 * scale,
             length_includes_head=True)


def panel_normals(ax, R, A, m):
    th = np.linspace(0, 2 * np.pi, 1200)
    r = R + A * np.cos(m * th)
    ax.plot(R * np.cos(th), R * np.sin(th), color="0.55", lw=1.6, ls=(0, (5, 4)))
    ax.plot(r * np.cos(th), r * np.sin(th), color="k", lw=2.6)
    antinodes = np.arange(2 * m) * np.pi / m
    nodes = np.pi / (2 * m) + np.arange(2 * m) * np.pi / m
    for t in antinodes:                        # flat shell: flow along the mean radius
        p, n, *_ = shell(t, R, A, m)
        arrow(ax, p, 0.32 * n, "0.45", 2.2)
    t0 = nodes[0]
    for t in nodes:                            # steepest shell: normal leans sideways
        p, n, *_ = shell(t, R, A, m)
        ax.plot(*p, "o", color="k", ms=8)
        if t != t0:
            arrow(ax, p, 0.34 * n, "k", 2.4)
    p, n, er, et = shell(t0, R, A, m)          # split the normal at one nodal line
    L = 0.62
    pr, pt = L * np.dot(n, er) * er, L * np.dot(n, et) * et
    arrow(ax, p, pr, "0.55", 2.2)
    arrow(ax, p + pr, pt, "0.55", 2.2)
    arrow(ax, p, L * n, "k", 2.8)
    ax.text(*(p + L * n + np.array([-0.05, 0.06])), "normal $n$", fontsize=17)
    ax.text(*(p + 0.6 * pr + np.array([0.07, -0.10])), r"part along $\hat{r}$", fontsize=15, color="0.35")
    ax.text(*(p + pr + 0.5 * pt + np.array([0.06, 0.0])), "part along the\nsphere, " + r"$-\nabla_s w$",
            fontsize=15, color="0.35")
    pa, *_ = shell(0.0, R, A, m)
    ax.text(pa[0] + 0.36, pa[1], "antinode", va="center", fontsize=16)
    ax.text(*(p * 0.80 + np.array([-0.08, -0.04])), "nodal\nline", fontsize=16, ha="center")
    ax.text(0, 0.06, "fluid inside", ha="center", fontsize=16, color="0.4")
    ax.text(0, -0.12, "dashed: mean radius $R$", ha="center", fontsize=14, color="0.5")
    ax.set_title("(A) Flow Leaves Along the Shell Normal")


def panel_mean_flux(ax, R, m, band=0.12, ring=1.16):
    th = np.linspace(0, 2 * np.pi, 1200)
    Y2 = np.cos(m * th) ** 2
    ax.plot(R * np.cos(th), R * np.sin(th), color="0.55", lw=1.6, ls=(0, (5, 4)))
    outer = R + band * Y2                      # grey band: Y^2, where the shell moves most
    ax.fill(np.r_[outer * np.cos(th), R * np.cos(th[::-1])], np.r_[outer * np.sin(th), R * np.sin(th[::-1])],
            color="0.85", lw=0)
    antinodes = np.arange(2 * m) * np.pi / m
    nodes = np.pi / (2 * m) + np.arange(2 * m) * np.pi / m
    half = np.pi / (2 * m)
    for t in antinodes:                        # <J_s> ~ -d(Y^2)/dtheta: from each antinode to its two nodes
        for sgn in (1, -1):
            a = np.linspace(t + sgn * 0.06, t + sgn * (half - 0.07), 30)
            ax.plot(ring * np.cos(a), ring * np.sin(a), color="k", lw=2.2)
            e = np.array([np.cos(a[-1]), np.sin(a[-1])])
            tg = sgn * np.array([-np.sin(a[-1]), np.cos(a[-1])])
            arrow(ax, ring * e - 0.02 * tg, 0.02 * tg, "k", 2.2, scale=1.3)
    for t in nodes:
        ax.plot(R * np.cos(t), R * np.sin(t), "o", color="k", ms=11)
    for t in antinodes:
        ax.plot(R * np.cos(t), R * np.sin(t), "o", mfc="white", mec="k", ms=10, mew=1.6)
    ax.plot([], [], "o", color="k", ms=10, label="nodal line: flow collects")
    ax.plot([], [], "o", mfc="white", mec="k", ms=10, mew=1.6, label="antinode: flow leaves")
    ax.fill([], [], color="0.85", label=r"$Y^2$ (shell motion)")
    ax.legend(loc="center", handlelength=1.2)
    ax.set_title(r"(B) Mean Sideways Flux, $\langle J_s\rangle \propto -\nabla_s(Y^2)$")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, default=4, help="number of wavelengths around the slice, Y = cos(m theta)")
    p.add_argument("--amp", type=float, default=0.16, help="drawn amplitude in units of R (exaggerated)")
    p.add_argument("--out", default="shell_flux", help="output name, without extension")
    args = p.parse_args()
    paper_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(17, 8.6),
                                 gridspec_kw=dict(wspace=0.08, left=0.02, right=0.98, top=0.9, bottom=0.04))
    for a in (ax, bx):
        a.set_aspect("equal")
        a.set_xlim(-1.55, 1.75)
        a.set_ylim(-1.45, 1.45)
        a.axis("off")
    panel_normals(ax, 1.0, args.amp, args.mode)
    panel_mean_flux(bx, 1.0, args.mode)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    print(f"saved {args.out}.png and {args.out}.pdf")


if __name__ == "__main__":
    main()
