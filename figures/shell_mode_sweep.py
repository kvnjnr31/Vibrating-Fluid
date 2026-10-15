"""Flow through a permeable vibrating shell as the mode number m grows (one slice, Y = cos(m theta)).

    python shell_mode_sweep.py                          # modes 2 4 8 16, writes shell_mode_sweep.png and .pdf
    python shell_mode_sweep.py --modes 3 6 12 24 --slope 0.5

Rows:  (top)    wt = 0,  antinodes at their peaks
       (middle) wt = pi, antinodes at their troughs
       (bottom) cycle-averaged sideways flux <J_s> ~ -grad_s(Y^2), for k > 0
Columns: one mode each.

The drawn amplitude is A = slope / m, so the steepest tilt of the shell (A m) is the same in every
column; with a fixed A the shell would turn into spikes at large m. The flow field is the potential
flow of shell_flow_extremes.py: a central source q0 plus the mode term, which decays as (r/R)^(m-1)
inside and (R/r)^(m+1) outside, so the mode's flow stays within a layer about R/m thick.

Needs shell_flow_extremes.py in the same folder (for paper_style and velocity).
"""
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shell_flow_extremes import paper_style, velocity  # noqa: E402

LIM = 1.75


def extreme(ax, R, m, A, q0, kW, s):
    g = np.linspace(-LIM, LIM, 420)
    X, Y = np.meshgrid(g, g)
    U, V = velocity(X, Y, R, m, q0, kW, s)
    speed = np.hypot(U, V)
    core = np.hypot(X, Y) < 0.12 * R
    U, V = np.where(core, np.nan, U), np.where(core, np.nan, V)
    lw = 0.5 + 1.4 * np.clip(speed / np.nanpercentile(speed, 95), 0, 1)
    ax.streamplot(X, Y, U, V, color="0.62", linewidth=lw, density=1.5, arrowsize=0.9, arrowstyle="-|>")
    th = np.linspace(0, 2 * np.pi, 3000)
    rs = R + s * A * np.cos(m * th)
    ax.plot(R * np.cos(th), R * np.sin(th), color="0.5", lw=1.2, ls=(0, (5, 4)))
    ax.plot(rs * np.cos(th), rs * np.sin(th), color="k", lw=2.2)
    L0 = 0.38 * min(1.0, 6.0 / m)                       # shorter arrows when the antinodes crowd
    for t in np.arange(2 * m) * np.pi / m:              # antinodes: shell flat, flow along the radius
        r0 = R + s * A * np.cos(m * t)
        e = np.array([np.cos(t), np.sin(t)])
        q = q0 + s * kW * np.cos(m * t)
        L = L0 * abs(q) / (q0 + kW)
        p = r0 * e
        if q > 0:
            ax.arrow(*p, *(L * e), color="k", lw=2.2, head_width=0.06, head_length=0.075,
                     length_includes_head=True, zorder=5)
        else:
            ax.arrow(*(p + L * e), *(-L * e), color="0.45", lw=2.2, head_width=0.06, head_length=0.075,
                     length_includes_head=True, zorder=5)
    ax.plot(0, 0, "o", color="k", ms=6, zorder=6)


def mean_flux(ax, R, m, band=0.12, ring=1.2):
    th = np.linspace(0, 2 * np.pi, 3000)
    Y2 = np.cos(m * th) ** 2
    ax.plot(R * np.cos(th), R * np.sin(th), color="0.55", lw=1.2, ls=(0, (5, 4)))
    outer = R + band * Y2
    ax.fill(np.r_[outer * np.cos(th), R * np.cos(th[::-1])], np.r_[outer * np.sin(th), R * np.sin(th[::-1])],
            color="0.85", lw=0)
    half = np.pi / (2 * m)
    gap = 0.18 * half
    hs = min(1.0, 6.0 / m)
    for t in np.arange(2 * m) * np.pi / m:             # from each antinode toward its two nodal lines
        for sg in (1, -1):
            a = np.linspace(t + sg * gap, t + sg * (half - gap), 30)
            ax.plot(ring * np.cos(a), ring * np.sin(a), color="k", lw=1.8)
            e = np.array([np.cos(a[-1]), np.sin(a[-1])])
            tg = sg * np.array([-np.sin(a[-1]), np.cos(a[-1])])
            ax.arrow(*(ring * e - 0.02 * tg), *(0.02 * tg), color="k", lw=1.8, head_width=0.07 * hs,
                     head_length=0.09 * hs, length_includes_head=True)
    nodes = half + np.arange(2 * m) * np.pi / m
    anti = np.arange(2 * m) * np.pi / m
    ms = 9 * min(1.0, 8.0 / m) + 3
    ax.plot(R * np.cos(nodes), R * np.sin(nodes), "o", color="k", ms=ms, ls="none")
    ax.plot(R * np.cos(anti), R * np.sin(anti), "o", mfc="white", mec="k", ms=ms, mew=1.3, ls="none")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--modes", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--slope", type=float, default=0.56, help="steepest shell tilt A*m (drawn)")
    p.add_argument("--q0", type=float, default=0.3, help="steady outflow from the source, relative to k W0")
    p.add_argument("--out", default="shell_mode_sweep")
    a = p.parse_args()
    paper_style()
    n = len(a.modes)
    fig, axes = plt.subplots(3, n, figsize=(5.2 * n, 16.6),
                             gridspec_kw=dict(wspace=0.04, hspace=0.12, left=0.075, right=0.995,
                                              top=0.955, bottom=0.05))
    axes = np.atleast_2d(axes).reshape(3, n)
    R, kW = 1.0, 1.0
    for j, m in enumerate(a.modes):
        A = a.slope / m
        extreme(axes[0, j], R, m, A, a.q0, kW, +1)
        extreme(axes[1, j], R, m, A, a.q0, kW, -1)
        mean_flux(axes[2, j], R, m)
        axes[0, j].set_title(f"$m = {m}$", loc="center", fontsize=24)
        for i in range(3):
            ax = axes[i, j]
            ax.set_aspect("equal")
            ax.set_xlim(-LIM, LIM)
            ax.set_ylim(-LIM, LIM)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    labels = [r"(A) $\omega t = 0$" + "\npeaks", r"(B) $\omega t = \pi$" + "\ntroughs",
              r"(C) mean" + "\n" + r"$\langle J_s\rangle$"]
    for i, lab in enumerate(labels):
        axes[i, 0].set_ylabel(lab, fontsize=20, rotation=0, ha="right", va="center", labelpad=8)
    fig.text(0.5, 0.012, "black arrows: fluid leaves     grey arrows: fluid enters     grey lines: streamlines     "
             "(C) black dots: nodal lines, where the mean flux collects     open dots: antinodes",
             ha="center", fontsize=15, color="0.25")
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=170)
    print(f"saved {a.out}.png")


if __name__ == "__main__":
    main()
