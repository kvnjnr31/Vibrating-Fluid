"""Flow through a permeable vibrating shell at the two extremes of the antinodes (one slice).

    python shell_flow_extremes.py                       # mode 4, writes shell_flow_extremes.png and .pdf
    python shell_flow_extremes.py --mode 6 --q0 0.2

The shell sits at r = R + w with w = W0 cos(wt) cos(m theta). At the two extremes, wt = 0 and
wt = pi, the shell is still (dw/dt = 0), so the flow through it is q = q0 + k w along the normal.
Qualitative flow field in the slice: an incompressible, irrotational flow that matches this
radial flow at r = R, built from a central source (steady part q0) and the mode (part k w):

    inside   phi = q0 R ln r + s (k W0 R / m) (r / R)^m  cos(m theta)
    outside  phi = q0 R ln r - s (k W0 R / m) (R / r)^m  cos(m theta),     s = cos(wt) = +1 or -1

so u_r(R) = q0 + s k W0 cos(m theta) on both sides. Streamlines in grey, shell in black,
flow through the shell as arrows along the shell normal: black where fluid leaves, grey where it enters.
"""
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def paper_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlesize": 22,
        "font.size": 20, "legend.fontsize": 15, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
    })


def velocity(x, y, R, m, q0, kW, s):
    """Cartesian velocity of the potential flow above (s = +1 or -1)."""
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    inside = r < R
    rr = np.where(r == 0, 1e-9, r)
    # radial and angular components
    ur_in = q0 * R / rr + s * kW * (rr / R) ** (m - 1) * np.cos(m * th)
    ut_in = -s * kW * (rr / R) ** (m - 1) * np.sin(m * th)
    ur_out = q0 * R / rr + s * kW * (R / rr) ** (m + 1) * np.cos(m * th)
    ut_out = -s * kW * (R / rr) ** (m + 1) * np.sin(m * th)
    ur = np.where(inside, ur_in, ur_out)
    ut = np.where(inside, ut_in, ut_out)
    return ur * np.cos(th) - ut * np.sin(th), ur * np.sin(th) + ut * np.cos(th)


def panel(ax, R, m, A, q0, kW, s, title):
    n = 400
    g = np.linspace(-1.9, 1.9, n)
    X, Y = np.meshgrid(g, g)
    U, V = velocity(X, Y, R, m, q0, kW, s)
    speed = np.hypot(U, V)
    core = np.hypot(X, Y) < 0.12 * R                       # hide the point source at the center
    U, V = np.where(core, np.nan, U), np.where(core, np.nan, V)
    lw = 0.6 + 1.6 * np.clip(speed / np.nanpercentile(speed, 95), 0, 1)
    ax.streamplot(X, Y, U, V, color="0.6", linewidth=lw, density=1.6, arrowsize=1.1, arrowstyle="-|>")
    th = np.linspace(0, 2 * np.pi, 1200)
    rs = R + s * A * np.cos(m * th)
    ax.plot(R * np.cos(th), R * np.sin(th), color="0.5", lw=1.4, ls=(0, (5, 4)))
    ax.plot(rs * np.cos(th), rs * np.sin(th), color="k", lw=2.8)
    # flow through the shell along the normal, at the antinodes
    for t in np.arange(2 * m) * np.pi / m:
        r0 = R + s * A * np.cos(m * t)
        dr = -s * A * m * np.sin(m * t)                    # zero at an antinode: normal along the radius
        er, et = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
        nrm = r0 * er - dr * et
        nrm /= np.linalg.norm(nrm)
        q = q0 + s * kW * np.cos(m * t)
        p = r0 * er
        L = 0.42 * q / (q0 + kW)
        col = "k" if q > 0 else "0.45"
        start = p if q > 0 else p - L * nrm
        ax.arrow(*start, *(L * nrm) if q > 0 else (L * nrm), color=col, lw=3.0, head_width=0.07,
                 head_length=0.09, length_includes_head=True, zorder=5)
    ax.plot(0, 0, "o", color="k", ms=7, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    ax.axis("off")
    ax.set_title(title)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", type=int, default=4)
    p.add_argument("--amp", type=float, default=0.14, help="drawn shell amplitude, in units of R")
    p.add_argument("--q0", type=float, default=0.3, help="steady outflow from the source, relative to k W0")
    p.add_argument("--out", default="shell_flow_extremes")
    a = p.parse_args()
    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(17, 9.2), gridspec_kw=dict(wspace=0.05, left=0.01, right=0.99,
                                                                        top=0.9, bottom=0.07))
    R, kW = 1.0, 1.0
    panel(axes[0], R, a.mode, a.amp, a.q0, kW, +1, r"(A) $\omega t = 0$: Antinodes at Their Peaks")
    panel(axes[1], R, a.mode, a.amp, a.q0, kW, -1, r"(B) $\omega t = \pi$: Antinodes at Their Troughs")
    fig.text(0.5, 0.025, "black arrows: fluid leaves the shell     grey arrows: fluid enters     "
             "grey lines: streamlines     dot: source", ha="center", fontsize=15, color="0.25")
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=200)
    print(f"saved {a.out}.png")


if __name__ == "__main__":
    main()
