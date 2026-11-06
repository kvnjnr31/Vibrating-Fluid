"""Draws figures/maxwell_stress.png for Part 1, Step 3 of the Maxwell notebooks.

    python notebooks/figures/maxwell_stress.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Ellipse

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titlelocation": "left"})


def arrow(ax, start, vec, color="k", lw=2.0, ls="-", ms=15, style="-|>"):
    ax.annotate("", xy=np.add(start, vec), xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls, mutation_scale=ms))


def tube(ax, x0, y0, r=0.42, h=3.2, spin_label=False, fc="0.93"):
    """A vortex tube along y, drawn in light perspective, spinning about its axis."""
    e = 0.32 * r                                      # half-height of the end ellipses
    ax.add_patch(plt.Rectangle((x0 - r, y0 - h / 2), 2 * r, h, fc=fc, ec="none", zorder=1))
    ax.plot([x0 - r, x0 - r], [y0 - h / 2, y0 + h / 2], color="k", lw=1.3, zorder=2)
    ax.plot([x0 + r, x0 + r], [y0 - h / 2, y0 + h / 2], color="k", lw=1.3, zorder=2)
    ax.add_patch(Ellipse((x0, y0 + h / 2), 2 * r, 2 * e, fc=fc, ec="k", lw=1.3, zorder=3))
    ax.add_patch(Arc((x0, y0 - h / 2), 2 * r, 2 * e, theta1=180, theta2=360, color="k", lw=1.3, zorder=3))
    ax.add_patch(Arc((x0, y0 - h / 2), 2 * r, 2 * e, theta1=0, theta2=180, color="k", lw=1.0, ls=(0, (3, 3)), zorder=3))
    # spin: arcs around the middle, front side solid, with an arrow head
    for yy in (y0 - h * 0.22, y0 + h * 0.22):
        ax.add_patch(Arc((x0, yy), 2.7 * r, 4.0 * e, theta1=195, theta2=335, color="0.3", lw=2.0, zorder=4))
        a = np.radians(335)
        p = np.array([x0 + 1.35 * r * np.cos(a), yy + 2.0 * e * np.sin(a)])
        d = np.array([1.35 * r * np.sin(a), -2.0 * e * np.cos(a)]) * -1          # tangent, direction of travel
        d = d / np.hypot(*d) * 0.12 * (r / 0.42)
        arrow(ax, p - d * 0.2, d, color="0.3", lw=2.0, ms=16)
    if spin_label:
        return (x0 + 1.25 * r, y0 + h * 0.22)


def element(ax, x0, y0, top, side, title, s=0.55, k=0.55):
    """Square element; top = pull on top/bottom faces (>0 tension), side = pull on side faces."""
    ax.add_patch(plt.Rectangle((x0 - s, y0 - s), 2 * s, 2 * s, fc="0.93", ec="k", lw=1.4))
    for n in ((0, 1), (0, -1)):
        if top:
            L = k * abs(top)
            face = np.array([x0, y0]) + np.array(n) * s
            if top > 0: arrow(ax, face, np.array(n) * L)
            else: arrow(ax, face + np.array(n) * L, -np.array(n) * L)
    for n in ((1, 0), (-1, 0)):
        if side:
            L = k * abs(side)
            face = np.array([x0, y0]) + np.array(n) * s
            if side > 0: arrow(ax, face, np.array(n) * L)
            else: arrow(ax, face + np.array(n) * L, -np.array(n) * L)
    ax.text(x0, y0 - s - 1.45, title, ha="center", va="top", fontsize=11.5)


fig = plt.figure(figsize=(16, 10.5))

# (a) one vortex -------------------------------------------------------------------------------------
ax = fig.add_axes([0.02, 0.43, 0.30, 0.52])
tube(ax, 0, 0, r=0.75, h=3.6)
ax.plot([0, 0], [-2.4, 2.4], color="k", lw=1, ls=(0, (6, 4)))
arrow(ax, (0, 2.45), (0, 0.9), lw=2.6)
ax.text(0.15, 3.15, r"$\boldsymbol{\alpha}$", fontsize=18)
ax.text(1.0, 2.75, "axis of the vortex\n= direction of the\nline of force", fontsize=11, va="center")
ax.annotate("rim speed $= |\\boldsymbol{\\alpha}|$", xy=(0.9, -0.85), xytext=(1.35, -1.6), fontsize=12,
            arrowprops=dict(arrowstyle="-", color="0.4", lw=1))
ax.annotate("spin", xy=(0.9, 0.7), xytext=(1.35, 0.35), fontsize=11, color="0.35",
            arrowprops=dict(arrowstyle="-", color="0.4", lw=1))
ax.set_xlim(-2.2, 3.3); ax.set_ylim(-3.0, 3.6); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(a) One vortex: what $\\boldsymbol{\\alpha}$ means")

# (b) a bundle of vortices along the field lines -------------------------------------------------------
ax = fig.add_axes([0.35, 0.43, 0.63, 0.52])
xs = np.arange(-3, 4) * 1.35
for xl in xs:
    tube(ax, xl, 0, r=0.36, h=3.4)
for xa, xb in zip(xs[:-1], xs[1:]):
    xm = 0.5 * (xa + xb)
    arrow(ax, (xm, 0), (-0.27, 0), color="k", lw=1.8, ms=12)
    arrow(ax, (xm, 0), (0.27, 0), color="k", lw=1.8, ms=12)
for xl in xs[::2]:
    arrow(ax, (xl, 2.85), (0, -0.6), color="0.45", lw=1.8, ms=12)
    arrow(ax, (xl, -2.85), (0, 0.6), color="0.45", lw=1.8, ms=12)
ax.text(xs[-1] + 0.75, 0, "spinning tubes push\nsideways on their\nneighbors: pressure\nacross the lines", fontsize=11.5, va="center")
ax.text(xs[-1] + 0.75, 2.35, "along their axes the tubes\ndo not push; the lines try\nto shorten, like stretched\nrubber bands (grey arrows):\ntension along the lines", fontsize=11.5, va="center")
ax.set_xlim(xs[0] - 0.8, xs[-1] + 4.6); ax.set_ylim(-3.2, 3.6); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(b) Many vortices, axes along the field (field along y)")

# (c) the two terms of the stress ---------------------------------------------------------------------
ax = fig.add_axes([0.03, 0.02, 0.94, 0.38])
element(ax, 0, 0, 2, 0, "rope term  $\\frac{\\mu}{4\\pi}\\alpha_i\\alpha_j$\npull $2P$ along $\\boldsymbol{\\alpha}$ only")
ax.text(2.35, 0, "+", fontsize=30, ha="center", va="center")
element(ax, 4.7, 0, -1, -1, "gas term  $-p_1\\delta_{ij}$, with $p_1 = P$\npush $P$ on every face")
ax.text(7.05, 0, "=", fontsize=30, ha="center", va="center")
element(ax, 9.4, 0, 1, -1, "Maxwell's stress\npull $P$ along $\\boldsymbol{\\alpha}$, push $P$ across")
for xl in np.linspace(8.65, 10.15, 5):
    ax.plot([xl, xl], [-1.75, 1.75], color="0.85", lw=1, zorder=0)
ax.text(12.0, 0, "$P = \\dfrac{\\mu}{8\\pi}|\\boldsymbol{\\alpha}|^2$\n\narrows away from a face: pull\narrows into a face: push",
        fontsize=12, va="center")
ax.set_xlim(-1.8, 15.6); ax.set_ylim(-3.2, 2.0); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(c) The two terms of the stress, on a small square (field along y)")

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "maxwell_stress.png"), dpi=105)
