"""Draws figures/rope_term.png for Part 1, Step 3 ("Where the rope term comes from").

    python notebooks/figures/rope_term.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Ellipse

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titlelocation": "left"})


def arrow(ax, start, vec, color="k", lw=2.0, ms=15):
    ax.annotate("", xy=np.add(start, vec), xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms))


def element(ax, x0, y0, side, top, title, s=0.6, k=0.45):
    """Square; side/top = traction normal part on side/top faces (>0 pull, <0 push). Axis along y."""
    ax.add_patch(plt.Rectangle((x0 - s, y0 - s), 2 * s, 2 * s, fc="0.93", ec="k", lw=1.4))
    for val, normals in ((side, ((1, 0), (-1, 0))), (top, ((0, 1), (0, -1)))):
        if not val:
            continue
        for n in normals:
            n = np.array(n, float); L = k * abs(val); face = np.array([x0, y0]) + n * s
            if val > 0: arrow(ax, face, n * L)
            else: arrow(ax, face + n * L, -n * L)
    ax.text(x0, y0 - s - 1.25, title, ha="center", va="top", fontsize=11.5)


fig = plt.figure(figsize=(16, 10.5))

# (a) cross-section of the spinning tube ------------------------------------------------------------
ax = fig.add_axes([0.02, 0.45, 0.30, 0.50])
R = 1.6
ax.add_patch(plt.Circle((0, 0), R, fc="0.93", ec="k", lw=1.5))
ax.plot(0, 0, "o", color="k", ms=5)
for r in (0.55, 1.05, 1.55):
    for th in np.radians(np.arange(0, 360, 45) + (22.5 if r == 1.05 else 0)):
        p = r*np.array([np.cos(th), np.sin(th)]); t = np.array([-np.sin(th), np.cos(th)])
        arrow(ax, p - 0.18*r*t, 0.36*r*t, color="0.3", lw=1.6, ms=11)
ax.annotate("rim speed $v = \\omega R$", xy=(0, R), xytext=(0.9, 2.15), fontsize=12,
            arrowprops=dict(arrowstyle="-", color="0.4", lw=1))
ax.text(-2.2, -2.15, "speed grows from the axis: $\\omega r$\nthe fluid only circles; nothing moves along the axis", fontsize=11)
ax.text(0.12, 0.12, "axis", fontsize=10, color="0.35")
ax.set_xlim(-2.3, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(a) The tube seen end-on: solid-body spin")

# (b) side view: extra push across, none along -------------------------------------------------------
ax = fig.add_axes([0.36, 0.45, 0.30, 0.50])
r, h, e = 0.75, 3.2, 0.24
ax.add_patch(plt.Rectangle((-r, -h/2), 2*r, h, fc="0.93", ec="none"))
ax.plot([-r, -r], [-h/2, h/2], color="k", lw=1.3); ax.plot([r, r], [-h/2, h/2], color="k", lw=1.3)
ax.add_patch(Ellipse((0, h/2), 2*r, 2*e, fc="0.93", ec="k", lw=1.3))
ax.add_patch(Arc((0, -h/2), 2*r, 2*e, theta1=180, theta2=360, color="k", lw=1.3))
ax.plot([0, 0], [-h/2 - 0.5, h/2 + 0.9], color="k", lw=1, ls=(0, (6, 4)))
arrow(ax, (0, h/2 + 0.9), (0, 0.5), lw=2.4)
ax.text(0.12, h/2 + 1.25, "$\\mathbf{e}$ (axis)", fontsize=12)
for yy in (-0.9, 0.0, 0.9):
    arrow(ax, (r, yy), (0.7, 0), lw=2.2)
    arrow(ax, (-r, yy), (-0.7, 0), lw=2.2)
ax.text(r + 0.85, 0.0, "extra push\n$\\rho v^2/4$\nacross the axis", fontsize=11.5, va="center")
ax.text(-1.9, h/2 + 0.65, "no extra push\nalong the axis", fontsize=11.5, color="0.35", ha="center")
ax.set_xlim(-2.6, 2.8); ax.set_ylim(-2.6, 3.4); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(b) Side view: the spin pushes sideways only")

# (c) numbers: where rho v^2/4 comes from -------------------------------------------------------------
ax = fig.add_axes([0.70, 0.45, 0.29, 0.50])
rr = np.linspace(0, 1, 200)
ax.plot(rr, rr**2, color="k", lw=2.2)
ax.fill_between(rr, 0, rr**2, color="0.88")
ax.axhline(0.5, color="0.4", ls="--", lw=1.4)
ax.text(0.03, 0.53, "average over the disk area = 1/2", fontsize=11, color="0.3")
ax.set_xlabel("$r/R$"); ax.set_ylabel("$(\\omega r)^2 / v^2$")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("(c) Moving fluid carries a push $\\rho u^2$")
ax.text(0.03, 0.9, "$\\langle u_x^2 + u_y^2 \\rangle = v^2/2$\nsplit evenly: $v^2/4$ along $x$, $v^2/4$ along $y$", fontsize=11, va="top")

# (d) the split into gas + rope ------------------------------------------------------------------------
ax = fig.add_axes([0.03, 0.02, 0.94, 0.38])
element(ax, 0, 0, -1.6, -1.0, "the tube's stress (axis along y)\npush $\\bar p + \\rho v^2/4$ across, $\\bar p$ along")
ax.text(2.4, 0, "=", fontsize=30, ha="center", va="center")
element(ax, 4.8, 0, -1.6, -1.6, "gas term\nequal push $p_1 = \\bar p + \\rho v^2/4$ on every face")
ax.text(7.2, 0, "+", fontsize=30, ha="center", va="center")
element(ax, 9.6, 0, 0, 0.6, "rope term\n\"pull\" $\\rho v^2/4$ along the axis only")
ax.text(12.1, 0, "the rope term is not a real pull:\nit is the push that is missing\nalong the axis\n\n"
        "with $\\boldsymbol{\\alpha} = v\\,\\mathbf{e}$:  $\\frac{\\rho}{4} v^2 e_i e_j = \\frac{\\rho}{4}\\,\\alpha_i\\alpha_j$",
        fontsize=12, va="center")
ax.set_xlim(-1.8, 16.4); ax.set_ylim(-3.1, 1.9); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(d) Take out the equal push: what is left is the rope term")

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rope_term.png"), dpi=105)
