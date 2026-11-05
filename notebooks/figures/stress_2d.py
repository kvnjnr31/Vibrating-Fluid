"""Draws figures/stress_2d.png for Part 1, Step 1 of the Maxwell notebooks.

    python notebooks/figures/stress_2d.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titlelocation": "left"})

def arrow(ax, start, vec, ls="-", lw=2.0, color="k"):
    ax.annotate("", xy=np.add(start, vec), xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=16))

def draw_element(ax, S, title, labels=False, scale=0.8, field=False):
    S = np.asarray(S, float); big = np.abs(S).max()
    if field:                                                    # faint field lines along y
        for xl in np.linspace(-1.6, 1.6, 9): ax.plot([xl, xl], [-2, 2], color="0.88", lw=1, zorder=0)
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fc="0.92", ec="k", lw=1.5, zorder=1))
    names = {(1, 0): (r"$\sigma_{xx}$", r"$\tau_{yx}$"), (0, 1): (r"$\sigma_{yy}$", r"$\tau_{xy}$")}
    for nvec in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nvec = np.array(nvec, float); tang = np.array([-nvec[1], nvec[0]])
        t = S @ nvec; tn, ts = t @ nvec, t @ tang
        face = nvec*1.0
        if abs(tn) > 1e-9:
            L = scale*abs(tn)/big
            if tn > 0: arrow(ax, face, nvec*L)                   # tension: away from the face
            else: arrow(ax, face + nvec*L, -nvec*L)              # pressure: into the face
        if abs(ts) > 1e-9:                                       # shear: along the face, just outside it, off to one side
            L = 0.7; d = np.sign(ts)*tang                      # drawn at a fixed length for clarity
            arrow(ax, face + nvec*0.12 + tang*0.55 - d*L/2, d*L, color="0.45")
        key = tuple(nvec.astype(int))
        if labels and key in names:
            ax.text(*(face + nvec*1.15), names[key][0], fontsize=13, ha="center", va="center")
            ax.text(*(face + nvec*0.38 + tang*0.55), names[key][1], fontsize=13, ha="center", va="center", color="0.35")
    ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.3, 2.3); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)

fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
draw_element(axs[0], [[1.0, 0.45], [0.45, 0.7]], "(a) General 2D stress", labels=True)
draw_element(axs[1], [[0, 0], [0, 1.0]], "(b) Rod pulled along y")
axs[1].text(0, -2.15, r"$\sigma = [[0,\ 0],\ [0,\ F/A]]$", ha="center", fontsize=12)
draw_element(axs[2], [[-1.0, 0], [0, 1.0]], "(c) Maxwell: field along y", field=True)
axs[2].text(0, -2.15, r"$\frac{\mu}{8\pi}|\alpha|^2\,[[-1,\ 0],\ [0,\ +1]]$", ha="center", fontsize=12)

ax = axs[3]                                                     # (d) slanted cut through the rod
S = np.array([[0, 0], [0, 1.0]]); nvec = np.array([-1, 1])/np.sqrt(2)   # outward normal of the cut
ax.add_patch(plt.Polygon([[-1, -1], [1, -1], [1, 1]], fc="0.92", ec="k", lw=1.5))
ax.plot([-1, 1], [-1, 1], color="k", lw=2.5)
arrow(ax, (0, -1), (0, -0.8))                                   # the pull on the bottom face
t = S @ nvec; tn = (t @ nvec)*nvec; ts = t - tn
p = np.array([0.0, 0.0])
arrow(ax, p, 1.3*t, lw=2.6)
arrow(ax, p, 1.3*tn, ls="--", color="0.35"); arrow(ax, p, 1.3*ts, ls=":", color="0.35")
ax.text(*(1.3*t + [0.08, 0.1]), r"$\mathbf{t} = \sigma\mathbf{n}$", fontsize=13)
ax.text(*(1.3*tn + [-0.75, 0.05]), "normal", fontsize=11, color="0.35")
ax.text(*(1.3*ts + [0.05, -0.1]), "shear", fontsize=11, color="0.35")
arrow(ax, (-0.55, -0.55), 0.55*nvec, color="0.6", lw=1.5); ax.text(-1.15, -0.2, r"$\mathbf{n}$", fontsize=13, color="0.45")
ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.3, 2.3); ax.set_aspect("equal"); ax.axis("off"); ax.set_title("(d) Rod cut at 45 degrees")
plt.tight_layout()
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stress_2d.png"), dpi=110)

