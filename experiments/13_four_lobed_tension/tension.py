"""Experiment 13: experiment 12 with a four-lobed tension pattern around every shell.

Each shell has tension T(theta) = T0 (1 + delta cos 4 theta), the same in every cell: highest at 0 and
90 degrees, lowest at 45 and 135 degrees. The shell energy is taken as that of a string bent around the
circle, U = (1/2R) oint T (eta'^2 - eta^2) dtheta, with eta the wall displacement. For the two oval modes
    eta = a cos 2theta   ->   stiffness K (1 - eps)
    eta = b sin 2theta   ->   stiffness K (1 + eps),     eps = 5 delta / 6,
with no cross term. (The pattern also couples the ovals to 6-lobed shapes; those are left out.) The
oval stretched along x or y (a) gets softer, the oval along the diagonals (b) gets stiffer. The fluid
and its coupling are unchanged from experiment 12.

    python experiments/13_four_lobed_tension/tension.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "12_ideal_fluid_cells"))
import ideal as I  # noqa: E402
import run as Rn  # noqa: E402

EPS = 0.2
DELTA = 6 * EPS / 5
W_LIST = (0.88, 0.99, 1.15)      # a only, both, b only
W_MOVIE = (0.88, 1.15)


def bands(bl, Mi, K, kx, ky, eps=EPS):
    """Frequencies and oval axis angle (deg) of the two bands; [..., 0] lower, [..., 1] upper."""
    keys = np.array(list(bl.keys()))
    B = np.array(list(bl.values()))
    kx, ky = np.atleast_1d(kx), np.atleast_1d(ky)
    ph = np.exp(1j * (np.multiply.outer(kx, keys[:, 0]) + np.multiply.outer(ky, keys[:, 1])))
    Mk = np.einsum("...j,jab->...ab", ph, B).real + Mi * np.eye(2)
    Kd = np.diag([K * (1 - eps), K * (1 + eps)])
    w = np.empty(kx.shape + (2,))
    ang = np.empty_like(w)
    for idx in np.ndindex(kx.shape):
        ev, v = eigh(Kd, Mk[idx])
        w[idx] = np.sqrt(ev)
        ang[idx] = np.degrees(0.5 * np.arctan2(v[1], v[0]))
    return w, ang


def band_axes(ax, bl, Mi, K, mark=None, legend_small=False):
    k, s, ticks = Rn.path_k()
    w0, _ = Rn.bands(bl, Mi, K, k[:, 0], k[:, 1])
    w, ang = bands(bl, Mi, K, k[:, 0], k[:, 1])
    ax.plot(s, w0[:, 0], color="0.75", lw=1.2, label="uniform tension (exp. 12)")
    ax.plot(s, w0[:, 1], color="0.75", lw=1.2)
    ax.plot(s, w[:, 0], color="k", lw=2.2, label="mostly oval a (axis 0$^\\circ$)")
    ax.plot(s, w[:, 1], color="k", lw=2.2, ls="--", label="mostly oval b (axis 45$^\\circ$)")
    if mark is not None:
        for m in np.atleast_1d(mark):
            ax.axhline(m, color="k", lw=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$\\Gamma$\nlong waves", "M\nshortest\nup-down", "K\nshortest\nleft-right", "$\\Gamma$"],
                       fontsize=9 if legend_small else 11)
    ax.set_xlim(0, ticks[-1])
    ax.set_ylim(0.6, 1.75)
    ax.set_ylabel("frequency")
    return w, ang


def build(bl, Mi, K):
    z, ring, M, D = Rn.big_lattice(bl, Mi, K)
    N = len(z)
    Kv = np.tile([K * (1 - EPS), K * (1 + EPS)], N)
    return z, ring, M, D, Kv


def steady(z, M, D, Kv, w):
    N = len(z)
    f = np.zeros(2 * N)
    f[0] = f[1] = 1 / np.sqrt(2)              # drive the centre oval with its axis at 22.5 degrees
    q = Rn.response(M, D, Kv, w, f)
    return q[0::2], q[1::2]


def checks_figure(path, bl, Mi, K, maps, res):
    plt = Rn.style()
    fig = plt.figure(figsize=(17, 13))
    ax0 = fig.add_axes([0.06, 0.58, 0.40, 0.37])
    band_axes(ax0, bl, Mi, K, mark=W_LIST)
    for w in W_LIST:
        ax0.text(0.02, w + 0.01, f"{w}", transform=ax0.get_yaxis_transform(), fontsize=10)
    ax0.legend(loc="upper right", fontsize=10, frameon=False)
    ax0.set_title("(A) Bands: tension 1 + %.2f cos 4$\\theta$ vs uniform" % DELTA)
    ax1 = fig.add_axes([0.55, 0.58, 0.40, 0.37])
    g = np.linspace(-7.26 / 2 * 1.15, 7.26 / 2 * 1.15, 121)
    KX, KY = np.meshgrid(g, g)
    W, ANG = bands(bl, Mi, K, KX, KY)
    cs = ax1.contour(KX, KY, W[..., 1], levels=[1.05, 1.15, 1.3, 1.45], colors="k")
    ax1.clabel(cs, fontsize=9)
    cs2 = ax1.contour(KX, KY, W[..., 0], levels=[0.8, 0.88, 0.95], colors="0.45", linestyles="--")
    ax1.clabel(cs2, fontsize=9)
    th = np.linspace(0, 2 * np.pi, 7)
    ax1.plot(4 * np.pi / 3 * np.cos(th), 4 * np.pi / 3 * np.sin(th), color="0.6", lw=1)
    ax1.set_aspect("equal")
    ax1.set_xlabel("$k_x$")
    ax1.set_ylabel("$k_y$")
    ax1.set_title("(B) Lines of equal frequency (solid upper, dashed lower)")
    z, ring = maps["z"], maps["ring"]
    m = ring < Rn.SPONGE
    for c, w in enumerate(W_LIST):
        a, b = maps[w]
        top = max(np.abs(a[m]).max(), np.abs(b[m]).max())
        for r_, (comp, name) in enumerate(((a, "a (0$^\\circ$)"), (b, "b (45$^\\circ$)"))):
            ax = fig.add_axes([0.04 + c * 0.32, 0.29 - r_ * 0.26, 0.28, 0.24])
            ax.scatter(z.real[m], z.imag[m], c=(np.abs(comp[m]) / top) ** 0.5, cmap="Greys", s=5, vmin=0,
                       vmax=0.8, lw=0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{'(C) ' if c == 0 and r_ == 0 else ''}w = {w}: oval {name}", fontsize=13)
    fig.text(0.04, 0.015, "(C) Amplitude from one cell driven with its oval at 22.5$^\\circ$ (equal a and b); "
             "darker = larger, same scale for a and b in each column.", fontsize=12)
    fig.savefig(path, dpi=105)


def far_ratio(z, ring, a, b):
    sel = (np.abs(z) > 6) & (ring < Rn.SPONGE)
    near = (np.abs(z) > 0.9) & (np.abs(z) < 1.1)
    return {"a_far_over_near": float(np.abs(a[sel]).mean() / np.abs(a[near]).mean()),
            "b_far_over_near": float(np.abs(b[sel]).mean() / np.abs(b[near]).mean()),
            "far_b_over_a": float(np.abs(b[sel]).mean() / np.abs(a[sel]).mean())}


def movies(bl, Mi, K, z, ring, M, D, Kv, ws, outname):
    N = len(z)
    Rn.band_axes = lambda ax, bl_, Mi_, K_, mark=None: band_axes(ax, bl_, Mi_, K_, mark, legend_small=True)
    parts = []
    for w in ws:
        Rn.W_DRIVE = w
        Tp = 2 * np.pi / w
        path = os.path.join(HERE, f"run_w{int(round(w * 100))}.npz")
        if not os.path.exists(path):
            def drive(t, w=w, Tp=Tp):
                f = np.zeros(2 * N)
                f[0] = f[1] = 0.05 / np.sqrt(2) * np.sin(w * t) * min(1.0, t / (3 * Tp))
                return f
            tt, qq, vv, aa = Rn.newmark(M, D, Kv, drive, 40 * Tp, Tp / 32, 32 // Rn.PER)
            np.savez_compressed(path, t=tt, q=qq, v=vv, a=aa)
        run = dict(np.load(path))
        which = "mostly oval a (0$^\\circ$) travels" if w < 0.95 else "mostly oval b (45$^\\circ$) travels"
        which += ", crests in" if w < 1.25 else ", crests out"
        out = os.path.join(HERE, f"part_w{int(round(w * 100))}.mp4")
        Rn.movie(out, run, bl, Mi, K, split=True,
                 title=f"Four-lobed tension, drive {w}: {which}",
                 note="Every shell: tension highest at 0 and 90 degrees, lowest at 45. Centre oval driven at "
                      "22.5 degrees (equal a and b). Ideal fluid, no loss; outer rings hidden: sponge.")
        parts.append(out)
    lst = os.path.join(HERE, "parts.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    os.system(f"ffmpeg -loglevel error -y -f concat -safe 0 -i {lst} -c copy {os.path.join(HERE, outname)}")
    for p in parts + [lst]:
        os.remove(p)


def main(movie_only=False):
    bl, Mi, K = Rn.setup()
    z, ring, M, D, Kv = build(bl, Mi, K)
    N = len(z)
    if not movie_only:
        t0 = time.perf_counter()
        res = {"delta": DELTA, "eps": EPS, "K_a": K * (1 - EPS), "K_b": K * (1 + EPS)}
        for name, kk in (("Gamma", (0.0, 0.0)), ("M", (0.0, 2 * np.pi / np.sqrt(3))),
                         ("K", (2 * np.pi / 3, 2 * np.pi / np.sqrt(3)))):
            w, ang = bands(bl, Mi, K, *kk)
            res["band_" + name] = {"w": w[0].tolist(), "oval_axis_deg": ang[0].round(1).tolist()}
        g = np.linspace(-7.26, 7.26, 161)
        KX, KY = np.meshgrid(g, g)
        W, _ = bands(bl, Mi, K, KX, KY)
        res["lower_band"] = [float(W[..., 0].min()), float(W[..., 0].max())]
        res["upper_band"] = [float(W[..., 1].min()), float(W[..., 1].max())]
        res["uniform_K_gap_closed_at"] = 0.912
        maps = {"z": z, "ring": ring}
        for w in W_LIST:
            a, b = steady(z, M, D, Kv, w)
            maps[w] = (a, b)
            res[f"steady_w{w}"] = far_ratio(z, ring, a, b)
        res["seconds"] = round(time.perf_counter() - t0, 1)
        json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=2)
        print(json.dumps(res, indent=2), flush=True)
        checks_figure(os.path.join(HERE, "four_lobed_checks.png"), bl, Mi, K, maps, res)
    movies(bl, Mi, K, z, ring, M, D, Kv, W_MOVIE, "four_lobed.mp4")
    print("done", flush=True)


if __name__ == "__main__":
    if "--w" in sys.argv:                      # e.g. --w 1.5 : one extra movie at that drive
        w = float(sys.argv[sys.argv.index("--w") + 1])
        bl, Mi, K = Rn.setup()
        z, ring, M, D, Kv = build(bl, Mi, K)
        movies(bl, Mi, K, z, ring, M, D, Kv, (w,), f"four_lobed_w{int(round(w * 100))}.mp4")
    else:
        main("--movie" in sys.argv)
