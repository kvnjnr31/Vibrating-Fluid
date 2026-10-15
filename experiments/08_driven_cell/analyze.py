"""Numbers and figure for Experiment 6 from the saved runs.

    python experiments/08_driven_cell/analyze.py

Linear lattice model. With the neighbors held still, the fluid force on the oval modes of cell i
(a and b) is Q_i = G(i - 0) x_0 when cell 0 moves with complex amplitude x_0 (x(t) = Re x e^{iwt}).
G(0) = madd w^2 - i w c gives the added mass and damping; G(d), d != 0, is the coupling to a cell
at lattice offset d. For free cells with M x'' + K x = Q and K = (M + madd) w^2 (resonant at w):

    i w c x_i = sum over j != i of G(i - j) x_j

(the 2 x 2 version for the a and b modes). Solving this with the center cell given predicts every
cell's steady response; the free run checks it.

A lattice pattern x_i = exp(i k . r_i) feels an extra fluid force sum_{d != 0} G(d) exp(i k . d), so
its effective damping is c_eff(k) = c - Im[sum_{d != 0} G(d) exp(i k . d)] / w. The aligned pattern
is k = (0, 0) and the geared (checkerboard) pattern is k = (pi, pi).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    d = dict(np.load(os.path.join(HERE, name)))
    return {k: (v.item() if v.ndim == 0 else v) for k, v in d.items()}


def offsets(d):
    """Lattice offset of every cell from the driven cell, folded into the periodic box."""
    n = d["n"]
    off = d["ij"] - d["ij"][d["k0"]]
    return (off + n // 2) % n - n // 2


def coupling(ia, ib):
    """G[k] is the 2x2 matrix (rows: force on a, b; cols: drive a, b) for offset off[k]."""
    G = np.zeros((ia["Ga"].size, 2, 2), complex)
    G[:, 0, 0], G[:, 1, 0] = ia["Ga"], ia["Gb"]
    G[:, 0, 1], G[:, 1, 1] = ib["Ga"], ib["Gb"]
    return G


def predict(G, off, n, k0, X0):
    """Steady response of free cells to the center cell driven in mode a with amplitude X0."""
    nc = n * n
    # spring and cell mass cancel the a-mode added mass, leaving the rest of the self force
    Sres = G[k0] - G[k0, 0, 0].real * np.eye(2)
    lut = {tuple(o): k for k, o in enumerate(off)}
    ij = np.array([(i, j) for i in range(n) for j in range(n)])
    Amat = np.zeros((2 * nc, 2 * nc), complex)
    for i in range(nc):
        for j in range(nc):
            dd = tuple((ij[i] - ij[j] + n // 2) % n - n // 2)
            blk = Sres if i == j else G[lut[dd]]
            Amat[2 * i:2 * i + 2, 2 * j:2 * j + 2] = blk
    # equations for free cells: sum_j blk_ij x_j = 0 (forces balance: spring+mass cancel madd,
    # leaving the residual self force plus the neighbor forces)
    free = np.ones(2 * nc, bool)
    free[2 * k0:2 * k0 + 2] = False
    xk = np.array([X0, 0.0])
    rhs = -Amat[np.ix_(free, ~free)] @ xk
    x = np.zeros(2 * nc, complex)
    x[~free] = xk
    x[free] = np.linalg.solve(Amat[np.ix_(free, free)], rhs)
    return x[0::2], x[1::2]


def decay_a(ia, nbig=15):
    """a-mode only: free cells on an nbig x nbig lattice, coupling G(d) from ia for |d| <= 2.
    Returns |a|/A and phase lag (units of pi) along +x, from the linear model."""
    off = offsets(ia)
    k0, w, A = ia["k0"], ia["w"], ia["A"]
    G0 = ia["Ga"][k0]
    lut = {tuple(o): ia["Ga"][k] for k, o in enumerate(off) if 0 < max(abs(o[0]), abs(o[1])) <= 2}
    ij = np.array([(i, j) for i in range(nbig) for j in range(nbig)])
    c0 = (nbig // 2) * nbig + nbig // 2
    nc = nbig * nbig
    M = np.zeros((nc, nc), complex)
    for i in range(nc):
        M[i, i] = 1j * G0.imag
        for j in range(nc):
            dd = tuple(ij[i] - ij[j])
            if dd in lut:
                M[i, j] = lut[dd]
    free = np.arange(nc) != c0
    x = np.zeros(nc, complex)
    x[c0] = -1j * A
    x[free] = np.linalg.solve(M[np.ix_(free, free)], -M[free, c0] * x[c0])
    along = [c0 + dx * nbig for dx in range(0, nbig // 2 + 1)]
    amp = np.abs(x[along]) / A
    lag = -np.unwrap(np.angle(x[along] / x[c0])) / np.pi
    return amp, lag


def demod(t, x, w, t0, t1):
    s = (t >= t0) & (t < t1)
    return 2 * np.mean(x[s] * np.exp(-1j * w * t[s])[:, None], axis=0)


def envelope(t, x, w, T):
    """Complex amplitude over a sliding window of one period, sampled once per period."""
    edges = np.arange(t[0], t[-1] - T + 1e-9, T / 4)
    return edges + T / 2, np.array([demod(t, x, w, e, e + T) for e in edges])


def main():
    ia, ib = load("impedance_a.npz"), load("impedance_b.npz")
    n, k0, w, T, A = ia["n"], ia["k0"], ia["w"], ia["period"], ia["A"]
    off = offsets(ia)
    G = coupling(ia, ib)
    madd = G[k0, 0, 0].real / w ** 2
    c = -G[k0, 0, 0].imag / w
    out = dict(added_mass=madd, damping=c, added_mass_b=G[k0, 1, 1].real / w ** 2,
               damping_b=-G[k0, 1, 1].imag / w, period=T, w=w)
    # held-neighbor forces, scaled by the self damping force w c A (force a still neighbor feels)
    for name, o in (("x1", (1, 0)), ("x2", (2, 0)), ("diag1", (1, 1)), ("y1", (0, 1))):
        k = int(np.where((off == o).all(1))[0][0])
        g = G[k, 0, 0]
        out[f"G_{name}_over_wc"] = [round(float(abs(g) / (w * c)), 4), round(float(np.angle(g / (1j * w * c)) / np.pi), 3)]
    # lattice patterns: effective damping c_eff(k)
    def ceff(kv, mode):
        s = 0j
        for kk, o in enumerate(off):
            if kk == k0:
                continue
            s += G[kk, mode, mode] * np.exp(1j * (kv[0] * o[0] + kv[1] * o[1]))
        return (c if mode == 0 else out["damping_b"]) - s.imag / w, s.real / w ** 2
    for name, kv in (("aligned", (0, 0)), ("geared", (np.pi, np.pi)), ("stripes_x", (np.pi, 0))):
        ce, dm = ceff(kv, 0)
        out[f"c_eff_{name}"] = round(float(ce), 3)
        out[f"extra_added_mass_{name}"] = round(float(dm), 1)
    # steady prediction
    Xd = -1j * A
    pa, pb = predict(G, off, n, k0, Xd)
    res = dict(out=out, off=off, pa=pa, pb=pb)
    path = os.path.join(HERE, "free.npz")
    if os.path.exists(path):
        fr = load("free.npz")
        t, a, b = fr["t"], fr["a"], fr["b"]
        P = int(round(t[-1] / T))
        Xa = demod(t, a, w, (P - 4) * T, P * T)
        Xb = demod(t, b, w, (P - 4) * T, P * T)
        te, Ea = envelope(t, a, w, T)
        res.update(fr=fr, Xa=Xa, Xb=Xb, te=te, Ea=Ea, P=P)
        # arrival: first time |a| reaches half its final amplitude, along +x
        arr = []
        for dx in range(1, n // 2 + 1):
            k = int(np.where((off == (dx, 0)).all(1))[0][0])
            fin = abs(Xa[k])
            hit = np.where(np.abs(Ea[:, k]) >= 0.5 * fin)[0]
            arr.append((dx, float(te[hit[0]] / T) if hit.size else np.nan, float(fin / A),
                        float(np.angle(Xa[k] / Xa[k0]) / np.pi), float(abs(pa[k]) / A),
                        float(np.angle(pa[k] / pa[k0]) / np.pi)))
        res["arrival"] = arr
        out["along_x"] = [dict(cells=r[0], arrival_periods=round(r[1], 2), amp_free=round(r[2], 4),
                               phase_free_pi=round(r[3], 3), amp_theory=round(r[4], 4),
                               phase_theory_pi=round(r[5], 3)) for r in arr]
        for o, name in (((1, 1), "diag1"), ((2, 1), "knight")):
            kk = int(np.where((off == o).all(1))[0][0])
            out[f"{name}_free"] = dict(amp_a=round(float(abs(Xa[kk]) / A), 4),
                                       phase_a_pi=round(float(np.angle(Xa[kk] / Xa[k0]) / np.pi), 3),
                                       amp_b=round(float(abs(Xb[kk]) / A), 4),
                                       amp_theory_a=round(float(abs(pa[kk]) / A), 4),
                                       phase_theory_a_pi=round(float(np.angle(pa[kk] / pa[k0]) / np.pi), 3),
                                       amp_theory_b=round(float(abs(pb[kk]) / A), 4))
        out["M"], out["K"] = float(fr["M"]), float(fr["K"])
        out["Q_lone"] = round(float((fr["M"] + madd) * w / c), 2)
        out["ringup_periods"] = round(float((fr["M"] + madd) / c / T), 2)
    # cell-size sweep: same fluid and Stokes layer, R = 10, 20, 40 (delta / R = 0.6, 0.3, 0.15)
    sweep = []
    for tag, fname in (("R10", "impedance_a_R10.npz"), ("R20", "impedance_a.npz"), ("R40", "impedance_a_R40.npz")):
        if not os.path.exists(os.path.join(HERE, fname)):
            continue
        im = load(fname)
        of = offsets(im)
        kk = int(np.where((of == (1, 0)).all(1))[0][0])
        kd = int(np.where((of == (1, 1)).all(1))[0][0])
        c_ = -im["Ga"][im["k0"]].imag / im["w"]
        g1 = im["Ga"][kk] / (im["w"] * c_)
        gd = im["Ga"][kd] / (im["w"] * c_)
        amp, lag = decay_a(im)
        sweep.append(dict(R=im["R"], delta_over_R=round(float(im["delta"] / im["R"]), 3),
                          added_mass_over_rho_pi_R2=round(float(im["Ga"][im["k0"]].real / im["w"] ** 2 / (np.pi * im["R"] ** 2)), 3),
                          G1_over_wc=[round(float(g1.real), 4), round(float(g1.imag), 4)],
                          Gdiag_over_wc=[round(float(gd.real), 4), round(float(gd.imag), 4)],
                          c_geared_over_c=round(float(1 + 4 * g1.imag - 4 * gd.imag), 3),
                          c_aligned_over_c=round(float(1 - 4 * g1.imag - 4 * gd.imag), 3),
                          model_amp_along_x=[round(float(v), 5) for v in amp[1:5]],
                          model_lag_pi_along_x=[round(float(v), 3) for v in lag[1:5]]))
    out["size_sweep"] = sweep
    res["sweep"] = sweep
    if os.path.exists(os.path.join(HERE, "free_m10.npz")) and "fr" in res:
        f10 = load("free_m10.npz")
        P10 = int(round(f10["t"][-1] / T))
        X10 = demod(f10["t"], f10["a"], w, (P10 - 4) * T, P10 * T)
        te10, E10 = envelope(f10["t"], f10["a"], w, T)
        r10 = []
        for dx in range(1, n // 2 + 1):
            k = int(np.where((off == (dx, 0)).all(1))[0][0])
            hit = np.where(np.abs(E10[:, k]) >= 0.5 * abs(X10[k]))[0]
            r10.append(dict(cells=dx, arrival_periods=round(float(te10[hit[0]] / T), 2) if hit.size else None,
                            amp=round(float(abs(X10[k]) / A), 4),
                            phase_pi=round(float(np.angle(X10[k] / X10[k0]) / np.pi), 3)))
        out["mass10_along_x"] = r10
        out["mass10_Q_lone"] = round(float((f10["M"] + madd) * w / c), 2)
        res.update(f10=f10, X10=X10, te10=te10, E10=E10)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    if "fr" in res:
        figure(ia, res)


def figure(d, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                         "font.size": 15, "axes.titlelocation": "left", "axes.titlesize": 16})
    fr, off, Xa, Xb = res["fr"], res["off"], res["Xa"], res["Xb"]
    n, D, R, A, k0, T, w, m = d["n"], d["D"], d["R"], d["A"], d["k0"], d["period"], d["w"], d["m"]
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 11.5), gridspec_kw=dict(left=0.05, right=0.99, top=0.95,
                                                                            bottom=0.08, wspace=0.25, hspace=0.3))
    th = np.linspace(0, 2 * np.pi, 400)
    L = iter("ABCDEF")
    ref = Xa[k0]

    # (A) shapes at the moment the driven cell is widest along x; amplitude in grey and drawn larger
    ax = axes[0, 0]
    amp = np.abs(Xa) / A
    for k, (x0, y0) in enumerate(d["centers"]):
        za = Xa[k] / ref * A          # complex amplitude with the driven cell's phase set to 0
        zb = Xb[k] / ref * A
        ax.add_patch(plt.Circle((x0 / D, y0 / D), R / D * 0.98, color=str(1 - 0.85 * min(1, amp[k] ** 0.5)), lw=0))
        z = np.array([za, zb]) / max(abs(za), abs(zb), 1e-12) * 2.5 * A    # same drawn size for every cell
        r = R + z[0].real * np.cos(m * th) + z[1].real * np.sin(m * th)
        ax.plot((x0 + r * np.cos(th)) / D, (y0 + r * np.sin(th)) / D, color="k", lw=1.6)
        if k != k0 and amp[k] >= 0.01:
            ax.text(x0 / D, y0 / D, f"{amp[k]:.2f}", ha="center", va="center", fontsize=10)
    ax.text(d["centers"][k0, 0] / D, d["centers"][k0, 1] / D, "drive", ha="center", va="center", fontsize=10,
            color="w")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, n + 1, 3))
    ax.set_yticks(range(0, n + 1, 3))
    ax.set_xlabel("$x / D$")
    ax.set_ylabel("$y / D$")
    ax.set_title(f"({next(L)}) Shapes when the driven cell is widest")

    # (B) flow at that moment, zoomed on the middle 5 x 5 cells
    ax = axes[0, 1]
    U, V = fr["q_u"], fr["q_v"]
    g = np.arange(d["L"]) + 0.5
    sp = np.hypot(U, V)
    lo, hi = (n // 2 - 2) * D, (n // 2 + 3) * D
    s = slice(lo, hi)
    gg = g[s] / D
    ax.contourf(gg, gg, (sp[s, s] / sp[s, s].max()).T, levels=np.linspace(0, 1, 11), cmap="Greys", alpha=0.5)
    ax.streamplot(gg, gg, U[s, s].T, V[s, s].T, color="k", linewidth=0.7, density=2.0, arrowsize=0.8)
    for k, (x0, y0) in enumerate(d["centers"]):
        if lo < x0 < hi and lo < y0 < hi:
            r = R + fr["q_a"][k] * np.cos(m * th) + fr["q_b"][k] * np.sin(m * th)
            ax.plot((x0 + r * np.cos(th)) / D, (y0 + r * np.sin(th)) / D, color="k", lw=2.0)
    ax.set_xlim(lo / D, hi / D)
    ax.set_ylim(lo / D, hi / D)
    ax.set_aspect("equal")
    ax.set_xlabel("$x / D$")
    ax.set_ylabel("$y / D$")
    ax.set_title(f"({next(L)}) Flow when the driven wall moves fastest")

    # (C) a(t) of cells along +x
    ax = axes[0, 2]
    t = fr["t"] / T
    styles = ["-", (0, (5, 2)), (0, (2, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2))]
    for dx in range(0, min(4, n // 2) + 1):
        k = int(np.where((off == (dx, 0)).all(1))[0][0])
        gain = 1 if dx == 0 else A / max(abs(Xa[k]), 1e-12)
        lab = "driven" if dx == 0 else f"{dx} cell{'s' if dx > 1 else ''} along $x$, x{gain:.0f}"
        ax.plot(t, fr["a"][:, k] / A * (1 if dx == 0 else gain) + 2.6 * dx, color="k", lw=0.8, ls=styles[dx % 5])
        ax.text(t[-1] * 1.01, 2.6 * dx, lab if dx else "driven", fontsize=10, va="center")
    ax.set_xlim(0, t[-1] * 1.0)
    ax.set_yticks([])
    ax.set_xlabel("time (drive periods)")
    ax.set_title(f"({next(L)}) Oval amplitude $a(t)$, rescaled")
    ax.set_xlim(0, t[-1] * 1.32)

    # (D) amplitude vs distance, free run and linear theory
    ax = axes[1, 0]
    dist = np.hypot(off[:, 0], off[:, 1])
    mask = np.arange(len(dist)) != k0
    ax.semilogy(dist[mask], np.abs(Xa[mask]) / A, "o", mfc="k", mec="k", ms=6, label="free run")
    ax.semilogy(dist[mask], np.abs(res["pa"][mask]) / A, "o", mfc="none", mec="0.4", ms=11, mew=1.2,
                label="linear model")
    if "X10" in res:
        ax.semilogy(dist[mask], np.abs(res["X10"][mask]) / A, "s", mfc="0.6", mec="0.6", ms=5,
                    label="free run, cells 10x heavier")
    ax.set_xlabel("distance from driven cell / $D$")
    ax.set_ylabel("$|a| / A$")
    ax.legend(frameon=False, fontsize=12)
    ax.set_title(f"({next(L)}) Steady amplitude of each cell")

    # (E) unwrapped phase lag along +x, against the checkerboard (pi per cell)
    ax = axes[1, 1]
    cells = np.arange(0, n // 2 + 1)
    ks = [int(np.where((off == (dx, 0)).all(1))[0][0]) for dx in cells]
    lag = -np.unwrap(np.angle(Xa[ks] / ref)) / np.pi
    lag_m = -np.unwrap(np.angle(res["pa"][ks] / res["pa"][k0])) / np.pi
    ax.plot(cells, cells * 1.0, color="0.5", lw=1, ls=(0, (4, 3)), label=r"checkerboard, $\pi$ per cell")
    ax.plot(cells, lag_m, "o", mfc="none", mec="0.4", ms=12, mew=1.2, label="linear model")
    ax.plot(cells, lag, "o", mfc="k", mec="k", ms=6, label="free run")
    if "X10" in res:
        lag10 = -np.unwrap(np.angle(res["X10"][ks] / res["X10"][k0])) / np.pi
        ax.plot(cells, lag10, "s", mfc="0.6", mec="0.6", ms=5, label="cells 10x heavier")
    slope = np.polyfit(cells, lag, 1)[0]
    ax.text(0.97, 0.05, f"slope {slope:.2f}" + r"$\,\pi$ per cell", transform=ax.transAxes, ha="right", fontsize=12)
    ax.set_xticks(cells)
    ax.set_yticks(range(0, int(cells[-1]) + 1))
    ax.set_yticklabels(["0"] + [rf"${k}\pi$" if k > 1 else r"$\pi$" for k in range(1, int(cells[-1]) + 1)])
    ax.set_xlabel("cells from the driven cell, along $x$")
    ax.set_ylabel("phase lag behind driven cell")
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    ax.set_title(f"({next(L)}) Phase of mode $a$ along $x$")

    # (F) envelope growth along x
    ax = axes[1, 2]
    te = res["te"] / T
    for dx in range(1, min(4, n // 2) + 1):
        k = int(np.where((off == (dx, 0)).all(1))[0][0])
        ax.plot(te, np.abs(res["Ea"][:, k]) / abs(Xa[k]), color="k", lw=1.2, ls=styles[dx % 5],
                label=f"{dx} cell{'s' if dx > 1 else ''}")
    if "E10" in res:
        for dx in range(1, min(4, n // 2) + 1):
            k = int(np.where((off == (dx, 0)).all(1))[0][0])
            ax.plot(res["te10"] / T, np.abs(res["E10"][:, k]) / abs(res["X10"][k]), color="0.6", lw=1.2,
                    ls=styles[dx % 5])
        ax.text(0.97, 0.97, "black: cells as in (A)\ngrey: cells 10x heavier", transform=ax.transAxes, ha="right", va="top",
                fontsize=11, color="0.3")
    ax.axhline(0.5, color="0.5", lw=1, ls=(0, (4, 3)))
    ax.set_xlim(0, 30 if "E10" in res else 10)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("time (drive periods)")
    ax.set_ylabel("amplitude / final amplitude")
    ax.legend(frameon=False, fontsize=12, title="along $x$", title_fontsize=12, loc="lower right")
    ax.set_title(f"({next(L)}) Growth of the response")
    fig.text(0.5, 0.012, "Center cell driven, a = A sin(wt); other cells free, each oval mode a mass on a spring tuned to "
             "the drive. (A) grey and numbers: steady amplitude |a|/A; outlines show shape and phase, all drawn at 2.5 A.",
             ha="center", fontsize=12, color="0.3")
    fig.savefig(os.path.join(HERE, "driven_cell.png"), dpi=130)


if __name__ == "__main__":
    main()
