"""Experiment 12 runs: bands, energy check, wave patterns, time run, checks figure and movie.

    python experiments/12_ideal_fluid_cells/run.py           # everything (about 10 minutes)
    python experiments/12_ideal_fluid_cells/run.py --movie   # movie only, from saved run

Lattice: hexagonal, spacing d = 1, shell radius R = 0.45 (gap 0.1). Shell mass zero, so all the
inertia is fluid. Shell stiffness K per oval mode equals the total self mass of a cell in the lattice,
so a cell alone in its cage would ring at frequency 1.
"""
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ideal as I  # noqa: E402

R = 0.45
NR = 26            # rings in the big lattice
SPONGE = 17        # sponge starts at this ring
RCUT = 6.0         # coupling kept out to this distance
W_DRIVE = 1.05     # drive frequency for the time run and movie
NW = 8             # rings in the window where the fluid is drawn
PER = 16           # movie frames per period


def setup():
    path = os.path.join(HERE, "blocks_R045.pkl")
    if not os.path.exists(path):
        bl, _, _ = I.bulk_blocks(R, 8, 8)
        pickle.dump(bl, open(path, "wb"))
    bl = pickle.load(open(path, "rb"))
    Mi = np.pi * I.RHO * R ** 2 / 2           # fluid inside the shell, per mode
    K = bl[(0.0, 0.0)][0, 0] + Mi
    return bl, Mi, K


def bands(bl, Mi, K, kx, ky):
    keys = np.array(list(bl.keys()))
    B = np.array(list(bl.values()))
    ph = np.exp(1j * (np.multiply.outer(kx, keys[:, 0]) + np.multiply.outer(ky, keys[:, 1])))
    Mk = np.einsum("...j,jab->...ab", ph, B).real + Mi * np.eye(2)
    ev, vec = np.linalg.eigh(Mk)
    return np.sqrt(K / ev), vec               # [..., 0] upper band, [..., 1] lower band


def path_k(n=120):
    G = np.array([0.0, 0.0])
    M = np.array([0.0, 2 * np.pi / np.sqrt(3)])
    Kp = np.array([2 * np.pi / 3, 2 * np.pi / np.sqrt(3)])
    segs, s, ticks = [], [], [0.0]
    s0 = 0.0
    for a, b in ((G, M), (M, Kp), (Kp, G)):
        t = np.linspace(0, 1, n)[:, None]
        segs.append(a + (b - a) * t)
        s.append(s0 + np.linalg.norm(b - a) * t[:, 0])
        s0 += np.linalg.norm(b - a)
        ticks.append(s0)
    return np.vstack(segs), np.concatenate(s), ticks


def big_lattice(bl, Mi, K):
    z, ring = I.hex_lattice(NR)
    N = len(z)
    M = I.lattice_mass(z, bl, RCUT) + Mi * np.eye(2 * N)
    g = 0.4 * np.clip((ring - SPONGE) / (NR - SPONGE), 0, 1) ** 2
    D = np.repeat(g, 2) * K
    return z, ring, M, D


def response(M, D, K, w, f):
    n = M.shape[0]
    return np.linalg.solve(K * np.eye(n) - w * w * M - 1j * w * np.diag(D), f)


def newmark(M, D, K, f_of_t, T, dt, every, q0=None, v0=None):
    """Average-acceleration Newmark: exact energy balance for a linear system."""
    n = M.shape[0]
    q = np.zeros(n) if q0 is None else q0.copy()
    v = np.zeros(n) if v0 is None else v0.copy()
    Mf = M
    a = np.linalg.solve(Mf, f_of_t(0.0) - D * v - K * q)
    S = np.linalg.inv(Mf + 0.5 * dt * np.diag(D) + 0.25 * dt * dt * K * np.eye(n))
    steps = int(round(T / dt))
    out_q, out_a, out_v, out_t = [q.copy()], [a.copy()], [v.copy()], [0.0]
    for s in range(1, steps + 1):
        t = s * dt
        qp = q + dt * v + 0.25 * dt * dt * a
        vp = v + 0.5 * dt * a
        a = S @ (f_of_t(t) - D * vp - K * qp)
        q = qp + 0.25 * dt * dt * a
        v = vp + 0.5 * dt * a
        if s % every == 0:
            out_q.append(q.copy()); out_a.append(a.copy()); out_v.append(v.copy()); out_t.append(t)
    return np.array(out_t), np.array(out_q), np.array(out_v), np.array(out_a)


def energy(M, K, q, v):
    return 0.5 * np.einsum("ti,ij,tj->t", v, M, v) + 0.5 * K * np.einsum("ti,ti->t", q, q)


# ------------------------------------------------------------------------------------------------
def fluid_map(xg, yg, P=6):
    """Matrix from shell mode speeds of the window cells to the flow potential on the grid."""
    z, ring = I.hex_lattice(NW)
    N = len(z)
    Q = 4 * P + 8
    th = 2 * np.pi * np.arange(Q) / Q
    eith = np.exp(1j * th)
    n = np.arange(1, P + 1)
    proj = np.vstack([np.cos(np.outer(n, th)), np.sin(np.outer(n, th))]) * (2.0 / Q)
    nu = 2 * P * N
    A = np.zeros((nu, nu))
    for i in range(N):
        zz = z[i] + R * eith
        s = R / (zz[:, None] - z[None, :])
        sn = s[:, :, None] ** n[None, None, :]
        ur = -(n / R)[None, None, :] * sn * s[:, :, None] * eith[:, None, None]
        Ui = np.empty((Q, N, 2, P))
        Ui[:, :, 0, :] = ur.real
        Ui[:, :, 1, :] = -ur.imag
        A[2 * P * i:2 * P * (i + 1)] = proj @ Ui.reshape(Q, nu)
    G = np.zeros((nu, 2 * N))
    for i in range(N):
        G[2 * P * i + 1, 2 * i] = 1.0
        G[2 * P * i + P + 1, 2 * i + 1] = 1.0
    C = np.linalg.solve(A, G)                                     # (nu, 2N)
    zg = (xg + 1j * yg).ravel()
    out = np.zeros((zg.size, 2 * N))
    dist = np.abs(zg[:, None] - z[None, :])
    inside = dist.min(axis=1) < R
    owner = dist.argmin(axis=1)
    for c0 in range(0, zg.size, 4000):
        zc = zg[c0:c0 + 4000]
        s = R / (zc[:, None] - z[None, :])
        sn = s[:, :, None] ** n[None, None, :]
        F = np.empty((zc.size, N, 2, P))
        F[:, :, 0, :] = sn.real
        F[:, :, 1, :] = -sn.imag
        out[c0:c0 + 4000] = F.reshape(zc.size, nu) @ C
    # inside a shell: phi = (r^2 / 2R)(adot cos 2th + bdot sin 2th)
    for k in np.nonzero(inside)[0]:
        j = owner[k]
        dz = zg[k] - z[j]
        r2, ang = abs(dz) ** 2, np.angle(dz)
        out[k] = 0.0
        out[k, 2 * j] = r2 / (2 * R) * np.cos(2 * ang)
        out[k, 2 * j + 1] = r2 / (2 * R) * np.sin(2 * ang)
    return out, inside.reshape(xg.shape), N


# ------------------------------------------------------------------------------------------------
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                         "font.size": 14, "axes.titlelocation": "left", "axes.titlesize": 15})
    return plt


def band_axes(ax, bl, Mi, K, mark=None):
    k, s, ticks = path_k()
    w, _ = bands(bl, Mi, K, k[:, 0], k[:, 1])
    ax.axhspan(0, w[:, 1].min(), color="0.9", lw=0)
    ax.axhspan(w[:, 0].max(), 2.0, color="0.9", lw=0)
    ax.plot(s, w[:, 0], color="k", lw=2.2, label="upper band")
    ax.plot(s, w[:, 1], color="0.5", lw=2.2, ls="--", label="lower band")
    ax.plot(s, np.minimum(np.linalg.norm(k, axis=1), 2.0), color="k", lw=1, ls=":", label="option A (light line)")
    if mark is not None:
        ax.axhline(mark, color="k", lw=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$\\Gamma$", "M", "K", "$\\Gamma$"])
    ax.set_xlim(0, ticks[-1])
    ax.set_ylim(0, 1.75)
    ax.set_ylabel("frequency")
    ax.text(ticks[-1] * 0.98, 1.62, "no wave (gap)", ha="right", fontsize=11, color="0.3")
    ax.text(ticks[-1] * 0.98, 0.35, "no wave (gap)", ha="right", fontsize=11, color="0.3")
    return w


def checks_figure(path, bl, Mi, K, closed, fd, res):
    plt = style()
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    band_axes(ax[0, 0], bl, Mi, K, mark=W_DRIVE)
    ax[0, 0].legend(loc="upper center", fontsize=11, frameon=False, ncol=1)
    ax[0, 0].set_title("(A) Bands of the shell lattice vs option A")
    g = np.linspace(-2 * np.pi / np.sqrt(3) * 1.15, 2 * np.pi / np.sqrt(3) * 1.15, 241)
    KX, KY = np.meshgrid(g, g)
    W, _ = bands(bl, Mi, K, KX, KY)
    ax[0, 1].contour(KX, KY, W[..., 1], levels=[0.85, 0.95, W_DRIVE, 1.10], colors="0.45", linestyles="--")
    cs = ax[0, 1].contour(KX, KY, W[..., 0], levels=[1.15, 1.25, 1.35, 1.45], colors="k")
    ax[0, 1].clabel(cs, fontsize=10)
    th = np.linspace(0, 2 * np.pi, 7)
    rM = 4 * np.pi / 3
    ax[0, 1].plot(rM * np.cos(th), rM * np.sin(th), color="0.6", lw=1)
    ax[0, 1].set_aspect("equal")
    ax[0, 1].set_xlabel("$k_x$")
    ax[0, 1].set_ylabel("$k_y$")
    ax[0, 1].set_title("(B) Lines of equal frequency (solid upper, dashed lower)")
    t, E = closed
    ax[1, 0].plot(t, E / E[0], color="k", lw=2)
    ax[1, 0].set_ylim(0.99, 1.01)
    ax[1, 0].set_xlabel("time")
    ax[1, 0].set_ylabel("energy / start")
    ax[1, 0].set_title("(C) Closed lattice, no drive, no sponge: no loss")
    z, ring, a = fd
    m = ring < SPONGE
    amp = np.abs(a[m]) / np.abs(a[m]).max()
    ax[1, 1].scatter(z.real[m], z.imag[m], c=amp ** 0.5, cmap="Greys", s=22, vmin=0, vmax=0.8, lw=0)
    ax[1, 1].set_aspect("equal")
    ax[1, 1].set_title(f"(D) Wave amplitude from one cell (oval a driven), frequency {W_DRIVE}")
    ax[1, 1].set_xlabel("x (cells)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def movie(path, run, bl, Mi, K, fps=24, split=False, title=None, note=None):
    plt = style()
    from matplotlib.animation import FFMpegWriter
    from matplotlib.collections import LineCollection
    z, ring = I.hex_lattice(NR)
    zw, _ = I.hex_lattice(NW)
    Nw = len(zw)
    t, q, v, acc = run["t"], run["q"], run["v"], run["a"]
    half = 6.0
    xs = np.linspace(-half, half, 260)
    xg, yg = np.meshgrid(xs, xs * 0.75)
    Fm, inside, _ = fluid_map(xg, yg)
    p = -(acc[:, :2 * Nw] @ Fm.T)                                 # pressure = -rho dphi/dt
    psc = np.percentile(np.abs(p[-PER * 4:]), 99) + 1e-12
    qsc = np.percentile(np.abs(q[-PER * 4:, :2 * Nw]), 99) + 1e-12
    fig = plt.figure(figsize=(16, 9.4))
    axA = fig.add_axes([0.03, 0.08, 0.52, 0.80])
    if split:
        axB = fig.add_axes([0.575, 0.47, 0.205, 0.41])
        axB2 = fig.add_axes([0.785, 0.47, 0.205, 0.41])
    else:
        axB = fig.add_axes([0.60, 0.47, 0.36, 0.41])
    axC = fig.add_axes([0.63, 0.08, 0.33, 0.30])
    im = axA.imshow(np.zeros_like(xg), extent=(-half, half, -0.75 * half, 0.75 * half), origin="lower",
                    cmap="Greys", vmin=-1, vmax=1, interpolation="bicubic")
    sel = np.nonzero((np.abs(zw.real) < half + 0.6) & (np.abs(zw.imag) < 0.75 * half + 0.6))[0]
    th = np.linspace(0, 2 * np.pi, 61)
    lc = LineCollection([], colors="k", linewidths=1.3)
    axA.add_collection(lc)
    lc_ax = LineCollection([], colors="k", linewidths=0.8)
    axA.add_collection(lc_ax)
    axA.set_xlim(-half, half)
    axA.set_ylim(-0.75 * half, 0.75 * half)
    axA.set_aspect("equal")
    axA.set_xticks([])
    axA.set_yticks([])
    axA.set_title("(A) Shells (ovals drawn larger; line = long axis) and fluid pressure (shading)")
    m = ring < SPONGE
    sc = axB.scatter(z.real[m], z.imag[m], c=np.zeros(m.sum()), cmap="Greys", s=16, vmin=-1, vmax=1, lw=0)
    axB.plot([-half, half, half, -half, -half], [-0.75 * half, -0.75 * half, 0.75 * half, 0.75 * half, -0.75 * half],
             color="k", lw=0.8)
    axB.set_aspect("equal")
    axB.set_xticks([])
    axB.set_yticks([])
    axB.set_title("(B) Whole lattice: oval a (axis 0$^\\circ$)" if split else "(B) Whole lattice: oval amplitude a")
    if split:
        sc2 = axB2.scatter(z.real[m], z.imag[m], c=np.zeros(m.sum()), cmap="Greys", s=6, vmin=-1, vmax=1, lw=0)
        axB2.set_aspect("equal")
        axB2.set_xticks([])
        axB2.set_yticks([])
        axB2.set_title("oval b (axis 45$^\\circ$)", fontsize=13)
        sc.set_sizes([6])
        qb_all = q[:, 1::2][:, m]
    band_axes(axC, bl, Mi, K, mark=W_DRIVE)
    axC.set_title("(C) Bands; line = drive")
    axC.legend(loc="upper left", fontsize=8.5, frameon=False)
    stamp = fig.text(0.5, 0.955, "", ha="center", fontsize=17)
    fig.text(0.5, 0.015, note or ("Ideal fluid, no viscosity: the shells push the fluid, the fluid pushes the neighbours. "
             "Nothing is lost. Square in (B) = view in (A). Outer rings of (B) hidden: sponge."),
             ha="center", fontsize=11.5, color="0.3")
    qa_all = q[:, 0::2][:, m]
    bsc = np.percentile(np.abs(qa_all[-PER * 4:]), 98) + 1e-12
    if split:                                   # one scale for a and b, so the weaker one shows as weaker
        bsc = np.percentile(np.abs(np.r_[qa_all[-PER * 4:].ravel(), qb_all[-PER * 4:].ravel()]), 98) + 1e-12
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "24"])
    with writer.saving(fig, path, dpi=100):
        for k in range(len(t)):
            pk = np.tanh(p[k] / psc).reshape(xg.shape)
            im.set_data(pk)
            segs, axes_ = [], []
            for j in sel:
                a, b = q[k, 2 * j], q[k, 2 * j + 1]
                amp = np.tanh(np.hypot(a, b) / qsc)                  # 0..1, compressed
                s = 0.048 * amp / (np.hypot(a, b) + 1e-30)            # largest bulge 0.048: under half the gap
                r = R + s * (a * np.cos(2 * th) + b * np.sin(2 * th))
                segs.append(np.c_[zw[j].real + r * np.cos(th), zw[j].imag + r * np.sin(th)])
                phi = 0.5 * np.arctan2(b, a)
                L = R * amp                                           # long axis line, length ~ amplitude
                axes_.append([[zw[j].real - L * np.cos(phi), zw[j].imag - L * np.sin(phi)],
                              [zw[j].real + L * np.cos(phi), zw[j].imag + L * np.sin(phi)]])
            lc.set_segments(segs)
            lc_ax.set_segments(axes_)
            sc.set_array(np.tanh(qa_all[k] / bsc))
            if split:
                sc2.set_array(np.tanh(qb_all[k] / bsc))
            stamp.set_text(f"{title or 'Option B: oval shells coupled by an ideal fluid'} — t = {t[k]:6.1f}  "
                           f"(drive period {2 * np.pi / W_DRIVE:.1f})")
            writer.grab_frame()


# ------------------------------------------------------------------------------------------------
def main(movie_only=False):
    bl, Mi, K = setup()
    if not movie_only:
        t0 = time.perf_counter()
        res = {"R": R, "d": 1.0, "K": K, "self_mass_isolated_outer": float(I.outer_mass(np.array([0j]), R, 4)[0, 0]),
               "self_mass_theory": I.self_mass(R)}
        pair = {P: I.outer_mass(np.array([0j, 1 + 0j]), R, P)[0, 2] for P in (2, 4, 8, 12)}
        res["pair_coupling_vs_P"] = {str(P): float(v) for P, v in pair.items()}
        res["pair_coupling_leading_order"] = 3 * np.pi * R ** 6
        G = np.array([0.0]), np.array([0.0])
        Mpt = np.array([0.0]), np.array([2 * np.pi / np.sqrt(3)])
        Kpt = np.array([2 * np.pi / 3]), np.array([2 * np.pi / np.sqrt(3)])
        for name, kk in (("Gamma", G), ("M", Mpt), ("K", Kpt)):
            res["band_" + name] = bands(bl, Mi, K, *kk)[0][0].tolist()
        g = np.linspace(-8, 8, 401)
        KX, KY = np.meshgrid(g, g)
        W, _ = bands(bl, Mi, K, KX, KY)
        res["upper_band"] = [float(W[..., 0].min()), float(W[..., 0].max())]
        res["lower_band"] = [float(W[..., 1].min()), float(W[..., 1].max())]
        dk = g[1] - g[0]
        vg = [np.hypot(*np.gradient(W[..., b], dk)).max() for b in (0, 1)]
        res["max_group_velocity"] = [float(v) for v in vg]
        # polarization at small k along x and y
        pol = {}
        for name, kk in (("k_along_x", (np.array([0.3]), np.array([0.0]))),
                         ("k_along_y", (np.array([0.0]), np.array([0.3])))):
            w, vec = bands(bl, Mi, K, *kk)
            pol[name] = {"upper_w": float(w[0, 0]), "upper_oval_axis_deg": float(np.degrees(0.5 * np.arctan2(vec[0, 1, 0], vec[0, 0, 0]))),
                         "lower_w": float(w[0, 1]), "lower_oval_axis_deg": float(np.degrees(0.5 * np.arctan2(vec[0, 1, 1], vec[0, 0, 1])))}
        res["polarization"] = pol
        z, ring, M, D = big_lattice(bl, Mi, K)
        N = len(z)
        res["cells"] = N
        res["min_mass_eigenvalue"] = float(np.linalg.eigvalsh(M).min())
        # closed lattice, random start, no sponge
        rng = np.random.default_rng(1)
        q0 = rng.standard_normal(2 * N) * np.repeat(ring < 6, 2)
        tc, qc, vc, _ = newmark(M, np.zeros(2 * N), K, lambda t: np.zeros(2 * N), 200.0, 0.2, 5, q0=q0)
        Ec = energy(M, K, qc, vc)
        res["closed_energy_swing"] = float(Ec.max() / Ec.min() - 1)
        # steady patterns
        f = np.zeros(2 * N); f[0] = 1.0
        a_fd = response(M, D, K, W_DRIVE, f)[0::2]
        # wavenumber along +y column (x = 0): phase gradient
        col = np.nonzero((np.abs(z.real) < 1e-6) & (z.imag > 2) & (ring < SPONGE - 2))[0]
        col = col[np.argsort(z.imag[col])]
        ph = np.unwrap(np.angle(a_fd[col]))
        ky = np.polyfit(z.imag[col], ph, 1)[0]
        wth, _ = bands(bl, Mi, K, np.array([0.0]), np.array([abs(ky)]))
        res["drive_w"] = W_DRIVE
        res["y_column_phase_slope"] = float(ky)
        res["y_column_note"] = ("time factor exp(-i w t): positive slope = phase moves outward; "
                                "negative = phase moves inward while energy goes out (backward wave)")
        res["band_at_measured_ky"] = wth[0].tolist()
        # above the band: decay
        a_gap = response(M, D, K, 1.6, f)[0::2]
        rr = np.abs(z)
        res["gap_w1.6_amp_ratio_r3_over_r1"] = float(np.abs(a_gap[(rr > 2.9) & (rr < 3.1)]).max() /
                                                     np.abs(a_gap[(rr > 0.9) & (rr < 1.1)]).max())
        res["pass_w1.3_amp_ratio_r3_over_r1"] = float(
            np.abs(response(M, D, K, 1.3, f)[0::2][(rr > 2.9) & (rr < 3.1)]).max() /
            np.abs(response(M, D, K, 1.3, f)[0::2][(rr > 0.9) & (rr < 1.1)]).max())
        # time run for the movie
        Tp = 2 * np.pi / W_DRIVE
        dt = Tp / 32
        ramp = 3 * Tp

        def drive(t):
            f = np.zeros(2 * N)
            f[0] = 0.05 * np.sin(W_DRIVE * t) * min(1.0, t / ramp)
            return f
        tt, qq, vv, aa = newmark(M, D, K, drive, 45 * Tp, dt, 32 // PER)
        np.savez_compressed(os.path.join(HERE, "run_point.npz"), t=tt, q=qq, v=vv, a=aa)
        res["seconds"] = round(time.perf_counter() - t0, 1)
        json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=2)
        print(json.dumps(res, indent=2), flush=True)
        checks_figure(os.path.join(HERE, "ideal_fluid_checks.png"), bl, Mi, K, (tc, Ec), (z, ring, a_fd), res)
    run = dict(np.load(os.path.join(HERE, "run_point.npz")))
    movie(os.path.join(HERE, "ideal_fluid_cells.mp4"), run, bl, Mi, K)
    print("done", flush=True)


def upper_band_movie(w=1.30):
    """Same run and movie with the drive in the upper band, where crests and energy both move out."""
    global W_DRIVE
    W_DRIVE = w
    tag = f"w{int(round(w * 100))}"
    bl, Mi, K = setup()
    z, ring, M, D = big_lattice(bl, Mi, K)
    N = len(z)
    f = np.zeros(2 * N); f[0] = 1.0
    q_fd = response(M, D, K, w, f)
    # crest direction along the +y column: phase slope (time factor exp(-i w t))
    col = np.nonzero((np.abs(z.real) < 1e-6) & (z.imag > 2) & (ring < SPONGE - 2))[0]
    col = col[np.argsort(z.imag[col])]
    ky = np.polyfit(z.imag[col], np.unwrap(np.angle(q_fd[0::2][col])), 1)[0]
    res = {"drive_w": w, "y_column_phase_slope": float(ky),
           "band_at_measured_ky": bands(bl, Mi, K, np.array([0.0]), np.array([abs(ky)]))[0][0].tolist()}
    Tp = 2 * np.pi / w
    dt = Tp / 32

    def drive(t):
        g = np.zeros(2 * N)
        g[0] = 0.05 * np.sin(w * t) * min(1.0, t / (3 * Tp))
        return g
    tt, qq, vv, aa = newmark(M, D, K, drive, 45 * Tp, dt, 32 // PER)
    amp = np.hypot(qq[:, 0::2], qq[:, 1::2])
    front = {}
    for T in (10, 20, 40, 60):
        k = np.argmin(np.abs(tt - T))
        a = amp[max(0, k - PER):k + 1].max(0)
        front[str(T)] = int(ring[(a > 0.05 * a[ring == 1].max()) & (ring < SPONGE)].max())
    res["energy_front_ring_vs_time"] = front
    json.dump(res, open(os.path.join(HERE, f"results_{tag}.json"), "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)
    np.savez_compressed(os.path.join(HERE, f"run_point_{tag}.npz"), t=tt, q=qq, v=vv, a=aa)
    movie(os.path.join(HERE, f"ideal_fluid_cells_{tag}.mp4"), dict(t=tt, q=qq, v=vv, a=aa), bl, Mi, K)
    print("done", flush=True)


if __name__ == "__main__":
    if "--upper" in sys.argv:
        upper_band_movie()
    else:
        main("--movie" in sys.argv)
