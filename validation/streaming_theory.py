"""Small-amplitude theory of the mean flow above a string carrying a standing wave.

The string sits at y = eta = A sin(kx) cos(wt) and moves only up and down. A wall at y = H is
still. To first order in A the flow is the exact unsteady Stokes solution

    psi_1 = Re[ f(y) cos(kx) e^{iwt} ],   (iw - nu (D^2 - k^2)) (D^2 - k^2) f = 0,
    f(0) = i w A / k,  f'(0) = 0,  f(H) = f'(H) = 0,

so no thin-layer approximation is needed (k delta need not be small). At second order the
period-averaged flow is a slow Stokes flow driven by the Reynolds stress,

    psi_bar = F(y) sin(2kx),   u_bar = F' sin(2kx),   v_bar = -2k F cos(2kx),
    (D^2 - 4k^2)^2 F = -(k / 4 nu) Re[ f g'^* - f' g^* ],   g = k^2 f - f'',

with the string condition moved from y = eta to y = 0 by a Taylor step:
    u_bar(0) = -<eta du_1/dy>  ->  F'(0) = -(A/4) Re f''(0),     F(0) = 0,     F(H) = F'(H) = 0.

This is the Eulerian mean (the average at a fixed point), which is what the LBM average gives.
Valid for A much smaller than delta and 1/k, and a small streaming Reynolds number.

    python validation/streaming_theory.py      # compares with validation/string_streaming_A*.npz
"""
import glob
import os

import numpy as np
from scipy.integrate import solve_bvp

HERE = os.path.dirname(os.path.abspath(__file__))


def first_order(k, w, nu, A, H):
    m = np.sqrt(k * k + 1j * w / nu)
    m = m if m.real > 0 else -m
    lam = np.array([k, -k, m, -m])
    # f = sum c_j exp(lam_j (y - y_ref_j)); scale growing modes from y = H to keep numbers small
    ref = np.array([H, 0.0, H, 0.0])

    def basis(y, d):
        y = np.atleast_1d(y)[:, None]
        return lam ** d * np.exp(lam * (y - ref))

    M = np.vstack([basis(0.0, 0), basis(0.0, 1), basis(H, 0), basis(H, 1)])
    rhs = np.array([1j * w * A / k, 0, 0, 0])
    c = np.linalg.solve(M, rhs)
    return lambda y, d=0: basis(y, d) @ c


def mean_flow(k, w, nu, A, H, n=4000):
    f = first_order(k, w, nu, A, H)

    def Q(y):
        f0, f1, f2, f3 = f(y, 0), f(y, 1), f(y, 2), f(y, 3)
        g, g1 = k * k * f0 - f2, k * k * f1 - f3
        return -(k / (4 * nu)) * np.real(f0 * np.conj(g1) - f1 * np.conj(g))

    beta = -(A / 4) * np.real(f(0.0, 2)[0])
    K2 = 4 * k * k

    def ode(y, Y):                       # Y = [F, F', F'', F''']
        F, F1, F2, F3 = Y
        F4 = Q(y) + 2 * K2 * F2 - K2 * K2 * F
        return np.vstack([F1, F2, F3, F4])

    def bc(Ya, Yb):
        return np.array([Ya[0], Ya[1] - beta, Yb[0], Yb[1]])

    y = np.concatenate([np.linspace(0, 10 / np.sqrt(w / (2 * nu)), n // 2), np.linspace(0, H, n // 2)])
    y = np.unique(y)
    sol = solve_bvp(ode, bc, y, np.zeros((4, y.size)), tol=1e-8, max_nodes=200000)
    return sol, beta


def compare(path):
    r = np.load(path)
    mu, mv, y0 = r["mu"], r["mv"], float(r["y0"])
    k, w, nu, A, d = float(r["k"]), float(r["w"]), float(r["nu"]), float(r["A"]), float(r["delta"])
    nx, ny = mu.shape
    H = (ny - 1.5) - y0                   # half-way bounce-back: top wall half a spacing inside
    sol, beta = mean_flow(k, w, nu, A, H)
    x = np.arange(nx) + 0.5
    sc = A * A * w * k
    js = np.arange(int(np.floor(y0)) + 1, ny - 1)
    yy = js - y0
    u_lbm = np.array([2 * np.mean(mu[:, j] * np.sin(2 * k * x)) for j in js]) / sc
    v_lbm = np.array([2 * np.mean(mv[:, j] * np.cos(2 * k * x)) for j in js]) / sc
    Y = sol.sol(yy)
    u_th = Y[1] / sc
    v_th = -2 * k * Y[0] / sc
    return dict(y=yy, d=d, u_lbm=u_lbm, v_lbm=v_lbm, u_th=u_th, v_th=v_th, A=A, kd=k * d, sol=sol, sc=sc, k=k)


def summary():
    rows, comps = [], []
    for p in sorted(glob.glob(os.path.join(HERE, "string_streaming_A*.npz")),
                    key=lambda q: float(q.split("_A")[-1][:-4])):
        c = compare(p)
        out = (c["y"] > 2.5) & (c["y"] < 6 * c["d"])        # outside the smeared string
        eu = np.abs(c["u_lbm"] - c["u_th"])[out].max() / np.abs(c["u_th"][out]).max()
        ev = np.abs(c["v_lbm"] - c["v_th"])[out].max() / np.abs(c["v_th"][out]).max()
        i = int(np.argmax(c["u_th"]))
        rows.append(dict(A=c["A"], kd=c["kd"], peak_u_theory=float(c["u_th"][i]), peak_u_lbm=float(c["u_lbm"][i]),
                         y_peak_over_delta=float(c["y"][i] / c["d"]), err_u=float(eu), err_v=float(ev)))
        comps.append((p, c))
    return rows, comps


def figure(comps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                         "font.size": 16, "axes.titlelocation": "left", "axes.titlesize": 17,
                         "legend.frameon": False, "legend.fontsize": 13})
    p, c = comps[-1]
    r = np.load(p)
    mu, mv, y0, A = r["mu"], r["mv"], float(r["y0"]), float(r["A"])
    nx, ny = mu.shape
    lam = nx
    x = (np.arange(nx) + 0.5) / lam
    y = (np.arange(ny) - y0) / lam
    win = np.abs(y) < 0.6
    fig = plt.figure(figsize=(17, 6.2))
    ax = fig.add_axes([0.04, 0.13, 0.36, 0.78])
    U, V = mu[:, win], mv[:, win]
    sp = np.hypot(U, V)
    ax.contourf(*np.meshgrid(x, y[win], indexing="ij"), sp / sp.max(), levels=np.linspace(0, 1, 11),
                cmap="Greys", alpha=0.5)
    ax.streamplot(x, y[win], U.T, V.T, color="k", linewidth=0.9, density=1.5, arrowsize=1.0)
    xs = np.linspace(0, 1, 400)
    ax.fill_between(xs, -A / lam * np.abs(np.sin(2 * np.pi * xs)), A / lam * np.abs(np.sin(2 * np.pi * xs)),
                    color="0.3", lw=0)
    for xn in (0, 0.5, 1):
        ax.plot(xn, 0, "o", color="k", ms=10, zorder=5, clip_on=False)
    for xa in (0.25, 0.75):
        ax.plot(xa, 0, "o", mfc="white", mec="k", ms=10, mew=1.6, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/\lambda$")
    ax.set_ylabel(r"$y/\lambda$")
    ax.set_title("(A) Mean Flow Around the String")
    marks = ("s", "^", "o")
    for n, (lab, key_l, key_t, pos) in enumerate(((r"(B) $\bar u$, along the string", "u_lbm", "u_th", 0.48),
                                                   (r"(C) $\bar v$, away from the string", "v_lbm", "v_th", 0.76))):
        bx = fig.add_axes([pos, 0.13, 0.21, 0.78])
        for (pp, cc), mk in zip(comps, marks):
            m = (cc["y"] > 0) & (cc["y"] < 6 * cc["d"])
            bx.plot(cc[key_l][m][::2], cc["y"][m][::2] / cc["d"], mk, color="k", mfc="white", ms=5,
                    label=f"LBM, $A = {cc['A']:g}$")
        m = (c["y"] > 0) & (c["y"] < 6 * c["d"])
        bx.plot(c[key_t][m], c["y"][m] / c["d"], color="0.55", lw=3, zorder=0, label="theory")
        bx.axvline(0, color="k", lw=0.8)
        bx.axhspan(0, 2.5 / c["d"], color="0.9", zorder=-1)
        bx.set_ylim(0, 6)
        bx.set_ylabel(r"$y/\delta$")
        bx.set_xlabel(("amplitude of " + (r"$\sin 2kx$" if n == 0 else r"$\cos 2kx$")) + r" / $A^2\omega k$")
        bx.set_title(lab)
        if n == 0:
            bx.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    fig.savefig(path, dpi=150)


def main():
    rows, comps = summary()
    for row in rows:
        print({k: round(v, 4) for k, v in row.items()})
    figure(comps, os.path.join(HERE, "string_streaming.png"))


if __name__ == "__main__":
    main()
