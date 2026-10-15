"""Check the numba LBM solver against exact solutions, and time it against the numpy original.

    python validation/run_checks.py              # all checks, about 2-3 min on 2 cores
    python validation/run_checks.py --only stokes

Checks
  poiseuille  channel flow driven by a body force:  u(s) = g s (H - s) / (2 nu)
  stokes      Stokes layer above a wall oscillating along itself (Stokes' second problem):
              u(s, t) = U exp(-s/d) cos(w t - s/d),  d = sqrt(2 nu / w)
              -- the boundary layer that drives steady streaming next to a vibrating string
  rotlet      cylinder spinning in still fluid: u_theta(r) = Omega R^2 / r  (exact Navier-Stokes)
  speed       lattice updates per second, numba against the original numpy solver

Writes validation/checks.json and validation/checks.png (grey scale).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from lbm import LBM  # noqa: E402


def poiseuille(tau=0.8, H=32, g=1e-6, steps=30000):
    sim = LBM(4, H + 2, tau)
    sim.solid[:, 0] = sim.solid[:, -1] = True
    sim.Fx[:] = g
    sim.Fx[sim.solid] = 0.0
    sim.run(steps)
    _, ux, _ = sim.fields()
    s = np.arange(1, H + 1) - 0.5                        # distance from the wall (half-way bounce-back)
    num = ux[0, 1:-1]
    exact = g * s * (H - s) / (2 * sim.nu)
    err = float(np.abs(num - exact).max() / exact.max())
    return dict(err=err, ok=err < 0.01, s=s, num=num, exact=exact)


def stokes(tau=1.0, period=2000, U=0.01, ny=160, periods=6, samples=16):
    sim = LBM(2, ny, tau)
    sim.solid[:, 0] = sim.solid[:, -1] = True             # bottom wall oscillates, top wall far away
    w = 2 * np.pi / period
    d = np.sqrt(2 * sim.nu / w)
    bottom = np.zeros_like(sim.solid)
    bottom[:, 0] = True

    def wall(t, s):
        s.uwx[bottom] = U * np.cos(w * (t + 0.5))         # wall speed at mid-step

    sim.run(periods * period, wall)
    s = np.arange(1, ny - 1) - 0.5
    errs, prof = [], []
    for k in range(samples):
        sim.run(period // samples, wall)
        _, ux, _ = sim.fields()
        exact = U * np.exp(-s / d) * np.cos(w * sim.t - s / d)
        errs.append(np.abs(ux[0, 1:-1] - exact).max() / U)
        if k % 4 == 0:
            prof.append((ux[0, 1:-1].copy(), exact))
    err = float(max(errs))
    return dict(err=err, ok=err < 0.02, delta=float(d), s=s, prof=prof)


def rotlet(n=200, R=20.0, Omega=0.002, tau=0.8, steps=60000):
    sim = LBM(n, n, tau)
    c = (n - 1) / 2.0
    X, Y = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing="ij")
    r = np.hypot(X, Y)
    sim.solid[:] = r <= R
    sim.uwx[:] = -Omega * Y
    sim.uwy[:] = Omega * X
    sim.run(steps)
    _, ux, uy = sim.fields()
    ut = (-ux * Y + uy * X) / np.where(r == 0, 1, r)
    bins = np.arange(R + 1, 1.5 * R + 0.01, 1.0)
    rc, num = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (r >= a) & (r < b)
        rc.append(r[m].mean())
        num.append(ut[m].mean())
    rc, num = np.array(rc), np.array(num)
    exact = Omega * R ** 2 / rc
    err = float(np.sqrt(np.mean((num - exact) ** 2) / np.mean(exact ** 2)))
    rr = np.linspace(R, c, 60)
    prof = [ut[(r >= a) & (r < a + 1)].mean() for a in rr[:-1]]
    return dict(err=err, ok=err < 0.03, r=rr[:-1] + 0.5, prof=np.array(prof), R=R, Omega=Omega)


def speed(n=256, steps_numba=2000, steps_numpy=40):
    sim = LBM(n, n, 0.8)
    sim.solid[:, 0] = sim.solid[:, -1] = True
    sim.run(10)                                          # compile
    t0 = time.perf_counter()
    sim.run(steps_numba)
    mlups_numba = n * n * steps_numba / (time.perf_counter() - t0) / 1e6
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lbm", "reference"))
    from core_numpy import feq, macroscopic, stream, bounce_back
    solid = np.zeros((n, n), bool)
    solid[:, 0] = solid[:, -1] = True
    f = feq(np.ones((n, n)), np.zeros((n, n, 2)))
    t0 = time.perf_counter()
    for _ in range(steps_numpy):
        rho, u = macroscopic(f)
        f = f - (f - feq(rho, u)) / 0.8
        f = stream(f)
        bounce_back(f, solid)
    mlups_numpy = n * n * steps_numpy / (time.perf_counter() - t0) / 1e6
    return dict(mlups_numba=mlups_numba, mlups_numpy=mlups_numpy, speedup=mlups_numba / mlups_numpy, ok=True)


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                         "font.size": 16, "axes.titlelocation": "left", "axes.titlesize": 17,
                         "legend.frameon": False, "legend.fontsize": 13})
    keys = [k for k in ("poiseuille", "stokes", "rotlet") if k in res]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.6 * len(keys), 4.8), squeeze=False)
    for ax, k in zip(axes[0], keys):
        r = res[k]
        if k == "poiseuille":
            ax.plot(r["exact"] * 1e3, r["s"], color="0.6", lw=4, label="exact")
            ax.plot(r["num"] * 1e3, r["s"], "o", color="k", ms=4, label="LBM")
            ax.set_xlabel(r"$u_x$ ($10^{-3}$)")
            ax.set_ylabel("distance from wall")
            ax.set_title(f"(A) Poiseuille, error {100 * r['err']:.2f}%")
        elif k == "stokes":
            for j, (num, ex) in enumerate(r["prof"]):
                ax.plot(ex / 0.01, r["s"], color="0.6", lw=3.5, label="exact" if j == 0 else None)
                ax.plot(num / 0.01, r["s"], "o", color="k", ms=3, label="LBM" if j == 0 else None)
            ax.set_ylim(0, 5 * r["delta"])
            ax.set_xlabel(r"$u_x / U$")
            ax.set_ylabel("distance from wall")
            ax.set_title(f"(B) Stokes layer, error {100 * r['err']:.2f}%")
        else:
            rr = np.linspace(r["R"], r["r"].max(), 200)
            ax.plot(rr / r["R"], r["Omega"] * r["R"] ** 2 / rr / (r["Omega"] * r["R"]), color="0.6", lw=4,
                    label=r"exact $\Omega R^2/r$")
            ax.plot(r["r"] / r["R"], r["prof"] / (r["Omega"] * r["R"]), "o", color="k", ms=4, label="LBM")
            ax.axvspan(1, 1.5, color="0.9", zorder=0)
            ax.set_xlabel("$r / R$")
            ax.set_ylabel(r"$u_\theta / \Omega R$")
            ax.set_title(f"(C) Spinning cylinder, error {100 * r['err']:.1f}%")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="+", choices=["poiseuille", "stokes", "rotlet", "speed"])
    a = p.parse_args()
    todo = a.only or ["poiseuille", "stokes", "rotlet", "speed"]
    res, summary = {}, {}
    for name in todo:
        t0 = time.perf_counter()
        r = globals()[name]()
        res[name] = r
        summary[name] = {k: v for k, v in r.items() if isinstance(v, (bool, float, int))}
        summary[name]["seconds"] = round(time.perf_counter() - t0, 1)
        print(name, "PASS" if r["ok"] else "FAIL", summary[name], flush=True)
    with open(os.path.join(HERE, "checks.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    figure(res, os.path.join(HERE, "checks.png"))
    sys.exit(0 if all(r["ok"] for r in res.values()) else 1)


if __name__ == "__main__":
    main()
