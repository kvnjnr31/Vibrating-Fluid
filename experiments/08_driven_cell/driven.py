"""Experiment 8: one cell of the square array is driven; the others are free to respond.

    python experiments/08_driven_cell/driven.py impedance      # held neighbors, about 4 min on 2 cores
    python experiments/08_driven_cell/driven.py free           # free neighbors, about 15 min
    python experiments/08_driven_cell/driven.py --plot          # figure and numbers from saved runs

Same cells as Experiment 5 (R = 20, spacing D = 48, gap 8, oval mode m = 2, A = 2, Stokes layer 6),
on an n x n lattice in a periodic box, with the center cell driven:

    r = R + a(t) cos 2 theta + b(t) sin 2 theta

The driven cell has a = A sin(wt), b = 0. Every other cell carries two oval modes, a (bulges along
x and y) and b (bulges along the diagonals). Each mode is a mass on a spring:

    M a'' + K a = Q_a,       Q_a = - sum over markers of (F . n) cos 2 theta ds

where F is the force per unit length that the immersed boundary puts on the fluid, so -F is the
force of the fluid on the cell wall. K is set so the cell, alone in the fluid, resonates at the
drive frequency (from the impedance run). No damping in the cell itself: all loss is in the fluid.

impedance   the other cells are held still (rigid circles). The fluid force on each held cell
            gives the coupling from the driven cell to every other cell, and the force on the
            driven cell gives its added mass and damping.
free        the other cells move. Records a(t), b(t) of every cell.

Time step: tau = 0.65 (nu = 0.05) with the same Stokes layer 6 as Experiment 5, so the period is
2262 steps instead of 679. The sound wavelength (c_s x period = 1306) is then three times the box
(432), so the fluid acts as incompressible across the array. The flow is the same in scaled terms:
delta / R = 0.3, A / R = 0.1, A^2 w / nu = 0.22, as in Experiment 5.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from lbm import LBM  # noqa: E402
from lbm.ib import IBString, interpolate, spread  # noqa: E402
from lbm.kernels import moments  # noqa: E402


class FastIB(IBString):
    """Same multi-direct forcing as IBString, without full-grid temporaries.

    Interpolation is linear, so interp(u* + F/2) = interp(u*) + interp(F)/2: u* is interpolated
    once and each iteration only interpolates the force field. Like IBString's force law
    dF = 2 (U - u), this takes rho = 1 (density varies by about 1e-3 here). No solid nodes.
    """

    def force(self, sim, X, Y, Ux, Uy, ds):
        if self._zero is None:
            self._zero = np.zeros_like(sim.Fx)
            self._u0 = np.empty(self.n)
            self._v0 = np.empty(self.n)
            self._uf = np.empty(self.n)
            self._vf = np.empty(self.n)
        moments(sim.f, sim.solid, self._zero, self._zero, sim.rho, sim.ux, sim.uy)   # u*
        interpolate(sim.ux, sim.uy, X, Y, self._u0, self._v0)
        sim.Fx[:] = 0.0
        sim.Fy[:] = 0.0
        self.Fl_x[:] = 0.0
        self.Fl_y[:] = 0.0
        for it in range(self.n_iter):
            if it == 0:
                self.Ui[:] = self._u0
                self.Vi[:] = self._v0
            else:
                interpolate(sim.Fx, sim.Fy, X, Y, self._uf, self._vf)
                self.Ui[:] = self._u0 + 0.5 * self._uf
                self.Vi[:] = self._v0 + 0.5 * self._vf
            dFx = 2.0 * (Ux - self.Ui)
            dFy = 2.0 * (Uy - self.Vi)
            spread(dFx, dFy, X, Y, ds, sim.Fx, sim.Fy)
            self.Fl_x += dFx
            self.Fl_y += dFy


class Array:
    """n x n cells on a square lattice; cell k = i * n + j sits at ((i + .5) D, (j + .5) D)."""

    def __init__(self, n=9, D=48, R=20.0, A=2.0, tau=0.65, delta=6.0, m=2):
        self.n, self.D, self.R, self.A, self.m = n, D, R, A, m
        self.L = L = n * D
        self.sim = LBM(L, L, tau)
        w = 2 * self.sim.nu / delta ** 2
        self.period = int(round(2 * np.pi / w))
        self.w = 2 * np.pi / self.period
        self.nm = nm = int(np.ceil(2 * np.pi * R))
        self.nc = nc = n * n
        th = 2 * np.pi * np.arange(nm) / nm
        ij = np.array([(i, j) for i in range(n) for j in range(n)])
        self.ij = ij
        self.centers = (ij + 0.5) * D + 0.25
        self.k0 = (n // 2) * n + n // 2                      # driven cell
        self.TH = np.tile(th, nc)
        self.cx = np.repeat(self.centers[:, 0], nm)
        self.cy = np.repeat(self.centers[:, 1], nm)
        self.ca, self.sa = np.cos(m * self.TH), np.sin(m * self.TH)
        self.ib = FastIB(self.TH.size)
        self.cosT, self.sinT = np.cos(self.TH), np.sin(self.TH)
        self.dth = 2 * np.pi / nm
        # mode amplitudes and speeds
        self.a = np.zeros(nc)
        self.b = np.zeros(nc)
        self.va = np.zeros(nc)
        self.vb = np.zeros(nc)
        self.Qa = np.zeros(nc)
        self.Qb = np.zeros(nc)

    def drive(self, t):
        """Driven cell: a = A sin(wt), b = 0."""
        return self.A * np.sin(self.w * t), self.A * self.w * np.cos(self.w * t)

    def markers(self, a, b, va, vb):
        """Marker positions, velocities and arc lengths from per-cell mode amplitudes."""
        A_ = np.repeat(a, self.nm)
        B_ = np.repeat(b, self.nm)
        VA = np.repeat(va, self.nm)
        VB = np.repeat(vb, self.nm)
        r = self.R + A_ * self.ca + B_ * self.sa
        rt = VA * self.ca + VB * self.sa
        drdth = self.m * (-A_ * self.sa + B_ * self.ca)
        X = (self.cx + r * self.cosT) % self.L
        Y = (self.cy + r * self.sinT) % self.L
        ds = np.sqrt(r * r + drdth * drdth) * self.dth
        # wall velocity of a radial mode: radial only (markers keep their angle)
        return X, Y, rt * self.cosT, rt * self.sinT, ds

    def loads(self, ds):
        """Generalized fluid force on each cell's a and b modes (force of fluid on wall)."""
        Fn = self.ib.Fl_x * self.cosT + self.ib.Fl_y * self.sinT
        q = -Fn * ds
        self.Qa[:] = (q * self.ca).reshape(self.nc, self.nm).sum(1)
        self.Qb[:] = (q * self.sa).reshape(self.nc, self.nm).sum(1)


def run_impedance(arr, driven_mode="a", periods=12, avg=4, every=10):
    """Center cell driven in mode a (or b), all others held still."""
    T = arr.period
    nsteps = periods * T
    rec_t, rec_qa, rec_qb = [], [], []

    def hook(t, s):
        tm = t + 0.5
        p, v = arr.drive(tm)
        arr.a[:] = arr.b[:] = arr.va[:] = arr.vb[:] = 0.0
        if driven_mode == "a":
            arr.a[arr.k0], arr.va[arr.k0] = p, v
        else:
            arr.b[arr.k0], arr.vb[arr.k0] = p, v
        X, Y, U, V, ds = arr.markers(arr.a, arr.b, arr.va, arr.vb)
        arr.ib.force(s, X, Y, U, V, ds)
        arr.loads(ds)

    t0 = time.perf_counter()
    for k in range(nsteps):
        arr.sim.step(hook)
        if k >= (periods - avg) * T:
            rec_t.append(k + 0.5)
            rec_qa.append(arr.Qa.copy())
            rec_qb.append(arr.Qb.copy())
        if k % (2 * T) == 0:
            print(f"  impedance {driven_mode}: period {k // T}/{periods}, {time.perf_counter() - t0:.0f} s", flush=True)
    t = np.array(rec_t)
    e = np.exp(-1j * arr.w * t)[:, None]                   # complex amplitude: x(t) = Re(X e^{iwt})
    Qa = 2 * np.mean(np.array(rec_qa) * e, axis=0)
    Qb = 2 * np.mean(np.array(rec_qb) * e, axis=0)
    # drive displacement: A sin(wt) = Re(-i A e^{iwt})
    X0 = -1j * arr.A
    _, ux, uy = arr.sim.fields()
    return dict(Ga=Qa / X0, Gb=Qb / X0, Qa=Qa, Qb=Qb, ux=ux.copy(), uy=uy.copy())


def run_free(arr, M, K, Mhat, periods=40, every=10):
    """Center cell driven in mode a; others are masses on springs, moved by the fluid.
    Added-mass stabilization: (M + Mhat) acc_new = Q - K a + Mhat acc_old."""
    T = arr.period
    nsteps = periods * T
    free = np.ones(arr.nc, bool)
    free[arr.k0] = False
    acc_a = np.zeros(arr.nc)
    acc_b = np.zeros(arr.nc)
    rec = dict(t=[], a=[], b=[])
    snaps = {}

    def hook(t, s):
        tm = t + 0.5
        p, v = arr.drive(tm)
        arr.a[arr.k0], arr.va[arr.k0], arr.b[arr.k0], arr.vb[arr.k0] = p, v, 0.0, 0.0
        # positions at t + 1/2 from the leapfrog state (a at t, v at t + 1/2)
        ah = arr.a + 0.5 * arr.va
        bh = arr.b + 0.5 * arr.vb
        ah[arr.k0], bh[arr.k0] = p, 0.0
        X, Y, U, V, ds = arr.markers(ah, bh, arr.va, arr.vb)
        arr.ib.force(s, X, Y, U, V, ds)
        arr.loads(ds)
        # update free cells (unit time step)
        na = (arr.Qa - K * ah + Mhat * acc_a) / (M + Mhat)
        nb = (arr.Qb - K * bh + Mhat * acc_b) / (M + Mhat)
        acc_a[:] = na
        acc_b[:] = nb
        arr.va[free] += na[free]
        arr.vb[free] += nb[free]
        arr.a[free] += arr.va[free]
        arr.b[free] += arr.vb[free]

    t0 = time.perf_counter()
    for k in range(nsteps):
        arr.sim.step(hook)
        if k % every == 0:
            rec["t"].append(k + 1.0)
            a = arr.a.copy()
            a[arr.k0] = arr.drive(k + 1.0)[0]
            rec["a"].append(a)
            rec["b"].append(arr.b.copy())
        if k == nsteps - T - 1:                            # w(k + 1) = 2 pi (periods - 1): a = 0, driven wall fastest outward
            _, ux, uy = arr.sim.fields()
            snaps["q_u"], snaps["q_v"] = ux.copy(), uy.copy()
            snaps["q_a"], snaps["q_b"] = arr.a.copy(), arr.b.copy()
        if k % (4 * T) == 0:
            print(f"  free: period {k // T}/{periods}, {time.perf_counter() - t0:.0f} s, "
                  f"max |a| others {np.abs(np.where(free, arr.a, 0)).max():.3f}", flush=True)
    return dict(t=np.array(rec["t"]), a=np.array(rec["a"]), b=np.array(rec["b"]), **snaps)


def params(arr):
    return dict(n=arr.n, D=arr.D, R=arr.R, A=arr.A, m=arr.m, L=arr.L, w=arr.w, period=arr.period,
                nu=arr.sim.nu, tau=arr.sim.tau, delta=np.sqrt(2 * arr.sim.nu / arr.w), centers=arr.centers,
                ij=arr.ij, k0=arr.k0, nm=arr.nm)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("what", nargs="?", choices=["impedance", "free"])
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--periods", type=int, default=40)
    p.add_argument("--mass", type=float, default=1.0, help="cell mass / added mass of a lone cell")
    p.add_argument("--tau", type=float, default=0.65)
    p.add_argument("--R", type=float, default=20.0, help="cell radius; spacing 2.4 R, amplitude 0.1 R")
    p.add_argument("--modes", default="ab", help="impedance: which modes to drive")
    p.add_argument("--tag", default="")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    tag = a.tag
    if a.what == "impedance":
        for mode in a.modes:
            arr = Array(n=a.n, tau=a.tau, R=a.R, D=int(round(2.4 * a.R)), A=0.1 * a.R)
            t0 = time.perf_counter()
            out = run_impedance(arr, mode)
            np.savez_compressed(os.path.join(HERE, f"impedance_{mode}{tag}.npz"), **out, **params(arr))
            print(mode, round(time.perf_counter() - t0), "s", flush=True)
    elif a.what == "free":
        imp = np.load(os.path.join(HERE, "impedance_a.npz"))
        arr = Array(n=a.n, tau=a.tau)
        G = complex(imp["Ga"][imp["k0"]])                    # self coupling: Q = G x
        w = arr.w
        madd = G.real / w ** 2                               # Q = (madd w^2 - i w c) x
        M = a.mass * madd
        K = (M + madd) * w ** 2                              # lone cell resonates at w
        print(f"added mass {madd:.1f}, damping c = {-G.imag / w:.2f}, M = {M:.1f}, K = {K:.4f}", flush=True)
        t0 = time.perf_counter()
        out = run_free(arr, M, K, Mhat=0.8 * madd, periods=a.periods)
        np.savez_compressed(os.path.join(HERE, f"free{tag}.npz"), **out, **params(arr), M=M, K=K, madd=madd, mass=a.mass,
                            c=-G.imag / w)
        print("free", round(time.perf_counter() - t0), "s", flush=True)
    if a.plot:
        import analyze
        analyze.main()


if __name__ == "__main__":
    main()
