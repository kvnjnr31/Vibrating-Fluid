"""Experiment 10: waves along a chain of free vibrating cells (three routes toward lasting waves).

    python experiments/10_wave_chain/chain.py            # impedance + 3 free runs, about 30 min on 2 cores
    python experiments/10_wave_chain/chain.py --plot     # figures and movie from saved runs

Geometry. N = 10 cells in a row along x, spacing D = 2.4 R, in a box N D x D, periodic both ways.
Periodic in y with one row means every cell has in-phase copies above and below: the runs follow
plane waves along x. Cell 0 is driven; the wave leaves it to the right and to the left (through the
periodic edge) and the two meet at cell 5.

Each cell: r = R + a(t) cos 2 theta + b(t) sin 2 theta. Mode a bulges toward the x neighbors, mode b
toward the diagonals. Free cells: each mode is a mass on a spring, as in experiment 8,
    M a'' + K a = Q_a + gain,     Q_a = fluid force on the mode.
M = added mass of a lone cell; K tunes a lone cell (neighbors held) to the drive frequency.

Route 1, thin Stokes layers: R = 40, so delta / R = 0.15 (experiment 8 had 0.3). Every run uses it.
Route 2, self-driven cells: each free mode gets an energy supply that cancels the cell's own fluid
         loss and levels off at amplitude A (van der Pol type):  gain = g c (1 - (a^2 + b^2)/A^2) a'.
         g = 1: at small amplitude the supply equals the loss of a lone cell. A stand-in for the
         tension feedback of the self-regulating conveyor.
Route 3, spin: cell 0 is driven with its nodes moving around it, a = A sin wt, b = -A cos wt (the two
         modes a quarter cycle apart). Free cells are self-driven in both modes. Spin of a cell:
         S = (a b' - b a') / (w A^2), +1 for nodes moving counterclockwise at full amplitude.

Runs: impedance (a and b, neighbors held: tuning and coupling), passive (free cells, no gain),
active (route 2), spin (route 3). tau = 0.65 as in experiment 8 (period 2262 steps, near-
incompressible), A = 0.1 R.
Output: impedance_<mode>.npz, run_<case>.npz, results.json, wave_chain.mp4, em_analog.png.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "08_driven_cell"))
from lbm import LBM  # noqa: E402
from driven import FastIB  # noqa: E402


class Chain:
    def __init__(self, N=10, R=40.0, tau=0.65, delta=6.0, m=2):
        self.N, self.R, self.m = N, R, m
        self.D = D = int(round(2.4 * R))
        self.A = 0.1 * R
        self.Lx, self.Ly = N * D, D
        self.sim = LBM(self.Lx, self.Ly, tau)
        w = 2 * self.sim.nu / delta ** 2
        self.period = int(round(2 * np.pi / w))
        self.w = 2 * np.pi / self.period
        self.nm = nm = int(np.ceil(2 * np.pi * R))
        th = 2 * np.pi * np.arange(nm) / nm
        self.centers = np.c_[(np.arange(N) + 0.5) * D + 0.25, np.full(N, 0.5 * D + 0.25)]
        self.TH = np.tile(th, N)
        self.cx = np.repeat(self.centers[:, 0], nm)
        self.cy = np.repeat(self.centers[:, 1], nm)
        self.ca, self.sa = np.cos(m * self.TH), np.sin(m * self.TH)
        self.cosT, self.sinT = np.cos(self.TH), np.sin(self.TH)
        self.dth = 2 * np.pi / nm
        self.ib = FastIB(self.TH.size)
        z = lambda: np.zeros(N)  # noqa: E731
        self.a, self.b, self.va, self.vb, self.Qa, self.Qb = z(), z(), z(), z(), z(), z()

    def markers(self, a, b, va, vb):
        A_, B_ = np.repeat(a, self.nm), np.repeat(b, self.nm)
        VA, VB = np.repeat(va, self.nm), np.repeat(vb, self.nm)
        r = self.R + A_ * self.ca + B_ * self.sa
        rt = VA * self.ca + VB * self.sa
        drdth = self.m * (-A_ * self.sa + B_ * self.ca)
        X = (self.cx + r * self.cosT) % self.Lx
        Y = (self.cy + r * self.sinT) % self.Ly
        ds = np.sqrt(r * r + drdth * drdth) * self.dth
        return X, Y, rt * self.cosT, rt * self.sinT, ds

    def loads(self, ds):
        q = -(self.ib.Fl_x * self.cosT + self.ib.Fl_y * self.sinT) * ds
        self.Qa[:] = (q * self.ca).reshape(self.N, self.nm).sum(1)
        self.Qb[:] = (q * self.sa).reshape(self.N, self.nm).sum(1)

    def params(self):
        return dict(N=self.N, R=self.R, D=self.D, A=self.A, m=self.m, Lx=self.Lx, Ly=self.Ly, w=self.w,
                    period=self.period, nu=self.sim.nu, delta=np.sqrt(2 * self.sim.nu / self.w),
                    centers=self.centers)


def impedance(mode, periods=10, avg=4):
    """Cell 0 driven in one mode, the others held: self coupling (added mass, damping) and neighbors."""
    ch = Chain()
    T = ch.period
    rec_t, rec_q = [], []

    def hook(t, s):
        tm = t + 0.5
        ch.a[:] = ch.b[:] = ch.va[:] = ch.vb[:] = 0.0
        p, v = ch.A * np.sin(ch.w * tm), ch.A * ch.w * np.cos(ch.w * tm)
        if mode == "a":
            ch.a[0], ch.va[0] = p, v
        else:
            ch.b[0], ch.vb[0] = p, v
        X, Y, U, V, ds = ch.markers(ch.a, ch.b, ch.va, ch.vb)
        ch.ib.force(s, X, Y, U, V, ds)
        ch.loads(ds)

    for k in range(periods * T):
        ch.sim.step(hook)
        if k >= (periods - avg) * T:
            rec_t.append(k + 0.5)
            rec_q.append((ch.Qa if mode == "a" else ch.Qb).copy())
    e = np.exp(-1j * ch.w * np.array(rec_t))[:, None]
    Q = 2 * np.mean(np.array(rec_q) * e, axis=0)
    G = Q / (-1j * ch.A)                                   # drive A sin wt = Re(-i A e^{iwt})
    return dict(G=G, **ch.params())


def free(case, tun, periods=24, frames_per_period=24, gain=1.0, every=10, down=3):
    """case: passive, active (route 2) or spin (route 3)."""
    ch = Chain()
    T, w, A = ch.period, ch.w, ch.A
    Ma, Ka, ca, Mha = tun["a"]
    Mb, Kb, cb, Mhb = tun["b"]
    acc_a, acc_b = np.zeros(ch.N), np.zeros(ch.N)
    free_ = np.ones(ch.N, bool)
    free_[0] = False
    ga = gain * ca if case in ("active", "spin") else 0.0
    gb = gain * cb if case == "spin" else 0.0
    rec = dict(t=[], a=[], b=[])
    frames = dict(t=[], a=[], b=[], speed=[])
    step_f = T // frames_per_period
    first_frame = (periods - 12) * T                        # last 12 periods go in the movie

    def drive(tm):
        if case == "spin":
            return (A * np.sin(w * tm), A * w * np.cos(w * tm), -A * np.cos(w * tm), A * w * np.sin(w * tm))
        return A * np.sin(w * tm), A * w * np.cos(w * tm), 0.0, 0.0

    def hook(t, s):
        tm = t + 0.5
        pa, va0, pb, vb0 = drive(tm)
        ch.a[0], ch.va[0], ch.b[0], ch.vb[0] = pa, va0, pb, vb0
        ah, bh = ch.a + 0.5 * ch.va, ch.b + 0.5 * ch.vb
        ah[0], bh[0] = pa, pb
        X, Y, U, V, ds = ch.markers(ah, bh, ch.va, ch.vb)
        ch.ib.force(s, X, Y, U, V, ds)
        ch.loads(ds)
        sat = 1.0 - (ah * ah + bh * bh) / (A * A)
        na = (ch.Qa - Ka * ah + ga * sat * ch.va + Mha * acc_a) / (Ma + Mha)
        nb = (ch.Qb - Kb * bh + gb * sat * ch.vb + Mhb * acc_b) / (Mb + Mhb)
        acc_a[:], acc_b[:] = na, nb
        ch.va[free_] += na[free_]
        ch.vb[free_] += nb[free_]
        ch.a[free_] += ch.va[free_]
        ch.b[free_] += ch.vb[free_]

    t0 = time.perf_counter()
    for k in range(periods * T):
        ch.sim.step(hook)
        if k % every == 0:
            a_ = ch.a.copy()
            b_ = ch.b.copy()
            a_[0], _, b_[0], _ = drive(k + 1.0)
            rec["t"].append(k + 1.0)
            rec["a"].append(a_)
            rec["b"].append(b_)
        if k >= first_frame and (k - first_frame) % step_f == 0:
            _, ux, uy = ch.sim.fields()
            a_ = ch.a.copy()
            b_ = ch.b.copy()
            a_[0], _, b_[0], _ = drive(k + 1.0)
            frames["t"].append(k + 1.0)
            frames["a"].append(a_)
            frames["b"].append(b_)
            frames["speed"].append(np.hypot(ux, uy)[::down, ::down].astype(np.float16))
        if k % (4 * T) == 0:
            print(f"  {case}: period {k // T}/{periods}, {time.perf_counter() - t0:.0f} s, "
                  f"|a| max others {np.abs(ch.a[1:]).max():.3f}", flush=True)
    return dict(case=case, t=np.array(rec["t"]), a=np.array(rec["a"]), b=np.array(rec["b"]),
                ft=np.array(frames["t"]), fa=np.array(frames["a"]), fb=np.array(frames["b"]),
                speed=np.array(frames["speed"]), down=down, gain=gain, **ch.params())


def tuning():
    out = {}
    for mode in "ab":
        d = np.load(os.path.join(HERE, f"impedance_{mode}.npz"))
        G0 = complex(d["G"][0])
        w = float(d["w"])
        madd = G0.real / w ** 2                             # Q = (madd w^2 - i w c) x
        c = -G0.imag / w
        M = madd
        K = (M + madd) * w ** 2
        out[mode] = (M, K, c, 0.8 * madd)
    return out


def load(name):
    d = dict(np.load(os.path.join(HERE, f"{name}.npz"), allow_pickle=False))
    return {k: (v.item() if v.ndim == 0 else v) for k, v in d.items()}


def response(d, periods_avg=4):
    """Complex amplitude of a and b of every cell over the last periods (x = Re(X e^{iwt}))."""
    t, w, T = d["t"], d["w"], d["period"]
    m = t > t[-1] - periods_avg * T
    e = np.exp(-1j * w * t[m])[:, None]
    Xa = 2 * np.mean(d["a"][m] * e, axis=0)
    Xb = 2 * np.mean(d["b"][m] * e, axis=0)
    return Xa, Xb


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plot", action="store_true")
    p.add_argument("--cases", nargs="+", default=["passive", "active", "spin"])
    p.add_argument("--periods", type=int, default=24)
    a = p.parse_args()
    res_path = os.path.join(HERE, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    if not a.plot:
        for mode in "ab":
            f = os.path.join(HERE, f"impedance_{mode}.npz")
            if not os.path.exists(f):
                t0 = time.perf_counter()
                np.savez_compressed(f, **impedance(mode))
                print("impedance", mode, round(time.perf_counter() - t0), "s", flush=True)
        tun = tuning()
        res["tuning"] = {k: dict(M=v[0], K=v[1], c=v[2]) for k, v in tun.items()}
        imp = load("impedance_a")
        G = imp["G"]
        res["coupling_a"] = dict(G1_over_wc=[float((G[1] / (imp["w"] * tun["a"][2])).real),
                                             float((G[1] / (imp["w"] * tun["a"][2])).imag)])
        print("tuning", res["tuning"], "G1/(wc)", res["coupling_a"], flush=True)
        for case in a.cases:
            t0 = time.perf_counter()
            out = free(case, tun, periods=a.periods)
            np.savez_compressed(os.path.join(HERE, f"run_{case}.npz"), **out)
            Xa, Xb = response(out)
            res[case] = dict(amp_a=np.abs(Xa / out["A"]).tolist(), phase_a_over_pi=(np.angle(Xa / Xa[0]) / np.pi).tolist(),
                             amp_b=np.abs(Xb / out["A"]).tolist(), seconds=round(time.perf_counter() - t0, 1))
            print(case, {k: np.round(v, 3).tolist() if isinstance(v, list) else v for k, v in res[case].items()},
                  flush=True)
            json.dump(res, open(res_path, "w"), indent=2)
    import plots
    plots.main()


if __name__ == "__main__":
    main()
