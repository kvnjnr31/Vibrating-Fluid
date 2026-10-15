"""Experiment 11 (option A): Maxwell's vortex cells and idle wheels as a lattice without loss.

    python experiments/11_maxwell_lattice/lattice.py           # checks, figure and movie, a few minutes

The medium (Maxwell 1861-62, Parts II and III, idealized):
  * Cells on a square lattice, spacing D = 1, radius R = D/2 (neighbors touch). Each cell turns as a
    rigid body: angle theta, angular speed Omega, moment of inertia I.
  * One idle particle sits at each contact between two neighboring cells. It rolls between the two
    surfaces, so it moves at the mean of the two surface speeds (Maxwell, eq. 27).
  * In an insulator the particles cannot travel; they are held in place elastically (Part III). The
    rolling then strains the cell substance at the contact: d = rolling displacement, spring k,
    tangential force F = k d. The strain stores energy; nothing is lost.
  * In a conductor the particles can slip through the medium against friction: d relaxes at rate
    1 / t_c, a flow of particles = current.

Equations, cell (i, j), horizontal contact h between (i, j) and (i+1, j), vertical contact v between
(i, j) and (i, j+1):
    d_h' = (R/2) (Omega_ij - Omega_i+1,j) - d_h / t_c,     F_h = k d_h
    d_v' = (R/2) (Omega_i,j+1 - Omega_ij) - d_v / t_c,     F_v = k d_v
    I Omega_ij' = (R/2) (F_h[i-1,j] - F_h[i,j] + F_v[i,j] - F_v[i,j-1])
These are Maxwell's equations for a 2D wave (H along z, E in the plane) on a staggered (Yee) grid:
    H_z = Omega,  E_y = F_h,  E_x = F_v,  mu = 2 I / R,  1/epsilon = k R / 2,  sigma = epsilon / t_c
    mu dH/dt = -(curl E)_z,    epsilon dE/dt = curl(H z) - sigma E
so the wave speed is c = (R/2) sqrt(k / I). Here I = 1, k = 16, R = 1/2: c = 1 cell per unit time.
The elastic displacement of the particles, d = F / k, is Maxwell's electric displacement.

Runs:
  point     one cell turned back and forth at the center; absorbing border (6 cells, friction).
  slab      a plane wave from the left onto a conductor slab; periodic top to bottom.
  closed    an initial twist in a closed box, no border: total energy must stay constant.
  dispersion plane waves at several frequencies: measured wavenumber against the lattice formula.
Output: results.json, maxwell_lattice_checks.png, maxwell_lattice.mp4.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R, I_, K = 0.5, 1.0, 16.0
C = (R / 2) * np.sqrt(K / I_)                              # wave speed, cells per unit time


class Lattice:
    def __init__(self, nx, ny, dt=0.05, periodic_y=False):
        self.nx, self.ny, self.dt, self.py = nx, ny, dt, periodic_y
        self.Om = np.zeros((nx, ny))
        self.th = np.zeros((nx, ny))
        self.dh = np.zeros((nx - 1, ny))                    # contacts between (i,j) and (i+1,j)
        self.dv = np.zeros((nx, ny if periodic_y else ny - 1))
        self.tch = np.full(self.dh.shape, np.inf)          # particle slip time (inf = insulator)
        self.tcv = np.full(self.dv.shape, np.inf)
        self.drag = np.zeros((nx, ny))                     # friction on cells (absorbing border only)
        self.t = 0.0

    def sponge(self, width, strength=0.6):
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        fx = np.clip((width - np.minimum(x, self.nx - 1 - x)) / width, 0, 1)
        fy = np.zeros(self.ny) if self.py else np.clip((width - np.minimum(y, self.ny - 1 - y)) / width, 0, 1)
        self.drag = strength * np.maximum(fx[:, None], fy[None, :]) ** 2

    def torque(self):
        Fh, Fv = K * self.dh, K * self.dv
        tq = np.zeros_like(self.Om)
        tq[1:, :] += Fh                                      # contact on the left of the cell
        tq[:-1, :] -= Fh                                     # contact on the right
        if self.py:
            tq += Fv - np.roll(Fv, 1, axis=1)               # above minus below
        else:
            tq[:, :-1] += Fv
            tq[:, 1:] -= Fv
        return (R / 2) * tq

    def step(self, drive=None):
        dt = self.dt
        # contacts (E) from the cells' turning (H); particle slip in conductors
        rh = (R / 2) * (self.Om[:-1, :] - self.Om[1:, :])
        rv = (R / 2) * ((np.roll(self.Om, -1, axis=1) - self.Om) if self.py else (self.Om[:, 1:] - self.Om[:, :-1]))
        self.dh = (self.dh * (1 - 0.5 * dt / self.tch) + dt * rh) / (1 + 0.5 * dt / self.tch)
        self.dv = (self.dv * (1 - 0.5 * dt / self.tcv) + dt * rv) / (1 + 0.5 * dt / self.tcv)
        # cells (H) from the contact forces
        self.Om += dt * self.torque() / I_
        self.Om /= (1 + dt * self.drag)
        self.t += dt
        if drive is not None:
            drive(self)
        self.th += dt * self.Om

    def energy(self):
        return 0.5 * I_ * np.sum(self.Om ** 2) + 0.5 * K * (np.sum(self.dh ** 2) + np.sum(self.dv ** 2))

    def shear(self):
        """Symmetric shear stress in each cell from the tangential pulls at its four contacts
        (drawn as the cell's oval distortion, the diagonal mode)."""
        Fh, Fv = K * self.dh, K * self.dv
        syx = np.zeros_like(self.Om)
        sxy = np.zeros_like(self.Om)
        syx[:-1, :] -= Fh / 4
        syx[1:, :] += Fh / 4
        if self.py:
            sxy += (np.roll(Fv, 1, axis=1) - Fv) / 4
        else:
            sxy[:, :-1] -= Fv / 4
            sxy[:, 1:] += Fv / 4
        return 0.5 * (sxy + syx)


# ------------------------------------------------------------------------------------------------
def run_point(n=81, w=2 * np.pi / 12.0, T=90.0, frame_dt=0.25, amp=1.0):
    L = Lattice(n, n)
    L.sponge(22, strength=0.25)
    c0 = n // 2

    def drive(lat):
        lat.Om[c0, c0] = amp * np.sin(w * lat.t) * min(1.0, lat.t / 8.0)

    return record(L, drive, T, frame_dt)


def run_slab(nx=96, ny=8, w=2 * np.pi / 12.0, T=90.0, frame_dt=0.25, slab=(54, 66), t_c=1.5, amp=1.0):
    L = Lattice(nx, ny, periodic_y=True)
    L.sponge(12, strength=0.3)
    L.drag[:14, :] = 0.0                                    # source side: no border on the left
    L.tch[slab[0]:slab[1], :] = t_c
    L.tcv[slab[0]:slab[1] + 1, :] = t_c

    def drive(lat):
        lat.Om[2, :] = amp * np.sin(w * lat.t) * min(1.0, lat.t / 8.0)

    out = record(L, drive, T, frame_dt)
    out.update(slab=np.array(slab), t_c=t_c)
    return out


def record(L, drive, T, frame_dt):
    nsteps = int(round(T / L.dt))
    every = int(round(frame_dt / L.dt))
    fr = dict(t=[], Om=[], th=[], dh=[], dv=[], shear=[])
    for s in range(nsteps):
        L.step(drive)
        if s % every == 0:
            fr["t"].append(L.t)
            fr["Om"].append(L.Om.copy())
            fr["th"].append(L.th.copy())
            fr["dh"].append(L.dh.copy())
            fr["dv"].append(L.dv.copy())
            fr["shear"].append(L.shear())
    out = {k: np.array(v) for k, v in fr.items()}
    out.update(nx=L.nx, ny=L.ny, periodic_y=L.py)
    return out


def run_closed(n=30, T=200.0):
    L = Lattice(n, n)
    x = np.arange(n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    L.Om[:] = np.exp(-((X - n / 2) ** 2 + (Y - n / 2) ** 2) / 8.0)
    E0 = L.energy()
    ts, Es = [], []
    for s in range(int(T / L.dt)):
        L.step()
        if s % 20 == 0:
            ts.append(L.t)
            Es.append(L.energy() / E0)
    return np.array(ts), np.array(Es)


def run_dispersion(periods=(5, 6, 8, 10, 14, 20, 30)):
    """Plane waves: drive a column, measure the wavenumber from the phase along x."""
    ks, ws = [], []
    for P in periods:
        w = 2 * np.pi / P * C                               # P = free-space wavelength in cells
        nx = 120
        L = Lattice(nx, 2, periodic_y=True)
        L.sponge(15)
        L.drag[:16, :] = 0.0
        T = 0.0

        def drive(lat):
            lat.Om[2, :] = np.sin(w * lat.t) * min(1.0, lat.t / 10.0)

        recs, tt = [], []
        while L.t < 60.0 + 2 * nx / C:
            L.step(drive)
            if L.t > 2 * nx / C - 10:
                recs.append(L.Om[:, 0].copy())
                tt.append(L.t)
        recs, tt = np.array(recs), np.array(tt)
        Z = 2 * np.mean(recs * np.exp(-1j * w * tt)[:, None], axis=0)
        xs = np.arange(10, 70)
        ph = np.unwrap(np.angle(Z[xs]))
        k = -np.polyfit(xs, ph, 1)[0]
        ks.append(k)
        ws.append(w)
    return np.array(ks), np.array(ws)


def plane_snapshot(slab):
    """E_y and H_z along x at one moment, in the insulating part before the slab."""
    i = len(slab["t"]) - 1
    x_h = np.arange(slab["nx"])
    x_e = np.arange(slab["nx"] - 1) + 0.5
    H = slab["Om"][i][:, 0]
    E = K * slab["dh"][i][:, 0]
    Ex = K * slab["dv"][i][:, 0]
    return x_h, H, x_e, E, Ex


# ------------------------------------------------------------------------------------------------
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                         "font.size": 14, "axes.titlelocation": "left", "axes.titlesize": 15})
    return plt


def checks_figure(path, closed, disp, slab, w_slab, res):
    plt = style()
    fig, ax = plt.subplots(2, 2, figsize=(16, 11))
    t, E = closed
    ax[0, 0].plot(t, E, color="k", lw=2)
    ax[0, 0].set_ylim(0.99, 1.01)
    ax[0, 0].set_xlabel("time")
    ax[0, 0].set_ylabel("total energy / initial")
    ax[0, 0].set_title(f"(A) Closed box: no loss (energy swings within {100 * res['energy_drift']:.1f}%, no trend)")
    ks, ws = disp
    kk = np.linspace(0, np.pi, 200)
    ax[0, 1].plot(kk, 2 * C * np.sin(kk / 2), color="0.55", lw=3, label=r"lattice: $\omega = 2c\,\sin(kD/2)$")
    ax[0, 1].plot(kk, C * kk, color="0.3", lw=1.5, ls=(0, (5, 3)), label=r"Maxwell, free space: $\omega = c k$")
    ax[0, 1].plot(ks, ws, "o", color="k", mfc="white", ms=8, label="simulation")
    ax[0, 1].set_xlabel("wavenumber $k D$")
    ax[0, 1].set_ylabel(r"frequency $\omega$")
    ax[0, 1].set_title("(B) Dispersion: light-like at long wavelengths")
    ax[0, 1].legend(frameon=False, fontsize=12)
    s0, s1 = slab["slab"]
    zH = np.sqrt((2 * I_ / R) / (2 / (K * R)))             # wave impedance sqrt(mu/epsilon)
    tt = slab["t"]
    mm = tt > tt[-1] - 40
    e = np.exp(-1j * w_slab * tt[mm])
    Hc = 2 * np.mean(slab["Om"][mm][:, :, 0] * e[:, None], axis=0)
    Ec = 2 * np.mean(K * slab["dh"][mm][:, :, 0] * e[:, None], axis=0)
    Exc = 2 * np.mean(K * slab["dv"][mm][:, :, 0] * e[:, None], axis=0)
    x_h, x_e = np.arange(slab["nx"]), np.arange(slab["nx"] - 1) + 0.5
    m, me = x_h < s0, x_e < s0
    ph = np.exp(1j * 0.3)
    ax[1, 0].plot(x_h[m], (Hc * ph).real[m], color="k", lw=2, label=r"$H_z$ = cell turning speed $\Omega$")
    ax[1, 0].plot(x_e[me], (Ec * ph).real[me] / zH, color="0.55", lw=2.5, ls=(0, (5, 3)),
                  label=r"$E_y / Z$ = contact pull, scaled by the wave impedance")
    ax[1, 0].plot(x_h[m], (Exc * ph).real[m] / zH, color="0.7", lw=1.5, label=r"$E_x / Z$ (along the wave)")
    Hm = 0.5 * (Hc[:-1] + Hc[1:])
    sel = (x_e > 8) & (x_e < s0 - 4)
    res["E_over_ZH"] = float(np.mean(np.abs(Ec[sel] / zH) / np.abs(Hm[sel])))
    res["E_H_phase_over_pi"] = float(np.mean(np.angle(Ec[sel] / Hm[sel])) / np.pi)
    ax[1, 0].axvspan(s0, s1, color="0.88", zorder=0)
    ax[1, 0].axvspan(slab["nx"] - 12, slab["nx"], color="0.95", zorder=0)
    ax[1, 0].set_xlim(0, slab["nx"])
    ax[1, 0].set_xlabel("x (cells)")
    ax[1, 0].set_title(f"(C) Plane wave: E across the wave, in step with H (E/ZH = {res['E_over_ZH']:.2f})")
    ax[1, 0].legend(frameon=False, fontsize=11, loc="lower left")
    amp = np.abs(slab["Om"][-80:, :, 0]).max(axis=0)
    ax[1, 1].semilogy(np.arange(slab["nx"]), np.maximum(amp, 1e-6), color="k", lw=2, label="|H| envelope")
    ax[1, 1].axvspan(s0, s1, color="0.88", zorder=0, label="conductor (particles slip)")
    x_in = np.arange(s0, s1)
    mu, eps = 2 * I_ / R, 2 / (K * R)
    sig = eps / slab["t_c"]
    kc = w_slab * np.sqrt(mu * eps + 0j) * np.sqrt(1 - 1j * sig / (eps * w_slab))
    ax[1, 1].semilogy(x_in, amp[s0] * np.exp(-abs(kc.imag) * (x_in - s0)), color="0.55", lw=2.5,
                      ls=(0, (5, 3)), label=f"Maxwell, conductor: decay length {1 / abs(kc.imag):.1f} cells")
    ax[1, 1].set_xlim(0, slab["nx"] - 12)
    ax[1, 1].set_ylim(1e-4, 3)
    ax[1, 1].set_xlabel("x (cells)")
    ax[1, 1].set_title("(D) Conductor: reflection in front, skin decay inside")
    ax[1, 1].legend(frameon=False, fontsize=11, loc="lower left")
    res["skin_theory"] = float(1 / abs(kc.imag))
    seg = np.log(np.maximum(amp[s0 + 1:s0 + 6], 1e-12))
    res["skin_measured"] = float(-1 / np.polyfit(np.arange(5), seg, 1)[0])
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def movie(path, point, slab, fps=24, exag=0.22):
    plt = style()
    from matplotlib.animation import FFMpegWriter
    from matplotlib.collections import LineCollection
    th = np.linspace(0, 2 * np.pi, 40)
    fig = plt.figure(figsize=(16, 9))
    axA = fig.add_axes([0.03, 0.08, 0.44, 0.8])
    axB = fig.add_axes([0.5, 0.42, 0.48, 0.46])
    axC = fig.add_axes([0.5, 0.08, 0.48, 0.26])

    def cells(ax, nx, ny, x0=0, y0=0, w=None):
        cx, cy = np.meshgrid(np.arange(nx) + x0, np.arange(ny) + y0, indexing="ij")
        lc = LineCollection([], colors="k", linewidths=0.8, zorder=4)
        ax.add_collection(lc)
        mk = ax.plot([], [], "-", color="k", lw=0.8, zorder=5)[0]
        pt = ax.plot([], [], "o", color="0.35", ms=1.8, zorder=6)[0]
        return cx, cy, lc, mk, pt

    def scales(d):
        return dict(sh=np.percentile(np.abs(d["shear"]), 90) + 1e-12,
                    th=np.percentile(np.abs(d["th"]), 90) + 1e-12,
                    dd=max(np.percentile(np.abs(d["dh"]), 90), np.percentile(np.abs(d["dv"]), 90)) + 1e-12)

    def draw(ax_art, d, k, sc, ny_show=None):
        cx, cy, lc, mk, pt, im = ax_art
        Om, thv, sh = d["Om"][k], d["th"][k], d["shear"][k]
        ny = Om.shape[1] if ny_show is None else ny_show
        Om, thv, sh = Om[:, :ny], thv[:, :ny], sh[:, :ny]
        im.set_data(Om.T)
        bshape = exag * np.tanh(sh / sc["sh"])              # oval (diagonal mode), compressed so far cells show
        segs = []
        for i in range(Om.shape[0]):
            for j in range(ny):
                r = 0.42 * (1 + bshape[i, j] * np.sin(2 * th))
                segs.append(np.c_[cx[i, j] + r * np.cos(th), cy[i, j] + r * np.sin(th)])
        lc.set_segments(segs)
        ang = np.pi / 2 + 0.8 * np.tanh(thv / sc["th"])       # radius line: twist, compressed
        xs = np.c_[cx.ravel(), cx.ravel() + 0.38 * np.cos(ang.ravel()), np.full(cx.size, np.nan)].ravel()
        ys = np.c_[cy.ravel(), cy.ravel() + 0.38 * np.sin(ang.ravel()), np.full(cy.size, np.nan)].ravel()
        mk.set_data(xs, ys)
        dh = d["dh"][k][:, :ny]
        px = (cx[:-1, :] + 0.5).ravel()
        pyv = (cy[:-1, :] + 0.3 * np.tanh(dh / sc["dd"])).ravel()
        dv = d["dv"][k][:, :ny - 1]
        qx = (cx[:, :ny - 1] + 0.3 * np.tanh(dv / sc["dd"])).ravel()
        qy = (cy[:, :ny - 1] + 0.5).ravel()
        pt.set_data(np.r_[px, qx], np.r_[pyv, qy])

    # panel A: point source, central window
    n = point["nx"]
    win = 10
    c0 = n // 2
    sl = slice(c0 - win, c0 + win + 1)
    sub = {k: (v[:, sl, sl] if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[1] == n and v.shape[2] == n
               else v) for k, v in point.items()}
    sub["dh"] = point["dh"][:, sl.start:sl.stop - 1, sl]
    sub["dv"] = point["dv"][:, sl, sl.start:sl.stop - 1]
    HmA = np.abs(point["Om"]).max()
    imA = axA.imshow(sub["Om"][0].T, origin="lower", extent=(-0.5, 2 * win + 0.5, -0.5, 2 * win + 0.5), cmap="Greys",
                     vmin=-0.3 * HmA, vmax=0.3 * HmA, zorder=1, interpolation="bicubic")
    artA = (*cells(axA, 2 * win + 1, 2 * win + 1), imA)
    axA.set_xlim(-0.5, 2 * win + 0.5)
    axA.set_ylim(-0.5, 2 * win + 0.5)
    axA.set_aspect("equal")
    axA.set_xticks([])
    axA.set_yticks([])
    axA.set_title("(A) One cell twisted back and forth: a circular wave")
    # panel B: plane wave and conductor
    nxs, nys = 72, 4
    HmB = np.abs(slab["Om"]).max()
    imB = axB.imshow(slab["Om"][0][:nxs, :nys].T, origin="lower", extent=(-0.5, nxs - 0.5, -0.5, nys - 0.5), cmap="Greys",
                     vmin=-HmB, vmax=HmB, zorder=1, aspect="equal", interpolation="bicubic")
    artB = (*cells(axB, nxs, nys), imB)
    s0, s1 = slab["slab"]
    axB.axvspan(s0 - 0.5, s1 - 0.5, color="0.6", alpha=0.25, zorder=2)
    axB.text((s0 + s1) / 2 - 0.5, nys - 0.1, "conductor", ha="center", va="bottom", fontsize=12)
    axB.set_xlim(-0.5, nxs - 0.5)
    axB.set_ylim(-0.5, nys - 0.5)
    axB.set_yticks([])
    axB.set_title("(B) Plane wave from the left onto a conductor")
    xh = np.arange(nxs)
    lH, = axC.plot([], [], color="k", lw=2, label=r"$H_z$: cell turning")
    lE, = axC.plot([], [], color="0.55", lw=2, ls=(0, (5, 3)), label=r"$E_y$: contact pull (scaled)")
    axC.axvspan(s0 - 0.5, s1 - 0.5, color="0.6", alpha=0.25)
    axC.set_xlim(-0.5, nxs - 0.5)
    axC.set_ylim(-1.3 * HmB, 1.3 * HmB)
    axC.set_xlabel("x (cells)")
    axC.legend(frameon=False, fontsize=11, loc="lower left", ncol=2)
    axC.set_title("(C) Fields along the row", fontsize=14)
    zH = np.sqrt((2 * I_ / R) / (2 / (K * R)))
    stamp = fig.text(0.5, 0.95, "", ha="center", fontsize=16)
    fig.text(0.5, 0.015, "Shells: Maxwell's cells, twisting (radius line) and sheared into ovals by the pulls at their "
             "contacts (twist, oval and particle motion drawn larger and compressed, so far cells show). Dots: idle particles, moved by their elastic displacement (E). "
             "Shading: turning speed (H).", ha="center", fontsize=11.5,
             color="0.3")
    scA = scales(sub)
    scB = scales(dict(shear=slab["shear"][:, :nxs, :nys], th=slab["th"][:, :nxs, :nys], dh=slab["dh"][:, :nxs - 1, :nys],
                      dv=slab["dv"][:, :nxs, :nys]))
    nf = min(len(point["t"]), len(slab["t"]))
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "24"])
    with writer.saving(fig, path, dpi=100):
        for k in range(0, nf):
            draw(artA, sub, k, scA)
            slab_k = dict(Om=slab["Om"][:, :nxs], th=slab["th"][:, :nxs], shear=slab["shear"][:, :nxs],
                          dh=slab["dh"][:, :nxs - 1], dv=slab["dv"][:, :nxs])
            draw(artB, slab_k, k, scB, ny_show=nys)
            lH.set_data(xh, slab["Om"][k][:nxs, 0])
            lE.set_data(np.arange(nxs - 1) + 0.5, K * slab["dh"][k][:nxs - 1, 0] / zH)
            stamp.set_text(f"Maxwell's vortex cells and idle wheels, no loss: t = {point['t'][k]:5.1f} "
                           f"(wave period 12)")
            writer.grab_frame()


def main():
    res = {"wave_speed_c": C}
    t, E = run_closed()
    res["energy_drift"] = float(np.abs(E - 1).max())
    ks, ws = run_dispersion()
    res["dispersion"] = dict(k=ks.tolist(), w=ws.tolist(),
                             w_lattice=(2 * C * np.sin(ks / 2)).tolist())
    res["dispersion_max_error"] = float(np.max(np.abs(ws - 2 * C * np.sin(ks / 2)) / ws))
    w = 2 * np.pi / 12.0
    point = run_point(w=w)
    tt, Om = point["t"], point["Om"]
    n = Om.shape[1]
    c0 = n // 2
    mm = tt > tt[-1] - 24
    Zp = 2 * np.mean(Om[mm] * np.exp(-1j * w * tt[mm])[:, None, None], axis=0)
    X, Y = np.meshgrid(np.arange(n) - c0, np.arange(n) - c0, indexing="ij")
    rr, ang = np.hypot(X, Y), np.arctan2(Y, X)
    circ = {}
    for r0 in (4, 8, 12, 16):
        ring = np.abs(rr - r0) < 0.5
        on_axis = ring & ((np.abs(ang) % (np.pi / 2) < 0.2) | (np.abs(ang) % (np.pi / 2) > np.pi / 2 - 0.2))
        diag = ring & (np.abs(np.abs(ang) % (np.pi / 2) - np.pi / 4) < 0.2)
        circ[r0] = dict(amp_axis=float(np.abs(Zp[on_axis]).mean()), amp_diag=float(np.abs(Zp[diag]).mean()),
                        phase_axis_over_pi=float(np.angle(Zp[on_axis].mean()) / np.pi),
                        phase_diag_over_pi=float(np.angle(Zp[diag].mean()) / np.pi))
    res["point_source_rings"] = circ
    slab = run_slab(w=w)
    np.savez_compressed(os.path.join(HERE, "run_point.npz"), **point)
    np.savez_compressed(os.path.join(HERE, "run_slab.npz"), **slab)
    checks_figure(os.path.join(HERE, "maxwell_lattice_checks.png"), (t, E), (ks, ws), slab, w, res)
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "dispersion"}, indent=1), flush=True)
    movie(os.path.join(HERE, "maxwell_lattice.mp4"), point, slab)
    print("done", flush=True)


if __name__ == "__main__":
    main()
