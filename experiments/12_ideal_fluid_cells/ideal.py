"""Experiment 12 (option B): oval-mode shells on a hexagonal lattice, coupled only through an ideal fluid.

Each cell is a closed shell of radius R, centre fixed, with two oval modes
    r = R + a cos 2theta + b sin 2theta        (theta from the global x axis)
held by the shell tension (stiffness K per mode). The fluid inside and outside is inviscid and
incompressible and starts at rest, so it has no swirl (Kelvin) and its flow is a potential flow. The
fluid then only adds inertia: its kinetic energy is (1/2) qdot^T M qdot, with M the added-mass matrix,
and the shells obey
    (M + m_s) qddot + K q = f.
Nothing loses energy. The off-diagonal blocks of M are the coupling between cells.

M is computed exactly (to multipole order P) with complex potentials: the outer flow is
w(z) = sum_j sum_n c_jn (R / (z - z_j))^n, and the wall speed is matched mode by mode on every circle.

    python experiments/12_ideal_fluid_cells/ideal.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RHO = 1.0


def hex_lattice(nr, d=1.0):
    """Cells within nr rings of the centre; returns positions (complex) and axial coords. Index 0 = centre."""
    pts = []
    for a in range(-nr, nr + 1):
        for b in range(-nr, nr + 1):
            if abs(a) <= nr and abs(b) <= nr and abs(a + b) <= nr:
                pts.append((max(abs(a), abs(b), abs(a + b)), a, b))
    pts.sort()
    ring = np.array([p[0] for p in pts])
    z = np.array([d * (p[1] + 0.5 * p[2]) + 1j * d * (np.sqrt(3) / 2 * p[2]) for p in pts])
    return z, ring


def outer_mass(z, R, P=8, Q=None):
    """Exterior added-mass matrix (2N x 2N), ordering [a_0, b_0, a_1, b_1, ...]."""
    N = len(z)
    Q = Q or 4 * P + 8
    th = 2 * np.pi * np.arange(Q) / Q
    eith = np.exp(1j * th)
    n = np.arange(1, P + 1)
    # projection of a function on the circle onto cos(n th), sin(n th), n = 1..P  -> (2P, Q)
    proj = np.vstack([np.cos(np.outer(n, th)), np.sin(np.outer(n, th))]) * (2.0 / Q)
    nu = 2 * P * N
    A = np.zeros((nu, nu))       # equations: radial speed modes on each circle
    B = np.zeros((2 * N, nu))    # phi projected on cos2, sin2 on each circle
    for i in range(N):
        zz = z[i] + R * eith                                      # (Q,)
        s = R / (zz[:, None] - z[None, :])                        # (Q, N)
        sn = s[:, :, None] ** n[None, None, :]                    # (Q, N, P): (R/(z-zj))^n
        dw = -(n / R)[None, None, :] * sn * s[:, :, None]         # d/dz of (R/(z-zj))^n
        ur = dw * eith[:, None, None]                             # u_r = Re(dw/dz e^{i th}) per unit c
        # unknown ordering per cell j: [Re c_1..Re c_P, Im c_1..Im c_P]
        cols_re = ur.real.reshape(Q, N * P)
        cols_im = (-ur.imag).reshape(Q, N * P)
        phi_re = sn.real.reshape(Q, N * P)
        phi_im = (-sn.imag).reshape(Q, N * P)
        Ui = np.empty((Q, nu)); Fi = np.empty((Q, nu))
        Ui.reshape(Q, N, 2, P)[:, :, 0, :] = cols_re.reshape(Q, N, P)
        Ui.reshape(Q, N, 2, P)[:, :, 1, :] = cols_im.reshape(Q, N, P)
        Fi.reshape(Q, N, 2, P)[:, :, 0, :] = phi_re.reshape(Q, N, P)
        Fi.reshape(Q, N, 2, P)[:, :, 1, :] = phi_im.reshape(Q, N, P)
        A[2 * P * i:2 * P * (i + 1)] = proj @ Ui
        B[2 * i:2 * i + 2] = proj[[1, P + 1]] @ Fi
    G = np.zeros((nu, 2 * N))    # rhs: cos2 row of cell i <- a_i, sin2 row <- b_i
    for i in range(N):
        G[2 * P * i + 1, 2 * i] = 1.0
        G[2 * P * i + P + 1, 2 * i + 1] = 1.0
    C = np.linalg.solve(A, G)
    # T = -(rho R / 2) sum_i oint phi u_r dtheta ; oint phi cos2 = pi * coefficient
    M = -RHO * R * np.pi * (B @ C)
    return 0.5 * (M + M.T)


def self_mass(R):
    """Isolated cell, per oval mode: outside pi rho R^2 / 2, inside pi rho R^2 / 2."""
    return np.pi * RHO * R ** 2 / 2


def bulk_blocks(R, nr=8, P=8, rcut=None):
    """2x2 coupling blocks between the centre cell and every cell of a cluster (bulk values)."""
    z, ring = hex_lattice(nr)
    M = outer_mass(z, R, P)
    blocks = {}
    for j in range(len(z)):
        dz = z[j] - z[0]
        if rcut is None or abs(dz) <= rcut + 1e-9:
            blocks[(round(dz.real, 6), round(dz.imag, 6))] = M[0:2, 2 * j:2 * j + 2].copy()
    return blocks, M, z


def bloch(blocks, k):
    """M(k) = sum_j M_0j exp(i k . r_j)."""
    Mk = np.zeros((2, 2), complex)
    for (x, y), b in blocks.items():
        Mk += b * np.exp(1j * (k[0] * x + k[1] * y))
    return Mk


def lattice_mass(z, blocks, rcut):
    """Big lattice with translation-invariant bulk blocks (edge cells inside the sponge)."""
    N = len(z)
    M = np.zeros((2 * N, 2 * N))
    keys = {k: v for k, v in blocks.items()}
    for i in range(N):
        dz = z - z[i]
        idx = np.nonzero(np.abs(dz) <= rcut + 1e-9)[0]
        for j in idx:
            key = (round(dz[j].real, 6), round(dz[j].imag, 6))
            M[2 * i:2 * i + 2, 2 * j:2 * j + 2] = keys[key]
    return 0.5 * (M + M.T)


if __name__ == "__main__":
    R = float(sys.argv[1]) if len(sys.argv) > 1 else 0.45
    t0 = time.perf_counter()
    z = np.array([0j])
    print("single cell outer mass", outer_mass(z, R, 4)[0, 0], "theory", self_mass(R))
    z2 = np.array([0j, 1 + 0j])
    for P in (2, 4, 8, 12):
        M = outer_mass(z2, R, P)
        print("pair P", P, "M_aa(0,1)", M[0, 2], "M_bb", M[1, 3], "self", M[0, 0], "leading", 1.5 * np.pi * R ** 6)
    print(time.perf_counter() - t0)
