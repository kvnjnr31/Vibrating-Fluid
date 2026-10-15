"""Numba kernels for the D2Q9 BGK lattice Boltzmann method.

Populations are stored as f[i, x, y] (one contiguous plane per direction). One time step is

    collide(f)                  BGK relaxation with Guo forcing, on fluid nodes only
    stream(f -> f_new)          pull from the upstream node; periodic in x and y

Walls are solid nodes. A population that would be pulled from a solid node is instead bounced
back on the same link (half-way bounce-back), so the wall sits halfway between the last fluid
node and the first solid node. A moving wall adds Ladd's momentum term:

    f_i(x, t+1) = f*_opp(i)(x, t) + 2 w_i rho (c_i . u_wall) / cs^2

with u_wall the velocity of the solid node the link points to. This replaces the full-way
bounce-back of the original numpy solver (see reference/), which placed the wall about half a
lattice spacing less precisely.

Force is a force density (force per unit volume), so the fluid velocity is
u = (sum_i f_i c_i + F/2) / rho  (Guo et al. 2002).
"""
import numpy as np
from numba import njit, prange

from .lattice import CX, CY, W, OPP


@njit(parallel=True, cache=True)
def collide(f, solid, Fx, Fy, omega, rho, ux, uy):
    """BGK collision with Guo forcing. Writes rho, ux, uy (the physical velocity) as it goes."""
    _, nx, ny = f.shape
    pref = 1.0 - 0.5 * omega
    for x in prange(nx):
        for y in range(ny):
            if solid[x, y]:
                continue
            r = 0.0
            jx = 0.0
            jy = 0.0
            for i in range(9):
                fi = f[i, x, y]
                r += fi
                jx += fi * CX[i]
                jy += fi * CY[i]
            fx = Fx[x, y]
            fy = Fy[x, y]
            u = (jx + 0.5 * fx) / r
            v = (jy + 0.5 * fy) / r
            rho[x, y] = r
            ux[x, y] = u
            uy[x, y] = v
            uu = 1.5 * (u * u + v * v)
            for i in range(9):
                cu = CX[i] * u + CY[i] * v
                feq = W[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - uu)
                force = pref * W[i] * (3.0 * ((CX[i] - u) * fx + (CY[i] - v) * fy)
                                       + 9.0 * cu * (CX[i] * fx + CY[i] * fy))
                f[i, x, y] += omega * (feq - f[i, x, y]) + force


@njit(parallel=True, cache=True)
def stream(f, f_new, solid, uwx, uwy, rho):
    """Pull streaming with half-way (moving) bounce-back on links into solid nodes."""
    _, nx, ny = f.shape
    for x in prange(nx):
        for y in range(ny):
            if solid[x, y]:
                continue
            for i in range(9):
                xs = (x - CX[i]) % nx
                ys = (y - CY[i]) % ny
                if solid[xs, ys]:
                    f_new[i, x, y] = (f[OPP[i], x, y]
                                      + 6.0 * W[i] * rho[x, y] * (CX[i] * uwx[xs, ys] + CY[i] * uwy[xs, ys]))
                else:
                    f_new[i, x, y] = f[i, xs, ys]


@njit(parallel=True, cache=True)
def equilibrium(rho, ux, uy, f):
    _, nx, ny = f.shape
    for x in prange(nx):
        for y in range(ny):
            r = rho[x, y]
            u = ux[x, y]
            v = uy[x, y]
            uu = 1.5 * (u * u + v * v)
            for i in range(9):
                cu = CX[i] * u + CY[i] * v
                f[i, x, y] = W[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - uu)


@njit(parallel=True, cache=True)
def moments(f, solid, Fx, Fy, rho, ux, uy):
    """Density and physical velocity from the populations (zero inside solids)."""
    _, nx, ny = f.shape
    for x in prange(nx):
        for y in range(ny):
            if solid[x, y]:
                rho[x, y] = 1.0
                ux[x, y] = 0.0
                uy[x, y] = 0.0
                continue
            r = 0.0
            jx = 0.0
            jy = 0.0
            for i in range(9):
                fi = f[i, x, y]
                r += fi
                jx += fi * CX[i]
                jy += fi * CY[i]
            rho[x, y] = r
            ux[x, y] = (jx + 0.5 * Fx[x, y]) / r
            uy[x, y] = (jy + 0.5 * Fy[x, y]) / r
