"""Open boundaries for the lattice Boltzmann solver.

Inlet (source): make the bottom row solid and give it a wall velocity (0, q0). The moving-wall
bounce-back in kernels.stream then injects fluid at speed q0 (Ladd's term with a normal wall
velocity), so the bottom row acts as a uniform source.

Outlet: the top row is fluid. After streaming, outlet_top resets it by non-equilibrium
extrapolation (Guo, Zheng & Shi 2002): density fixed at rho_out, velocity and non-equilibrium
part copied from the row below. Fluid leaves freely at a fixed pressure.
"""
from numba import njit, prange

from .lattice import CX, CY, W


@njit(parallel=True, cache=True)
def outlet_top(f, rho_out):
    _, nx, ny = f.shape
    j, jb = ny - 1, ny - 2
    for x in prange(nx):
        r = 0.0
        jx = 0.0
        jy = 0.0
        for i in range(9):
            fi = f[i, x, jb]
            r += fi
            jx += fi * CX[i]
            jy += fi * CY[i]
        u = jx / r
        v = jy / r
        uu = 1.5 * (u * u + v * v)
        for i in range(9):
            cu = CX[i] * u + CY[i] * v
            e = 1.0 + 3.0 * cu + 4.5 * cu * cu - uu
            f[i, x, j] = W[i] * rho_out * e + (f[i, x, jb] - W[i] * r * e)
