"""Immersed-boundary string for the lattice Boltzmann solver.

The string is a chain of marker points X_l with prescribed velocities U_l (from a string
solver, or from a formula). Each step the markers push on the fluid so that the fluid at the
markers moves with them (direct forcing, Feng & Michaelides 2004; Wang et al. 2008):

    1. u* = sum_i f_i c_i / rho                  fluid velocity without the force
    2. repeat n_iter times (multi-direct forcing):
           u(X_l) = interpolate(u* + F / 2 rho)    Peskin 4-point delta
           dF_l   = 2 rho (U_l - u(X_l))           force per unit length of string
           F     += spread(dF_l) ds                same delta
    3. collide with F (Guo forcing), stream

Since u = u* + F / (2 rho) in Guo's scheme, step 2 sets the fluid velocity at the markers to
the marker velocity. The string is a smeared interface about 4 lattice spacings thick, so the
grid must resolve the Stokes layer (delta = sqrt(2 nu / omega)) well beyond that.

Periodic in x; markers must stay at least 2 lattice spacings from the top and bottom rows.
"""
import numpy as np
from numba import njit

from .kernels import moments


@njit(inline="always")
def _phi(r):
    r = abs(r)
    if r < 1.0:
        return 0.125 * (3.0 - 2.0 * r + np.sqrt(1.0 + 4.0 * r - 4.0 * r * r))
    if r < 2.0:
        return 0.125 * (5.0 - 2.0 * r - np.sqrt(-7.0 + 12.0 * r - 4.0 * r * r))
    return 0.0


@njit(cache=True)
def interpolate(ux, uy, X, Y, Ux, Uy):
    nx, ny = ux.shape
    for l in range(X.shape[0]):
        x0 = int(np.floor(X[l])) - 1
        y0 = int(np.floor(Y[l])) - 1
        su = 0.0
        sv = 0.0
        for a in range(4):
            wx = _phi(X[l] - (x0 + a))
            i = (x0 + a) % nx
            for b in range(4):
                j = y0 + b
                w = wx * _phi(Y[l] - j)
                su += w * ux[i, j]
                sv += w * uy[i, j]
        Ux[l] = su
        Uy[l] = sv


@njit(cache=True)
def spread(Fl_x, Fl_y, X, Y, ds, Fx, Fy):
    nx, ny = Fx.shape
    for l in range(X.shape[0]):
        x0 = int(np.floor(X[l])) - 1
        y0 = int(np.floor(Y[l])) - 1
        for a in range(4):
            wx = _phi(X[l] - (x0 + a))
            i = (x0 + a) % nx
            for b in range(4):
                j = y0 + b
                w = wx * _phi(Y[l] - j) * ds[l]
                Fx[i, j] += w * Fl_x[l]
                Fy[i, j] += w * Fl_y[l]


class IBString:
    """Markers along a string; call force(sim, X, Y, Ux, Uy) before each step."""

    def __init__(self, n_markers, n_iter=3):
        self.n = int(n_markers)
        self.n_iter = int(n_iter)
        self.Ui = np.zeros(self.n)
        self.Vi = np.zeros(self.n)
        self.Fl_x = np.zeros(self.n)        # total force per unit length on the fluid, last step
        self.Fl_y = np.zeros(self.n)
        self._zero = None

    def force(self, sim, X, Y, Ux, Uy, ds):
        if self._zero is None:
            self._zero = np.zeros_like(sim.Fx)
        moments(sim.f, sim.solid, self._zero, self._zero, sim.rho, sim.ux, sim.uy)   # u*
        u_star_x, u_star_y = sim.ux.copy(), sim.uy.copy()
        sim.Fx[:] = 0.0
        sim.Fy[:] = 0.0
        self.Fl_x[:] = 0.0
        self.Fl_y[:] = 0.0
        dFx = np.empty(self.n)
        dFy = np.empty(self.n)
        for _ in range(self.n_iter):
            ux = u_star_x + 0.5 * sim.Fx / sim.rho
            uy = u_star_y + 0.5 * sim.Fy / sim.rho
            interpolate(ux, uy, X, Y, self.Ui, self.Vi)
            dFx[:] = 2.0 * (Ux - self.Ui)
            dFy[:] = 2.0 * (Uy - self.Vi)
            spread(dFx, dFy, X, Y, ds, sim.Fx, sim.Fy)
            self.Fl_x += dFx
            self.Fl_y += dFy
        sim.Fx[sim.solid] = 0.0
        sim.Fy[sim.solid] = 0.0

    def slip(self, sim, X, Y, Ux, Uy):
        """Largest gap between fluid and marker velocity after the step (a check on no-slip)."""
        _, ux, uy = sim.fields()
        interpolate(ux, uy, X, Y, self.Ui, self.Vi)
        return float(np.max(np.hypot(self.Ui - Ux, self.Vi - Uy)))
