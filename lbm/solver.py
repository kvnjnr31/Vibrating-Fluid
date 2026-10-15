"""Lattice Boltzmann solver object: holds the fields and runs time steps.

    from lbm import LBM
    sim = LBM(nx=256, ny=128, tau=0.8)
    sim.solid[:, 0] = sim.solid[:, -1] = True     # channel walls
    sim.Fx[:] = 1e-6                               # body force density
    sim.run(20000)
    rho, ux, uy = sim.fields()

Moving walls: set sim.uwx, sim.uwy on solid nodes (they may change every step).
Body forces (later: the immersed-boundary string) go in sim.Fx, sim.Fy.
"""
import numpy as np

from .kernels import collide, stream, equilibrium, moments
from .lattice import viscosity


class LBM:
    def __init__(self, nx, ny, tau, rho0=1.0):
        if tau <= 0.5:
            raise ValueError("tau must exceed 0.5")
        self.nx, self.ny, self.tau = int(nx), int(ny), float(tau)
        self.omega = 1.0 / self.tau
        self.nu = viscosity(self.tau)
        shape = (self.nx, self.ny)
        self.solid = np.zeros(shape, dtype=np.bool_)
        self.uwx = np.zeros(shape)
        self.uwy = np.zeros(shape)
        self.Fx = np.zeros(shape)
        self.Fy = np.zeros(shape)
        self.rho = np.full(shape, float(rho0))
        self.ux = np.zeros(shape)
        self.uy = np.zeros(shape)
        self.f = np.empty((9, *shape))
        self.f_new = np.empty_like(self.f)
        equilibrium(self.rho, self.ux, self.uy, self.f)
        self.f_new[:] = self.f
        self.t = 0

    def set_velocity(self, ux, uy, rho=None):
        """Start from equilibrium at the given velocity (and density)."""
        self.ux[:] = ux
        self.uy[:] = uy
        if rho is not None:
            self.rho[:] = rho
        equilibrium(self.rho, self.ux, self.uy, self.f)
        self.f_new[:] = self.f

    def step(self, wall=None):
        """One time step. wall(t, sim) may update uwx, uwy, Fx, Fy before the step."""
        if wall is not None:
            wall(self.t, self)
        collide(self.f, self.solid, self.Fx, self.Fy, self.omega, self.rho, self.ux, self.uy)
        stream(self.f, self.f_new, self.solid, self.uwx, self.uwy, self.rho)
        self.f, self.f_new = self.f_new, self.f
        self.t += 1

    def run(self, n, wall=None, every=0, callback=None):
        for _ in range(int(n)):
            self.step(wall)
            if every and callback is not None and self.t % every == 0:
                callback(self)

    def fields(self):
        """Density and physical velocity at the current time (velocity zero inside solids)."""
        moments(self.f, self.solid, self.Fx, self.Fy, self.rho, self.ux, self.uy)
        return self.rho, self.ux, self.uy

    def scales(self, L):
        """Peak speed, Reynolds number U L / nu and Mach number U / cs."""
        _, ux, uy = self.fields()
        U = float(np.sqrt(ux ** 2 + uy ** 2).max())
        return dict(umax=U, Re=U * L / self.nu, Ma=U * np.sqrt(3.0), nu=self.nu)
