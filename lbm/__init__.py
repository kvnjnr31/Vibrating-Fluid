"""D2Q9 lattice Boltzmann solver (numba), refactored from K. L. Joiner's numpy LBM code (see reference/)."""
from .lattice import CX, CY, W, OPP, CS2, viscosity, tau_from_viscosity
from .solver import LBM

__all__ = ["LBM", "CX", "CY", "W", "OPP", "CS2", "viscosity", "tau_from_viscosity"]
