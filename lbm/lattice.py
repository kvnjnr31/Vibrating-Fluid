"""D2Q9 lattice constants (lattice units: dx = dt = 1).

Same direction order as the original numpy solver (reference/core_numpy.py):

    6  2  5
    3  0  1
    7  4  8
"""
import numpy as np

CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=np.float64)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
CS2 = 1.0 / 3.0


def viscosity(tau):
    """Kinematic viscosity of the BGK model, nu = cs^2 (tau - 1/2)."""
    return CS2 * (tau - 0.5)


def tau_from_viscosity(nu):
    return nu / CS2 + 0.5
