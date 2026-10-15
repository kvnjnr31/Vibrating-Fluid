"""
rotating_body.py
===============================================================================
Pure geometry/kinematics module for a rigidly rotating circular obstacle.

This module has NO dependency on the LBM solver (lbm_core.py /
lbm_rotating_obstacle.py) and no dependency on the phage/receptor agent
model (tiger_encounter_coupled.py). It only knows about circles and rigid
rotation. That separation is deliberate: this module can be unit-tested and
sanity-checked entirely on its own (see `self_check()` below and
`if __name__ == "__main__"`), independent of whether the fluid solver or the
phage code are working correctly. If something looks wrong downstream, you
can rule geometry in or out first before touching either of the other two.

-------------------------------------------------------------------------------
WHY A CIRCLE, AND WHY THIS IS NOT AN OVERSIMPLIFICATION
-------------------------------------------------------------------------------
The 3D bacterium model used elsewhere in this project (encounter_rates_main.m
and its Python port, tiger_encounter_coupled.py) is a capsule -- a cylinder
capped by two hemispheres -- that is rigidly rotated about its long axis
(the y-axis in that code). That capsule is rotationally symmetric about the
y-axis, which means every cross-section perpendicular to y (i.e. every
constant-y slice, taken in the x-z plane) is an exact circle: radius Rb in
the cylindrical body region, and a smaller radius sqrt(Rb^2 - (y-y0)^2) in
either hemispherical cap region.

So a 2D LBM simulation in a plane perpendicular to the rotation axis, with a
rigidly rotating circular obstacle, is not an arbitrary simplification of
the 3D geometry -- it is the exact cross-section of the real model, sliced
at whatever y you choose to place the LBM plane. What this 2D slice
deliberately does NOT capture is any axial (y-direction) flow -- e.g. from
flagellar propulsion, or net swimming translation -- since that motion is
out-of-plane for this cross-section. This module and the solver built on it
are scoped to the rotational hydrodynamics only, matching what was asked
for; axial/translational flow would need a separate (or 3D) treatment.

-------------------------------------------------------------------------------
API
-------------------------------------------------------------------------------
    RotatingCircle(center, radius, omega)
        .solid_mask(nx, ny)         -> (nx, ny) bool array, True = solid
        .wall_velocity(x, y)        -> (ux, uy) rigid-rotation velocity at
                                        any point(s) (x, y), vectorized
        .boundary_velocity_field(nx, ny) -> (nx, ny, 2) wall-velocity field,
                                        defined everywhere but only meaningful
                                        where used (adjacent to solid_mask)

Units are "lattice units" throughout (same convention as lbm_core.py):
center/radius/positions in lattice-node coordinates, omega in radians per
lattice time step.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RotatingCircle:
    """A rigid, steadily-rotating circular obstacle.

    Because rotation about a circle's own center leaves the circle's shape
    (and hence its solid/fluid mask) unchanged, `solid_mask` is time
    independent -- only the imparted wall *velocity* depends on omega. That
    means the disturbance flow this drives reaches a genuine steady state,
    which is what lets the LBM solver below reuse the same
    "iterate to convergence" pattern as the existing Poiseuille solver
    instead of needing to track a time-dependent boundary.
    """

    center: tuple[float, float]  # (cx, cy) in lattice units
    radius: float                 # R, in lattice units
    omega: float                  # rad / lattice-time-step, +ve = counter-clockwise

    def solid_mask(self, nx: int, ny: int) -> np.ndarray:
        """Boolean (nx, ny) mask, True where the node lies inside the circle."""
        cx, cy = self.center
        xs = np.arange(nx)[:, None]
        ys = np.arange(ny)[None, :]
        return (xs - cx) ** 2 + (ys - cy) ** 2 <= self.radius**2

    def wall_velocity(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rigid-body rotation velocity u = omega * zhat x (r - center) at
        the given point(s). Vectorized: x, y may be scalars or arrays of the
        same shape.

        In 2D, zhat x (rx, ry) = (-ry, rx), so:
            ux = -omega * (y - cy)
            uy =  omega * (x - cx)
        This is exact rigid-body kinematics (not an approximation) -- it is
        the boundary condition the fluid must satisfy at the obstacle
        surface, valid for ANY point (used both exactly on the boundary,
        which is what the LBM bounce-back needs, and for the self-check
        below, which spot-checks tangential-speed = omega * r).
        """
        cx, cy = self.center
        rx, ry = np.asarray(x) - cx, np.asarray(y) - cy
        ux = -self.omega * ry
        uy = self.omega * rx
        return ux, uy

    def boundary_velocity_field(self, nx: int, ny: int) -> np.ndarray:
        """Wall-velocity field evaluated at every lattice node (nx, ny, 2).
        Only the values at/adjacent to the solid boundary are physically
        used by the LBM moving bounce-back, but it's cheap and simpler to
        evaluate everywhere with one vectorized call."""
        xs = np.arange(nx)[:, None] * np.ones((1, ny))
        ys = np.ones((nx, 1)) * np.arange(ny)[None, :]
        ux, uy = self.wall_velocity(xs, ys)
        return np.stack([ux, uy], axis=-1)


def self_check(nx: int = 64, ny: int = 64, R: float = 8.0, omega: float = 0.02) -> None:
    """Independent sanity check of this module ONLY -- no LBM, no phage
    code. Verifies:
      1. solid_mask area matches pi*R^2 to a discretization-limited tolerance.
      2. wall_velocity at boundary points is purely tangential (radial
         component ~ 0) with magnitude omega*R (to within discretization).
    Run directly: `python rotating_body.py`
    """
    body = RotatingCircle(center=(nx / 2, ny / 2), radius=R, omega=omega)

    mask = body.solid_mask(nx, ny)
    area_measured = mask.sum()
    area_expected = np.pi * R**2
    rel_area_err = abs(area_measured - area_expected) / area_expected
    print(f"[rotating_body.self_check] solid area: measured={area_measured}, "
          f"expected={area_expected:.1f}, rel_err={rel_area_err:.3f}")
    assert rel_area_err < 0.05, "solid_mask area is off by more than 5% -- geometry bug"

    # sample boundary points at several angles, confirm tangential-only velocity
    cx, cy = body.center
    thetas = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    bx = cx + R * np.cos(thetas)
    by = cy + R * np.sin(thetas)
    ux, uy = body.wall_velocity(bx, by)

    rhat_x, rhat_y = np.cos(thetas), np.sin(thetas)
    radial_component = ux * rhat_x + uy * rhat_y
    speed = np.sqrt(ux**2 + uy**2)
    expected_speed = omega * R

    print(f"[rotating_body.self_check] max |radial component| = "
          f"{np.max(np.abs(radial_component)):.2e} (should be ~0)")
    print(f"[rotating_body.self_check] speed: measured={speed[0]:.5f}, "
          f"expected=omega*R={expected_speed:.5f}")

    assert np.max(np.abs(radial_component)) < 1e-10, "wall velocity has a radial component -- not pure rotation"
    assert np.allclose(speed, expected_speed), "wall speed != omega*R -- kinematics bug"

    print("[rotating_body.self_check] PASSED")


if __name__ == "__main__":
    self_check()
