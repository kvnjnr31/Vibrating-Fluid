"""
lbm_rotating_obstacle.py
===============================================================================
Extends the lab's existing, validated D2Q9 BGK LBM engine (lbm_core.py --
a vendored, byte-identical copy of
01_LBM_paper/01_overdamped_validation/deliverables/core.py) with a rigidly
rotating circular obstacle, so the solver can generate the disturbance flow
field induced by a spinning bacterium cross-section.

DELIBERATELY NOT MODIFIED: lbm_core.py. Everything here is additive --
this module only *imports* feq/macroscopic/stream/viscosity_from_tau/
compute_scales/C/W/CS2/OPP from it. That keeps the lab's existing
channel/Couette validation (00_biophysics_validation, 01_overdamped_
validation, 02_parameter_fitting) completely untouched and independently
re-runnable/checkable, while this obstacle capability lives and is checked
on its own. See rotating_body.py for the (also independent) geometry piece.

-------------------------------------------------------------------------------
WHAT'S NEW: MOVING BOUNCE-BACK
-------------------------------------------------------------------------------
lbm_core.py's `bounce_back()` implements static, no-slip, full-way
bounce-back: f_i(solid) <- f_opp_i(solid). That's correct for a stationary
wall (u_wall = 0 everywhere on it) but says nothing about a *moving* wall.

For a moving/rotating wall, the standard extension (Ladd 1994) adds a
momentum-exchange correction term on top of the same opposite-direction
swap:

    f_i(solid) <- f_opp_i(solid) + 2 * w_i * rho_wall * (c_i . u_wall) / cs^2

where u_wall is the local velocity of the solid boundary at that node. This
module applies that correction using the *same* full-way convention
lbm_core.py already uses (swap the whole solid node, not just a per-link
half-way correction) -- i.e. it is the direct, consistent extension of the
existing scheme, not a different scheme. See the note in the project's own
PROJECT_README.md about full-way vs. half-way bounce-back placing the
no-slip surface at slightly different sub-grid locations -- that caveat
applies here too, and matters if you need decay-law parameters to
sub-percent accuracy; a half-way version would be a natural follow-up
refinement, not implemented here to keep this extension minimal.

u_wall itself comes from rotating_body.RotatingCircle.wall_velocity(), i.e.
exact rigid-body kinematics (omega x r), not a fitted or empirical value.

-------------------------------------------------------------------------------
WHY NO BODY FORCE
-------------------------------------------------------------------------------
Unlike run_lbm_poiseuille() (driven by a constant pressure-gradient body
force), this flow is driven purely by the rotating wall boundary condition.
That's a clean, physically separate test case from the channel/Couette
benchmarks -- exactly the kind of modularity asked for: this obstacle+
rotation solver can be validated completely on its own, against its own
analytical solution (see validate_rotating_obstacle.py), independent of
whether the Poiseuille/Couette benchmarks pass or fail, and vice versa.

-------------------------------------------------------------------------------
ANALYTICAL TARGET: THE 2D ROTLET
-------------------------------------------------------------------------------
For a rigid cylinder of radius R rotating at angular velocity omega in an
unbounded, purely azimuthal, axisymmetric 2D incompressible flow (u_r = 0,
u_theta = u_theta(r) only), the steady incompressible Navier-Stokes
equations reduce to:

    d/dr[ (1/r) d(r u_theta)/dr ] = 0

Integrating with u_theta -> 0 as r -> infinity and u_theta(R) = omega*R
(no-slip at the rotating surface) gives the exact solution

    u_theta(r) = omega * R^2 / r ,   r >= R

Notably, the nonlinear convective terms vanish identically for this flow
(u_r = 0 kills them outright), so this is an EXACT solution of the full
nonlinear Navier-Stokes equations, valid at any Reynolds number -- not just
a Stokes/creeping-flow linearization. That makes it a clean, Re-independent
target: this check is really testing "does the moving bounce-back correctly
impose the rotation and let the flow relax to the right steady state,"
not "has Re been pushed low enough for a Stokes approximation to hold"
(that question is what 01_overdamped_validation's Poiseuille/Couette sweep
is already for).

The one real-world caveat: our domain is a finite, periodic box, not an
infinite one, so u_theta -> 0 as r -> infinity isn't exactly enforceable --
periodic images perturb the far field. validate_rotating_obstacle.py
therefore only fits/compares in an annulus R < r < ~0.3*min(nx, ny), where
image effects are still small.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core_numpy import C, W, CS2, OPP, feq, macroscopic, stream, viscosity_from_tau, compute_scales, RunLog
from rotating_body import RotatingCircle


def moving_bounce_back(
    f: np.ndarray, solid: np.ndarray, u_wall: np.ndarray, rho_wall: float = 1.0
) -> np.ndarray:
    """Full-way moving bounce-back (see module docstring). Modifies f
    in-place on solid nodes and also returns it for convenience.

    Parameters
    ----------
    f : (nx, ny, 9)
    solid : (nx, ny) bool
    u_wall : (nx, ny, 2) -- local wall velocity, only used where solid=True
    rho_wall : reference density used in the momentum-exchange term (Eq. in
        module docstring). Using a fixed constant (nominal rho=1, the same
        value the domain is initialized to) rather than the locally
        *measured* density at solid nodes is the standard choice in the
        literature and is important here for a practical reason: interior
        solid-node populations are not physically meaningful (they never
        feed back into the fluid -- only the solid/fluid interface layer
        does), so their measured "density" can drift arbitrarily far from 1
        over iterations. Feeding that drifting value back into the
        correction term creates a runaway feedback loop. An earlier version
        of this function used the locally measured density and the flow
        blew up (umax several times the driving wall speed) for exactly
        this reason -- switching to a fixed rho_wall fixed it.
    """
    fs = f[solid].copy()
    f[solid] = fs[:, OPP]

    ux = u_wall[solid, 0]
    uy = u_wall[solid, 1]
    for i in range(9):
        cu = C[i, 0] * ux + C[i, 1] * uy
        f[solid, i] += 2.0 * W[i] * rho_wall * cu / CS2
    return f


@dataclass
class RotatingObstacleResult:
    u: np.ndarray            # (nx, ny, 2) final velocity field
    rho: np.ndarray          # (nx, ny) final density field
    solid: np.ndarray        # (nx, ny) bool obstacle mask
    body: RotatingCircle
    scales: RunLog
    converged: bool
    n_iters: int
    history: dict | None


def run_lbm_rotating_obstacle(
    nx: int,
    ny: int,
    tau: float,
    R: float,
    omega: float,
    center: tuple[float, float] | None = None,
    *,
    max_steps: int = 40000,
    tol: float = 1e-7,
    check_every: int = 200,
    probe_radius_factor: float = 2.0,
    record: bool = False,
    record_every: int = 200,
) -> RotatingObstacleResult:
    """Solve for the steady 2D flow induced by a rigidly rotating circular
    obstacle in an otherwise quiescent, periodic fluid domain.

    Mirrors run_lbm_poiseuille()'s structure (same collide -> stream ->
    boundary-condition -> convergence-check loop) so it's recognizable to
    anyone who has already read that solver, but swaps the body-force +
    static-wall boundary condition for no forcing + moving bounce-back.

    Convergence is checked via the relative change of the (x-averaged over
    an annulus) azimuthal speed profile, evaluated at a fixed probe radius,
    rather than a full-field profile -- this is a genuinely rotating flow, so
    there's no single line-profile the way there is for a channel; we track
    a scalar (mean speed at r = probe_radius_factor * R) which is stable and
    cheap once the flow has spun up.
    """
    if center is None:
        center = (nx / 2.0, ny / 2.0)

    body = RotatingCircle(center=center, radius=R, omega=omega)
    solid = body.solid_mask(nx, ny)
    u_wall = body.boundary_velocity_field(nx, ny)

    rho = np.ones((nx, ny), dtype=float)
    u = np.zeros((nx, ny, 2), dtype=float)
    f = feq(rho, u)

    # probe ring for convergence tracking
    cx, cy = center
    xs = np.arange(nx)[:, None]
    ys = np.arange(ny)[None, :]
    rr = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    probe_r = probe_radius_factor * R
    probe_mask = (rr > probe_r - 1) & (rr < probe_r + 1) & (~solid)

    it_hist, speed_hist = [], []
    last = None
    converged = False
    n_iters = max_steps

    for it in range(int(max_steps)):
        rho, u = macroscopic(f)
        f = f - (f - feq(rho, u)) / tau
        f = stream(f)
        f = moving_bounce_back(f, solid, u_wall, rho_wall=1.0)

        if record and (it % int(record_every) == 0):
            it_hist.append(it)
            speed = np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2)
            speed_hist.append(float(np.mean(speed[probe_mask])))

        if it % int(check_every) == 0:
            speed = np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2)
            probe_speed = float(np.mean(speed[probe_mask]))
            if last is not None:
                rel = abs(probe_speed - last) / (abs(last) + 1e-30)
                if rel < float(tol):
                    converged = True
                    n_iters = it
                    break
            last = probe_speed

    rho, u = macroscopic(f)
    scales = compute_scales(u, L=2 * R, tau=tau, U_def="umax")

    history = None
    if record:
        history = {"it": np.asarray(it_hist, dtype=int), "speed": np.asarray(speed_hist, dtype=float)}

    return RotatingObstacleResult(
        u=u, rho=rho, solid=solid, body=body, scales=scales,
        converged=converged, n_iters=n_iters, history=history,
    )


if __name__ == "__main__":
    # Quick smoke test (small grid, loose tolerance) -- full validation with
    # figures/metrics lives in validate_rotating_obstacle.py.
    result = run_lbm_rotating_obstacle(nx=64, ny=64, tau=0.8, R=6, omega=0.02, max_steps=5000, check_every=100)
    print(f"converged={result.converged} in {result.n_iters} iters, "
          f"Re={result.scales.Re:.3f}, Ma={result.scales.Ma:.3f}, umax={result.scales.umax:.4f}")
