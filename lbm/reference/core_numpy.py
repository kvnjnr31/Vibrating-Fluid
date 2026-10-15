# core.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# ===== D2Q9 (lattice units) ==================================================
C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=int,
)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=float)
CS2 = 1.0 / 3.0
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=int)


# ===== Geometry ===============================================================
def channel_walls(nx: int, ny: int) -> np.ndarray:
    """
    Construct a simple 2D channel wall mask for a periodic-x domain.

    The geometry is a rectangular lattice (nx, ny) with solid (no-slip) walls
    at the bottom and top boundaries in y:
        - solid[:, 0]    = True   (bottom wall)
        - solid[:, ny-1] = True   (top wall)
    All other nodes are fluid.

    Parameters
    ----------
    nx
        Number of lattice nodes in the x-direction.
    ny
        Number of lattice nodes in the y-direction.

    Returns
    -------
    solid : (nx, ny) bool ndarray
        Boolean mask where True indicates a solid wall node and False indicates
        a fluid node.
    """
    solid = np.zeros((nx, ny), dtype=bool)
    solid[:, 0] = True
    solid[:, -1] = True
    return solid


# ===== LBM building blocks ====================================================
def feq(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Compute the D2Q9 equilibrium distribution f_eq for a given (rho, u).

    This uses the standard second-order (in Mach number) expansion of the
    Maxwell-Boltzmann equilibrium in lattice units with c_s^2 = CS2.

    Parameters
    ----------
    rho
        Density field.
        Shape: (nx, ny)
    u
        Velocity field.
        Shape: (nx, ny, 2) with u[..., 0] = u_x and u[..., 1] = u_y

    Returns
    -------
    f_eq : (nx, ny, 9) float ndarray
        Equilibrium populations for each discrete velocity direction i=0..8.
        The last axis corresponds to the D2Q9 directions (same ordering as C, W).
    """
    ux, uy = u[..., 0], u[..., 1]
    uu = ux * ux + uy * uy
    f = np.zeros((*rho.shape, 9), dtype=float)
    for i in range(9):
        cu = C[i, 0] * ux + C[i, 1] * uy
        f[..., i] = W[i] * rho * (
            1 + cu / CS2 + 0.5 * (cu * cu) / (CS2 * CS2) - 0.5 * uu / CS2
        )
    return f


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover macroscopic density and velocity from distribution populations f.

    Definitions (lattice units):
        rho = sum_i f_i
        j   = sum_i f_i * c_i
        u   = j / rho

    A small epsilon is added in the denominator to avoid division-by-zero.

    Parameters
    ----------
    f
        Distribution populations.
        Shape: (nx, ny, 9)

    Returns
    -------
    rho : (nx, ny) float ndarray
        Density field.
    u : (nx, ny, 2) float ndarray
        Velocity field (u_x, u_y).
    """
    rho = np.sum(f, axis=-1)
    jx = np.sum(f * C[:, 0], axis=-1)
    jy = np.sum(f * C[:, 1], axis=-1)
    u = np.zeros((*rho.shape, 2), dtype=float)
    u[..., 0] = jx / (rho + 1e-12)
    u[..., 1] = jy / (rho + 1e-12)
    return rho, u


def stream(f: np.ndarray) -> np.ndarray:
    """
    Stream (propagate) D2Q9 populations to neighbor nodes via periodic shifts.

    For each lattice direction i with discrete velocity C[i] = (cx, cy),
    streaming moves f_i(x, y) -> f_i(x+cx, y+cy). This is implemented via
    np.roll along axis=0 (x) and axis=1 (y). The roll implies periodicity
    in both x and y; in practice, no-slip walls are enforced afterward via
    bounce-back, so the effective y-boundaries become solid.

    Parameters
    ----------
    f
        Distribution populations prior to streaming.
        Shape: (nx, ny, 9)

    Returns
    -------
    g : (nx, ny, 9) float ndarray
        Streamed populations.
    """
    g = np.empty_like(f)
    for i in range(9):
        g[..., i] = np.roll(np.roll(f[..., i], C[i, 0], axis=0), C[i, 1], axis=1)
    return g


def bounce_back(f: np.ndarray, solid: np.ndarray) -> None:
    """
    Apply full-way bounce-back boundary condition on solid nodes.

    For each solid node, populations are swapped with their opposite-direction
    counterparts using the OPP mapping:
        f_i(solid) <- f_OPP[i](solid)

    This enforces a no-slip condition at the wall (for standard BGK LBM).

    Parameters
    ----------
    f
        Distribution populations to be modified in-place.
        Shape: (nx, ny, 9)
    solid
        Boolean mask of solid nodes.
        Shape: (nx, ny)

    Returns
    -------
    None
        The array f is modified in-place.
    """
    fs = f[solid].copy()
    f[solid] = fs[:, OPP]


def guo_force(rho: np.ndarray, u: np.ndarray, a: np.ndarray, tau: float) -> np.ndarray:
    """
    Compute the D2Q9 forcing term using the Guo forcing scheme (BGK).

    This returns the additive term F_i that is added during collision:
        f <- f - (f - f_eq)/tau + F

    Here, "a" is the physical acceleration field (body force per unit mass),
    and the Guo scheme incorporates a prefactor (1 - 0.5/tau) for second-order
    accuracy in time.

    Parameters
    ----------
    rho
        Density field.
        Shape: (nx, ny)
    u
        Velocity field at which the forcing term is evaluated.
        Shape: (nx, ny, 2)
    a
        Acceleration field (body force per unit mass).
        Shape: (nx, ny, 2)
    tau
        BGK relaxation time (in lattice time units). Must satisfy tau > 0.5
        for positive viscosity.

    Returns
    -------
    F : (nx, ny, 9) float ndarray
        Guo forcing term for each D2Q9 population direction.
    """
    ux, uy = u[..., 0], u[..., 1]
    ax, ay = a[..., 0], a[..., 1]
    pref = 1.0 - 0.5 / tau
    F = np.zeros((*rho.shape, 9), dtype=float)
    for i in range(9):
        cx, cy = C[i]
        cu = cx * ux + cy * uy
        ca = cx * ax + cy * ay
        F[..., i] = pref * W[i] * rho * (
            (((cx - ux) * ax + (cy - uy) * ay) / CS2) + (cu * ca) / (CS2 * CS2)
        )
    return F


# ===== Scales =================================================================
def viscosity_from_tau(tau: float, dt: float = 1.0) -> float:
    """
    Convert BGK relaxation time tau to kinematic viscosity nu.

    In lattice Boltzmann (BGK, lattice units), the kinematic viscosity is:
        nu = c_s^2 (tau - 1/2) dt
    where c_s^2 = CS2.

    Parameters
    ----------
    tau
        Relaxation time (lattice units). Typically tau > 0.5.
    dt
        Lattice time step (default 1.0 in lattice units).

    Returns
    -------
    nu : float
        Kinematic viscosity in lattice units.
    """
    return CS2 * (tau - 0.5) * dt


@dataclass
class RunLog:
    """
    Convenience container for nondimensional and summary metrics.

    Attributes
    ----------
    nu : float
        Kinematic viscosity (lattice units).
    umax : float
        Maximum speed magnitude over the domain.
    umean : float
        Mean speed magnitude over the domain.
    Re : float
        Reynolds number defined as Re = U * L / nu.
    Ma : float
        Mach number defined as Ma = U / c_s, with c_s = sqrt(CS2).
    """

    nu: float
    umax: float
    umean: float
    Re: float
    Ma: float


def compute_scales(
    u: np.ndarray,
    L: float,
    tau: float,
    dt: float = 1.0,
    U_def: str = "umax",
) -> RunLog:
    """
    Compute basic scales and nondimensional numbers from a velocity field.

    The characteristic speed U is chosen as either:
        - "umax": maximum speed magnitude over the grid
        - otherwise: mean speed magnitude over the grid (typically "umean")

    Reynolds number:
        Re = U * L / nu
    Mach number:
        Ma = U / c_s, with c_s = sqrt(CS2)

    Parameters
    ----------
    u
        Velocity field.
        Shape: (nx, ny, 2)
    L
        Characteristic length scale (in lattice units), e.g., channel height or
        hydraulic diameter as used by the caller.
    tau
        BGK relaxation time used to compute viscosity.
    dt
        Lattice time step (default 1.0).
    U_def
        Speed definition: "umax" (default) uses max speed; any other value uses
        mean speed.

    Returns
    -------
    log : RunLog
        Struct containing nu, umax, umean, Re, Ma.
    """
    nu = viscosity_from_tau(tau, dt)
    speed = np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2)
    umax, umean = float(np.max(speed)), float(np.mean(speed))
    U = umax if U_def == "umax" else umean
    return RunLog(
        nu=nu,
        umax=umax,
        umean=umean,
        Re=float(U * L / (nu + 1e-30)),
        Ma=float(U / np.sqrt(CS2)),
    )


# ===== Poiseuille solver ======================================================
def run_lbm_poiseuille(
    nx: int,
    ny: int,
    tau: float,
    ax: float,
    *,
    u0: np.ndarray | None = None,
    v0: np.ndarray | None = None,
    max_steps: int = 30000,
    tol: float = 1e-10,
    check_every: int = 200,
    record_every: int = 50,
    record: bool = False,
) -> tuple[np.ndarray, dict | None]:
    """
    Solve steady 2D Poiseuille flow in a channel using D2Q9 BGK LBM + Guo forcing.

    Domain and boundary conditions
    ------------------------------
    - Grid shape: (nx, ny)
    - Periodic in x (implemented by streaming with np.roll in x)
    - No-slip walls at y=0 and y=ny-1 via full-way bounce-back
    - Constant body-force acceleration in +x: ax

    Numerical scheme (per iteration)
    --------------------------------
    1) Compute (rho, u) from populations f
    2) Apply half-step acceleration: u <- u + 0.5 * a
    3) BGK collision with equilibrium and Guo forcing:
           f <- f - (f - feq(rho, u))/tau + guo_force(rho, u, a, tau)
    4) Stream
    5) Bounce-back on solid nodes

    Convergence check
    -----------------
    Every `check_every` steps, compute the x-averaged u_x(y) profile and compare
    it to the previous stored profile (excluding wall nodes). Stop if the
    relative L2 change is below `tol`.

    Parameters
    ----------
    nx, ny
        Lattice resolution in x and y.
    tau
        BGK relaxation time (lattice units). Must satisfy tau > 0.5 for positive viscosity.
    ax
        Constant acceleration applied in the x-direction (body force per unit mass).
    u0
        Optional initial condition for u_x.
        Shape: (nx, ny). If provided, sets u[..., 0] = u0.
    v0
        Optional initial condition for u_y.
        Shape: (nx, ny). If u0 is provided but v0 is None, u_y is initialized to 0.
    max_steps
        Maximum number of LBM iterations.
    tol
        Relative tolerance for convergence of the x-averaged u_x(y) profile.
    check_every
        Convergence check interval (iterations).
    record_every
        Recording interval (iterations) when `record=True`.
    record
        If True, store the x-averaged u_x(y) profile over time.

    Returns
    -------
    u : (nx, ny, 2) float ndarray
        Final velocity field. The returned velocity includes the final half-step
        acceleration correction u <- u + 0.5 * a.
    history : dict or None
        If record=False: None.
        If record=True: dict with:
            - "it":   (n_snap,) int ndarray of iteration indices
            - "ubar": (n_snap, ny) float ndarray, where each row is the x-averaged u_x(y)
                     at the recorded iteration.
    """
    solid = channel_walls(nx, ny)
    rho = np.ones((nx, ny), dtype=float)

    u = np.zeros((nx, ny, 2), dtype=float)
    if u0 is not None:
        u[..., 0] = u0
        u[..., 1] = 0.0 if v0 is None else v0

    f = feq(rho, u)

    a = np.zeros_like(u)
    a[..., 0] = float(ax)
    a[solid] = 0.0

    it_hist, ubar_hist = [], []
    last = None

    for it in range(int(max_steps)):
        rho, u = macroscopic(f)
        u = u + 0.5 * a  # half-step acceleration (standard with Guo forcing)
        f = f - (f - feq(rho, u)) / tau + guo_force(rho, u, a, tau)
        f = stream(f)
        bounce_back(f, solid)

        if record and (it % int(record_every) == 0):
            it_hist.append(it)
            ubar_hist.append(np.mean(u[..., 0], axis=0))  # x-averaged u_x(y)

        if it % int(check_every) == 0:
            prof = np.mean(u[..., 0], axis=0)[1:-1]  # ignore wall nodes
            if last is not None:
                rel = np.linalg.norm(prof - last) / (np.linalg.norm(last) + 1e-30)
                if rel < float(tol):
                    break
            last = prof.copy()

    rho, u = macroscopic(f)
    u = u + 0.5 * a

    if not record:
        return u, None
    return u, {"it": np.asarray(it_hist, dtype=int), "ubar": np.asarray(ubar_hist, dtype=float)}