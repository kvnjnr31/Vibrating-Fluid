# Vibrating Fluid

Flow of a fluid inside and around a permeable vibrating spherical shell (K. L. Joiner).

The shell sits at r = R + w, with w = W0 cos(wt) Y(theta, phi). Fluid passes through the shell along its normal at the rate q = q0 + k w. The local continuity and Navier–Stokes equations stay the same; the shell enters through the conditions at its surface.

## Contents

| Folder | Contents |
|---|---|
| `notes/` | LaTeX draft `vibrating_hydrodynamics.tex` (continuity, Navier–Stokes with a source, net outflow, mean flow) |
| `figures/` | Figure scripts, grey scale, one slice with Y = cos(m theta) |
| `lbm/` | D2Q9 lattice Boltzmann solver in numba, with moving walls and body forces |
| `lbm/reference/` | The original numpy solver it was refactored from (kept for comparison) |
| `validation/` | Checks of the solver against exact solutions |

## Figures

    pip install -r requirements.txt
    cd figures
    python shell_flux_figure.py          # shell normals and mean sideways flux
    python shell_flow_extremes.py        # flow at the two extremes of the cycle
    python shell_mode_sweep.py           # the same for modes 2, 4, 8, 16

`shell_mode_sweep.py` imports from `shell_flow_extremes.py`, so run it from the `figures/` folder.

The flow fields are potential flows (a central source plus the mode term), not viscous solutions.

## Lattice Boltzmann solver

    from lbm import LBM
    sim = LBM(nx=256, ny=128, tau=0.8)
    sim.solid[:, 0] = sim.solid[:, -1] = True     # walls
    sim.uwx[:, 0] = 0.01                           # moving bottom wall
    sim.Fx[:] = 1e-6                               # body force density
    sim.run(20000)
    rho, ux, uy = sim.fields()

Walls use half-way bounce-back with Ladd's moving-wall term; body forces use Guo's scheme. Check it with

    python validation/run_checks.py                # about 2.5 min on 2 cores

| Check | Exact solution | Error |
|---|---|---|
| Poiseuille channel | u = g s (H - s) / 2 nu | 0.05% |
| Stokes layer over an oscillating wall | u = U exp(-s/d) cos(wt - s/d) | 0.07% |
| Spinning cylinder, R < r < 1.5R | u_theta = Omega R^2 / r | 1.2% |
| Speed, 256 x 256, 2 cores | numba 21 million updates/s, numpy original 2.4 | 8.8x faster |

The string is an immersed boundary (`lbm/ib.py`): marker points with prescribed positions and velocities push on the fluid (direct forcing, Peskin 4-point delta). Check it with

    python validation/string_checks.py               # about 15 min on 2 cores

| Check | Comparison | Error |
|---|---|---|
| Flat string oscillating along itself | exact Stokes layer on each side; the string acts as a plate of half-thickness 0.8 | 0.4% |
| String with a standing wave, A = 1, 1.5, 2 | small-amplitude theory of the mean flow (`validation/streaming_theory.py`) | 6-7% |

The mean flow grows as A^2 to within 1%. Near the string the mean flow runs from the antinodes to the nodes; it leaves the string at the nodes and returns at the antinodes.

The spinning cylinder sits in a periodic box, so far from the cylinder the flow falls below Omega R^2 / r; the check uses only the band near the cylinder, as in the original validation.

## Planned

- Tracer particles (Lagrangian mean flow) and moving nodes driven by the string solver
- Interaction of several shells
- `aws/` folder for long parameter sweeps on EC2
