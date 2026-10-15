# Vibrating Fluid

Flow of a fluid inside and around a permeable vibrating spherical shell (K. L. Joiner).

The shell sits at r = R + w, with w = W0 cos(wt) Y(theta, phi). Fluid passes through the shell along its normal at the rate q = q0 + k w. The local continuity and Navier–Stokes equations stay the same; the shell enters through the conditions at its surface.

## Contents

| Folder | Contents |
|---|---|
| `notes/` | LaTeX draft `vibrating_hydrodynamics.tex` (continuity, Navier–Stokes with a source, net outflow, mean flow) |
| `figures/` | Figure scripts, grey scale, one slice with Y = cos(m theta) |

## Figures

    pip install -r requirements.txt
    cd figures
    python shell_flux_figure.py          # shell normals and mean sideways flux
    python shell_flow_extremes.py        # flow at the two extremes of the cycle
    python shell_mode_sweep.py           # the same for modes 2, 4, 8, 16

`shell_mode_sweep.py` imports from `shell_flow_extremes.py`, so run it from the `figures/` folder.

The flow fields are potential flows (a central source plus the mode term), not viscous solutions.

## Planned

- 2D simulation of a fluid around a vibrating string with moving nodes
- Interaction of several shells
- `aws/` folder for long parameter sweeps on EC2
