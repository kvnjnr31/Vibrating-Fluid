# Experiment 11: Maxwell's lattice (option A)

Maxwell's vortex cells and idle wheels with the liquid replaced by an ideal medium. Each cell turns as a rigid wheel with moment of inertia I. Each contact holds an idle particle on an elastic spring of stiffness k. The particle moves by the difference of the rim speeds of its two neighbor cells. The spring pull turns both cells.

## Map to the fields (2D, TE)

| Lattice | Field |
|---|---|
| cell spin `Om` | H_z |
| pull at a vertical contact `dh` | E_y |
| pull at a horizontal contact `dv` | E_x |
| moment of inertia 2I/R | permeability mu |
| contact stiffness kR/2 | 1/epsilon |
| particle slip time t_c | conductivity sigma = epsilon / t_c |

The update is a Yee (FDTD) scheme. With I = 1, k = 16 and R = 0.5, the wave speed is c = (R/2) sqrt(k/I) = 1 cell per time unit.

## Runs

- `run_closed`: no drive, no sponge. Total energy swings 0.9% with no trend.
- `run_dispersion`: periods 5 to 30 cells. The lattice follows the Yee dispersion law within 1.6%.
- `run_point`: one cell twisted back and forth (period 12). The wave spreads as circular rings. Axis and diagonal amplitudes agree within about 4%. Amplitude falls about as r^(-1/2).
- `run_slab`: a plane wave from the left onto a conductor strip (slip time 1.5). Ahead of the strip E_y / (Z H_z) = 1.04 with phase 0: E and H move in step, as in a light wave. Inside the strip the wave decays with skin depth 3.40 cells (theory 3.43).

## Outputs

- `maxwell_lattice_checks.png`: (A) energy, (B) dispersion, (C) E and H along the slab run, (D) skin decay.
- `maxwell_lattice.mp4`: (A) the circular wave with every cell drawn (twist line, oval shear, idle particles); (B) the plane wave and the conductor; (C) H and E along the row. Twist, oval and particle motion are drawn larger and compressed so far cells show.
- `results.json`: the numbers above.

## Run

    python experiments/11_maxwell_lattice/lattice.py

About 3 minutes on one core, plus about 4 minutes for the movie.
