# Experiment 12: oval shells coupled by an ideal fluid (option B)

The shells keep their oval modes, r = R + a cos 2theta + b sin 2theta, and sit on a hexagonal lattice (spacing 1, R = 0.45, gap 0.1). The shell tension is a spring on each mode. The fluid inside and outside has no viscosity and cannot be compressed. The fluid starts at rest, so it never swirls (Kelvin). The fluid then only adds mass: each shell pushes the fluid, and the fluid pushes the neighbors. The equations are (M + m_s) q'' + K q = f, with M the added-mass matrix. Nothing loses energy.

`ideal.py` computes M exactly with complex potentials (multipoles up to order P, wall speed matched on every circle). `run.py` does the rest.

## Checks

- One cell alone: outer added mass pi rho R^2 / 2, exact.
- Two cells: coupling converges with multipole order (0.0915 at P = 8 and 12). Leading-order theory 3 pi rho R^6 / d^4 = 0.078; the rest comes from reflections between the two shells.
- Closed lattice (2107 cells), random start: energy constant to 1e-14 (Newmark average acceleration, no loss term).
- Drive at frequency 1.05: along the y axis the phase slope is -0.97 per cell; the band at that wavenumber is 1.040.

## Results

| | option A (Maxwell lattice, exp. 11) | option B (shells in ideal fluid) |
|---|---|---|
| frequencies that travel | 0 up to the grid limit (light line, w = c k) | two bands only: 0.80 to 1.12 and 0.91 to 1.47 |
| outside the bands | none | gaps: the wave dies near the source (above 1.47 and below 0.80) |
| long waves | speed c | almost stand still (group speed near 0 at small k) |
| fastest energy speed | 1 cell per time unit | 0.42 (upper band), 0.16 (lower band) |
| shape of the wave from one cell | circles | beams along preferred lattice directions |
| polarization | E across k (a vector, spin 1) | oval axes (a rank-2 shape, spin 2): axes along/across k or at 45 degrees |
| backward waves | no | yes, lower band: phase moves in while energy moves out |
| spin field H | carries the wave | not excited: an ideal fluid cannot change the swirl |

Reading: the ideal fluid makes the shells a lossless lattice of resonators coupled through their mass, like rings coupled by mutual inductance (magneto-inductive waves) or a photonic crystal. It does not give light. A light-like wave needs a spring between neighbors (Maxwell's elastic idle wheels); an incompressible fluid only shares mass.

## Outputs

- `ideal_fluid_checks.png`: (A) bands vs the option A light line, (B) lines of equal frequency, (C) energy, (D) wave amplitude from one driven cell.
- `ideal_fluid_cells.mp4`: (A) shells drawn as ovals (bulge drawn larger, line = long axis) with the fluid pressure, (B) the whole lattice, (C) the bands with the drive frequency.
- `results.json`, `blocks_R045.pkl` (bulk coupling blocks).

## Run

    python experiments/12_ideal_fluid_cells/run.py

About 1 minute for the runs plus 4 minutes for the movie.
