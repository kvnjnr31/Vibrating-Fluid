# Experiment 6: Flow From the Left Through Permeable Vibrating Cells (stopped)

Date: 2026-10-15. Code: `flow_through.py` in this folder; right-side outlet `outlet_right` in `lbm/boundaries.py`.

Status: the aligned run finished (data in `run_aligned.npz`, not in git); the geared run was stopped partway. No figure or movie yet.

## Setup

- The 4 x 4 array of oval cells from experiment 5, with an open stretch of one cell spacing on each side (box 6D x 4D, periodic top to bottom).
- Fluid enters on the left at q0 = 0.0025 and leaves on the right at fixed pressure.
- Cell walls are permeable as in the derivation: q = kappa (p_in - p_out) along the wall normal, kappa = 4.
- Aligned and geared lockings; 50 cycles of spin-up, then 60 cycles with tracers.

Smoke test (aligned, 15,000 steps): density stays within about 1% to 2% of 1; the flow through the walls swings between about -3 and +5 q0 over a cycle.

## Aligned run

- Net flux: q0 times the box height at the inlet side, and 1% more downstream, from the 1% density drop across the array (mass flux is conserved).
- Flow through the walls: the outflow summed over all 16 cells is 3.5 times the incoming flux. The flow crosses 4 columns of cells in series, so on average about 87% of the fluid passes through each column of cells rather than around it.

## Next

Following Maxwell's molecular vortices: replace the standing waves with waves that travel around each cell, so each cell turns and the fluid in the gaps acts as idle wheels.

    python experiments/06_permeable_cells/flow_through.py    # about 25 min on 2 cores
