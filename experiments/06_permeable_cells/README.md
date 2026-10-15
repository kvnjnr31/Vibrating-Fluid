# Experiment 6: Flow From the Left Through Permeable Vibrating Cells (stopped)

Date: 2026-10-15. Code: `flow_through.py` in this folder; right-side outlet `outlet_right` in `lbm/boundaries.py`.

Status: set up and smoke-tested, then stopped before the full runs finished. No results yet.

## Setup

- The 4 x 4 array of oval cells from experiment 5, with an open stretch of one cell spacing on each side (box 6D x 4D, periodic top to bottom).
- Fluid enters on the left at q0 = 0.0025 and leaves on the right at fixed pressure.
- Cell walls are permeable as in the derivation: q = kappa (p_in - p_out) along the wall normal, kappa = 4.
- Aligned and geared lockings; 50 cycles of spin-up, then 60 cycles with tracers.

Smoke test (aligned, 15,000 steps): density stays within about 1% to 2% of 1; the flow through the walls swings between about -3 and +5 q0 over a cycle.

## Next

Following Maxwell's molecular vortices: replace the standing waves with waves that travel around each cell, so each cell turns and the fluid in the gaps acts as idle wheels.

    python experiments/06_permeable_cells/flow_through.py    # about 25 min on 2 cores
