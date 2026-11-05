"""Builds maxwell_1861.ipynb: the Maxwell-only part (Parts 1-3) of the full study notebook, as one short notebook.

    python notebooks/build_maxwell_notebook.py      # full notebook first (source of the cells)
    python notebooks/build_maxwell_1861.py
"""
import os
import subprocess
import sys

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(HERE, "build_maxwell_notebook.py")], check=True, capture_output=True)
full = nbf.read(os.path.join(HERE, "maxwell_to_vibrating_cells.ipynb"), 4)
src = [c for c in full.cells]
start = next(i for i, c in enumerate(src) if c.cell_type == "markdown" and "## Part 1." in c.source)
stop = next(i for i, c in enumerate(src) if c.cell_type == "markdown" and "## Part 4." in c.source)

title = nbf.v4.new_markdown_cell(r"""# Maxwell's Molecular Vortices, Worked Through

A short study notebook on J. C. Maxwell, *On Physical Lines of Force* (Philosophical Magazine, 1861–62). It covers the first three parts of the paper, one idea each, and ends where the Vibrating-Fluid experiments begin.

| Part | Maxwell's idea | What you compute | Course links |
|---|---|---|---|
| 1 | Spinning vortices make a stress (Part I) | The squeeze inside a wire, from stresses alone | Theory of Elasticity, PDEs |
| 2 | Idle wheels between vortices carry the current (Part II) | The current for four spin patterns, as in experiment 9 | Discrete Mathematical Modeling, PDEs |
| 3 | Elastic cells carry waves at the speed of light (Part III) | The lattice wave equation, its speed and its dispersion | PDEs, Numerical Matrix Analysis, Linear Statistical Models |

**How to use it.** Read a part, run its cells, then try the exercises marked **Try it**. Each part takes about an hour. All cells together run in a few seconds and need only `numpy`, `scipy` and `matplotlib`.""")

end = nbf.v4.new_markdown_cell(r"""---
## What comes next

Part 3 ends with Maxwell's lattice: rigid wheels, springy idle particles, and a lossless wave at speed $c$. The Vibrating-Fluid experiments then swap the wheels for vibrating shells:

- **Experiment 12** couples oval shells through an ideal fluid. The fluid shares mass between cells but gives no spring, so the waves form a narrow band with gaps and backward waves instead of a light line.
- **Experiment 13** adds a four-lobed tension around each shell and splits the two oval modes, like a crystal that splits light by polarization.

The full study notebook in the repository, `notebooks/maxwell_to_vibrating_cells.ipynb`, continues with those steps and adds nonlinear dynamics (Duffing shells, chaos, breathers) and statistics (equipartition, Maxwell's 1860 speed law, bootstrap, PCA, Fermi–Pasta–Ulam–Tsingou).

### Reading

- J. C. Maxwell, *On Physical Lines of Force*, Philosophical Magazine, 1861–62, Parts I–IV.
- D. M. Siegel, *Innovation in Maxwell's Electromagnetic Theory: Molecular Vortices, Displacement Current, and Light*, Cambridge University Press, 1991.
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*.""")

nb = nbf.v4.new_notebook()
nb.cells = [title, src[1]] + src[start:stop] + [end]
nb.metadata = full.metadata
for c in nb.cells:
    if c.cell_type == "code":
        c.outputs, c.execution_count = [], None
out = os.path.join(HERE, "maxwell_1861.ipynb")
nbf.write(nb, out)
print("wrote", out, len(nb.cells), "cells")
