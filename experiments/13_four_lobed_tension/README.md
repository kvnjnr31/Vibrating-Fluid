# Experiment 13: four-lobed tension around every shell

Experiment 12 (oval shells coupled by an ideal fluid) with the shell tension varying around each shell: T(theta) = T0 (1 + 0.24 cos 4 theta), the same in every cell. Tension is highest at 0 and 90 degrees and lowest at 45 and 135 degrees.

## Model

The shell energy is taken as that of a string bent around the circle, U = (1/2R) oint T (eta'^2 - eta^2) dtheta. For the two oval modes this gives:

- oval a (eta = a cos 2 theta, long axis at 0 or 90 degrees): stiffness K (1 - eps)
- oval b (eta = b sin 2 theta, long axis at 45 or 135 degrees): stiffness K (1 + eps)
- eps = 5 delta / 6 = 0.2, and no cross term.

The oval a bends the wall most where the tension is low, so it gets softer. The pattern also couples the ovals to 6-lobed shapes; those are left out. The fluid coupling is unchanged.

## Results

| | uniform tension (exp. 12) | four-lobed tension |
|---|---|---|
| long waves (Gamma) | both ovals at 1.12 | a at 1.005, b at 1.23: split |
| shortest left-right (K) | bands touch at 0.91 | 0.82 and 1.00: touching point opened |
| lower band | 0.80 to 1.12 | 0.71 to 1.01, mostly oval a |
| upper band | 0.91 to 1.47 | 0.96 to 1.61, mostly oval b |
| oval axis of each wave | set by the wave direction | locked near the cell axes (0 or 45 degrees) along the main directions; mixed in oblique directions |

Drive: centre oval at 22.5 degrees (equal a and b). Far-field ratio of oval b to oval a (cells beyond 6 rings):

| drive | bands crossed | far b / a |
|---|---|---|
| 0.88 | lower only | 0.48 (mostly a) |
| 0.99 | both (overlap 0.96 to 1.01) | 0.42 |
| 1.15 | upper only | 1.96 (mostly b) |

Crest direction (first-zone wavenumber dotted with the outward direction, power-weighted over 12 sectors): 0.88 gives -1.86 and 1.15 gives -1.58 (crests in, energy out: backward waves); 1.30 gives +1.30, 1.40 gives +2.65 and 1.50 gives +1.32 (crests out). At 1.50 the far field is almost pure oval b (b / a = 78): rows of 45 and 135 degree ovals that alternate up the lattice. Movie: `four_lobed_w150.mp4` (`tension.py --w 1.5`).

Reading: the lattice acts like a birefringent crystal. The same source sends out a different oval axis at different frequencies, and each wave keeps an axis tied to the cell tension pattern instead of to its direction. The filter is not perfect: waves travelling at oblique angles carry both ovals. The waves form narrow beams along a few lattice directions. The lower band falls away from long waves, so (as at 1.05 in experiment 12) the crests of the 0.88 wave should run inward while the energy runs outward.

## Outputs

- `four_lobed_checks.png`: (A) bands vs uniform tension, (B) lines of equal frequency, (C) a and b amplitude maps at 0.88, 0.99 and 1.15.
- `four_lobed.mp4`: drive 0.88, then drive 1.15. Panel B shows oval a and oval b side by side on one scale.
- `results.json`.

## Run

    python experiments/13_four_lobed_tension/tension.py

About 10 seconds for the checks and 6 minutes for the two time runs and the movie.
