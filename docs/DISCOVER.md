# Equation discovery on a simulated Earth (`brain/discover`)

A side experiment, unrelated to the market bank. `earth.py` simulates a planet: its magnetic field,
its gravity, and the satellites and charged particles that move in them. `engine.py` is shown only
measurements (magnetometer readings, positions over time) and a library of candidate terms, most of
them wrong, and has to recover the laws. It never reads the world's constants; they are used only to
grade what it found.

```
python -m brain discover earth            # every experiment and control, ~5 minutes on 4 cores
python -m brain discover earth --quick    # skips the slow q/m = 500 run
```

The engine is not the spiking circuit of `docs/NEURONS.md`. A circuit with hand-assigned weights
cannot find an equation it was not wired with. This is sparse regression in the style of SINDy
(Brunton, Proctor & Kutz, PNAS 2016), and it obeys this repo's rule: nothing counts until it passes
a test on data it never saw.

## The world (normalized units: Earth radius 1, equatorial surface field 1, GM 1)

- **Field:** a dipole tilted 11° plus a quadrupole (about a tenth of the dipole at the surface).
  Outside the planet there are no currents, so ∇·B = 0 and ∇×B = 0.
- **Gravity:** a = −GM r̂/r².
- **Protons:** a = (q/m) v × B − GM r̂/r², with q/m = 50. They are integrated with the Boris scheme
  and they gyrate, bounce along field lines and drift, as radiation-belt particles do.

## The engine

1. **Families.** Terms come in families: one shape at one power of r, in each orientation. The
   engine keeps or drops whole families, because space has no preferred axes.
2. **Grow.** Group orthogonal matching pursuit adds the family that best explains what is left.
3. **Choose.** By BIC, then prune, then swap families when BIC drops by more than 10 log n.
4. **Stable.** A family must be chosen in 90% of 30 bootstrap resamples.
5. **Test.** Held-out data, usually farther from Earth than anything fitted (extrapolation).
   A law is FOUND if its test error is under 5%, or no more than 1.25× its fitting error, since noise
   sets a floor no law can beat; and under 30% in any case.

Three things failed first, and each fix is described in `engine.py`:

- Thresholding a full least-squares fit (plain SINDy) kept all 85 terms: over 1-2.5 Earth radii,
  r⁻¹ to r⁻⁵ look alike.
- Choosing terms one axis at a time made the quadrupole unstable.
- A "within 5% of the best error" pruning rule dropped a real quadrupole family once noise dominated
  the error.

## What it found (`python -m brain discover earth`)

| experiment | result | recovered | truth |
|---|---|---|---|
| field, no noise | FOUND: dipole (r⁻³, both families) + quadrupole (r⁻⁴, both families) | tilt 11.00°, dipole and quadrupole exact, monopole 0 | 11°, monopole 0 |
| field, 1% noise | FOUND: dipole + half of the quadrupole | tilt 10.84°, dipole error 0.006; quadrupole off by 70% | |
| field, 5% noise | FOUND: same (held-out error 5.3%, at the noise floor) | tilt 10.91° | |
| gravity from satellites | FOUND: a = −1.000 r̂/r² | GM 1.000, inverse square | 1 |
| protons, unaided | NOT FOUND: a = 49.93 v×B alone misses gravity (13% held-out) | q/m 49.93 | 50 |
| protons, orbits 1.5-4 radii | NOT FOUND: gravity chosen in 73% of resamples, under the 90% bar | q/m 49.95 | |
| protons, gravity carried over from satellites | FOUND: a + r̂/r² = 50.000 v×B (3.7% held-out) | q/m 50.000 | 50 |
| Maxwell, from field derivatives | FOUND, 4 relations: ∇·B = 0 and the three components of ∇×B = 0 | held-out error 0.03-0.04% | |
| invariants, q/m 50 | FOUND: v∥² + v⊥² − 2/r constant (energy); v⊥²/\|B\| varies 10.7%, so not conserved | | |
| invariants, q/m 500 | FOUND: energy again; v⊥²/\|B\| varies 1.4%, so conserved: the first adiabatic invariant | | |

Controls, all NOT FOUND as they should be:

| control | result |
|---|---|
| field readings shuffled | nothing fits (98% error) |
| field library without r⁻³ | a blend of r⁻² and r⁻⁴ fits the training range to 1.1%, then misses by 75% farther out |
| proton accelerations shuffled | no term chosen |
| force library without v×B | 100% error |

What this shows:

- **The dipole is the robust discovery.** Its power (r⁻³), its shape and its tilt (within 0.2°) are
  recovered at every noise level tried, and the monopole term is always zero.
- **The quadrupole's weaker half is lost from 1% noise on.** Over 1-2.5 Earth radii, a blend of r⁻¹,
  r⁻³ and r⁻⁵ can stand in for r⁻⁴ within the noise, and the stability test refuses to choose
  between them. That is a limit of these measurements, not of the search. I did not loosen the bar
  until it passed.
- **Small forces hide.** Gravity is about 8% of a proton's acceleration here, and alone the engine
  could not pin its power. Carrying the law found from satellites over to the protons, as science
  builds on earlier results, settled it: what remains is exactly the Lorentz force with q/m = 50.
- **Extrapolation is what catches a wrong law.** The library without r⁻³ fit its own range to 1%
  and failed by 75% outside it.
- **A physical prediction, tested.** The magnetic moment v⊥²/|B| is only conserved when a particle's
  gyration is small next to the scale over which the field changes. At q/m = 50 it is not (10.7%),
  and the engine says so. At q/m = 500 it is (1.4%), and the engine finds it.
- **A blemish.** At q/m = 500 one form of the energy law carries small junk terms (r, |B|, r²) with
  1.2% held-out error; the clean form was also found.
- **The library sets the limit.** The engine can only find what its library can express. Every term
  it was offered was written by a person, and a law outside the library is reported NOT FOUND,
  never approximated into a false discovery.

# Against the field's standard tools (`python -m brain discover compare`)

Every system gets the same training rows, the same 85-term (field) or 17-term (motion) library, and
the same held-out test rows farther from Earth. Each is graded against the terms the world is made of.

The systems:
- **PySINDy 2.1** (the reference SINDy implementation): STLSQ, SR3 (L0), SSR, FROLS and E-SINDy.
  None of them chooses its own sparsity, so each gets a sweep of its knob and the lowest-BIC model on
  the training rows, the same criterion this engine uses.
- **scikit-learn's LassoCV**, as the classical baseline.
- **PySR 1.x** (Cranmer 2023), which searches formulas over the raw variables with no library.

These are recorded runs, 3 seeds per task, on 4 shared cores; seconds are wall time per run. "Exact"
means the chosen terms are the true ones, no more and no fewer.

| task | ours | STLSQ | SR3 | SSR | FROLS | E-SINDy | LassoCV |
|---|---|---|---|---|---|---|---|
| field, no noise: exact / false terms | **3/3** / 0 | **3/3** / 0 | **3/3** / 0 | 0/3 / 6.7 | 0/3 / 13.3 | 2/3 / 22.3 | 0/3 / 10.7 |
| field, 1% noise: true of 16 / false / held-out error | 9.3 / **0** / 2.2% | 15.3 / 57 / 48% | 15.7 / 59 / 48% | 15.3 / 54 / 46% | **16 / 0 / 1.0% (3/3 exact)** | 15.3 / 63 / 43% | 16 / 12 / 1.1% |
| field, 5% noise: true / false / held-out error | 7.7 / **0** / 5.8% | 15.7 / 64 / 243% | 16 / 65 / 243% | 15.7 / 58 / 242% | **15 / 1 / 5.0%** | 15.3 / 63 / 224% | 16 / 23 / 14% |
| gravity from satellites: exact | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 |
| protons unaided: exact (a = 50 v×B − r̂/r²) | **0/3** (gravity unstable, NOT FOUND) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| protons, gravity carried over: exact | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| control, shuffled field: verdict / held-out error | **NOT FOUND** ×3 | 85 terms / 4574% | 85 / 4574% | 1 / 100% | 1.7 / 118% | 85 / 4574% | 7 / 117% |
| control, library without r⁻³: verdict / held-out error | **NOT FOUND** ×3 / 71% | 47 terms / 28% | 52 / 28% | 42 / 28% | 32 / 26% | 55 / 28% | 40 / 42% |
| mean seconds, field tasks | 180-620 | 50-68 | 25-32 | 26-30 | **6-9** | 250-340 | 1 |

PySR (2 minutes per target, raw variables, no library, seed 0):

| target | PySR's formula | truth | held-out error |
|---|---|---|---|
| a_x of satellites from (x, y, z) | −0.997 x / (√(x²+y²) · r²) | −x / r³ | 1.9%: right fit, wrong formula |
| a_x of protons from (v, B), gravity carried over | 49.98 v_z·(−B_y) + 49.99 v_y B_z | 50 (v_y B_z − v_z B_y) | 3.7%: **exact** |
| B_z from (x, y, z), no noise | a 6-constant rational function | dipole + quadrupole | 3,900%: blows up near r ≈ 0.9 |

The satellites orbit near the equator, where √(x²+y²) ≈ r, so PySR's distance from the axis fits as
well as the distance from the centre. Nothing in a free formula search rules that out; a library of
radial terms does.

What this shows, plainly:

- **FROLS beats this engine.** It is greedy forward selection like this one, but without families,
  swaps or bootstrap. It recovered the full field law exactly at 1% noise, and all but one term at 5%,
  in 6-9 seconds against this engine's 3-10 minutes. It also found gravity on unaided protons, where
  this engine refused.
- **This engine is the most conservative.** In every real task it chose **zero false terms**, and it is
  the only system that says NOT FOUND. That verdict was right on both controls, where every other
  system returned a confident-looking law. The price is missed truths: half the quadrupole under
  noise, and gravity on protons, which a plain BIC choice gets right 3 times out of 3. Its 90%
  bootstrap-stability bar is too strict for small effects.
- **Thresholding fails under noise, as found earlier.** STLSQ, SR3, SSR and E-SINDy take in 54-65
  false terms at 1-5% noise and miss by 43-243% when extrapolating: the collinear r⁻¹ to r⁻⁵ library
  defeats them. E-SINDy's bagging did not save it.
- **LassoCV keeps every true term but always adds false ones** (10-23), a known L1 behaviour.
- **PySR is the most capable when a law is compact** (the Lorentz force, exact, with no library), and
  the weakest on the multi-term field, inside a 2-minute budget. It can also fit a wrong formula that
  the sampled data cannot tell apart from the right one.
- **Caveats.** The PySINDy optimizers' thresholds were selected by BIC, not tuned by hand, and an
  expert's tuning might do better. PySR had 2 minutes per target, and longer runs may find the field.
  The tasks are mine and this engine was developed on them, which favours it. FROLS won anyway.

The next engine would take FROLS's path and keep this one's verdict. That would be a declared new
version, run against the same benchmark; this one stays as recorded.
