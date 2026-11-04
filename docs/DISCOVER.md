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
