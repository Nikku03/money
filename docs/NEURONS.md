# Hand-wired spiking neurons (`brain/neurons`)

A side experiment, unrelated to the market bank: a circuit of 107 spiking neurons that plays Pong with
**no learning**. Every weight, delay and connection is assigned by a written rule (`brain/neurons/pong.py`).
The idea comes from DishBrain (Kagan et al., *Neuron* 2022), where cultured neurons learned Pong. Here
the designer does the learning, and the circuit's behaviour can be read off its wiring.

```
python -m brain neurons play --compare --games 3 --frames 800   # circuit vs still vs random paddle
python -m brain neurons play --noise 2                          # membrane noise
python -m brain neurons play --kill UP1,DOWN1                   # targeted lesion
python -m brain neurons play --html pong.html                   # a replay page: court, circuit, raster, voltages
python -m brain neurons wiring UP3                              # who feeds a neuron, whom it feeds, weights, delays
python -m brain neurons topology                                # directed simplices by dimension
```

## What a neuron is here

- **Voltage over time.** Izhikevich cells: membrane voltage `v` and recovery `u`, stepped every 0.25 ms.
  A spike at +30 mV resets `v`. RS (regular spiking), FS (fast inhibitory) and FE (fast-recovering
  excitatory) parameter sets.
- **Synapses are conductances.** Excitatory ones reverse at 0 mV and decay over 5 ms; inhibitory ones
  reverse at -80 mV and decay over 10 ms. Each neuron has one sign (Dale's law).
- **Delays.** Every synapse has its own axonal delay, so *when* a spike arrives matters as much as
  whether it does.
- **Connections are explicit.** `Circuit.inputs(n)` and `Circuit.outputs(n)` name the senders and targets.

## The circuit

| layer | neurons | rule |
|---|---|---|
| eyes | 8 BALL, 8 PADDLE | A 1 ms electrode pulse every 20 ms frame, place-tuned: the cells nearest the object fire. |
| compare | 64 C(i,j) | 0.08 from BALL i plus 0.08 from PADDLE j. Threshold is about 0.135, so a cell fires only on coincidence. |
| decide | | i > j feeds UP, i < j feeds DOWN, i = j feeds HOLD (inhibits both sources). |
| amplify | UP, DOWN: 12 each | Each is a **directed 11-simplex**: j feeds every k > j (66 synapses). Neuron k's weights share 0.3 across its k inputs. The delay from j to k is (k-j-1)·2.5 + 1 ms, so all inputs to k arrive together. |
| choose | INH_UP, INH_DOWN | Each source silences the other clique's source (winner-take-all). |
| move | | Each sink spike moves the paddle 1/32, at most one step per frame. |

"11-dimensional" means what it does in Reimann et al. (2017, *Front. Comput. Neurosci.*): the
dimension of a directed clique, twelve neurons all wired forward. It is not a dimension of space.
`topology.simplices` counts them; this circuit has two 11-simplices.

## Recorded results (3 games × 800 frames, seeds 0-2)

| condition | hits / rallies |
|---|---|
| still paddle | 10/32 (31%) |
| random paddle | 10/33 (30%) |
| the circuit | 21/21 (100%) |
| kill neuron 2, or 9, or 5-7 of both cliques | 21/21 |
| kill neuron 1 of both cliques | 19/25 (76%) |
| kill neurons 3, 5, 7, 9 | 18/24 (75%) |
| random 20% / 30% / 50% of clique middles killed | 70% / 46% / 21% (800 frames) |
| membrane noise 2 / 4 / 6 | 83% / 50% / 30% |

What this shows:

- The clique's weak point is its start. Neuron 1 hears only the source, so losing it breaks most
  volleys. Later neurons hear many inputs and can be lost.
- Noise near 4 already fires cells on its own, and play falls apart there.
- The game is easy, and perfect play comes from the wiring rule, not from the neurons discovering anything.

# Catch: a game that needs prediction (`brain/neurons/catch.py`)

```
python -m brain neurons catch --compare --games 5        # every controller, same serves (seeds 10-14)
python -m brain neurons catch --html catch.html          # replay page: court, guess vs truth, raster
python -m brain neurons wiring --game catch P3_9_+2      # one prediction cell: its inputs and its landing
```

Each ball is served from the left at a random height and angle. It crosses in 30 frames, with vertical
speed up to 0.05 per frame, and bounces off the top and bottom walls. The paddle (half-height 0.1) has
a top speed of 0.025 per frame, half the ball's. A paddle that chases the ball's current height arrives
late, so the circuit has to predict where the ball will land.

| layer | cells | rule |
|---|---|---|
| eyes | 16 height, 8 distance, 8 paddle | A fixed electrode pulse to the cell (or two at a band edge) holding the object, so every eye spike has the same latency. |
| motion | 146 R(i,s) | Hears Y_i directly and Y_(i-s) through a 121 ms delay line. It fires when the ball moved s bands in 6 frames (a Reichardt detector). |
| predict | 1,168 P(x,i,s) | R(i,s) plus distance cell x, delayed 8 ms to arrive together. One synapse to the landing cell for fold(y + v·frames left), the bounces folded in. |
| land | 8 L | These take the place of Pong's ball cells and feed the same compare → 11-simplex → paddle motor. The paddle cells' delay is 18.5 ms, so their spikes meet the landing spikes. |

## Recorded results (5 games × 900 frames, seeds 10-14, 150 serves; tuning used seeds 0-2)

| controller | caught |
|---|---|
| still paddle | 30 (20%) |
| random paddle | 29 (19%) |
| Pong circuit (steers to the ball's current height) | 64 (43%) |
| this circuit, landing wired as if there were no walls | 86 (57%) |
| **this circuit** | **113 (75%)** |
| oracle: true landing, same top speed, no neurons | 147 (98%) |
| this circuit, membrane noise 2 | 48 (32%) |

Median landing-guess error is 0.065 with the ball far away, 0.042 halfway and 0.03 near. Without
the folding, the far-off error is 0.168.

What this shows:

- Prediction beats chasing by 32 points, and folding the walls into the wiring is worth 18 of them.
- The gap to the oracle comes from the motor. The guess arrives about 2 frames late and jitters
  between bands, so the paddle overshoots or starts late. In this game the motor loses more than the
  prediction does.
- It is fragile to noise, which Pong was not (83% at noise 2). 1,168 prediction cells wait one input
  short of threshold, noise fires stray ones, and each stray one votes for a wrong landing.
- Three things found while tuning, kept because they say something about spike timing:
  - A graded eye code broke the motion detectors: a cell's latency depended on where the ball sat in
    its band, so a spike now and one from 120 ms ago no longer coincided.
  - A code firing two eye cells too often smeared the landing vote (61%).
  - A code too narrow left the paddle blind at band edges: it never moved (18%).

# School: learning alone and from others (`brain/neurons/learn.py`)

```
python -m brain neurons school --seeds 3 --html school.html    # all five modes, three schools each (~17 min)
```

The world changes: **gravity** (vy loses 0.0015 per frame). The hand-wired straight-line rule now
predicts the wrong landing. One layer becomes plastic: each prediction cell P connects to all 8 landing
cells. Its eight weights are its expectation of the landing; the strongest is 0.2, the rest in proportion.

- **From itself.** When a ball crosses the paddle line, every P cell that fired during that serve (from
  its 7th frame on) moves 30% of the way toward the landing it felt: `p <- p + 0.3 (onehot - p)`. It is
  a delayed teaching signal, like the cerebellum's climbing fibres.
- **From others.** Four circuits play their own serves. After each round of 10 serves, each learns from
  every peer's serves as if they were its own (0.3 × trust) and copies its knowledge (0.2 × trust). They
  are wired alike, so a peer's serve is one it could have seen.
- **Trust is earned.** Circuit A3's landing sense is broken: it reports every landing mirrored. Trust in
  a peer is the lower of two tests: how well its knowledge predicts *my* last 40 serves, and how often
  its reported landings agree with what I'd predict. Each is scaled from 40% (zero) to 100% (one). This
  is the bank's own rule for sources: weight comes only from a tested record.

## Recorded results (3 schools per mode, 15 rounds × 10 serves; exams on 200 held-out serves)

The **early** exam asks for the exact landing band from the predictions made in the ball's far half,
which is what a paddle at half the ball's speed needs. Under gravity the rule scores 12.5% (chance is
12.5%); in its own no-gravity world it scores 49%. The **lenient** exam (within one band, whole serve)
gives the rule 92% under gravity and saturates near 98% for every learner, so it cannot separate modes.

| mode | early: start | early: end | broken A3: end | lenient: end | honest catch, final exam |
|---|---|---|---|---|---|
| fixed rule, no learning | 12% | 12% | 12% | 92% | 56% |
| learns alone | 12% | 32% | 5% | 98% | 67% |
| learns from all, trusts all | 12% | 36% | 38% | 92% | 60% |
| **learns from all, trust earned** | 12% | **41%** | 5% | 99% | **71%** |
| learns from all, no broken peer | 12% | 42% | – | 99% | 72% |

Early-exam end per school: alone 34/30/32%, naive 38/34/35%, trusted 43/39/41%, honest 43/39/43%.
Trust in the trusted mode: honest → honest 0.95-0.98, honest → broken 0.00, in every round of every school.

What this shows:

- Learning from experience works. Alone, a circuit's early guess goes from 12% to 32% exact in 150
  serves, and its catch rate from 56% to 67%.
- Learning from others works faster and goes further: 41-42% exact with honest peers. Each circuit
  gets four times the experience.
- Trusting everyone costs. The broken peer's mirrored lessons cut the honest circuits' gain. Their
  early guess still rises (36%), because the honest majority outvotes the liar. But their whole-serve
  knowledge falls back to the rule's 92%, and their catch rate (60%) ends *below* learning alone (67%).
- Earned trust recovers almost all of it. It matches a school with no broken peer (41% vs 42%, catch
  71% vs 72%) and gives the broken peer zero weight from round 1 on.
- What trust cannot fix: the broken circuit. It tests peers against its own mirrored sense, so it
  trusts no one and learns itself into 5%. Under naive trust, the others had dragged it up to 38%. A
  learner that tests sources against a broken sense rejects the people who could correct it.
- The learning is small and tuned by hand: the rates, the trust floor of 0.4, 4 circuits, one fault
  type, one world change. These are recorded results for this setup, not general claims.

# Runner: a three-lane endless runner (`brain/neurons/runner.py`)

An original game in the endless-runner genre. Rows of obstacles come down three lanes: a **wall**
means switch lanes, a **hurdle** means jump, a **beam** means duck. Every row leaves a lane without a
wall. Speed rises from 1 to 2.4 units per frame, and a crash restarts the run at the start speed. A
lane switch has a 4-frame cooldown; a jump or duck lasts 8 frames.

```
python -m brain neurons run --compare                 # every controller, 4 fresh tracks x 5,000 frames
python -m brain neurons run --games 1 --frames 1200 --html runner.html
python -m brain neurons wiring --game run GO_R20
```

| layer | cells | rule |
|---|---|---|
| eyes | 15 | LANE_l (the runner is in lane l); WALL_l (wall arrives in 3-12 frames); CLEAR_l (no wall within 13 frames: an OFF cell, since a spike can't signal absence); HURDLE_l and BEAM_l (arrive in 2-5 frames). Obstacle cells are tuned to **time-to-contact**, like the locust's looming detector, so they fire the same number of frames ahead at any speed. |
| decide | 12 | GO_R(l) = LANE_l + WALL_l + CLEAR_(l+1), 0.055 per input: two inputs make 0.11, under threshold, three fire. GO_L mirrors it; GO_R2/GO_L2 handle escapes two lanes away. JUMP(l) = LANE_l + HURDLE_l, and DUCK likewise (0.08 each). |
| act | 4 × 12 | LEFT, RIGHT, JUMP and DUCK are each a directed 11-simplex (`pong.add_clique`); each sink spike is a command. LEFT and RIGHT race through inhibitory cells, and GO_L axons are 1.5 ms longer, so the circuit goes right when both sides are clear. |

77 neurons, 310 synapses, four 11-simplices. Commands come out about 2 frames after the eyes see,
so every window leads the needed action by 2 frames.

## Recorded results (4 tracks × 5,000 frames, seeds 20-23; tuned on seed 0)

| controller | crashes | rows per crash | top speed |
|---|---|---|---|
| standing still | 335 | 1.5 | 1.23 |
| random moves | 319 | 1.7 | 1.25 |
| same circuit, eyes tuned to distance (at the start speed) | 38 | 24.9 | 2.23 |
| **this circuit** | **0** | all 974 rows | 2.40 |
| oracle: true state, no delay | 0 | all 974 rows | 2.40 |
| this circuit, membrane noise 2 | 191 | 3.8 | 1.66 |
| this circuit, membrane noise 4 | 256 | 2.5 | 1.44 |

What this shows:

- On these tracks the circuit plays as well as the oracle. The game is fully observed and the rules
  are simple, so a correct wiring rule is enough. That says more about the game than about the neurons.
- Timing in the eye is what scales with speed. The same circuit with distance-tuned eyes crashes 38
  times, almost all into hurdles and beams: as speed rises, its fixed-distance windows fire too late.
- My first wall window (3-9 frames) lost every crash the same way. An escape two lanes away needs two
  switches 4 frames apart, and the lane it ends in may hold a hurdle the runner then sees too late.
  Widening the window to 12 frames fixed it.
- It is fragile to noise, more than Pong (83% at noise 2). Every stray command acts and can be fatal,
  and the three-input cells sit one input from firing, so noise fires them with only two.

This is not Subway Surfers or any other existing game: it is an original runner with the genre's
mechanics, and the circuit does not connect to any real game.

# The ladder of cells: rich cells against simple ones (`circuit.CELLS`)

He et al. (2024, *Nat Comput Sci* 4:584-599, doi:10.1038/s43588-024-00674-9) argue that a network can
get its abilities two ways: more simple cells ("external complexity") or richer cells ("internal
complexity"), and that a network of Hodgkin-Huxley cells matches a bigger network of leaky
integrate-and-fire cells. Our circuits put their complexity in the wiring (delays, coincidence cells,
directed cliques). The paper had three things they lacked: cells built from real physics, a measured
exchange rate between rich and simple cells, and training on tasks nobody wired. These are added below.
Nothing here is the paper's code; every number is measured on this simulator.

```
python -m brain neurons play --cells hh          # any game on any cell: --cells lif|izh|hh
python -m brain neurons ladder                   # every game on each cell, clean and under noise (~45 CPU-min)
python -m brain neurons exchange                 # how many LIF cells copy one Hodgkin-Huxley cell (~40 CPU-min)
python -m brain neurons train                    # networks of each cell trained on a timing task (~70 CPU-min)
```

## The three cells

| cell | state variables | what it is |
|---|---|---|
| `lif` leaky integrate-and-fire | 1 | voltage leaks to -70 mV; at -50 mV it fires, resets to -65 mV and rests 1 ms. Its membrane time constant (1.25 ms) is the Izhikevich cells' near rest, so the two differ only in what happens at threshold |
| `izh` Izhikevich (the default) | 2 | voltage and recovery; reproduces how real cells fire, not why |
| `hh` Hodgkin-Huxley | 4 | voltage and the sodium (m, h) and potassium (n) gates of the squid giant axon, at its measured conductances. No reset by hand: the spike, threshold and refractory period come from the channels. Exponential Euler at 0.05 ms inside each 0.25 ms step |

Synapses (conductances with reversal potentials), delays, electrode pulses and noise are shared. So the
same wiring means the same thing on each cell, every cell has two gains (`circuit.GAINS`, recomputed by
`circuit.calibrate()`): synaptic, so the weight of one event that just fires a resting cell equals the
Izhikevich FE cell's (0.124); and current, so the 1 ms electrode pulse that just fires it equals the
Izhikevich RS cell's (19.0). Hodgkin-Huxley: 0.4705 and 0.3642. LIF: 4.957 and 1.910.

What stayed: Izhikevich is the default, and every result recorded above reproduces exactly on it (Pong
21/21, 83% at noise 2, 46% with 30% of the clique middles gone; Catch 113/150; Runner 0 crashes, 191 at
noise 2 and 256 at noise 4). Wiring, topology, kills, lesions, noise and the school take any cell.

## 1. The same hand-wired games on each cell (`ladder`)

Wired exactly as above; seeds as above (Pong 0-2 x 800 frames, Catch 10-14 x 900, Runner 20-23 x 5,000).

| game, condition | LIF (1) | Izhikevich (2) | Hodgkin-Huxley (4) |
|---|---|---|---|
| Pong, clean | 21/21 (100%) | 21/21 (100%) | 21/21 (100%) |
| Pong, noise 2 | 21/21 (100%) | 20/24 (83%) | 22/24 (92%) |
| Pong, noise 4 | 18/24 (75%) | 14/28 (50%) | 15/28 (54%) |
| Pong, 30% of clique middles killed | 21/21 (100%) | 12/26 (46%) | 16/27 (59%) |
| Catch, clean | 42/150 (28%) | 113/150 (75%) | **130/150 (87%)** |
| Catch, noise 2 | 34/150 (23%) | 48/150 (32%) | 29/150 (19%) |
| Catch, noise 4 | 29/150 (19%) | 40/150 (27%) | 37/150 (25%) |
| Runner, clean (crashes) | 37 | **0** | 44 |
| Runner, noise 2 | 193 | 191 | 210 |
| Runner, noise 4 | 229 | 256 | 279 |
| CPU time against Izhikevich | 1.1-1.4x | 1x | 8-10x |

What this shows:

- One wiring runs on all three cells: every game plays well above a still paddle on every cell.
- The Hodgkin-Huxley cell catches more (87% against 75%) with the same landing guess (off by 0.062 far,
  0.043 mid, 0.030 near, against 0.065, 0.042, 0.029 on games 10-11): its gain is in the motor stage,
  where the docs above place the circuit's loss. Under noise it falls further (19% at noise 2).
- The LIF cell is the hardest to break in Pong (100% with 30% of the cliques gone, 75% at noise 4) and the
  worst at Catch (28%, a still paddle catches 20%). Its landing guesses are worse (0.084 far, 0.069 near):
  it fires 0.25-1 ms after its input where the Izhikevich cell takes 2-5 ms, and the motion detectors'
  121 ms delay lines and 8 ms alignments were timed for the slower cell.
- Catch and Runner were tuned on the Izhikevich cell, so the other two run wiring timed for another cell.
  That is no fair race for them; the training below is.
- The physics costs: a Hodgkin-Huxley circuit takes 8-10 times the CPU of the same Izhikevich circuit.

## 2. The exchange rate: how many simple cells copy one Hodgkin-Huxley cell (`exchange`)

One Hodgkin-Huxley cell is driven by a fluctuating current (noise with a 3 ms memory around a mean that
jumps every 250 ms between -4 and 14 uA/cm2). Candidates copy its spikes: a circuit of k LIF cells (one
output and k - 1 helpers; every cell hears the input with its own gain and sign, each helper feeds the
output through its own synapse, each output spike kicks every helper) and one Izhikevich cell, every
parameter fitted, plus an output delay of up to 3 ms. The fit: the cross-entropy method, best of three
runs of 80 generations x 128 candidates, on 4 s of input. The score: the coincidence factor Gamma
(Kistler et al. 1997), spikes matched within 2 ms beyond chance, on 4 s of input never fitted (1 is a
perfect copy, 0 chance), for three seeds.

| model | cells | state variables | Gamma on unseen input | spikes vs HH | silent at rest | rebound |
|---|---|---|---|---|---|---|
| Hodgkin-Huxley (target) | 1 | 4 | 1 | 1x | yes | yes |
| LIF x1 | 1 | 1 | 0.34 (0.32, 0.38, 0.31) | 2.1x | 2/3 | 0/3 |
| Izhikevich x1 | 1 | 2 | 0.47 (0.50, 0.48, 0.43) | 2.0x | 0/3 | 0/3 |
| LIF x2 | 2 | 3 | 0.53 (0.53, 0.53, 0.53) | 1.7x | 3/3 | 0/3 |
| LIF x4 | 4 | 7 | 0.50 (0.50, 0.52, 0.48) | 1.6x | 2/3 | 0/3 |
| LIF x8 | 8 | 15 | 0.53 (0.51, 0.56, 0.52) | 1.4x | 3/3 | 1/3 |

CPU per neural second, one cell circuit at a time: Hodgkin-Huxley 1.15 s, LIF circuits 0.03-0.09 s,
Izhikevich 0.07 s.

What this shows:

- Two LIF cells copy the Hodgkin-Huxley cell better than one Izhikevich cell (0.53 against 0.47), and one
  Izhikevich cell better than one LIF cell (0.34): richer inside does buy what more cells outside buy.
- None came near a copy. More than two LIF cells did not help on unseen input (0.50 at four, 0.53 at
  eight), though they fitted the training input a little better (eight cells 0.61-0.66, two cells
  0.59-0.60): the extra cells learn that input, not the cell. Every fitted model fires 1.4-2.1 times as many spikes as the
  target.
- Rebound after inhibition, which Hodgkin-Huxley does by its channels, appeared in 1 of the 12 fitted LIF
  circuits and in no Izhikevich fit.
- The paper reports a small LIF circuit reproducing one Hodgkin-Huxley cell. Ours did not get there with
  plain LIF cells, this circuit shape and this search; that limit may be ours rather than the cells'.
  These are recorded results for this setup, not general claims.
- In this code the physics is expensive: one Hodgkin-Huxley cell costs 12-35 times a small LIF circuit
  to simulate.

## 3. Training on a task nobody wired (`train`)

Results to follow: the first run is in progress.
