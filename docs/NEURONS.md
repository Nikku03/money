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
