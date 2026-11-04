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
