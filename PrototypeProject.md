# One-day supervisor prototype

## Goal

Build a small, separate Python prototype that demonstrates the full idea:

**track → observations → NEAT driver → car movement → fitness → evolution → replay**

The supervisor should see a car driving, understand what evolves, and see recorded results from a short training run. A completed lap by an evolved agent is a stretch goal, not a promise for one day.

This is a demonstration of the learning pipeline, not the finished BSP simulator or evidence of realistic F1 performance.

**Use this file as the prototype's single working checklist and technical reference.** All choices needed to start are below. Update its checkboxes and record any changed settings here as you work; other project documents are background, not required reading for this demo.

## Scope for the day

- One easy, wide, synthetic oval; no live telemetry downloads.
- One fixed car configuration shared by every controller.
- Simple kinematic bicycle physics with bounded speed, steering, acceleration, and braking.
- A headless environment with 16 observations and 3 actions.
- A simple rule-based driver to verify that the track is driveable.
- A short `neat-python` run, a saved champion, a fitness plot, and a replay.
- Deterministic seeds and saved settings.

Defer dynamic tyre forces, aero, ICE/ERS, gears, opponents, multiple circuits, setup optimization, and DEAP. These are outside this one-day prototype. Label the simplified physics as a modelling assumption.

## Fixed car: what to implement today

These values are **prototype assumptions**, not official F1 specifications. Dimensions and steering limits follow the main project's initial assumptions; acceleration, braking, and speed limits below are deliberately simple demo settings. Keep them identical for all genomes.

| Parameter | Value | Use today |
| --- | ---: | --- |
| Mass | 760 kg | Recorded metadata; acceleration is prescribed, so mass does not enter today's equations. |
| Body length | 5.20 m | Oriented rectangular collision footprint. |
| Body width | 1.90 m | Oriented rectangular collision footprint. |
| Wheelbase `L` | 3.40 m | Steering-to-turning relationship. |
| CG to front axle `lf` | 1.55 m | `lf + lr = L`. |
| CG to rear axle `lr` | 1.85 m | Defines the CG kinematic model below. |
| Maximum steering angle | ±0.35 rad (about ±20°) | Limit requested wheel angle. |
| Maximum steering rate | 0.80 rad/s | Limit how quickly wheel angle changes. |
| Maximum drive acceleration | 4.0 m/s² | Full throttle, with no braking. |
| Maximum braking deceleration | 8.0 m/s² | Full brake. |
| Maximum speed | 30 m/s (108 km/h) | Conservative demonstration cap. |
| Initial speed | 0 m/s | Same start for baseline and every genome. |
| Physics / controller timestep | 0.01 s / 0.05 s | Five physics updates per decision. |
| Track width | 12 m total | Initial synthetic road width; validate the baseline before training. |
| Seed | 42 | Initial reproducible run. |

State: `x, y, heading, speed, steering_angle`, with `x, y` at the centre of gravity (CG). Assume the collision rectangle is centred at the CG. Positive heading and steering mean counterclockwise/left; positive lateral offset means left of the reference path.

Use this CG kinematic bicycle model, with all angles in radians:

```text
steering_target = steering_command × 0.35
steering_angle approaches steering_target by at most 0.80 × dt per step

effective_throttle = throttle × (1 - brake)
acceleration = 4.0 × effective_throttle - 8.0 × brake
speed_next = clip(speed + acceleration × dt, 0, 30)

beta = atan((lr / L) × tan(steering_angle))
vx = speed_next × cos(beta)                 # body-frame longitudinal speed
vy = speed_next × sin(beta)                 # body-frame lateral speed at CG
yaw_rate = (speed_next / L) × cos(beta) × tan(steering_angle)

x_next = x + speed_next × cos(heading + beta) × dt
y_next = y + speed_next × sin(heading + beta) × dt
heading_next = wrap_to_pi(heading + yaw_rate × dt)
```

Here `beta` is a geometric velocity angle, not a dynamic tyre-slip model. There are no tyre forces, drag, downforce, battery, or engine-power equations today. The speed cap and prescribed acceleration replace those details for this demonstration. Use the current state speed, steering, and the same formulas to compute observation velocities and yaw rate at reset and after each step.

## Neural network: 16 inputs, 3 outputs, evolving weights

The network receives observations every 0.05 s and outputs driving commands. **Vehicle mass (760 kg) stays fixed. Neural-network weights are connection strengths that NEAT evolves.** Inputs and outputs are values, not themselves weights.

### Exact input order

Define `clip(z)` here as clipping to `[-1, 1]`, and wrap angular differences to `[-π, π]`. These normalization scales are initial demo settings; save them with the champion.

| Index | Input | Normalized value |
| ---: | --- | --- |
| 0 | Body longitudinal speed | `clip(vx / 30)` |
| 1 | Body lateral speed at CG | `clip(vy / 10)` |
| 2 | Yaw rate | `clip(yaw_rate / 4)` |
| 3 | Heading error | `sin(heading - local_path_heading)` |
| 4 | Signed lateral offset | `clip(offset_m / 6)` for the 12 m road |
| 5 | Ray at −90° (right) | `distance_m / 150` |
| 6 | Ray at −67.5° | `distance_m / 150` |
| 7 | Ray at −45° | `distance_m / 150` |
| 8 | Ray at −22.5° | `distance_m / 150` |
| 9 | Ray at 0° (forward) | `distance_m / 150` |
| 10 | Ray at +22.5° | `distance_m / 150` |
| 11 | Ray at +45° | `distance_m / 150` |
| 12 | Ray at +67.5° | `distance_m / 150` |
| 13 | Ray at +90° (left) | `distance_m / 150` |
| 14 | Path heading 30 m ahead relative to car | `wrap(path_heading(s + 30) - heading) / π` |
| 15 | Path heading 100 m ahead relative to car | `wrap(path_heading(s + 100) - heading) / π` |

Cast rays relative to the car heading from the CG. Clamp distances to `[0, 150]` m; use 150 m when no boundary is hit within range. Look-ahead uses the path tangent at future arc length, wrapping around the closed loop. Rays provide perception; the full rectangle determines collisions.

### Exact output order

Use sigmoid output activations so each raw network output `y` lies in `[0, 1]`.

| Index | Output | Mapping and effect |
| ---: | --- | --- |
| 0 | Steering | `command = 2*y0 - 1`; −1 requests full right, +1 full left; target wheel angle is `command × 0.35` rad. |
| 1 | Throttle | `throttle = y1`; requests a fraction of the 4 m/s² acceleration. |
| 2 | Brake | `brake = y2`; requests a fraction of the 8 m/s² deceleration and suppresses throttle via `throttle × (1 - brake)`. |

Hold these commands for five physics steps. Apply steering-rate limits and speed limits in the simulator, not inside the network.

### What the weights do

A simple output node computes:

```text
output_j = sigmoid(bias_j + sum(weight_ij × input_i))
```

Use sum aggregation and fixed node response of 1 for this prototype. Weights may be positive or negative. For example, a connection from a boundary ray can influence steering or braking; the useful relationship is discovered through evolution, not assigned manually.

Start with **16 inputs → 3 outputs, fully connected, no hidden nodes**:

- `16 × 3 = 48` connection weights.
- `3` output biases.
- **51 initial evolving numeric values** when node responses and activation functions remain fixed.

There is no extra layer of “input weights” or “output weights”: each weight belongs to a connection between two nodes. Input nodes simply supply the observations. NEAT can mutate weights/biases and add or remove connections and hidden nodes, so 51 is the starting count, not a fixed final architecture. Record champion hidden-node and enabled-connection counts.

### How one generation works

1. Create 30 genomes, each describing a driver network.
2. Reset the same track and fixed car for each genome; evaluate without graphics.
3. At each controller step, pass the 16 observations through its network and apply its 3 commands.
4. Stop the episode on completion, failure, stalling, or the time limit.
5. Assign `fitness = maximum_validated_progress / track_length + 2 × completed`, where `completed` is 0 or 1 and progress is capped at one lap.
6. Let NEAT perform speciation, selection, crossover, and mutation; repeat for up to 20 generations within the wall-clock budget.

Fitness is calculated after the episode, not supplied as a network input or output. No backpropagation is needed. The completion bonus ensures a valid completed lap scores above any incomplete lap. Keep lap-time optimization for later.

Initial termination settings: 120 simulated seconds maximum (12,000 physics steps / 2,400 decisions); stop after 5 simulated seconds without at least 1 m of new validated progress. Stop sustained wrong-way travel after 1 s with longitudinal path velocity below −1 m/s. Treat these as settings to check against the baseline before training. The initial zero-speed spawn must receive the full stall grace period.

## Separate folder

Create a folder such as `F1_Prototype/` beside the main project. Keep this plan with it as `PLAN.md`. Build only the modules needed:

```text
F1_Prototype/
├── PLAN.md
├── README.md
├── requirements.txt
├── config.json          # fixed car, simulation, track, and seed settings
├── neat.cfg             # complete runnable NEAT configuration
├── track.py             # geometry, rays, checkpoints, containment
├── vehicle.py           # configuration validation and motion update
├── environment.py       # reset, step, observations, termination, metrics
├── baseline.py          # simple track-following driver
├── train.py             # genome evaluation and evolution
├── replay.py            # optional rendering of saved trajectories
├── tests/
└── outputs/             # champion, metrics, trajectories, plot, run settings
```

Prefer reusing the main project's working track geometry, sensors, and saved oval, with their small required dependencies. Copy a documented, self-contained subset or export a small track asset; avoid a hidden dependency on a sibling checkout. Do not copy unfinished physics or training scaffolds as if they already work.

Use the existing project environment where practical. Keep dependencies small: `numpy`, `shapely`, `neat-python`, `matplotlib`, and `pytest` for checks. Add `pygame` only if needed for the replay. A plain `reset`/`step` interface is sufficient; a Gymnasium wrapper is unnecessary today.

## Eight-hour implementation schedule

| Time | Work | Acceptance checkpoint |
| --- | --- | --- |
| 0:00–0:30 | Create folder, environment, configuration, and oval track. | Track loads locally and can be plotted; no network access needed for simulation. |
| 0:30–2:00 | Implement fixed-car kinematic motion, footprint containment, and baseline driver. | Baseline completes one valid lap; steering and braking behave correctly. |
| 2:00–3:15 | Add observations, ordered progress, episode termination, and fitness. | Headless episode returns 16 finite inputs and accepts 3 bounded actions; basic exploit checks pass. |
| 3:15–4:15 | Connect NEAT evaluation and artifact saving. | A tiny smoke run finishes and a saved champion can be reconstructed. |
| 4:15–5:15 | Run a bounded training experiment; record metrics. | Generation metrics, champion, exact configs, and seed are saved. |
| 5:15–6:15 | Add champion replay and fitness plot. | Saved trajectory replays with speed, progress, and termination reason visible. |
| 6:15–7:00 | Run focused checks and rehearse the demo. | Fresh commands reproduce the baseline and load the champion. |
| 7:00–8:00 | Fix problems, document limitations, and prepare backup screenshots/replay. | Demo works without starting a new training run. |

These are time limits, not estimates to keep extending. If a milestone slips, reduce polish and training duration first.

## Implementation checklist

### 1. Track and vehicle

- [ ] Use SI units and a fixed physics timestep of 0.01 s; hold controller actions for five steps (0.05 s).
- [ ] Start on the reference path facing the correct direction.
- [ ] Store and validate positive dimensions, wheelbase, limits, and timesteps in a serializable vehicle/simulation configuration.
- [ ] Keep car parameters identical across all genomes.
- [ ] Use a kinematic bicycle model; document the pose reference point and place the collision rectangle consistently with it.
- [ ] Clamp steering angle/rate, throttle, brake, and speed. Let braking override throttle when both are requested.
- [ ] Use the entire oriented car rectangle for off-track termination; allow boundary contact consistently with the main project.
- [ ] Verify ordered, forward checkpoint crossings before declaring a lap complete.

### 2. Environment and controller interface

- [ ] Implement independent episode state with `reset(seed)` and `step(action)`.
- [ ] Keep rendering outside physics and genome evaluation.
- [ ] Produce the project's 16 inputs: longitudinal speed, lateral speed, yaw rate, sine of heading error, normalized lateral offset, nine boundary rays, and two look-ahead heading errors.
- [ ] Use the CG kinematic model above for `vx`, `vy`, and yaw rate; keep pose and collision-footprint conventions consistent.
- [ ] Normalize and clip observations; document scales, ray angles/range, and look-ahead distances.
- [ ] Map three sigmoid network outputs to steering `2y - 1`, throttle `y`, and brake `y`.
- [ ] End episodes on off-track, invalid state, sustained wrong-way motion, stalling, valid completion, or a fixed simulated-time limit.
- [ ] Bound progress changes using reachable distance per step and checkpoint order; do not reward raw distance travelled.
- [ ] Use the fitness formula above: maximum validated forward progress / lap length, plus 2 for valid completion. Keep lap time diagnostic today.
- [ ] Record progress, completion, lap time for completed laps only, and termination reason separately from fitness.

### 3. NEAT and saved results

- [ ] Start with fully connected feed-forward networks, 16 inputs, 3 outputs, and no hidden nodes; allow weight, bias, and topology mutation. Fix node responses to 1 and activations to sigmoid.
- [ ] Use a small initial budget, for example 30 genomes × 20 generations, then adjust to measured runtime. These are starting assumptions, not tuned settings.
- [ ] Measure a tiny run first. Set an episode step limit and a wall-clock training budget so training cannot consume the day.
- [ ] Reset an independent environment for each genome under identical conditions.
- [ ] Save the best genome, exact NEAT configuration, vehicle/track settings, seed, package versions, and generation metrics.
- [ ] Plot best and mean fitness by generation; do not claim improvement unless the measurements show it.
- [ ] Compare baseline, initial population performance, and saved champion under the same conditions. Report one run as a demonstration, not statistical evidence.

## Minimum validation before the meeting

- [ ] Zero steering gives straight motion; positive steering turns in the documented direction; braking cannot create negative speed.
- [ ] Footprint collision catches a corner outside the road even when the car centre is inside.
- [ ] A stationary car, reverse finish crossing, skipped gate, repeated gate crossing, and start-line oscillation cannot earn a completed lap or repeated progress.
- [ ] Identical seed, settings, and action sequence reproduce the same trajectory.
- [ ] Observations always contain 16 finite values; controller action mapping respects all three ranges.
- [ ] Baseline completes a valid lap using the same collision and checkpoint rules as NEAT.
- [ ] A short training run works headlessly, and the saved champion reloads and runs.

ERS energy-accounting and dynamic-grip tests remain requirements for later physics work; this prototype has neither model.

## What to show the supervisor: five minutes

1. **Explain the question:** can evolution produce a neural driver that progresses around a track?
2. **Show the baseline lap:** demonstrate that the environment is driveable.
3. **Show initial versus evolved behaviour:** replay recorded episodes and display measured progress/completion.
4. **Show the training plot:** explain fitness and that NEAT evolves network weights and topology while the car stays fixed.
5. **State limitations and the next step:** one easy track, simplified physics, one short run; next establish repeatable completion before increasing realism.

## Fallback and definition of done

If NEAT does not complete a lap, show its actual best progress and failure reason alongside the successful baseline. Keep the learned and rule-based drivers clearly labelled. Do not spend the final hour hoping another run succeeds.

If rendering causes delays, use a trajectory plot and saved frames. If training is slow, shorten the experiment and report its actual budget. Keep a recorded replay and screenshots available for the meeting.

**Done for the day:** a baseline lap, a real end-to-end NEAT run, a reloadable champion, recorded metrics, a visual demonstration, and a README with commands that were actually verified. Do not expand into the full simulator before these work.

