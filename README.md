# NEAT driving prototype

This project evolves a neural driver on a simulated track. NEAT changes the network between episodes. During an episode, the weights stay fixed and the network controls steering, throttle and braking.

The current version uses a synthetic technical track, a dynamic bicycle model and serial genome evaluation. It is a small learning experiment, not a calibrated racing simulator. See [CHANGES_SINCE_BASIC.md](CHANGES_SINCE_BASIC.md) for the comparison with the earlier oval version.

## Setup and first run

Use Python 3.11 or newer. Create a fresh environment on each machine:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -q
python train.py --generations 2 --budget-seconds 120 --render-every 0
```

On Debian/Ubuntu, you may need `python3-venv` to create the environment. For normal training, run `python train.py`. That uses the full limits below. The short command above is a smoke run, not enough training to establish learning performance.

No previous training artifacts are included in the supervisor ZIP. New runs create timestamped folders inside `outputs/`.

```sh
python baseline.py
python replay.py outputs/<training-folder>
python replay.py outputs/<training-folder> --export outputs/champion.png
python train.py --reload outputs/<training-folder>
python track.py --plot outputs/track.png
```

Replace `<training-folder>` with the folder printed by training. Use a new filename for exports. Space pauses replay, arrows seek five seconds, R restarts and Escape closes it. Replay animates recorded samples; `--reload` drives a fresh episode with the saved network. Only load trusted pickle genomes. Reloading a champion does not resume evolution: the full population is not saved.

## Evaluation and evolution

Each genome starts from the same stationary pose. Every 0.05 s, its feed-forward network reads 16 inputs and chooses an action. The environment runs five 0.01 s updates per decision. Dynamic integration uses smaller 0.001 s substeps. The full car rectangle must stay on the road, and gates must be crossed forwards in order.

The input order is:

- 0–2: body longitudinal speed, lateral speed and yaw rate.
- 3–4: sine of heading error and signed path offset.
- 5–13: nine boundary rays, from −90° to +90° relative to the car.
- 14–15: relative path headings 30 m and 100 m ahead.

Inputs are normalised using `observations` in `config.json`. Rays have a 150 m range. The speed scales are 30 m/s longitudinally and 10 m/s laterally, yaw uses 4 rad/s and offset uses 6 m. Lookahead angles are wrapped and divided by pi. These are ideal geometric inputs, including map information, not camera measurements.

Three sigmoid outputs produce steering `2*y0 - 1`, throttle `y1` and brake `y2`. Braking reduces requested acceleration through `throttle * (1 - brake)`.

`neat-python` implements speciation, reproduction and mutation. The project supplies the simulator, observations, fitness and evaluation loop. There is no gradient descent.

| Setting in `neat.cfg` | Current value |
| --- | --- |
| Population | 100 genomes |
| Initial network | 16 inputs, 3 outputs, no hidden nodes, all 48 direct connections |
| Activation / aggregation | Sigmoid / sum |
| Elite genomes copied | 2 per surviving species |
| Parent pool | Best 20% within each species, rounded up; at least two where available |
| Compatibility threshold | 3.0 |
| Stagnation limit / protected species | 10 generations / best 2 |
| Weight mutation rate / replacement rate | 0.8 / 0.1 |
| Bias mutation rate / replacement rate | 0.7 / 0.1 |
| Mutation standard deviation | 0.5 |
| Node addition / deletion attempt rates | 0.2 / 0.2 |
| Connection addition / deletion attempt rates | 0.5 / 0.5 |

The parent pool belongs to each species, not a single global ranking. Mutation can change the network structure. Some structural attempts fail, for example if a connection would create a feed-forward cycle. Fitness-threshold stopping is disabled.

## Physics and tracks

The dynamic bicycle model integrates longitudinal speed, lateral speed and yaw rate. Front and rear slip angles determine lateral axle forces:

```text
alpha_front = steering - atan2(vy + lf * yaw_rate, vx)
alpha_rear = -atan2(vy - lr * yaw_rate, vx)
Fy = clip(cornering_stiffness * alpha, -mu * Fz, mu * Fz)
abs(Fx) <= sqrt((mu * Fz)^2 - Fy^2)
```

Lateral forces get priority in the friction circle, leaving less grip for acceleration or braking in a corner. Static axle loads follow the CG position. Aerodynamic loads add:

```text
drag = 0.5 * air_density * drag_area * speed^2
downforce = 0.5 * air_density * downforce_area * speed^2
```

Drag opposes motion. Downforce increases available tyre load, with 45% applied at the front. Slip calculations are regularised at low speed. The kinematic model remains available with `vehicle.model = "kinematic"`.

| Setting in `config.json` | Current value |
| --- | --- |
| Car mass / yaw inertia | 760 kg / 1900 kg m² |
| Body / wheelbase | 5.2 × 1.9 m / 3.4 m |
| Front / rear cornering stiffness | 55,000 / 60,000 N/rad |
| Friction coefficient | 1.0 |
| Steering angle / rate limits | ±0.35 rad / 0.8 rad/s |
| Acceleration / braking limits | 4 / 8 m/s² before force allocation |
| Hard speed cap | 80 m/s, or 288 km/h |
| Air density | 1.225 kg/m³ |
| Drag / downforce areas | 1.1 / 3.0 m² |
| Track | `tracks/technical.json`, about 3030.46 m, 12 m wide |
| Target / episode limit | 3 laps / 1200 simulated seconds |
| Seed | 42 |
| Training limit | 500 generations or 7200 wall-clock seconds |
| Evaluation | One genome at a time in the main process |

These are demonstration parameters. The tyre law is saturated linear, with no wheel rotation, longitudinal slip, load transfer, temperature or wear. The car is not calibrated against measurements. An 80 m/s cap does not mean the car can reach that speed or take every corner at it.

`configs/oval.json` selects the original oval. `configs/monza.json` selects the supplied `monza.json` asset, which is not the active track. Asset lengths describe the supplied geometry, not official circuit accuracy. File tracks must have a simple closed reference path in metres. Saved run settings embed the loaded points.

## Fitness and termination

Let `D` be track length times target laps. Fitness is:

```text
progress = maximum_validated_progress / D
completion = 2 if all target laps finish, otherwise 0
time_bonus = 45 / (45 + mean_lap_time) if completed, otherwise 0
speed_bonus = 0.5 / D * sum(new_progress * clip(forward_path_speed / 50, 0, 1))
fitness = progress + completion + time_bonus + speed_bonus
```

The constants come from `fitness` in `config.json`. The speed reward uses 50 m/s as its reference, independently of the 80 m/s cap. It credits only new maximum validated progress. Reversing or repeating a section cannot earn it again. The time reference is fixed, not derived from the baseline. Completion dominates incomplete runs. The first lap includes acceleration from rest.

Episodes stop on completion, leaving the road, invalid actions/states/progress, time limit, stalling or sustained wrong-way travel. Stalling means less than 1 m of new progress within 5 s. Wrong-way travel below −1 m/s for 1 s also stops the episode. Config validation and these runtime checks remain in place.

## Saving and replay

This Mac version evaluates genomes serially in the main process. There is no worker-count setting, worker benchmark or GPU evaluation.

`training.render_every_generations` accepts `"auto"`, a positive interval, or 0. Auto saves every strict improvement in all-time fitness, including multiple improvements in a generation. Numeric intervals save the best driver in that generation. PNG is the default; `window` also plays the recording.

Run folders save exact settings, package versions, evaluation and generation records, champion genome/trajectory, baseline comparison and fitness plots. Champion snapshots go into `generation_champions/` and `champion_laps/`. A partial episode receives no fitness. Final artifact saving and reload checks happen after the training budget, so total command time can exceed it.

Trajectory colour runs from light yellow for slow segments to dark blue for fast segments. The display scale is 30 m/s, so faster values share the darkest colour. This does not limit physics. Min/max speed excludes the first lap. PNG exports default to 4400 × 3040 pixels.

## Files and limits of the experiment

| File | Purpose |
| --- | --- |
| `vehicle.py` | Kinematic/dynamic motion, axle forces, aero and car footprint |
| `track.py` | Oval and file geometry, rays, path projection, gates and previews |
| `environment.py` | Episode loop, observations, scoring and termination |
| `baseline.py` | Conservative rule-based driver using the same inputs |
| `train.py` | Serial NEAT evaluation, saving and champion reload |
| `replay.py` | Recorded playback and PNG export |
| `config.json`, `neat.cfg` | Simulation/training settings and NEAT settings |
| `configs/`, `tracks/`, `monza.json` | Alternative configs and required track assets |
| `tests/` | Physics, geometry, environment, training and replay checks |

The baseline targets 12 m/s and checks driveability. It is not an optimised racing controller. Comparing its time with an evolved driver does not establish superiority over a strong controller. A useful study still needs multiple seeds, equal evaluation budgets and tests on unseen tracks. Basic oval results cannot be compared directly with technical-track fitness: the geometry, physics and reward changed.
