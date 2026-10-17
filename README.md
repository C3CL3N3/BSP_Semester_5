# F1 learning prototype

Separate Python scaffold for the pipeline **track → observations → NEAT driver
→ movement → fitness → evolution → replay**. Read [PLAN.md](PLAN.md) for the
implementation checklist and exact modelling assumptions.

## Current objective and generation rendering

The current default is **three laps**, with a 360 s episode limit. Earlier
one-lap measurements below are historical; previous saved recordings still
play using their original configuration. Configurations without `target_laps`
retain one-lap behaviour; configurations without `fitness` retain no time bonus.

Fitness is:

```text
progress = maximum_validated_progress / (track_length * target_laps)
time_bonus = weight * reference_lap_time / (reference_lap_time + mean_lap_time)
fitness = progress + 2 * completed + (time_bonus if completed else 0)
```

Defaults: `simulation.target_laps = 3`, `fitness.time_bonus_weight = 1`,
`fitness.reference_lap_time_s = 45`. Faster completed runs score higher;
incomplete runs stay below 1 and earn no time bonus. Every lap requires all
32 ordered forward gate crossings. Metrics include `laps_completed`,
`lap_times`, `mean_lap_time`, and `total_time`. `lap_time` now means mean lap
time for a completed episode; `total_time` is the full three-lap duration.
The first lap includes acceleration from rest. Fitness values from earlier
one-lap runs are not directly comparable.

`neat.cfg` disables fitness-threshold termination so training runs to the
generation or wall-clock limit even after a driver completes three laps.

Configure periodic rendering in config.json:

```json
"render_every_generations": 5,
"render_mode": "png",
"render_speed": 4.0
```

Each training invocation creates a separate timestamped subfolder by default:

```text
outputs/training-<timestamp>/
├── champion_laps/
│   ├── generation-0005.png
│   └── generation-0010.png
├── generation_champions/
│   └── generation-0005/    # genome, recording, config and trajectory PNG
├── champion.pkl
├── champion_episode.json
├── fitness.png
└── summary.json
```

Use `--render-every X` to select the PNG interval. The run directory and saved
PNG paths are printed during training. Separate runs preserve previous results.
`--output NEW_FOLDER` optionally names a run folder explicitly.

These keys belong in `training`. `png` saves a trajectory picture and recorded
episode every fifth completed generation (5, 10, 15, ...). `window` also opens
the generation champion replay, then automatically closes it and resumes
training; Escape/closing the window skips playback. Set the interval to 0 to
disable. The chosen network is the best **in that generation**, not necessarily
the best across the entire run. Partial generations are not rendered.
Snapshots contain champion episode/genome, exact configuration, NEAT config and
PNG under `generation_champions/generation-0005/`. Both replay and trusted
champion reconstruction accept that directory.

```sh
# Watch every fifth generation at 4x playback speed
.venv/bin/python train.py --generations 20 --render-every 5 --render-mode window --render-speed 4
# Save PNGs and recordings every generation without opening windows
.venv/bin/python train.py --generations 2 --render-every 1 --render-mode png
```

Periodic re-evaluation and playback count toward the wall-clock training
budget. If the budget expires before a periodic recording completes, it is
skipped; final champion saving/reload verification still happens afterwards.
Interactive window playback remains a manual check; PNG export is tested.

Verified new-objective smoke run:
`outputs/training-20261017T102545465510Z/`: two full generations, 60 genomes,
96 gates for the champion, 117.98 s total / 39.33 s mean lap, fitness 3.533639.
Baseline: 113.94 s total / 37.98 s mean lap. Best fitness was unchanged across
the two generations; mean population fitness rose from 0.124012 to 0.240933.

## Current status

The project skeleton, settings, state/action types and controller output mapping
are present. Vehicle physics, parameter validation, body velocities, footprint
corners, oval geometry, boundary rays, ordered progress tracking and headless episodes
and the rule-based baseline are implemented. NEAT training, artifact saving and
champion reconstruction are also implemented. Vehicle, track, environment, training and replay tests cover the pipeline.
Pygame replay and static trajectory PNG export are also implemented.
The baseline completed a physically driven lap in 39.00 simulated seconds.
The first NEAT smoke run also produced a reloadable lap-completing champion
from the initial population; this is not evidence of evolutionary improvement.

The car uses a CG-centred footprint and a simplified kinematic bicycle model.
Settings are prototype assumptions, not official F1 specifications.

## Setup

Run from this directory. Python 3.11 or newer is recommended.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies have version bounds; save exact installed versions with each run.
Shapely and its NumPy dependency were installed and verified in `.venv`.
neat-python and matplotlib were also installed and verified. Baseline and NEAT
smoke runs work headlessly. Replay uses pygame-ce (imported as pygame),
which supplies a working Python 3.14 wheel; pygame itself failed to build
because local SDL headers were missing.

## Files

| File | Responsibility |
| --- | --- |
| `config.json` | Fixed car, timing, oval, observations, seed and training budget |
| `neat.cfg` | 30 feed-forward genomes, 16 inputs, 3 sigmoid outputs, no initial hidden nodes |
| `track.py` | Oval geometry, path projection/lookahead, rays, footprint containment and ordered gates/progress |
| `vehicle.py` | Validated configuration, bounded CG bicycle motion, velocities and footprint corners |
| `environment.py` | Validated configuration, reset/step, 16 observations, termination and episode metrics |
| `baseline.py` | Observation-only rule-based driver and measured lap/trajectory recording |
| `train.py` | Bounded NEAT evolution, fitness plot, run artifacts and champion reconstruction |
| `replay.py` | Pygame animation, interactive playback and static trajectory PNG export |
| `tests/` | Validation cases to implement alongside behaviour |
| `outputs/` | Generated run artifacts, ignored by Git |

Oval straight length (100 m), turn radius (40 m), gate count (32), and training
wall-clock budget (1800 s) are initial scaffold choices. Validate them against
the baseline before training. They do not change the brief's fixed car settings.

## Scaffold checks

These scaffold checks were verified:

```sh
.venv/bin/python -m compileall -q vehicle.py track.py environment.py baseline.py train.py replay.py
.venv/bin/python -c 'from environment import load_config; from train import network_action; assert load_config()["seed"] == 42; assert network_action([0.5, 1, 0]).steering == 0'
```

Baseline, training, champion reconstruction, replay rendering and trajectory
export are now implemented. Interactive window use remains a manual demo check.
Run `python -m pytest tests` once pytest is installed. All current vehicle, track and environment checks run with:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Use `VehicleConfig(**load_config()["vehicle"])` once per environment, then pass
that config to `advance(state, action, config, 0.01)` on each physics step.
The vehicle section dictionary is also accepted directly. `advance` returns a
new state; finite action values are clamped, while invalid state values,
non-finite inputs, and non-positive timesteps raise `ValueError`.
`body_velocities` returns `(vx, vy, yaw_rate)`, and `footprint_corners` returns
four world-space corners for the track's containment check.

Construct `Track(load_config()["track"])` once per environment. The current
configuration produces a 451.33 m counterclockwise stadium oval, 12 m wide,
with 32 gates. `start_state()` returns a fresh stationary pose on the lower
straight. `path_pose(s)` returns `(x, y, heading)` at wrapped arc length;
`path_heading(s)` supports future lookahead observations. `project(x, y)`
returns arc length, closest path position, tangent heading and signed offset
(positive left). Rays return metres in the supplied angle order.

`contains_vehicle` checks the whole oriented rectangle with the configured
boundary-contact policy. Paths are analytic; the road polygon and ray boundaries
use 256 segments per quadrant, giving under 0.3 mm radial error at these dimensions.

Create a separate `ProgressTracker(track)` for each episode. After each valid,
on-road physics step, call `tracker.update(state, max_speed * physics_dt)`.
It exposes `maximum_progress`, `gates_passed` and `completed`. It validates
reachable movement and forward crossings in order; invalid moves raise
`ValueError` without changing progress. The environment must handle off-track,
invalid-state and invalid-progress termination before rewarding the episode.
The supplied distance bound must be less than half the checkpoint spacing.

## Headless episodes

```python
from environment import Environment, load_config
from vehicle import Action

env = Environment(load_config())
observations = env.reset(seed=42)
observations, done, metrics = env.step(Action(steering=0, throttle=1, brake=0))
```

`reset` returns a tuple of 16 inputs in PLAN.md order. `step` accepts simulator
commands and returns `(observations, done, metrics)`. Use `train.network_action`
to map three sigmoid outputs before stepping. Each decision holds the action
for five 0.01 s physics updates, stopping early when an episode ends.

Metrics include `fitness`, `maximum_validated_progress` (metres),
`progress_fraction`, `completed`, `lap_time` (seconds, or `None` if incomplete),
`elapsed_time`, `termination_reason`, step counts, gates passed and seed.
Reasons are `completed`, `off_track`, `invalid_state`, `invalid_action`,
`invalid_progress`, `wrong_way`, `stalled`, and `time_limit`.
An invalid state/action returns the last finite observation. Call `reset`
before the first step and after termination; further steps after done raise
`RuntimeError`. Read the current `env.state` to record trajectories.

Seeds are recorded without changing global random generators; the current
physics and track contain no random effects. Configurations are copied per
environment. Timing and normalization settings are validated at construction.
The stall timer resets after at least 1 m of new maximum progress; spawn gets
the full 5 s grace period. Continuous wrong-way travel below -1 m/s lasts at
most 1 s. Collision/progress failures take priority over completion; valid
completion takes priority over timer expiry on the same physics step.

Environment unit tests include synthetic poses through real gates.
The baseline integration test also completes a real lap using only vehicle
physics and verifies that a second run reproduces its entire trajectory.
NEAT evaluation and champion reconstruction are covered by smoke and integration tests.

## Historical one-lap baseline

```sh
.venv/bin/python baseline.py
```

The driver uses only the 16 observations, with a conservative target speed of
12 m/s. It starts at rest and uses the same fixed car, physics, footprint
containment and gate validation intended for NEAT. With config.json and seed 42,
it completed 451.327 m in 39.00 simulated seconds, passed all 32 gates, and
scored fitness 3.0. This validates driveability under the simplified model.

Each CLI run saves a new timestamped JSON in `outputs/`, containing metrics,
exact configuration, controller gains, seed, Python/Shapely versions, and a
trajectory sampled every controller decision (including start and terminal
poses). Collision and progress checks still run every physics step.
Use `--config PATH`, `--seed N`, or `--output NEW_PATH` to override defaults.
Existing output files are never overwritten; an incomplete lap exits with code 1.

The first measured result is
[baseline-20261017T101155142104Z.json](outputs/baseline-20261017T101155142104Z.json).
Generated outputs are ignored by Git; preserve them separately for the demo.

## Historical one-lap NEAT smoke run

```sh
.venv/bin/python train.py --generations 2 --budget-seconds 120
.venv/bin/python train.py --reload outputs/training-20261017T101537376963Z
```

The first smoke run evaluated all 30 initial genomes in 3.31 wall-clock seconds.
Genome 9 completed the lap in 42.03 simulated seconds, passing all 32 gates
with fitness 3.0, 48 enabled connections and no hidden nodes. The initial
population's mean fitness was 0.118671. The same-settings baseline took 39.00 s.
Training stopped at the configured fitness threshold after generation 0;
the requested two generations are an upper bound. This result demonstrates
the pipeline, not improvement through evolution. Separate tests execute two
full generations on short episodes and reproduce metrics and trajectories.

Each new run directory contains `champion.pkl`, exact `neat.cfg` and
`config.json`, `versions.json`, `evaluations.json`, `generation_metrics.json`,
`champion_episode.json`, `baseline_episode.json`, `fitness.png` and
`summary.json`. Champion metrics are checked against a freshly reconstructed
network before the run is reported as verified. Reload only trusted locally
generated pickle artifacts. Configurations and the seed are saved, and NEAT's
random generator state is restored after training.

Use `--seed N`, `--config PATH`, `--neat-config PATH`, or `--output NEW_DIR`
to customize a run. Existing directories are never overwritten. Without CLI
overrides, config.json supplies 20 generations and an 1800 s budget.
The budget covers evolution and genome evaluation and is checked each
controller decision. Artifact generation, baseline comparison and champion
reconstruction run afterwards. Interrupted episodes receive no fitness and
partial generations are marked `complete: false` in metrics. A budget too
small to finish any genome saves a summary without a champion and exits 1.

## Recorded replay

Play the champion or baseline without running physics or training:

```sh
.venv/bin/python replay.py outputs/training-20261017T101537376963Z
.venv/bin/python replay.py outputs/baseline-20261017T101155142104Z.json --speed 2
```

Inputs may be a baseline JSON, a `champion_episode.json`, or a training directory.
Champion episodes load the exact config.json beside the episode. The display
shows the track, gates, full-size car rectangle, trajectory trail, driver label,
speed, elapsed time, progress, final fitness and termination reason. Motion
interpolates recorded samples and wraps heading changes across ±pi.
Space pauses/resumes, left/right seek five seconds, R restarts, Escape quits.
Playback holds the final frame until the window is closed.

Export a static PNG or check animation without a display:

```sh
.venv/bin/python replay.py outputs/training-20261017T101537376963Z --export outputs/champion-backup.png
.venv/bin/python replay.py outputs/training-20261017T101537376963Z --headless --frames 5
```

Export paths must be new; existing files are preserved. Verified backup plots
are `outputs/champion-trajectory.png` and `outputs/baseline-trajectory.png`.
Tests cover artifact loading, invalid recordings, time/angle interpolation,
PNG generation and display-free animation. An interactive desktop window has
not been manually checked in this session.
