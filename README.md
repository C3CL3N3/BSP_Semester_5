# F1 learning prototype

Separate Python scaffold for the pipeline **track → observations → NEAT driver
→ movement → fitness → evolution → replay**. Read [PLAN.md](PLAN.md) for the
implementation checklist and exact modelling assumptions.

## Current tracks, aero and replay display

The active `config.json` now uses `tracks/technical.json`: a 3,030.46 m
synthetic track, 12 m wide, with alternating left/right bends, tightening
transitions, and longer straights. It has 122 ordered gates per lap and a
1,200 s episode limit for three laps. The original oval remains available in
`configs/oval.json`. Oval-specific regression tests use that explicit fixture.

The supplied `monza.json` is supported as a separate closed-reference asset:
5,760.15 m, 12 m wide and 58 gates, selected by `configs/monza.json`. **Monza is
not active and has not been driven/trained in this change.** Its geometry,
spawn and path queries were validated. Track lengths describe the supplied
reference data, not a claim of official circuit accuracy.

File-based tracks require schema_version 1, metre units, width_m and explicitly
closed reference_points. They use a buffered polyline road and arc-length
projection/tangents. Self-crossing centrelines and roads with merged loops are
rejected. Assets use paths relative to the project directory. Environment and
training snapshots embed the loaded points, so old runs don't depend on later
changes to a source asset. The analytical oval implementation remains intact.

Preview a track without activating it or running physics:

```sh
.venv/bin/python track.py --plot outputs/technical-preview-new.png
.venv/bin/python track.py --config configs/monza.json --plot outputs/monza-preview-new.png
```

Verified previews: `outputs/technical-track.png`,
`outputs/monza-track-preview.png`. Export filenames must be new.

Dynamic physics now adds aerodynamic drag and downforce:

```text
drag = 0.5 * air_density * (Cd*A) * speed^2
downforce = 0.5 * air_density * (Cl*A) * speed^2
Fz_front += downforce * front_downforce_fraction
Fz_rear  += downforce * (1 - front_downforce_fraction)
```

Drag opposes both components of body velocity. Downforce increases the normal
loads used by each axle's existing friction circle; it does not propel the car.
The quadratic laws follow NASA's [drag equation](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/drag-equation/)
and [lift equation](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/lift-equation-2/).
Current demonstration settings are air density 1.225 kg/m³, `drag_area_m2 = 1.1`
(Cd*A), `downforce_area_m2 = 3.0` (positive downward Cl*A), and front downforce
fraction 0.45. These are uncalibrated assumptions. Set either area to zero to
disable that effect. Old configurations default to zero aero coefficients.
No wind, ride-height effects, pitch or dynamic load transfer are simulated.
The 80 m/s hard speed cap remains; drag can produce a lower terminal speed.

Speed-coloured replay lines are thin again (1 px colour / 3 px outline).
Replay/PNG displays show min/max speeds **after the first completed lap**, so
standing-start acceleration is excluded. New metrics store
`min_speed_after_first_lap_m_s` and `max_speed_after_first_lap_m_s`, measured at
physics rate. Values are null until lap one finishes. During replay, extrema
use available recorded samples up to playback time; the terminal frame uses
exact saved extrema. Older recordings derive extrema from their stored samples.

Validation: 64 tests pass. The baseline drove three real laps on the technical
track with aero enabled: 763.54 s total, 366 gate crossings; post-lap-one min/max
11.74 / 11.94 m/s. Recording:
`outputs/baseline-20261017T141311024092Z.json`. Thin speed-coloured export:
`outputs/technical-baseline-speed.png`.
A 30-genome one-generation smoke run saved/reloaded its champion and periodic
PNG under `outputs/training-20261017T141502757883Z/`. No initial network completed
a lap; best validated progress was 826.39 m. Earlier oval results below are
historical and not directly comparable to this new circuit/aero configuration.

## Speed reward and speed-coloured trajectories

Checkpoint `465bb09` saves the dynamic-physics project before these changes.
The current objective adds an explicit speed bonus alongside progress,
completion and mean-lap-time rewards:

```text
speed_bonus = speed_bonus_weight / target_distance
              * sum(new_validated_progress * clip(forward_path_speed / reference_speed, 0, 1))
fitness = progress_fraction + 2 * completed + time_bonus + speed_bonus
```

`fitness.speed_bonus_weight` defaults to 0.5 and `reference_speed_m_s` to 30.
The bonus is available during incomplete runs, helping selection before a car
can finish. It is bounded by 0.5 for a full target distance. Only new maximum
validated progress is credited; stationary motion, reversing and re-driving
previously credited sections cannot farm it. Collision/invalid moves earn no
bonus. Completion still dominates: incomplete fitness stays below 1.5 while
complete episodes score at least 3. Metrics expose `speed_bonus` separately.
Set the weight to 0 to disable it. Old saved settings without this field keep
their original objective. Earlier fitness measurements below are historical.

Animated trails, static exports and periodic champion PNGs now colour each
segment by speed: light yellow is slow, dark blue is fast, with a pale outline
for contrast. The fixed legend range is `replay.speed_color_max_m_s` (default
30 m/s); values above it share the darkest colour. This display scale is
independent of the 80 m/s vehicle speed cap, and is fixed across generations.
Overlapping lap traces show the most recently drawn lap segment.

```sh
.venv/bin/python replay.py outputs/training-20261017T104242234640Z --export outputs/my-speed-plot.png
```

## Dynamic tyres (current branch)

`config.json` now selects `vehicle.model = "dynamic"`. Old saved configs without
this key continue to select the kinematic model. The 16-input / 3-action driver
interface, fixed car dimensions, track and fitness remain the same. Dynamic
states add body lateral velocity and yaw rate; total `speed` remains the
magnitude of body velocity. Observations read these integrated velocities.
All trajectory artifacts now retain the extra dynamic state fields.

Implemented force model (left-positive conventions):

```text
alpha_front = steering - atan2(vy + lf * yaw_rate, vx)
alpha_rear  = -atan2(vy - lr * yaw_rate, vx)
Fz_front = mass * g * lr / wheelbase
Fz_rear  = mass * g * lf / wheelbase
Fy_axle = clip(cornering_stiffness * alpha, -mu * Fz, mu * Fz)
abs(Fx_axle) <= sqrt((mu * Fz)^2 - Fy_axle^2)
```

Mass and yaw inertia now enter the longitudinal, lateral and yaw equations.
Front forces rotate through the wheel steering angle into body coordinates.
The model structure follows the [MathWorks dynamic bicycle equations](https://www.mathworks.com/help/ident/ug/modeling-a-vehicle-dynamics-system.html).
Tyre forces are axle totals, not individual wheel forces. The friction-circle
allocator gives lateral force priority and limits the requested drive/brake
force to the remaining grip. Drive/braking are distributed by static axle load;
this is a simple all-wheel force allocation assumption. Brakes cannot launch
the car from rest. Body-forward velocity is constrained nonnegative.

Prototype parameters: yaw inertia 1900 kg m², front/rear axle cornering stiffness
55,000 / 60,000 N/rad, mu = 1.0, g = 9.81 m/s². These are demonstration settings,
not calibrated F1 tyre data. Below 1 m/s the slip denominator is regularized,
and lateral forces ramp from zero at rest. Physics is internally substepped
at 0.001 s while preserving the 0.01 s environment update. The existing speed
cap remains. This is a saturated linear tyre law, with no longitudinal slip,
wheel rotation, load transfer, tyre temperatures. Aerodynamic drag and downforce are now included as described above.

Inspect current slip angles, axle loads and forces using
`vehicle.tyre_forces(state, action, vehicle_config)`. Returned `TyreForces` values
use radians and newtons. Use `model: "kinematic"` for comparison.

### Measured increase in challenge

The same 30 seeded initial networks completed three laps in 1/30 cases under
each model. This does not demonstrate a general increase in learning difficulty.
The unchanged 12 m/s baseline completed in 113.94 s with kinematics versus
114.52 s with dynamics (+0.51%). Pushing the same controller harder exposed
the grip limit:

| Requested speed | Kinematic | Dynamic |
| --- | --- | --- |
| 12 m/s | Three laps, 113.94 s | Three laps, 114.52 s |
| 18 m/s | Three laps, 77.22 s | Three laps, 78.33 s |
| 22 m/s | Three laps, 64.09 s | Off-track after 312.82 m |
| 26 m/s | Three laps, 55.16 s | Off-track after 301.11 m |

The change mainly challenges speed optimization: networks must budget grip
between steering and throttle/braking, and handle lateral/yaw response lag.
A rough constant-radius estimate with mu=1 and R=40 m gives sqrt(mu*g*R)
= 19.8 m/s; it is an approximation, not an enforced speed limit. Measurements
are a single seed/controller comparison, not a statistical training study.
Detailed settings, per-genome metrics, baseline recordings and the speed sweep
are in `outputs/physics-comparison-20261017T103853822623Z/`.

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

The car uses a CG-centred footprint and a simplified dynamic bicycle model.
The legacy kinematic model remains available for old recordings/comparisons.
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
original oval configuration produces a 451.33 m counterclockwise stadium oval, 12 m wide,
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

## Large exports for zooming

Trajectory PNGs now render natively at 4400×3040 pixels, including periodic
champion exports. Configure `replay.export_width_px` / `export_height_px`, or
pass `--export-width 6000 --export-height 4000` to replay.py. This adds geometry
detail rather than enlarging an existing bitmap; trajectory lines remain thin.
Playback windows keep their normal size. Track previews are 6400×4800 PNGs;
use a `.svg` output suffix for vector track previews with sharp unlimited zoom.
New training runs save fitness plots at 4200×2400 plus vector `fitness.svg`.

Generated examples: `outputs/technical-baseline-large.png`,
`outputs/technical-track-large.png`, and `outputs/technical-track.svg`.
Open these in an image/SVG viewer and zoom to inspect the line.

## Save every new champion automatically

`training.render_every_generations` accepts `"auto"` (now selected in config.json),
a positive generation interval, or 0 to disable snapshots. CLI equivalent:

```sh
.venv/bin/python train.py --render-every auto --render-mode png
```

Auto mode immediately saves each **strict improvement in all-time fitness**,
including multiple improvements within one generation. The first evaluated
genome establishes the first champion. Ties, repeated elites and worse fitness
do not create duplicate snapshots. Filenames include generation and genome ID,
e.g. `champion_laps/generation-0005-genome-000123.png`. Full reloadable snapshots
remain in the matching `generation_champions/` subfolder.

Auto mode records evaluated trajectories so saving a champion doesn't require a
second simulation. Completed champions are saved even if the budget subsequently
ends part-way through a generation. `render_mode: "window"` also plays every
new champion; PNG mode saves without opening windows. Numeric intervals preserve
the original end-of-generation behaviour. This saves champions, not the complete
population required to resume evolution.
