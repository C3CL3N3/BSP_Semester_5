# What I changed since the basic version

In the basic version, I used a short oval and a kinematic bicycle model to check the full workflow: observations, driving, fitness, evolution and replay. This version keeps that workflow but adds tyre forces, aerodynamic loads and a longer track. I also changed the fitness so that faster driving can earn a reward before a network finishes all three laps.

The comparison here is with `cleanup/basic-demo` (`8bd8e82`). The current Mac version comes from `feature/dynamic-tyre-physics` (`0c5f55c`). Evaluation is still serial. The Intel multiprocessing branch is separate.

## 1. Vehicle model

The basic model calculated the direction of movement from speed and steering geometry. It had steering and acceleration limits, but no tyre grip limit. A network could therefore learn cornering behaviour that the model allowed even when a physical car would run out of grip.

I added a dynamic bicycle model with body lateral velocity and yaw rate as integrated state variables. Front and rear slip angles now determine lateral axle forces. Each axle uses a linear cornering-stiffness law, clipped at its available grip.

The longitudinal and lateral forces share a friction circle. In this implementation, lateral force gets priority. A heavily loaded tyre then has less force available for acceleration or braking. Steering no longer sets the motion direction immediately; the network has to deal with the car's lateral and yaw response.

Mass now enters the force equations, and yaw inertia enters the yaw equation. The current settings use 760 kg, 1900 kg m² yaw inertia, front/rear cornering stiffness of 55,000/60,000 N/rad and friction coefficient 1.0. These are demonstration values, not measured F1 parameters.

The dynamic model uses 0.001 s integration substeps. The environment still updates every 0.01 s, and the controller still chooses an action every 0.05 s. I kept the kinematic model available for comparisons.

## 2. Drag, downforce and speed

I added aerodynamic drag opposing motion and downforce increasing the normal load available to the tyres. Both vary with speed squared. The current drag area is 1.1 m² and downforce area is 3.0 m², with 45% of downforce applied at the front.

I raised the hard speed cap from 30 to 80 m/s, or from 108 to 288 km/h. This is a numerical ceiling, not a target speed or a guarantee that the car reaches it. Drag and cornering grip still affect the speed it can sustain.

The physics is still simplified. It has no wheel rotation, longitudinal tyre slip, load transfer, tyre temperature or wear. Adding forces makes the driving problem more constrained, but does not make the model calibrated or fully realistic.

## 3. Track

I replaced the active 451.33 m oval with a synthetic technical track of about 3030.46 m. It stays 12 m wide and has 122 ordered gates per lap, so three laps require 366 forward crossings. The alternating bends and longer straights introduce more varied steering and speed decisions.

I kept the original oval in `configs/oval.json`. The supplied Monza reference is also supported through `configs/monza.json`, but it is not the active circuit. Its geometry is an input asset, not a claim of official circuit accuracy.

File-based tracks use a closed reference path in metres. The saved run configuration embeds the loaded points so that a recording does not depend on later edits to the source asset. The episode limit increased from 360 to 1200 simulated seconds for the longer task.

## 4. Fitness: what changed and why

The basic version already rewarded progress, completing three laps and lower mean lap time. It did not have a separate speed reward. I kept those terms and added a speed bonus on newly validated progress.

Let `D` be the track length multiplied by the target number of laps, and `p` the maximum validated progress. With the basic settings:

```text
progress_score = p / D
completion_bonus = 2 if all three laps finish, otherwise 0
time_bonus = 45 / (45 + mean_lap_time) if completed, otherwise 0

basic_fitness = progress_score + completion_bonus + time_bonus
```

The time bonus only becomes available after completion. Before that, two genomes reaching the same distance would receive the same score even if one covered that distance faster.

The current version adds:

```text
speed_bonus = 0.5 / D * sum(delta_p * clip(v_path / 50, 0, 1))
current_fitness = basic_fitness + speed_bonus
```

Here, `delta_p` is the increase in maximum validated progress at a physics update. `v_path` is velocity along the local forward path direction, not just the magnitude of the car's speed. The sum runs over valid updates during the episode.

This gives selection a speed-related signal while networks are still learning to finish. For example, consider two hypothetical runs that both cover half the target distance. If one maintains 10 m/s forward path speed and the other 20 m/s over that new progress, their speed bonuses are 0.05 and 0.10. Their total scores become 0.55 and 0.60. These numbers illustrate the formula; they are not measured training results.

The bonus is distance-weighted rather than time-weighted. Waiting cannot earn it, and reversing or driving over an already credited section cannot earn the same progress again. Off-track or invalid-progress updates are rejected before this reward is added. At the current weight, the speed bonus is at most 0.5 over the full target distance.

Completion still dominates. An incomplete run scores below 1.5, while a completed run has a progress score of 1 and completion bonus of 2 before its time and speed bonuses. Faster completed laps still increase the time term. Both terms now encourage speed, but at different stages: the speed term works during partial runs, while the time term compares completed runs.

The settings are explicit in `config.json`:

| Fitness setting | Basic | Current |
| --- | ---: | ---: |
| `time_bonus_weight` | 1.0 | 1.0 |
| `reference_lap_time_s` | 45 | 45 |
| `speed_bonus_weight` | Absent, equivalent to 0 | 0.5 |
| `reference_speed_m_s` | Not used | 50 |

The 45 s reference is fixed and does not come from the baseline. The first lap still includes acceleration from rest. Setting `speed_bonus_weight` to 0 disables the added term. There is no separate steering-smoothness reward.

A fitness value from this version should not be compared directly with a basic-version value. The reward changed, and so did the track and vehicle model.

## 5. NEAT and training settings

I kept the same 16 inputs and three sigmoid outputs. The network still starts with no hidden nodes and 48 direct input-to-output connections. `neat-python` handles speciation, crossover and mutation; the project code handles the driving episodes and scoring. I did not replace the library's genetic algorithm.

The population increased from 30 to 100 genomes. The generation limit increased from 20 to 500, and the wall-clock training budget from 1800 to 7200 seconds. These are upper limits. A run stops when either limit is reached, and a partial episode receives no fitness.

The reproduction settings remain the same: two elites per surviving species and a parent pool drawn from the top 20% within each species, subject to the library's minimum pool size. Evaluation remains one genome at a time on this Mac branch. Saving a champion lets me replay or re-evaluate it, but does not save the full population needed to resume evolution.

## 6. Saving and visual inspection

I changed champion snapshots from every five generations to `"auto"` by default. Auto mode saves each strict improvement in all-time fitness, even if several occur in one generation. Ties do not create another snapshot. A numeric generation interval is still available, and 0 disables snapshots.

The trajectory line now shows speed: light yellow for slow sections and dark blue for fast sections. It stays thin so nearby lines remain visible when zooming. The colour scale tops out at 30 m/s, independently of the 80 m/s physics cap. Values above 30 m/s share the darkest colour.

PNG exports default to 4400 × 3040 pixels. I also added minimum and maximum speed after the first completed lap, excluding standing-start acceleration from those extrema.

## 7. Cleanup and checks

For this handover, I kept the code, settings, track assets, tests and documentation. The ZIP leaves out old training outputs, earlier reports, caches, virtual environments and Git history. I removed outdated planning documents, unused imports/constants, an unused NEAT callback and the unused pytest dependency. The finite-value, collision and progress checks remain.

The Mac version passes all 66 tests. Details are in `VALIDATION.md`. No new long technical-track training was run for this handover, so the changes above describe the implementation rather than a new measured learning result.

The next useful comparison would use several seeds, equal evaluation budgets and one change at a time. I cannot assign a percentage increase in difficulty from this update: track, physics, reward and population all changed together. Testing on an unseen track would also be needed before claiming that the driver generalises.
