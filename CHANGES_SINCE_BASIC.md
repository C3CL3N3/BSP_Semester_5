# Changes since the basic supervisor version

This compares `cleanup/basic-demo` (commit `8bd8e82`) with this supervisor version, based on the Mac development branch `feature/dynamic-tyre-physics` (commit `0c5f55c`). It describes implementation changes, not a controlled comparison of learning results.

| Area | Basic version | Current version |
| --- | --- | --- |
| Motion | Kinematic bicycle | Dynamic bicycle with lateral velocity and yaw response; kinematic mode still available |
| Tyres | No grip limit | Front/rear slip angles, saturated lateral forces and shared lateral/longitudinal grip |
| Aero | None | Quadratic drag and downforce |
| Active track | 451.33 m oval | About 3030.46 m technical track with 122 gates per lap |
| Other tracks | Oval only | Original oval and optional supplied Monza asset |
| Speed cap | 30 m/s | 80 m/s |
| Fitness | Progress, completion and completed-lap time | Same terms plus speed reward on new validated progress |
| Population | 30 | 100 |
| Training limit | 20 generations / 1800 s | 500 generations / 7200 s |
| Episode limit | 360 s | 1200 s |
| Evaluation | Serial | Serial, unchanged |
| Champion snapshots | Every 5 generations | Every strict new all-time champion by default; interval option retained |
| Trajectories | Plain line | Thin speed-coloured line, larger exports and post-first-lap speed extrema |

## What stayed the same

The driver still receives 16 observations and produces three actions. The controller runs every 0.05 s, with 0.01 s environment steps. Each episode starts from rest, targets three laps, checks the full car rectangle and validates ordered forward gates. NEAT uses species, crossover and mutation rather than gradient descent. The baseline remains conservative.

## Why this changes the task

The driver now has to manage grip and lateral response. Accelerating hard in a corner can leave too little grip, and steering changes do not instantly set the car's direction. The longer track has different bend directions and braking demands. The speed term also provides a signal before a genome can complete all three laps.

These changes make the task different, but do not establish a numerical increase in learning difficulty. To measure that, compare models with several seeds and equal evaluation budgets. Changing population, circuit, reward and physics together prevents attributing a result to one feature.

## Cleanup for this handover

Removed old plans, duplicate setup notes, historical output links, an unused NEAT callback, unused constants/imports and the unused pytest dependency. The README now describes the current settings rather than the sequence of development. Runtime finite-value, collision and progress checks remain. Action/state validation avoids unnecessary dataclass deep copies; equations and rewards were not changed for this cleanup.

The ZIP contains source, settings, track assets, tests and the handover documents. It excludes saved trainings, earlier reports, Git metadata, virtual environments, caches and writing prompts. Existing local artifacts are preserved outside the package. The basic branch's uncommitted comment edits were preserved in a Git stash before switching branches. The separate Intel multiprocessing implementation is not included.

## Validation

Validation for this handover is recorded in `VALIDATION.md`. Test results check implementation behaviour; they do not prove realistic vehicle dynamics or successful training on the technical track. No new long training or target-laptop performance result is claimed.
