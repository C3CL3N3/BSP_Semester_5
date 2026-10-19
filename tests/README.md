Add focused behavioural checks here as each subsystem is implemented.

Required cases are in PLAN.md under “Minimum validation before the meeting”:
vehicle limits and turning direction, full footprint collision, ordered gates
and progress exploits, deterministic rollouts, observation shape/ranges,
baseline completion, and champion save/reload after a tiny training run.

Vehicle behaviour is covered in `test_vehicle.py`; geometry, rays, collision,
ordered gates and progress exploits are covered in `test_track.py`.
`test_environment.py` checks observations, timestep grouping, episode lifecycle,
termination, deterministic independent rollouts and completion metrics.
Run with `.venv/bin/python -m unittest discover -s tests -v` or, once pytest is
installed, `python -m pytest tests`. Track tests require Shapely.
`test_baseline.py` verifies a physically driven completed lap and identical
trajectory on a repeated run, without mocked physics. `test_train.py` checks actual two-generation evolution, seeded
reproducibility, champion reconstruction, output mapping and budget expiry.

`test_replay.py` verifies baseline/champion loading, input validation, timestamp
and heading interpolation, PNG export and SDL dummy-display animation.
Replay tests require pygame-ce.

`test_dynamic_vehicle.py` verifies axle slip/load conventions, friction circles,
reduced longitudinal force under cornering load, transient steering symmetry,
stationary steering/braking, lateral-force direction, deterministic integration
and timestep convergence. Legacy kinematic checks remain in test_vehicle.py.

`test_aerodynamics.py` checks quadratic drag/downforce, coasting drag, zero-speed
behaviour and increased axle load/grip. `test_track_assets.py` validates technical
and optional Monza assets, projection, containment, ordered gates, reachable
progress and self-contained snapshots. Oval regression tests explicitly use
configs/oval.json so changing the active track does not invalidate them.
