"""Deterministic headless episodes; no rendering or global random state.

step accepts simulator Action values, not raw sigmoid network outputs.
"""

from copy import deepcopy
import json
import math
from pathlib import Path

from track import ProgressTracker, Track
from vehicle import Action, VehicleConfig, advance, body_velocities, wrap_to_pi

ROOT = Path(__file__).resolve().parent
OBSERVATION_COUNT = 16
ACTION_COUNT = 3


def load_config(path: str | Path = ROOT / "config.json") -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _clip(value):
    return max(-1.0, min(value, 1.0))


class Environment:
    def __init__(self, config: dict):
        self.config = deepcopy(config)
        self.vehicle_config = VehicleConfig(**self.config["vehicle"])
        self.track = Track(self.config["track"])
        self.simulation = self.config["simulation"]
        self.target_laps = self.simulation.get("target_laps", 1)
        if isinstance(self.target_laps, bool) or not isinstance(self.target_laps, int) or self.target_laps < 1:
            raise ValueError("target_laps must be a positive integer.")
        self.target_distance = self.track.length * self.target_laps
        fitness = self.config.get("fitness", {})
        self.time_bonus_weight = fitness.get("time_bonus_weight", 0.0)
        if not isinstance(self.time_bonus_weight, (int, float)) or isinstance(self.time_bonus_weight, bool) or not math.isfinite(self.time_bonus_weight) or self.time_bonus_weight < 0:
            raise ValueError("time_bonus_weight must be finite and nonnegative.")
        self.reference_lap_time = fitness.get("reference_lap_time_s", 45.0)
        _positive("reference_lap_time_s", self.reference_lap_time)
        self.observation_config = self.config["observations"]
        for key in ("physics_dt_s", "controller_dt_s", "max_episode_s",
                    "stall_timeout_s", "stall_progress_m", "wrong_way_timeout_s"):
            _positive(key, self.simulation[key])
        threshold = self.simulation["wrong_way_velocity_m_s"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold >= 0:
            raise ValueError("wrong_way_velocity_m_s must be finite and negative.")
        self.dt = self.simulation["physics_dt_s"]
        ratio = self.simulation["controller_dt_s"] / self.dt
        if not math.isclose(ratio, round(ratio), abs_tol=1e-9) or ratio < 1:
            raise ValueError("Controller timestep must be an integer multiple of physics timestep.")
        self.physics_steps_per_action = round(ratio)
        limit = self.simulation["max_episode_s"] / self.dt
        if not math.isclose(limit, round(limit), abs_tol=1e-9):
            raise ValueError("Episode time limit must be an integer number of physics steps.")
        self.max_physics_steps = round(limit)
        if self.vehicle_config.max_speed_m_s * self.dt >= self.track.checkpoint_spacing / 2:
            raise ValueError("Physics timestep is too large for checkpoint validation.")
        for key in ("longitudinal_speed_scale_m_s", "lateral_speed_scale_m_s",
                    "yaw_rate_scale_rad_s", "lateral_offset_scale_m", "ray_range_m"):
            _positive(key, self.observation_config[key])
        if self.observation_config["ray_angles_deg"] != [-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90]:
            raise ValueError("Nine ray angles must match the documented input order.")
        lookahead = self.observation_config["lookahead_distances_m"]
        if len(lookahead) != 2:
            raise ValueError("Exactly two lookahead distances are required.")
        for value in lookahead:
            _positive("lookahead distance", value)
        self._ready = False

    def reset(self, seed: int | None = None) -> tuple[float, ...]:
        """Return 16 observations and reset all independent episode state.

        The seed is recorded; this environment currently has no random effects.
        Omission uses config.json's seed, without touching global RNGs.
        """
        seed = self.config["seed"] if seed is None else seed
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        self.seed = seed
        self.state = self.track.start_state()
        self.state.speed = self.vehicle_config.initial_speed_m_s
        if not self.track.contains_vehicle(self.state, self.vehicle_config):
            raise ValueError("Vehicle does not fit on the road at spawn.")
        self.progress = ProgressTracker(self.track, self.target_laps)
        self.lap_times = []
        self._last_lap_finish = 0.0
        self.physics_steps = 0
        self.decisions = 0
        self.done = False
        self.termination_reason = None
        self._progress_anchor = 0.0
        self._last_progress_time = 0.0
        self._wrong_way_steps = 0
        self._ready = True
        self._last_observations = self.observations()
        return self._last_observations

    @property
    def elapsed_time(self) -> float:
        return self.physics_steps * self.dt

    def observations(self) -> tuple[float, ...]:
        """Exact PLAN.md input order; angles radians, rays relative to heading."""
        if not self._ready:
            raise RuntimeError("Call reset before reading observations.")
        cfg = self.observation_config
        vx, vy, yaw_rate = body_velocities(self.state, self.vehicle_config)
        path = self.track.project(self.state.x, self.state.y)
        rays = self.track.ray_distances(self.state, cfg["ray_angles_deg"], cfg["ray_range_m"])
        values = (
            _clip(vx / cfg["longitudinal_speed_scale_m_s"]),
            _clip(vy / cfg["lateral_speed_scale_m_s"]),
            _clip(yaw_rate / cfg["yaw_rate_scale_rad_s"]),
            math.sin(self.state.heading - path.heading),
            _clip(path.offset / cfg["lateral_offset_scale_m"]),
            *(max(0.0, min(distance / cfg["ray_range_m"], 1.0)) for distance in rays),
            *(wrap_to_pi(self.track.path_heading(path.s + distance) - self.state.heading) / math.pi
              for distance in cfg["lookahead_distances_m"]),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite observations.")
        return values

    def metrics(self) -> dict:
        if not self._ready:
            raise RuntimeError("Call reset before reading metrics.")
        completed = self.done and self.termination_reason == "completed"
        progress = self.progress.maximum_progress
        mean_lap_time = sum(self.lap_times) / len(self.lap_times) if self.lap_times else None
        time_bonus = (self.time_bonus_weight * self.reference_lap_time /
                      (self.reference_lap_time + mean_lap_time)) if completed else 0.0
        return {
            "fitness": progress / self.target_distance + 2 * int(completed) + time_bonus,
            "time_bonus": time_bonus,
            "target_laps": self.target_laps,
            "laps_completed": len(self.lap_times),
            "lap_times": list(self.lap_times),
            "mean_lap_time": mean_lap_time,
            "total_time": self.elapsed_time if completed else None,
            "maximum_validated_progress": progress,
            "progress_fraction": progress / self.target_distance,
            "completed": completed,
            "lap_time": mean_lap_time if completed else None,
            "elapsed_time": self.elapsed_time,
            "termination_reason": self.termination_reason,
            "gates_passed": self.progress.gates_passed,
            "physics_steps": self.physics_steps,
            "decisions": self.decisions,
            "seed": self.seed,
        }

    def _finish(self, reason):
        self.done = True
        self.termination_reason = reason

    def step(self, action: Action):
        """Return (observations, done, metrics); hold commands for five updates.

        Configuration can change the timestep ratio. Collision, progress and
        termination are checked after every physics update. Terminal invalid
        states return the last finite observation. Calls after done raise.
        """
        if not self._ready:
            raise RuntimeError("Call reset before stepping.")
        if self.done:
            raise RuntimeError("Episode has ended; call reset before stepping again.")
        if not isinstance(action, Action):
            raise TypeError("step expects an Action; map network outputs before calling it.")
        self.decisions += 1
        try:
            action = action.bounded()
        except ValueError:
            self._finish("invalid_action")
        for _ in range(self.physics_steps_per_action):
            if self.done:
                break
            try:
                next_state = advance(self.state, action, self.vehicle_config, self.dt)
            except (ValueError, OverflowError):
                self._finish("invalid_state")
                break
            self.state = next_state
            self.physics_steps += 1
            if not self.track.contains_vehicle(self.state, self.vehicle_config):
                self._finish("off_track")
                break
            try:
                self.progress.update(self.state, self.vehicle_config.max_speed_m_s * self.dt)
            except ValueError:
                self._finish("invalid_progress")
                break
            laps_completed = self.progress.gates_passed // len(self.track.checkpoints)
            if laps_completed > len(self.lap_times):
                self.lap_times.append(self.elapsed_time - self._last_lap_finish)
                self._last_lap_finish = self.elapsed_time
            if self.progress.completed:
                self._finish("completed")
                break
            path = self.track.project(self.state.x, self.state.y)
            vx, vy, _ = body_velocities(self.state, self.vehicle_config)
            error = self.state.heading - path.heading
            path_velocity = vx * math.cos(error) - vy * math.sin(error)
            if path_velocity < self.simulation["wrong_way_velocity_m_s"]:
                self._wrong_way_steps += 1
            else:
                self._wrong_way_steps = 0
            if self._wrong_way_steps * self.dt + 1e-9 >= self.simulation["wrong_way_timeout_s"]:
                self._finish("wrong_way")
                break
            if self.progress.maximum_progress - self._progress_anchor + 1e-9 >= self.simulation["stall_progress_m"]:
                self._progress_anchor = self.progress.maximum_progress
                self._last_progress_time = self.elapsed_time
            if self.elapsed_time - self._last_progress_time + 1e-9 >= self.simulation["stall_timeout_s"]:
                self._finish("stalled")
                break
            if self.physics_steps >= self.max_physics_steps:
                self._finish("time_limit")
                break
        if self.termination_reason not in ("invalid_state", "invalid_action"):
            self._last_observations = self.observations()
        return self._last_observations, self.done, self.metrics()
