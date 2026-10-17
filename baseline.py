"""Rule-based driver using only the same 16 observations supplied to NEAT."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
from importlib.metadata import version

from environment import Environment, ROOT, load_config
from vehicle import Action

# Deliberately conservative settings for environment validation, not lap-time tuning.
DRIVER_SETTINGS = {
    "target_speed_m_s": 12.0,
    "heading_gain": 1.5,
    "offset_gain_rad_per_m": 0.15,
    "lookahead_gain": 0.08,
    "speed_gain": 0.5,
}


def choose_action(observations, config=None) -> Action:
    """Follow the path using heading, offset and the near lookahead input.

    The sine heading input is inverted on its small-error branch; this driver
    is intended for the documented forward spawn, not wrong-way recovery.
    Configuration supplies normalization scales and fixed vehicle parameters,
    never the car pose or privileged track queries.
    """
    if len(observations) != 16 or not all(math.isfinite(value) for value in observations):
        raise ValueError("Expected 16 finite observations.")
    config = load_config() if config is None else config
    scales = config["observations"]
    heading_error = math.asin(max(-1.0, min(observations[3], 1.0)))
    offset = observations[4] * scales["lateral_offset_scale_m"]
    lookahead = observations[14] * math.pi
    wheel_angle = (-DRIVER_SETTINGS["heading_gain"] * heading_error
                   - DRIVER_SETTINGS["offset_gain_rad_per_m"] * offset
                   + DRIVER_SETTINGS["lookahead_gain"] * lookahead)
    vx = observations[0] * scales["longitudinal_speed_scale_m_s"]
    vy = observations[1] * scales["lateral_speed_scale_m_s"]
    speed = math.hypot(vx, vy)
    speed_error = DRIVER_SETTINGS["target_speed_m_s"] - speed
    effort = DRIVER_SETTINGS["speed_gain"] * speed_error
    return Action(wheel_angle / config["vehicle"]["max_steering_rad"],
                  max(0.0, effort), max(0.0, -effort)).bounded()


def run_baseline(config: dict, seed=None) -> dict:
    """Drive a real episode and return controller-rate samples plus exact settings.

    All motion is produced by Environment.step. Containment and ordered gates
    are checked at physics rate by the unchanged environment.
    """
    env = Environment(config)
    observations = env.reset(seed)
    trajectory = [{"elapsed_time": 0.0, **asdict(env.state),
                   "maximum_validated_progress": 0.0, "action": None}]
    while not env.done:
        action = choose_action(observations, env.config)
        observations, _, metrics = env.step(action)
        trajectory.append({"elapsed_time": env.elapsed_time, **asdict(env.state),
                           "maximum_validated_progress": metrics["maximum_validated_progress"],
                           "action": asdict(action)})
    return {
        "driver": "rule_based_baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "driver_settings": dict(DRIVER_SETTINGS),
        "config": env.config,
        "versions": {"python": platform.python_version(), "shapely": version("shapely")},
        "metrics": env.metrics(),
        "trajectory": trajectory,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="New result JSON path; existing files are never overwritten.")
    args = parser.parse_args()
    output = args.output or ROOT / "outputs" / f"baseline-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
    if output.exists():
        parser.error(f"Output already exists: {output}")
    result = run_baseline(load_config(args.config), args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved baseline result and trajectory: {output.resolve()}")
    return 0 if result["metrics"]["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
