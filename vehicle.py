"""Fixed-car CG kinematic bicycle model; SI units throughout.

The footprint is centred at the CG. Positive heading/steering turn left.
These are prototype assumptions, not realistic F1 dynamics. See PLAN.md.
"""

from dataclasses import asdict, dataclass
import math
from typing import Mapping


def _finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def wrap_to_pi(angle: float) -> float:
    """Wrap a finite angle to [-pi, pi)."""
    _finite("angle", angle)
    return (angle + math.pi) % math.tau - math.pi


@dataclass(frozen=True)
class VehicleConfig:
    """Validated, serializable fixed-car parameters; mass is metadata only."""

    mass_kg: float = 760.0
    body_length_m: float = 5.2
    body_width_m: float = 1.9
    wheelbase_m: float = 3.4
    lf_m: float = 1.55
    lr_m: float = 1.85
    max_steering_rad: float = 0.35
    max_steering_rate_rad_s: float = 0.8
    max_acceleration_m_s2: float = 4.0
    max_braking_m_s2: float = 8.0
    max_speed_m_s: float = 30.0
    initial_speed_m_s: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            _finite(name, value)
            if value < 0 or (name != "initial_speed_m_s" and value == 0):
                raise ValueError(f"{name} must be positive (initial speed may be zero).")
        if not math.isclose(self.lf_m + self.lr_m, self.wheelbase_m, abs_tol=1e-9):
            raise ValueError("lf_m + lr_m must equal wheelbase_m.")
        if self.max_steering_rad >= math.pi / 2:
            raise ValueError("max_steering_rad must be less than pi/2.")
        if self.initial_speed_m_s > self.max_speed_m_s:
            raise ValueError("initial_speed_m_s must not exceed max_speed_m_s.")

    def to_dict(self) -> dict:
        return asdict(self)


def _config(config: VehicleConfig | Mapping) -> VehicleConfig:
    """Accept a validated config or the vehicle section of config.json."""
    return config if isinstance(config, VehicleConfig) else VehicleConfig(**config)


@dataclass(frozen=True)
class Action:
    """Simulator commands: steering in [-1, 1], throttle/brake in [0, 1]."""

    steering: float
    throttle: float
    brake: float

    def bounded(self) -> "Action":
        for name, value in asdict(self).items():
            _finite(name, value)
        return Action(_clip(self.steering, -1, 1),
                      _clip(self.throttle, 0, 1), _clip(self.brake, 0, 1))


@dataclass
class VehicleState:
    """Position at CG, angles in radians, speed in metres per second."""

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    steering_angle: float = 0.0


def _validate_state(state: VehicleState, config: VehicleConfig) -> None:
    for name, value in asdict(state).items():
        _finite(name, value)
    if not 0 <= state.speed <= config.max_speed_m_s:
        raise ValueError("State speed is outside vehicle limits.")
    if abs(state.steering_angle) > config.max_steering_rad:
        raise ValueError("State steering angle is outside vehicle limits.")


def body_velocities(state: VehicleState, config: VehicleConfig | Mapping) -> tuple[float, float, float]:
    """Return (vx, vy, yaw_rate) at the CG, including at reset.

    The geometric velocity angle beta is not dynamic tyre slip.
    """
    cfg = _config(config)
    _validate_state(state, cfg)
    beta = math.atan((cfg.lr_m / cfg.wheelbase_m) * math.tan(state.steering_angle))
    vx = state.speed * math.cos(beta)
    vy = state.speed * math.sin(beta)
    return vx, vy, vx / cfg.wheelbase_m * math.tan(state.steering_angle)


def footprint_corners(state: VehicleState, config: VehicleConfig | Mapping) -> tuple[tuple[float, float], ...]:
    """Return four CG-centred rectangle corners in counterclockwise order.

    Track geometry is responsible for containment and boundary contact rules.
    """
    cfg = _config(config)
    _validate_state(state, cfg)
    half_length, half_width = cfg.body_length_m / 2, cfg.body_width_m / 2
    cosine, sine = math.cos(state.heading), math.sin(state.heading)
    return tuple((state.x + x * cosine - y * sine,
                  state.y + x * sine + y * cosine)
                 for x, y in ((half_length, half_width), (-half_length, half_width),
                              (-half_length, -half_width), (half_length, -half_width)))


def advance(state: VehicleState, action: Action, config: VehicleConfig | Mapping, dt: float) -> VehicleState:
    """Advance one physics step without mutating the input state.

    Use dt=0.01 s for this prototype; the environment holds each action for
    five steps. Finite commands are clamped; invalid states/configs fail fast.
    Position uses the updated speed/steering and the previous heading.
    """
    cfg = _config(config)
    _finite("dt", dt)
    if dt <= 0:
        raise ValueError("dt must be positive.")
    _validate_state(state, cfg)
    command = action.bounded()
    target = command.steering * cfg.max_steering_rad
    max_change = cfg.max_steering_rate_rad_s * dt
    steering = _clip(state.steering_angle + _clip(target - state.steering_angle,
                                               -max_change, max_change),
                     -cfg.max_steering_rad, cfg.max_steering_rad)
    acceleration = (cfg.max_acceleration_m_s2 * command.throttle * (1 - command.brake)
                    - cfg.max_braking_m_s2 * command.brake)
    speed = _clip(state.speed + acceleration * dt, 0, cfg.max_speed_m_s)
    next_state = VehicleState(state.x, state.y, state.heading, speed, steering)
    vx, vy, yaw_rate = body_velocities(next_state, cfg)
    cosine, sine = math.cos(state.heading), math.sin(state.heading)
    next_state.x += (vx * cosine - vy * sine) * dt
    next_state.y += (vx * sine + vy * cosine) * dt
    next_state.heading = wrap_to_pi(state.heading + yaw_rate * dt)
    _validate_state(next_state, cfg)
    return next_state
