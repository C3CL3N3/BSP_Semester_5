"""Fixed-car CG bicycle models (legacy kinematic and dynamic tyre forces); SI units throughout.

The footprint is centred at the CG. Positive heading/steering turn left.
These are prototype assumptions, not realistic F1 dynamics.
"""

from dataclasses import asdict, dataclass, fields
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
    """Fixed-car assumptions; mass/inertia enter the dynamic force equations."""

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
    model: str = "kinematic"
    yaw_inertia_kg_m2: float = 1900.0
    front_cornering_stiffness_n_rad: float = 55000.0
    rear_cornering_stiffness_n_rad: float = 60000.0
    friction_coefficient: float = 1.0
    gravity_m_s2: float = 9.81
    low_speed_regularization_m_s: float = 1.0
    integration_dt_s: float = 0.001
    air_density_kg_m3: float = 1.225
    drag_area_m2: float = 0.0
    downforce_area_m2: float = 0.0
    front_downforce_fraction: float = 0.45

    def __post_init__(self):
        if self.model not in ("kinematic", "dynamic"):
            raise ValueError("model must be kinematic or dynamic.")
        for name, value in asdict(self).items():
            if name == "model":
                continue
            _finite(name, value)
            if value < 0 or (name not in ("initial_speed_m_s", "drag_area_m2", "downforce_area_m2", "front_downforce_fraction") and value == 0):
                raise ValueError(f"{name} must be positive (initial speed may be zero).")
        if self.front_downforce_fraction > 1:
            raise ValueError("front_downforce_fraction must lie in [0, 1].")
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
        for name, value in (("steering", self.steering), ("throttle", self.throttle), ("brake", self.brake)):
            _finite(name, value)
        return Action(_clip(self.steering, -1, 1),
                      _clip(self.throttle, 0, 1), _clip(self.brake, 0, 1))


@dataclass
class VehicleState:
    """CG pose, total speed and dynamic body lateral speed/yaw rate (SI).

    The legacy model derives velocities geometrically and ignores the last
    two fields. The dynamic model integrates them as independent states.
    """

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    steering_angle: float = 0.0
    lateral_speed: float = 0.0
    yaw_rate: float = 0.0


def _validate_state(state: VehicleState, config: VehicleConfig) -> None:
    for field in fields(state):
        _finite(field.name, getattr(state, field.name))
    if not 0 <= state.speed <= config.max_speed_m_s:
        raise ValueError("State speed is outside vehicle limits.")
    if abs(state.lateral_speed) > state.speed + 1e-9:
        raise ValueError("Lateral speed cannot exceed total speed.")
    if abs(state.steering_angle) > config.max_steering_rad:
        raise ValueError("State steering angle is outside vehicle limits.")


def body_velocities(state: VehicleState, config: VehicleConfig | Mapping) -> tuple[float, float, float]:
    """Return (vx, vy, yaw_rate) at the CG, including at reset.

    Dynamic mode reads integrated velocities. Legacy mode uses a geometric
    velocity angle beta, which is not dynamic tyre slip.
    """
    cfg = _config(config)
    _validate_state(state, cfg)
    if cfg.model == "dynamic":
        return math.sqrt(max(0.0, state.speed ** 2 - state.lateral_speed ** 2)), state.lateral_speed, state.yaw_rate
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
    if cfg.model == "dynamic":
        return _advance_dynamic(state, command, cfg, dt)
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


@dataclass(frozen=True)
class TyreForces:
    """Axle totals in wheel coordinates; positive slip/force acts left.

    Longitudinal force gets the friction-circle capacity remaining after lateral
    force. This lateral-priority allocation is a deliberate prototype assumption.
    """

    front_slip_angle: float
    rear_slip_angle: float
    front_normal_load: float
    rear_normal_load: float
    front_lateral_force: float
    rear_lateral_force: float
    front_longitudinal_force: float
    rear_longitudinal_force: float


def aerodynamic_forces(speed: float, config: VehicleConfig | Mapping) -> tuple[float, float]:
    """Return drag/downforce magnitudes in N; areas mean Cd*A and Cl*A.

    No wind, pitch or ride-height model. Positive downforce increases tyre load.
    """
    cfg = _config(config)
    _finite("speed", speed)
    if speed < 0:
        raise ValueError("speed must be nonnegative.")
    pressure = 0.5 * cfg.air_density_kg_m3 * speed * speed
    return pressure * cfg.drag_area_m2, pressure * cfg.downforce_area_m2


def _tyre_forces(vx, vy, yaw, steering, command, cfg):
    _, downforce = aerodynamic_forces(math.hypot(vx, vy), cfg)
    front_load = cfg.mass_kg * cfg.gravity_m_s2 * cfg.lr_m / cfg.wheelbase_m
    rear_load = cfg.mass_kg * cfg.gravity_m_s2 * cfg.lf_m / cfg.wheelbase_m
    front_load += downforce * cfg.front_downforce_fraction
    rear_load += downforce * (1 - cfg.front_downforce_fraction)
    front_vy, rear_vy = vy + cfg.lf_m * yaw, vy - cfg.lr_m * yaw
    # At low speed the slip law is regularized and its force ramps from zero.
    denominator = max(vx, cfg.low_speed_regularization_m_s)
    alpha_f = steering - math.atan2(front_vy, denominator)
    alpha_r = -math.atan2(rear_vy, denominator)
    front_ramp = min(1.0, math.hypot(vx, front_vy) / cfg.low_speed_regularization_m_s)
    rear_ramp = min(1.0, math.hypot(vx, rear_vy) / cfg.low_speed_regularization_m_s)
    if math.hypot(vx, front_vy) < 1e-12:
        alpha_f = 0.0
    if math.hypot(vx, rear_vy) < 1e-12:
        alpha_r = 0.0
    front_limit, rear_limit = cfg.friction_coefficient * front_load, cfg.friction_coefficient * rear_load
    fyf = _clip(cfg.front_cornering_stiffness_n_rad * alpha_f * front_ramp, -front_limit, front_limit)
    fyr = _clip(cfg.rear_cornering_stiffness_n_rad * alpha_r * rear_ramp, -rear_limit, rear_limit)
    request = cfg.mass_kg * (cfg.max_acceleration_m_s2 * command.throttle * (1 - command.brake)
                             - cfg.max_braking_m_s2 * command.brake)
    # Drive and brake demands are shared by static axle load (no load transfer).
    remaining_f = math.sqrt(max(0.0, front_limit ** 2 - fyf ** 2))
    remaining_r = math.sqrt(max(0.0, rear_limit ** 2 - fyr ** 2))
    fxf = _clip(request * cfg.lr_m / cfg.wheelbase_m, -remaining_f, remaining_f)
    fxr = _clip(request * cfg.lf_m / cfg.wheelbase_m, -remaining_r, remaining_r)
    if request < 0:
        # A brake resists wheel motion; it must not launch a stopped car or
        # create a lateral force from rotated front wheels at standstill.
        front_forward = max(0.0, vx * math.cos(steering) + front_vy * math.sin(steering))
        fxf = max(fxf, -cfg.mass_kg * cfg.lr_m / cfg.wheelbase_m * front_forward / cfg.integration_dt_s)
        fxr = max(fxr, -cfg.mass_kg * cfg.lf_m / cfg.wheelbase_m * max(vx, 0.0) / cfg.integration_dt_s)
    return TyreForces(alpha_f, alpha_r, front_load, rear_load, fyf, fyr, fxf, fxr)


def tyre_forces(state: VehicleState, action: Action, config: VehicleConfig | Mapping) -> TyreForces:
    """Inspect axle slip, load and friction-limited forces at the current state."""
    cfg = _config(config)
    if cfg.model != "dynamic":
        raise ValueError("Tyre forces require the dynamic model.")
    vx, vy, yaw = body_velocities(state, cfg)
    return _tyre_forces(vx, vy, yaw, state.steering_angle, action.bounded(), cfg)


def _advance_dynamic(state, command, cfg, dt):
    # Substepping avoids stiff low-speed tyre oscillations at the 0.01 s API rate.
    steps = math.ceil(dt / cfg.integration_dt_s)
    h = dt / steps
    vx, vy, yaw = body_velocities(state, cfg)
    x, y, heading, steering = state.x, state.y, state.heading, state.steering_angle
    for _ in range(steps):
        target = command.steering * cfg.max_steering_rad
        steering += _clip(target - steering, -cfg.max_steering_rate_rad_s * h,
                          cfg.max_steering_rate_rad_s * h)
        forces = _tyre_forces(vx, vy, yaw, steering, command, cfg)
        cosine, sine = math.cos(steering), math.sin(steering)
        front_x = forces.front_longitudinal_force * cosine - forces.front_lateral_force * sine
        front_y = forces.front_longitudinal_force * sine + forces.front_lateral_force * cosine
        speed = math.hypot(vx, vy)
        drag, _ = aerodynamic_forces(speed, cfg)
        drag_x, drag_y = (drag * vx / speed, drag * vy / speed) if speed > 0 else (0.0, 0.0)
        dvx = (front_x + forces.rear_longitudinal_force - drag_x) / cfg.mass_kg + yaw * vy
        dvy = (front_y + forces.rear_lateral_force - drag_y) / cfg.mass_kg - yaw * vx
        dyaw = (cfg.lf_m * front_y - cfg.lr_m * forces.rear_lateral_force) / cfg.yaw_inertia_kg_m2
        vx, vy, yaw = max(0.0, vx + dvx * h), vy + dvy * h, yaw + dyaw * h
        speed = math.hypot(vx, vy)
        if speed > cfg.max_speed_m_s:
            scale = cfg.max_speed_m_s / speed
            vx, vy = vx * scale, vy * scale
        x += (vx * math.cos(heading) - vy * math.sin(heading)) * h
        y += (vx * math.sin(heading) + vy * math.cos(heading)) * h
        heading = wrap_to_pi(heading + yaw * h)
    result = VehicleState(x, y, heading, min(math.hypot(vx, vy), cfg.max_speed_m_s),
                          _clip(steering, -cfg.max_steering_rad, cfg.max_steering_rad), vy, yaw)
    _validate_state(result, cfg)
    return result
