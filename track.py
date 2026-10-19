"""Self-contained counterclockwise stadium oval, in metres.

The start is the midpoint of the lower straight, facing east. Path queries
are analytic. Collision/ray geometry uses circular arcs approximated by
256 segments per quadrant (under 0.3 mm radial error at default dimensions).
"""

from dataclasses import dataclass
import math
import json
from pathlib import Path

import numpy as np

from shapely.geometry import LineString, Point, Polygon
from shapely.prepared import prep

from vehicle import VehicleConfig, VehicleState, footprint_corners, wrap_to_pi


def _number(name, value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")


@dataclass(frozen=True)
class PathProjection:
    s: float
    x: float
    y: float
    heading: float
    offset: float  # positive to the left of the path


class Track:
    def __init__(self, config: dict):
        self.config = dict(config)
        if config.get("shape") in ("file", "polyline"):
            self._init_polyline(config)
            return
        if config.get("shape") != "oval":
            raise ValueError("Only the oval track is supported.")
        for key in ("width_m", "straight_length_m", "turn_radius_m"):
            _number(key, config[key], positive=True)
        self.width = config["width_m"]
        self.straight_length = config["straight_length_m"]
        self.radius = config["turn_radius_m"]
        if self.width / 2 >= self.radius:
            raise ValueError("Track half-width must be less than turn radius.")
        count = config["checkpoint_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 4:
            raise ValueError("checkpoint_count must be an integer >= 4.")
        self.boundary_contact_allowed = config.get("boundary_contact_allowed", True)
        if not isinstance(self.boundary_contact_allowed, bool):
            raise ValueError("boundary_contact_allowed must be boolean.")
        self.length = 2 * self.straight_length + math.tau * self.radius
        axis = LineString([(-self.straight_length / 2, 0), (self.straight_length / 2, 0)])
        outer = axis.buffer(self.radius + self.width / 2, quad_segs=256)
        inner = axis.buffer(self.radius - self.width / 2, quad_segs=256)
        self.road = outer.difference(inner)
        self.boundary = self.road.boundary
        self._prepared_road = prep(self.road)
        self.checkpoint_spacing = self.length / count
        self.checkpoints = tuple(self._gate(i * self.checkpoint_spacing) for i in range(count))
        self.progress_distance_factor = self.radius / (self.radius - self.width / 2)

    def _init_polyline(self, config):
        data = dict(config)
        if "reference_points" not in data:
            path = Path(data["path"])
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            with path.open(encoding="utf-8") as handle:
                asset = json.load(handle)
            if asset.get("schema_version") != 1 or asset.get("units") != "m":
                raise ValueError("Track asset requires schema_version 1 and metre units.")
            data = {**asset, **data}
        points = np.asarray(data["reference_points"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 5 or not np.isfinite(points).all():
            raise ValueError("Track requires at least four finite 2D points and closure.")
        if not np.allclose(points[0], points[-1], rtol=0, atol=1e-6):
            raise ValueError("Reference path must be explicitly closed.")
        points[-1] = points[0]
        self._segment_start = points[:-1]
        self._segment_vector = np.diff(points, axis=0)
        self._segment_length = np.linalg.norm(self._segment_vector, axis=1)
        if np.any(self._segment_length < 1e-6):
            raise ValueError("Reference path has repeated consecutive points.")
        self._arc = np.concatenate(([0.0], np.cumsum(self._segment_length)))
        self.length = float(self._arc[-1])
        self._line = LineString(points)
        if not self._line.is_simple:
            raise ValueError("Reference path must not cross itself.")
        self.width = data["width_m"]
        _number("width_m", self.width, positive=True)
        self.boundary_contact_allowed = data.get("boundary_contact_allowed", True)
        if not isinstance(self.boundary_contact_allowed, bool):
            raise ValueError("boundary_contact_allowed must be boolean.")
        self.road = self._line.buffer(self.width / 2, quad_segs=16)
        if self.road.geom_type != "Polygon" or len(self.road.interiors) != 1:
            raise ValueError("Road must form a single closed loop without merged sections.")
        self.boundary = self.road.boundary
        self._prepared_road = prep(self.road)
        spacing = data.get("checkpoint_spacing_m", 25.0)
        _number("checkpoint_spacing_m", spacing, positive=True)
        count = data.get("checkpoint_count", max(4, math.ceil(self.length / spacing)))
        if isinstance(count, bool) or not isinstance(count, int) or count < 4:
            raise ValueError("checkpoint_count must be an integer >= 4.")
        self.checkpoint_spacing = self.length / count
        self.checkpoints = tuple(self._gate(i * self.checkpoint_spacing) for i in range(count))
        # Conservative allowance for arc-length magnification on inner bends.
        # Gate order and actual Euclidean reachability are independently checked.
        self.progress_distance_factor = 4.0
        self.config = {**data, "shape": "polyline", "reference_points": points.tolist(), "checkpoint_count": count}
        self.config.pop("path", None)

    def path_pose(self, s: float) -> tuple[float, float, float]:
        """Return exact (x, y, tangent heading) at wrapped arc length."""
        _number("s", s)
        if hasattr(self, "_arc"):
            s %= self.length
            index = min(int(np.searchsorted(self._arc, s, side="right")) - 1, len(self._segment_length) - 1)
            fraction = (s - self._arc[index]) / self._segment_length[index]
            x, y = self._segment_start[index] + fraction * self._segment_vector[index]
            dx, dy = self._segment_vector[index]
            return float(x), float(y), math.atan2(dy, dx)
        # Start halfway along the bottom straight; parameter u starts at its left end.
        u = (s + self.straight_length / 2) % self.length
        d, r = self.straight_length, self.radius
        if u < d:
            return -d / 2 + u, -r, 0.0
        u -= d
        if u < math.pi * r:
            angle = -math.pi / 2 + u / r
            return d / 2 + r * math.cos(angle), r * math.sin(angle), wrap_to_pi(angle + math.pi / 2)
        u -= math.pi * r
        if u < d:
            return d / 2 - u, r, -math.pi
        angle = math.pi / 2 + (u - d) / r
        return -d / 2 + r * math.cos(angle), r * math.sin(angle), wrap_to_pi(angle + math.pi / 2)

    def path_heading(self, s: float) -> float:
        return self.path_pose(s)[2]

    def project(self, x: float, y: float) -> PathProjection:
        """Nearest centreline point and signed lateral offset; s is in [0, length)."""
        _number("x", x)
        _number("y", y)
        if hasattr(self, "_arc"):
            s = float(self._line.project(Point(x, y))) % self.length
            px, py, heading = self.path_pose(s)
            offset = -(x - px) * math.sin(heading) + (y - py) * math.cos(heading)
            return PathProjection(s, px, py, heading, offset)
        d, r = self.straight_length, self.radius
        candidates = []
        bottom_x = max(-d / 2, min(x, d / 2))
        candidates.append(bottom_x + d / 2)
        candidates.append(d + math.pi * r + d / 2 - bottom_x)
        for cx, start_angle, base in ((d / 2, -math.pi / 2, d),
                                      (-d / 2, math.pi / 2, 2 * d + math.pi * r)):
            angle = math.atan2(y, x - cx)
            # Right semicircle has x >= cx; left semicircle has x <= cx.
            if cx > 0:
                angle = max(-math.pi / 2, min(angle, math.pi / 2))
            else:
                if angle < 0:
                    angle += math.tau
                angle = max(math.pi / 2, min(angle, 3 * math.pi / 2))
            candidates.append(base + (angle - start_angle) * r)
        poses = [(u - d / 2) % self.length for u in candidates]
        s = min(poses, key=lambda v: (self.path_pose(v)[0] - x) ** 2 + (self.path_pose(v)[1] - y) ** 2)
        px, py, heading = self.path_pose(s)
        offset = -(x - px) * math.sin(heading) + (y - py) * math.cos(heading)
        return PathProjection(s, px, py, heading, offset)

    def start_state(self) -> VehicleState:
        """Return a fresh stationary CG pose at the start, facing forward."""
        x, y, heading = self.path_pose(0)
        return VehicleState(x=x, y=y, heading=heading)

    def contains_vehicle(self, state: VehicleState, vehicle_config: VehicleConfig | dict) -> bool:
        footprint = Polygon(footprint_corners(state, vehicle_config))
        if self.boundary_contact_allowed:
            return self._prepared_road.covers(footprint)
        return self._prepared_road.contains_properly(footprint)

    def ray_distances(self, state: VehicleState, angles_deg, max_range_m: float) -> tuple[float, ...]:
        """Distances to the first boundary, clamped to range; misses use range.

        A ray whose origin touches the boundary returns zero. Rays from outside
        the road still report intersections; containment must be checked separately.
        """
        for name in ("x", "y", "heading"):
            _number(name, getattr(state, name))
        _number("max_range_m", max_range_m, positive=True)
        origin = Point(state.x, state.y)
        distances = []
        for angle in angles_deg:
            _number("ray angle", angle)
            theta = state.heading + math.radians(angle)
            ray = LineString([(state.x, state.y),
                              (state.x + max_range_m * math.cos(theta),
                               state.y + max_range_m * math.sin(theta))])
            hit = self.boundary.intersection(ray)
            distances.append(max_range_m if hit.is_empty else min(max_range_m, origin.distance(hit)))
        return tuple(distances)

    def _gate(self, s: float) -> LineString:
        x, y, heading = self.path_pose(s)
        nx, ny = -math.sin(heading) * self.width / 2, math.cos(heading) * self.width / 2
        return LineString([(x - nx, y - ny), (x + nx, y + ny)])

    def crosses_gate(self, previous: VehicleState, current: VehicleState, index: int) -> bool:
        """True only for a forward CG crossing of the finite checkpoint gate."""
        if not 0 <= index < len(self.checkpoints):
            raise ValueError("Checkpoint index out of range.")
        for state in (previous, current):
            _number("x", state.x)
            _number("y", state.y)
        x, y, heading = self.path_pose(index * self.checkpoint_spacing)
        tx, ty = math.cos(heading), math.sin(heading)
        before = (previous.x - x) * tx + (previous.y - y) * ty
        after = (current.x - x) * tx + (current.y - y) * ty
        return (before < 0 <= after and
                LineString([(previous.x, previous.y), (current.x, current.y)]).intersects(self.checkpoints[index]))


class ProgressTracker:
    """Independent multi-lap progress state, reset by constructing a new tracker.

    Call update after each valid, on-road physics step. Pass max_speed * dt
    as max_distance_m. Invalid/unreachable moves raise ValueError without
    changing progress. The environment must terminate on invalid moves.
    """

    def __init__(self, track: Track, target_laps: int = 1):
        if isinstance(target_laps, bool) or not isinstance(target_laps, int) or target_laps < 1:
            raise ValueError("target_laps must be a positive integer.")
        self.target_laps = target_laps
        self.target_distance = track.length * target_laps
        self.track = track
        self.previous = track.start_state()
        self._s = 0.0
        self._unwrapped = 0.0
        self.maximum_progress = 0.0
        self.gates_passed = 0
        self.completed = False

    def update(self, current: VehicleState, max_distance_m: float) -> float:
        _number("max_distance_m", max_distance_m, positive=True)
        if max_distance_m >= self.track.checkpoint_spacing / 2:
            raise ValueError("Progress updates must be shorter than half a checkpoint interval.")
        projection = self.track.project(current.x, current.y)
        delta = (projection.s - self._s + self.track.length / 2) % self.track.length - self.track.length / 2
        distance = math.hypot(current.x - self.previous.x, current.y - self.previous.y)
        # Arc-length motion may exceed CG motion on the inner edge of a turn.
        arc_bound = max_distance_m * self.track.progress_distance_factor
        if distance > max_distance_m + 1e-8 or abs(delta) > arc_bound + 1e-8:
            raise ValueError("Unreachable progress change.")
        if self.completed:
            return self.maximum_progress
        next_gate = (self.gates_passed + 1) % len(self.track.checkpoints)
        unwrapped = self._unwrapped + delta
        gate_s = (self.gates_passed + 1) * self.track.checkpoint_spacing
        crossed = self.track.crosses_gate(self.previous, current, next_gate)
        if unwrapped > gate_s + 1e-8 and not crossed:
            raise ValueError("Checkpoint skipped.")
        if crossed and delta > 0 and abs(unwrapped - gate_s) <= arc_bound + 1e-8:
            self.gates_passed += 1
            self.completed = self.gates_passed == len(self.track.checkpoints) * self.target_laps
        self._s, self._unwrapped = projection.s, unwrapped
        self.previous = VehicleState(current.x, current.y, current.heading, current.speed, current.steering_angle)
        self.maximum_progress = max(self.maximum_progress, min(max(0.0, unwrapped), self.target_distance))
        if self.completed:
            self.maximum_progress = self.target_distance
        return self.maximum_progress


def save_track_plot(track: Track, output):
    """Export a layout preview without changing the active simulation track."""
    import os
    cache = Path(__file__).resolve().parent / "outputs" / ".plot-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    fig, ax = plt.subplots(figsize=(16, 12))
    x, y = track.road.exterior.xy
    ax.fill(x, y, color="#65717b")
    for hole in track.road.interiors:
        ax.fill(*hole.xy, color="white")
    points = np.array([track.path_pose(s)[:2] for s in np.linspace(0, track.length, 2500)])
    ax.plot(points[:, 0], points[:, 1], color="#18242d", linewidth=0.6)
    start = track.start_state()
    ax.plot(start.x, start.y, "o", color="#d86a16", label="Start / finish")
    ax.set(title=f'{track.config.get("name", "Oval")} | {track.length:.1f} m | {track.width:g} m wide',
           xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400)
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preview a configured track without running an episode.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.json")
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    track = Track(config["track"])
    save_track_plot(track, args.plot)
    print(f"Saved {track.length:.2f} m track preview: {args.plot.resolve()}")


if __name__ == "__main__":
    main()
