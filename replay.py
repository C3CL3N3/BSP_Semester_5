"""Pygame replay of saved baseline/champion episodes; no physics or training.

Space pauses, arrows seek five seconds, R restarts, Escape exits.
Use --export for a static PNG, or --headless for a display-free smoke check.
"""

import argparse
from bisect import bisect_right
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

from environment import ROOT
from track import Track
from vehicle import VehicleConfig, VehicleState, footprint_corners, wrap_to_pi


@dataclass
class Replay:
    driver: str
    config: dict
    metrics: dict
    trajectory: list

    @property
    def duration(self):
        return self.trajectory[-1]["elapsed_time"]

    def sample(self, elapsed):
        """Interpolate recorded positions/times, taking the short angular arc."""
        elapsed = max(0.0, min(elapsed, self.duration))
        index = max(0, bisect_right(self.times, elapsed) - 1)
        a = self.trajectory[index]
        if index == len(self.trajectory) - 1 or elapsed == a["elapsed_time"]:
            return dict(a)
        b = self.trajectory[index + 1]
        fraction = (elapsed - a["elapsed_time"]) / (b["elapsed_time"] - a["elapsed_time"])
        result = dict(a)
        for key in ("x", "y", "speed", "steering_angle", "maximum_validated_progress"):
            result[key] = a[key] + fraction * (b[key] - a[key])
        for key in ("lateral_speed", "yaw_rate"):
            if key in a and key in b:
                result[key] = a[key] + fraction * (b[key] - a[key])
        result["heading"] = wrap_to_pi(a["heading"] + fraction * wrap_to_pi(b["heading"] - a["heading"]))
        result["elapsed_time"] = elapsed
        return result

    def __post_init__(self):
        self.times = [row["elapsed_time"] for row in self.trajectory]


def load_replay(source) -> Replay:
    """Load baseline JSON, champion episode JSON, or a training run directory."""
    source = Path(source)
    if source.is_dir():
        source = source / "champion_episode.json"
    with source.open(encoding="utf-8") as handle:
        data = json.load(handle)
    config = data.get("config")
    if config is None:
        with (source.parent / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
    vehicle = VehicleConfig(**config["vehicle"])
    track = Track(config["track"])
    driver = data.get("driver")
    labels = {"rule_based_baseline": "Rule-based baseline", "neat": "NEAT champion"}
    if driver not in labels:
        raise ValueError("Expected a baseline or NEAT episode artifact.")
    trajectory = data.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("Replay trajectory is empty or missing.")
    previous_time = -1.0
    for row in trajectory:
        for key in ("elapsed_time", "x", "y", "heading", "speed", "steering_angle", "maximum_validated_progress"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid trajectory value: {key}")
        if row["elapsed_time"] < 0 or row["elapsed_time"] <= previous_time:
            raise ValueError("Trajectory timestamps must strictly increase from zero.")
        if not 0 <= row["speed"] <= vehicle.max_speed_m_s:
            raise ValueError("Replay speed outside vehicle limits.")
        if abs(row["steering_angle"]) > vehicle.max_steering_rad:
            raise ValueError("Replay steering outside vehicle limits.")
        if not 0 <= row["maximum_validated_progress"] <= track.length * config["simulation"].get("target_laps", 1):
            raise ValueError("Replay progress outside track length.")
        previous_time = row["elapsed_time"]
    if trajectory[0]["elapsed_time"] != 0:
        raise ValueError("Replay must include the initial pose at time zero.")
    metrics = data["metrics"]
    if not math.isclose(metrics["elapsed_time"], trajectory[-1]["elapsed_time"], abs_tol=1e-8):
        raise ValueError("Metrics and trajectory durations disagree.")
    return Replay(labels[driver], config, metrics, trajectory)


class Renderer:
    def __init__(self, replay, size=(1100, 760)):
        import pygame
        self.pg = pygame
        pygame.font.init()
        self.replay = replay
        self.track = Track(replay.config["track"])
        self.vehicle = VehicleConfig(**replay.config["vehicle"])
        self.surface = pygame.Surface(size)
        self.font = pygame.font.Font(None, 27)
        self.small_font = pygame.font.Font(None, 22)
        min_x, min_y, max_x, max_y = self.track.road.bounds
        self.scale = min((size[0] - 100) / (max_x - min_x), (size[1] - 230) / (max_y - min_y))
        self.cx, self.cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        self.origin = (size[0] / 2, 140 + (size[1] - 220) / 2)
        self.base = pygame.Surface(size)
        self.base.fill((18, 27, 34))
        pygame.draw.polygon(self.base, (65, 75, 83), [self.point(x, y) for x, y in self.track.road.exterior.coords])
        for hole in self.track.road.interiors:
            pygame.draw.polygon(self.base, (18, 27, 34), [self.point(x, y) for x, y in hole.coords])
        points = [self.point(*self.track.path_pose(i * self.track.length / 500)[:2]) for i in range(501)]
        pygame.draw.lines(self.base, (115, 128, 136), False, points, 1)
        for gate in self.track.checkpoints:
            pygame.draw.line(self.base, (88, 99, 105), *[self.point(x, y) for x, y in gate.coords], 1)
        pygame.draw.line(self.base, (240, 240, 240), *[self.point(x, y) for x, y in self.track.checkpoints[0].coords], 3)

    def point(self, x, y):
        return (round(self.origin[0] + (x - self.cx) * self.scale),
                round(self.origin[1] - (y - self.cy) * self.scale))

    def draw(self, elapsed, *, static=False, paused=False, playback_speed=1.0):
        pg = self.pg
        self.surface.blit(self.base, (0, 0))
        row = self.replay.sample(elapsed)
        rows = self.replay.trajectory if static else self.replay.trajectory[:bisect_right(self.replay.times, elapsed)]
        points = [self.point(r["x"], r["y"]) for r in rows]
        if not static:
            points.append(self.point(row["x"], row["y"]))
        if len(points) >= 2:
            pg.draw.lines(self.surface, (56, 208, 193), False, points, 2)
        state = VehicleState(**{key: row[key] for key in ("x", "y", "heading", "speed", "steering_angle")})
        corners = [self.point(x, y) for x, y in footprint_corners(state, self.vehicle)]
        pg.draw.polygon(self.surface, (255, 190, 70), corners)
        tip = self.point(state.x + self.vehicle.body_length_m / 2 * math.cos(state.heading),
                         state.y + self.vehicle.body_length_m / 2 * math.sin(state.heading))
        pg.draw.line(self.surface, (30, 30, 30), self.point(state.x, state.y), tip, 2)
        target_laps = self.replay.config["simulation"].get("target_laps", 1)
        target_distance = self.track.length * target_laps
        progress = row["maximum_validated_progress"] / target_distance
        laps = min(target_laps, int((row["maximum_validated_progress"] + 1e-8) / self.track.length))
        terminal = elapsed >= self.replay.duration
        status = self.replay.metrics["termination_reason"] if terminal or static else ("paused" if paused else "playing")
        lines = [self.replay.driver + (" — recorded trajectory" if static else " — recorded replay"),
                 f'Time {row["elapsed_time"]:.2f} / {self.replay.duration:.2f} s    Speed {row["speed"]:.2f} m/s ({row["speed"] * 3.6:.1f} km/h)',
                 f'Progress {progress:.1%} | Laps {laps}/{target_laps} | {row["maximum_validated_progress"]:.1f}/{target_distance:.1f} m | {status}',
                 f'Final fitness {self.replay.metrics["fitness"]:.4f}    Completed: {self.replay.metrics["completed"]}']
        for i, text in enumerate(lines):
            self.surface.blit(self.font.render(text, True, (232, 239, 242)), (24, 15 + i * 29))
        model = self.replay.config["vehicle"].get("model", "kinematic")
        help_text = f"Saved samples; interpolated motion. Simplified {model} bicycle physics."
        if not static:
            help_text += f"  {playback_speed:g}x | Space: pause | arrows: seek | R: restart | Esc: quit"
        self.surface.blit(self.small_font.render(help_text, True, (186, 199, 205)), (24, self.surface.get_height() - 34))
        return self.surface


def export_plot(replay, output):
    """Save a static whole-lap trajectory without opening a display."""
    import pygame
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    renderer = Renderer(replay)
    surface = renderer.draw(replay.duration, static=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(output))


def play(replay, *, speed=1.0, headless=False, frames=None, close_on_finish=False):
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Playback speed must be positive and finite.")
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame
    pygame.display.init()
    try:
        renderer = Renderer(replay)
        screen = pygame.display.set_mode(renderer.surface.get_size())
        pygame.display.set_caption(replay.driver)
        clock = pygame.time.Clock()
        elapsed, paused, count = 0.0, False, 0
        while frames is None or count < frames:
            dt = clock.tick(60) / 1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    if event.key == pygame.K_r:
                        elapsed = 0.0
                    if event.key == pygame.K_LEFT:
                        elapsed = max(0.0, elapsed - 5)
                    if event.key == pygame.K_RIGHT:
                        elapsed = min(replay.duration, elapsed + 5)
            if not paused:
                elapsed = min(replay.duration, elapsed + dt * speed)
            screen.blit(renderer.draw(elapsed, paused=paused, playback_speed=speed), (0, 0))
            pygame.display.flip()
            count += 1
            if (headless or close_on_finish) and elapsed >= replay.duration:
                return
    finally:
        pygame.display.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Baseline JSON, champion episode JSON, or training directory")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--export", type=Path, help="Save static trajectory PNG instead of opening a window")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, help="Stop after N animation frames (smoke checks)")
    args = parser.parse_args()
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    try:
        replay = load_replay(args.source)
        if args.export:
            export_plot(replay, args.export)
            print(f"Saved trajectory plot: {args.export.resolve()}")
        else:
            play(replay, speed=args.speed, headless=args.headless, frames=args.frames)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Replay error: {error}\n")


if __name__ == "__main__":
    main()
