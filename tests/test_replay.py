"""Saved artifact loading, angle interpolation and display-free rendering."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from environment import load_config
from replay import Renderer, export_plot, load_replay, play


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.data = {
            "driver": "rule_based_baseline", "config": load_config(),
            "metrics": {"elapsed_time": 1.0, "termination_reason": "time_limit",
                        "fitness": 0.001, "completed": False},
            "trajectory": [
                {"elapsed_time": 0.0, "x": 0, "y": -40, "heading": math.pi - 0.1,
                 "speed": 0, "steering_angle": 0, "maximum_validated_progress": 0},
                {"elapsed_time": 1.0, "x": 1, "y": -40, "heading": -math.pi + 0.1,
                 "speed": 2, "steering_angle": 0, "maximum_validated_progress": 1},
            ],
        }

    def load(self, data=None):
        path = self.root / "episode.json"
        path.write_text(json.dumps(self.data if data is None else data))
        return load_replay(path)

    def test_baseline_and_champion_config_loading(self):
        self.assertEqual(self.load().driver, "Rule-based baseline")
        data = deepcopy(self.data)
        config = data.pop("config")
        data["driver"] = "neat"
        (self.root / "config.json").write_text(json.dumps(config))
        (self.root / "champion_episode.json").write_text(json.dumps(data))
        self.assertEqual(load_replay(self.root).driver, "NEAT champion")

    def test_time_and_heading_interpolation(self):
        replay = self.load()
        halfway = replay.sample(0.5)
        self.assertAlmostEqual(halfway["x"], 0.5)
        self.assertAlmostEqual(halfway["speed"], 1)
        self.assertAlmostEqual(abs(halfway["heading"]), math.pi)
        self.assertEqual(replay.sample(-1), replay.trajectory[0])
        self.assertEqual(replay.sample(100), replay.trajectory[-1])

    def test_malformed_trajectories_rejected(self):
        for kind in ("empty", "nan", "timestamps", "duration", "speed"):
            data = deepcopy(self.data)
            if kind == "empty":
                data["trajectory"] = []
            elif kind == "nan":
                data["trajectory"][0]["x"] = math.nan
            elif kind == "timestamps":
                data["trajectory"][1]["elapsed_time"] = 0
            elif kind == "duration":
                data["metrics"]["elapsed_time"] = 3
            else:
                data["trajectory"][1]["speed"] = -1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.load(data)

    def test_png_export_and_headless_animation(self):
        replay = self.load()
        renderer = Renderer(replay)
        renderer.draw(0.5)
        output = self.root / "trajectory.png"
        export_plot(replay, output)
        self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        with self.assertRaises(FileExistsError):
            export_plot(replay, output)
        play(replay, headless=True, frames=3)
        with self.assertRaises(ValueError):
            play(replay, speed=0, headless=True, frames=1)


if __name__ == "__main__":
    unittest.main()
