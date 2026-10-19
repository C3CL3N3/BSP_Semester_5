"""Optional Monza and active technical geometry, progress and portable assets."""

import math
import tempfile
import unittest
from pathlib import Path
import json

from environment import ROOT, Environment, load_config
from track import Track, ProgressTracker
from vehicle import VehicleState, VehicleConfig


class TrackAssetTests(unittest.TestCase):
    def test_monza_loads_but_is_not_active(self):
        monza = Track(load_config(ROOT / "configs/monza.json")["track"])
        active = Track(load_config()["track"])
        self.assertEqual(monza.config["name"], "monza")
        self.assertNotEqual(active.config["name"], "monza")
        self.assertAlmostEqual(monza.length, 5760.1527586, places=4)
        self.assertTrue(monza.contains_vehicle(monza.start_state(), VehicleConfig()))
        pose = monza.path_pose(100)
        self.assertAlmostEqual(monza.project(*pose[:2]).s, 100)

    def test_technical_path_wrap_projection_rays_and_containment(self):
        track = Track(load_config()["track"])
        self.assertGreater(track.length, 2500)
        self.assertEqual(track.config["name"], "technical_test")
        self.assertTrue(track.road.is_valid)
        self.assertEqual(track.path_pose(0), track.path_pose(track.length))
        self.assertTrue(track.contains_vehicle(track.start_state(), VehicleConfig()))
        for i in range(50):
            s = (i + 0.5) * track.length / 50
            x, y, heading = track.path_pose(s)
            projected = track.project(x, y)
            self.assertAlmostEqual(projected.s, s, places=6)
            self.assertAlmostEqual(projected.offset, 0, places=6)
            state = VehicleState(x, y, heading)
            self.assertTrue(track.contains_vehicle(state, VehicleConfig()))
            self.assertTrue(all(0 <= value <= 150 for value in track.ray_distances(state, [-90, 0, 90], 150)))

    def test_ordered_lap_and_teleport_rejection_on_polyline(self):
        track = Track(load_config()["track"])
        tracker = ProgressTracker(track)
        count = math.ceil(track.length / 0.5)
        for i in range(1, count + 2):
            tracker.update(VehicleState(*track.path_pose(i * track.length / count)), 0.8)
        self.assertTrue(tracker.completed)
        self.assertEqual(tracker.gates_passed, len(track.checkpoints))
        tracker = ProgressTracker(track)
        with self.assertRaises(ValueError):
            tracker.update(VehicleState(*track.path_pose(100)), 0.8)

    def test_saved_geometry_is_self_contained(self):
        env = Environment(load_config())
        self.assertIn("reference_points", env.config["track"])
        self.assertNotIn("path", env.config["track"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(env.config))
            reloaded = Environment(load_config(path))
            self.assertEqual(env.reset(), reloaded.reset())

    def test_invalid_asset_geometry_rejected(self):
        for points in ([[0, 0], [10, 0], [0, 10], [0, 0]],
                       [[0, 0], [100, 100], [0, 100], [100, 0], [0, 0]]):
            with self.assertRaises(ValueError):
                Track({"shape": "polyline", "reference_points": points, "width_m": 12})
