"""Geometry and progress checks using the actual configured oval."""

import json
import math
from pathlib import Path
import unittest

from shapely.geometry import Point

from track import ProgressTracker, Track
from vehicle import VehicleConfig, VehicleState


class TrackTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())["track"]
        self.track = Track(self.config)
        self.vehicle = VehicleConfig()

    def pose(self, s):
        x, y, heading = self.track.path_pose(s)
        return VehicleState(x=x, y=y, heading=heading)

    def test_length_start_and_wrapping(self):
        self.assertAlmostEqual(self.track.length, 200 + 80 * math.pi)
        self.assertEqual(self.track.start_state(), VehicleState(y=-40))
        self.assertEqual(self.track.path_pose(0), self.track.path_pose(self.track.length))
        self.assertAlmostEqual(self.track.path_heading(30), 0)
        self.assertTrue(self.track.contains_vehicle(self.track.start_state(), self.vehicle))
        a = self.track.start_state()
        a.x = 200
        self.assertEqual(self.track.start_state().x, 0)

    def test_projection_on_all_segments_and_signed_offsets(self):
        for i in range(500):
            s = (i + 0.25) * self.track.length / 500
            x, y, h = self.track.path_pose(s)
            for offset in (-3, 0, 3):
                p = self.track.project(x - offset * math.sin(h), y + offset * math.cos(h))
                with self.subTest(s=s, offset=offset):
                    self.assertAlmostEqual(p.s, s, places=8)
                    self.assertAlmostEqual(p.offset, offset, places=8)
                    self.assertAlmostEqual(math.sin(p.heading - h), 0, places=8)

    def test_full_rectangle_collision_including_rotation(self):
        state = VehicleState(y=-45.5)
        self.assertTrue(self.track.road.covers(Point(state.x, state.y)))
        self.assertFalse(self.track.contains_vehicle(state, self.vehicle))
        state = VehicleState(y=-44, heading=math.pi / 4)
        self.assertFalse(self.track.contains_vehicle(state, self.vehicle))
        self.assertFalse(self.track.contains_vehicle(VehicleState(), self.vehicle))

    def test_boundary_contact_policy(self):
        touching = VehicleState(y=-46 + self.vehicle.body_width_m / 2)
        self.assertTrue(self.track.contains_vehicle(touching, self.vehicle))
        strict = Track({**self.config, "boundary_contact_allowed": False})
        self.assertFalse(strict.contains_vehicle(touching, self.vehicle))

    def test_rays_order_range_and_rotated_heading(self):
        state = self.track.start_state()
        distances = self.track.ray_distances(state, [-90, 0, 90], 150)
        self.assertAlmostEqual(distances[0], 6)
        self.assertAlmostEqual(distances[2], 6)
        self.assertGreater(distances[1], 50)
        self.assertEqual(self.track.ray_distances(state, [0], 2), (2,))
        state.heading = math.pi / 2
        self.assertAlmostEqual(self.track.ray_distances(state, [0], 150)[0], 6)
        state.y = -46
        self.assertEqual(self.track.ray_distances(state, [90], 150), (0,))

    def test_curve_rays(self):
        state = self.pose(50 + math.pi * 40 / 2)
        right, left = self.track.ray_distances(state, [-90, 90], 150)
        self.assertAlmostEqual(right, 6, places=7)
        self.assertAlmostEqual(left, 6, places=7)

    def test_forward_finite_gate_crossing(self):
        self.assertTrue(self.track.crosses_gate(self.pose(-0.1), self.pose(0.1), 0))
        self.assertFalse(self.track.crosses_gate(self.pose(0.1), self.pose(-0.1), 0))
        self.assertFalse(self.track.crosses_gate(self.pose(0), self.pose(0), 0))
        self.assertFalse(self.track.crosses_gate(VehicleState(x=-1, y=-60), VehicleState(x=1, y=-60), 0))

    def test_valid_lap_with_ordered_gates(self):
        tracker = ProgressTracker(self.track)
        count = math.ceil(self.track.length / 0.2)
        for i in range(1, count + 2):
            tracker.update(self.pose(i * self.track.length / count), 0.25)
        self.assertTrue(tracker.completed)
        self.assertEqual(tracker.gates_passed, len(self.track.checkpoints))
        self.assertEqual(tracker.maximum_progress, self.track.length)

    def test_stationary_and_start_line_oscillation(self):
        tracker = ProgressTracker(self.track)
        for _ in range(10):
            tracker.update(self.pose(0), 0.25)
        self.assertEqual(tracker.maximum_progress, 0)
        for _ in range(10):
            for s in (-0.1, 0.1):
                tracker.update(self.pose(s), 0.25)
        self.assertAlmostEqual(tracker.maximum_progress, 0.1)
        self.assertEqual(tracker.gates_passed, 0)
        self.assertFalse(tracker.completed)

    def test_teleport_and_skipped_gate_are_rejected(self):
        tracker = ProgressTracker(self.track)
        with self.assertRaises(ValueError):
            tracker.update(self.pose(2 * self.track.checkpoint_spacing), 0.3)
        self.assertEqual(tracker.maximum_progress, 0)
        with self.assertRaises(ValueError):
            tracker.update(self.pose(1), self.track.checkpoint_spacing)
        self.assertEqual(tracker.gates_passed, 0)

    def test_reverse_lap_cannot_arm_forward_completion(self):
        tracker = ProgressTracker(self.track)
        count = math.ceil(self.track.length / 0.2)
        for i in range(1, count + 1):
            tracker.update(self.pose(-i * self.track.length / count), 0.25)
        for i in range(1, count + 1):
            tracker.update(self.pose(i * self.track.length / count), 0.25)
        self.assertEqual(tracker.gates_passed, 0)
        self.assertFalse(tracker.completed)
        self.assertAlmostEqual(tracker.maximum_progress, 0)

    def test_trackers_are_independent_and_repeated_gates_do_not_count(self):
        tracker, other = ProgressTracker(self.track), ProgressTracker(self.track)
        s = 0
        target = self.track.checkpoint_spacing + 0.1
        while s < target:
            s = min(target, s + 0.1)
            tracker.update(self.pose(s), 0.25)
        self.assertEqual(tracker.gates_passed, 1)
        for _ in range(5):
            tracker.update(self.pose(target - 0.2), 0.25)
            tracker.update(self.pose(target), 0.25)
        self.assertEqual(tracker.gates_passed, 1)
        self.assertEqual(other.maximum_progress, 0)
        self.assertEqual(other.gates_passed, 0)

    def test_invalid_configuration_and_queries(self):
        for values in ({"shape": "unknown"}, {"width_m": 80},
                       {"turn_radius_m": -1}, {"checkpoint_count": 3},
                       {"checkpoint_count": True}, {"width_m": math.nan}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Track({**self.config, **values})
        with self.assertRaises(ValueError):
            self.track.project(math.nan, 0)
        with self.assertRaises(ValueError):
            self.track.ray_distances(self.pose(0), [math.inf], 150)
        with self.assertRaises(ValueError):
            self.track.ray_distances(self.pose(0), [0], -1)


if __name__ == "__main__":
    unittest.main()
