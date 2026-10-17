"""Vehicle behavioural checks, runnable without third-party dependencies."""

import json
import math
from pathlib import Path
import unittest

from vehicle import (Action, VehicleConfig, VehicleState, advance,
                     body_velocities, footprint_corners)


class VehicleTests(unittest.TestCase):
    def setUp(self):
        self.config = VehicleConfig()

    def test_project_config_round_trip(self):
        path = Path(__file__).resolve().parents[1] / "config.json"
        values = json.loads(path.read_text())["vehicle"]
        self.assertEqual(VehicleConfig(**values).to_dict(), values)

    def test_straight_motion_and_input_is_unchanged(self):
        start = VehicleState(speed=10)
        end = advance(start, Action(0, 1, 0), self.config, 0.01)
        self.assertAlmostEqual(end.speed, 10.04)
        self.assertAlmostEqual(end.x, 0.1004)
        self.assertEqual((end.y, end.heading), (0, 0))
        self.assertEqual(start, VehicleState(speed=10))

    def test_left_and_right_turns_are_symmetric(self):
        left = advance(VehicleState(speed=10), Action(1, 0, 0), self.config, 0.01)
        right = advance(VehicleState(speed=10), Action(-1, 0, 0), self.config, 0.01)
        self.assertGreater(left.y, 0)
        self.assertGreater(left.heading, 0)
        self.assertAlmostEqual(left.x, right.x)
        self.assertAlmostEqual(left.y, -right.y)
        self.assertAlmostEqual(left.heading, -right.heading)

    def test_steering_rate_and_angle_limits(self):
        state = advance(VehicleState(), Action(10, 0, 0), self.config, 0.01)
        self.assertAlmostEqual(state.steering_angle, 0.008)
        for _ in range(100):
            state = advance(state, Action(1, 0, 0), self.config, 0.01)
        self.assertEqual(state.steering_angle, 0.35)
        state = advance(state, Action(-1, 0, 0), self.config, 0.01)
        self.assertAlmostEqual(state.steering_angle, 0.342)

    def test_brake_overrides_throttle_and_speed_is_bounded(self):
        state = advance(VehicleState(speed=10), Action(0, 1, 1), self.config, 0.01)
        self.assertAlmostEqual(state.speed, 9.92)
        state = advance(VehicleState(speed=0.01), Action(0, 1, 2), self.config, 0.01)
        self.assertEqual(state.speed, 0)
        state = advance(VehicleState(speed=30), Action(0, 2, -1), self.config, 0.01)
        self.assertEqual(state.speed, 30)
        partial = advance(VehicleState(speed=10), Action(0, 1, 0.5), self.config, 0.01)
        self.assertAlmostEqual(partial.speed, 9.98)

    def test_body_velocities_and_heading_wrap(self):
        self.assertEqual(body_velocities(VehicleState(), self.config), (0, 0, 0))
        vx, vy, yaw = body_velocities(VehicleState(speed=10, steering_angle=0.2), self.config)
        self.assertAlmostEqual(math.hypot(vx, vy), 10)
        self.assertGreater(vy, 0)
        self.assertAlmostEqual(yaw, vx / 3.4 * math.tan(0.2))
        state = advance(VehicleState(heading=math.pi - 0.001, speed=30,
                                     steering_angle=0.35), Action(1, 0, 0), self.config, 0.01)
        self.assertTrue(-math.pi <= state.heading < 0)

    def test_rotated_footprint_is_centred_at_cg(self):
        corners = footprint_corners(VehicleState(x=4, y=5, heading=math.pi / 2), self.config)
        self.assertAlmostEqual(corners[0][0], 4 - 0.95)
        self.assertAlmostEqual(corners[0][1], 5 + 2.6)
        self.assertAlmostEqual(sum(x for x, _ in corners) / 4, 4)
        self.assertAlmostEqual(sum(y for _, y in corners) / 4, 5)

    def test_invalid_inputs_are_rejected(self):
        for values in ({"wheelbase_m": 4}, {"body_width_m": 0},
                       {"max_steering_rad": math.pi / 2},
                       {"mass_kg": math.nan}, {"initial_speed_m_s": 31}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                VehicleConfig(**values)
        for dt in (0, -1, math.nan, math.inf):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                advance(VehicleState(), Action(0, 0, 0), self.config, dt)
        with self.assertRaises(ValueError):
            advance(VehicleState(), Action(math.nan, 0, 0), self.config, 0.01)
        with self.assertRaises(ValueError):
            advance(VehicleState(speed=-1), Action(0, 0, 0), self.config, 0.01)

    def test_action_sequence_is_deterministic(self):
        def rollout():
            state = VehicleState()
            for action in [Action(0, 1, 0)] * 100 + [Action(0.3, 0.4, 0)] * 100:
                state = advance(state, action, self.config, 0.01)
            return state
        self.assertEqual(rollout(), rollout())


if __name__ == "__main__":
    unittest.main()
