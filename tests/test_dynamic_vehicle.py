"""Force limits, transient response, low-speed handling and integration checks."""

from dataclasses import replace
import math
import unittest

from vehicle import Action, VehicleConfig, VehicleState, advance, body_velocities, tyre_forces


class DynamicVehicleTests(unittest.TestCase):
    def setUp(self):
        self.cfg = VehicleConfig(model="dynamic")

    def test_slip_signs_static_axle_loads_and_friction_circles(self):
        state = VehicleState(speed=20, steering_angle=0.1, lateral_speed=1, yaw_rate=0.1)
        for action in (Action(1, 1, 0), Action(1, 0, 1)):
            f = tyre_forces(state, action, self.cfg)
            vx, vy, yaw = body_velocities(state, self.cfg)
            self.assertAlmostEqual(f.front_slip_angle, 0.1 - math.atan2(vy + 1.55 * yaw, vx))
            self.assertAlmostEqual(f.rear_slip_angle, -math.atan2(vy - 1.85 * yaw, vx))
            self.assertAlmostEqual(f.front_normal_load + f.rear_normal_load, 760 * 9.81)
            for fx, fy, fz in ((f.front_longitudinal_force, f.front_lateral_force, f.front_normal_load),
                               (f.rear_longitudinal_force, f.rear_lateral_force, f.rear_normal_load)):
                self.assertLessEqual(math.hypot(fx, fy), self.cfg.friction_coefficient * fz + 1e-9)

    def test_lateral_saturation_reduces_drive_and_braking(self):
        straight = VehicleState(speed=20)
        corner = VehicleState(speed=20, steering_angle=0.35, lateral_speed=5)
        for action in (Action(0, 1, 0), Action(0, 0, 1)):
            a = tyre_forces(straight, action, self.cfg)
            b = tyre_forces(corner, action, self.cfg)
            self.assertLess(abs(b.front_longitudinal_force) + abs(b.rear_longitudinal_force),
                            abs(a.front_longitudinal_force) + abs(a.rear_longitudinal_force))
            self.assertLessEqual(abs(b.rear_lateral_force), b.rear_normal_load + 1e-9)

    def test_steering_is_a_transient_response_and_turns_are_symmetric(self):
        start = VehicleState(speed=15)
        left, right = start, start
        for _ in range(50):
            left = advance(left, Action(0.5, 0, 0), self.cfg, 0.01)
            right = advance(right, Action(-0.5, 0, 0), self.cfg, 0.01)
        self.assertGreater(left.y, 0)
        self.assertGreater(left.yaw_rate, 0)
        self.assertAlmostEqual(left.x, right.x)
        self.assertAlmostEqual(left.y, -right.y)
        self.assertAlmostEqual(left.yaw_rate, -right.yaw_rate)
        # Merely changing wheel angle doesn't instantly change velocity/yaw.
        self.assertEqual(body_velocities(VehicleState(speed=15, steering_angle=0.2), self.cfg), (15, 0, 0))

    def test_straight_acceleration_braking_and_stationary_steering(self):
        state = advance(VehicleState(speed=10), Action(0, 1, 0), self.cfg, 0.01)
        self.assertAlmostEqual(state.speed, 10.04)
        state = advance(VehicleState(speed=10), Action(0, 1, 1), self.cfg, 0.01)
        self.assertAlmostEqual(state.speed, 9.92)
        stopped = VehicleState(steering_angle=0.35)
        for _ in range(500):
            stopped = advance(stopped, Action(1, 0, 1), self.cfg, 0.01)
        self.assertEqual(stopped, VehicleState(steering_angle=0.35))
        state = VehicleState(speed=1)
        for _ in range(100):
            state = advance(state, Action(0, 0, 1), self.cfg, 0.01)
        self.assertEqual(state.speed, 0)

    def test_force_opposes_existing_lateral_motion(self):
        forces = tyre_forces(VehicleState(speed=10, lateral_speed=1), Action(0, 0, 0), self.cfg)
        self.assertLess(forces.front_lateral_force, 0)
        self.assertLess(forces.rear_lateral_force, 0)

    def test_substep_convergence_and_determinism(self):
        def rollout(cfg):
            state = VehicleState(speed=20)
            for _ in range(100):
                state = advance(state, Action(0.3, 0.5, 0), cfg, 0.01)
            return state
        a = rollout(self.cfg)
        self.assertEqual(a, rollout(self.cfg))
        b = rollout(replace(self.cfg, integration_dt_s=0.0005))
        self.assertLess(math.hypot(a.x - b.x, a.y - b.y), 0.03)
        self.assertLess(abs(a.heading - b.heading), 0.005)
        self.assertLessEqual(a.speed, 30)

    def test_invalid_dynamic_parameters_rejected(self):
        for values in ({"friction_coefficient": 0}, {"yaw_inertia_kg_m2": -1},
                       {"front_cornering_stiffness_n_rad": math.nan}, {"model": "unknown"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(self.cfg, **values)
