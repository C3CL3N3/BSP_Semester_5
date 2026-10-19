"""Quadratic aero, dissipative drag and downforce-dependent tyre grip."""

from dataclasses import replace
import math
import unittest

from vehicle import Action, VehicleConfig, VehicleState, aerodynamic_forces, advance, tyre_forces


class AerodynamicsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = VehicleConfig(model="dynamic", max_speed_m_s=80,
                                 drag_area_m2=1.1, downforce_area_m2=3.0)

    def test_quadratic_scaling_and_zero_at_rest(self):
        self.assertEqual(aerodynamic_forces(0, self.cfg), (0, 0))
        drag20, down20 = aerodynamic_forces(20, self.cfg)
        drag40, down40 = aerodynamic_forces(40, self.cfg)
        self.assertAlmostEqual(drag40, 4 * drag20)
        self.assertAlmostEqual(down40, 4 * down20)
        self.assertAlmostEqual(drag20, 0.5 * 1.225 * 1.1 * 20 ** 2)

    def test_drag_slows_coasting_and_cannot_move_a_stopped_car(self):
        start = VehicleState(speed=40)
        drag = advance(start, Action(0, 0, 0), self.cfg, 0.01)
        none = advance(start, Action(0, 0, 0), replace(self.cfg, drag_area_m2=0), 0.01)
        self.assertLess(drag.speed, none.speed)
        self.assertEqual(none.speed, 40)
        self.assertEqual(advance(VehicleState(), Action(0, 0, 0), self.cfg, 0.01), VehicleState())

    def test_downforce_adds_load_and_grip_without_propulsion(self):
        state = VehicleState(speed=40, steering_angle=0.35)
        aero = tyre_forces(state, Action(1, 0, 0), self.cfg)
        no_aero = tyre_forces(state, Action(1, 0, 0), replace(self.cfg, downforce_area_m2=0))
        _, down = aerodynamic_forces(40, self.cfg)
        self.assertAlmostEqual(aero.front_normal_load + aero.rear_normal_load,
                               self.cfg.mass_kg * self.cfg.gravity_m_s2 + down)
        self.assertGreater(abs(aero.front_lateral_force), abs(no_aero.front_lateral_force))
        self.assertAlmostEqual(aero.front_normal_load - no_aero.front_normal_load,
                               down * self.cfg.front_downforce_fraction)
        for fx, fy, fz in ((aero.front_longitudinal_force, aero.front_lateral_force, aero.front_normal_load),
                            (aero.rear_longitudinal_force, aero.rear_lateral_force, aero.rear_normal_load)):
            self.assertLessEqual(math.hypot(fx, fy), fz * self.cfg.friction_coefficient + 1e-9)
        # Downforce by itself changes neither speed nor heading on a straight.
        result = advance(VehicleState(speed=20), Action(0, 0, 0), replace(self.cfg, drag_area_m2=0), 0.01)
        self.assertAlmostEqual(result.speed, 20)
        self.assertEqual(result.heading, 0)

    def test_invalid_parameters_and_zero_coefficients(self):
        for values in ({"drag_area_m2": -1}, {"downforce_area_m2": -1},
                       {"front_downforce_fraction": 1.1}, {"air_density_kg_m3": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(self.cfg, **values)
        self.assertEqual(aerodynamic_forces(40, replace(self.cfg, drag_area_m2=0, downforce_area_m2=0)), (0, 0))
