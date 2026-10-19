"""Episode contract, termination ordering and deterministic integration checks."""

import math
import unittest
from unittest.mock import patch

from environment import Environment, ROOT, load_config as _load_config
from vehicle import Action, VehicleState, advance, body_velocities


def load_config():
    """Keep oval-specific regression checks independent of the active circuit."""
    return _load_config(ROOT / "configs" / "oval.json")


class EnvironmentTests(unittest.TestCase):
    def make_env(self, **simulation):
        config = load_config()
        config["simulation"].update(simulation)
        env = Environment(config)
        env.reset()
        return env

    def test_reset_and_observation_order(self):
        env = self.make_env()
        obs = env.observations()
        self.assertEqual(len(obs), 16)
        self.assertEqual(obs[:5], (0, 0, 0, 0, 0))
        self.assertAlmostEqual(obs[5], 6 / 150)
        self.assertAlmostEqual(obs[13], 6 / 150)
        self.assertEqual(obs[14], 0)
        self.assertAlmostEqual(obs[15], (50 / 40) / math.pi)
        self.assertTrue(all(math.isfinite(x) and -1 <= x <= 1 for x in obs))
        self.assertEqual(env.seed, 42)
        self.assertIsNone(env.metrics()["min_speed_after_first_lap_m_s"])
        self.assertIsNone(env.metrics()["max_speed_after_first_lap_m_s"])

    def test_five_updates_match_vehicle(self):
        env = self.make_env()
        state = env.track.start_state()
        for _ in range(5):
            state = advance(state, Action(0.2, 1, 0), env.vehicle_config, 0.01)
        obs, done, metrics = env.step(Action(0.2, 1, 0))
        self.assertEqual(env.state, state)
        self.assertFalse(done)
        self.assertEqual(metrics["physics_steps"], 5)
        self.assertEqual(metrics["elapsed_time"], 0.05)
        vx, vy, yaw = body_velocities(state, env.vehicle_config)
        self.assertAlmostEqual(obs[0], vx / 30)
        self.assertAlmostEqual(obs[1], vy / 10)
        self.assertAlmostEqual(obs[2], yaw / 4)
        self.assertAlmostEqual(obs[3], math.sin(state.heading))
        self.assertAlmostEqual(obs[4], (state.y + 40) / 6)
        self.assertIsNone(metrics["lap_time"])

    def test_full_spawn_stall_grace(self):
        env = self.make_env()
        for _ in range(99):
            _, done, _ = env.step(Action(0, 0, 0))
            self.assertFalse(done)
        _, done, metrics = env.step(Action(0, 0, 0))
        self.assertTrue(done)
        self.assertEqual(metrics["termination_reason"], "stalled")
        self.assertEqual(metrics["elapsed_time"], 5)
        self.assertEqual(metrics["fitness"], 0)
        self.assertIsNone(metrics["lap_time"])

    def test_timeout_stops_mid_decision(self):
        env = self.make_env(max_episode_s=0.03)
        obs, done, metrics = env.step(Action(0, 1, 0))
        self.assertTrue(done)
        self.assertEqual(metrics["termination_reason"], "time_limit")
        self.assertEqual(metrics["physics_steps"], 3)
        self.assertEqual(len(obs), 16)

    def test_invalid_action_and_state_return_finite_observation(self):
        env = self.make_env()
        initial = env.observations()
        obs, done, metrics = env.step(Action(math.nan, 1, 0))
        self.assertEqual(obs, initial)
        self.assertTrue(done)
        self.assertEqual(metrics["termination_reason"], "invalid_action")
        self.assertEqual(metrics["physics_steps"], 0)
        env.reset()
        env.state.x = math.inf
        obs, done, metrics = env.step(Action(0, 1, 0))
        self.assertTrue(done)
        self.assertEqual(metrics["termination_reason"], "invalid_state")
        self.assertTrue(all(math.isfinite(x) for x in obs))

    def test_off_track_checked_before_progress(self):
        env = self.make_env()
        # Centre is on road but the rear/front corner is outside.
        env.state.y = -45.5
        _, done, metrics = env.step(Action(0, 0, 0))
        self.assertTrue(done)
        self.assertEqual(metrics["termination_reason"], "off_track")
        self.assertEqual(metrics["physics_steps"], 1)
        self.assertEqual(metrics["fitness"], 0)

    def test_unreachable_progress_terminates(self):
        env = self.make_env()
        env.state.x = 10
        _, done, metrics = env.step(Action(0, 0, 0))
        self.assertTrue(done)
        self.assertEqual(metrics["termination_reason"], "invalid_progress")
        self.assertEqual(metrics["fitness"], 0)

    def test_wrong_way_sustained_timer_and_recovery(self):
        env = self.make_env(wrong_way_timeout_s=0.1)
        env.state.heading, env.state.speed = math.pi, 2
        self.assertFalse(env.step(Action(0, 0, 0))[1])
        env.state.heading = 0
        self.assertFalse(env.step(Action(0, 0, 0))[1])
        env.state.heading = math.pi
        self.assertFalse(env.step(Action(0, 0, 0))[1])
        _, done, metrics = env.step(Action(0, 0, 0))
        self.assertTrue(done)
        self.assertEqual(metrics["termination_reason"], "wrong_way")

    def test_config_is_copied_and_episodes_are_independent(self):
        config = load_config()
        a, b = Environment(config), Environment(config)
        config["simulation"]["stall_timeout_s"] = 0.01
        initial = a.reset(7)
        b.reset(7)
        a.step(Action(0, 1, 0))
        self.assertEqual(b.state.speed, 0)
        self.assertEqual(a.reset(7), initial)
        self.assertEqual(a.metrics()["physics_steps"], 0)
        self.assertEqual(a.metrics()["fitness"], 0)

    def test_same_actions_reproduce_trajectory(self):
        a, b = self.make_env(), self.make_env()
        for _ in range(20):
            self.assertEqual(a.step(Action(0.1, 0.5, 0)), b.step(Action(0.1, 0.5, 0)))
            self.assertEqual(a.state, b.state)

    def test_action_bounds_and_episode_lifecycle(self):
        env = Environment(load_config())
        with self.assertRaises(RuntimeError):
            env.step(Action(0, 0, 0))
        env.reset()
        with self.assertRaises(TypeError):
            env.step([0.5, 1, 0])
        env.step(Action(100, 100, -100))
        self.assertGreater(env.state.speed, 0)
        self.assertLessEqual(env.state.speed, 0.2)
        env.step(Action(math.nan, 0, 0))
        with self.assertRaises(RuntimeError):
            env.step(Action(0, 0, 0))
        env.reset()
        self.assertFalse(env.done)

    def test_valid_completion_has_bonus_and_lap_time(self):
        env = self.make_env()
        # Synthetic on-centreline steps exercise real gates/containment/episode
        # completion, independently of a driver. This is not a baseline lap.
        count = math.ceil(env.track.length / 0.2)
        samples = iter(range(1, count * env.target_laps + 2))
        def next_pose(*args):
            x, y, h = env.track.path_pose(next(samples) * env.track.length / count)
            return VehicleState(x, y, h, 20, 0)
        with patch("environment.advance", side_effect=next_pose):
            while not env.done:
                _, _, metrics = env.step(Action(0, 1, 0))
                if 0 < metrics["laps_completed"] < 3:
                    self.assertFalse(metrics["completed"])
                    self.assertEqual(metrics["time_bonus"], 0)
                    self.assertLess(metrics["fitness"], 1 + env.speed_bonus_weight)
        self.assertEqual(metrics["termination_reason"], "completed")
        self.assertTrue(metrics["completed"])
        self.assertGreater(metrics["fitness"], 3)
        self.assertEqual(metrics["progress_fraction"], 1)
        self.assertEqual(metrics["laps_completed"], 3)
        self.assertEqual(len(metrics["lap_times"]), 3)
        self.assertAlmostEqual(metrics["lap_time"], metrics["elapsed_time"] / 3)
        self.assertEqual(metrics["gates_passed"], 96)
        self.assertEqual(metrics["min_speed_after_first_lap_m_s"], 20)
        self.assertEqual(metrics["max_speed_after_first_lap_m_s"], 20)

    def test_faster_completed_runs_score_higher(self):
        scores = []
        for distance_per_step in (0.1, 0.2):
            env = self.make_env()
            count = math.ceil(env.track.length / distance_per_step)
            samples = iter(range(1, count * env.target_laps + 2))
            def next_pose(*args):
                x, y, h = env.track.path_pose(next(samples) * env.track.length / count)
                return VehicleState(x, y, h, distance_per_step / env.dt, 0)
            with patch("environment.advance", side_effect=next_pose):
                while not env.done:
                    env.step(Action(0, 1, 0))
            self.assertTrue(env.metrics()["completed"])
            scores.append(env.metrics()["fitness"])
        self.assertGreater(scores[1], scores[0])

    def test_old_config_keeps_single_lap_objective(self):
        config = load_config()
        config["simulation"].pop("target_laps")
        config.pop("fitness")
        env = Environment(config)
        env.reset()
        self.assertEqual(env.target_laps, 1)
        self.assertEqual(env.time_bonus_weight, 0)
        self.assertEqual(env.speed_bonus_weight, 0)

    def test_speed_bonus_rewards_faster_new_progress(self):
        results = []
        for speed, decisions in ((5, 10), (10, 5)):
            config = load_config()
            config["vehicle"]["model"] = "kinematic"
            env = Environment(config)
            env.reset()
            env.state.speed = speed
            for _ in range(decisions):
                env.step(Action(0, 0, 0))
            results.append(env.metrics())
        self.assertAlmostEqual(results[0]["maximum_validated_progress"], results[1]["maximum_validated_progress"])
        self.assertAlmostEqual(results[1]["speed_bonus"], results[0]["speed_bonus"] * 2)
        self.assertGreater(results[1]["fitness"], results[0]["fitness"])

    def test_speed_bonus_cannot_be_farmed_by_repeating_a_section(self):
        config = load_config()
        config["vehicle"]["model"] = "kinematic"
        env = Environment(config)
        env.reset()
        env.state.speed = 10
        for _ in range(10):
            env.step(Action(0, 0, 0))
        original = env.metrics()["speed_bonus"]
        for heading in (math.pi, 0):
            env.state.heading = heading
            for _ in range(10):
                env.step(Action(0, 0, 0))
        self.assertFalse(env.done)
        self.assertAlmostEqual(env.metrics()["speed_bonus"], original)
        env.reset()
        self.assertEqual(env.metrics()["speed_bonus"], 0)

    def test_settings_are_validated(self):
        for changes in ({"physics_dt_s": 0}, {"controller_dt_s": 0.015},
                        {"max_episode_s": 0.015}, {"wrong_way_velocity_m_s": 1},
                        {"stall_timeout_s": math.nan}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.make_env(**changes)
        config = load_config()
        config["observations"]["ray_angles_deg"] = [0] * 9
        with self.assertRaises(ValueError):
            Environment(config)


if __name__ == "__main__":
    unittest.main()
