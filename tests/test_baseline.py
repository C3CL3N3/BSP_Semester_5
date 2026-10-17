"""A real, unmocked baseline lap is the environment's integration checkpoint."""

import json
import unittest

from baseline import run_baseline
from environment import load_config


class BaselineTests(unittest.TestCase):
    def test_physically_driven_lap_is_valid_and_repeatable(self):
        config = load_config()
        first = run_baseline(config)
        second = run_baseline(config)
        self.assertEqual(first["trajectory"], second["trajectory"])
        self.assertEqual(first["metrics"], second["metrics"])
        metrics = first["metrics"]
        self.assertTrue(metrics["completed"])
        self.assertEqual(metrics["termination_reason"], "completed")
        self.assertEqual(metrics["gates_passed"], config["track"]["checkpoint_count"] * 3)
        self.assertGreater(metrics["fitness"], 3.0)
        self.assertEqual(metrics["laps_completed"], 3)
        self.assertGreater(metrics["lap_time"], 0)
        self.assertLess(metrics["lap_time"], config["simulation"]["max_episode_s"])
        self.assertEqual(first["trajectory"][0]["speed"], 0)
        self.assertEqual(first["trajectory"][-1]["elapsed_time"], metrics["total_time"])
        self.assertTrue(all(0 <= row["speed"] <= config["vehicle"]["max_speed_m_s"]
                            for row in first["trajectory"]))
        # Saved artifacts must be strict JSON and preserve the measured result.
        self.assertEqual(json.loads(json.dumps(first, allow_nan=False))["metrics"], metrics)


if __name__ == "__main__":
    unittest.main()
