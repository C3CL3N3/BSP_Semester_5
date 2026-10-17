"""Real evolution, artifact reconstruction, reproducibility and budget checks."""

import json
import math
from pathlib import Path
import random
import tempfile
import unittest

import neat

from environment import ROOT, load_config
from train import (BudgetExpired, GenomeEvaluator, load_neat_config,
                   network_action, reload_champion, train)


class TrainingTests(unittest.TestCase):
    def test_output_mapping_rejects_invalid_values(self):
        action = network_action([0.25, 0.7, 0.1])
        self.assertEqual((action.steering, action.throttle, action.brake), (-0.5, 0.7, 0.1))
        for values in ([0, 1], [math.nan, 0, 0], [0, math.inf, 0], [0, 0, -0.1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                network_action(values)

    def test_evolution_reload_and_seed_reproducibility(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = root / "neat.cfg"
            cfg.write_text((ROOT / "neat.cfg").read_text().replace("pop_size = 30", "pop_size = 4"))
            config = load_config()
            config["simulation"]["max_episode_s"] = 0.1
            config["training"]["render_every_generations"] = 2
            config["training"]["render_mode"] = "png"
            rng = random.getstate()
            a = train(config, cfg, root / "a", generations=2, budget_s=30)
            self.assertEqual(random.getstate(), rng)
            b = train(config, cfg, root / "b", generations=2, budget_s=30)
            self.assertEqual(a["evaluated_genomes"], 8)
            self.assertEqual(a["completed_generations"], 2)
            self.assertTrue(a["champion"]["reload_verified"])
            for filename in ("generation_metrics.json", "evaluations.json", "champion_episode.json"):
                self.assertEqual(json.loads((root / "a" / filename).read_text()),
                                 json.loads((root / "b" / filename).read_text()))
            self.assertEqual(reload_champion(root / "a"),
                             json.loads((root / "a" / "champion_episode.json").read_text()))
            self.assertTrue((root / "a" / "fitness.png").is_file())
            snapshot = root / "a" / "generation_champions" / "generation-0002"
            self.assertTrue((snapshot / "trajectory.png").is_file())
            gallery = root / "a" / "champion_laps"
            self.assertEqual([p.name for p in gallery.iterdir()], ["generation-0002.png"])
            self.assertEqual((gallery / "generation-0002.png").read_bytes(),
                             (snapshot / "trajectory.png").read_bytes())
            self.assertTrue((snapshot / "champion.pkl").is_file())
            self.assertFalse((snapshot.parent / "generation-0001").exists())
            from replay import load_replay
            self.assertEqual(load_replay(snapshot).metrics["target_laps"], 3)
            self.assertEqual(reload_champion(snapshot)["metrics"], load_replay(snapshot).metrics)
            self.assertEqual(a["champion"], b["champion"])
            with self.assertRaises(FileExistsError):
                train(config, cfg, root / "a", generations=1, budget_s=30)

    def test_expired_budget_does_not_assign_partial_fitness(self):
        cfg = load_neat_config(ROOT / "neat.cfg")
        genome = next(iter(neat.Population(cfg).population.values()))
        self.assertEqual(len(genome.connections), 48)
        self.assertEqual(len(genome.nodes), 3)
        self.assertTrue(all(node.response == 1 and node.activation == "sigmoid"
                            for node in genome.nodes.values()))
        evaluator = GenomeEvaluator(load_config(), deadline=-1)
        with self.assertRaises(BudgetExpired):
            evaluator([(1, genome)], cfg)
        self.assertIsNone(genome.fitness)
        self.assertIsNone(evaluator.champion)
        self.assertEqual(evaluator.evaluations, [])
        with tempfile.TemporaryDirectory() as folder:
            result = train(load_config(), ROOT / "neat.cfg", Path(folder) / "run",
                           generations=2, budget_s=1e-12)
            self.assertEqual(result["stop_reason"], "wall_clock_budget")
            self.assertEqual(result["evaluated_genomes"], 0)
            self.assertIsNone(result["champion"])


if __name__ == "__main__":
    unittest.main()
