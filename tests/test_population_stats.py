"""Population measurements must describe evaluated genomes, not their offspring."""

from datetime import datetime, timezone
import unittest
from unittest.mock import patch

import neat

from environment import ROOT, load_config
from train import BudgetExpired, GenomeEvaluator, load_neat_config, run_name
from population_stats import summarise


class PopulationStatsTests(unittest.TestCase):
    def test_known_population_and_partial_denominator(self):
        cfg = load_neat_config(ROOT / 'neat.cfg')
        population = neat.Population(cfg)
        genomes = list(population.population.items())[:3]
        rows = []
        for (key, genome), score, success in zip(genomes, [1., 3.5], [False, True]):
            genome.fitness = score
            rows.append(dict(genome_id=key, fitness=score, completed=success, laps_completed=3 if success else 0))
        summary, species = summarise(0, genomes, rows, cfg, population.species)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['success_rate'], .5)
        self.assertEqual(summary['median_fitness'], 2.25)
        self.assertEqual(summary['fitness_std'], 1.25)
        self.assertEqual(summary['fitness_q1'], 1.625)
        self.assertEqual(summary['fitness_q3'], 2.875)
        self.assertEqual(rows[1]['distance_to_generation_best'], 0)
        self.assertEqual(rows[0]['hidden_nodes'], 0)
        self.assertEqual(rows[0]['enabled_connections'], 48)
        self.assertEqual(sum(r['evaluated'] for r in species), 2)
        self.assertTrue(all(not r['complete'] for r in species))

    def test_budget_records_only_finished_evaluations(self):
        cfg = load_neat_config(ROOT / 'neat.cfg')
        genomes = list(neat.Population(cfg).population.items())[:3]
        evaluator = GenomeEvaluator(load_config())
        with patch('train.run_genome', side_effect=[{'metrics': {'fitness': .2, 'completed': False}}, BudgetExpired]):
            with self.assertRaises(BudgetExpired):
                evaluator(genomes, cfg)
        self.assertEqual(len(evaluator.evaluations), 1)
        self.assertFalse(evaluator.generations[0]['complete'])
        self.assertEqual(evaluator.generations[0]['evaluated'], 1)
        self.assertEqual(evaluator.generations[0]['population_size'], 3)
        self.assertIsNone(genomes[1][1].fitness)

    def test_readable_names_are_safe_and_seed_specific(self):
        config = load_config()
        stamp = datetime(2026, 11, 10, 14, 32, 5, tzinfo=timezone.utc)
        name = run_name(config, '../Grip study!', stamp)
        self.assertTrue(name.startswith('grip-study-technical-dynamic-s42-'))
        self.assertNotIn('/', name)
        config['seed'] = 7
        self.assertIn('-s7-', run_name(config, now=stamp))
