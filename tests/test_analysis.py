"""Check derived statistics, old formats, and read-only report generation."""

import json
from pathlib import Path
import tempfile
import unittest

from analysis.study import discover, load_run, nearest, study


class AnalysisTests(unittest.TestCase):
    def make_run(self, root, name='run', legacy=False):
        run = root/name
        run.mkdir()
        rows = []
        for key, fitness in enumerate([1., 2., 4.]):
            row = dict(generation=0, genome_id=key, fitness=fitness, completed=key==2,
                       mean_lap_time=30 if key==2 else None)
            if not legacy:
                row.update(hidden_nodes=key, enabled_connections=48+key,
                           distance_to_generation_best=2-key, species_id=1)
            rows.append(row)
        (run/'evaluations.json').write_text(json.dumps(rows))
        (run/'generation_metrics.json').write_text(json.dumps([dict(generation=0, complete=False)]))
        (run/'config.json').write_text(json.dumps(dict(seed=42, fitness={'speed_bonus_weight': .5})))
        (run/'neat.cfg').write_text('[NEAT]\npop_size = 4\n')
        (run/'versions.json').write_text('{"python":"test"}')
        return run

    def test_statistics_legacy_and_cohorts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            a=load_run(self.make_run(root))
            self.assertAlmostEqual(a['generations'][0]['success_rate'], 1/3)
            self.assertEqual(a['generations'][0]['median_fitness'], 2)
            self.assertFalse(a['generations'][0]['complete'])
            self.assertEqual([r['genome_id'] for r in nearest(a)], [1, 0])
            b=load_run(self.make_run(root, 'old', legacy=True))
            self.assertIsNone(b['generations'][0]['mean_hidden_nodes'])
            self.assertEqual(nearest(b), [])
            self.assertEqual(a['cohort'], b['cohort'])
            config=b['path']/'config.json'
            config.write_text('{"seed":7,"fitness":{"speed_bonus_weight":0.5}}')
            self.assertEqual(a['cohort'], load_run(b['path'])['cohort'])
            config.write_text('{"seed":7,"fitness":{"speed_bonus_weight":0}}')
            self.assertNotEqual(a['cohort'], load_run(b['path'])['cohort'])

    def test_reports_preserve_sources_and_handle_missing_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            run=self.make_run(root, legacy=True)
            before={p.name:p.read_bytes() for p in run.iterdir()}
            self.assertEqual(discover([run,root]), [run.resolve()])
            out=study([run], root/'report')
            self.assertTrue((out/'comparison.png').is_file())
            self.assertTrue(next(out.glob('01-*/population.png')).is_file())
            self.assertIn('cohort', (out/'runs.csv').read_text())
            self.assertEqual(before, {p.name:p.read_bytes() for p in run.iterdir()})
            with self.assertRaises(FileExistsError): study([run],out)
            with self.assertRaises(ValueError): study([run],run/'report')
            with self.assertRaises(ValueError): study([run],root/'missing-generation',99)
            self.assertFalse((root/'missing-generation').exists())

    def test_invalid_duplicate_and_empty_evaluations(self):
        with tempfile.TemporaryDirectory() as folder:
            run=self.make_run(Path(folder))
            path=run/'evaluations.json'
            rows=json.loads(path.read_text())
            path.write_text(json.dumps(rows+[rows[0]]))
            with self.assertRaises(ValueError): load_run(run)
            path.write_text('[]')
            self.assertEqual(load_run(run)['generations'], [])
