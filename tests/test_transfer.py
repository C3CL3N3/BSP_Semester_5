"""A transfer changes the track, not the saved driver or its input scales."""

from copy import deepcopy
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import neat

from environment import Environment, ROOT, load_config
from train import load_neat_config, run_genome
from transfer import prepare, drive


class TransferTests(unittest.TestCase):
    def test_saved_driver_and_headless_trajectory(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)
            config=load_config(ROOT/'configs/oval.json')
            config['simulation']['max_episode_s']=.1
            original=deepcopy(config)
            (source/'config.json').write_text(json.dumps(config))
            text=(ROOT/'neat.cfg').read_text()
            (source/'neat.cfg').write_text(text)
            cfg=load_neat_config(source/'neat.cfg')
            genome=next(iter(neat.Population(cfg).population.values()))
            with (source/'champion.pkl').open('wb') as handle: pickle.dump(genome,handle)
            env, network, key=prepare(source, ROOT/'tracks/monza.json')
            self.assertEqual(key,genome.key)
            for field in ('vehicle','observations','fitness','simulation'):
                self.assertEqual(env.config[field],original[field])
            expected=run_genome(genome,cfg,env.config,record=True)
            actual=drive(env,network)
            self.assertEqual(actual['metrics'],expected['metrics'])
            self.assertEqual(actual['trajectory'],expected['trajectory'])
            self.assertFalse(actual['user_stopped'])
            self.assertEqual(json.loads((source/'config.json').read_text()),original)
