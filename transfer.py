"""Drive a saved champion on another track, without training it again."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle

import neat

from environment import Environment, ROOT, load_config
from train import load_neat_config, network_action


def prepare(source, track, max_seconds=None):
    config = load_config(source / 'config.json')
    config['track'] = {'shape': 'file', 'path': str(track.resolve()),
                       'boundary_contact_allowed': config['track'].get('boundary_contact_allowed', True)}
    if max_seconds is not None:
        config['simulation']['max_episode_s'] = max_seconds
    env = Environment(config)
    env.reset()
    neat_config = load_neat_config(source / 'neat.cfg')
    # Only use trusted genomes: pickle can execute code while loading.
    with (source / 'champion.pkl').open('rb') as handle:
        genome = pickle.load(handle)
    network = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    return env, network, genome.key


def drive(env, network, *, live=False, speed=1.0):
    from replay import Replay, Renderer
    trajectory = [{'elapsed_time': 0., **asdict(env.state), 'maximum_validated_progress': 0., 'action': None}]
    replay = Replay('Transferred NEAT champion', env.config, env.metrics(), trajectory)
    stopped = False
    renderer = None
    if live:
        import pygame
        pygame.display.init()
        screen = pygame.display.set_mode((1100, 760))
        pygame.display.set_caption('Champion transfer - live simulation')
        renderer = Renderer(replay)
        clock = pygame.time.Clock()
        paused = False
        accumulator = 0.
    try:
        while not env.done and not stopped:
            if live:
                elapsed = clock.tick(60) / 1000
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        stopped = True
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        paused = not paused
                if not paused and not stopped:
                    accumulator += min(elapsed, .1) * speed
                steps = 0 if paused or stopped else min(20, int(accumulator / env.simulation['controller_dt_s']))
            else:
                steps = 1
            for _ in range(steps):
                action = network_action(network.activate(env.observations()))
                env.step(action)
                trajectory.append({'elapsed_time': env.elapsed_time, **asdict(env.state),
                                   'maximum_validated_progress': env.progress.maximum_progress,
                                   'action': asdict(action)})
                if live:
                    replay.times.append(env.elapsed_time)
                    accumulator -= env.simulation['controller_dt_s']
                if env.done:
                    break
            replay.metrics = env.metrics()
            if live:
                screen.blit(renderer.draw(env.elapsed_time, live=True, paused=paused, playback_speed=speed), (0, 0))
                pygame.display.flip()
        if live and env.done and not stopped:
            print(json.dumps(env.metrics(), indent=2), flush=True)
            print('Episode ended. Close the window or press Escape.', flush=True)
            waiting = True
            while waiting:
                clock.tick(30)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        waiting = False
    finally:
        if live:
            pygame.quit()
    return {'driver': 'neat', 'config': env.config, 'metrics': env.metrics(),
            'trajectory': trajectory, 'user_stopped': stopped and not env.done}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Trusted training folder or champion snapshot')
    parser.add_argument('--track', type=Path, default=ROOT/'tracks/monza.json')
    parser.add_argument('--live', action='store_true', help='Watch physics run in a window')
    parser.add_argument('--speed', type=float, default=1., help='Target live playback rate; limited by available CPU')
    parser.add_argument('--max-seconds', type=float, help='Optional episode time limit; other source settings stay unchanged')
    parser.add_argument('--output', type=Path, help='New transfer-result folder')
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be finite and positive')
    try:
        env, network, key = prepare(args.source, args.track, args.max_seconds)
        output = args.output or ROOT/'outputs'/f'transfer-{args.source.name}-{args.track.stem}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S-%fZ}'
        output.mkdir(parents=True, exist_ok=False)
        print(f'Transfer output: {output.resolve()}', flush=True)
        print(f'Genome {key}, {env.track.length:.2f} m track, {env.target_laps} target laps. No training.', flush=True)
        episode = drive(env, network, live=args.live, speed=args.speed)
        episode.update(source_run=str(args.source.resolve()), genome_id=key)
        (output/'champion_episode.json').write_text(json.dumps(episode, indent=2)+'\n')
        print(json.dumps(episode['metrics'], indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Transfer failed: {exc}\n')


if __name__ == '__main__':
    main()
