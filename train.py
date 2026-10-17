"""Bounded headless NEAT training and reproducible champion reconstruction."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import pickle
import platform
import random
import statistics
import time

import neat

from environment import Environment, ROOT, load_config
from vehicle import Action


class BudgetExpired(Exception):
    """The wall-clock training budget expired; discard any unfinished episode."""


def network_action(outputs) -> Action:
    """Map three finite sigmoid outputs to simulator commands."""
    if len(outputs) != 3:
        raise ValueError("Expected three sigmoid outputs.")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in outputs):
        raise ValueError("Sigmoid outputs must be finite and in [0, 1].")
    return Action(2.0 * outputs[0] - 1.0, outputs[1], outputs[2])


def load_neat_config(path):
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation, str(path))


def run_genome(genome, neat_config, config, *, record=False, deadline=None):
    """Evaluate an independent episode under fixed settings; optional trajectory."""
    env = Environment(config)
    observations = env.reset()
    network = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    trajectory = []
    if record:
        trajectory.append({"elapsed_time": 0.0, **asdict(env.state),
                           "maximum_validated_progress": 0.0, "action": None})
    while not env.done:
        if deadline is not None and time.monotonic() >= deadline:
            raise BudgetExpired
        action = network_action(network.activate(observations))
        observations, _, metrics = env.step(action)
        if record:
            trajectory.append({"elapsed_time": env.elapsed_time, **asdict(env.state),
                               "maximum_validated_progress": metrics["maximum_validated_progress"],
                               "action": asdict(action)})
    return {"driver": "neat", "metrics": env.metrics(), "trajectory": trajectory}


class GenomeEvaluator:
    def __init__(self, config, deadline=None, output_dir=None):
        self.config = deepcopy(config)
        self.deadline = deadline
        self.champion = None
        self.champion_metrics = None
        self.generations = []
        self.evaluations = []
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.render_every = config["training"].get("render_every_generations", 0)
        if isinstance(self.render_every, bool) or not isinstance(self.render_every, int) or self.render_every < 0:
            raise ValueError("render_every_generations must be a nonnegative integer.")
        self.render_mode = config["training"].get("render_mode", "png")
        if self.render_mode not in ("png", "window"):
            raise ValueError("render_mode must be png or window.")
        self.render_speed = config["training"].get("render_speed", 4.0)
        if not math.isfinite(self.render_speed) or self.render_speed <= 0:
            raise ValueError("render_speed must be positive and finite.")

    def render_generation(self, genome, neat_config, generation):
        """Record/render this generation's best, numbered from 1."""
        if self.output_dir is None or not self.render_every or (generation + 1) % self.render_every:
            return
        try:
            episode = run_genome(genome, neat_config, self.config, record=True, deadline=self.deadline)
        except BudgetExpired:
            return
        episode.update(config=self.config, generation=generation + 1, genome_id=genome.key)
        folder = self.output_dir / "generation_champions" / f"generation-{generation + 1:04d}"
        folder.mkdir(parents=True)
        _json(folder / "champion_episode.json", episode)
        _json(folder / "config.json", self.config)
        (folder / "neat.cfg").write_text((self.output_dir / "neat.cfg").read_text())
        with (folder / "champion.pkl").open("wb") as handle:
            pickle.dump(genome, handle, protocol=pickle.HIGHEST_PROTOCOL)
        from replay import export_plot, load_replay, play
        replay = load_replay(folder / "champion_episode.json")
        export_plot(replay, folder / "trajectory.png")
        # A single gallery makes interval champions easy to browse/compare.
        gallery = self.output_dir / "champion_laps"
        gallery.mkdir(exist_ok=True)
        from shutil import copyfile
        png = gallery / f"generation-{generation + 1:04d}.png"
        copyfile(folder / "trajectory.png", png)
        print(f"Saved generation {generation + 1} champion PNG: {png.resolve()}", flush=True)
        if self.render_mode == "window":
            play(replay, speed=self.render_speed, close_on_finish=True)

    def __call__(self, genomes, neat_config):
        generation = len(self.generations)
        results = []
        interrupted = False
        generation_champion = None
        try:
            for genome_id, genome in genomes:
                result = run_genome(genome, neat_config, self.config, deadline=self.deadline)
                metrics = result["metrics"]
                genome.fitness = metrics["fitness"]
                results.append(genome.fitness)
                if generation_champion is None or genome.fitness > generation_champion.fitness:
                    generation_champion = deepcopy(genome)
                self.evaluations.append({"generation": generation, "genome_id": genome_id, **metrics})
                if self.champion is None or genome.fitness > self.champion.fitness:
                    self.champion = deepcopy(genome)
                    self.champion_metrics = metrics
        except BudgetExpired:
            interrupted = True
            raise
        finally:
            if results:
                self.generations.append({"generation": generation, "evaluated": len(results),
                                         "complete": not interrupted,
                                         "best_fitness": max(results),
                                         "mean_fitness": statistics.mean(results)})
                row = self.generations[-1]
                print(f"Generation {generation}: {len(results)} evaluated, "
                      f"best={row['best_fitness']:.6f}, mean={row['mean_fitness']:.6f}", flush=True)
                if not interrupted:
                    rng = random.getstate()
                    try:
                        self.render_generation(generation_champion, neat_config, generation)
                    finally:
                        random.setstate(rng)


def evaluate_genomes(genomes, neat_config):
    """Standard NEAT callback using the default project settings."""
    GenomeEvaluator(load_config())(genomes, neat_config)


def _json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def reload_champion(run_dir):
    """Reconstruct a locally generated, trusted run and record its episode."""
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.json")
    neat_config = load_neat_config(run_dir / "neat.cfg")
    with (run_dir / "champion.pkl").open("rb") as handle:
        genome = pickle.load(handle)
    return run_genome(genome, neat_config, config, record=True)


def _plot(rows, output):
    # Keep plotting caches in the writable run area, including in sandboxes.
    cache = ROOT / "outputs" / ".plot-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for field, label in (("best_fitness", "Best"), ("mean_fitness", "Mean")):
        ax.plot([r["generation"] for r in rows], [r[field] for r in rows], marker="o", label=label)
    ax.set(xlabel="Generation (0 = initial population)", ylabel="Fitness", title="Measured NEAT fitness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def train(config, neat_path, output_dir, *, generations=None, budget_s=None):
    """Train within a wall-clock budget; save only fully evaluated champions.

    Budget covers population creation/evolution/evaluation, checked at every
    controller decision. Artifact saving, baseline comparison and reload checking
    happen afterwards. A partial generation is recorded and labelled explicitly.
    """
    config = deepcopy(config)
    generations = config["training"]["generations"] if generations is None else generations
    budget_s = config["training"]["wall_clock_budget_s"] if budget_s is None else budget_s
    if isinstance(generations, bool) or not isinstance(generations, int) or generations <= 0:
        raise ValueError("generations must be a positive integer.")
    if isinstance(budget_s, bool) or not math.isfinite(budget_s) or budget_s <= 0:
        raise ValueError("budget_s must be finite and positive.")
    # Validate environment and seed before creating output artifacts.
    Environment(config).reset()
    neat_text = Path(neat_path).read_text()
    neat_config = load_neat_config(neat_path)
    GenomeEvaluator(config)  # Validate rendering settings before creating files.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Training output folder: {output_dir.resolve()}", flush=True)
    config["training"].update(generations=generations, wall_clock_budget_s=budget_s)
    _json(output_dir / "config.json", config)
    (output_dir / "neat.cfg").write_text(neat_text)
    _json(output_dir / "versions.json", {"python": platform.python_version(),
          **{name: version(name) for name in ("neat-python", "numpy", "shapely", "matplotlib", "pygame-ce")}})
    previous_rng = random.getstate()
    start = time.monotonic()
    evaluator = GenomeEvaluator(config, deadline=start + budget_s, output_dir=output_dir)
    stop_reason = "generation_limit"
    try:
        random.seed(config["seed"])
        population = neat.Population(neat_config)
        try:
            population.run(evaluator, generations)
            if not neat_config.no_fitness_termination and evaluator.champion and evaluator.champion.fitness >= neat_config.fitness_threshold:
                stop_reason = "fitness_threshold"
        except BudgetExpired:
            stop_reason = "wall_clock_budget"
    finally:
        random.setstate(previous_rng)
    training_seconds = time.monotonic() - start
    _json(output_dir / "generation_metrics.json", evaluator.generations)
    _json(output_dir / "evaluations.json", evaluator.evaluations)
    summary = {"stop_reason": stop_reason, "training_wall_seconds": training_seconds,
               "evaluated_genomes": len(evaluator.evaluations),
               "completed_generations": sum(r["complete"] for r in evaluator.generations),
               "seed": config["seed"], "champion": None}
    if evaluator.champion is not None:
        with (output_dir / "champion.pkl").open("wb") as handle:
            pickle.dump(evaluator.champion, handle, protocol=pickle.HIGHEST_PROTOCOL)
        replay = reload_champion(output_dir)
        if replay["metrics"] != evaluator.champion_metrics:
            raise RuntimeError("Reloaded champion did not reproduce its evaluation metrics.")
        _json(output_dir / "champion_episode.json", replay)
        from baseline import run_baseline
        baseline = run_baseline(config)
        _json(output_dir / "baseline_episode.json", baseline)
        champion = evaluator.champion
        summary["champion"] = {**replay["metrics"], "genome_id": champion.key,
                               "hidden_nodes": len(set(champion.nodes) - set(neat_config.genome_config.output_keys)),
                               "enabled_connections": sum(c.enabled for c in champion.connections.values()),
                               "reload_verified": True}
        summary["baseline"] = baseline["metrics"]
        summary["initial_population"] = evaluator.generations[0]
    if evaluator.generations:
        _plot(evaluator.generations, output_dir / "fitness.png")
    _json(output_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--neat-config", type=Path, default=ROOT / "neat.cfg")
    parser.add_argument("--generations", type=int)
    parser.add_argument("--budget-seconds", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--render-every", type=int, help="Render every N completed generations; 0 disables")
    parser.add_argument("--render-mode", choices=("png", "window"))
    parser.add_argument("--render-speed", type=float)
    parser.add_argument("--output", type=Path, help="New run folder; default: outputs/training-<timestamp>")
    parser.add_argument("--reload", type=Path, help="Reload a trusted local run instead of training.")
    args = parser.parse_args()
    if args.reload:
        print(json.dumps(reload_champion(args.reload)["metrics"], indent=2))
        return 0
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    for key, value in (("render_every_generations", args.render_every),
                       ("render_mode", args.render_mode), ("render_speed", args.render_speed)):
        if value is not None:
            config["training"][key] = value
    output = args.output or ROOT / config["training"]["output_dir"] / f"training-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    summary = train(config, args.neat_config, output, generations=args.generations, budget_s=args.budget_seconds)
    print(json.dumps(summary, indent=2))
    print(f"Saved run: {output.resolve()}")
    return 0 if summary["champion"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
