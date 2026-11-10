"""Inspect training JSON records without running physics or loading pickle files."""

import argparse
import configparser
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


def read_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def discover(paths):
    found = []
    for path in paths:
        path = path.resolve()
        if (path / 'evaluations.json').is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(p for p in sorted(path.iterdir()) if p.is_dir() and (p / 'evaluations.json').is_file())
        else:
            raise ValueError(f'Not a run or output directory: {path}')
    found = list(dict.fromkeys(found))
    if not found:
        raise ValueError('No training runs with evaluations.json found.')
    return found


def load_run(path):
    evaluations = read_json(path / 'evaluations.json')
    if not isinstance(evaluations, list):
        raise ValueError(f'{path}: evaluations.json must contain a list.')
    seen = set()
    for row in evaluations:
        if not isinstance(row, dict):
            raise ValueError(f'{path}: invalid evaluation row.')
        for key in ('generation', 'genome_id'):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f'{path}: {key} must be a nonnegative integer.')
        score = row.get('fitness')
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError(f'{path}: fitness must be finite.')
        if 'completed' in row and type(row['completed']) is not bool:
            raise ValueError(f'{path}: completed must be a boolean.')
        key = (row['generation'], row['genome_id'])
        if key in seen:
            raise ValueError(f'{path}: duplicate evaluation {key}.')
        seen.add(key)
    evaluations.sort(key=lambda r: r['generation'])
    config = read_json(path / 'config.json', {})
    meta = {r['generation']: r for r in read_json(path / 'generation_metrics.json', [])}
    neat_path = path / 'neat.cfg'
    parser = configparser.ConfigParser()
    if neat_path.exists():
        parser.read(neat_path)
    expected = parser.getint('NEAT', 'pop_size', fallback=None)
    generations = []
    cumulative = 0
    best_so_far = None
    for g in sorted({r['generation'] for r in evaluations}):
        members = [r for r in evaluations if r['generation'] == g]
        scores = [r['fitness'] for r in members]
        quartiles = statistics.quantiles(scores, n=4, method='inclusive') if len(scores) > 1 else [scores[0]]*3
        success_known = all('completed' in r for r in members)
        successes = sum(r.get('completed', False) for r in members) if success_known else None
        cumulative += len(members)
        best_so_far = max(scores + ([best_so_far] if best_so_far is not None else []))
        row = dict(generation=g, evaluated=len(members), cumulative_evaluations=cumulative,
                   complete=meta.get(g, {}).get('complete', len(members) == expected if expected else None),
                   best_fitness=max(scores), best_so_far=best_so_far, mean_fitness=statistics.mean(scores),
                   median_fitness=statistics.median(scores), fitness_std=statistics.pstdev(scores),
                   fitness_q1=quartiles[0], fitness_q3=quartiles[2], success_count=successes,
                   success_rate=successes/len(members) if success_known else None)
        for field in ('hidden_nodes', 'enabled_connections'):
            values = [r.get(field) for r in members]
            row['mean_'+field] = statistics.mean(values) if all(type(v) in (int, float) for v in values) else None
        generations.append(row)
    # Seeds and recording choices do not define the task; NEAT and software versions do.
    signature = {'task': {k: v for k, v in config.items() if k not in ('seed', 'training', 'replay')},
                 'neat': {s: dict(parser[s]) for s in parser.sections()},
                 'versions': read_json(path / 'versions.json', {})}
    cohort = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
    return dict(path=path, config=config, evaluations=evaluations, generations=generations,
                species=read_json(path / 'species_history.json', []), cohort=cohort,
                provenance_complete=bool(config and neat_path.exists() and (path/'versions.json').exists()))


def nearest(run, generation=None, limit=10):
    rows = run['evaluations']
    if not rows:
        return []
    g = max(r['generation'] for r in rows) if generation is None else generation
    members = [r for r in rows if r['generation'] == g]
    if not members:
        raise ValueError(f"{run['path']}: generation {g} has no evaluations.")
    best = max(members, key=lambda r: r['fitness'])
    # Distances were recorded to the generation best; never substitute a different reference.
    reference = min(members, key=lambda r: r.get('distance_to_generation_best', math.inf))
    if reference.get('distance_to_generation_best') != 0 or reference['fitness'] != best['fitness']:
        return []
    return sorted((r for r in members if r['genome_id'] != reference['genome_id']
                   and r.get('distance_to_generation_best') is not None),
                  key=lambda r: (r['distance_to_generation_best'], r['genome_id']))[:limit]


def pyplot():
    cache = Path(tempfile.gettempdir()) / 'prototype-analysis-mpl'
    cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_run(run, folder, generation):
    plt = pyplot()
    rows = run['generations']
    if not rows:
        return
    x = [r['generation'] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ax = axes[0, 0]
    for field in ('best_fitness', 'mean_fitness', 'median_fitness'):
        ax.plot(x, [r[field] for r in rows], label=field.replace('_fitness', ''))
    ax.fill_between(x, [r['fitness_q1'] for r in rows], [r['fitness_q3'] for r in rows], alpha=.2, label='Middle 50%')
    ax.set(title='Fitness distribution', ylabel='Fitness'); ax.legend()
    ax = axes[0, 1]
    if all(r['success_rate'] is not None for r in rows):
        ax.plot(x, [100*r['success_rate'] for r in rows], marker='o')
    else:
        ax.text(.1, .5, 'Completion fields missing', transform=ax.transAxes)
    ax.set(title='Target-lap completion', ylabel='% of evaluated genomes', ylim=(-2, 102))
    ax = axes[0, 2]
    for field in ('mean_hidden_nodes', 'mean_enabled_connections'):
        if all(r[field] is not None for r in rows):
            ax.plot(x, [r[field] for r in rows], label=field.replace('mean_', ''))
    ax.set(title='Stored topology counts', ylabel='Mean gene count')
    if ax.lines: ax.legend()
    else: ax.text(.1, .5, 'Topology fields missing', transform=ax.transAxes)
    ax = axes[1, 0]
    for sid in sorted({r['species_id'] for r in run['species']}):
        values = {r['generation']: r['population_size'] for r in run['species'] if r['species_id'] == sid}
        ax.plot(x, [values.get(g, 0) for g in x], label=str(sid))
    ax.set(title='Species before reproduction', ylabel='Assigned genomes')
    if not run['species']: ax.text(.1, .5, 'Species history missing', transform=ax.transAxes)
    elif len({r['species_id'] for r in run['species']}) <= 10: ax.legend(title='Species')
    g = x[-1] if generation is None else generation
    members = [r for r in run['evaluations'] if r['generation'] == g]
    ax = axes[1, 1]
    points = [r for r in members if r.get('distance_to_generation_best') is not None]
    if points:
        ax.scatter([r['distance_to_generation_best'] for r in points], [r['fitness'] for r in points], s=18)
    else: ax.text(.1, .5, 'Genetic distance missing', transform=ax.transAxes)
    ax.set(title=f'Genetic similarity, generation {g}', xlabel='Distance to generation best', ylabel='Fitness')
    ax = axes[1, 2]
    points = [r for r in members if r.get('enabled_connections') is not None]
    if points: ax.scatter([r['enabled_connections'] for r in points], [r['fitness'] for r in points], s=18)
    else: ax.text(.1, .5, 'Topology fields missing', transform=ax.transAxes)
    ax.set(title=f'Complexity and fitness, generation {g}', xlabel='Enabled connection count', ylabel='Fitness')
    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]):
        ax.set_xlabel('Generation (0 = initial population)')
        for row in rows:
            if row['complete'] is False: ax.axvline(row['generation'], color='red', linestyle=':', alpha=.5)
    fig.suptitle(run['path'].name + '\nRed dotted lines mark partial generations; absent historical fields stay blank')
    fig.tight_layout(); fig.savefig(folder/'population.png', dpi=180); plt.close(fig)


def study(paths, output, generation=None):
    runs = [load_run(p) for p in discover(paths)]
    output = output.resolve()
    if any(output == r['path'] or r['path'] in output.parents for r in runs):
        raise ValueError('Choose an analysis output outside the training folders.')
    # Validate generation selection before creating any report files.
    neighbours = [nearest(run, generation) for run in runs]
    output.mkdir(parents=True, exist_ok=False)
    summary = []
    for i, (run, neighbours_rows) in enumerate(zip(runs, neighbours), 1):
        folder = output / f'{i:02d}-{run["path"].name}'
        folder.mkdir()
        write_csv(folder/'evaluations.csv', run['evaluations'])
        write_csv(folder/'generations.csv', run['generations'])
        if run['species']: write_csv(folder/'species.csv', run['species'])
        if neighbours_rows: write_csv(folder/'nearest_to_best.csv', neighbours_rows)
        plot_run(run, folder, generation)
        completed = [r for r in run['evaluations'] if r.get('completed')]
        times = [r['mean_lap_time'] for r in completed if r.get('mean_lap_time') is not None]
        summary.append(dict(run=str(run['path']), cohort=run['cohort'], seed=run['config'].get('seed'),
                            provenance_complete=run['provenance_complete'], evaluations=len(run['evaluations']),
                            completed_generations=sum(r['complete'] is True for r in run['generations']),
                            best_fitness=max((r['fitness'] for r in run['evaluations']), default=None),
                            successful_evaluations=len(completed) if all("completed" in r for r in run["evaluations"]) else None, fastest_mean_lap_s=min(times) if times else None,
                            final_success_rate=run['generations'][-1]['success_rate'] if run['generations'] else None))
    write_csv(output/'runs.csv', summary)
    plt = pyplot()
    cohorts = sorted({run['cohort'] for run in runs})
    fig, axes = plt.subplots(len(cohorts), 1, figsize=(12, 5*len(cohorts)), squeeze=False)
    for ax, cohort in zip(axes.flat, cohorts):
        for i, run in enumerate(runs, 1):
            if run['cohort'] != cohort or not run['generations']:
                continue
            ax.plot([r['cumulative_evaluations'] for r in run['generations']],
                    [r['best_so_far'] for r in run['generations']], label=f'{i}: seed {run["config"].get("seed", "?")}')
        ax.set(xlabel='Completed genome evaluations', ylabel='Best fitness so far', title=f'Cohort {cohort}: compare at a common evaluation budget')
        if ax.lines: ax.legend(fontsize=8)
        else: ax.text(.1, .5, 'No finished evaluations', transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(output/'comparison.png', dpi=180); plt.close(fig)
    lines = ['# Output study', '', 'Training inputs were read only. No pickle files were loaded.', '',
             '## Runs', '', '| Run | Cohort | Seed | Evaluations | Best fitness |', '| --- | --- | --- | --- | --- |']
    for i, row in enumerate(summary, 1):
        lines.append(f'| {i} | {row["cohort"]} | {row["seed"]} | {row["evaluations"]} | {row["best_fitness"]} |')
    lines += ['', '## Interpretation', '',
              '- Cohorts fingerprint simulation/task settings, NEAT settings and recorded package versions. Seed, training limits and recording settings are excluded. Matching cohorts are a prerequisite, not proof of identical code.',
              '- Curves use evaluation counts rather than generations. Compare at a common completed evaluation budget; final results can have different budgets.',
              '- No runs are pooled. Repeated seeds and repeated elite evaluations are not independent experimental replicates. Several distinct seeds are needed for uncertainty estimates.',
              '- Completion means all target laps. Partial-generation rates use only evaluated genomes. Red dotted lines mark partial generations; missing fields remain blank, never zero-filled.',
              '- The similarity table uses the selected generation (last available by default), excludes its reference best and lists up to ten nearest genomes. Genetic distance is not driving-line similarity.',
              '- Complexity counts include unused genes. Counts do not reconstruct every topology; only stored generation-best topology is available in the source JSON.',
              '- Missing provenance prevents a reliable cross-run comparison. Check provenance_complete in runs.csv. Successful evaluation counts include repeated elites and do not count unique successful individuals.',
              '- Fastest lap summaries use mean lap time from completed episodes; they are best-case values, not estimates of typical performance.', '',
              '## Sources', '']
    lines += [f'- {r["path"]}' for r in runs]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+', help='Run folders or a parent containing runs')
    parser.add_argument('--output', type=Path, default=None, help='New report folder, outside training runs')
    parser.add_argument('--generation', type=int, help='Generation for similarity and complexity scatter plots')
    args = parser.parse_args()
    output = args.output or Path(__file__).parent/'reports'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')
    try:
        print(study(args.runs, output, args.generation))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Analysis failed: {exc}\n')


if __name__ == '__main__':
    main()
