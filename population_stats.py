"""Summaries of evaluated genomes before NEAT breeds the next population."""

from collections import Counter
import statistics


def topology(genome, neat_config):
    return {
        "hidden_nodes": len(set(genome.nodes) - set(neat_config.genome_config.output_keys)),
        "enabled_connections": sum(c.enabled for c in genome.connections.values()),
        "total_connections": len(genome.connections),
    }


def summarise(generation, genomes, rows, neat_config, species_set=None):
    """Distances use this generation's evaluated best, even in a partial generation."""
    evaluated = {row["genome_id"]: row for row in rows}
    best = max((g for key, g in genomes if key in evaluated), key=lambda g: g.fitness)
    for key, genome in genomes:
        if key in evaluated:
            evaluated[key].update(topology(genome, neat_config))
            evaluated[key]["distance_to_generation_best"] = genome.distance(best, neat_config.genome_config)
            evaluated[key]["species_id"] = (species_set.genome_to_species.get(key)
                                             if species_set is not None else None)
    fitness = [r["fitness"] for r in rows]
    q1, _, q3 = statistics.quantiles(fitness, n=4, method="inclusive") if len(fitness) > 1 else [fitness[0]] * 3
    successes = sum(r.get("completed", False) for r in rows)
    summary = {
        "generation": generation, "population_size": len(genomes), "evaluated": len(rows),
        "complete": len(rows) == len(genomes), "best_genome_id": best.key,
        "best_fitness": max(fitness), "mean_fitness": statistics.mean(fitness),
        "median_fitness": statistics.median(fitness), "min_fitness": min(fitness),
        "fitness_std": statistics.pstdev(fitness), "fitness_q1": q1, "fitness_q3": q3,
        "success_count": successes, "success_rate": successes / len(rows),
        "one_lap_count": sum(r.get("laps_completed", 0) >= 1 for r in rows),
        "mean_hidden_nodes": statistics.mean(r["hidden_nodes"] for r in rows),
        "mean_enabled_connections": statistics.mean(r["enabled_connections"] for r in rows),
        "termination_counts": dict(Counter(r.get("termination_reason", "unknown") for r in rows)),
        "best_topology": {"nodes": sorted(best.nodes),
                          "enabled_edges": [list(k) for k, c in sorted(best.connections.items()) if c.enabled]},
    }
    species_rows = []
    if species_set is not None:
        for sid, species in sorted(species_set.species.items()):
            members = [evaluated[k] for k in species.members if k in evaluated]
            values = [r["fitness"] for r in members]
            species_rows.append({
                "generation": generation, "species_id": sid,
                "age": generation - species.created, "population_size": len(species.members),
                "evaluated": len(members), "complete": summary["complete"],
                "best_fitness": max(values) if values else None,
                "mean_fitness": statistics.mean(values) if values else None,
                "success_count": sum(r.get("completed", False) for r in members),
            })
    return summary, species_rows
