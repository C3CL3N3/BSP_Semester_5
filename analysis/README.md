# Studying training outputs

I use this folder to inspect the population records after training. The script reads JSON and settings, exports CSV tables and draws figures. It does not run the simulator, load pickle genomes or modify the training folders.

Run these commands from the project root, with the same virtual environment used for training. No additional packages are needed.

## One run

```sh
python analysis/study.py outputs/<run-folder>
```

Reports go into a new timestamped folder under `analysis/reports/`. Select a particular generation for the similarity and complexity scatter plots with:

```sh
python analysis/study.py outputs/<run-folder> --generation 10
```

Generation 0 is the initial population. By default, the scatter plots use the last recorded generation, which may be partial. Choose a complete generation for a full-population comparison.

## Several runs

```sh
python analysis/study.py outputs/<run-a> outputs/<run-b> --output analysis/reports/my-comparison
```

Or inspect every direct child run in a parent directory:

```sh
python analysis/study.py outputs
```

Discovery is one level deep. It does not treat champion snapshot folders as separate experiments. Repeated input paths are deduplicated. The output folder must be new and outside the selected training folders.

## Generated files

| File | What I use it for |
| --- | --- |
| `REPORT.md` | Run index, comparison rules and limits of the measurements |
| `runs.csv` | One row per run: seed, evaluation budget, best fitness, completion counts and fastest completed mean lap time |
| `comparison.png` | Best-so-far fitness against completed evaluations, with each run shown separately |
| `<run>/evaluations.csv` | Individual genome results, including the fields present in the source JSON |
| `<run>/generations.csv` | Recomputed fitness summaries, completion rates and mean topology counts |
| `<run>/species.csv` | Recorded species history, when available |
| `<run>/nearest_to_best.csv` | Up to ten nearest genomes to the selected generation's reference best, excluding that best |
| `<run>/population.png` | Six panels: fitness spread, completion, complexity, species sizes, genetic distance versus fitness and connection count versus fitness |

CSV files open in a spreadsheet or can be loaded into Python/R. Nested JSON fields are written as JSON strings inside CSV cells. Empty cells indicate unavailable values. The original JSON remains the source record.

## Where the original recordings live

The CSVs are analysis exports. The original measurements are in the training folder, usually:

```text
outputs/<label>-<track>-<model>-s<seed>-<UTC timestamp>/
```

For example, a run may be named `grip-study-technical-dynamic-s42-20261110-114708-928281Z`. The label is optional and comes from `train.py --name`. `--output` can instead choose an exact folder name.

The older October recordings are backed up locally in `.archive/cleanup-20261027/previous-results-and-docs.zip`, under its `outputs/` directory. They were not deleted. Extract a run into a separate folder before analysing it. Older recorders saved fewer population fields, so not every plot will be available.

A report folder is separate:

```text
analysis/reports/<UTC timestamp>/
    REPORT.md
    runs.csv
    comparison.png
    01-<training-folder-name>/
        evaluations.csv
        generations.csv
        species.csv             # only when species history exists
        nearest_to_best.csv     # only when suitable distance records exist
        population.png
```

The `01-` prefix links a run to its number in `REPORT.md` and comparison plot. It does not change its original training name.

## Training files: what is saved and why

All paths in this table are relative to one training folder.

| File | What it records | Why I keep it |
| --- | --- | --- |
| `config.json` | Seed, car parameters/model, embedded track points, simulation timing, observation scales, fitness weights, training limits and replay settings | Check exactly which task the population faced and reconstruct the environment |
| `neat.cfg` | Population size, initial network, activation, species, reproduction and mutation settings | Explain how evolution was configured and reload the network |
| `versions.json` | Python, neat-python, NumPy, Shapely, Matplotlib and pygame-ce versions | Identify software differences between experiments |
| `evaluations.json` | One result per finished genome evaluation, including an episode that terminated unsuccessfully | Study individuals, completion, fitness components, complexity and genetic similarity |
| `generation_metrics.json` | One summary per generation with at least one finished evaluation, including a partial generation | Follow the population over time and retain the best topology |
| `species_history.json` | One row per species per recorded generation | Follow species size, survival and performance before reproduction |
| `summary.json` | Stop reason, time spent, evaluation/generation counts, seed, champion and baseline metrics | Get a quick overview without opening every evaluation |
| `champion.pkl` | The final all-time best genome, including its genes | Reconstruct the neural driver; this is not a saved population |
| `champion_episode.json` | Final champion's driver label, terminal metrics and sampled trajectory | Replay and inspect its behaviour without retraining |
| `population.png` | Four training plots: fitness spread, target-lap success, mean topology counts and species sizes | Quickly check whether the population improves, rather than only its champion |
| `baseline_episode.json` | Optional baseline recording, settings, controller gains, versions, creation time and metrics | Replay the conservative reference controller when needed |
| `generation_champions/<snapshot>/` | Selected genome, episode, settings and one `trajectory.png` | Inspect intermediate champions and reconstruct a selected one independently |

The baseline metrics stay in `summary.json` even when its trajectory is not saved. Enable the recording with `--save-baseline-trajectory` or `training.save_baseline_trajectory: true`. This baseline is a driveability check, not an optimised racing controller.

With `render_every_generations: "auto"`, snapshots record every strict improvement in all-time fitness. With a positive integer, they record the best of each selected generation. Zero disables snapshots. Each snapshot contains `champion.pkl`, `champion_episode.json`, `config.json`, `neat.cfg` and `trajectory.png`. Its episode also records `generation`, `genome_id` and embedded `config`.

**Generation numbering differs between records and snapshots.** Evaluation, generation and species tables use 0 for the initial population. Snapshot folders and their episode `generation` field start at 1. So `generation-0001` corresponds to generation 0 in the tables.

Population JSON files are refreshed after each generation with finished evaluations, including a partial generation. Final champion files, summary and plots are written during finalisation. A running or interrupted run may not have all of them. If no genome finishes an evaluation, there is no champion or population plot.

Only load pickle files from a trusted source. `study.py` does not read them; it uses the JSON records and settings.

## Individual measurements

Find these in `outputs/<run>/evaluations.json` and the report's `<run>/evaluations.csv`. The CSV copies all fields present in the source rows. The pair `generation` and `genome_id` identifies one evaluation; an elite can retain its genome ID across generations.

### Identity, fitness and outcome

| Field | Meaning | What I use it for |
| --- | --- | --- |
| `generation`, `genome_id` | Zero-based generation and NEAT genome identifier | Join records and follow retained elites |
| `seed` | Recorded episode seed; physics currently has no random effects | Link the episode to the experiment; NEAT uses the run seed for evolution |
| `fitness` | Total selection score, dimensionless | Compare genomes under the same objective |
| `progress_fraction` | Maximum validated progress divided by the full target distance | Separate progress from the other rewards |
| `speed_bonus`, `time_bonus` | Contributions to fitness | Check whether score improvements come from speed or completed-lap time |
| `completed` | True only when all target laps finish validly | Define success independently of the fitness value |
| `target_laps`, `laps_completed` | Required laps and number finished | Distinguish partial milestones from full success |
| `termination_reason` | `completed`, `off_track`, `invalid_state`, `invalid_action`, `invalid_progress`, `wrong_way`, `stalled` or `time_limit` | Understand why individuals fail |
| `maximum_validated_progress` | Furthest credited distance along ordered gates, in metres | Measure partial performance without rewarding repeated sections |
| `gates_passed` | Number of ordered forward gate crossings | Check progress validation |

With the current settings, fitness is `progress_fraction + 2*completed + time_bonus + speed_bonus`. The completion bonus is implicit; it is not a separate recorded column. The full formulas and weights are in the main README and saved `config.json`. Fitness from different objectives is not directly comparable.

### Time and speed

| Field | Meaning | What I use it for |
| --- | --- | --- |
| `elapsed_time` | Simulated seconds until episode termination, including failures | Measure episode duration; this is not laptop execution time |
| `total_time` | Full episode time only when the target laps finish; otherwise null | Compare completed race times |
| `lap_times` | Simulated duration of each completed lap, in seconds | Inspect consistency and standing-start effects |
| `mean_lap_time` | Mean of the laps completed so far; null if none finished | Inspect partial runs carefully; not necessarily a fully completed episode |
| `lap_time` | Compatibility field: mean lap time only for full completion | Read older results; it does not mean the final individual lap |
| `min_speed_after_first_lap_m_s`, `max_speed_after_first_lap_m_s` | Physics-rate speed extrema after lap one; null until the relevant samples exist | Exclude standing-start acceleration from the extrema |
| `physics_steps`, `decisions` | Environment update count and controller decision count | Check timing and the evaluation workload |

The first lap includes acceleration from rest. Lap times are stored in seconds as floating-point values. To display milliseconds, multiply by 1000; this does not change the simulation resolution. A short unsuccessful episode can have a low `elapsed_time`, so it must not be interpreted as a fast lap.

### Structure and similarity

| Field | Meaning | What I use it for |
| --- | --- | --- |
| `hidden_nodes` | Stored node count excluding output nodes | Follow structural growth |
| `enabled_connections` | Enabled connection genes | Compare connection count with fitness |
| `total_connections` | All connection genes, including disabled ones | Distinguish stored genetic material from enabled edges |
| `species_id` | Species assignment before reproduction | Group individuals within one run |
| `distance_to_generation_best` | NEAT compatibility distance to that generation's evaluated best | Compare genetically similar individuals with different fitness |

Counts can include genes disconnected from outputs. They are not counts of only the effective network. Distance is dimensionless and depends on the saved NEAT settings, including gene differences. It is not racing-line distance. The reference is the current generation's best, not necessarily the final champion. Species IDs have no shared meaning across separate runs.

## Generation measurements

The source is `generation_metrics.json`. The analysis recomputes a smaller table in `generations.csv` from individual evaluations; it does not copy every source field. This distinction matters when looking for topology or failure counts.

| Field | Where to find it | Meaning and use |
| --- | --- | --- |
| `generation`, `evaluated`, `complete` | JSON and CSV | Generation index, finished evaluation count and whether the whole population was evaluated |
| `population_size` | JSON | Assigned population size, including genomes not evaluated before a budget expiry |
| `best_genome_id` | JSON | Identifies this generation's evaluated winner |
| `best_fitness`, `mean_fitness`, `median_fitness` | JSON and CSV | Show whether improvement reaches the population centre or only the best individual |
| `fitness_std` | JSON and CSV | Population standard deviation of the evaluated scores, not a standard error or confidence interval |
| `fitness_q1`, `fitness_q3` | JSON and CSV | Inclusive 25th and 75th percentiles; these bound the shaded middle 50% in the fitness plots |
| `min_fitness` | JSON | Worst evaluated score in the generation |
| `success_count`, `success_rate` | JSON and CSV | Full-completion count and count divided by finished evaluations; rate is a fraction from 0 to 1 |
| `one_lap_count` | JSON | Number completing at least one lap, even if they did not finish the target |
| `mean_hidden_nodes`, `mean_enabled_connections` | JSON and CSV | Mean stored topology counts across evaluated genomes |
| `termination_counts` | JSON | Mapping from termination reason to count, for studying failure modes |
| `best_topology.nodes` | JSON | Stored non-input node IDs of the generation best, including outputs |
| `best_topology.enabled_edges` | JSON | Enabled `[source, destination]` pairs of that best genome; input IDs can occur as sources |
| `cumulative_evaluations` | Analysis CSV only | Finished evaluation count up to the generation, used as the comparison plot's horizontal axis |
| `best_so_far` | Analysis CSV only | Highest score seen through that generation, including a partial generation |

For one evaluated genome, the standard deviation is 0 and both quartiles equal its fitness. A partial generation still describes only the evaluated subset, which may not represent the whole population. An unfinished episode receives no evaluation row and is not counted as a failure. Missing completion status in an older run leaves its rate blank.

The stored best topology contains connectivity, not weights or biases. Full numeric genes are available only for saved champion pickles. `generations.csv` does not contain the topology graph; look in the original JSON.

## Species measurements

Find these in `species_history.json` and, when available, `<run>/species.csv`. There is one row per assigned species per recorded generation, before reproduction.

| Field | Meaning and use |
| --- | --- |
| `generation`, `species_id` | Identify the species within that run and generation |
| `age` | Current generation minus the species creation generation; starts at 0 |
| `population_size` | All members assigned to the species, including unevaluated members in partial generations |
| `evaluated` | Members with a finished evaluation in this generation |
| `complete` | Whether the generation as a whole was fully evaluated; not whether an individual completed the track |
| `best_fitness`, `mean_fitness` | Performance of evaluated members; null if no member was evaluated |
| `success_count` | Evaluated members that completed all target laps |

Species success rate is not stored separately. Derive it as `success_count / evaluated` when `evaluated > 0`; leave it undefined otherwise. These are raw fitness summaries, not NEAT's adjusted reproductive fitness. Representatives, parent choices and full ancestry are not recorded.

## Champion, baseline and trajectory measurements

`summary.json` records `stop_reason`, `training_wall_seconds`, `evaluated_genomes`, `completed_generations` and `seed`. Stop reasons can include `generation_limit`, `wall_clock_budget` or `fitness_threshold`, although threshold stopping is disabled in the current NEAT settings. Training wall time includes work inside the training loop, including recording overhead, but excludes final baseline evaluation, plotting and reload verification.

When a champion exists, `summary.champion` contains the episode metrics above plus `genome_id`, `hidden_nodes`, `enabled_connections` and `reload_verified`. The last field confirms that reloading reproduced its metrics. `summary.baseline` contains the conservative driver's episode metrics. `summary.initial_population` copies the first recorded generation summary for a quick reference. There is no baseline summary if no champion was available for finalisation.

A champion episode has `driver: "neat"`, `metrics` and a `trajectory` list. The root episode uses the adjacent `config.json`. Snapshot episodes also embed their config and identify the snapshot generation/genome. Baseline episodes use `driver: "rule_based_baseline"` and additionally record `created_at_utc`, `driver_settings`, embedded `config` and Python/Shapely `versions`.

The baseline's `driver_settings` records `target_speed_m_s` (12), `heading_gain` (1.5), `offset_gain_rad_per_m` (0.15), `lookahead_gain` (0.08) and `speed_gain` (0.5). These explain its chosen speed and steering response; they are controller gains rather than evolved network parameters.

Each trajectory sample contains:

| Fields | Meaning and use |
| --- | --- |
| `elapsed_time` | Sample time in simulated seconds, for playback and time-based plots |
| `x`, `y` | CG world position in metres, for the driven line |
| `heading` | Vehicle heading in radians, for orientation and animation |
| `speed` | Total speed in m/s, for speed-coloured trajectories |
| `steering_angle` | Actual wheel steering angle in radians, after steering-rate limits |
| `lateral_speed`, `yaw_rate` | Dynamic body lateral velocity in m/s and yaw rate in rad/s; legacy kinematic states do not integrate these fields |
| `maximum_validated_progress` | Credited path distance at the sample, in metres |
| `action.steering`, `action.throttle`, `action.brake` | Requested normalised commands: steering in [-1, 1], throttle/brake in [0, 1] |

Trajectories include the reset pose with `action: null`, then one sample after each controller decision, including the terminal pose. Each later action is the command applied over the preceding interval. They are not full 0.001 s or 0.01 s physics logs; a terminal sample may arrive earlier than the usual 0.05 s decision interval. Physics-rate extrema can therefore differ from extrema calculated from sampled trajectories.

## Analysis run summary and plots

The report-level `runs.csv` has these columns:

| Field | Meaning and use |
| --- | --- |
| `run` | Absolute source folder, so the figures can be traced back to their recordings |
| `cohort` | Short settings/version fingerprint, used to separate comparison groups |
| `seed` | Seed from the saved config, for checking distinct experimental repeats |
| `provenance_complete` | Whether config, NEAT settings and version files are present; not proof that all metadata or source code matches |
| `evaluations` | Total finished genome evaluations |
| `completed_generations` | Generations marked fully evaluated; not the number of successful driving episodes |
| `best_fitness` | Highest evaluated score anywhere in the run |
| `successful_evaluations` | Number of full-completion evaluation rows; repeated elite evaluations count again |
| `fastest_mean_lap_s` | Lowest mean lap time among episodes that completed the full target; blank if none |
| `final_success_rate` | Completion fraction in the last recorded generation, which may be partial |

`nearest_to_best.csv` contains the ordinary evaluation columns for up to ten nearest neighbours, sorted by genetic distance then genome ID. It excludes the reference best. The scatter plot includes all available distances, including the best at distance 0, rather than only those ten neighbours. Neither output measures similarity between driven trajectories.

The training `population.png` has four panels. The analysis `<run>/population.png` adds genetic-distance-versus-fitness and connection-count-versus-fitness panels for the selected generation, making six. `comparison.png` shows best-so-far fitness against cumulative completed evaluations in separate panels for different settings groups. It does not average seeds or automatically truncate them to a shared budget.

## What is deliberately not recorded

I do not save every genome's full trajectory or numeric genes, observations at every decision, per-substep tyre forces, full parentage, or population checkpoints. That keeps recording size manageable. Selected champions can be re-evaluated from their genomes, but the current files cannot resume the whole evolutionary run or reconstruct every past individual. The run files also do not automatically record the Git commit or hardware specifications.

Duplicate champion PNG galleries and separate fitness PNG/SVG files are no longer produced by the current recorder. Older runs may still have them. Analysis reports can be regenerated from the source JSON; keep the source training folders rather than only their CSV exports.

## Questions to ask

1. Does the completion rate rise, or does only the champion improve?
2. Does the middle of the fitness distribution improve too?
3. Does network size grow alongside fitness? Extra genes alone do not establish that complexity caused the improvement.
4. Do several species remain, or does one dominate? Species IDs are local to a run.
5. Are genetically close individuals similarly fit? The distance is NEAT's compatibility distance to that generation's best, not physical distance between racing lines.
6. Do distinct seeds show similar trends at the same evaluation budget?

Fitness summaries use all evaluated genomes. Success means completing all target laps. A partial generation's denominator includes only its finished evaluations. Its unevaluated genomes are not counted as failures. Red dotted lines mark partial generations in the population plots.

The nearest-genome table needs the distance fields added by the new recorder. Older runs can still be studied for fitness and completion, but missing topology, species or distance fields cannot be recovered from final champion files. Counts include disconnected genes; they describe the stored genome rather than only the effective network. Full generation-best topology remains in the original `generation_metrics.json`.

## Fair comparisons

The report gives each run a cohort fingerprint based on its environment/task settings, NEAT settings and recorded software versions. Training budgets, recording options and seeds are excluded. Different fingerprints mean the runs should not be pooled as repeats of one experiment. Matching fingerprints do not guarantee matching source code; also check the code version used for each run. Missing settings or version files are flagged in `runs.csv`.

Use distinct seeds and compare at a common evaluation budget. Final scores from runs with different budgets are not an equal-budget comparison. Repeated elites and adjacent generations are not independent replicates. The script deliberately does not compute significance tests or confidence intervals from them.

A best lap time is a best-case observation, not typical driving performance. Fitness also combines several rewards, so inspect completion and lap time separately. These tools analyse recorded training; they do not yet perform parameter sweeps or evaluate champions on unseen tracks.

## Watch a saved driver on another track

From the project root:

```sh
python transfer.py outputs/training-20261018T173254651701Z --track tracks/monza.json --live
```

This loads the trusted saved genome and its own NEAT configuration. It preserves the source car, observation scales, fitness and simulation settings, then replaces only the track. No evolution or retraining happens. Optional `--max-seconds` changes the episode limit explicitly. `--speed 2` requests twice real-time simulation, subject to CPU performance.

Space pauses and Escape stops. The final frame stays visible until the window is closed. Without `--live`, the same episode runs without a display. Closing early records `user_stopped: true`; this is an interrupted test, not an environment termination or completed lap.

After the window closes, the result is saved in a new `outputs/transfer-.../champion_episode.json`, including source run, genome ID, embedded target configuration, metrics and trajectory. It can be viewed again with `python replay.py outputs/<transfer-folder>`. Transfer folders are episode records, not training populations, so `study.py` does not treat them as training runs.
