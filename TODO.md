# Notes for the next part of the project

Based on my discussion with my supervisor. I want to look at what happens to the population during training, not just show the best car at the end. Then I want to test whether a driver can handle a different circuit.

These notes were organised and improved with help from Codex.

Checked items are already implemented. The others still need work or an experiment. Having the recording code does not mean I have answered the research question yet.

## What is already there

- [x] Save the result of each finished genome evaluation, including failed episodes: fitness, reward components, progress, completion, lap times and reason for stopping.
- [x] Count how many genomes finish all target laps in each generation and calculate the success rate.
- [x] Save best, mean and median fitness, standard deviation and quartiles for each generation.
- [x] Save node/connection counts for every evaluated genome, and the best network's nodes and enabled edges each generation.
- [x] Save species membership, size, age, fitness and successful evaluations before reproduction.
- [x] Measure genetic distance to the generation best and export its nearest neighbours with their fitness.
- [x] Generate CSVs and population plots. Runs with different saved settings or software versions are put in separate comparison groups.
- [x] Mark partial generations and leave missing older data blank.
- [x] Give training folders readable names: experiment label, track, model, seed and time.
- [x] Explain the recordings in `analysis/README.md`.
- [x] Run a saved champion on another circuit without retraining, with a live view if needed.

For now, transfer testing is one champion at a time. I still need to automate repeated runs and compare the results across seeds.

## First: make the analysis easier to use

### Fitness and completion

- [ ] Plot progress, completion, time bonus and speed bonus separately. I need to see why fitness increases.
- [ ] Show how many genomes finish one lap as well as all three. Otherwise early improvement can be hidden behind a 0% success rate.
- [ ] Plot why runs fail: off-track, stalled, wrong way, etc.
- [ ] Keep completed lap times separate from the duration of failed attempts. Stopping quickly is not a good lap time.
- [ ] Make partial generations clear in the CSVs and report text too, not just the plots.

### Best genome and similar genomes

- [ ] Use the last complete generation by default for similarity plots. Still allow a partial one if I select it, and explain what happens if none is complete.
- [ ] Put the reference best in the nearest-neighbour CSV as a labelled row. Add its ID, each neighbour's fitness difference and whether they share a species.
- [ ] Decide how to handle equal best fitness. Save the actual reference ID instead of guessing it from a distance of zero.
- [ ] Keep the distinction clear: genetic similarity does not mean the cars drive the same line.
- [ ] If I compare driving behaviour later, save or re-evaluate a few selected genomes. Compare speed and lateral offset at the same positions along the track, rather than recording every car's full trajectory.

### Network structure and species

- [ ] Draw some of the generation-best networks, with inputs, hidden nodes, outputs and enabled connections.
- [ ] Count the effective network as well as stored genes. Some genes are disconnected from the outputs.
- [ ] Plot species fitness and success rates as well as size. Only calculate a rate when members were evaluated.
- [ ] Save distance to the species representative when the assignment is made, plus its ID. This should help explain why the current run has only one species.
- [ ] Maybe sample distances between other genome pairs too. Keep the sample small and reproducible without affecting evolution's random numbers.
- [ ] Be careful with conclusions: a larger network having better fitness does not prove the extra nodes caused the improvement.

The aim here is to explain whether the whole population improves, what changes in the score, and how the best differs from nearby genomes.

## Then: repeat experiments with different seeds

- [ ] Choose and save a list of distinct seeds before the main experiments. The number of runs depends on the compute time available.
- [ ] Use the same number of completed evaluations for comparisons. Record wall-clock time separately.
- [ ] Add an evaluation-count stopping option, so a faster laptop or extra plotting does not change the learning budget.
- [ ] Keep the task, input scales, reward, population settings and recording options fixed within each condition.
- [ ] Add a batch script that saves the experiment settings and starts separate runs without overwriting results.
- [ ] Compare curves at shared evaluation counts. Do not extend a curve beyond where a run actually stopped.
- [ ] Show individual seed curves, then summarise completion and performance across runs with medians, spread and uncertainty estimates where the number of seeds supports them.
- [ ] Treat separate seeded runs as the repeats. Neighbouring generations and repeated elites are not independent experiments.
- [ ] Include unsuccessful seeds in the results. Reporting only successful lap times would hide failures.

I should be able to repeat an experiment from its saved settings and see how much the outcome varies between seeds.

## Parameter sensitivity

- [ ] Start with the compatibility threshold because the run I inspected has only one species.
- [ ] Try 3.0, 1.5 and 1.0 with the same task, seeds and evaluation budget. These are candidate values, not known good settings.
- [ ] Compare species sizes/counts, success, fitness and network growth. More species is not automatically better.
- [ ] Check whether a lower threshold creates lots of tiny species. Look at how elitism and minimum species size affect this.
- [ ] After that, choose a few other parameters to study, such as population size, mutation rates or speed-reward weight. Change one at a time initially.
- [ ] For population size, compare equal evaluation counts, not equal generations.
- [ ] When changing the reward, compare completion, progress and lap time. The raw fitness scales will differ.
- [ ] Keep tuning separate from final testing. Do not choose settings based on the held-out transfer results.

I want parameter choices supported by several runs, rather than picking whatever worked in one run.

## Main experiment idea: train on three circuits

Train separate populations on three circuits, one circuit per training condition. Then freeze each champion and test it on all three circuits.

- [ ] Choose the circuits and check geometry, spawn, width and driveability. Decide whether the oval is useful as an easy control or whether to use three technical tracks.
- [ ] Keep physics, input scales, target laps and fitness the same. Document any necessary differences in episode time limits.
- [ ] Use the same seed list and evaluation budget for each training circuit. One run per circuit would only be a pilot.
- [ ] Decide champion selection before testing, for example the all-time best from training. Do not pick a champion using its test-track score.
- [ ] Automate transfer evaluation. No mutation, retraining or input rescaling on the target track.
- [ ] Save source run, genome ID, training/test circuits, evaluation settings and outcome for every test.
- [ ] Make a 3 × 3 training-track versus test-track table across seeds. The diagonal shows performance on the training track; the other cells show transfer.
- [ ] Report target-lap completion, validated progress fraction and lap time for fully completed attempts. Failed attempt duration is not a lap time.
- [ ] Compare lap times within each target circuit, since track lengths differ.
- [ ] Run the conservative baseline on each track for context, without presenting it as an optimised controller.
- [ ] If I tune settings after seeing these results, keep another circuit aside for a final unseen-track test.

The question is which training conditions transfer, how consistently across seeds, and where the unsuccessful drivers lose the track.

## Recording and code checks

- [ ] Save the Git commit, whether there are uncommitted changes, and a source snapshot or code hash with each run. Matching settings alone do not prove the code was identical.
- [ ] Save hardware and OS information when comparing execution times.
- [ ] Add a recording-format version and keep older outputs readable.
- [ ] Check the fields analysis depends on and give clear errors for missing or malformed data.
- [ ] Make record writes atomic so an interrupted write does not damage a JSON file. Save useful partial-run status and distinguish budget expiry, manual stop and crashes.
- [ ] If I study training speed, time simulation, recording and finalisation separately.
- [ ] Only save extra genomes for a specific analysis need. Full-population checkpoints can come later if resuming evolution is needed.
- [ ] Keep tests for known statistics, ties, partial generations, old recordings, comparison groups and leaving source files unchanged.

## Before the next supervisor handover

- [ ] Update `analysis/README.md` when fields, plots or defaults change. Follow `HowToWrite.md` and avoid changing CSV names unnecessarily.
- [ ] Simplify the generated report wording, especially the settings-group ID and metadata-completeness flag.
- [ ] Keep implemented features, measured results and future plans clearly separated.
- [ ] Review and commit the current work without mixing in unrelated local edits.
- [ ] Rebuild the supervisor ZIP. The old ZIP does not include later analysis and transfer changes automatically.
- [ ] Keep original training recordings and settings. CSVs and figures can be regenerated from them.

## Order I would follow

1. Improve the single-run analysis, especially fitness components and the comparison with the reference best.
2. Add equal-evaluation budgets and a batch runner for several seeds.
3. Run a small compatibility-threshold pilot before choosing the main settings.
4. Run the three-circuit experiment and transfer tests with fixed champions.
5. Write up the results and limits, keeping each figure traceable to its run and settings.
