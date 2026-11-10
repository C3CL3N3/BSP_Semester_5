# Mac handover checks

Checked on 3 November 2026 on the development Mac using the existing Python environment.

```sh
python -m unittest discover -s tests -q
```

All 66 tests passed. The suite covers kinematic and dynamic motion, aerodynamics, track assets, episode termination/progress, training, champion saving and replay loading/export. This version uses serial genome evaluation.

`git diff --check` passed. The supervisor ZIP was checked for corrupt members and compared byte-for-byte with the selected working files.

No new long technical-track training was performed for this handover. Interactive playback was not manually checked. Test success does not establish realistic tyre behaviour or generalisation to unseen circuits.

## Population recording update, 10 November 2026

The suite now has 69 passing tests, including partial-generation denominators, known fitness quartiles, species membership/totals, genetic distance to the generation best, repeatable population records and safe readable run names. Baseline replay saving is tested both enabled and disabled. A two-generation, four-genome smoke run also produced the population figure, which was visually checked. This short run validates recording, not learning performance. Physics and fitness were not changed by this update.

## GitHub source preparation, 10 November 2026

All 73 tests pass in the development checkout. The same source was exported without Git history, local recordings, archives or the virtual environment and tested separately using the existing interpreter/dependencies. The export includes source, tests, settings, track assets and documentation. This checks that the tests do not depend on local experiment artifacts; it is not a fresh dependency-install test.

The current `neat.cfg` uses 50 genomes. Earlier sections describe checks made before the analysis and transfer additions. No long training was run as part of repository cleanup.
