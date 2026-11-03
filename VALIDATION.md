# Mac handover checks

Checked on 3 November 2026 on the development Mac using the existing Python environment.

```sh
python -m unittest discover -s tests -q
```

All 66 tests passed. The suite covers kinematic and dynamic motion, aerodynamics, track assets, episode termination/progress, training, champion saving and replay loading/export. This version uses serial genome evaluation.

`git diff --check` passed. The supervisor ZIP was checked for corrupt members and compared byte-for-byte with the selected working files.

No new long technical-track training was performed for this handover. Interactive playback was not manually checked. Test success does not establish realistic tyre behaviour or generalisation to unseen circuits.
