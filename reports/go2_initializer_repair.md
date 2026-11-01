# Raised-leg initializer repair — 2026-10-29

The scripted Stage B initializer now produces sustained raised poses on both
front legs in the tested simulation distribution. No training was started.

## Diagnosis and change

The old preparation could leave the selected foot loaded and allow accepted
poses to drift at roughly 0.1–0.4 m/s. Its admission gate checked orientation and
foot placement but did not bound body translation speed.

The replacement controls body support force and torque, transfers the center of
mass into the remaining three-foot support triangle, and unloads the selected
foot before lifting. It permits a small mirrored lean and lowers the target base
height to .28 m. All movement uses physics and the stock position actuators;
there are no raised-state root/joint writes or reward changes.

The stages are 2 s weight transfer, at least .10 s measured unloading, 1.5 s lift,
and at least 1 s continuous stationary endpoint hold. Admission requires three
support contacts, selected foot airborne, endpoint tolerances, tilt <.20 rad,
full angular speed <.25 rad/s, body speed <.06 m/s, and every foot speed <.06 m/s.
A bad sample restarts the hold. Timeout is 10 s.

## Qualification

| Probe | Result |
|---|---|
| Seed 9030, 64 first attempts, 3 s hold | 32/32 FL and 32/32 FR; zero resets/falls |
| Seed 9031, 64 first attempts, 3 s hold | 32/32 FL and 32/32 FR; zero resets/falls |
| Seed 9032, bilateral video probes, 8 environments each | Both passed; zero resets/falls |
| Hierarchical component/config tests | 31 passed; all six configs validated |
| Mixed-entry reset/export smoke, 550 steps x16 | 16 raised admissions; zero preparation failures/falls; 100/112/12 dimensions and JIT parity/ONNX export passed |

Across the two 64-environment probes, all 19,328 recorded post-lift hold samples
passed the stability gate. Maximum height/lateral errors during these holds were
1.33/1.39 mm. At admission, maximum body speed was .265 mm/s, maximum foot speed
1.85 mm/s, full angular speed .00132 rad/s, and tilt .0717 rad (4.11 degrees).

The smoke used zero policy actions and an untrained network for export. Its
STEP trials did not succeed; these checks do not demonstrate learned landing,
kick execution, or live kick-to-step transitions. There were zero PPO updates.
The tested reset/endpoint variation does not establish robustness to changed
friction, mass, pushes, terrain or hardware.

## Artifacts and reproduction

- [Seed 9030](../logs/hierarchical_initializer_repair/hold3_seed9030/result.json)
- [Seed 9031](../logs/hierarchical_initializer_repair/hold3_seed9031/result.json)
- [Mixed smoke](../logs/hierarchical_initializer_repair/mixed_smoke/summary.json)
- [FL video](../logs/hierarchical_initializer_repair/video_fl/preparation.mp4)
- [FR video](../logs/hierarchical_initializer_repair/video_fr/preparation.mp4)

Both videos were visually checked at the held pose. Each result directory
contains the exported environment configuration. Earlier failed baseline and
controller API probes are retained separately in the same parent directory.

```bash
/home/rml2/anaconda3/envs/nick_sillywalk/bin/python scripts/hierarchical_preflight.py --headless --stage B --num_envs 64 --seed 9030 --hold-seconds 3 --trace --output logs/hierarchical_initializer_repair/new_probe
```

Use a fresh output directory. Read result.json; simulator process exit status
alone is not a reliable pass indicator.
