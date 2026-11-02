# Quad-SDK CPG/Gazebo integration — 2026-11-01

The existing 75-input controller now runs through Quad-SDK's `cpg` controller
mode in Gazebo. No policy weights, observation dimensions or training rewards
were changed. No training ran. Hardware activation is explicitly disabled.

## Final comparisons

The following use the first 1500 target samples (29.98 seconds, including runtime
settling) with the final CPG contact model. Walking and experimental dance used
approximately 15 seconds of +0.10 m/s forward input, with neutral input before
and after. Original dance used zero commanded velocity throughout.

| Policy | Completed steps | Post-kick landings | Strict completed kick chains | Net forward / lateral | Max tilt |
|---|---:|---:|---:|---:|---:|
| Walking-only model899 | 43/43 | — | — | 1.477 / -0.112 m | 5.61° |
| Experimental traveling dance model2499 | 33/33 | 10 | 6/10 | 1.447 / -0.066 m | 4.22° |
| Preserved original dance U | 17/23 | 0 | 0/10 | 0.451 / -0.017 m | 15.15° |

No body falls occurred in these windows: minimum body heights were 0.262,
0.275 and 0.252 m respectively. This is a bounded flat-ground integration test,
not broad robustness or hardware qualification. Zero commands still produce
marching/dancing, and there is noticeable uncommanded drift.

The experimental dance reached all 10 completed endpoints and retained enough
support to incorporate all 10 post-kick landings, but only 6 chains satisfied
endpoint delay <=60 ms, >=80% valid hold samples and a successful post-kick step.
An eleventh kick was in progress at the window boundary. These checks use measured
Gazebo contact forces and the deployable encoder/IMU clearance/speed estimator;
they are **not an independent simulator-foot-ground-truth audit** and are not
interchangeable with the earlier MuJoCo qualification results. Original dance
lost support during all 10 completed holds and entered recovery, skipping 10
scheduled steps in addition to its 6 unsuccessful attempted steps.

Within the 14.98-second nonzero-command interval, forward displacements were
1.102 m walking and 1.047 m experimental dance. Those intervals include kick
pauses for the dance; these are displacement measurements, **not pause-excluded
velocity-tracking scores**. Total-window displacement also includes neutral drift.

A longer, earlier walking test before collision deduplication completed 257/257
steps over 172.76 s, including backward, lateral, turning and combined commands.
No falls or limit clamps; maximum tilt 6.89°. Backward motion was weak and neutral
marching drifted. That earlier dynamics/contact condition is recorded separately;
it does not substitute for the final-model regression above.

## Changes

Quad-SDK: simulation-only `CpgController`, state-machine mode feedback, dedicated
`quad_cpg_gazebo.py` launch and raw sensor/worker bridge. RobotDriver remains the
sole actuator command publisher. Existing walking mode and old policies remain.
Runtime: opt-in worker telemetry only; existing 75-input actor behavior unchanged.
Tests now resolve relocatable bundles after the repository move; archived selection
files and manifests were not rewritten.

Startup fixes: proper policy-path override and overlay resource discovery from
prior work; IMU publication corrected from 1 to500 Hz. Four overlapping fixed-lump
foot collision spheres are removed only from the generated CPG Go2 SDF. Sensor
spheres remain, and inertias/friction/joint dynamics are unchanged. This corrected
force reporting but did not fix the old dance's support-loss failures. Individual
Gazebo contact forces expire after10 ms when their event stream stops during flight.

No training was used to obtain these integration results. Bundle SHA checks were
performed by the existing selection loader each time the worker started.

## Validation

- `robot_driver` and `quad_utils` rebuilt successfully in `install_nick_conga`.
- 11 deployment tests passed, including observation/action mapping and archived
  controller/JIT parity;4 Gazebo force-decoding regression tests passed.
- SDF check: exactly4 redundant collisions removed; toe sensors, inertia and
  joint dynamics unchanged. Python syntax and git whitespace checks passed.
- Real localhost ROS/Gazebo runs exercised settling, marching, direction changes,
  zero/stale motion input, kicks and post-kick transitions.
- Stopping the adapter triggered the C++ stale-target latch and standing fallback.
  Injecting an empty command also latched a fault; later valid commands did not
  clear it. Both are simulation-only fallback tests.
- Worker processing p99:3.05 ms walking and3.67 ms experimental dance in final
  windows, below the20 ms policy period. This is not a hardware timing guarantee.

All test instances started for this integration were stopped after testing.
No physical Go2 or wireless controller was connected.

## Before an 81-observation experiment

The most important unresolved mismatch is physical dynamics, not input shape.
The Gazebo SDF applies passive springs3 Nm/rad toward sitting and damping0.5.
At the nominal standing knee angle, the spring contributes about3.2 Nm, comparable
to the maximum3 Nm-equivalent correction from a0.12 rad residual atKp25. The model
mass is16.247 kg versus nominal16.087 kg. These parameters were left unchanged;
audit them against the physical Go2 before training a policy to compensate.

An offline audit of the support-foot velocity estimator against Gazebo root truth
found world-x/y RMSE about0.0135/0.0040 m/s for walking and0.0137/0.0048 m/s for
experimental dance. This uses ideal simulated encoder/IMU signals, flat ground,
and logged support contacts. It does not establish real-world accuracy or robustness
to slipping/noise. The current actor does not consume this estimated velocity.

Recommendation for review: retain the75-input integration baseline, reconcile
actuator dynamics, then compare an81-input actor adding estimated horizontal
velocity and measured contacts using the same estimator during training. Adding
observations alone does not resolve artificial spring loads or imperfect holds.
No81-input policy or training run has been started or scheduled.

## Reproduction

Launch/build/command instructions: Quad-SDK `docs/cpg_gazebo.md`.
Evidence: `logs/cpg_gazebo_integration_20261101/` contains per-case traces and
launch logs, executable analysis, bounded command probes, source snapshots and
checksums. Run `analyze.py` with the policy Python interpreter to regenerate
`summary.json`. Only final comparison windows are used for the table above;
traces also retain extra neutral operation and intentional fault-injection tails.
