# Gazebo passive spring/damping comparison — 2026-11-01

Six bounded headless runs: two repeats per condition, 40 seconds of CPG control
per run. Frozen experimental traveling-dance model 2499, 75 observations, unchanged
policy checksum, motor PD 25/0.5, nominal references, command limits and contact
handling. Command schedule relative to first target: zero 0–9 s, +0.10 m/s 9–24 s,
zero 24–40 s. Each run started at simulation time 11.02 s after a fresh sit/stand
transition. No training or hardware actions. Defaults were not changed.

| Condition | Completed steps | Strict completed kick chains | Hold passes | Mean endpoint delay | Forward distance during 15 s input | Forward drift during final 16 s neutral |
|---|---:|---:|---:|---:|---:|---:|
| Original | 88/88 | 13/28 | 13/28 | 47.9 ms | 1.049 m | +0.516 m |
| Springs off; passive damping retained | 88/88 | 10/28 | 28/28 | 65.7 ms | 0.705 m | -0.007 m |
| Springs and passive damping off | 88/88 | 28/28 | 28/28 | 40.7 ms | 0.977 m | +0.004 m |

Distances are means of two repeats; counts pool both runs. The forward interval
includes the choreography's kick pauses and is not a pause-excluded velocity
score. Final neutral includes deceleration immediately after the command changes.
All runs completed without falls, latched faults, missed steps or recovery skips.
All 84 kicks reached endpoints and had successful post-kick landings; the strict
failures were endpoint timing or hold quality, not lost choreography.

The strict criterion remains <=60 ms endpoint delay, >=80% valid hold samples,
and successful post-kick landing with no hold failure. These are the same
controller/estimator-based checks used in the integration report. Ground-truth
foot positions are logged separately and never enter observations/control.

## Interpretation

Removing springs improved held-foot clearance and eliminated the large neutral
forward drift, but slowed the kick enough to reduce strict timing success.
Removing passive damping next, with springs already off, restored timely
extension and preserved good holds. This sequential comparison isolates the
spring change at original damping, then the damping change at zero springs;
it is not a full factorial identification of every interaction.

The springs and damping being disabled are the passive SDF terms, not motor PD.
Motor Kp 25 and Kd 0.5 stayed enabled throughout. Original passive values are
3 Nm/rad springs toward sitting and 0.5 Nm·s/rad damping on all 12 revolute joints.
Generated model dumps were compared: after restoring those changed values,
all six SDFs are identical. Source SDF and policy SHA checks remained unchanged.

The simulator foot-truth cross-check shows average held toe clearance:

- Original: mean 3.84 cm; minimum 3.31 cm. Maximum body tilt 4.32°.
- Springs off: mean 4.12 cm; minimum 3.78 cm. Maximum body tilt 6.96°.
- Both passive terms off: mean 4.55 cm; minimum 4.11 cm. Maximum body tilt 7.21°.

Toe clearance here is Gazebo toe-link world-z minus 22 mm, using the same reference
convention as the policy. The collision sphere has a 2 mm local offset, so this
is not an exact geometric minimum-distance calculation. Truth timestamps lagged
control by at most a few milliseconds (recorded in summary.json). The truth
cross-check supports a physical clearance improvement beyond threshold effects
in the deployable estimator. The successful condition increased maximum body tilt from 4.32° to 7.21°,
so the improvement does not mean every balance metric improved. Two repeats
of one short command schedule do not establish broad robustness.
Zero passive terms still do not establish the
real Go2's damping/friction or guarantee hardware transfer.

## Recommendation and reproduction

Keep the 75-input policy as the baseline. The clean passive-dynamics configuration
is a strong next simulation candidate, with broader directional/recovery tests
and physical actuator calibration still pending. This test does not justify an
immediate 81-input retraining run. No defaults were promoted automatically.

To reproduce the best diagnostic configuration, add these arguments to the
existing `quad_cpg_gazebo.py` launch with the experimental dance selection:

```bash
cpg_disable_springs:=true cpg_disable_damping:=true
```

Omitting them preserves the original spring/damping settings. `cpg_model_dump`
optionally writes the exact generated SDF for audit. Full launch instructions
remain in Quad-SDK `docs/cpg_gazebo.md`.

Evidence: `logs/cpg_passive_ab_20261101/` contains case traces, generated models,
launch/probe logs, run metadata, policy/source checksums, analysis and source
snapshots. `summary.json` lists individual kick delays, hold fractions and truth
clearances. `aggregate.json` backs the table. All test simulations were stopped.

![Passive-dynamics comparison](../logs/cpg_passive_ab_20261101/comparison.png)
