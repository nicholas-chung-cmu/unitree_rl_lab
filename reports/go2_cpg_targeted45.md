# Targeted Go2 robustness — 45-minute session, 2026-11-02

The selected timing-prioritized checkpoint at300 additional updates improves neutral drift and tested robustness while preserving the75-input deployment contract. MuJoCo90s neutral drift falls0.840→0.294m (65% reduction); Gazebo90s drift falls0.634→0.106m (83% reduction). Matched stronger Isaac uncertainty improves strict kick chains628/672→656/672, with no falls in either policy. The selected candidate passes all three Gazebo runs. It remains simulation/shadow-only, with backward travel and hardware fault handling unqualified.

## Scope and implementation

Authorized window: 04:52:12–05:37:12 UTC. No overnight extension or hardware activation. Start: the 75-input, 250-update robustness checkpoint (`logs/cpg_robust_compare_20261101/obs75_train1/training/model_249.pt`). Original exports, source checkpoint and hardware guard remain intact.

Added opt-in kick-extension body-stability and neutral-travel precision rewards. The actor still has 75 deployable observations. Physical simulator velocity is used only by rewards/critic. Travel rewards continue to exclude kick/hold/recovery pauses, and the 90 BPM clock and kick geometry are unchanged.

Added configurable directional sampling: 10% backward among translating commands, 15% pure-turn sampling, and 40–45% neutral commands. This changes training sampling only; external joystick commands are unchanged. Default settings preserve legacy sampling and RNG behavior.

Both branches use 4,096 environments, 24 steps/update, fixed learning rate 5e-5, existing mass/friction/gain/armature/COM/disturbance randomization, and upstream sensor uncertainty. Training includes 0–20 ms sensor-value delay and 0–10 ms actuator delay. The held-out stress test uses stronger noise and actuator delays up to 20 ms.

Precision branch: 600 updates. Initially kick-body reward 2.0, neutral reward 1.0, timing 0.25, kick tracking 1.5. After 300 updates, timing increased to 0.75 and kick tracking to 2.5; kick-body reward decreased to 1.0 after a higher-inertia timing regression.

Timing branch: 450 updates, restarted from the same preserved checkpoint. Kick-body reward 0.5, neutral 1.0, timing 1.0 and kick tracking 3.0 from the outset. Total training: 1,050 updates / 103,219,200 transitions, plus a separate two-update smoke test. One seed sequence per branch; these are engineering comparisons, not statistical superiority claims.

## Matched 40-second MuJoCo command checks

Non-backward totals include neutral, forward 0.12 m/s, lateral ±0.06 m/s, yaw ±0.1 rad/s, and combined 0.08 m/s +0.1 rad/s. Backward remains a diagnostic and is reported separately. Strict chains require endpoint timing, airborne hold and controller-confirmed post-kick landing.

| Checkpoint | Non-backward physical steps | Non-backward strict kicks | Neutral net drift (m) | Backward strict kicks |
|---|---:|---:|---:|---:|
| starting_75 | 308/308 | 98/98 | 0.370 | 14/14 |
| precision_150 | 307/308 | 98/98 | 0.158 | 5/14 |
| precision_300 | 302/308 | 95/98 | 0.107 | 3/14 |
| precision_600 | 306/308 | 96/98 | 0.145 | 4/14 |
| timing_150 | 307/308 | 95/98 | 0.139 | 2/14 |
| timing_300 | 308/308 | 98/98 | 0.134 | 3/14 |
| timing_450 | 304/308 | 96/98 | 0.144 | 3/14 |

## Longer and perturbed MuJoCo checks

All runs use the same startup/ending neutral intervals and matched inputs. The demo is 120 seconds with forward, lateral and yaw changes; no reverse segment. Neutral runs last 90 seconds, forward 60 seconds at +0.10 m/s, perturbations 40 seconds.

| Candidate | Case | Physical steps | Strict kick chains | Net x / y (m) | Fault |
|---|---|---:|---:|---:|---|
| baseline_transfer | mj_action_delay20 | 44/44 | 13/14 | +0.832 / +0.007 | none |
| baseline_transfer | mj_armature_015 | 44/44 | 11/14 | +1.000 / -0.003 | none |
| baseline_transfer | mj_demo | 134/134 | 43/44 | +1.741 / +0.408 | none |
| baseline_transfer | mj_forward | 66/66 | 21/21 | +2.046 / -0.033 | none |
| baseline_transfer | mj_gains_090 | 44/44 | 12/14 | +0.711 / +0.000 | none |
| baseline_transfer | mj_mass_110 | 44/44 | 14/14 | +0.773 / +0.002 | none |
| baseline_transfer | mj_sensor_delay20 | 44/44 | 14/14 | +0.832 / +0.007 | none |
| baseline_transfer | mj_sensor_noise | 44/44 | 14/14 | +1.043 / +0.001 | none |
| baseline_transfer | mj_zero | 100/100 | 33/33 | -0.840 / +0.007 | none |
| precision150_transfer | mj_action_delay20 | 44/44 | 14/14 | +1.074 / -0.017 | none |
| precision150_transfer | mj_armature_015 | 44/44 | 2/14 | +1.127 / -0.010 | none |
| precision150_transfer | mj_demo | 134/134 | 43/44 | +2.073 / +0.442 | none |
| precision150_transfer | mj_forward | 66/66 | 20/21 | +2.145 / -0.027 | none |
| precision150_transfer | mj_gains_090 | 41/44 | 13/14 | +0.766 / -0.009 | none |
| precision150_transfer | mj_mass_110 | 42/44 | 14/14 | +0.834 / -0.004 | none |
| precision150_transfer | mj_sensor_delay20 | 44/44 | 14/14 | +1.074 / -0.017 | none |
| precision150_transfer | mj_sensor_noise | 44/44 | 14/14 | +1.134 / -0.004 | none |
| precision150_transfer | mj_zero | 100/100 | 33/33 | -0.332 / +0.004 | none |
| precision600_transfer | mj_action_delay20 | 42/44 | 14/14 | +1.322 / +0.000 | none |
| precision600_transfer | mj_armature_015 | 44/44 | 13/14 | +1.430 / -0.013 | none |
| precision600_transfer | mj_demo | 134/134 | 43/44 | +2.778 / +0.591 | none |
| precision600_transfer | mj_forward | 66/66 | 17/21 | +2.781 / -0.032 | none |
| precision600_transfer | mj_gains_090 | 42/44 | 14/14 | +1.164 / +0.001 | none |
| precision600_transfer | mj_mass_110 | 43/44 | 14/14 | +1.233 / -0.003 | none |
| precision600_transfer | mj_sensor_delay20 | 42/44 | 14/14 | +1.322 / +0.000 | none |
| precision600_transfer | mj_sensor_noise | 44/44 | 13/14 | +1.457 / -0.008 | none |
| precision600_transfer | mj_zero | 100/100 | 33/33 | -0.307 / +0.003 | none |
| timing150_transfer | mj_action_delay20 | 44/44 | 14/14 | +0.897 / -0.010 | none |
| timing150_transfer | mj_armature_015 | 44/44 | 12/14 | +1.039 / +0.002 | none |
| timing150_transfer | mj_demo | 134/134 | 43/44 | +1.970 / +0.434 | none |
| timing150_transfer | mj_forward | 66/66 | 21/21 | +2.039 / -0.014 | none |
| timing150_transfer | mj_gains_090 | 44/44 | 14/14 | +0.762 / +0.001 | none |
| timing150_transfer | mj_mass_110 | 44/44 | 14/14 | +0.834 / +0.001 | none |
| timing150_transfer | mj_sensor_delay20 | 44/44 | 14/14 | +0.897 / -0.010 | none |
| timing150_transfer | mj_sensor_noise | 44/44 | 14/14 | +1.076 / +0.003 | none |
| timing150_transfer | mj_zero | 100/100 | 33/33 | -0.295 / +0.008 | none |
| timing300_transfer | mj_action_delay20 | 44/44 | 14/14 | +1.117 / -0.006 | none |
| timing300_transfer | mj_armature_015 | 44/44 | 13/14 | +1.282 / -0.007 | none |
| timing300_transfer | mj_demo | 134/134 | 43/44 | +2.456 / +0.534 | none |
| timing300_transfer | mj_forward | 66/66 | 21/21 | +2.484 / -0.023 | none |
| timing300_transfer | mj_gains_090 | 44/44 | 14/14 | +0.910 / +0.004 | none |
| timing300_transfer | mj_mass_110 | 44/44 | 14/14 | +0.992 / +0.005 | none |
| timing300_transfer | mj_sensor_delay20 | 44/44 | 14/14 | +1.117 / -0.006 | none |
| timing300_transfer | mj_sensor_noise | 44/44 | 14/14 | +1.321 / -0.000 | none |
| timing300_transfer | mj_zero | 100/100 | 33/33 | -0.294 / +0.005 | none |
| timing450_transfer | mj_action_delay20 | 43/44 | 13/14 | +1.175 / -0.002 | none |
| timing450_transfer | mj_armature_015 | 44/44 | 13/14 | +1.326 / -0.005 | none |
| timing450_transfer | mj_demo | 134/134 | 44/44 | +2.553 / +0.568 | none |
| timing450_transfer | mj_forward | 66/66 | 21/21 | +2.564 / -0.024 | none |
| timing450_transfer | mj_gains_090 | 44/44 | 14/14 | +0.998 / +0.002 | none |
| timing450_transfer | mj_mass_110 | 44/44 | 14/14 | +1.076 / +0.004 | none |
| timing450_transfer | mj_sensor_delay20 | 43/44 | 13/14 | +1.175 / -0.002 | none |
| timing450_transfer | mj_sensor_noise | 44/44 | 14/14 | +1.359 / +0.003 | none |
| timing450_transfer | mj_zero | 100/100 | 33/33 | -0.319 / +0.007 | none |

## Gazebo / Quad-SDK validation

The complete Quad-SDK path uses the unchanged opt-in`go2hv_mujoco_v1` profile. Tests ran after GPU training/evaluation stopped. All three completed without faults.

| Case | Physical steps | Strict kick chains | Peak tilt (deg) | Net x / y (m) |
|---|---:|---:|---:|---:|
| gz_demo | 134/134 | 44/44 | 7.5 | +2.613 / +1.102 |
| gz_forward | 66/66 | 21/21 | 6.7 | +2.807 / -0.076 |
| gz_zero | 100/100 | 33/33 | 5.4 | -0.105 / +0.015 |

The neutral and forward conditions match the prior comparison. The demo uses the new non-backward sequence, so its displacement is not directly comparable with the older mixed-command run.

## Matched combined uncertainty test

32 environments,60seconds each, seed9321, seven non-backward commands switching every6seconds. Stronger upstream noise/bias/contact dropout and0–20ms sensor/actuator delays are enabled alongside physics randomization.

| Policy | Physical steps | Strict kick chains | Falls |
|---|---:|---:|---:|
| baseline_stress | 2095/2111 | 628/672 | 0 |
| selected_stress | 2097/2111 | 656/672 | 0 |

The selected MuJoCo neutral run retained a median step peak clearance of4.9cm (starting policy5.6cm), with minimum body height0.2995m versus0.3000m. Reduced drift did not come from sliding the feet on the floor, but the clearance reduction should be monitored in later training.

## Diagnostics and limits

A frozen-policy reference scan found little benefit from reducing the backward kick-preparation shift (3 cm to 2/1 cm). Increasing support-velocity damping from 6 to 14 reduced 40-second neutral drift from 0.370 to 0.317 m, but this parameter change was not adopted. The selected training candidates retain the original reference gains.

The early precision policy reduced drift but regressed to 2/14 strict kick chains at armature 0.015, versus 11/14 for the starting policy. Physical endpoint measurements identified narrow planar-placement and foot-settling margins, not a height or body-tilt violation. Increasing timing/placement emphasis restored the later precision checkpoint to 13/14 under that condition, while other checks still regressed.

An instrumented replay of the precision600 forward case identified four post-kick controller rejections: both commanded feet landed on time, but the opposite front support foot lifted about 2.7 mm and lost contact for 40 ms. This reached the rejection threshold at about 67 ms after the beat. The independent physical-step audit ends at the 60 ms window, so it counts 66/66 while the controller counts 62/66. The strict kick-chain audit retains the controller-confirmed post-kick requirement and reports 17/21. Thresholds were not relaxed, and this is a real support-margin issue plus an audit-window discrepancy to investigate. Evidence: `step_rejection_diagnosis.json`, `support_unloading_details.json`, and `step_diagnostic/`.

Backward reliability remains poor and is not qualified. The intended next hardware demo should have an explicitly restricted operating envelope; this session did not change joystick clamping or enable hardware. Physical actuator/contact calibration, real-time timing validation, total PD torque limiting and airborne fault recovery remain unresolved.

## Verification and artifacts

18 regression checks passed, including the new directional-curriculum contracts and legacy RNG preservation. The 4,096-environment smoke completed. Training and evaluation settings, exact code snapshots, hashes, gates, physical traces, perturbation readbacks and selected export metadata are under `logs/cpg_targeted45_20261102/`.

P=precision branch, T=timing-prioritized branch; numbers are additional PPO updates.

![Neutral drift and strict non-backward kick completion](go2_cpg_targeted45.png)

## Selected artifacts and next-session boundary

Portable candidate: `exports/go2_cpg_targeted45_20261102/`. Checkpoint:
`logs/cpg_targeted45_20261102/timing_train2/training/model_299.pt`.
Training and tests are stopped at handoff; no overnight job was launched.
See `go2_cpg_targeted45_handoff.md` for the proposed next experiments and the
remaining hardware boundary. Preserve the prior robust75 export as fallback.
