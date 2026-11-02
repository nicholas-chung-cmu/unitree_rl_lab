# Go2 CPG robustness comparison — 2026-11-01

Plant alignment and matched 75/81-input fine-tuning are complete. Both candidates remain simulation/shadow-only; no hardware controller was enabled. Existing checkpoints, exports, source SDF and default launch profile were preserved.

## Protocol

Each arm received 1,000 PPO updates using 4,096 environments × 24 steps: 98,304,000 training transitions per arm, 196,608,000 total. Both started from the experimental model2499, copied actor/critic weights, reset optimizer state at learning rate 0.0003, and used the same stage seeds. The 81-input actor appended contact-kinematic vx/vy and four measured contact flags; its six new first-layer columns started at zero. Critic width stayed 90. This is one seed per arm, not a statistical superiority study.

The first 250 updates used mild upstream sensor uncertainty. The remaining 750 increased sensor/actuator delays and noise. Randomization covers friction, mass/inertia, gains, armature, passive friction/damping, COM, small disturbances, sensor bias, packet hold and contact dropout. Both arms use the same changes; improvement over the baseline cannot be attributed to one change alone. See [configuration and observation contract](../docs/go2_cpg_robustness.md).

Travel tracking excludes kick, hold, recovery and reset. A separate HOLD-body stability reward was added; the controller never requests catch-up travel during the kick pause. The 90 BPM musical clock and kick specification are preserved.

## Nominal evaluation

Every checkpoint ran the same eight 40-second MuJoCo commands and 32-environment Isaac evaluation. Strict kick chains require timely endpoint, airborne hold, and the following measured landing. Physical contact traces are used where sensor corruption is present.

| Candidate | Isaac steps | Isaac strict kicks | MuJoCo steps | MuJoCo strict kicks | MuJoCo faults/falls |
|---|---:|---:|---:|---:|---:|
| baseline | 1408/1408 | 379/448 | 340/352 | 19/112 | 0 |
| obs75_250 | 1408/1408 | 448/448 | 352/352 | 112/112 | 0 |
| obs81_250 | 1408/1408 | 448/448 | 346/352 | 99/112 | 0 |
| obs75_1000 | 1408/1408 | 448/448 | 343/352 | 94/112 | 0 |
| obs81_1000 | 1408/1408 | 448/448 | 337/352 | 98/112 | 0 |

Within each arm, checkpoint selection prioritizes failures, worst-command step success, worst-command kick success, total kick success, then neutral drift. Selected: obs75_250, obs81_250. Latest checkpoints are not automatically preferred.

## Longer and perturbed transfer tests

The following tests use the selected checkpoint from each arm. Fixed commands have 6 seconds of neutral startup and 10 seconds of neutral ending; mixed commands change every 10 seconds. Neutral runs last 90 seconds, backward/forward 60 seconds, mixed 120 seconds; individual perturbations last 40 seconds. Gazebo uses the full Quad-SDK ROS path with opt-in `go2hv_mujoco_v1`.

| Simulator / candidate / case | Status | Physical contact step audit | Strict physical kick chains | Peak tilt (deg) | Net x / y (m) |
|---|---|---:|---:|---:|---:|
| gazebo/obs75_gz_backward | fault Excessive body tilt | 30/31 | 9/10 | 34.7 | -1.875 / -0.077 |
| gazebo/obs75_gz_forward | complete  | 66/66 | 21/21 | 9.1 | +2.398 / -0.065 |
| gazebo/obs75_gz_mixed | complete  | 134/134 | 44/44 | 9.6 | +0.771 / +0.311 |
| gazebo/obs75_gz_zero | complete  | 100/100 | 33/33 | 5.6 | -0.634 / +0.016 |
| gazebo/obs81_gz_backward | fault Excessive body tilt | 31/32 | 10/10 | 36.5 | -1.921 / +0.074 |
| gazebo/obs81_gz_forward | fault Missed simulation control tick | 27/27 | 8/8 | 6.4 | +1.061 / -0.018 |
| gazebo/obs81_gz_forward_retry | complete  | 66/66 | 21/21 | 6.4 | +2.568 / -0.086 |
| gazebo/obs81_gz_mixed | complete  | 134/134 | 44/44 | 7.1 | +1.073 / +0.335 |
| gazebo/obs81_gz_zero | complete  | 100/100 | 33/33 | 4.9 | -0.396 / +0.009 |
| mujoco/obs75_mj_action_delay20 | complete  | 44/44 | 13/14 | 8.7 | +0.832 / +0.007 |
| mujoco/obs75_mj_armature_005 | complete  | 44/44 | 14/14 | 9.2 | +1.064 / -0.003 |
| mujoco/obs75_mj_armature_015 | complete  | 44/44 | 11/14 | 9.4 | +1.000 / -0.003 |
| mujoco/obs75_mj_backward | complete  | 66/66 | 21/21 | 4.4 | -3.255 / -0.059 |
| mujoco/obs75_mj_gains_090 | complete  | 44/44 | 12/14 | 12.0 | +0.711 / +0.000 |
| mujoco/obs75_mj_gains_110 | complete  | 44/44 | 14/14 | 8.5 | +1.285 / -0.018 |
| mujoco/obs75_mj_mass_110 | complete  | 44/44 | 14/14 | 11.1 | +0.773 / +0.002 |
| mujoco/obs75_mj_mixed | complete  | 134/134 | 44/44 | 10.0 | +0.411 / +0.182 |
| mujoco/obs75_mj_reference40 | complete  | 44/44 | 14/14 | 9.1 | +1.042 / +0.001 |
| mujoco/obs75_mj_sensor_delay20 | complete  | 44/44 | 14/14 | 8.7 | +0.832 / +0.007 |
| mujoco/obs75_mj_sensor_noise | complete  | 44/44 | 14/14 | 9.2 | +1.043 / +0.001 |
| mujoco/obs75_mj_zero | complete  | 100/100 | 33/33 | 4.5 | -0.840 / +0.007 |
| mujoco/obs81_mj_action_delay20 | complete  | 44/44 | 14/14 | 6.7 | +0.996 / +0.001 |
| mujoco/obs81_mj_armature_005 | complete  | 44/44 | 14/14 | 7.1 | +1.182 / +0.001 |
| mujoco/obs81_mj_armature_015 | complete  | 44/44 | 11/14 | 7.0 | +1.149 / -0.000 |
| mujoco/obs81_mj_backward | complete  | 58/66 | 7/21 | 8.1 | -2.791 / -0.481 |
| mujoco/obs81_mj_gains_090 | complete  | 44/44 | 13/14 | 8.2 | +0.822 / +0.010 |
| mujoco/obs81_mj_gains_110 | complete  | 44/44 | 14/14 | 6.3 | +1.436 / -0.011 |
| mujoco/obs81_mj_mass_110 | complete  | 44/44 | 14/14 | 8.1 | +0.892 / +0.011 |
| mujoco/obs81_mj_mixed | complete  | 133/134 | 41/44 | 7.6 | +0.801 / +0.143 |
| mujoco/obs81_mj_reference40 | complete  | 44/44 | 14/14 | 6.7 | +1.167 / +0.003 |
| mujoco/obs81_mj_sensor_delay20 | complete  | 44/44 | 14/14 | 6.7 | +0.996 / +0.001 |
| mujoco/obs81_mj_sensor_noise | complete  | 44/44 | 14/14 | 6.9 | +1.173 / +0.006 |
| mujoco/obs81_mj_zero | complete  | 100/100 | 33/33 | 4.2 | -0.487 / +0.008 |

The step and strict kick columns use physical contact audits; controller success counters are retained separately in the raw summaries. Torque estimates in the machine-readable summaries use nominal gains and 50 Hz samples; they are not measured motor torque and do not represent peak torque between samples.

## Held-out randomized Isaac checks

Each check uses 32 environments, 60 seconds, changing commands, stronger sensor/physics uncertainty and a held-out seed (9217). Final checkpoints and any earlier selected checkpoints are evaluated separately.

| Candidate | Physical contact step audit | Strict physical kick chains | Falls |
|---|---:|---:|---:|
| obs75_selected_stress | 2088/2112 | 609/672 | 0 |
| obs75_stress | 2090/2112 | 645/672 | 0 |
| obs81_selected_stress | 2085/2112 | 612/672 | 0 |
| obs81_stress | 2090/2112 | 643/672 | 0 |

Final adaptive learning rates: obs75_train1: 0.000171, obs75_train2: 7.59e-05, obs81_train1: 0.000384, obs81_train2: 0.000256. Mean episode length reached 1,500 control steps at the end of every stage; this did not prevent transfer regressions.


## Interpretation and remaining work

Keep the 75-input, 250-update checkpoint as the next simulation development baseline. It has the strongest nominal contact timing and mixed-command transfer. The 81-input candidate reduces neutral drift and performs similarly under randomized Isaac conditions, but loses backward kick/step reliability. This single-seed experiment does not establish that the six extra inputs are intrinsically unhelpful. Neither candidate is hardware-ready: both hit excessive tilt during sustained backward Gazebo travel.

The 75-input Gazebo forward trial traveled +2.398 m and averaged 0.082 m/s during travel phases at a 0.10 m/s command. Its neutral 90-second drift was 0.634 m in Gazebo and 0.840 m in MuJoCo; 81 inputs reduced those to 0.396 m and 0.487 m. These are net displacements, not path lengths. Much of the residual drift occurs outside HOLD, so increasing only the hold reward would miss the problem.

Next work should diagnose the sustained backward instability against actuator/inertia/contact differences, then address body translation during kick extension and zero-command steps without adding the kick pause to travel tracking. Longer fine-tuning alone is not the immediate recommendation: the 1,000-update checkpoints improved randomized performance but regressed on nominal transfer. Keep hardware activation blocked until calibration and fault handling are validated.

The physical Go2 configuration remains uncalibrated: actual foot-force units/thresholds, encoder/IMU noise and latency, motor gains/envelope, friction and effective rotor inertia must be measured. Gazebo rotor inertia is not reconciled by the profile, and simulator friction implementations differ. Flat firm ground reduces terrain uncertainty but does not remove these actuator/sensor gaps.

The hardware barriers from the prior audit remain: validated airborne fault recovery and enforcement of total PD-plus-feedforward torque limits in the hardware path. The current RobotDriver effort clamp alone does not establish that limit. Joystick loss continues dancing at zero velocity; stale sensor data latches output off. No physical robot was connected.

## Integration verification

Regression checks passed for the controller/deployment path, sensor delay resets,
physical-truth isolation, 81-input reflection/warm-start equivalence and kinematic
velocity estimation. The 4,096-environment smoke completed, and exported 81-input
ONNX/Torch outputs matched to a maximum absolute difference of 8.35e-7. The
Gazebo plugin/launch build and live actuator-profile readback passed.

Both selected policies passed 600-packet localhost ROS replay and stale-sensor
latch checks. The first 81-input replay stopped on a stale stream while other
work was running; its retry passed after GPU evaluation stopped. Similarly,
the first 81-input Gazebo forward run stopped on a missed simulation tick;
a later isolated 60-second retry completed. These traces are retained, not
removed from the results. They demonstrate fault latching, but do not establish
a cause or a hardware real-time guarantee. No deadline was relaxed for retries.

The longer suite comprises 32 planned cases plus that forward retry. The full
experiment, including these evaluations, remained within the two-hour budget.
Original exported ONNX/metadata and Quad-SDK source YAML hashes match the prior
audit. Training is stopped. The first 81-input evaluation summary predates a
width-annotation correction; ONNX shape and `physics_readback.json` are the
authoritative 81-input evidence, recorded in `export_dimensions_audit.json`.

## Reproduction and evidence

- Main runner: `scripts/run_cpg_robust_compare.py`; immutable source snapshot and hashes under `logs/cpg_robust_compare_20261101/source/`.
- Stage settings, checkpoints, readbacks, ONNX exports, gates and runner status are in that session directory.
- Longer-test runner/probe, model readbacks, sensor traces, physical kick audits and summaries are under `posteval/`.
- The evaluation-only physical-contact logging correction is recorded in `truth_audit_revision.json`; it does not change training dynamics.
- Selected portable exports contain scheduler, IK, actor, runtime, metadata and a SHA-256 manifest. They do not install a hardware controller.

![Nominal transfer results and neutral drift](go2_cpg_robust_compare.png)
