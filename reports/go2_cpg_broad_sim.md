# Broader conga simulation checks and configuration audit

Completed 2026-11-01. Frozen experimental dance model2499, 75 actor inputs,
12 residual actions. No training, policy edits, hardware commands, or default
physics changes. The existing hardware-mode rejection remains in place.

## Direction and transition trials

Eight scenarios per simulator: neutral 90 s; forward/backward/lateral/turning
60 s each; changing commands 120 s. Fixed tests use neutral input for the first
6 s and last 10 s. Commands: vx ±0.10 m/s, vy ±0.05 m/s, yaw ±0.10 rad/s.
Changing inputs alternate those directions every 10 s and include combined
translation/turning. Both front legs kick repeatedly. These are one rollout per
scenario, not statistical qualification.

Gazebo runs the complete Quad-SDK stack with the opt-in passive springs and
passive damping disabled; motor PD remains 25/0.5. MuJoCo uses the unchanged
local official robot model with flat floor, armature .01, passive damping .1,
and joint frictionloss .2. Original source models and policy remain unchanged.
Startup differs: Gazebo includes Quad-SDK sit/stand and READY; MuJoCo uses its
measured settling procedure. This is a cross-stack check, not an isolated test
of the simulation engine alone.

The table uses the same independent physical foot-position/velocity kick audit
in both simulators. It checks endpoint, timing, hold, and post-kick landing,
including scheduled kicks that never reached a successful endpoint. The raw
controller's kick_success counter is less strict and must not replace this table.

| Scenario | Gazebo strict chains | MuJoCo strict chains | Gazebo outcome |
|---|---:|---:|---|
| zero | 33/33 | 0/33 | Completed |
| forward | 21/21 | 16/21 | Completed |
| backward | 8/11 | 0/21 | Tilt shutdown at 31.3 s |
| left | 21/21 | 0/21 | Completed |
| right | 20/21 | 0/21 | Completed |
| yaw_left | 21/21 | 0/21 | Completed |
| yaw_right | 21/21 | 0/21 | Completed |
| mixed | 42/44 | 14/44 | Completed |

Gazebo backward instability triggered the runtime's excessive-tilt latch. This
reproduced at approximately 31.28 s in a fresh repeat (same tilt fault). This
is a failed readiness test even though the controller stopped before we continued
the trial to a physical fall. Its last recorded frame precedes the rejected sensor
frame. Other per-case stop reasons, missed steps, recovery skips, and leg-specific
kick counts are in `broad_summary.json` and each `kick_audit.json`.

The 90-second neutral run drifted backward about 0.36 m in Gazebo and 0.69 m in
MuJoCo. The earlier short neutral interval's small endpoint drift was not evidence
of long-term station keeping. Movement over a full phrase and travel-only velocity
are separate metrics: the latter excludes kick/hold/recovery by design. Any future
hold-drift reward should target stable support during the pause, rather than demand
travel or catch-up speed during it. A position-increment breakdown of the neutral
Gazebo run attributes +0.759 m to travel phases and -1.114 m to pause phases
(plus -0.009 m startup). MuJoCo has +0.002 m travel and -0.687 m pause displacement.
This makes hold-phase stability a separate priority from travel tracking.

In steady forward travel, mean measured vx was approximately 0.092 m/s in Gazebo
and 0.052 m/s in MuJoCo for a 0.10 m/s command. The report's tracking audits exclude
kick pauses, start-up, recovery and command-ramp transients.

## Sensitivity tests

Eleven single-factor MuJoCo perturbations plus an identical 40-second reference:
neutral 0–6 s, forward .1 m/s 6–30 s, neutral 30–40 s. Seed 42 for sensor noise.
No randomization optimization occurred; these are deterministic sensitivity probes.

| Perturbation | Strict chains | Correct / attempted steps |
|---|---:|---:|
| reference40 | 9/14 | 44/44 |
| armature_0 | 14/14 | 44/44 |
| armature_005 | 13/14 | 44/44 |
| armature_015 | 0/14 | 44/44 |
| damping_0 | 2/14 | 43/44 |
| damping_02 | 9/14 | 44/44 |
| mass_110 | 6/14 | 44/44 |
| gains_090 | 8/14 | 44/44 |
| gains_110 | 4/14 | 42/44 |
| sensor_noise | 8/14 | 43/44 |
| sensor_delay20 | 9/14 | 43/44 |
| action_delay20 | 9/14 | 43/44 |

Mass perturbation scales all link masses/inertias by 1.10. Gain perturbations scale
both motor Kp and Kd by 0.90 or 1.10. Noise is uniform encoder ±.002 rad, velocity
±.1 rad/s, gyro ±.02 rad/s, and foot force ±1 N (clipped nonnegative), injected
BEFORE the estimator and scheduler. Sensor delay holds sensor values back one
20 ms tick while preserving current controller time; it is a control-latency
probe, not an authentic timestamp/transport replay. Action delay holds targets
one 20 ms tick. All settings and loaded model readbacks are archived per case.

The same policy's armature sweep passes 14/14 at 0, 13/14 at .005, 9/14 at .010,
and 0/14 at .015. This demonstrates sensitivity inside the range used in training;
including a parameter range in randomization is not the same as passing its extremes.
None of these altered models is claimed to be the real robot's calibrated model.

## Robot-free shadow substitute

600 Gazebo sensor packets replayed through actual Quad-SDK ROS 2 message types,
Python worker and `/cpg_sim/` command output on localhost domain 77. Validated
command shape, finite bounds and exact 25/0.5 gains; sensor loss latched output off
and fresh messages did not rearm it. The test passed, with 602 output messages
because the timer and packet publisher are asynchronous. This is software replay,
not a robot hardware shadow test, firmware test, or latency calibration.

## Training, YAML and physical configuration

The exported `assets/settings.json` describes an EVALUATION setup, not the complete
training history. It says randomization=false, delay=0, armature=.01. The selected
model2499 actually comes from hold-transfer followed by walking rehearsal, with
randomization enabled; recorded physics readback spans approximately 0–.015 armature.
The later model2799 armature-focused training is not the selected policy. Preserve
this distinction when interpreting the exported metadata.

| Item | Audit finding | Retraining implication |
|---|---|---|
| Policy control/action contract | 50 Hz, 75 inputs, 12 outputs, .12 rad bounded residual on scheduled IK; gains 25/.5 | Matches current CPG path; no retraining needed for wiring alone |
| Joint order and defaults | Explicit SDK/Quad/policy mapping; nominal thigh .789, calf -1.58; shared deployment metadata | Mapping fixes belong in adapter, not in learned weights |
| Stand/planner YAML | Stand uses .8/-1.5 and 60/2; planner uses different stance gains/period | These are separate modes; copying them into CPG would change its plant |
| Nominal mass | Training URDF 16.087 kg; Gazebo 16.247 kg; actual payload unmeasured | About 1% model difference, within trained mass range; not alone a reason to restart training |
| Passive dynamics | Gazebo default sitting springs/damping differ from training and MuJoCo; sweep confirms sensitivity | Calibrate/model these explicitly before dynamics fine-tuning |
| Torque/speed limits | Training and MuJoCo use Go2HV 20.2/23.4 Nm envelope, knee at 13.5 rad/s, cutoff 30; Gazebo plugin uses fixed YAML limits 33.5/33.5/50 Nm and disables its speed model | Model definitions differ; reconcile before relying on Gazebo qualification |
| URDF/SDF joint limits | First-leg hip/thigh/calf position and nominal effort/speed values match each other; calf URDF speed 15.7 differs from all-joint training actuator cutoff 30 | Audit per-joint actuator limits against hardware, not just YAML names |
| Sensor processing | Training actor sees noise; estimator/scheduler use clean simulation state/contact. Runtime consumes actual packets | Needs upstream sensor/contact uncertainty, not just extra actor observations |

The torque-demand audit sampled nominal PD demand at 50 Hz. The checked normal
rollouts did not exceed the training envelope at those samples, so the torque
model discrepancy is not established as the cause of the observed kick failures.
This is not a full-rate measured-torque audit.

A hardware-specific code issue also needs attention: `RobotDriver` clamps the
computed `effort` telemetry field, while `UnitreeInterface` sends q/dq/Kp/Kd/tau_ff,
not that clamped scalar. Therefore this code path does not establish a software
limit on total motor PD torque. Firmware protections are separate and unverified
here. CPG hardware activation remains blocked; no safety changes were made during
this evaluation.

## Domain randomization and recommendation

Existing training coverage:

- Static/dynamic friction .4–1.0, restitution zero.
- Per-body mass/inertia scale .95–1.05.
- Motor Kp/Kd scale .9–1.1; armature 0–.015.
- Actuator command delay 0–1 physics step, i.e. 0–5 ms, not 0–20 ms.
- Actor noise: gyro ±.2 rad/s, gravity components ±.05, joint position ±.01 rad,
  joint velocity ±1.5 rad/s. Initial x/y/yaw variation; flat ground.

Main gaps: sensor bias/noise entering the estimator and scheduler; foot-force
scale/offset/threshold and latency; correlated dropouts and transport jitter;
payload COM variation; passive damping/friction variation; nonzero disturbance
pushes. Flat, firm ground remains the target; extreme terrain is not required to
address these gaps. Add bounded ranges progressively and evaluate held-out extremes
and combinations rather than selecting solely on mean training reward.

Recommendation: retain this 75-input policy as a regression baseline and plan a
robustness fine-tune after aligning the software plant definitions. An 81-input
actor (+ estimated vx/vy and four measured contacts) is worth an equal-budget
comparison, using the same noisy estimator/filtering in training and deployment.
It is not a substitute for fixing dynamics, latency and upstream contact handling.
The current results do not support full-direction hardware dancing. No new
training was authorized or launched during this task.

## Evidence

`logs/cpg_broad_sim_20261101/`: per-case traces, models, launches, kick/tracking
audits, ROS replay, configuration audit, source snapshots and checksums. Scripts
and scenario manifests are preserved. `config_audit.json` separates selected
checkpoint training from export evaluation settings. Ground truth is for audits
only, never injected into the actor. All tests are local simulation/replay.

![Broader simulation checks](../logs/cpg_broad_sim_20261101/comparison.png)
