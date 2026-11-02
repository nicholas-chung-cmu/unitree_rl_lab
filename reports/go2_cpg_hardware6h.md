# Go2 conga hardware preparation — 2026-11-02

**Keep the prior timing+300 policy.** The additional training did not improve
transfer reliably. The useful result of this session is a complete, guarded
Quad-SDK hardware test path, stronger verification, a restricted command envelope,
read-only calibration/shadow tools, and a portable handoff. No real robot was
connected or commanded; this is not hardware qualification.

Start with [the operator guide](../docs/go2_cpg_hardware.md). The portable export
is `exports/go2_cpg_hardware_prep_20261102/`, with a native Quad-SDK source patch,
policy/runtime, calibration template, videos, evidence and checksums. The user's
six-hour authorization runs 06:06:45–12:06:45 UTC; final elapsed time and cleanup
are recorded in the session's `final_status.json`. Preparation finished in 111.2
minutes, within that budget; no autonomous jobs remain. No extension was used.

## Selected motion and policy

- Original checkpoint: `logs/cpg_targeted45_20261102/timing_train2/training/model_299.pt`.
- Actor remains 75 observations; no 81-input retraining or hardware-only actor inputs.
- ONNX SHA-256: `bfe0617876fd09abc972161193004974975f402927119023e707ac0037df3051`.
- 90 BPM; scheduled diagonal landings, three steps then alternating front kicks.
  Cartesian/IK reference plus bounded ±0.12 rad residual; no bounce.
- Final commands: **0–0.10 m/s forward, no lateral/backward, ±0.10 rad/s yaw**.
  Neutral/stale velocity keeps dancing. Deliberate stand and emergency damping
  are distinct controls. Kick/hold pauses remain excluded from velocity tracking.
- Nominal joint PD remains 25/0.5. The hardware sink applies an 18 Nm sampled
  demand cap and motor speed envelope. Final MuJoCo trials explicitly enable the
  same sampled cap; earlier uncapped comparisons are retained separately.

The attempted turn-dependent forward cap of 0.08 m/s did not improve sustained
turning and was rejected. The optional envelope feature remains available but is
not selected. Do not infer that every smaller command necessarily performs better.

## Motion evidence

| Evaluation | Physical steps (audit v2) | Strict kick chains | Falls/faults |
|---|---:|---:|---|
| Final MuJoCo: 12 × 90 s, both initial diagonals, neutral/forward/turn/combined | 1,196/1,200 | 394/396 | 0 / 0 |
| Native Gazebo through rebuilt Quad-SDK: four 90 s + one 120 s trials | 534/534 | 176/176 | 0 / 0 |
| Final MuJoCo perturbations: nine 90 s trials | 884/900 | 290/297 | 0 / 0 |
| Strong randomized Isaac: three seeds × 64 environments × 120 s | 25,369/25,689 | 8,167/8,448 | 0 falls |

Gazebo broad tests included commands up to 0.12 m/s and ±0.15 rad/s; the final
matched video uses the smaller selected envelope. The final nominal MuJoCo
misses were confined to combined forward/turning trials. All 800 opportunities
in neutral, straight forward and turning in place passed. The misses include
brief opposite-support loss around post-kick landing, not a failure to lift the
selected pair. No thresholds were relaxed to improve the scores.

Audit v2 preserves ±60 ms touchdown tolerance while extending the opposite-support
check through 100 ms after the beat. This catches the previously identified
contact loss near +67 ms that the older 60 ms window could miss. Counts include
scheduled opportunities suppressed by the controller. V2 can omit an incomplete
final opportunity, so its denominator can differ slightly from v1. For the
randomized Isaac set, v1 was 25,376/25,696; v2 is shown above.

A strict kick chain requires endpoint geometry/height/speed/support, timely arrival,
a stable airborne hold, and the next diagonal landing. Controller counters alone
are not qualification. Ground-truth geometry/velocity are used only for offline
physics audits, never as deployable actor inputs.

### Perturbations at the final sampled torque limit

| MuJoCo condition | Steps | Strict kicks |
|---|---:|---:|
| armature_005 | 99/100 | 32/33 |
| armature_015 | 100/100 | 32/33 |
| mass_110 | 100/100 | 33/33 |
| gains_090 | 100/100 | 33/33 |
| gains_110 | 95/100 | 28/33 |
| sensor_noise | 100/100 | 33/33 |
| sensor_delay20 | 96/100 | 33/33 |
| action_delay20 | 100/100 | 33/33 |
| combined | 94/100 | 33/33 |

Each trial commands 0.10 m/s during seconds 6–80 and neutral otherwise. Sensor
noise is ±0.002 rad encoder, ±0.1 rad/s joint velocity, ±0.02 rad/s gyro and ±1
contact-force unit. Delays are one 20 ms policy tick; this deliberately delays
values with a current logical controller timestamp to test dynamics, separately
from the transport stale-stamp tests. The combined case adds 5% mass, 5% lower
gains, sensor noise, 20 ms sensing and 20 ms actuation delay. Mass/inertia and
armature edits are confined to generated diagnostic models with readback saved.

The 10% higher-gain condition degraded strict kicks most. This is evidence to
measure real gain/lag response, not a reason to turn up gains on the robot.
Strong Isaac tests additionally include randomized contact/material, mass,
armature, gains, passive dynamics, COM, sensor offsets/noise and disturbances;
exact settings are preserved in `heldout_settings.json`.

## Matched videos

- [MuJoCo, final cap, 60 seconds](../logs/cpg_hardware6h_20261102/videos/mujoco_final/dance.mp4)
- [Native Gazebo through Quad-SDK, 60 seconds](../logs/cpg_hardware6h_20261102/videos/gazebo_final/dance.mp4)

Both use the same sequence: neutral 0–6 s; +0.10 m/s 6–22 s; +0.08 m/s with
+0.10 rad/s yaw 22–32 s; +0.08 m/s with −0.10 rad/s yaw 32–42 s; neutral
thereafter; deliberate standing requested at 46 s. These are actual simulator
renders with a fixed camera, not a reconstruction of one simulator in another.
Gazebo was run at a slower wall-clock rate for capture; playback is 1× simulation
time at 20 fps. Its largest recorded camera gap was 100 ms; missing frame slots
repeat the preceding image. MuJoCo is 50 fps. Earlier takes remain archived.

| Video measurement | MuJoCo | Gazebo |
|---|---:|---:|
| Net forward displacement | 1.714 m | 1.969 m |
| Net lateral displacement (turns were commanded) | 0.149 m | 0.240 m |
| Active-travel speed at +0.10 m/s | 0.079 m/s | 0.092 m/s |
| Steps | 52/52 | 52/52 |
| Strict kick chains | 16/17 | 17/17 |
| Standing reached | 47.28 s | 47.98 s |

MuJoCo's one strict miss was a late FL endpoint; it still held and completed the
post-kick step. The controller reported 17 successes, which is why independent
truth auditing matters. Both videos show the complete stand transition and more
than 12 seconds afterward. Net displacement includes pauses; active-travel speed
excludes kick/hold/recovery and is not inflated to compensate for them.

## Training review

Three branches used 4096 environments and **1,900 additional PPO updates**,
186,777,600 training transitions, plus a separate 2-update/32-environment smoke.
The baseline and all old exports/checkpoints remain intact. Fixed learning rates
were 3e-5 for the restricted branch and 1e-5 for conservative/sustained branches.

Restricted training removed lateral/backward sampling, retained neutral and
turn-in-place examples, and added mild support-margin/clearance emphasis.
Conservative training retained the prior rewards at a lower learning rate.
Sustained training used 120-second episodes, 30-second command holds and a mild
support-force margin penalty to expose long-run landing support failures. The
penalty excludes the selected airborne leg; it does not reward keeping it planted.

| New checkpoint | MuJoCo horizon per case | Steps, v2 | Strict kicks |
|---|---:|---:|---:|
| restricted_150 | 40 s | 348/352 | 110/112 |
| restricted_300 | 40 s | 342/352 | 109/112 |
| restricted_600 | 40 s | 339/352 | 109/112 |
| conservative_100 | 40 s | 337/352 | 107/112 |
| conservative_300 | 40 s | 345/352 | 111/112 |
| sustained_200 | 90 s | 784/800 | 260/264 |
| sustained_500 | 90 s | 784/800 | 256/264 |
| sustained_1000 | 90 s | 773/800 | 255/264 |

The matched prior-policy 90-second/eight-command comparison was **796/800 steps,
262/264 strict kicks**. All sustained checkpoints had worse aggregate transfer;
the final +1000 checkpoint also regressed straight forward (84/100 steps under v2; 92/100 under v1,
26/33 kicks). Nominal Isaac evaluations remained strong, demonstrating that more
updates and higher training return are insufficient selection criteria. No new
checkpoint was promoted based on its iteration number or reward alone. The
persistent runners checkpointed each segment, recorded source manifests, enforced
the original wall-clock deadline and reserved its final hour for handoff.

## Integration and fault handling

The driver previously only clamped effort telemetry while transmitting unchanged
PD gains/targets. The CPG path now scales the **transmitted** gains/feedforward
against measured joint state. Both the C++ sink and Unitree interface enforce the
sampled demand bound. This is not a physical torque guarantee between samples;
logs labeled estimated PD demand use nominal gains and are not measured torque.

Raw joints, IMU, contact readings, source tick and receipt stamp now travel in one
`CpgSensorFrame`. Repeated MCU samples cannot refresh the CPG stream. The adapter
checks source/reception age, ordering, synchronization and plausible encoders,
then runs the complete worker at 50 Hz. Accumulated control-clock drift is limited
to 20 ms, preventing sustained timer slowness from changing tempo. Worker replies have a bounded 15 ms read;
partial JSON lines cannot bypass that timeout. Targets retain the sensor stamp.

The driver starts CPG hardware in SAFETY and needs an explicit READY request.
Standing and sitting ramp from measured joint positions. A deliberate stand
request finishes a stable landing and holds the last joint target; sensors remain
validated. Stale velocity continues neutral dancing; stale sensors/targets or
invalid data latch damping. There is no automatic rearming. CPG SAFETY and fault
handlers consistently use Kp=0, Kd=0.5. A separate interface watchdog checks MCU
and command freshness; it cannot cover process/OS/network/power failure.

Hardware launch checks the verified profile, actor hash, ROS/custom-message
imports, worker startup and trace path before the vendor interface can release
Sport ownership. The shipped profile is disabled. Read-only raw capture,
per-foot calibration drafting, offline preflight, actual ROS shadow and Unitree
wireless ROS input are included. A requests deliberate standing; B requests
latched SAFETY. These custom controls do not replace independent robot protection.

## Verification and practical limits

- Native desktop build: `quad_msgs`, `quad_utils`, `robot_driver` passed.
- 109 CPG/hierarchical/original-conga component tests; 29 Python deployment,
  calibration, command and audit tests; 4 Gazebo force-decoder tests passed.
- 8 C++ command-sink/gate tests and 4 coherent-sensor tests passed.
  100,000 randomized sampled PD/feedforward envelope checks passed.
- Five actual ROS shadow fault cases (600 requested frames each) passed:
  stream loss, frozen timestamps, frozen MCU ticks, future stamps and nonfinite
  state. Neutral stale input continued the dance; faults stayed latched. Desktop
  observed worker maxima ranged up to 11.12 ms during concurrent testing. The
  final five-case replay with the accumulated clock guard passed with maximum
  phase error 0.156 ms and maximum worker response 3.80 ms.
- Five activation-rejection checks passed before vendor initialization, including
  unverified profile, missing worker and occupied trace. Positive CPG attestation
  testing was constructor-only; it never created a Unitree motor interface.
- Six MuJoCo deliberate stops, including raised-leg phases, passed: standing in
  1.32–2.16 s, all four contacts afterward, frozen targets.
  Gazebo standing took 1.783 s; SAFETY damping appeared in 2 ms and target-loss
  damping in 61 ms in the final native checks. Emergency damping is not upright
  recovery and is not included in the no-fall motion claims.

The desktop preflight replay processed 602 packets with a 3.72 ms maximum worker
response; the final relocated export passed again at a 3.12 ms maximum. The
Quad-SDK patch applied cleanly to the recorded base and all bundled payload
checksums matched. This is **not Orin timing qualification**. The exact onboard model,
Ubuntu/ROS/JetPack versions, native build, raw contact calibration, physical
motor response, payload/COM and independent stop remain unverified. Both sims
are useful evidence; neither proves that the real robot will behave identically.
The operator guide allocates the four-hour morning window around those checks
and preserves the existing walking setup as rollback.

Initial test-harness failures are retained rather than erased: repeated NPZ
field decompression slowed an early ROS replay; integer command values violated
ROS Python float-field checks; and one copied cap-evaluation runner targeted
existing output directories (MuJoCo refused them). Corrected reruns and the
invalid-attempt labels are saved. They are not counted as passed motion trials.

Source evidence: [session summary](../logs/cpg_hardware6h_20261102/summary.json),
[final envelope](../logs/cpg_hardware6h_20261102/capped_envelope_results.json),
[perturbations](../logs/cpg_hardware6h_20261102/capped_heldout_mj_results.json),
[Gazebo](../logs/cpg_hardware6h_20261102/baseline_gz/broad_summary.json), and
[hardware guide](../docs/go2_cpg_hardware.md). Full traces and all rejected
checkpoints remain in the source workspace; the bundle carries compact evidence.
