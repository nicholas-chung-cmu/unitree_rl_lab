# Go2 CPG-guided residual controller

This is a separate simulation controller, registered as `Unitree-Go2-CPG-*`.
The phase-conditioned and hierarchical tasks and their checkpoints are preserved.
The session evidence and selected checkpoints are in
[`reports/conga_cpg_overnight.md`](../reports/conga_cpg_overnight.md).

## Control path

1. A deterministic scheduler dispatches individual diagonal STEP, front-leg KICK,
   airborne HOLD, and recovery states. It keeps the musical clock running through
   failures and uses measured contact feedback to admit and complete skills.
2. A Cartesian reference generator constructs scheduled stance and swing paths.
   Quintic interpolation gives zero endpoint velocity; a squared-sine clearance
   profile provides the STEP lift. A raised front foot can enter the next STEP
   directly, while the matching hind foot performs its ordinary swing.
3. Analytic Go2 inverse kinematics produces nominal joint positions, checks forward
   kinematics error, and clamps against the articulation's soft joint limits.
   A bounded gravity-load position bias compensates the stock compliant actuators.
   Three-foot support can additionally allocate the required support wrench and
   damp body motion estimated from encoders, contact states, and IMU measurements.
4. PPO supplies bounded corrections:
   `q_target = clamp(q_nominal + 0.12 * tanh(policy_output), joint_limits)`.
   The stock position gains (25/0.5) and motor limits remain in use.

An optional exact sagittal-symmetry actor averages a shared MLP with its reflected
prediction: `f_s(x) = (f(x) + R_action f(R_observation x))/2`. It preserves the
75-input/12-output interface and is differentiable through both evaluations.
Reflection handles polar/axial vectors, joint signs, diagonal phase and raised-foot
flags. It requires unnormalized actor observations. Checkpoints using it must be
loaded with `equivariant_actor: true` through `cpg_experiment.py`; saved actor keys
have the `actor.base` prefix. JIT/ONNX exports include the complete wrapper.
Those exports contain the residual actor; the scheduler, IK and observation
construction remain separate controller components.

The simulator's joint order is hip FL/FR/RL/RR, thigh FL/FR/RL/RR, then calf
FL/FR/RL/RR. Geometry helpers use leg-major FL/FR/RL/RR triples; the adapter performs
the explicit mapping. Coordinates are body x forward, y left, z up. Foot targets
refer to spherical foot centers; sole clearance subtracts the 0.022 m radius.
The tested ground is a flat plane.

The local URDF gives a neutral whole-body center of mass near
`(-0.0008, 0, -0.0198) m`, with total mass 16.087 kg. Simulator measurements confirm
the near-zero fore-aft offset. Earlier dance experiments used an incorrect
`(-0.025, 0, 0)` approximation; retain their saved settings/source when reproducing
those results. `support_com_x/z` expose the calibrated model values.
`kick_shift_back/side` optionally add a planted body transfer in the existing gap
before KICK. The tested candidate uses 3 cm backward and 1.5 cm toward the support
side, retains the bias through the raised entry, and restores it after four-foot
support returns. The kick's body-frame endpoint remains unchanged. Calibration and
reference reachability alone do not establish physical hold success.

The actor has 75 inputs: angular velocity, projected gravity, joint positions and
velocities, previous actions, nominal joint targets, gait and musical phases,
expected contacts, operator velocity/yaw, selected feet, time remaining, kick
state, and preparation/recovery state. Its 12 outputs are residual joint actions.
The 90-input critic additionally receives body linear velocity and joint torques.
Actual simulator foot positions/velocities, ground height and body linear velocity
are used for rewards and evaluation, not directly in actor observations. Contact
sensors feed the deterministic scheduler. Hardware use would require an appropriate
contact estimator, calibrated geometry/mass, observation adapter and recovery
validation; the exported metadata explicitly marks hardware compatibility false.

## Final motion specification

Tempo is fixed at 90 BPM (2/3 s per beat), independently of travel commands.
At zero velocity the robot continues marching. Speed changes stride displacement,
not cadence; the hard horizontal vector cap is 0.5 m/s. Initial command training
uses at most 0.15 m/s and 0.25 rad/s yaw. Backward, lateral, moving-turn and in-place
turn commands are supported. Broader ranges require separate qualification.

Command changes use a planned velocity with bounded acceleration: 0.075 m/s²
for the XY vector and 0.15 rad/s² for yaw. This deliberately slows reversals while
preserving the musical clock. The actor and velocity rewards use planned velocity;
evaluation traces distinguish it from requested operator velocity. Initial episode
commands are established during settling. Walking integrates stance positions and
fixes swing endpoints at dispatch, preventing command changes from teleporting
foot targets. Both continuity and acceleration bounds are required: unbounded
integration can otherwise demand unreachable stance positions. The tested bounds
do not establish feasibility for every possible command/terrain combination.

| Phrase event | Target time from phrase start | Dispatch/reference behavior |
|---|---:|---|
| STEP 1 touchdown | 0 s | Dispatch 0.20 s before touchdown |
| STEP 2 touchdown | 0.6667 s | Opposite diagonal |
| STEP 3 touchdown | 1.3333 s | Opposite diagonal again |
| KICK begins | 1.6667 s | FR and FL alternate between phrases |
| KICK endpoint | 1.9167 s | 8 cm outward, 5 cm sole clearance |
| Next STEP begins | 2.4667 s | Contains the raised front leg |
| Next phrase touchdown | 2.6667 s | Direct raised-leg landing |

STEP uses a 0.20 s swing and 6 cm peak reference clearance after deterministic
trajectory validation. Startup has a one-beat pre-roll following measured settling.
The 0.25 s KICK reference ends with zero velocity and retains its airborne endpoint.
There is no automatic retraction or intermediate landing. Travel is paused during
kick/hold; the subsequent stride does not compensate for the lost travel distance.
The kick endpoint's fore-aft coordinate is fixed at calibrated nominal even when
the operator requests travel or yaw; a regression test prevents stride offsets
from leaking into this endpoint. Isolated HOLD also keeps travel paused through
musical-clock wraps.

The musical schedule leaves **0.75 s from kick endpoint to the next touchdown**.
With the 0.20 s landing swing, this means **0.55 s stationary reference hold**.
A full 0.75 s stationary hold followed by a 0.20 s STEP would miss that beat.
`Kick-Hold` separately tests longer stationary holds without claiming they fit the
musical phrase. The normal choreography and 90 BPM clock are unchanged.

## Contact contracts and recovery

Contact detection uses 5 N/2 N hysteresis and 40 ms confirmation. A verified swing
requires at least 40 ms without contact and more than 8 mm sole clearance. Each
STEP must have exactly one verified swing and landing per selected foot, retain
the opposite support pair, and land within the configured timing window. Walk
starts with 100 ms tolerance; Timing and Dance use 60 ms. Contact onset timestamps
are retained before confirmation, so confirmation does not shift touchdown time.

Unadmitted opportunities count as misses. Completing a reference trajectory is
never sufficient for physical success. The independent contact auditor includes
every expected opportunity rather than only attempted steps. Kick audits separately
check planar/height endpoint accuracy, foot speed, support, body stability, endpoint
timing, airborne hold quality and direct post-kick STEP success.

Default recovery is `land_and_rejoin`: gradually return foot targets toward nominal
support, then rejoin a future scheduled event after measured support. The clock and
physical episode age continue. Configurable `land_and_stop` and `hold_and_stop`
variants are provided but have not received physical qualification. The latter is
a reference-hold option, not a demonstrated safe response to every failure.

Initialization uses nominal IK joints and a grounded base height, then inhibits
residual actions until at least 1 s has elapsed and measured support, tilt,
encoder-derived body velocity and IMU angular velocity remain stable for 0.25 s.
A 5 s preparation timeout ends an invalid episode. No raised-state teleport is
used. Physical timeouts are never reset at skill boundaries.

## Configurable curriculum

| Registered suffix | Purpose |
|---|---|
| `Walk` | Low-speed diagonal stepping, including zero-command marching |
| `Timing` | Same gait with 60 ms touchdown window |
| `Travel` | Qualified broader XY/yaw range, default 0.20 m/s and 0.30 rad/s |
| `Dance` | Three STEP events, alternating kicks, hold and direct post-kick STEP |
| `Kick-Hold` | Isolated extended hold qualification/training |
| `Bounce` | Dance with 0.0075 m amplitude (1.5 cm peak to peak) |

Registration is functionality, not evidence that a stage has been learned. Advance
only after reading the held-out contact and stability audits. Keep the best walking
checkpoint independently if later dance training loses command robustness.

`extended_hold_fraction` optionally assigns a bilateral subset of training
environments to prolonged zero-velocity holds while the rest keep dancing. It
defaults to zero and does not change the normal musical schedule. This curriculum
was tested separately; its results must not be substituted for musical validation.
Hold reward variants and support damping gains are saved in each job's settings.
Bounce changes the reference body height by a beat-synchronous sinusoid. The
height cost tracks that sinusoid; it retains the original constant target when
bounce is zero. `scripts/audit_cpg_bounce.py` measures actual beat-frequency
amplitude, phase and height error separately from contact success. A configured
1.5 cm peak-to-peak reference does not prove an exact 1.5 cm physical bounce.
The current Dance defaults use the calibrated support model, planted transfer,
6/s translational damping, endpoint hold-phase freeze and hold-quality reward.
All current CPG task defaults include the qualified command-acceleration bounds;
walking additionally enables `integrated_walk_reference`. Earlier saved policies
can retain their original behavior through explicit complete settings. The selected
walking policy required no PPO retraining for this reference/command correction.
Use the selected checkpoint's complete settings file. Earlier experiments require
their archived source/configurations; several partial historical settings relied
on defaults that were subsequently corrected.

## Reproduction and persistence

Use `/home/rml2/anaconda3/envs/nick_sillywalk/bin/python`; no shared Isaac Lab files
or environment packages are modified. Example evaluation (supply the selected
checkpoint and its corresponding settings from the report):

```bash
/home/rml2/anaconda3/envs/nick_sillywalk/bin/python scripts/cpg_experiment.py \
  --headless --task Unitree-Go2-CPG-Timing --num_envs 64 --steps 3100 \
  --seed 9043 --checkpoint CHECKPOINT --settings SETTINGS --output NEW_OUTPUT
/home/rml2/anaconda3/envs/nick_sillywalk/bin/python scripts/analyze_cpg_contacts.py \
  NEW_OUTPUT --swing .2 --tolerance .06
```

For Dance also run `scripts/audit_cpg_kicks.py NEW_OUTPUT` and add `--dance` to the
contact auditor. Use `--video --num_envs 1 --steps 1000` for a 20 s recording.
For the isolated hold task use `scripts/audit_cpg_hold.py NEW_OUTPUT`; it audits
the first attempt per environment over a complete 0.76 s sampled interval and
reports continuous strict validity separately from the 80%-quality hold gate.
Optional `evaluation_command_period_s` rotates each environment through the
configured command list without resetting the robot or musical clock. Such
results are grouped by initial rotation offset, not mislabeled as fixed-command
performance. The original fixed-command evaluation is the default.
`--component-tests` runs core/adapter tests inside the required simulator process.
Read `status.json` as well as process exit status; simulator shutdown can mask errors.

`scripts/run_cpg_session.py --session SESSION --plan PLAN` runs immutable named
jobs sequentially, saves source/config/seed/Git snapshots and periodic checkpoints,
and enforces the session's original four-hour deadline. Run it in tmux. Completed
jobs are skipped when resuming an unfinished session; failed jobs are never retried
implicitly. A session marked complete is closed, protecting its final report from
being overwritten by a restart. A global GPU lock prevents overlapping CPG runners.
Resume preserves optimizer state and
restores the adaptive learning-rate scalar from the saved optimizer. Training uses
4,096 environments and 24-step rollouts. Evaluation is a separate sequential job.

The 2026-11-01 session began conservatively at 06:45 UTC and has an immutable
10:45 UTC deadline. Do not reuse that session to extend the authorized budget.

Selected replay manifests are `selected/walking.json`, `selected/dance_travel.json`
and `selected/dance_bounce.json` within `logs/conga_cpg_session_20261101/`. They
point to complete settings, checkpoint paths, qualifications and final source hashes.
Use the no-bounce dance as a retained alternative; the full-bounce policy has a
measured 12.28 mm beat-frequency height component versus 15 mm requested, with
about 58 ms phase lag. Stable contacts do not mean exact body-height tracking.
