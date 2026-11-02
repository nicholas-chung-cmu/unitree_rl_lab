# CPG robustness comparison and deployment interfaces

> Historical experiment/port notes. For the 2026-11-02 hardware test candidate,
> current controls and remaining qualification checks, use
> [the hardware handoff](go2_cpg_hardware.md). Earlier policy results and
> simulation-only restrictions below describe those earlier versions.

The original phase-conditioned, hierarchical, CPG checkpoints and exported bundles
remain separate. The 2026-11-01 robustness comparison fine-tunes the experimental
traveling-dance model2499 in `logs/cpg_robust_compare_20261101/`.

## Observation contract

The existing 75 values retain their order and units. The optional version appends:

| Indices | Values | Source |
|---|---|---|
| 75–76 | Body-frame vx, vy, clipped to ±1 m/s | Contacting-foot encoder kinematics and IMU gyro |
| 77–80 | FL, FR, RL, RR measured contact flags | Force hysteresis, 5 N on / 2 N off |

`cpg_residual.observation_dim` in deploy.yaml declares 75 or 81. Old metadata without
that key defaults to 75. Runtime verifies the ONNX input dimension against it.
The critic remains 90-dimensional and may use privileged simulation state.
Both actor versions output 12 bounded joint-position residuals on scheduled IK.
The six new input weights initialize to zero; all old actor/critic weights are
copied. Optimizer state is reset for BOTH arms at the same learning rate. The
reflection transform flips vy and swaps measured left/right contact flags.

The estimated velocity is not true simulation body velocity. It uses
`-mean(J(q) qdot + omega × foot_position)` over measured contacting feet. Airborne
feet are excluded; no supporting feet yields zero. Real force thresholds and
calibration remain unverified. No new root-height or foot-clearance truth enters
the actor.

## Sensor uncertainty and physical truth

Training uses one measured sensor packet for both the scheduler/estimator and the
actor. Noise is injected before contact detection, support estimation, kick/hold
monitoring and reference load compensation. Physical robot tensors are untouched
for rewards and independent evaluation. The critic remains privileged.

The first 250 PPO updates use milder noise and no sensor-value delay. The next
750 updates use these bounds:

| Quantity | Uncertainty |
|---|---|
| Encoder position | Uniform noise ±.002 rad; episode bias ±.001 rad |
| Encoder velocity | Uniform noise ±.1 rad/s |
| Gyro | Uniform noise ±.02 rad/s; episode bias ±.01 rad/s |
| Projected gravity | Component noise and bias ±.005, then normalized |
| Foot force | Noise ±.5 N; episode bias ±2 N; scale .85–1.15 |
| Sensor delay | 0 or 1 control tick (0–20 ms), sampled per episode |
| Sensor packet hold | Probability .001 per tick, keeping the prior packet |
| Individual contact dropout | Probability .001 per foot per tick |
| Actuator command delay | 0–4 physics ticks (0–20 ms) |

Delay histories reset per environment and cannot retain samples from an earlier
episode. Deployment does not inject synthetic noise: it consumes actual packets.
The sensor model is a bounded engineering hypothesis, not measured hardware data.
Contact qualification uses separately logged physical contact force/hysteresis
where available; the noisy controller's own success counters are diagnostics.

## Actuator profile and physical randomization

The `go2hv_mujoco_v1` profile uses motor PD 25/.5 and the existing Go2HV envelope:
20.2 Nm motoring / 23.4 Nm braking, full torque through 13.5 rad/s, tapering to
zero at 30 rad/s. It intentionally remains conservative on calf torque compared
with the URDF's larger nominal limit. Per-joint hardware limits need calibration.
Nominal armature is .01 kg m²; passive joint damping .1 Nm s/rad and Coulomb
friction .2 Nm. Isaac uses a smooth tanh friction approximation; MuJoCo uses its
native frictionloss. Matching these numbers does not make the engines identical.

Physics randomization includes material friction .4–1, per-body mass/inertia
scale .95–1.05, motor gain scale .9–1.1, armature .005–.015, Coulomb friction
.15–.25, passive damping .075–.125, and base COM offsets x±8/y±5/z±5 mm. Small
horizontal velocity disturbances (±.05 m/s) occur every 12–18 seconds. Flat ground
remains the task surface. No rough-terrain curriculum was introduced.

Quad-SDK's matching Gazebo option is:

```bash
cpg_actuator_profile:=go2hv_mujoco_v1
```

It generates a separate model/config with springs zero, damping .1, friction .2,
and enables the same speed-dependent torque envelope in the simulation plugin.
The default is `legacy`; old source SDF/YAML files are unchanged. Explicit
`cpg_disable_springs` / `cpg_disable_damping` diagnostic flags still override the
corresponding passive term. Gazebo's effective rotor inertia is not reconciled by
this flag, and engine friction implementations can differ. Hardware is rejected.

## Rewards and experiment protocol

A separate `hold_stationary` reward encourages low horizontal body velocity and
yaw rate during HOLD, regardless of travel command. Travel velocity/yaw tracking
continues to exclude kick/hold/recovery; no catch-up acceleration is requested.
The original beat clock, contact-based completion, kick geometry and post-kick
landing behavior remain unchanged.

Each arm receives 1000 PPO updates with 4096 environments × 24 steps, using the
same stage seeds, initial policy and curriculum. Checkpoints at 250 and 1000 are
evaluated on the same eight commands. A two-hour wall budget covers the main
comparison and queued longer tests. The runner owns a GPU lock, checkpoints and
status files; it refuses to overwrite a started experiment.

`runner_status.json` and `posteval/status.json` report live/completed work. Selection
within each arm considers failures, worst-command step/kick completion, total kick
completion and neutral drift; it does not automatically choose the latest file.
Held-out checks include longer neutral/backward/changing-command runs, dynamics
and sensor perturbations, and the complete Gazebo/Quad-SDK path. One training seed
per arm supports an engineering comparison, not a statistical superiority claim.

No low-level hardware publisher is enabled by these changes. In particular, the
previous audit's airborne recovery and total motor-PD torque limiting concerns
remain hardware-readiness work. See the final comparison report for measured
results before choosing a deployment candidate.

## Targeted 45-minute follow-up (2026-11-02)

Optional `training_backward_fraction` (default .5) changes the sign distribution
of sampled translation; `training_turn_fraction` (default0) creates pure-yaw
commands. They apply only to training command sampling, not external joystick
commands. Legacy defaults preserve samples and random-number consumption.

Two supplemental measured-policy reward terms default to weight zero:
`kick_stationary` rewards low physical body XY velocity (scale .06m/s) and yaw
rate (.2rad/s) during KICK; `neutral_stationary` uses a tighter .025m/s XY scale
during travel only when the full filtered command norm is below .005. They do
not change velocity targets or add kick time to travel tracking. Actor inputs
remain deployable; physical body velocity is a reward/critic-only measurement.

See `reports/go2_cpg_targeted45.md` for the controlled checkpoint comparisons,
post-kick support-unloading diagnosis and exact selected training weights.
Existing exported bundles and hardware blocking remain unchanged.
