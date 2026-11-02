# Conga through Quad-SDK in Gazebo

The `cpg` controller runs a metadata-declared **75- or 81-observation** CPG policy and reference
runtime through Quad-SDK's normal joint-command path. This launch is simulation-only.
A separate, explicitly gated hardware test path is documented in
[cpg_hardware.md](cpg_hardware.md); hardware calibration/timing remain unverified.
The later robustness experiment adds the optional 81-input actor; the original integration and policies remain selectable.

## Build

In a terminal with ROS Jazzy and the isolated workspace overlay sourced:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install_nick_conga/local_setup.bash
cd ~/ros2_ws
colcon --log-base log_nick_conga build \
  --base-paths src/quad-sdk \
  --build-base build_nick_conga --install-base install_nick_conga \
  --symlink-install --packages-select quad_utils robot_driver gazebo_plugins \
  --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release
```

The conga repository must contain its exported bundles and `.local/unitree_ros`
Go2 URDF. Its Python environment needs Torch, NumPy, PyYAML and ONNX Runtime.
ROS uses system Python 3.12; the policy runs in the separate existing Python 3.11
worker. `policy_python:=...` overrides the interpreter path.

## Launch and command

Use a normal desktop terminal with `DISPLAY` and `XAUTHORITY` intact. In every
terminal used for this simulation:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install_nick_conga/local_setup.bash
export ROS_DOMAIN_ID=87
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export GZ_PARTITION=nick_conga_demo
```

Stop previous simulations in this same domain/partition before launching another.
The dedicated launch defaults to the walking-only CPG candidate:

```bash
ros2 launch quad_utils quad_cpg_gazebo.py \
  conga_root:="$HOME/Documents/nicholas_chung/go2-conga/unitree_rl_lab" \
  trace:=/tmp/conga_walk.jsonl
```

Wait for both joint controllers to report activation and for the robot to settle.
In another terminal with the same environment, enter READY:

```bash
ros2 topic pub --once /robot_1/control/mode std_msgs/msg/UInt8 '{data: 1}'
```

The runtime completes measured startup settling and begins marching at 90 BPM.
For forward travel:

```bash
ros2 topic pub --rate 50 /robot_1/cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.1}, angular: {z: 0.0}}'
```

Ctrl+C stops this command stream. After 0.25 simulation seconds without a fresh
command, travel becomes zero **and marching/dancing continues**. This is different
from the basic walking controller's stale-command behavior.

To test the experimental traveling dance, stop the current simulation and relaunch
with this extra argument:

```bash
selection:="$HOME/Documents/nicholas_chung/go2-conga/unitree_rl_lab/exports/go2_cpg_demo_dance_experimental_20261101/assets/selection.json"
```

The preserved original dance is selectable through
`exports/go2_cpg_quad_sdk_candidate/assets/selection.json`, but its holds failed
this Gazebo evaluation. The experimental policy performed better here; neither
is hardware-qualified. Use `gui:=false rviz:=false` for headless evaluation.

Inspect status with:

```bash
ros2 topic echo /robot_1/cpg/status
```

The configured trace contains sensor packets, requested velocity, targets,
contact-event metrics and separately labeled simulator ground truth.

## Architecture and failure behavior

- Raw `/robot_1/joint_states`, `/robot_1/imu`, and four contact topics feed the
  adapter. It never injects simulator root position/velocity into the actor.
- The shared runtime supplies the scheduler, trajectories, IK, the declared observation vector,
  ONNX inference and bounded residuals at 50 Hz. Gains remain 25/0.5.
- `/robot_1/cpg/target` carries validated joint targets to the C++ `CpgController`.
  `robot_driver` remains the sole publisher on `control/joint_command`, preserves
  its sit/stand/READY state machine and sends commands through the existing
  Gazebo motor-limit/PD plugin.
- The adapter waits for the driver's actual `state/control_mode == 1`, rather
  than starting the choreography during the standing transition.
- Stale joint/IMU data, nonfinite inputs, excessive tilt, missed ticks and worker
  failures latch the adapter off. Leaving READY after starting also latches it.
- Missing targets for 0.1 simulation seconds or malformed commands latch the C++
  sink off. It then uses the existing **simulation standing fallback**. This is
  not a validated real-robot airborne recovery behavior.
- Restart the simulation to rearm after a fault or an explicit mode change.
  Neutral joystick input is not a fault and does not restart the musical clock.
- The simulation worker deadline is 50 ms of wall time, separate from the 20 ms
  simulation-time control interval. This is not a hardware real-time guarantee.

## Gazebo corrections and remaining model differences

The shared IMU model originally published at 1 Hz; it now publishes at 500 Hz.
The launch resolves resources/plugins from the active overlay rather than old
`.bashrc` paths. Desktop display variables must be preserved in clean shells.

For **Go2 + controller=cpg only**, generated SDF removes four duplicate foot
collision spheres inherited from fixed-joint lumping. Each instrumented toe is
then the sole foot sphere. The original SDF and other controller modes are
unchanged. Inertias and friction are unchanged. Joint dynamics retain their original values
unless the diagnostic overrides below are explicitly enabled.

Gazebo contact streams are event-like: individual airborne feet stop publishing.
Their force expires to zero after 10 ms; joint/IMU data and at least one support
contact stream must remain fresh. A lost individual contact stream is therefore
conservatively interpreted as lost contact, not as hardware-sensor qualification.
Real Unitree foot-force units and thresholds still require calibration.

The SDF still contains passive joint stiffness 3 Nm/rad toward the sitting pose,
damping 0.5, and a total modeled mass around 16.247 kg versus the runtime's nominal
16.087 kg. These remain the default values; opt-in diagnostic overrides are available below. At a standing knee
angle around -1.58 rad, the sitting spring contributes roughly -3.2 Nm; the
policy's maximum residual displacement corresponds to about 3 Nm at Kp=25.
Audit these dynamics against the physical robot before interpreting Gazebo as
hardware validation or training specifically to compensate for them.

## Results and review boundary

See the conga repository's `reports/go2_cpg_gazebo_integration.md` and
`logs/cpg_gazebo_integration_20261101/summary.json` for measured results. The
walking baseline and experimental traveling dance execute through the full stack.
The initial integration did not train a new actor. The later authorized 75/81-input robustness comparison is documented in the conga repository’s `reports/go2_cpg_robust_compare.md`.


## Passive-dynamics diagnostic overrides

For Go2 + `controller=cpg` only, add these arguments to the dedicated launch:

```bash
cpg_disable_springs:=true cpg_disable_damping:=true
```

Both default to `false`. They set only the generated model's revolute-joint
passive stiffness/damping to zero. Motor PD remains Kp=25, Kd=0.5, and the source
SDF is unchanged. `cpg_model_dump:=/tmp/cpg_model.sdf` optionally saves the exact
generated model. These switches are diagnostics, not a physical Go2 calibration.

Two 40-second repeats per condition with the frozen experimental dance policy:
original model 13/28 strict kick chains; springs off 10/28; springs and passive
damping off 28/28. All conditions completed 88/88 steps without falls. Springs
off corrected hold clearance; removing passive damping as well corrected timing.
Final neutral forward drift fell from 0.516 m to 0.004 m over 16 seconds with
both disabled, while maximum body tilt increased from 4.32° to 7.21°.

See the conga repository's `reports/go2_cpg_passive_ab.md` and
`logs/cpg_passive_ab_20261101/` for protocol, plots, raw traces, generated models
and checksums. No defaults were changed and no training was started.


## Broader readiness audit (2026-11-01)

The longer follow-up supersedes any readiness inference from the short forward
comparison: backward travel with the experimental dance policy triggers excessive
tilt at about 31 seconds in two fresh runs. Neutral dancing drifts backward over
90 seconds. Official-model MuJoCo kicks remain substantially less reliable than
the modified Gazebo model outside forward travel. Hardware mode remains blocked.

The conga repository's `reports/go2_cpg_broad_sim.md` records 29 closed-loop trials,
600-packet ROS replay, configuration provenance, and domain-randomization gaps.
Its `logs/cpg_broad_sim_20261101/config_audit.json` distinguishes actual checkpoint
training settings from exported evaluation metadata. No training/default changes.
The audit also notes that RobotDriver's clamped `effort` telemetry does not clamp
the separate position/gain command sent by UnitreeInterface; validate total-torque
limiting before any hardware activation.

## Matched actuator profile and robustness candidates

The optional `cpg_actuator_profile:=go2hv_mujoco_v1` is available only for Go2
CPG simulation. It generates a separate SDF with passive springs zero, damping
0.1 and friction 0.2, and a temporary plugin configuration enabling a motor
envelope of 20.2 Nm motoring / 23.4 Nm braking, tapering above 13.5 rad/s to
zero at 30 rad/s. PD remains 25/0.5. Source SDF/YAML and default `legacy`
behavior are preserved. Diagnostic disable flags override passive terms.
Gazebo effective rotor inertia and physical motor behavior still need calibration.

Select a robustness candidate explicitly, together with that profile:

```bash
ros2 launch quad_utils quad_cpg_gazebo.py \
  conga_root:="$HOME/Documents/nicholas_chung/go2-conga/unitree_rl_lab" \
  selection:="$HOME/Documents/nicholas_chung/go2-conga/unitree_rl_lab/exports/go2_cpg_robust75_20261101/assets/selection.json" \
  cpg_actuator_profile:=go2hv_mujoco_v1
```

The corresponding `go2_cpg_robust81_20261101` bundle contains the other arm.
The extra six values are contact-kinematic vx/vy and four measured contact flags;
they do not use simulator body velocity. Read the comparison report before
choosing a candidate. These exports remain simulation/shadow-only. The hardware
guard, stale-sensor latch and simulation-only fallback remain in force.
