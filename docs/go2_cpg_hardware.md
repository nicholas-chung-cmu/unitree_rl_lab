# Go2 EDU conga: hardware test handoff

This is an opt-in **hardware test candidate**, not a hardware-qualified policy.
No physical Go2 was connected during preparation. Use the complete bundle,
including the scheduler, Cartesian references, IK, contact estimator and residual
actor. Swapping the walking controller's ONNX file alone is incompatible.
The selected actor has 75 observations and 12 bounded residual joint actions.
See [the experiment report](../reports/go2_cpg_hardware6h.md) for results and misses.

## Which computer runs what?

Run both `robot_driver` and the Python policy worker **on the robot-facing Orin**
if its timing test passes. RViz and command input can run on a laptop; there is
no requirement to put the 50 Hz policy feedback loop over Wi-Fi. The exact Orin
Nano/NX model, JetPack, Ubuntu release and ROS distribution have not been verified.
Humble is the ROS distribution name, not the Ubuntu version. The development
machine tested here uses x86_64, Ubuntu 24.04 and ROS Jazzy; that build cannot be
copied as an executable install to an ARM64/Humble machine.

Without mocap/external localization, the driver's raw-sensor fallback pins the
reported body position to the origin. RViz can show joint motion but is not a
measurement of distance traveled; use floor markers/video or verified odometry.
The conga controller does not require a global position estimate.

The [Quad-SDK first-run tutorial](https://robomechanics.github.io/quad-sdk/ros2/tutorials/first-run/)
describes simulator/planning/RViz operation. Its mode numbers differ from this
`nick_conga` checkout. The [hardware guide](https://robomechanics.github.io/quad-sdk/ros2/getting-started/hardware-bringup/)
provides broader setup context; use the code and commands below for this branch.

| Local branch mode | Meaning |
|---|---|
| `0` | SIT |
| `1` | READY: run the selected leg controller after the stand transition |
| `2` | Internal SIT_TO_READY transition; **not a separate walking command** |
| `3` | Internal READY_TO_SIT transition |
| `4` | SAFETY: CPG damping, Kp=0, Kd=0.5 |

## Preserve the working walking setup and build natively

Record the robot's working walking launch command, workspace and environment
before changing anything. Keep that install and its policy available. Use a
separate checkout/overlay for CPG. Do not replace the robot's system ROS, Python,
JetPack or existing ONNX Runtime just to match this desktop.

The bundle's `quad_sdk/working_tree.patch` and `quad_sdk/base.json` record the
complete local Quad-SDK integration against its base commit, including earlier
Gazebo work. Review it and run `git apply --check` in the matching checkout before
applying. Do not force-apply it over a different robot branch or local changes.
`quad_sdk/source/` contains the corresponding source files for review/merging.
Never deploy the desktop `build_nick_conga` or `install_nick_conga` binaries.

After sourcing the robot's **existing working ROS/Quad-SDK environment**, build
only the updated hardware packages into a separate install:

```bash
# In the native robot workspace containing the updated Quad-SDK sources:
colcon build --build-base build_cpg_test --install-base install_cpg_test \
  --packages-select quad_msgs quad_utils robot_driver \
  --allow-overriding quad_msgs quad_utils robot_driver \
  --cmake-args -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=/usr/bin/python3
source install_cpg_test/local_setup.bash
ros2 interface show quad_msgs/msg/CpgSensorFrame
ros2 launch quad_utils quad_cpg_hardware.py --show-args
```

The CPG worker uses Python ONNX Runtime CPU inference. The existing walking
controller's native C++ ONNX dependency is separate; preserve it for rollback.
Use a dedicated policy environment with `torch`, `numpy`, `onnxruntime`, and
`PyYAML`. Isaac Lab, CUDA inference, and MuJoCo are not required on the robot.
The tested desktop versions are in `evidence/host_dependencies.json`; they are
provenance, not an instruction to install an x86/CUDA wheel on ARM. Follow the
[PyTorch platform installer](https://pytorch.org/get-started/locally/) and
[ONNX Runtime CPU installation guide](https://onnxruntime.ai/docs/install/)
for the robot's Python/architecture. Keep ROS in its system Python environment;
`--policy-python` selects the worker environment independently.

From the extracted bundle, set explicit paths (substitute your native locations):

```bash
export CONGA_ROOT="$PWD"
export CPG_PYTHON=/path/to/policy/environment/bin/python
export CPG_SELECTION="$CONGA_ROOT/assets/selection.json"
/usr/bin/python3 scripts/cpg_preflight.py --selection "$CPG_SELECTION" \
  --policy-python "$CPG_PYTHON" --packets evidence/preflight_packets.json \
  --output logs/orin_preflight.json
```

This preflight never initializes ROS or Unitree DDS. It checks worker startup and
replays saved sensor packets. Require all packets to complete without faults and
under the 15 ms response deadline; repeat under the intended onboard load.
An offline pass is necessary, but does not verify the complete ROS/driver timing.
The live adapter also limits accumulated control-clock drift to 20 ms so repeated
small timer delays cannot silently change the 90 BPM tempo.

## Read-only sensing and calibration

Build the two read-only SDK2 receivers natively with the existing SDK2 installation:

```bash
cmake -S deploy/cpg -B build/cpg_receivers
cmake --build build/cpg_receivers -j2
```

`ROBOT_MCU_IFACE` must be the interface used by the working robot setup. Inspect
that setup; do not assume `eth0`. The hardware driver uses Unitree DDS domain 0.
Neither receiver publishes motor commands or releases Sport mode.

With the robot supported by its established handling procedure, collect distinct
unloaded and stable standing recordings. Each requires at least 200 advancing
MCU samples; ten seconds is ample. The capture does not itself change posture:

```bash
build/cpg_receivers/lowstate_receiver "$ROBOT_MCU_IFACE" 0 logs/air.jsonl 10
# Put the robot into stable four-foot support using its established controller.
build/cpg_receivers/lowstate_receiver "$ROBOT_MCU_IFACE" 0 logs/standing.jsonl 10
"$CPG_PYTHON" scripts/cpg_contact_calibration.py \
  --air logs/air.jsonl --standing logs/standing.jsonl \
  --policy-sha256 "$("$CPG_PYTHON" -c 'import json,os; print(json.load(open(os.environ["CPG_SELECTION"]))["onnx_sha256"])')" \
  --output logs/go2_profile_draft.json
```

This maps each raw foot channel to contact **evidence**, not measured Newtons.
It preserves source hashes and leaves `hardware_enabled`, `contact_calibrated`
and `timing_verified` false. Verify each front and rear foot independently in
planted/unloaded states, including the real remote button/axis signs. Do not mark
all channels calibrated based only on an average standing reading. Offsets and
scales are in Quad-SDK order **FL, RL, FR, RR**; raw SDK order is FR, FL, RR, RL.
Also check the IMU world-from-body quaternion (SDK wxyz), gyro axes, encoder signs
and joint names against the existing known-good controller.

The updated driver publishes `/<robot>/cpg/sensors`, an atomic raw packet from
one LowState sample with its source tick and receipt timestamp. While the
**existing working controller** owns the robot, run the CPG adapter in shadow:

```bash
/usr/bin/python3 scripts/cpg_quad_hardware.py \
  --selection "$CPG_SELECTION" --profile logs/go2_profile_draft.json \
  --policy-python "$CPG_PYTHON" --robot-ns /robot_1 \
  --trace logs/go2_shadow.jsonl
```

Shadow publishes only `/cpg_sim/leg_command_array`, never motor targets. Do not
remap that output to a real controller. Check the actual trace for advancing
source ticks, no sensor/deadline faults, and worker response below 15 ms. Test
under the intended onboard load for several minutes. Shadow tests transport,
calibration, observations and timing; its unexecuted steps cannot qualify dance
motion. An unmodified driver without `CpgSensorFrame` cannot supply this adapter.

## Supported first test and normal controls

After real contact and timing checks, make a separate reviewed profile naming
the selected policy hash and set its three verification/enable flags true.
The shipped template deliberately cannot activate hardware. `hardware_compatible`
remains false in the selection to state that this development machine did not
qualify the physical robot; the measured profile is the explicit local opt-in.

**Launching the hardware driver releases Unitree Sport ownership and begins
CPG damping.** Start supported/seated using the lab's harness procedure, with
only one LowCmd owner and the robot's established independent stop available.
The custom remote B response is software damping, not an independent e-stop.
Never run the basic walking driver and CPG driver simultaneously.

```bash
ros2 launch quad_utils quad_cpg_hardware.py \
  conga_root:="$CONGA_ROOT" selection:="$CPG_SELECTION" \
  profile:="$CONGA_ROOT/logs/go2_profile_verified.json" \
  policy_python:="$CPG_PYTHON" trace:="$CONGA_ROOT/logs/go2_active_01.jsonl" \
  activate_hardware:=true
```

The launch checks the profile, actor checksum, ROS imports, trace path and worker
startup **before** creating the vendor interface. It starts in mode 4 and never
arms automatically. Once the operator is ready, the explicit READY request ramps
from measured joint positions, then the controller waits for measured settling:

```bash
ros2 topic pub --once /robot_1/control/mode std_msgs/msg/UInt8 '{data: 1}'
```

READY begins continuous dancing even at zero velocity. No planner or RViz goal
is needed. Begin with short supported, zero-travel trials; then try small forward
input and brief turns only after checking contacts, posture, motor response and
logs. The selected envelope is 0–0.10 m/s forward and ±0.10 rad/s yaw, recorded in
`assets/selection.json`; the MuJoCo final checks also use its 18 Nm sampled cap.
Backward and lateral commands are clipped to zero. Tempo remains 90 BPM; speed
changes stride length, and velocity tracking excludes the kick/hold pause.

For Unitree wireless input, run these on the same host (the snapshot uses its
monotonic clock), using separate terminals:

```bash
build/cpg_receivers/wireless_receiver "$ROBOT_MCU_IFACE" 0 /tmp/go2_wireless.json
/usr/bin/python3 scripts/cpg_wireless_ros2.py --input /tmp/go2_wireless.json \
  --selection "$CPG_SELECTION" --robot-ns /robot_1
```

| Input/event | Behavior |
|---|---|
| Left stick up (`ly`) | Forward travel, limited by selection |
| Right stick (`-rx`) | Turning, limited by selection |
| Neutral, missing or stale velocity (>250 ms) | Zero travel; **dance continues** |
| A | Deliberate finish-landing/settle request, then hold a fixed four-foot pose |
| B / mode 4 | Latched damping: Kp=0, Kd=0.5; no upright-recovery guarantee |
| Stale sensors/targets, invalid state, excessive tilt | Latched fault; no automatic restart |

A stand request can wait for a stable landing; it is not instantaneous emergency
stopping. It must finish within 8 seconds or fault. Watch `cpg/status` for
`standing: true`. For a deliberate CLI stop, then a controlled sit:

```bash
ros2 topic pub --once /robot_1/cpg/stand std_msgs/msg/Bool '{data: true}'
ros2 topic echo /robot_1/cpg/status
# Only after standing is confirmed:
ros2 topic pub --once /robot_1/control/mode std_msgs/msg/UInt8 '{data: 0}'
```

Sitting ramps from measured joint positions. Stop the processes after the robot
is seated/supported. Ctrl+C during an airborne kick is not the normal stopping
procedure. Deliberate standing or a fault does not automatically resume when the
stick moves; diagnose, seat/support, and restart the complete runtime/driver.
Neutral input alone does resume travel immediately when valid input returns,
while the musical clock keeps running throughout.

## Remaining physical checks and four-hour test window

The new CPG driver limits the **estimated sampled PD+feedforward demand** to
18 Nm and the configured speed envelope by scaling transmitted gains/torque.
That is not a physical torque guarantee between samples. Startup/SIT gains also
pass through the hardware limiter. The separate 5 ms watchdog reacts to stale
MCU/command streams with damping, but cannot survive process, OS, power or network
failure. Confirm the robot's independent protection using the lab procedure.

Suggested allocation: first hour for native environment/build, read-only sensing
and rollback; second hour for calibration and shadow timing; third hour for short
supported neutral/forward trials and stop checks; fourth hour for cautious free
forward trials with brief turns only if prior stages pass. If installation or
calibration consumes the window, use the preserved working walking policy rather
than bypassing the guards. Do not spend the morning retraining until measured
logs identify a specific actuator, estimator or timing mismatch.

Physical motor gain/lag/friction, contact scaling and onboard scheduling remain
unmeasured. The flat, firm floor matches the intended deployment setting, but
cross-simulator success does not settle those measurements. Save raw LowState,
`cpg/sensors`, `cpg/status`, `cpg/fault`, `control/joint_command`, `state/control_mode`
and the adapter JSONL trace for each trial. Compare real foot contacts and
sampled torque demand with the simulation report before increasing duration.
