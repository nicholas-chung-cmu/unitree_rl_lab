# Go2 CPG MuJoCo and Quad-SDK adapter

> Historical experiment/port notes. For the 2026-11-02 hardware test candidate,
> current controls and remaining qualification checks, use
> [the hardware handoff](go2_cpg_hardware.md). Earlier policy results and
> simulation-only restrictions below describe those earlier versions.

The full controller runs in MuJoCo without Isaac Lab: scheduler, Cartesian
references, IK, support compensation, contact feedback, and the selected U
residual actor. Neutral motion input continues dancing. No retraining was done.
The simulation executable has no robot command transport.

Results and limitations: [qualification report](../reports/go2_cpg_mujoco.md).
The current policy does **not** pass the strict MuJoCo kick timing gate. Do not
interpret the controller's relaxed endpoint counters as strict choreography success.

## Run the local simulation

From the repository root, with the existing `nick_sillywalk` environment:

```bash
/home/rml2/anaconda3/envs/nick_sillywalk/bin/python scripts/cpg_mujoco.py \
  --viewer --input zero --seconds 120 --output logs/my_cpg_view
```

Use a new output directory each time. The default model is the existing official
`/home/rml2/unitree/unitree_mujoco/unitree_robots/go2/go2.xml`; override it with
`--model /path/to/unitree_mujoco/unitree_robots/go2/go2.xml` on another machine.
`--input scripted` exercises forward/backward, lateral motion, yaw, a moving
turn, and return to neutral. `--video` also records an MP4. Headless runs omit
`--viewer`; video rendering defaults to EGL. Viewer mode requests real-time pacing;
rendering overload can make it slower, and it is not a hard real-time controller.

The robot initializes at a low standing height, then must pass the same measured
settling gate used in training. No physical reset occurs at phrase boundaries.
The actor runs at 50 Hz; the MuJoCo PD/torque loop runs at 500 Hz. Gains are 25/0.5,
with the training Go2HV torque-speed envelope. The official robot's inertia,
joint damping, armature and contact settings are retained, with a flat floor.
This is a different dynamics model, not a tuned recreation of Isaac physics.

The state adapter uses joint encoders, IMU orientation/gyro, and foot-force
readings. Support-foot kinematics estimate flat-ground height and foot speed.
Ground truth is recorded separately for independent audits. `--diagnostic-truth`
is explicitly diagnostic and is not a deployable-estimator qualification.

## Unitree wireless controller

The selected physical controller is Unitree wireless; the robot is not connected
yet. The optional receiver subscribes **only** to `rt/wireless_controller` and
writes an atomic local JSON snapshot. It has no LowCmd publisher, sport client,
or motion-switching calls. Build it with the already installed SDK2:

```bash
cmake -S deploy/cpg -B build/cpg_wireless
cmake --build build/cpg_wireless -j2
```

When a suitable input source is available, the receiver command is:

```bash
build/cpg_wireless/wireless_receiver INTERFACE DOMAIN /tmp/go2_wireless.json
```

Use `lo 77` for an isolated local test source. A real Go2 normally uses its robot
network interface and domain 0. The receiver itself is read-only, but a remote
paired with a powered robot may also operate that robot's existing controller;
that robot-side operating mode must be handled before moving the sticks.
No real-robot input session was run in this work.

Connect the file to the local simulator:

```bash
/home/rml2/anaconda3/envs/nick_sillywalk/bin/python scripts/cpg_mujoco.py \
  --viewer --input wireless --wireless-state /tmp/go2_wireless.json \
  --seconds 120 --output logs/my_wireless_dance
```

Mappings match the existing Go2 deployment convention: `ly` forward, `-lx`
left, `-rx` positive yaw. Deadband is 0.08, rescaled continuously outside it.
Full-stick commands are ±0.05 m/s forward, ±0.04 m/s lateral, ±0.10 rad/s yaw.
Planning acceleration limits remain 0.075 m/s² and 0.15 rad/s²; tempo stays 90 BPM.
Neutral, missing, invalid, or older-than-250-ms input requests zero travel while
the dance continues. B latches a **simulation stop**. Sensor loss is different:
the control runtime latches a fault and stops producing commands.

## Quad-SDK ROS 2 target

The user's target is `robomechanics/quad-sdk`, branch `devel_ros2`, inspected at
commit `67d2b64749e13cd924a778be0eb1e7914b014582`. A read-only source checkout is
under `.local/quad-sdk`. Its Go2 `UnitreeInterface` already maps raw encoders and
IMU to ROS and publishes `quad_msgs/FootContact`. Its existing `LearnedPolicy`
does not implement our scheduler, 75-element observation, or bounded residual
composition. Replacing its ONNX file alone is not compatible.

The new `deploy/cpg/quad_sdk.py` codec explicitly converts these conventions:

| Boundary | Ordering / meaning |
|---|---|
| Quad-SDK joints and commands | FL, RL, FR, RR; hip/thigh/calf per leg |
| Quad-SDK numeric joint names | `8,0,1,9,2,3,10,4,5,11,6,7` |
| Runtime sensor/action boundary | SDK order FR, FL, RR, RL; hip/thigh/calf |
| Internal CPG feet | FL, FR, RL, RR |
| IMU quaternion | wxyz, world-from-body; gyro in body frame |
| Joint output | position targets, zero desired velocity, Kp 25, Kd 0.5, zero feedforward torque |

Support compensation is already included in the position targets: do not add
the old MPC/inverse-dynamics feedforward force or apply the old policy's action
scale/offset on top. Name-based input mapping accepts both the numeric Go2 names
and standard `FL_hip_joint` names, and rejects unknown or duplicate names.

`scripts/cpg_quad_ros2.py` is a ROS 2 **simulation/shadow adapter**. It consumes
`state/joints`, `state/imu`, `state/foot_contact` and `cmd_vel` under a configurable
robot namespace. It emits real `quad_msgs/LegCommandArray` messages exclusively
under `/cpg_sim/`. It does not activate a Quad-SDK hardware mode. A separate
`cpg_worker.py` process runs the complete controller in the policy's Python
environment, avoiding the ROS Jazzy Python 3.12 / policy Python 3.11 ABI mismatch.

Build only the actual Quad-SDK messages locally:

```bash
colcon --log-base logs/quad_cpg_build build \
  --base-paths .local/quad-sdk/quad_msgs --packages-select quad_msgs \
  --build-base build/quad_cpg --install-base build/quad_cpg_install \
  --cmake-args -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=/usr/bin/python3
source build/quad_cpg_install/local_setup.bash
```

Then run against simulation or recorded sensor topics, using an isolated local
ROS domain for simulation-only work:

```bash
ROS_DOMAIN_ID=77 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  /usr/bin/python3 scripts/cpg_quad_ros2.py --robot-ns /robot_1
```

The adapter requires fresh, synchronized sensor streams. A missed policy deadline,
stale sensor data, or malformed state latches output off; restart the process after
diagnosing the fault. Joystick timeout alone continues the dance at zero travel.
`FootContact.foot_force_raw` is an int16 **raw reading**, not established Newtons;
the simulation scale of 1 must be calibrated on the real robot before deployment.
`--force-scale` and `--force-offset` expose that calibration explicitly.

Actual Quad-SDK hardware integration still needs a dedicated `LegController` mode
that accepts these targets at 50 Hz, retains them at the driver's servo rate,
and preserves the existing hardware ownership, state machine and watchdogs.
The existing driver does not consume our `/cpg_sim/` topic. Its fallback on a
failed leg-controller call is a standing target, which is not yet qualified for
interrupting an airborne kick. Do not remap the shadow topic to a hardware output
as a substitute for that integration and transition validation.

## Reproduce verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/rml2/anaconda3/envs/nick_sillywalk/bin/python -m pytest \
  tests/test_cpg_deployment.py -q
```

Artifact-dependent replay tests skip when the original selected checkpoints or
recorded MuJoCo traces are unavailable. CPU tests cover message permutations,
neutral input, stale/reconnected wireless input, sensor faults, observation and
action layout, original-controller parity, ONNX/JIT parity, and foot velocity
at the correct MuJoCo frame origin. The ROS replay uses actual generated messages,
the separate worker, and local DDS; it does not certify hardware scheduling.

For independent motion audits, run `scripts/analyze_cpg_contacts.py OUTPUT --dance
--swing .2 --tolerance .06` and `scripts/audit_cpg_kicks.py OUTPUT`. Expected
opportunities include skipped steps. Never count a finished reference as a
successful measured step or kick.
