# Go2 CPG MuJoCo and Quad-SDK work — 2026-11-01

The complete dance controller now runs outside Isaac Lab, with a Unitree wireless
input path and a tested Quad-SDK ROS 2 message adapter. Stationary diagonal stepping
transferred successfully in the tested MuJoCo cases. Strict kick timing did not.
No policy training, hardware commands, robot mode changes, or physical joystick
test occurred. The robot is not connected.

## Implemented

The controller logic moved into a shared, simulator-independent module,
`source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/cpg/controller.py`.
Isaac Lab retains thin lifecycle wrappers around the same code. Existing
phase-conditioned/hierarchical tasks, checkpoints, and the closed four-hour
training session are preserved.

The deployment runtime includes the deterministic scheduler, 90 BPM phase and
Cartesian references, IK, compliant support compensation, measured contact events,
75-element observation, ONNX actor, and bounded residual position targets. It
uses the selected U model2999 and its existing export, without changing rewards,
policy weights, kick geometry, or musical tolerance.

The new MuJoCo transport uses the installed official Go2 model with a flat floor.
Foot height and velocity supplied to control come from encoder/IMU/support-foot
estimation. Independent audits use true physical foot positions and velocities.
The sensor/actuator boundary is explicit and does not expose root simulation
state to the actor. Startup settling admits at approximately 1 second in these
cases; no repeated physical resets occur.

Unitree remote input maps to the tested small command range. Neutral or stale
motion input requests zero travel and continues dancing. A read-only SDK2 receiver
can supply input through a local JSON file; it never publishes robot motor
commands. The physical remote has not been tested because the robot is absent.

The user's target repository, [Quad-SDK devel_ros2](https://github.com/robomechanics/quad-sdk/tree/devel_ros2),
was inspected at `67d2b64749e13cd924a778be0eb1e7914b014582`. The ROS adapter uses
that branch's actual `JointState`, `Imu`, `FootContact`, and `LegCommandArray`
definitions, with explicit numeric joint-name and FL/RL/FR/RR permutations.
Output is restricted to `/cpg_sim/` for simulation/shadow evaluation. The existing
Go2 `robot_driver` hardware mode has not been replaced or activated.

## Final physical results

Each row is one deterministic 62-second simulation, with a different initial
diagonal or command profile. These are four trajectories, not a statistical
robustness campaign. Actual contacts define success, including suppressed steps.

| Input / first diagonal | Correct steps | Falls | Strict FL kick chains | Strict FR kick chains |
|---|---:|---:|---:|---:|
| Neutral / FL–RR | 69/69 | 0 | 0/11 | 0/11 |
| Neutral / FR–RL | 69/69 | 0 | 0/11 | 0/11 |
| Scripted / FL–RR | 67/69 | 0 | 2/11 | 3/11 |
| Scripted / FR–RL | 66/69 | 0 | 2/11 | 3/11 |

Stationary totals: **138/138 steps**, zero falls. Changing-command totals:
**133/138 steps**, zero falls; five opportunities were missed after kick/recovery
events. Scripted stick requests change every 4.1 seconds through neutral, forward,
backward, left, right, both yaw directions, a forward turn, then neutral. The
remaining time after 32.8 seconds is neutral. Commands are bounded to ±0.05 m/s
forward, ±0.04 m/s lateral and ±0.10 rad/s yaw, with the existing acceleration
limits. This does not qualify arbitrary commands or disturbances.

Median absolute touchdown error is approximately 6.9–7.4 ms at 20 ms sampling.
Mean contact-foot slip is 0.0084 m/s stationary and 0.0092–0.0094 m/s in the
changing-command cases. No joint-limit clamps were recorded. CPU controller
latency p99 was approximately 1.9–2.2 ms without video rendering, below the 20 ms
control interval on this workstation; this is not an onboard timing guarantee.

## What still fails

The strict kick gate requires a measured endpoint within 60 ms after the
scheduled 0.25-second extension. Stationary kicks reach the qualified endpoint
approximately **70 ms late** in the sampled trace. A representative first FR
kick has 21.6 mm planar error at +17 ms, 20.2 mm at +57 ms, then 19.3 mm at +77 ms;
the limit is 20 mm. The decisive failure in that example is endpoint position,
not inability to lift or remain upright. Height is approximately 6.2 cm, within
the configured 5 cm ±1.5 cm band, and endpoint speed is already below 0.1 m/s.

Once the endpoint is reached, the median valid fraction of the remaining hold is
100% in the final stationary cases, and post-kick steps land successfully. These
facts do not erase the timing failure. The live scheduler's more permissive
endpoint counter is not the strict audit. Changing commands also cause some
kick/recovery failures and suppressed next steps.

A separate 22-second no-bounce R-policy check completed 24/24 steps without a
fall, but also had 0/7 strict kick chains and roughly 100 ms median endpoint delay.
Simply switching to that earlier actor does not solve the transfer issue.

The planar endpoint offset needs further cross-simulator diagnosis of actuator,
contact and support dynamics. Any subsequent policy adjustment should retain the
current geometry and timing gates and recheck Isaac performance. No automatic
retraining or threshold relaxation was performed here.

## Integration and regression evidence

- All **106 existing component/legacy tests** passed after sharing the controller.
  A fresh 8-environment, 400-step Isaac evaluation/export smoke had zero falls or
  resets. This checks regression, not a new training result.
- **9 deployment tests** passed: mapping, bounds, neutral behavior, stale input,
  latched sensor faults, exact legacy-controller replay, ONNX/JIT agreement, and
  the physical velocity frame check.
- Replaying **3,100 recorded sensor frames** through the separate policy worker
  reproduced the direct runtime's joint targets with **0 rad maximum difference**.
- The actual Quad-SDK `quad_msgs` package built locally. A ROS 2 loopback test
  replayed **400 sensor packets**, received **402 valid command messages**, verified
  the stale-sensor fault latch, and exited cleanly. The two additional outputs
  were generated before the 50 ms freshness timeout; the test is a transport and
  fault test, not exact-time ROS replay parity.
- The SDK2 wireless receiver and isolated wireless-input publisher built locally.
  See `wireless_test_result.json` for the local receive/mapping/timeout check.
  This does not substitute for testing the actual handheld remote.

Early diagnostic runs used MuJoCo's inertial-body-origin velocity and initially
mixed pre/post-integration sensor values. Both errors were repaired: final runs
refresh sensor state and use `mjOBJ_XBODY`, checked against the body-position
Jacobian. Those early runs are retained but excluded from final qualification;
`superseded_diagnostics.json` identifies them. Final source and artifact hashes
are recorded separately from the closed training session.

## Quad-SDK deployment boundary

The deliverable includes the full controller and ROS 2 message adapter. It is
**not a drop-in replacement ONNX file** for Quad-SDK's existing `LearnedPolicy`.
An actual `LegController` mode still needs to own these targets within
`robot_driver`, retain them at its servo rate, and preserve its safety/state
machine and watchdogs. The current driver does not subscribe to `/cpg_sim/`.

Raw Unitree foot-force units need hardware calibration; the simulation assumes
one raw unit per Newton. The flat-ground kinematic estimator also needs checking
against the Go2/Quad-SDK estimator. Interrupting an airborne kick into the existing
standing fallback, sustained runtime deadlines, disturbances, and hardware
actuator differences remain unqualified. No robot-connected tests were possible.

## Artifacts

- [Zero-command MuJoCo video, 22 seconds](../logs/cpg_mujoco_20261101/video_zero_v2/dance.mp4):
  24/24 steps, zero falls; strict kicks fail the timing gate. Both front-leg hold
  frames were inspected and the recording ends with four contacts.
- [Scripted-input MuJoCo video, 40 seconds](../logs/cpg_mujoco_20261101/video_scripted/dance.mp4):
  42/44 expected steps, 5/14 strict kick chains, zero falls, ending with four contacts.
  Forward/lateral/turn/neutral frames were extracted; lateral and turn frames inspected.
- [Machine-readable final summary](../logs/cpg_mujoco_20261101/summary.json).
- [Launch and integration instructions](../docs/go2_cpg_mujoco.md).
- Portable bundle: `exports/go2_cpg_quad_sdk_candidate/`, with relative paths and
  a SHA-256 manifest. It is labelled as a simulation/shadow candidate.

No packages in the shared Isaac Lab environment or the external official Go2
MuJoCo checkout were modified. Original trained checkpoints remain intact.
