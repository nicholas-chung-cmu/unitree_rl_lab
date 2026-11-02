# Go2 CPG travel fine-tuning — 2026-11-01

Training completed: 1200 additional updates (117,964,800 transitions). The final gate triggered a step-retention review stop. Manual selection retains gate100/model3099; deployment replay and final video qualification are complete.

## Objective and scope

Fine-tune selected U (model 2999) with 4096 environments for at most 1200 additional PPO updates, with review gates at 100, 300, 700 and 1200. Preserve the original policy and all closed experiment records. Training and evaluations run sequentially under a GPU lock and a four-hour deadline. This is simulation training plus a Quad-SDK ROS 2 shadow integration; no robot is connected and no hardware controller is activated.

Velocity reward and reported travel speed exclude kick, airborne hold, recovery, initialization and reset samples. The 90 BPM clock remains independent of velocity. There is no compensating increase in speed after a pause. Continuous dancing at zero command remains enabled.

## Changes

- Linear/yaw tracking weights increase from 0.5/0.25 to 1.5/0.75; Gaussian widths narrow from 0.25 m/s and 0.4 rad/s to 0.10 m/s and 0.15 rad/s.
- Commands change every 3–5 seconds during training, with 25% zero-command samples. Initial limits remain 0.08 m/s and 0.15 rad/s; range expansion requires measured tracking and contact gates.
- The training controller uses the deployment support-foot/IMU/encoder estimate for foot clearance and endpoint speed, rather than privileged body state. Independent evaluation audits retain physical simulator truth.
- Modest mass, friction, armature and PD-gain randomization plus 0–1 physics-step actuator delay address some Isaac/MuJoCo differences. These are not calibrated real-Go2 distributions.
- A late-kick event cost and timely-arrival credit target the observed MuJoCo endpoint delay; the strict 60 ms timing tolerance remains unchanged.
- MuJoCo videos use a fixed camera and checker floor so displacement is visible. Unitree input caps can be selected explicitly rather than remaining hidden at old defaults.

## Qualification

Preflight: 108 component/legacy tests passed; randomized 32-environment, 500-step smoke had zero falls/resets. Separate deployment/metric tests passed 11/11. These checks validate implementation, not learned robustness.

Fixed-command gates use Isaac 32 environments × 1600 steps and eight MuJoCo cases × 22 seconds, alternating initial kick side. Commands include zero, ±0.05 m/s forward, ±0.04 m/s lateral, ±0.10 rad/s yaw, and mixed forward/yaw. Tracking statistics use only active travel intervals, with acceleration transients excluded from per-command mean bias. Foot and kick audits use measured physical contacts and positions.

## Deployment limits

Quad-SDK `devel_ros2` integration preserves joint/leg mappings, kp=25, kd=0.5, zero feedforward torque (load bias is already in nominal position targets), command freshness and the sensor-fault latch. The adapter still publishes only `/cpg_sim/` shadow messages. A hardware LegController mode, raw contact-force calibration, state-estimator validation, communication/actuation latency measurement and airborne fault recovery remain unqualified. Domain randomization and MuJoCo testing do not establish sim-to-real readiness.

## Fixed-command results

All reported speed statistics below exclude kick/hold/recovery. Strict kick chains include timely arrival, sustained hold and direct post-kick step. All gates had zero evaluation falls.

| Additional updates | Isaac steps | Isaac kick chains | MuJoCo steps | MuJoCo kick chains | MuJoCo mean XY bias (m/s) | Directional tracking gain |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1120/1120 | 351/352 | 185/192 | 20/56 | 0.01642 | 0.554 |
| 100 | 1120/1120 | 321/352 | 192/192 | 33/56 | 0.01464 | 0.562 |
| 300 | 1120/1120 | 218/352 | 192/192 | 30/56 | 0.01451 | 0.566 |
| 700 | 1100/1120 | 57/352 | 192/192 | 49/56 | 0.01952 | 0.532 |
| 1200 | 904/1120 | 18/352 | 191/192 | 47/56 | 0.01795 | 0.546 |

The automatic composite ranking selected gate700 because its MuJoCo kick score improved substantially. That score omitted Isaac kick retention. Manual review rejects it as the default replacement: only 57/352 strict Isaac kick chains remained, primarily because the raised pose was not maintained throughout the hold. Gate1200 also regressed to 904/1120 Isaac steps. `automatic_selection.json` and the supervisor records preserve the original automatic result; `selected.json` and `selection_review.json` record the reviewed choice, gate100/model3099. The supervisor's `selected_*` videos evaluate its automatic gate700 choice and are **diagnostics, not the final recommended policy videos**.

At the same +0.05 m/s forward request, the reviewed policy achieves +0.0205 m/s versus +0.0159 m/s before fine-tuning (about 29% improvement). Backward speed worsens from -0.0265 to -0.0209 m/s at -0.05 requested; lateral tracking remains about 0.032 m/s at 0.04 requested, and turning stays about 0.02 rad/s at 0.10 requested. Average directional gain barely changes (0.554→0.562). This is a modest forward/port improvement, not a general velocity-tracking solution. Isaac strict kick retention also drops from 351/352 to 321/352, so retain the original U as a dance-quality fallback.

The range never advanced above 0.08 m/s: no gate reached the 0.75 directional-gain threshold. Learning rate was not frozen at its floor: the final 500-update segment averaged 1.127e-4, ended at 5.0625e-5, and spent only one update at the 1e-5 floor. Action noise ended at 0.919. More updates under this objective did not resolve tracking and damaged dance retention. Full scalar statistics are in `learning_audit.json`.

## Higher commands and deployment verification

Independent seed9102 Isaac evaluation at the larger input caps retained 1120/1120 steps and 320/352 strict kick chains, with zero falls. Eight 22-second MuJoCo fixed-command tests retained 192/192 steps and 33/56 strict kick chains, with zero falls—the same kick count as the lower-speed gate100 cases. The selected simulation input caps are now 0.08 m/s forward/backward, 0.06 m/s lateral and 0.15 rad/s yaw; the controller still caps combined XY norm at 0.08 m/s. These are tested input limits, not achieved speeds or hardware qualifications.

| Requested command | Measured travel-only component |
|---|---:|
| [0.08, 0.0, 0.0] | (0.0344, 0.0030, 0.0001) |
| [-0.08, 0.0, 0.0] | (-0.0307, -0.0028, 0.0029) |
| [0.0, 0.06, 0.0] | (-0.0066, 0.0466, -0.0144) |
| [0.0, -0.06, 0.0] | (-0.0066, -0.0464, 0.0132) |
| [0.0, 0.0, 0.15] | (-0.0036, 0.0066, 0.0348) |
| [0.0, 0.0, -0.15] | (-0.0042, -0.0070, -0.0364) |
| [0.06, 0.0, 0.15] | (0.0260, 0.0055, 0.0425) |

Vectors are (forward m/s, left m/s, yaw rad/s). Stationary and forward cases each passed 7/7 strict MuJoCo kick chains. Backward passed 0/7, lateral and pure-turn cases each 3/7, and mixed forward/yaw passed 7/7. Thus raising the input caps makes travel faster, but reliable kicks while moving in every direction remain unresolved. These are short deterministic flat-ground tests, not reliability bounds.

Deployment checks: 1100 sensor frames produced exactly identical position targets through the JSON worker (maximum difference 0 rad). Exported ONNX versus JIT maximum action difference was 1.526e-5. Real Quad-SDK ROS2 messages replayed 400 sensor packets successfully, with 402 output messages including the short freshness grace period; sensor-stream loss latched output off and shutdown was clean. Testing used isolated localhost DDS domain77, no robot.

Recommended next experiment: retain gate100 and original U; isolate stronger travel tracking from kick/hold adaptation, explicitly gate checkpoint selection on sustained hold in both simulators, and investigate the forward/yaw tracking deficit under matched actuator and estimator models. Do not simply extend the same optimization: 1200 updates already showed regression. Calibrate real contact-force units, gains and latency before a hardware controller integration.

## Final changing-command tests and videos

- `reviewed_dynamic_isaac`: steps 1472/1472, strict kick chains 402/480, falls 0.
- `reviewed_zero_video`: steps 24/24, strict kick chains 7/7, falls 0.
- `reviewed_commands_video`: steps 42/44, strict kick chains 9/14, falls 0.

Final videos: `logs/cpg_finetune_20261101/reviewed_zero_video/dance.mp4` (22 seconds) and `reviewed_commands_video/dance.mp4` (40 seconds). Both use a fixed camera and floor grid. Scripted input includes forward/backward, both lateral directions, both yaw directions, mixed forward/yaw, and neutral. Use these rather than the automatic gate700 videos. Dynamic transitions still produce occasional missed steps/kick-chain failures; input caps are simulation test settings, not a complete motion qualification.

## Artifacts

Reviewed selection: `logs/cpg_finetune_20261101/selected.json` (model3099). Relocatable controller/export bundle: `exports/go2_cpg_quad_sdk_finetuned_20261101/`, with a matching `.tar.gz` archive. Bundle smoke completed 200 control samples with no fault or fall. Its manifest records policy/source/report hashes. The original deployment bundle remains intact. `source_final/` records final source; visualization/input-cap/packaging edits made after the training snapshot are retained separately and do not alter the completed training run.
