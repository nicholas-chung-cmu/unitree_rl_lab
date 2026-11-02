# Go2 travel-first curriculum results — 2026-11-01

Completed **2000 additional PPO updates with4096 environments and24-step rollouts:196,608,000 transitions**, within the original four-hour deadline (about one hour including evaluation). No training remains active. Original checkpoints, finished dance policies and bundles are preserved.

**Outcome:** meaningful simulated displacement is demonstrated. A walking-only candidate is qualified within the tested flat-ground cases. A traveling-dance candidate is useful for inspection but fails the full timing/contact/hold gates and does not replace the original finished dance. Hardware qualification remains pending.

## What to watch

| Video | Command | Forward displacement | Strict steps | Strict kick chains | Falls |
|---|---|---:|---:|---:|---:|
| [Walking,30s](../logs/cpg_travel_curriculum_20261101/armature_transfer/selected_forward_video/dance.mp4) | +0.12m/s | 2.19m | 43/43 | No kicks | 0 |
| [Experimental traveling dance,30s](../logs/cpg_travel_curriculum_20261101/experimental_dance_video/dance.mp4) | +0.12m/s | 1.49m | 30/33 | 9/10 | 0 |
| [Walking with changing input,40s](../logs/cpg_travel_curriculum_20261101/armature_transfer/selected_commands_video/dance.mp4) | Scripted forward/back/lateral/yaw | Mixed-direction sequence | 58/58 | No kicks | 0 |

These are fixed-camera videos using the official MuJoCo robot model. Walking also drifts0.33m laterally over the30-second forward clip; the experimental dance drifts0.08m. Precise heading/speed tracking is not established. The forward dance clip is a favorable specific case, not a general qualification: the same candidate passes only8/56 strict kick chains across the eight fixed commands.

Travel-only mean forward speeds are0.0767m/s walking and0.0614m/s dancing at the0.12m/s command. **Kick,hold,recovery,initialization and reset intervals are excluded from velocity rewards and travel-speed averages.** Net displacement above is separately measured from body position across the whole video, including pauses; there is no catch-up command. Mean contact-foot-center horizontal speed during travel is0.0184/0.0275m/s respectively (a slip proxy including contact transients). Verified airborne steps prevent interpreting these results as purely grounded sliding, but contact motion still merits improvement.

## Selected artifacts

- Walking: `walking_candidate.json`, model899 from the100-update walking adaptation. Continuous90BPM diagonal marching, no kicks or bounce. It passed1472/1472 fixed-command Isaac steps,1952/1952 changing-command steps,248/248 fixed-command MuJoCo steps, and the videos above, without falls.
- Experimental dance: `experimental_dance.json`, model2499 at1700 cumulative updates. It has the best worst-leg/worst-simulator strict kick rate among the final inertia-matched comparisons, but that rate is far below acceptance. The latest model2799 is not promoted.
- Original stationary/finished dance: unchanged; use its existing selection and bundle as the preserved dance fallback.
- Bundles: `exports/go2_cpg_demo_walk_20261101/` and `exports/go2_cpg_demo_dance_experimental_20261101/`, with matching `.tar.gz` archives. Their selection metadata explicitly distinguishes walking-only qualification from experimental dance.

Walking simulation joystick caps are0.12m/s forward/backward,0.08m/s lateral,0.15rad/s yaw; XY norm remains bounded by its0.15m/s gait cap. Experimental dance caps are0.12/0.06m/s and0.10rad/s yaw, XY norm0.12m/s. Neutral input continues marching/dancing. Neither bundle is a hardware LegController mode.

## Training and review results

Source was the earlier walking M/model799, before kicks and bounce. Walking100 updates preceded kick acquisition. Dance training alternated with ordinary walking rehearsal, then added the existing exact sagittal actor wrapper after a persistent absent-side kick was measured. Later stages tightened kick-planar shaping (scale0.03→0.015m), increased hold weight2→4, added angular hold shaping, shortened rehearsal50→25 updates, and reduced the lateral/yaw demo range. Bounce was never advanced because the dance gates did not pass.

Acceptance required>=95% strict steps in both simulators, zero falls, signed travel and>=0.8m forward displacement over22s. Final kick acceptance additionally required each leg separately to pass>=80% strict chains in Isaac and>=70% in MuJoCo. A strict chain includes endpoint arrival within60ms,>=80% valid sustained hold, and a direct post-kick step. These timing/pose criteria were not relaxed. Aggregate step rates are reported alongside direction-specific failures; an aggregate pass does not establish every command case.

| Cumulative updates / stage | Isaac steps | Isaac kick chains | MuJoCo steps | MuJoCo kick chains | Isaac/MuJoCo falls | Forward m/22s |
|---|---:|---:|---:|---:|---:|---:|
| 0 / walking walk_baseline | 1472/1472 | — | 248/248 | — | 0/0 | 1.71 |
| 100 / walking walk_gate_100 | 1472/1472 | — | 248/248 | — | 0/0 | 1.59 |
| 350 / kick_acquisition dance_gate_250 | 1100/1120 | 4/352 | 171/192 | 1/56 | 0/0 | 1.05 |
| 700 / kick_acquisition dance_gate_600 | 1079/1120 | 18/352 | 187/192 | 13/56 | 0/0 | 0.94 |
| 950 / symmetric_acquisition dance_gate_250 | 1098/1113 | 198/346 | 178/192 | 0/56 | 4/0 | 1.20 |
| 1275 / hold_transfer dance_gate_325 | 1086/1120 | 84/352 | 180/192 | 0/56 | 0/0 | 1.06 |
| 1700 / hold_transfer dance_gate_750 | 1114/1120 | 186/352 | 184/192 | 8/56 | 0/0 | 1.03 |
| 1700 / armature_transfer dance_transfer_baseline | 1120/1120 | 105/352 | 184/192 | 8/56 | 0/0 | 1.03 |
| 1825 / armature_transfer dance_gate_125 | 1120/1120 | 30/352 | 186/192 | 15/56 | 0/0 | 1.06 |
| 2000 / armature_transfer dance_gate_300 | 1120/1120 | 154/352 | 189/192 | 4/56 | 0/0 | 1.13 |

Forward comparisons use the same0.12m/s command. Initial lateral/yaw probes used±0.08m/s and±0.15rad/s; hold-transfer and later probes use±0.06 and±0.10. Earlier Isaac evaluations used nominal joint armature0; armature-transfer evaluations use0.01. Thus the latter are a different, explicitly matched dynamics condition, not a pure policy-only comparison. All MuJoCo qualification uses its unchanged official dynamics.

The first100 updates did not improve nominal walking speed over its source (1.59m versus1.71m forward/22s), but retained clean steps and passed changing-input/deployment checks. Direct untrained kick transfer initially produced42 Isaac falls. Learning eliminated those falls but initially taught only one kick side. The symmetry wrapper subsequently produced relaxed endpoints on170/173 FL and172/173 FR trials; timing and sustained holds remained incomplete. Four later falls were confined to lateral±0.08 cases; the reduced-range probe was fall-free.

Two subsequent training children were intentionally stopped **before main/PPO began**, to switch to symmetry and then inertia matching. Their supervisors record `failed` from process termination; `bilateral_review_stop.json` and `armature_review_stop.json` document the scoped review stops. No PPO updates occurred in those stopped children. Closed phases were not restarted. Counts above come from completed training result files and sum exactly2000.

## Actuator-inertia transfer finding

The port delay has a measurable dynamics component. With the same frozen policy and zero travel command, single-parameter diagnostic copies of the official MuJoCo model gave:

| MuJoCo configuration | Strict kick chains | Median endpoint delay | Steps / falls |
|---|---:|---:|---:|
| Official model | 0/7 | ~120ms | 24/24,0 |
| Joint friction loss set0 (normally0.2) | 3/7 | ~60–63ms | 24/24,0 |
| Passive damping set0 (normally0.1) | 0/7 | ~97ms | 24/24,0 |
| Armature set0 (normally0.01) | 7/7 | ~17ms | 24/24,0 |

These altered models are **diagnostics only**, not qualification results. The official file is unchanged. This isolates sensitivity in one policy/command case; it does not establish the correct real-Go2 inertia.

Physics-engine readback confirmed prior nominal Isaac armature0 and training range0–0.015 (observed mean0.00752). The final300 updates used nominal Isaac armature0.01 and training0.008–0.012 (observed mean0.010005). This improved some MuJoCo gates but did not produce reliable cross-simulator kick chains; more training under the old mismatch was not a justified default. The final checkpoint retained1120/1120 Isaac and189/192 MuJoCo steps but only154/352 and4/56 strict kick chains.

Adaptive learning rate remained active: the last150-update dance segment ranged1e-5–5.77e-4 and spent only one update at the floor. Final saved LR after rehearsal was2.56e-4 and action noise std0.658. The evidence does not support a frozen optimizer or collapsed exploration as the main explanation.

## Implementation and deployment checks

The portable runtime now selects the shared walking or dance controller from metadata. Actor/critic/action dimensions stay75/90/12; actor inputs remain encoders,IMU,reference/phase/command state, without privileged simulator root measurements. Body-position traces were added for direct displacement measurement. Existing joint mappings,kp25,kd0.5,zero feedforward torque,command freshness and latched sensor-fault behavior are preserved.

- 109 component tests and12 deployment/metric tests passed;26 core tests passed after reward-scale validation.
- Walking and experimental dance each passed1500-frame worker replay with exactly0rad target difference. Maximum ONNX/JIT action differences were3.34e-6 and3.81e-6.
- Both passed actual400-packet Quad-SDK ROS2 shadow replay, including the sensor-loss latch and clean shutdown. Each produced402 commands including the freshness grace period. DDS was isolated to localhost domain77; no robot was connected.
- Both portable bundles completed200-sample MuJoCo smoke tests without faults or falls. Export bundles have source/policy/report SHA-256 manifests and relocatable paths. Original walking checkpoint hash matches its archived selection.

The adapter remains restricted to `/cpg_sim/` shadow topics. Real hardware still needs its Quad-SDK controller mode,raw-force calibration,estimator/gain/latency validation and airborne fault recovery. All physical evaluations here are flat-ground simulation. The user's future multi-root Quad-SDK workspace can be integrated against the preserved75-input interface; no changes were made to that external workspace during this experiment.

## Next technical step

Retain the successful walking result and the original finished dance. For traveling dance, calibrate actuator response and training/evaluation dynamics before another broad reward sweep, then train with meaningful travel from the walking stage and protect both kick holds explicitly. Inspect the favorable forward video alongside the failed command cases; do not treat visible displacement or one good video as general deployment readiness.

Reproducibility: `results_summary.json`,`learning_audit.json`,`final_video_metrics.json`,`selection_review.json`, per-phase settings/checkpoints/source snapshots, and final `source_final_manifest.json` under `logs/cpg_travel_curriculum_20261101/`. The original four-hour deadline and2000-update budget were respected; no automatic extension or restart is scheduled.
