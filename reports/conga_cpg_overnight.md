# Go2 CPG residual conga: autonomous session, 2026-11-01

Stable 90 BPM stepping and alternating kick–hold–step choreography were demonstrated in simulation. The selected walking and dance policies are separate checkpoints. All earlier phase-conditioned and hierarchical tasks, logs and checkpoints were preserved. This report distinguishes implemented functionality from measured behavior; task registration alone is not qualification.

## Inspected videos

Exactly two final clips, each 22 simulated seconds (21.98 s encoded at 50 fps):

- [Marching: both diagonals, M policy with final walking controller](../logs/conga_cpg_session_20261101/video_selected_walk/video/rl-video-step-0.mp4): 31/31 complete steps, zero falls/resets.
- [Traveling dance with alternating kicks and bounce, U policy](../logs/conga_cpg_session_20261101/video_selected_dance/video/rl-video-step-0.mp4): 24/24 steps and 7/7 complete kick–hold–landing chains, zero falls/resets. Command is +0.04 m/s forward, +0.10 rad/s yaw.

I inspected raw frames for both walking diagonals, both raised front legs and both post-kick landings, and checked the complete recordings' contact traces. Both clips end in four-foot support. They include settling and multiple complete phrases; neither cuts off before the robot has time to step. The dance clip still shows the measured body-height dip during kicks. Per-video `video_review.json` files retain duration, frame indices and audit findings. Earlier diagnostic clips remain preserved but are not the selected final videos.

## Selected results

The final walking test used 64 environments, seed 9051, observation noise, eight commands, and 3,100 control steps (62 simulated seconds). Episodes last 60 seconds; the recording includes the beginning of the next episode. All 64 first episodes reached their time limits without falling. The dance test used the same dimensions/horizon with seed 9058. Neither used pushes, terrain variation or mass/friction randomization.

| Measurement | Walking M with continuous reference | Corrected traveling dance R |
|---|---:|---:|
| Expected diagonal STEP sequences completed | 5,680/5,680 | 4,288/4,288 |
| FL/RR sequences | 2,864/2,864 | 2,144/2,144 |
| FR/RL sequences | 2,816/2,816 | 2,144/2,144 |
| Physical falls | 0 | 0 |
| Median absolute touchdown error | 12.4 ms | 12.7 ms |
| 95 th-percentile absolute touchdown error | 20.0 ms | 25.7 ms |
| Maximum absolute touchdown error | 46.7 ms | 33.3 ms |
| Median / 95 th-percentile body tilt | 4.06° / 7.19° | 2.89° / 6.24° |
| Mean contacting-foot slip | 0.01524 m/s | 0.00867 m/s |
| Median body height | 0.2993 m | 0.2966 m |

Timing is sampled every 20 ms; numerical timestamp precision is not sensor temporal resolution. The independent audit requires one verified swing/landing per selected foot, the opposite pair retained, and touchdown within ±60 ms. It counts **all expected opportunities**, including ones a scheduler might suppress. No repeated swing or support-pair departure occurred in these selected evaluations. Walking median liftoff error was approximately 34 ms and median physical air duration 160 ms against a 200 ms reference swing.

Walking commands were zero, ±0.20 m/s forward, ±0.20 m/s lateral, +0.30 rad/s yaw, and two combined diagonal/turn cases (±0.14, ±0.14 m/s, ±0.30 rad/s). A separate noisy 62-second low-command retention test passed 5,696/5,696 steps. Zero-command XY drift in that test was approximately 0.0047 m/s, and yaw drift −0.0058 rad/s. Requested +0.30 rad/s yaw averaged approximately +0.271 rad/s. The hard 0.5 m/s software cap is **not** a demonstrated operating speed; an earlier 0.25 m/s/0.4 rad/s probe fell repeatedly and was rejected.

Dance commands were zero, ±0.05 m/s forward, ±0.04 m/s lateral, ±0.10 rad/s yaw, and +0.04 m/s forward with +0.10 rad/s yaw. Strict complete kick chains passed **FL 670/672 (99.70%) and FR 669/672 (99.55%)**. Every individual command/leg combination passed at least 97.5%. A chain requires a timely, accurate, slow airborne endpoint, stable three-foot hold, and successful direct landing in the following STEP. All 1,344 eligible post-kick landings passed.

The independent endpoint gate uses ≤2 cm planar error, 5 cm sole clearance ±1.5 cm, foot speed <0.1 m/s, tilt <0.2 rad, body angular speed <0.4 rad/s, confirmed support/air, and arrival no more than 60 ms after the scheduled 0.25-second extension. Hold quality must meet those conditions for at least 80% of the remaining musical hold, with no confirmed selected-foot touchdown. This is stronger than merely reaching the endpoint once, but does not mean every threshold holds continuously at every sample.

Separate first-attempt stationary-hold qualification, seed 9059, passed **32/32 FL and 32/32 FR** over 0.76 sampled seconds. Continuous strict validity at every sample passed 26/32 FL and 28/32 FR; both median valid fractions were 1.0. Later retries do not replace first attempts. Indefinite holds are not qualified.

Travel speed is less accurate than contact timing. For example, forward dance command +0.05 m/s produces a pause-adjusted average target of +0.0354 m/s and actual +0.0320 m/s. Backward actual is −0.0233 versus target −0.0354; turning actual is about ±0.051 versus target ±0.0709 rad/s. Zero-command dance has approximately +0.00635 m/s forward drift. The travel pause is excluded from the target; the controller does not speed up to compensate.

### Command changes and the final walking correction

A separate test rotated commands within each episode without resetting the robot or musical clock. The original stateless walking reference jumped its Cartesian targets when velocity changed: every 4 s command changes produced 4 falls and 4,682/5,680 correct sequences. Integrating stance positions and freezing each dispatched swing goal removed the jumps, but instantaneous reversals could then drive planted targets outside the leg workspace. That unbounded variant was rejected: its diagnostic had 515 falls. Neither variant received additional PPO training.

The selected walking controller combines continuous stance integration, fixed airborne goals, and bounded command acceleration: **0.075 m/s² XY vector acceleration and 0.15 rad/s² yaw acceleration**. A 62 s,64-environment fresh seed 9070 test changing requests every 4.1 s passed **5,680/5,680 sequences with zero falls**. Changing 4.1 s rather than a whole number of beats exercises multiple gait phases. Median/p 95 touchdown error was 13.1/33.3 ms; maximum was 60 ms within the documented float-roundoff allowance. IK target error stayed below 1.1e-7 m. A fixed-command retention test of the integrated reference at the original seed 9051 also retained 5,680/5,680, zero falls. CPU tests reproduce the original constant-command trajectory and check swing-goal immutability, planted continuity, reset isolation, command bounds and full-profile soft joint-limit reachability.

The unchanged M policy supplied the residuals throughout this correction. **This is a planner correction, not a newly trained walking checkpoint.** Speed response is intentionally gradual: 0→0.2 m/s takes at least 2.67 s and a full +0.2→−0.2 reversal takes 5.33 s. The four-second stress requests may change before planned velocity reaches them; these results do not establish instantaneous speed tracking. Traces record requested, planned and pause-adjusted target velocities separately.

R dance with the same bounded command transitions passed **4,288/4,288 steps, zero falls, and 640/672 FL /639/672 FR strict chains** under 4.1 s changes, seed 9071. Fixed-command R remains the higher 99.7%/99.6% result above. Phase/seed differ from the earlier unfiltered 4 s test, so the percentages are not a controlled causal estimate of the filter's effect.

### Checkpoints and exact replay settings

All paths below are relative to `logs/conga_cpg_session_20261101/`:

| Use | Checkpoint | Complete settings / manifest |
|---|---|---|
| Walking / wider command range | `walk_m_yaw_travel_200/training/model_799.pt` | `selected/walking_settings.json`, `selected/walking.json` |
| Alternating dance with low-speed travel | `dance_r_travel_300/training/model_2299.pt` | `selected/dance_travel_settings.json`, `selected/dance_travel.json` |
| Full dance with bounce | `bounce_u_full_300/training/model_2999.pt` | `selected/dance_bounce_settings.json`, `selected/dance_bounce.json` |
| Earlier stationary dance backup | `dance_q_equivariant_300/training/model_1999.pt` | `selected/dance_stationary_settings.json`, `selected/dance_stationary.json` |

**R requires the corrected kick endpoint reference in the final source**, not its pre-correction training snapshot. Its checkpoint was re-evaluated after that fix without further training. Complete selected settings include every gait field and active reward coefficient, avoiding reliance on changing defaults. Q/R/U use the exact-symmetry actor and require `equivariant_actor: true` in `scripts/cpg_experiment.py`. JIT/ONNX exports contain that wrapper. Generic legacy play/loading code does not automatically construct it.

Evaluation evidence: [walking audit](../logs/conga_cpg_session_20261101/walk_n_integrated_fixed/contact_audit.json), [dance contact audit](../logs/conga_cpg_session_20261101/dance_r_corrected_long/contact_audit.json), [strict kick audit](../logs/conga_cpg_session_20261101/dance_r_corrected_long/kick_audit.json), [separate hold audit](../logs/conga_cpg_session_20261101/dance_r_corrected_hold/hold_audit.json). The per-command kick files are beside the aggregate audit.

## Full-bounce stage U

The final full-dance checkpoint is **`bounce_u_full_300/training/model_2999.pt`**, with complete `selected/dance_bounce_settings.json` and manifest `selected/dance_bounce.json`. It uses the same bounded command transitions. R remains the no-bounce alternative; M remains the wider-range walking policy. Do not substitute one policy's qualified command envelope for another's.

Simply enabling the full 15 mm peak-to-peak reference on R initially degraded stepping to 1,601/2,176 and produced zero qualified kick chains, despite zero falls. A staged curriculum trained 200 updates at 5 mm peak-to-peak,200 at 10 mm, then 300 at 15 mm. The medium stage changed the phase-aware height penalty from −0.1 to −0.5; the full stage retained it. S/T retained every step and more than 99% of strict chains before advancing.

Fresh full-amplitude qualification, seed 9072,64 environments × 62 s with noisy observations and the same eight low-speed dance commands:

- **4,288/4,288 expected steps, zero falls.**
- **FL 670/672(99.70%) and FR 671/672(99.85%) complete kick chains.** Every command/leg combination exceeded 98.8%.
- All 1,344 eligible direct post-kick landings passed.
- The fitted beat-frequency body motion was **12.28 mm peak-to-peak versus 15 mm requested**, about 58 ms late. Mean fitted body height was 0.2971 m; target-height RMSE was 7.31 mm.

The harmonic fit removes offset and linear drift. It measures the beat-frequency component, not the full maximum-to-minimum body excursion or proof that every height change was caused by bounce. The actual waveform still contains gait/kick transients and a dip during kicks. **Exact 15 mm sinusoidal height tracking is not solved.** The no-bounce baseline itself had about 5.17 mm of beat-frequency height variation at a different phase.

Separate first-attempt full 0.76 s holds, seed 9073, passed 32/32 perleg under the 80%-quality gate; continuous strict validity passed 13/32 FL and 10/32 FR, with median valid fraction 97.4% both. Thus bounce retains airborne support, but its continuous endpoint precision is weaker than the no-bounce hold result.

With requests changing every 4.1 s, seed 9074, U achieved 4,272/4,288 steps(99.63%), zero falls, FL 634/672(94.35%) and FR 628/672(93.45%) strict chains. This qualifies the tested bounded-transition behavior under the initial engineering gates, not flawless response to arbitrary commands. A proposed stronger-height V continuation was **not launched**; the remaining work prioritized command continuity, qualification and final artifacts.

Evidence: [full-bounce contacts](../logs/conga_cpg_session_20261101/bounce_u_final_long/contact_audit.json), [kick chains](../logs/conga_cpg_session_20261101/bounce_u_final_long/kick_audit.json), [measured bounce](../logs/conga_cpg_session_20261101/bounce_u_final_long/bounce_audit.json), [isolated hold](../logs/conga_cpg_session_20261101/bounce_u_final_hold/hold_audit.json), [changing commands](../logs/conga_cpg_session_20261101/bounce_u_dynamic/kick_audit.json).

![Full-bounce reference and measured motion](../logs/conga_cpg_session_20261101/bounce_u_final_long/contact_detail.png)

## Implemented architecture and motion

The new `Unitree-Go2-CPG-*` family uses four components: a deterministic measured-feedback scheduler, scheduled Cartesian stance/swing references, Go2 analytic IK with explicit frame/joint-order mapping and limits, and a residual PPO controller. Tasks `Walk`, `Timing`, `Travel`, `Dance`, `Kick-Hold`, and `Bounce` are independently configurable. Detailed contracts and reproduction instructions are in [docs/go 2_cpg.md](../docs/go2_cpg.md).

The actuator target is `clamp(q_nominal + 0.12*tanh(policy_output), joint_limits)`. Nominal targets include bounded load compensation through the stock 25/0.5 PD gains; no physical state teleport is used for kicks. The actor has 75 deployable sensor/reference inputs, critic 90 inputs, and policy 12 actions. The critic additionally sees simulated body velocity and torques. World foot positions/velocities and CoM are used for rewards/audits, not directly fed to the actor. Scheduler completion uses measured contact and state feedback; hardware would need suitable estimators/adapters. Nothing was deployed to or moved on the physical robot.

Musical tempo remains 90 BPM independently of commands. Each STEP is one diagonal swing and landing; zero command marches. A phrase has STEP touchdown targets at 0, 2/3 and 4/3 seconds, then KICK starts at 5/3 seconds. STEP dispatch precedes touchdown by the configured 0.20-second swing. FR/FL kicks alternate between 8/3-second phrases. KICK extends 8 cm laterally from calibrated nominal to 5 cm sole clearance over 0.25 seconds, then remains raised. The following STEP contains that front leg and lands directly without intermediate retraction.

**The 0.75-second interval from kick endpoint to the next musical touchdown includes the landing swing:** with a 0.20-second STEP it provides a 0.55-second stationary hold. A separate full 0.75-second stationary hold was tested, but adding that plus landing to the musical phrase would miss the beat. Neither phrase duration nor kick timing was silently stretched.

Walking now integrates stance trajectories and fixes each airborne endpoint at dispatch. Operator changes are acceleration-limited before reference generation and velocity rewards, while the musical clock remains independent. These defaults were adopted only after the fixed- and changing-command tests described above.

The planted support transfer moves the body reference 3 cm backward and 1.5 cm toward the support side in the existing pre-kick gap. It keeps the bias through the raised entry and restores it after four-foot support. The calibrated neutral model has mass 16.087 kg and CoM approximately (−0.0008, 0, −0.0198) m; simulation confirmed the near-zero fore-aft offset. Weighted support forces, encoder/IMU velocity damping and bounded position bias support three-foot balance. Kick body-frame endpoint geometry remains fixed even under travel/yaw commands.

Initialization uses grounded nominal IK joints and waits for measured settling before residual control. Contact detection uses 5/2 N hysteresis and 40 ms confirmation; verified swing additionally requires >8 mm sole clearance. Confirmed touchdown retains its onset timestamp. Failed/missed events receive no completion credit. Default recovery gradually lands and rejoins a future event while preserving musical and physical episode clocks. Configurable `land_and_stop` / `hold_and_stop` are implemented but not physically qualified recovery guarantees.

## Experiments and diagnoses

Every substantial run used 4,096 environments, 24-step rollouts, immutable named output directories, saved seeds/configs/source/Git identifiers, and periodic checkpoints. Branches preserve their sources. The table summarizes meaningful experiments; startup/API fixes and short bounded validation jobs remain in the session directory.

| Branch | New PPO updates | PPO wall time | Change and outcome |
|---|---:|---:|---|
| A | 200 | 142.7 s | Higher-gain uncompensated reference remained planted: 0/1,856 expected sequences, zero falls. Rejected. |
| B | 200 + 300 | 367.5 s | Stock gains, gravity-load position bias and 0.20 s swing taught actual paired lifting. B 200 passed 1,856/1,856; continuation preserved stepping but increased slip. Selected earlier B 200 for subsequent work. |
| C | 200 | 144.7 s | Stronger, narrower XY/yaw tracking improved zero-command drift; noisy 62 s timing test passed 5,696/5,696. |
| D | 200 + 300 | 350.6 s | Introduced alternating kicks. Relaxed endpoint performance switched between sides; full hold-chain gates failed. |
| E | 200 | 162.5 s | Soft sagittal mirror loss improved bilateral endpoints, but neither side passed strict stable-hold chains. |
| F | 200 | 160.6 s | Explicit endpoint hold-quality reward: FL passed, FR still failed. |
| G | 200 | 161.6 s | Support velocity damping: clean evaluation FL 100%/FR 90%, but noisy long test FR 10.9%. General walking retention regressed, so walking checkpoint kept separately. |
| H / I | 200 each | 324.8 s combined | Isolated long holds / mixed long-hold curriculum produced extended poses but regressed musical precision. Rejected for choreography. |
| K | 200 | 136.7 s | Expanded walking to 0.20 m/s and 0.30 rad/s. Found and corrected three float 32 boundary false failures with a 10 µs numerical allowance; physical 60 ms tolerance unchanged. |
| L | 200 | 162.7 s | Angular-speed hold reward: FL 81.9%/FR 33.4%, still unqualified. Planned continuation was not launched. |
| M | 200 | 135.3 s | Refined walking yaw tracking from K. Selected walking result above. |
| P | 200 | 164.0 s | Calibrated CoM and planted support transfer. Remaining one-sided failure exposed asymmetric pre-roll observations. |
| Q | 300 | 232.1 s | Fixed mirrored pre-roll targets/features and trained an exactly equivariant actor. Noisy stationary chains reached about 95% both sides. Freezing the local HOLD phase additionally enabled the separate 0.75 s hold; musical clock continues. |
| R | 300 | 241.1 s | Low-speed traveling dance from Q. Independent per-command audit caught forward/yaw stride leaking into kick endpoint x. Fixed endpoint and re-evaluated existing R: >99% strict chains, all steps, no falls. |
| S | 200 | 157.0 s | Small-bounce curriculum: all steps and >99% strict chains. |
| T | 200 | 156.5 s | Intermediate bounce plus phase-aware height weight −0.5: all steps and 639/640 chains. |
| U | 300 | 234.8 s | Full 15 mm reference: final physical and contact results above. |

Comparisons are engineering diagnoses, not isolated causal studies across every branch. In particular, Q combines an initialization correction with exact actor symmetry; its gain cannot be attributed to symmetry alone. Deterministic references/IK passed geometric checks, but zero-residual dance was not independently stable: learning and compliant support compensation remain necessary.

## PPO, rewards and validation

PPO uses 256/128/128 ELU actor/critic networks; 5 epochs, 4 minibatches, clipping 0.2, gamma 0.99, GAE 0.95, entropy 0.01, value coefficient 1, gradient norm 1, and adaptive KL target 0.01. Initial learning rate is 1e-3 and Gaussian action standard deviation 0.5. Resume preserves optimizer state, advances the saved update index by one, and initializes the adaptive LR scalar from the saved optimizer LR. Actor observations are not normalized. The exact symmetric actor averages the MLP prediction with its reflected prediction and remains differentiable.

Selected M final LR was 3.844e-4, last 50 mean 3.006e-4, action std 0.493. Selected R final LR was 2.563e-4, last 50 mean 2.023e-4, action std 0.804. U final LR was 2.25e-5, last 50 mean 9.66e-5, and action std 0.643. The adaptive rate decreased, but action exploration did not collapse. No manual LR increase was needed to obtain the selected contact results. [Learning diagnostics](../logs/conga_cpg_session_20261101/selected/learning_diagnostics.json) retain min/max and final values.

Main per-second reward weights: contact +1, clearance +2, velocity +0.5, yaw +0.25, body-height −0.1, residual −0.05, foot-slip −0.5, action-rate −0.01, joint-velocity −0.0002, torque −0.0002, joint-acceleration −2.5e-7, joint limits −10, energy −2e-5, flat orientation −2.5, base angular velocity −0.05, undesired contacts −1. Event timing uses weight +0.25 with dt-normalized event rewards. Stock air-time/variance/joint-position and stock velocity rewards are disabled. M uses velocity +1.5 with 0.1 m/s Gaussian scale and yaw +1 with 0.1 rad/s scale. Dance adds kick tracking +1, endpoint speed −0.1 and hold quality +1. R retains broader velocity/yaw scales 0.25/0.4; speed precision is secondary to contacts. Exact selected files list all coefficients.

The hold reward combines planar endpoint error (0.02 m scale), sole-height error (0.015 m) and selected-foot speed (0.08 m/s). Its angular factor was tested only in L and is disabled in Q/R. Bounce experiments make the mild body-height penalty track the scheduled sinusoid; zero-bounce behavior is unchanged.

Latest complete validation: **106 component and legacy regression tests passed**, including IK/FK geometry, trajectory/schedule, mirrored initialization, bounded residual/preparation gate, contact timing, support shift/kick endpoint under travel and long holds, and phase-aware height cost. Fresh/resume PPO, JIT and ONNX smokes passed. Final physical analytic-FK discrepancy was at most about 4.1 µm for walking /2.6 µm for dance; IK target error was about 1e-7 m. These checks validate software contracts, not hardware safety.

### Expected versus measured contacts

![Walking expected and measured contacts](../logs/conga_cpg_session_20261101/walk_n_integrated_fixed/contact_detail.png)

![Dance expected and measured contacts](../logs/conga_cpg_session_20261101/dance_r_corrected_long/contact_detail.png)

Plots show measured sole clearance as well as contact state, making planted sliding distinguishable from airborne stepping. Orange/purple shading denotes kick/hold intervals. A displayed trace is illustrative; selection uses all-environment independent audits.

### Learning curves

![Walking learning diagnostics](../logs/conga_cpg_session_20261101/learning_curves_walking.png)

![Dance learning diagnostics](../logs/conga_cpg_session_20261101/learning_curves_dance.png)

Stored indices overlap between branches. Training reward and training contact fractions are diagnostics, not substitutes for held-out selection gates. Different reward variants' total returns are not directly comparable.

![Bounce learning diagnostics](../logs/conga_cpg_session_20261101/learning_curves_bounce.png)

## Runtime, persistence and preserved evidence

Actual training totaled **4,508 PPO updates and 442,392,576 environment transitions**: 4,500 updates at 4,096 environments plus four two-update smoke runs with 128 environments. Summed PPO-loop wall time was 3,440.1 s(**57.3 minutes**), excluding simulator startup, evaluation, exports, implementation and reporting. Substantial-run aggregate throughput was approximately 119,895–145,322 environment transitions/s. Hardware was an NVIDIA GeForce RTX 5070 Ti(16 GB), driver 580.178.04; installed versions are saved in environment.json. This is aggregate parallel throughput, not single-environment FPS. Per-run timings, seeds, sources and sample counts are in [training_totals.json](../logs/conga_cpg_session_20261101/training_totals.json).

Two initial preflights failed on installed-wrapper/export API mismatches and were corrected in new output directories before substantial learning. Later completed runs did not report NaNs or GPU OOM. Failed experiments, rejected policies and all source checkpoints remain available. Final inventory and source hashes are saved in the session directory. Two CPU runner guards verified rejection of a closed session and a budget exceeding four hours, without launching a child.

The persistent tmux supervisor ran one GPU-intensive child at a time. It saved immutable job plans, source/config/seed/Git snapshots and 100-update periodic checkpoints; child processes inherit the GPU/session locks. It preserves the original 06:45–10:45 UTC wall-clock budget and reserves evaluation/shutdown time. A failed job is not retried indefinitely. The four-hour timeout was not reached; the session's actual end time is recorded in session.json. No shared Isaac Lab files or installed packages were modified, and no physical robot was contacted.

## Remaining limits and next recommendation

The main objective—stable 90 BPM diagonal stepping with measured touchdown timing—was achieved in the tested flat-ground simulation. Alternating kicks, airborne holds and direct post-kick landings also passed the stated engineering gates. Bounce is implemented and trained, with stable contacts, but its exact physical amplitude/waveform still needs refinement.

The next useful training experiment is **command changes during training**, using the now-qualified bounded planner and a held-out phase-varying command schedule. Current PPO runs sampled commands per episode; changing-command tests were evaluation-only. Prioritize retaining all contacts and improving the remaining dynamic kick-chain failures before broadening speed or adding disturbance randomization. Then address residual kick-related height dip and bounce phase/amplitude with one controlled height/support-model change, retaining R as the no-bounce baseline.

Not qualified: arbitrary rapid/unbounded commands; the 0.5 m/s cap as a learned speed; terrain variation; pushes; randomized mass/friction/actuation; indefinite raised-leg holds; comprehensive recovery from every failure; or hardware deployment. Hardware use also requires the scheduler, IK/load model, reference observations, contact/state estimators and validated recovery around the exported residual policy. Existing legacy 55/70 and hierarchical 100/112 policies remain separate and incompatible with this 75/90 interface.
