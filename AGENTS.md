## CPG residual controller: verified architecture and session results (2026-11-01)

This section supersedes earlier CPG progress notes only. Preserve all historical
phase-conditioned/hierarchical implementations, checkpoints, logs and instructions
below. New tasks are `Unitree-Go2-CPG-{Walk,Timing,Travel,Dance,Kick-Hold,Bounce}`.
Full architecture/reproduction: `docs/go2_cpg.md`; final evidence/report:
`reports/conga_cpg_overnight.md`. Session: `logs/conga_cpg_session_20261101/`.
Session work is complete; no CPG GPU process remains. Final videos are
video_selected_walk/video/rl-video-step-0.mp4 and
video_selected_dance/video/rl-video-step-0.mp4 within the session, both22s
simulated/21.98s encoded and ending in four-foot support. Walking31/31steps;
dance24/24steps plus7/7kickchains, no falls/resets. Do not restart automatically.
Original budget starts06:45UTC and ends10:45UTC, four hours including development.
Never extend/reuse that deadline. Read session.json and runner_status.json before
claiming a current job finished. No hardware deployment or physical robot motion.

Control path is deterministic event scheduler -> scheduled Cartesian reference ->
frame/joint-limit-aware Go2 IK with bounded compliant-actuator load bias -> PPO
residual. q_target=clamp(q_nominal+.12*tanh(policy_output),joint limits), stock
25/.5 PD gains. Actor/critic/action75/90/12. Actor uses encoders, IMU and reference/
operator state; privileged body velocity/torques are critic-only. World foot pose,
speed and CoM measurements are reward/audit feedback, not direct actor inputs.
Hardware needs contact/state estimation, observation adapter and recovery validation.

Fixed90BPM (2/3s beat) independent of commands. Zero velocity marches. One STEP is
one diagonal swing/landing, alternating during normal walking. Three touchdowns
per8/3s phrase:0,2/3,4/3s, with .20s advance dispatch. KICK starts5/3s, extends
selected FL/FR8cm laterally from calibrated nominal, reaches5cm sole clearance in
.25s with zero-velocity reference endpoint, and remains airborne. Alternate front
legs each phrase. Next STEP must contain raised front leg and land it directly;
no automatic retraction/intermediate landing. Endpoint-to-next-touchdown interval
is.75s, comprising .55s stationary hold and .20s landing. A full .75s stationary
hold is tested separately; do not silently lengthen the musical phrase to fit it.
Travel pauses during KICK/HOLD; do not increase speed to catch up. XY hardcap.5m/s
is a software bound, not demonstrated capability. Start at0–.15m/s; speed adjusts
stride, never tempo. Backward/lateral/turn-in-place/moving-turn commands supported.

Measured settling precedes residual control: grounded nominal IK joints, minimum
1s physical age plus .25s continuous support/tilt/encoder-velocity/IMU stability;
5s preparation timeout. No raised-state teleport and no physical-age resets at
skill boundaries. Contacts use5/2N hysteresis and40ms confirmation, actual onset
preserved. Verified swing requires40ms air and>8mm sole clearance. STEP success
requires exactly one selected swing/landing, retained opposite support and timing
within±60ms in Timing/Dance (Walk initially100ms).10µs numerical allowance avoids
float32 boundary false failures; physical tolerance unchanged. All expected
opportunities count, including suppressed ones. Reference completion is not success.
Recovery defaults to gradual land_and_rejoin at a future scheduled event; clock
continues. land_and_stop/hold_and_stop exist but are not qualified safe responses.

Three-foot support model uses calibrated CoM(-.001,0,-.02)m and mass16.087kg,
weighted support wrench, encoder/IMU-derived velocity damping6/s, angular damping2.
Planted body transfer is3cm backward and1.5cm toward support in the existing
pre-kick gap, retained through raised STEP then restored in four-foot support.
Kick endpoint remains fixed body-frame nominal+8cm lateral. Corrected a bug that
let traveling/yaw stride contaminate kick endpoint x; current source fixes it.
Isolated HOLD no longer resumes stance travel after musical-clock wrap.

Q/R dance actors use exact sagittal equivariance f_s(x)=(f(x)+R_a*f(R_o*x))/2,
not just a soft mirror loss. Corrected bilateral pre-roll goals/features are also
required. HOLD freezes local gait phase, not musical time. Actor normalization
must remain Identity. Wrapped checkpoints have actor.base keys and require
`equivariant_actor:true` through scripts/cpg_experiment.py; generic legacy play
cannot load them without constructing the wrapper. JIT/ONNX include the wrapper.

The final walking reference integrates planted-foot motion and freezes each
swing endpoint at dispatch. Operator changes are bounded to.075m/s² XY vector
acceleration and.15rad/s² yaw acceleration; musical/episode clocks never restart.
The actor and velocity rewards receive planned velocity; traces separately record
requested/planned/pause-adjusted velocities. Unbounded stateless changes caused
4falls and4682/5680steps; unbounded integration was unreachable and rejected
(515diagnostic falls, no training). With both corrections, same M policy passed
5680/5680steps,0falls in fresh64x62s seed9070,requests rotating every4.1s. Fixed
command retention also5680/5680,0falls. Bounds are deliberately conservative:
0→.2m/s takes2.67s;+.2→−.2 takes5.33s. Do not claim instantaneous speed tracking.
Current task defaults use these bounds and integrated_walk_reference for walking.
Historical complete settings explicitly retain old flags when needed.

Selected walking M checkpoint:
`logs/conga_cpg_session_20261101/walk_m_yaw_travel_200/training/model_799.pt`.
Use `selected/walking_settings.json` (complete gait/reward fields) in that session.
Fresh noisy64envsx62s,60s episodes,seed9051:5680/5680 expected diagonal steps,
zero falls, both diagonals; median/p95/max touchdown error12.4/20.0/46.7ms at20ms
sampling. Commands0,±.20m/s forward/lateral,+.30rad/s yaw and±(.14,.14,.30)
combined. Separate low-command62s check5696/5696,zero falls. Low-command zero XY
bias~.0047m/s,yaw-.0058rad/s. Earlier.25m/s/.4rad/s probe failed; do not claim.5
or.25 qualified. Slip remains (expanded-command mean.01524m/s).

Selected low-speed dance R checkpoint:
`logs/conga_cpg_session_20261101/dance_r_travel_300/training/model_2299.pt`.
Use `selected/dance_travel_settings.json` and corrected final reference code.
Fresh noisy64x62s,seed9058:4288/4288steps,0falls; strict kick-hold-directSTEP chains
FL670/672(99.70%),FR669/672(99.55%),each command/leg>=97.5%. Commands0,±.05vx,
±.04vy,±.10yaw,+.04vx+.10yaw. Actual travel/yaw undershoot targets; contact timing
is stronger than speed tracking. Median/p95 touchdown12.7/25.7ms;mean slip.00867m/s.
Separate seed9059 first-attempt .76s stationary holds:32/32perleg qualify, strict
conditions continuously valid26/32FL,28/32FR. Do not claim indefinite holds.
Strict chain checks endpoint XY<=2cm,height5cm±1.5cm,speed<.1m/s,tilt<.2rad,
angularspeed<.4rad/s,confirmed3support+selectedair,arrival<=60mslate,>=80%valid
remaining hold with no touchdown,then directpostSTEP. Endpoints alone are weaker.
R with bounded4.1s command changes,seed9071,retained4288/4288steps,0falls and
640/672FL,639/672FR strictchains(~95%). Fixed-command performance is stronger.
Stationary Q model1999 remains a backup; selected/dance_stationary.json records it.

Latest106 component/legacy tests pass, plus fresh/resume PPO and JIT/ONNX smokes.
Sources/configs/seeds/Git IDs saved perjob. scripts/run_cpg_session.py runs immutable
plans in tmux, one GPU child, shared inherited GPU/session locks, periodic100update
checkpoints, originaldeadline enforcement, training/eval reserve and bounded kill.
No implicit failed-job retry. Resume preserves optimizer and initializes adaptive
LR from saved optimizerLR, advancing stored index by1. Main runs4096envs,24rollouts,
5epochs/4minibatches,256/128/128ELU,adaptiveKL.01,initialLR1e-3,entropy.01. Select
using held-out contacts/stability, not total reward. Historical partial settings
need archived sources/defaults; selected settings are complete. No shared IsaacLab
packages edited. Existing code reused read-only for support-force allocation.

Full-bounce U checkpoint:
`logs/conga_cpg_session_20261101/bounce_u_full_300/training/model_2999.pt`.
Use selected/dance_bounce_settings.json. S/T/U trained200/200/300updates through
5/10/15mm peak-to-peak references; medium stage height weight-.1→-.5 and the
height target follows musical bounce. Fresh noisy64x62s seed9072:4288/4288steps,
0falls,strictchainsFL670/672,FR671/672. Full.76s first holds32/32perleg qualify;
continuous strict validity13/32FL,10/32FR,medianvalidfraction97.4%. Actual fitted
body beat component12.28mm peak-to-peak versus15mm target,~58ms late,7.31mm height
RMSE; gait/kick transients remain. Exact sinusoidal bounce is not solved. Under
4.1s command changes seed9074:4272/4288steps,0falls,chainsFL634/672,FR628/672.
Keep R as no-bounce alternative and M for wider walking commands. A proposed V
height continuation was not launched; command-reference repair took priority.

All training totaled4508updates/442392576transitions,57.3minutes PPO-loop time,
plus development/startup/evaluation/reporting within the original four hours.
Next recommended learning is bounded command changes during training; present
training sampled fixed per-episode commands. Then improve dynamic kick retention
and exact bounce tracking before expanding speed or physical disturbances.
Remaining scope and final video evidence are recorded in the report/session status. Flat-plane/noise qualification does not establish
rough-terrain, pushes, randomized dynamics, arbitrary speeds or hardware readiness.


# Unitree RL Lab project instructions

## Final-video inspection and longer diagnostics (2026-11-01)

User suspected final clips ended before stepping. Inspected final model399 clips
and traces from logs/hierarchical_long_20261029_173703/. The run actually stopped
for review after200 new updates: success fell from79/155 (50.97%) at100 updates
to0/192 at200. Do not resume automatically. This was a review stop, not completion
of the1000-update budget.

Original clips were10 s each, with3 attempts per diagonal. Each FL/RR attempt
terminated after .24 s and each FR/RL after .12 s due to confirmed required
support-contact loss. Most footage therefore showed2.74 s settling repeatedly;
the video duration was not the cause of the short attempts. No claim is made
that sensor contact loss necessarily means a large physical support-foot lift.

Recorded exactly two new diagnostic clips,20 simulated seconds each, phase/reset
labels and quarter-speed active STEP playback, preserving physics and all failure
rules. New FL/RR has6 failed attempts and23.96 s playback; FR/RL has6 failures and
22.10 s playback. Both statuses passed; decoded frames and labels inspected.
Artifacts: logs/hierarchical_long_20261029_173703/videos_diagnostic_20261101/,
including video_review.json, per-diagonal MP4s and traces. No training or reward
changes. scripts/hierarchical_check.py --diagnostic-video is display-only.

## Authorized calibrated long run (2026-10-29, latest)

User approved restarting from the earlier200-update model and training up to
1000 additional updates with real review gates. Sliding calibration selected
-4.0 (from-2.0): smallest tested coefficient that removes grounded sliding's
advantage over both standing and planted body-shift cases at4/8/12 cm goals,
while preserving reference preference. Synthetic scoring is not a feasibility
proof. Actual baseline reward audit at-4 gave active STEP mean rates:slide-.666,
displacement+.919,placement+.406. Other current rewards/settling/strict gates
remain unchanged. Baseline model is the original aligned-contact model199 at
logs/rsl_rl/go2_hierarchical/2026-10-29_15-49-35_stage_a_20261029_154924/.

Run scripts/run_hierarchical_long.py, persistent tmux go2_stage_a_long.
Live status/source/results: logs/hierarchical_long_20261029_173703/.
Budget4096 envs,1000 additional updates,24-step rollouts,seed42 (98,304,000 samples
if completed), four-hour deadline, no StageB. Train/evaluate sequentially after
exactly50/100/200/500/1000 new updates. Resume from immutable source via private
copies advancing last-saved index by1, preserving optimizer/policy and initializing
the adaptive LR scalar from saved optimizer LR. Baseline LR .000260123; same
adaptive KL schedule. Final index1199 if completed. Segment boundaries reset
physical environments but not policy/optimizer learning. Source checkpoints intact.

Gates use32 envs x1000 steps,seed9041, both diagonal counts and reward/contact
audits. At200+ updates low paired lift or zero success triggers a stochastic
policy diagnostic. Review stops:deterministic fall count/trials>10%;at100+,
>20 percentage-point success drop from a prior>=20% gate;at200+,no verified lift
at two deterministic gates AND none in sampled probe;at500+,no completion in
either probe AND paired lifting<5%. No automatic extension/restart after review.
Two diagonal videos at terminal checkpoint, including review stop. Read status
before claiming completion. All38 component tests,7 guard checks and two-segment
one-update smoke (including stochastic evaluation) passed. Smoke is not learning
validation. Full rationale: reports/go2_hierarchical_long_plan.md.

## Strong-slip result and budget/LR review (2026-10-29)

The4096-env200-update -2.0 slip fine-tune completed. Final model597 still had
0/128 successes, zero falls. Terminal contact-foot mean slip worsened from
.005306 to .013100 m/s versus its source model398; foot error .04628→.07049m.
Increasing penalty20x did not establish a better gait. Latest last50 logged
reward magnitudes:feet_slide-.04983,displacement+.06069,placement+.03408,
clearance+.00153 (episode aggregates, not per-step reward rates).

Adaptive PPO LR is not stuck low: final8.65e-4,last50 mean9.86e-4; only1/200
logged updates was at1e-5 floor. Final action std .515, so do not claim exploration
collapsed. With24-step rollouts200 updates provide4800 steps/env (96 simulated
seconds). Mean training episode is~200 steps/4 s, including~2.74 s settling:
roughly24 attempts/env versus~86 at the earlier short-entry run. This is an
approximation; no active-attempt training counter is logged.

User asked for thoughts on restarting from the earlier200 checkpoint and a
longer budget. Recommendation (not launched/authorized by this discussion):
start from aligned-contact model199,retain settling/clearance/body-height and
strict gates,calibrate slip incentives offline,keep adaptive LR,then a bounded
1000-additional-update4096-env run with gates50/100/200/500/1000 and stop/review
on regression or persistent absent lift. Do not simply increase penalties or
LR based on the iteration count. No new training or reward change in this review.

## Authorized stronger-slip fine-tune (2026-10-29, latest)

User reports improved motion but persistent sliding and explicitly requested a
stronger anti-slide reward plus retraining. Hierarchical feet_slide weight is
now-2.0 versus-.1 (20x); the unchanged inherited function sums horizontal foot
speed when contact-force history exceeds1 N. No force threshold or clearance,
body-height, placement, displacement, action-rate or completion logic changes.
Legacy velocity/conga configs retain their original coefficients.

The preceding clearance/height model398 completed200 updates with0/128 strict
STEP successes and zero falls; mean terminal slip .005306 m/s, foot error .04628m,
body error .02876m. Visual improvement does not establish complete stepping.
Resume this latest checkpoint:
logs/rsl_rl/go2_hierarchical/2026-10-29_17-08-09_stage_a_20261029_170754/model_398.pt.
Budget4096 envs,200 additional updates,24-step rollouts,seed42 (19,660,800 samples).
Standard resume indices398..597; eval450/500/597 after53/103/200 new updates.
38 tests and a250-step x16 checkpoint/export smoke passed; actual reward-manager
log confirmed feet_slide=-2.0. Persistent tmux go2_stage_a_strong_slide;
status/source/config/results: logs/hierarchical_stage_a_20261029_171915/.
Supervisor repeats preflight, trains, evaluates held-out32x1000-step cases,
audits contacts and records exactly two diagonal videos. Keep settled entry and
strict single-swing gates. No StageB or automatic budget extension. Compare slip
AND genuine swing/completion to avoid calling standing an improvement. Read live
status before claiming completion; sparse updates requested.

## Authorized clearance/height training (2026-10-29, latest)

User explicitly requested training the installed clearance, gated placement and
mild body-height rewards. Launch:4096 environments,200 additional PPO updates,
24-step rollouts,seed42 (19,660,800 new samples). Resume the earlier lifting
model199 from logs/rsl_rl/go2_hierarchical/2026-10-29_15-49-35_stage_a_20261029_154924/,
not the later sliding policy. Keep settled initialization and strict single-swing
completion. Standard RSL index199..398 gives200 new updates; final model398.
Evaluate saved250/300/398 on held-out seed9041,32 environments,1000 steps;
run contact audits and record exactly two diagonal videos. No StageB advancement
or automatic budget extension. Four-hour deadline, one GPU child at a time.
Persistent tmux:go2_stage_a_clearance. Live status/source/config/results:
logs/hierarchical_stage_a_20261029_170754/. Read status.json before claiming
completion. Previous38 component tests passed; supervisor repeats normal-entry
preflight before training. Preserve original checkpoints and sparse updates.

## Clearance and mild height shaping installed (2026-10-29, latest)

User approved explicit clearance feedback and requested a mild body-height
deviation penalty. STEP now has clearance+1.0: signed reduction in absolute
selected-foot height error versus a grounded baseline, normalized by4 cm and
clamped[-1,1], averaged across selected feet. Grounded feet earn0, correct height
gets positive credit, overshoot loses credit, and staying airborne after requested
touchdown is penalized. It does not reward arbitrary air duration.

Placement remains+1.5 but selected STEP feet receive full placement credit only
when achieving the vertical reference. Gate blends near ground over0–8 mm;
correct reference tracking and landing retain full credit. It is clearance-based,
not a force-contact threshold, to preserve continuous lift feedback. KICK and
STABILIZE placement are unchanged. Body displacement+1 is unchanged.

New body_height weight-.1 tracks the existing .32 m body reference (including
configured bounce), with1 cm deadband and quadratic5 cm normalization, capped4
before weighting. A5 cm height deviation costs-.064/s;10 cm costs-.324/s;
maximum-.4/s. It is symmetric for excessive raising/lowering and is masked
through the existing settling/scripted-preparation gate. No extra roll/pitch
constraint. New helpers: skills/shaping.py; active hierarchical configs only.

38 component/config tests passed. A300-step x16 no-learning reset/export smoke
passed before the final near-ground gate refinement; final refinement has a
specific regression test. Offline exact-helper scoring gives synthetic changed-
term totals2.000 reference,.291 grounded slide,1.103 half-height,1.145 one-foot,
1.211 double-height,-.033 crouched slide. Physical feasibility/full PPO returns
are not established by these scores. See logs/hierarchical_clearance_height_scoring/
and scripts/score_hierarchical_clearance_height.py. These rewards are untrained;
no further long training was launched on this reward-edit request.

## Single-swing fine-tune result (2026-10-29)

Completed200 additional updates with4096 environments; final model398. All
three held-out checkpoint probes250/300/398 had0/128 successes and zero falls.
Final FL/RR0/63,FR/RL0/65; all128 terminal trials recorded zero verified swings.
Selected sole clearances stayed below1.95 mm and contact fractions were at least
99.96%. Final foot error .0376m, body error .0407m, contact-foot slip .01039m/s.
The user correctly noted sliding rather than stepping. Stricter completion
prevents false success but has not taught lifting. Reward configuration still
has body displacement+1 and 3D foot placement+1.5, contact+.5, support+.3;
stock air-time and variance are disabled, no explicit air-duration reward exists,
and inherited feet_slide is-.1. Body/XY progress rewards are not gated on swing.
No additional correction/training authorized by this results question.
Results/videos: logs/hierarchical_stage_a_20261029_163708/.

## Authorized single-swing correction and fine-tune (2026-10-29, latest)

User requested another correction and training. STEP now requires exactly one
verified swing and one verified landing per selected foot, and retention of the
opposite support pair. A verified swing requires >=40 ms without contact and
>8 mm sole clearance; its latch clears only on confirmed landing. Raised entry
counts as the first swing. A second verified swing or >=40 ms support-contact
loss immediately fails the trial (existing integrated -2 failure reward). One
sample contact chatter is tolerated. Added counters are diagnostic only; policy
100/112/12 layout is unchanged. Settled initialization, aligned contact reference,
other reward weights and PPO settings are retained. No extra lift reward added.
35 tests and one bounded learning/save-load/export smoke passed.

Fine-tune from the earlier aligned-contact model_199.pt that can lift from rest,
not from the later planted fresh model. Source:
logs/rsl_rl/go2_hierarchical/2026-10-29_15-49-35_stage_a_20261029_154924/model_199.pt.
Budget4096 environments,200 additional updates,24-step rollouts,seed42
(19,660,800 new samples). Standard RSL resume uses stored index199, hence new
updates199..398 (200 actual updates), final model_398.pt. Evaluations at saved
250/300/398, approximately52/102/200 new updates, with held-out seed9041 and
32x1000-step probes plus contact audits and exactly two diagonal videos.
Live status: logs/hierarchical_stage_a_20261029_163708/status.json;
tmux go2_stage_a_single_swing. Four-hour deadline; no automatic extension or
StageB progression. Launch is not completion. Original checkpoints preserved.

## Settled-entry experiment completed (2026-10-29)

The user-requested initialization fix,4096-env200-iteration fresh retrain,two
videos and analysis are complete. All64 preflight entries waited until2.74 s;
33 tests passed. Final fresh policy:0/128 STEP successes (FL/RR0/63,FR/RL0/65),
zero falls, no selected-foot lifts; same failure at50/100. The reset drop no
longer overlaps the policy step, but fresh training learned planted behavior.
Extra no-learning comparison of the prior policy from settled starts:99/155
executor successes, zero support-contact departures, but140/155 trials had
repeated >8mm airborne excursions. Reset impact was not the sole jitter cause.
Neither policy qualifies as clean single-step behavior. No StageB/extension.
Full results/checkpoint/two videos: reports/go2_stage_a_settled.md and
logs/hierarchical_stage_a_20261029_160817/. Keep settled initialization; review
single-swing/support-retention criteria before another experiment.

## Settled-entry Stage A retraining (2026-10-29, latest authorization)

User identified the reset drop and requested fixing the first-step entry, then
retraining, exactly two diagonal videos, and analysis. Normal startup now holds
stock default-pose actions and exposes the existing preparation flag until a
measured settling gate passes: age >=1 s, continuous .5 s four-foot confirmed
support, tilt <.15 rad, full body speed <.05 m/s, full angular speed <.2 rad/s,
and every foot speed <.05 m/s. Any invalid sample restarts confirmation. A 5 s
settling timeout or physical fall still terminates; physical episode age is not
reset. The old .3 s STABILIZE timer cannot dispatch or block the new gate.
Task reward terms are off during exogenous settling; inherited physical costs
remain. Foot trajectories, aligned contact rewards and active-skill criteria are
unchanged. No root/joint state teleport is added.

33 component tests and a 300-step x16 simulation/export check passed. 64/64
first-entry checks at seed9042 passed; all admitted at2.74 s. Same-seed9040
preflight: old entry at .32 s with mean body/angular speeds .274 m/s/.592 rad/s;
new entry at2.74 s with .0265 m/s/.0262 rad/s. Launch scope: fresh seed42,
4096 environments,200 iterations,24-step rollouts (19,660,800 samples), no budget
extension/StageB progression. Settling consumes more rollout samples, so equal
sample budgets do not provide equal numbers of active skill attempts.
Supervisor: scripts/run_hierarchical_stage_a.py --num-envs 4096 --record-videos,
tmux go2_stage_a_settled. Live status: logs/hierarchical_stage_a_20261029_160817/status.json.
Post-training evaluate50/100/199, record FL/RR and FR/RL views, and analyze
support-contact losses plus repeated airborne excursions. Do not claim completion
until status/evaluations/videos are checked.

## Aligned-contact Stage A result (2026-10-29)

Completed the authorized fresh 4,096-environment, 200-iteration run (19,660,800
samples). Checkpoints 50/100 had 0/353 executor successes; final checkpoint 199
had 607/607 (FL/RR 306/306, FR/RL 301/301), zero falls and zero missing target
touchdowns in the held-out seed-9041 forward-step evaluation. Final foot/body
errors were .01729/.00956 m. This is evidence of learned lifting/landing under
relaxed Stage A completion, NOT clean diagonal stepping or musical timing.
Every FR/RL trial had at least one supporting-foot departure; all FL/RR trials
had none. The contact-sequence metric passed only 306/607 (50.4%). Most FL/RR
trials also had 2–3 selected-foot contact onsets, so even that metric is not an
exact one-swing gate. Mean slip was .02822 m/s; mean completion .709 s versus
approximately 1 s requested duration. Timing gates are off at Stage A.
Do not advance automatically: inspect videos/contact traces and tighten the
one-swing/support-retention acceptance criteria before claiming Stage A solved.
Stage B has not started. Results: logs/hierarchical_stage_a_20261029_154924/evaluations.json.
Final checkpoint: logs/rsl_rl/go2_hierarchical/2026-10-29_15-49-35_stage_a_20261029_154924/model_199.pt.
Environment count increased as well as contact-target correction, so this run
cannot isolate which change caused improvement.

## Authorized aligned-contact Stage A run (2026-10-29, latest)

User approved the audit recommendation and requested 4,096 environments. STEP
contact targets now derive from the actual vertical reference: desired airborne
fraction clamps reference sole clearance / .008 m to [0,1], masked to selected
feet. This gives a soft 0–8 mm ground transition and keeps late-swing airborne
reference/contact goals consistent, including raised entries. KICK/STABILIZE
contact logic, trajectories, musical timing, other rewards and PPO settings are
unchanged. No separate height-progress reward was added.

Thirty-two component tests and a 250-step x16 simulation/export smoke passed.
Fresh Stage A: seed 42, 4,096 environments, 200 iterations, 24-step rollouts
(19,660,800 samples). Run scripts/run_hierarchical_stage_a.py --num-envs 4096;
persistent tmux go2_stage_a_aligned. Status/source snapshots/results:
logs/hierarchical_stage_a_20261029_154924/. Supervisor reruns normal-entry preflight,
then training, then checkpoint 50/100/199 evaluation at held-out seed 9041 with
32 environments and 1,000 steps. Four-hour deadline, no automatic extension or
Stage B advancement. Read status.json before claiming completion. The environment
count/sample budget also changed versus the earlier 1,024-environment pilot, so
this is not an isolated causal test of contact alignment. Sparse updates requested.



## Stage A offline reward audit (2026-10-29)

Read-only audit confirms all selected feet remained in contact throughout final
STEP samples. Found contact/reference conflict: contact requests landing after
70% progress while the 4 cm sine-squared reference remains 2.618 cm airborne.
Placement/contact allow substantial planted partial credit; action noise fell
.998 to .295 and final training successes were rare. Proposed next change is
reference/contact alignment first, then a separately controlled dense selected-foot
height-progress term if needed. Synthetic +.5/+1 scores are diagnostic only; no
runtime rewards or PPO settings changed and no further training launched.
Details: reports/go2_stage_a_reward_audit.md; reproducible script:
scripts/audit_hierarchical_stage_a.py. Do not auto-advance to Stage B.

## Stage A pilot result (2026-10-29)

The authorized fresh 1,024-environment, 200-iteration, 24-step-rollout pilot
completed (4,915,200 samples). Held-out seed 9041 evaluations used 32 environments
and 1,000 steps at checkpoints 50/100/199. Each had zero falls but 0/353 STEP
successes and 706 missing target touchdowns. Final diagonal outcomes: FL/RR
0/171, FR/RL 0/182. All final trials timed out; no selected-foot lift was registered
in their terminal records. Final mean body displacement error improved to .0091 m
(from .0752 m at checkpoint 50), but foot placement error remained .0726 m.
This is not learned stepping and does not meet the ~80% Stage A gate. Stage B
was not started. Review reward incentives and exploration before further training;
do not automatically extend the budget or advance stages.

Results: logs/hierarchical_stage_a_20261029_153221/evaluations.json.
Checkpoint: logs/rsl_rl/go2_hierarchical/2026-10-29_15-32-24_stage_a_20261029_153221/model_199.pt.

## Authorized Stage A pilot (2026-10-29, latest)

User approved running Stage A with sparse updates. Fresh policy: 1,024 environments,
200 PPO iterations, 24-step rollouts, seed 42 (4,915,200 samples). Normal-entry
preflight passed 16/16 at seed 9040. Persistent tmux session: go2_stage_a;
supervisor scripts/run_hierarchical_stage_a.py, four-hour deadline, one GPU child.
After training, evaluate saved checkpoints 50/100/199 with 32 environments,
1,000 steps, held-out seed 9041; retain both diagonal success counts. No automatic
Stage B advancement or budget extension. Status and source snapshots live in
logs/hierarchical_stage_a_20261029_153221/. Read status.json before claiming
completion. The user requested updates only at major milestones or problems.

## Raised-leg initializer repair (2026-10-29, latest)

The user requested a stable bilateral raised pose before training. Stage B now
uses support-force allocation through stock PD position targets: a 2 s CoM
transfer to the support triangle, .28 m base target with mirrored .06 rad roll
and -.04 rad pitch, measured unloading, then a 1.5 s lift. No physical state
teleports. Admission requires 1 s continuous stable hold after the lift, including
body speed <.06 m/s, full angular speed <.25 rad/s and all foot speeds <.06 m/s;
existing endpoint/contact/tilt checks remain. Timeout is 10 s.

Repair qualification: seeds 9030 and 9031, 64 balanced first attempts each, with
an extended 3 s hold, passed 128/128 (64 FL, 64 FR), zero resets/falls. Bilateral
videos at seed 9032 also passed. Thirty-one component tests and a 550-step x16
mixed-entry reset/JIT/ONNX smoke passed with 16 prepared entries, zero preparation
failures and zero falls. No PPO learning or overnight run was started. This
supersedes the earlier initializer reliability blocker only within this tested
simulation distribution; policy landing and kick/step transition gates remain
unevaluated. Current request is initializer repair, not automatic resumption of
the overnight experiment. See reports/go2_initializer_repair.md and
logs/hierarchical_initializer_repair/. Preserve historical failures below.

## Overnight experiment authorization and stopped preflight (2026-10-29)

The user authorized an approximately four-hour wall-clock hierarchical experiment,
including validation/evaluation, conditional on Phase 1 passing. This supersedes
older statements below that no long hierarchical training is authorized. Required
order: normal-entry STEP (~80% held-out success), mixed normal/raised STEP (~70%
raised success with normal retention), bilateral airborne KICK, then live
STEP→KICK→STEP. Never advance blindly or treat initialization admission as policy
success. At most two controlled reward/curriculum variations are allowed if basic
stepping is poor; one major factor per variation. One GPU-intensive job at a time,
periodic checkpoints, separate directories, source/config/seed snapshots and an
enforced deadline are required. Stop on invalid preflight, NaNs, OOM or repeated
errors. Preserve the legacy tasks. The user requested persistent tmux execution
only if infrastructure is ready, and a report in `reports/conga_overnight.md`.

The fresh preflight STOPPED before substantial training. At seed 9029, 16 balanced
raised-entry first attempts admitted 4/8 FL and 5/8 FR valid states; seven timed
out during preparation. All admitted joint/contact/diagonal checks passed, with
no preparation falls in this probe. The strict preflight required every sampled
initialization to succeed, so readiness failed. No invalid state was admitted;
56.25% admission is NOT learned STEP success. Thirty-one CPU core/reference tests
passed. The remaining fresh four-foot/PPO validation was not run after the failure.
No overnight PPO iterations, new policy checkpoints or tmux session were launched.

Verified diagnostic command (its readiness result is failed):
`/home/rml2/anaconda3/envs/nick_sillywalk/bin/python scripts/hierarchical_preflight.py --headless --stage B --num_envs 16 --seed 9029 --output logs/conga_overnight_20261029/preflight_raised`
Use a fresh output directory for repeats. Read `result.json`, not just the process
exit code. State/source snapshots/results live in `logs/conga_overnight_20261029/`;
report and admission plot are in `reports/`. Improve and requalify preparation
before resuming the conditional overnight pipeline; do not describe it as running
or claim the 80%/70% policy gates were evaluated. This record supersedes the prior
8-environment initializer probe only for this seed/batch, not as a general fix.


## Hierarchical conga architecture (2026-10-29, current branch)

The current objective on `feature/hierarchical-conga` is a separate hierarchical,
goal-conditioned skill controller. The user's October 29 request supersedes the
legacy motion/neutral-command decisions **for the new `Unitree-Go2-Hierarchical-*`
tasks only**. Preserve all legacy tasks, experiment records and checkpoints below.
The new family marches at zero velocity, uses 90 BPM (`60/90` s per beat), caps XY
vector speed at .5 m/s, and keeps yaw limits configurable. Operator velocity belongs
to the deterministic planner; the policy receives displacement, heading, foothold,
contact and duration goals. Active command updates affect future goals only.

- `STEP` is one diagonal-pair swing and landing. One shared goal-conditioned policy
  accepts either confirmed four-foot support or three-foot support with a raised
  front leg in the commanded pair. No intermediate landing/reset is inserted.
- `KICK` extends FL or FR outward .08 m from the calibrated nominal foot position,
  to .05 m sole clearance, in .25 s. It ends airborne; it does not retract. A smooth
  quintic reference has zero endpoint velocity. Success checks endpoint speed,
  support, pose and clearance. Foot order remains FL, FR, RL, RR.
- `STABILIZE` retains its supplied four-foot or three-foot support configuration.
  Completion is `RUNNING/SUCCESS/TIMEOUT/FAILURE`; feedback includes placement,
  displacement, heading, contacts, stability, elapsed time and raised-foot state.
  Contact force hysteresis (5/2 N) and 40 ms confirmation are configurable.
- Musical beats are **touchdown targets**, per the user's timing clarification.
  STEP commands are sent ahead of the beat (one-beat lead by default). Landings
  target beats 1/2/3 at 0, 2/3 and 4/3 s; kick starts at 5/3 s. All three ordinary
  steps therefore have about a full beat to execute, with a half-beat support gap
  before the kick. Startup includes one beat of pre-roll. The post-kick STEP starts
  directly at the airborne endpoint and has about .75 s to land on the next
  phrase's first beat. Contact confirmation does not shift the measured touchdown
  timestamp. Timing stages require confirmed touchdown onset within .06 s of its
  goal. Configurable landing offsets/lead times preserve the 90 BPM clock.
  Physical feasibility and learned timing have NOT been demonstrated.
- A/B/C/D/E/F tasks respectively implement normal-entry steps, mixed procedurally
  raised entries, isolated airborne kicks, actual-policy kick/step chains, musical
  timing, and full travel/bounce configuration. All share 100/112/12 actor/critic/action
  dimensions. Legacy 55/70 policies are incompatible. Bounce defaults to zero;
  only after transitions qualify may it be manually ramped toward .0075 m amplitude.
- Historical initial Stage B implementation generated states through a slow Jacobian/position-target controller under
  physics and stock actuators, never arbitrary raised joint resets. Admission checks
  measured three-foot support, extension/height and stability. Preparation is an
  observed substate with policy actions ignored. It remains experimental: the final
  bounded probe produced only 3 accepted entries across 8 environments, with 4
  preparation timeouts and 1 fall. The repair qualification above supersedes this initializer result.
- A/C task wiring, completion contracts, planner and PPO/export infrastructure can
  be checked independently of learned behavior. D/E/F require trained skills and
  physical transition/timing qualification. No skill is currently demonstrated
  learned and no long hierarchical training run has been authorized or launched.

Major files: `tasks/locomotion/skills/core.py` contains the tensor executor/planner;
`skills/adapter.py` connects sensors, commands, rewards, preparation and event
feedback to Isaac Lab; `robots/go2/hierarchical_env_cfg.py` holds stage configs;
`agents/hierarchical_ppo_cfg.py` supplies standard PPO settings;
`scripts/hierarchical_check.py` performs bounded smoke/evaluation with pre-reset
feedback and grouped metrics; `tests/test_hierarchical_*` check the contracts.
Paths under `tasks/` are relative to `source/unitree_rl_lab/unitree_rl_lab/`.
Verification: 27 new core/adapter tests and 10 legacy reference tests passed;
all six configs validate. A two-update fresh/save/load PPO smoke and JIT/ONNX
exports passed. The normal train CLI completed one iteration at 16 environments.
Step/kick/dance zero-action smokes correctly produced no successful skills.
Full commands, test evidence, stage limits and the recommended Stage A pilot live
in `docs/go2_hierarchical.md`. Inspect each check's `status.json` and `summary.json`;
simulator shutdown can mask a nonzero Python exception in its process exit code.
Keep new experiments under `logs/rsl_rl/go2_hierarchical*` and evaluation artifacts
under `logs/hierarchical_*`. Existing train/export infrastructure is reused, but
hardware deployment is NOT compatible without a planner, state estimator,
observation adapter and validated recovery controller. Never reset physical episode
ages/timeouts at a skill boundary. The existing environment/package restrictions
and explicit approval requirement for long training remain in force.


## Latest behavior decision (2026-10-28)

Neutral joystick/zero motion command permits standing in all future stages,
including deployment. A neutral-to-active translation or yaw command starts a
fresh step pattern from its beginning. Active-to-active joystick updates must
not restart the pattern; active cadence remains 120 BPM independent of speed.
This supersedes the earlier requirement to march continuously at zero command.
Use raw operator intent for activation, not the dance's internally zeroed kick
travel envelope. Input deadband/hysteresis, safe settling on return to neutral,
and rapid restart handling remain to be agreed before implementation. Do not
reset physical episode age or timeouts when restarting the motion clock.

Next proposed experiment is basic walking with gentle air/contact-duration
shaping, allowing standing at zero command. Timing targets are 0.20 s swing and
0.80 s per-foot stance (0.30 s four-foot gaps between alternating diagonals).
The duration-walking experiment described at the end of this file now implements
and has trained the accepted -.05/-.05 duration weights and tolerance bands.
Simulation phase observations restart at activation; safe deployment transitions
remain unimplemented. Do not claim hardware deployment compatibility.

## Completed experimental decisions (2026-10-28)

This section records the completed balanced-start experiment and supersedes older historical statements
below about pending proposals or training authorization. The final dance reference
in the next section remains unchanged; experimental timings and command caps must
not silently replace it.

- Current approach: learn timed marching from scratch at zero translation and yaw
  before reintroducing travel. Zero command must still produce alternating
  diagonals. Standing is allowed only in the legacy stock/basic control task.
- Retain 0.20 s swings and 0.30 s four-foot support gaps: diagonal opportunities
  every 0.5 s, alternating FL/RR and FR/RL, giving 1 Hz per foot. The experimental
  swing window is [0.12, 0.32] s within each beat, with 3 cm peak clearance.
  The 120 BPM clock is independent of velocity. Bounce and kick remain disabled.
- The accepted reduced travel cap in the recent walking/cadence experiments is
  0.5 m/s XY vector norm. The current march experiment has all commands zero;
  it does not automatically expand the range. The original stock/basic control
  used stock commands; do not describe its first 1,000 iterations as capped at
  0.5 m/s. The final dance's 1.25 m/s target remains a separate specification.
- Current march rewards: normalized phase matching +1.0 (80% swing / 20% support),
  individual clearance +0.25, paired completion +0.25, missed swing -0.5,
  missing pause contacts -0.25, and repeated departure -0.025 per foot event
  (-0.05 per repeated diagonal). Event costs are integrated costs, not plain
  per-second coefficients. No additional four-foot support bonus is installed.
  Pair completion is deliberately ungated to allow exploration; repetition
  penalties remain active. Stock joint-position/air-time/variance rewards are
  off; inherited physical costs and zero-velocity tracking remain active.
- Repetition detection counts each sampled foot departure, including staggered
  departures. Only a confirmed opposite pair clears repetition history: both
  feet at least 5 mm clear for two samples while the other pair supports. Do not
  reintroduce the simultaneous-takeoff loophole or assume contact noise is solved.
- Active task: `Unitree-Go2-Conga-March-Balanced`. Keep the preceding march
  rewards, physical initialization and 55/70/12 observation/action dimensions.
  A private seeded shuffle initially assigns half the environments each 0/0.5 s
  phase offset; every environment toggles its offset on each reset. This balances
  starts per environment over successive episodes, not necessarily the whole
  population at every instant. Observations and references share the offset;
  command resampling does not change it. Physical episode ages/timeouts are intact.
- Authorized budget: fresh policy, 4,096 environments, 1,000 PPO iterations,
  24-step rollouts (98.304M samples), gates at 50/100/200/500/1000. Launch mode:
  `scripts/run_early_experiment.py --march-balanced`. Read live status at
  `logs/conga_march_balanced_4096_experiment/status.json`; launch is not completion.
  Resume only toward that budget. Stop only this supervisor's own child on VRAM
  pressure; resume after at least 8 GiB free continuously for 60 seconds.
  Unrelated errors or deterioration review stops require review. Do not interrupt
  other training, extend this run, or advance to travel/bounce/kick automatically.
- The preceding fixed-start march run completed 1,000 iterations with zero
  controlled-evaluation falls but failed alternation: FL/RR stepped about .22 Hz
  in the first case while FR/RL stayed planted. Contact reward symmetry passed
  the offline audit; changing inference start phase did not switch anatomical
  preference. Balanced starts during learning are an experiment, not a proven fix.
- A history-gated pair reward plus balanced full-cycle bonus was scored only.
  It is NOT installed or approved for this run. Its bonus was zero on all recorded
  final cases, so stronger synthetic separation does not establish learnability.
- Balanced-start final outcome: all 1,000 iterations completed. All five gates
  had zero falls, but deterministic steady-state feet stayed planted. The final
  evaluation missed 2,304/2,304 targets (all per-foot rates zero), compared with
  2,048/2,304 missing in the fixed-start baseline. Last-50 training episodes
  averaged 19.901 s while paired reward remained effectively zero. Do not extend
  training or claim alternation from stable standing. This single-seed comparison
  does not establish a general causal disadvantage of balanced starts. Review
  exploration and reward incentives before agreeing another experiment. Full
  results and checkpoint path are in `docs/go2_conga.md`.
- Before balanced-run launch, 21 adapter tests and a 550-step x16 zero-action
  reset/simulation/PPO/export smoke passed. These establish implementation checks,
  not learned stability. Evaluate actual per-foot contacts, both diagonals, swing
  and pause durations, missing/excess touchdowns, drift, falls and slip. Use logged
  per-row phase offsets in timing analysis and inspect videos before declaring
  success. Keep final results and checkpoint paths in `docs/go2_conga.md`.

## Finalized Go2 conga motion specification

Implement a phase-conditioned PPO task on flat ground using the existing Go2
actuator, action, proprioception, PPO, and export infrastructure. This specification
defines the requested reference motion; physical feasibility and learned stability
remain to be established in simulation. Keep this file synchronized with changes.

- External simulation-time clock: 120 BPM, 0.5 s/beat, 2 s/phrase, 4 s/full
  alternating sequence. Speed commands must never affect clock frequency. Each
  environment resets its own clock. A neutral-to-active operator command also
  restarts the motion pattern; active command updates do not reset it.
- Foot order everywhere: FL, FR, RL, RR. Body coordinates: x forward, y left;
  vertical clearance uses world z above the flat local ground, corrected for the
  0.022 m spherical foot radius in the supplied Go2 URDF.
- Three traveling diagonal swings per phrase: [0.04, 0.40], [0.54, 0.90],
  [1.02, 1.23] s. First phrase: FL/RR, FR/RL, FL/RR; next phrase swaps
  diagonals. The shortened third swing allows touchdown before the kick.
- Kick FR during [1.25, 1.50] s of the first phrase; FL during the same interval
  of the next phrase. Thus the kick front leg belongs to the next diagonal swing.
  Only that front foot swings during the kick; the other three are encouraged to
  support the body. Beat 4 [1.50, 2.00] is recovery, including a deliberate
  diagonal phase reset for alternating kicks. After kick touchdown, contact
  targets relax smoothly to neutral 0.5 by 1.60 s and return to support targets
  over [1.92, 2.00] s. Recovery foot placement is free so resumed travel does not
  conflict with forced four-foot stance. Neutral contacts receive no clearance
  reward or touchdown timing penalty. Contact targets are soft near ordinary
  liftoff/touchdown too, not kinematic constraints.
- The kick uses a sin-squared extension/retraction pulse, zero velocity at start,
  maximum extension, and end. Full amplitude is 0.08 m outward from the nominal
  front-foot body-frame y position; sole clearance peaks at 0.05 m. The lateral
  target is body-relative and vertical clearance ground-relative. Fore-aft foot
  position is unconstrained. Nominal y comes from the default-pose URDF geometry.
- Commands support backward/forward, lateral motion, moving turns, and turns in
  place. Cap the XY command vector norm at 1.25 m/s, not each component separately.
  Smoothly reduce translation AND yaw over [1.10, 1.25] s, hold zero through
  the kick, and restore over [1.50, 1.70] s. No catch-up speed compensation.
  Report traveling speed separately from phrase-average speed.
- Zero operator commands permit standing, with no required stepping or bounce.
  Activation starts the pattern fresh; define safe settling/restart transitions
  before implementing. Internal kick-related command suppression is not a stop.
- Body bounce: sinusoidal 0.015 m peak-to-peak (0.0075 m amplitude), 2 Hz;
  nominal base height 0.32 m. Track height and reference vertical velocity rather
  than suppressing all vertical velocity. Initial reward weights are unvalidated.
- Preserve torque, joint limit, action rate, orientation, slip, and collision
  penalties. Replace velocity-dependent air time, symmetric air-time variance,
  near-zero joint-position penalties, and indiscriminate vertical-velocity cost.
- Curriculum stages: basic standing/walking preparation with stock locomotion
  rewards and commands plus the conga observation layout; then continuous
  fixed-cadence diagonal stepping including zero
  commands and yaw; add bounce; add kick with progressive amplitude; full command
  range and full dance; domain randomization and robustness evaluation. Keep the
  policy observation/action layout identical across stages for checkpoint resume.
  All stages permit standing at zero operator command. The basic stage's
  24-step PPO rollout matches the stock control experiment; later
  stages retain their 48-step rollouts. It retains stock training randomization
  for a controlled comparison, while the existing dance stages remain unchanged.

## Verification and operating rules

Use `/home/rml2/anaconda3/envs/nick_sillywalk/bin/python` (or activate
`nick_sillywalk`) for this project's setup, checks, and training. Do not install
or change packages in the original `isaaclab` environment. Isaac Lab source at
`/home/rml2/IsaacLab` is shared by editable links; do not modify that shared
checkout for this project. Isaac Sim and runtime packages live in the cloned
environment. The original environment's repository link has been restored.

Work incrementally: verify the numerical references, then rewards/commands and
configuration/registration, then simulation/reset/export smoke tests. Do not start
a long training run without explicit user approval. Do not claim a stable learned
gait from a reference test or an untrained simulation rollout. Preserve pre-existing
user edits, especially robot asset paths in `assets/robots/unitree.py`.

Evaluate actual touchdown timing and missing/excess contacts, cadence by commanded
speed, velocity error, kick lateral/height error, bounce error, falls, and slipping.
The short third step, a 0.25 s kick, and 1.25 m/s travel may be demanding; retain
these targets and evaluate feasibility rather than silently changing them.
Stock deployment needs the new clock/phase observations and travel envelope;
exporting weights alone does not make the existing hardware controller compatible.

## Historical experiment record

Entries below are chronological snapshots. Later results supersede earlier
"untrained", "not authorized", or "now running" statements; use the current
decisions above and live status files for present scope. Preserve these records
as evidence, not as instructions to restart old experiments.

Implementation, curriculum commands, evaluation procedures, and the latest
verification record live in `docs/go2_conga.md`. As of 2026-10-27, reference and
adapter tests, configuration/Hydra checks, GPU smoke simulation, and PPO/export
checks passed. A user-authorized walking-only validation run has now completed:
200 PPO iterations with 256 environments in `nick_sillywalk`. Its deterministic
policy remained planted and failed walking/marching/turning evaluation; do not
introduce kick yet. This separate temporary configuration does not change the
final motion specification. Results and proposed reward improvements are in
`docs/go2_conga_walk_validation.md`. No full-conga training has been performed.
A second authorized 200-iteration, 256-environment run tested walking reward v2.
It regressed to early orientation terminations (852 in the fixed-command
evaluation; episodes 0.36–0.38 s). The temporary validation task currently holds
these rejected experimental coefficients for review, not recommended use.
See `docs/go2_conga_walk_rewards_v2.md`; correct the termination incentive before
further training. Final conga motion and curriculum rewards remain unchanged.
The cloned checkout was also verified on rml2 with Isaac Lab 2.3.2 and an RTX
5070 Ti: 10 reference tests, 7 adapter tests, all five configs, an 8-environment
550-step robustness smoke test, and PPO/export checks passed. Local Go2 assets
live in ignored `.local/unitree_ros`; `UNITREE_ROS_DIR` can override that location.
See `docs/go2_conga.md` for the current setup and verification limitations.
An authorized stock `Unitree-Go2-Velocity` control run also completed: 200
iterations, 256 environments, unchanged 24-step PPO rollout (1,228,800 samples).
It failed early; a compatible mature archived policy walked under the same
evaluation. Those checkpoints have very different training budgets and terrain
histories. See `docs/go2_stock_validation.md` for the comparison and videos.
Force-contact crossings and sensor first-contact event counts disagreed in the
mature-policy test; audit this before relying on sparse conga timing diagnostics.
An authorized longer stock run completed with 1,000 iterations, 4,096 environments
and 24-step rollouts (98,304,000 samples). Walking emerged between inspected
checkpoints 200 and 500. The final policy walked forward/backward and turned with
zero falls in 16 controlled 40-second cases, but leftward tracking remained nearly
planted and rightward tracking was slow. A comparable-budget archived checkpoint
was also evaluated. Keep stock rewards for now; further walking validation is
needed before kick. See `docs/go2_stock_validation.md` for metrics and artifacts.
A subsequent authorized `Unitree-Go2-Conga-Basic` run completed 1,000 iterations
with 4,096 environments and 24-step rollouts. It uses exact stock rewards,
commands and dynamics with the ten conga inputs added (55/70 observations).
Basic walking emerged between inspected checkpoints 200 and 500. Final evaluation
had zero falls in the same 16 controlled 40-second cases and broadly comparable
stock tracking, with better lateral motion but worse backward slip. Only this
basic stage was trained; no automatic advancement is authorized. See
`docs/go2_conga.md` for the comparison, checks and video. The final dance motion
specification and later-stage configurations remain unchanged.
The user has authorized cadence training with the reviewed reward fade. Offline scoring
of a bounded repeat of the final basic checkpoint is recorded in `docs/go2_conga.md`.
The proposed two-term contact/clearance structure was accepted for investigation;
The user subsequently approved starting contact +0.25 and clearance +0.5, with
a 0.5 m/s XY vector-norm command cap. These are configured in the separate
`Unitree-Go2-Conga-Cadence-Start` task, preserving the legacy cadence/bounce/kick
configs. Joint-position, air-time and air-time-variance weights fade linearly to
zero over 200 cadence PPO iterations (4,800 control steps, 24 per iteration).
Cadence checkpoints persist fade progress; transferring basic starts it at zero.
Contact +0.25 and clearance +0.5 remain fixed; later coefficient increases are
not approved. Evaluate at 50, 100 and 200 iterations, pausing on deterioration.
Contact alone favors planted feet over the unsynchronized trot. Preserve the
stock physical costs and use the score report before deciding the transition.
The authorized cadence experiment has now completed 1,000 additional iterations
with 4,096 environments, starting from the basic checkpoint. The 200-iteration
fade completed and checkpointed correctly. All fixed-command gates and final
evaluation had zero falls, but forward steps remained ~2 Hz per foot instead of
1 Hz, and zero-command behavior mostly lifted the rear-left foot rather than
marching diagonally. Contact score barely improved while clearance increased.
Do not advance to bounce/kick. Results, checkpoints and videos are in
`docs/go2_conga.md`; reward balance/diagonal completion need review before another
training run. No further training or reward changes have been authorized.
An authorized offline comparison of higher contact weights and paired-diagonal
scores is complete in `docs/go2_conga.md`. Stronger mean-contact weighting retains
partial-foot credit; strict paired clearance rejects the observed rear-foot-only
motion. Whole-pattern contact plus paired clearance is a candidate for review,
not an installed reward change. It has sparser feedback, so learnability remains
unproven. Saved results: `logs/conga_cadence_candidate_comparison/`.
Offline scoring of 0.20-second swing windows (0.30-second four-foot support gaps)
is also complete. At fixed paired-reward coefficients, standing receives more
credit and the reference-over-standing advantage nearly halves; feedback becomes
sparser. The unchanged 3 cm pulse requires 1.8× peak vertical foot speed and
3.24× acceleration. These are reference calculations, not demonstrated stability
or feasibility. See `docs/go2_conga.md`; no active reference/reward changes or
training were performed for this comparison.

The subsequent user-requested reward adjustment is prepared in the separate
`Unitree-Go2-Conga-Cadence-Paired` task: 0.20 s diagonal swings, 0.30 s support
gaps, normalized support +0.10, paired step +0.50, and phase matching +0.50
(80% swing / 20% support). Synthetic cadence-only scores prefer the reference
(1.10) over standing/single-foot stepping (0.25). Ten reference and fifteen
adapter/configuration tests passed; a 220-step no-learning checkpoint smoke
completed without falls. This scheme is untrained. Final conga specification,
Cadence-Start, and basic training rewards are unchanged. Details and limitations
are in `docs/go2_conga.md`; do not interpret offline scores as learned stability.

The subsequently authorized phase-2 paired run completed 200 iterations from
basic with 4,096 environments and 24-step rollouts. Evaluations at 50/100/200
had zero falls, but zero-command stepping repeats FR/RL at 2 Hz with FL/RR
planted; forward gait remains near 2 Hz per foot. Half the scheduled touchdowns
are still missing at every gate. Velocity tracking mostly recovered, but slip
increased. The fade completed with fixed paired weights. Do not advance to
bounce/kick or extend training automatically. See `docs/go2_conga.md` and
`logs/conga_paired_analysis/`; early phase shaping during basic remains an
unperformed proposed experiment requiring a separately agreed schedule.

On 2026-10-28 the user authorized fresh early-shaping training: 1,024 environments,
1,000 iterations, 24-step PPO rollouts. `Unitree-Go2-Conga-Early` starts phase
weight at .05, ramps it to .25 and paired clearance to .10 over iterations
50–200 while fading the three conflicting stock rewards; no support bonus.
Translation ranges expand from +/-0.1 to vx +/-0.5, vy +/-0.25 over the same
window, with a .5 m/s vector cap. Final dance/basic configs are unchanged.
Sixteen tests and bounded PPO/evaluation/export smoke checks passed before launch.
`scripts/run_early_experiment.py` supervises gates at 200/500/1000 and monitors
VRAM, signaling only its own child process on a memory stop. Check live state
in `logs/conga_early_1024_experiment/status.json`; do not assume completion.
Details and regression-stop criteria are in `docs/go2_conga.md`. Do not interfere
with the user's separate training or extend beyond the authorized 1,000 iterations.
The user subsequently requested automatic recovery from VRAM pressure. Recovery
is attached to the running supervisor without interrupting training: after a
memory stop, wait for >=8 GiB free continuously for 60 seconds, then resume the
latest valid checkpoint toward the same 1,000-iteration target. Check
`memory_recovery.json` alongside `status.json`. Unrelated failures and stability
review stops still require review. Up to roughly 50 unsaved iterations may be
replayed; do not confuse replayed compute with an increased iteration target.
The early-shaping run subsequently completed all 1,000 iterations without memory
recovery. All evaluation gates had zero falls. Walking/turning improved by the
final gate, but zero-command stepping repeats FL/RR at 2 Hz with FR/RL planted;
forward gait remains ~2 Hz per foot and lateral tracking is weak. The cadence
objective is unmet. See `docs/go2_conga.md` and the experiment's analysis folder.
No additional training or advancement is authorized.
An authorized offline alternating-contact sequence audit is complete. Correct
synthetic alternating steps earn 35/35 bonuses; repeated diagonals, tested 2 Hz
patterns and chatter earn none. All six recorded policy evaluations earn zero
out of 1,120 eligible transitions each, including relaxed clearance thresholds.
This is a useful discriminator but a sparse learning signal; outright replacement
of dense stepping feedback is not recommended from these scores. No training
reward was changed. See `scripts/score_alternation.py`,
`logs/conga_alternation_scoring/` and `docs/go2_conga.md`.
The user's simpler last-opposite-liftoff gate has also been rescored offline.
It preserves paired-clearance feedback on 2.08% of latest-rollout timesteps,
versus zero strict sequence bonuses, while removing repeated-diagonal marching
credit. Some 2 Hz alternating credit remains; phase matching is still needed.
See `scripts/score_liftoff_history.py` and
`logs/conga_liftoff_history_scoring/scores.json`. No runtime reward or training
changes were made for either offline audit.
The user subsequently authorized a history-gated fine-tune: 200 additional PPO
iterations, 1,024 environments, 24-step rollouts from early-shaping model_999.pt.
`Unitree-Go2-Conga-History` keeps final early weights/commands fixed and changes
only paired eligibility to the audited last-opposite-liftoff gate. Seventeen
tests and a two-iteration resume smoke passed. Run supervision is via
`scripts/run_early_experiment.py --history-finetune`, with gates at absolute
1050/1100/1200 and automatic VRAM wait/resume. State and results are under
`logs/conga_history_1024_experiment/`. Do not extend the 200 additional iterations,
interfere with separate training, or advance to bounce/kick automatically.
The history-gated fine-tune has completed its 200 additional iterations without
VRAM recovery. Controlled evaluation gates had zero falls, but stationary
stepping still repeats FL/RR at 2 Hz while FR/RL stay planted. The paired reward
is verified zero for that behavior, so the gate works but has not induced the
desired alternation. Tracking improved modestly, slip rose about 18%, and half
the target touchdowns are still missed. No additional training is authorized;
see the completed experiment status and `docs/go2_conga.md` for results.
The user retained a 0.20 s swing and requested offline scoring of explicit
0.30 s pause rewards. `scripts/score_pause_rewards.py` tested a 3x3 coefficient
grid. An extra support bonus also rewards standing/repeated-pair behavior and
can cancel the intended fast-step penalty. Pause penalty -.25 without another
support bonus reduces fast forward stepping credit while preserving the
synthetic reference, but does not affect stationary repeated-diagonal behavior
that already respects the pause. This is an untrained candidate, not an installed
reward change. See `logs/conga_pause_scoring/scores.json` and `docs/go2_conga.md`.
The user-requested repeated-step penalty was scored offline with the pause cost.
An integrated .05 cost per repeated liftoff suppresses the observed repeated-pair
advantage without affecting the correct synthetic gait. However, staggered
takeoffs and one-sample chatter evade the detector while retaining phase credit.
The combined scheme is not yet considered training-ready. No runtime reward
was changed or training started. See `scripts/score_repeat_penalty.py` and
`logs/conga_repeat_penalty_scoring/scores.json`; event costs are not raw
Isaac reward-manager coefficients (manager rates are multiplied by step_dt).
The user authorized closing tested loopholes and updating rewards, not training.
`Unitree-Go2-Conga-Pause` now keeps the History configuration's fixed settings
and adds normalized pause cost -.25 plus integrated repeated-departure cost
-.025 per foot (-.05 per diagonal). Every sampled foot departure is accounted
for; only a confirmed opposite-pair swing clears its repetition history (both
feet >=5 mm clear for two samples with other-pair support). The 40 ms simultaneous
takeoff restriction no longer lets staggered repetitions avoid costs. Pair
reward +.10 uses confirmed history; phase remains +.25, no extra support bonus.
Eighteen checks and runtime-core adversarial scoring passed. Contact noise can
incur costs; standing and generalization remain learning risks. This separate
task is untrained, preserves 0.20 s swing / 0.30 s pause and the final dance
specification. See `docs/go2_conga.md` and `logs/conga_hardened_pause_scoring/`.
The user subsequently authorized training the Pause task with 4,096 environments
and delegated duration selection. A bounded 1,000 additional iterations (24-step
rollouts, 98.304M samples) is now supervised from history model_1199.pt, with gates
after 50/100/200/500/1000 new iterations. Launch mode is `--pause-finetune` in
`scripts/run_early_experiment.py`; status/results live in
`logs/conga_pause_4096_experiment/`. Reward weights stay fixed, no other training
is interrupted, and automatic VRAM wait/resume remains enabled. Evaluation adds
individual-foot/diagonal airborne and four-foot pause interval statistics.
Check live status before claiming completion; do not advance to bounce/kick.
The 4,096-environment Pause experiment completed all 1,000 additional iterations
(model_2199.pt). All five controlled evaluation gates had zero falls. Tracking
and slip improved substantially, but forward feet remain ~2 Hz, support pauses
only grew .08→.12 s, and stationary FL/RR repetition persists. Reward audit
confirms -.10/s repetition cost is active in the stationary case; target timing
is still unmet (576/1152 missing touchdowns). No further training has started.
See `docs/go2_conga.md` and `logs/conga_pause_4096_experiment/analysis/`.
The user requested scoring/testing a march-first candidate, not a training run.
`Unitree-Go2-Conga-March` is now isolated with all velocity commands zero and
timing active from the start: phase +1, individual clearance +.25, paired
completion +.25, missed-swing cost -.5, pause cost -.25 and repeated departure
-.025 per foot. Stock standing/air-time terms remain off; physical penalties and
observation/action layout are preserved. The paired term provides dense credit
without requiring previous alternation; repetition costs remain active. Synthetic
scores favor correct reference (1.5) over standing (-.2), repeated pair (.55),
and tested fast alternation (best .173), but partial-credit loopholes and learning
feasibility remain unproven. Nineteen tests and two bounded 550-step x16
simulation/export probes passed; neither performed PPO learning. See
`logs/conga_march_scoring/`, `logs/conga_march_smoke/`,
`logs/conga_march_existing_policy_probe/` and `docs/go2_conga.md`.
No fresh march-first training or automatic progression has been authorized.
The user subsequently approved the fresh march-first test. A 4,096-environment,
1,000-iteration run (24-step rollouts, seed 42) is now supervised with gates at
50/100/200/500/1000. It starts from random policy/optimizer weights, keeps all
commands zero and march rewards fixed, and does not advance automatically.
`MarchRunner` resumes without applying the earlier walking curriculum; 20 checks
passed before launch. VRAM wait/resume and stability review guards remain active.
Use `scripts/run_early_experiment.py --march-first`; inspect
`logs/conga_march_4096_experiment/status.json` before claiming completion.
The march-first run completed all 1,000 iterations. Final controlled evaluation
had zero falls, but alternating marching remains unmet: the first case repeats
FL/RR only ~.22 Hz with FR/RL planted. A requested zero-command video of the final
model_999.pt is in `logs/conga_march_video_final/videos/go2-step-0.mp4`.
No additional training or progression has started.
The user-requested starting-diagonal/reward audit found that initial segment
episode ages are randomized, but every normal reset returns the clock to zero
and schedules FL/RR first. Contact reward components are symmetric under diagonal
swap plus .5 s phase shift (recorded-case discrepancy <3e-14). An evaluation-only
opposite-start probe did not switch the learned policy to FR/RL; both probes had
zero falls. Balanced-cycle candidate scoring increases the correct-vs-repeated
gap ~39%, but gives zero bonus on all recorded final cases. Reset balancing is
recommended as a separately controlled fresh-learning test; neither it nor the
candidate reward was installed or trained. See `docs/go2_conga.md` and
`logs/conga_march_phase_audit/`.

The user authorized fresh balanced-start march training with unchanged march
rewards: `Unitree-Go2-Conga-March-Balanced`, 4,096 environments, 1,000 iterations,
24-step rollouts, evaluations at 50/100/200/500/1000. Half the environments start
with each diagonal and each environment toggles its starting diagonal on every
reset; episode ages, physics, commands, reward weights and final conga remain
unchanged. No balanced-cycle reward is installed. Twenty-one adapter tests and
a 550-step, 16-environment zero-action simulation/reset/PPO/export smoke passed
(zero falls; not evidence of learned gait). The supervisor was launched with
`--march-balanced`; consult `logs/conga_march_balanced_4096_experiment/status.json`
for live state. Existing VRAM recovery and deterioration gates apply. Do not
extend the budget or advance stages automatically. See `docs/go2_conga.md`.

## Current authorized duration-walking experiment (supersedes proposals above)

The user approved fixed air/stance penalties -.05/-.05 and fresh training.
`Unitree-Go2-Conga-Duration` uses stock basic tracking/physical/posture rewards,
disables stock air-time bonus/variance, and adds bounded duration costs for
.20 s swing / .80 s stance, with tolerance bands [.16,.24]/[.70,.90] s. Commands
retain stock gradual expansion, capped at .5 m/s XY norm. Neutral permits standing;
new costs are off when XY<=.05 m/s and |yaw|<=.05 rad/s. Completed costs are held
between events and ongoing overruns penalize indefinitely planted feet under
active commands. Initial partial durations are excluded. Phase observations
restart on activation; no phase reward or deployment transition controller exists.
Fresh budget: 4096 environments, 1000 iterations, 24 steps/rollout; evaluations
50/100/200/500/1000. Supervisor mode `--duration-walking`; inspect
`logs/conga_duration_4096_experiment/status.json` for actual progress. Twenty-two
adapter checks, exact-core synthetic/saved-rollout scoring, fresh/resume PPO and
bounded evaluation/export smoke passed. Do not infer learned gait from these.
Preserve VRAM recovery, don't interfere with other runs or extend automatically.
Duration distributions and walking matter; legacy phase misses are not acceptance
criteria for this task. Implementation/scoring details are in `docs/go2_conga.md`.

The duration experiment has completed 1000 iterations. Final controlled evaluation
had zero falls and 32 timeouts; neutral stands, forward .4 command tracks .375 m/s,
backward -.4 tracks -.348, lateral +/-.25 tracks +.196/-.191, yaw +/-1 tracks
+.977/-.986. Forward gait is diagonal but still 2 Hz per foot: median swing
[.12,.12,.16,.18] s and stance [.38,.38,.34,.32] s in FL/FR/RL/RR order, with
.08 s four-foot pauses. The .20/.80 duration goal is unmet. Last-50 training
mean episode duration was 19.936 s. Earlier evaluation-gate resumes inadvertently
reset command ranges due to saving initial config instead of live term config;
this was fixed around iteration 600 and subsequent persistence verified. See
range_repair.json and docs for this comparison limitation. No further training,
reward changes or stage advancement are authorized. Final artifacts are under
logs/conga_duration_4096_experiment/; checkpoint path is in status.json.

## Latest authorized experiment: stronger duration penalties

The user requested another run with harsher duration penalties. The separate
`Unitree-Go2-Conga-Duration-Strong` changes only air/stance weights from -.05 each
to -.20 each; bands, normalization, physical rewards, neutral standing and command
cap are unchanged. Fresh 4096-env, 1000-iteration, 24-step PPO budget; gates at
50/100/200/500/1000, no automatic extension. Use supervisor `--duration-strong`;
status is `logs/conga_duration_strong_4096_experiment/status.json`. Corrected live
command-range checkpoint persistence is used from the start. Separate checkpoint
phase-mode metadata prevents accidental gentle/strong training resume. Twenty-four
adapter/configuration checks and two-iteration PPO smoke passed. Counterfactual
scoring is in logs/conga_duration_strong_scoring; this does not establish learning.
VRAM recovery and deterioration checks remain active. Consult live status before
claiming completion. Earlier duration results remain the gentle-weight baseline;
its pre-600 command-range resume defect limits a strict controlled comparison.

The stronger-duration run is now complete (1000 iterations, 4096 environments).
Final evaluation had zero falls/32 timeouts, neutral standing and useful walking:
forward .4 tracks .380 m/s. However forward cadence remains about 2 Hz per foot,
median swing [.14,.12,.16,.16] s, stance [.36,.38,.34,.32] s and four-foot pause
.08 s. Increasing both weights fourfold did not achieve the slower gait. The
corrected command-range persistence worked throughout; no recovery or corrective
pause was needed. Baseline comparison still has its historical resume confound.
Do not extend or advance automatically. Results/checkpoint are recorded in
`docs/go2_conga.md` and `logs/conga_duration_strong_4096_experiment/status.json`.

An authorized offline velocity-averaging audit is complete; no reward changes or
training followed. Existing strong-policy forward XY reward improves only
1.49352 -> 1.49607/s with a 2 s average. Synthetic equal-mean speed fluctuations
receive substantially more credit, supporting the proposed mechanism but not
establishing the cause of learned 2 Hz steps. Instantaneous reward is insensitive
to frequency at a fixed speed distribution. Pure averaging can fully reward
large cancelling forward/backward excursions. An 80/20 averaged/instantaneous
blend was scored as a candidate only. See scripts/score_velocity_averaging.py,
logs/conga_velocity_average_scoring/ and docs/go2_conga.md. Command transitions,
neutral response, frame handling and reward history remain design decisions;
no implementation or new training is authorized by this scoring request.

## Latest authorized run: averaged XY tracking

User approved the 80% two-second average /20% instantaneous XY tracking blend,
then working on a kick with timing independent of gait speed. Separate task
`Unitree-Go2-Conga-Duration-Average` retains Strong timing costs -.20/-.20 and
all other settings. XY total weight1.5; yaw remains instantaneous .75. Rolling
body-frame vector error is averaged before the exponential, with actual-count
warmup; command changes clear reward history, neutral uses instantaneous tracking,
and selective resets clear affected histories. Phase clock does not restart on
active command updates. Observation/action layout is unchanged; full reward
history is not an actor input. No hardware transition compatibility is established.
26 adapter checks, fresh/resume two-iteration PPO and 220-step evaluation/export
smoke passed. Fresh 4096-env/1000-iteration/24-step run launched via
`--duration-average`; live state under `logs/conga_duration_average_4096_experiment/`.
Gates50/100/200/500/1000, correct curriculum persistence, memory recovery and
regression checks apply. No extra iterations or automatic advancement. User wants
kick work next; no kick training budget is agreed yet. Assess isolated support/
kick behavior without assuming that a slower learned gait is a prerequisite.

The averaged-velocity run completed all 1000 iterations. Final evaluation zero
falls/32 timeouts, neutral standing; forward .4 tracks .430 m/s, but cadence
remains ~2 Hz and four-foot pause shrank to .04 s (Strong baseline .08 s).
Lateral +/-.25 tracks only +.040/-.035 m/s, versus Strong +.202/-.201. This seed
regressed lateral walking and did not establish slower stepping; do not promote
the blend as an improvement. Results/checkpoint are in docs/go2_conga.md and
logs/conga_duration_average_4096_experiment/status.json. No extension started.
Next user interest is an isolated, simulation-time kick with three-foot support,
largely independent of gait speed. Existing full kick stage is not isolated;
prepare/review a separate kick experiment before agreeing its training budget.

The user-requested bounded isolated kick test is complete. Separate task
`Unitree-Go2-Conga-Kick-Test` uses explicit periodic FR/FL kick requests, .25 s,
.08 m outward/.05 m clearance, three other support feet, zero travel, no bounce
or walking swings. It is not neutral joystick deployment behavior. 28 adapter
checks, endpoint/speed-independence tests and 550-step x8 zero-action reset/PPO/
export smoke passed. A 2000-step x8 no-learning probe of Strong walking model999
had zero falls but failed the kick: FR stayed planted; FL lifted only .02488 m,
never extended outward, and three-foot support was only .25 of its kick samples.
FL timely touchdowns alone are not kick successes. Synthetic contact+kick score
advantage over standing is .10355/s before physical costs; candidate weights
need review for learning. No kick training was performed or budget agreed.
See docs/go2_conga.md, logs/conga_kick_isolation_scoring/scores.json and probe
artifacts. Keep final dance timings/geometry and previous task configs intact.

## Current authorization: small-kick training, review and continuation

User approved the split kick rewards and requested scoring,4096-env training,
then reward reevaluation and continuation. Separate task Kick-Intro fixes .02m
outward/.02m clearance/.25s duration, with no bounce/travel. Height+2,lateral+2,
airborne+.5,missing support-1; base height+.75,zero-velocity and physical costs
retained. Randomized first leg and per-reset0/2s offsets match observations.
Exact-core scoring prefers target over partial over standing; wrong supporting
foot lift reduces credit.30 adapter checks and transfer/PPO/reset/export smoke
passed. Fine-tune starts from Strong walking model999 for500 additional iterations,
gates1050/1100/1200/1500; review then up to500 more to2000 is authorized. Keep
intro amplitudes fixed unless explicitly reviewing/scoring a subsequent change.
Supervisor --kick-intro; state logs/conga_kick_intro_4096_experiment/status.json.
Continuation after review uses --kick-intro --kick-continue --resume-state.
Do not confuse old no-training-authorization snapshots with this latest request.
No unlimited extension or full-dance advancement. See docs/go2_conga.md for
scoring, progress qualification, and actual outcomes; inspect state before claims.

The first500 additional small-kick iterations completed. FL qualifies144/144 at
all gates; at+500 peaks .01954m outward/.02100m clearance. FR qualifies0/144:
peaks .01808m outward but .00136m clearance. All evaluations zero falls, full
three-foot support; outside requests all four contact. Data/first-block model1499
are recorded in docs. A first-gate evaluation harness environment failure was
fixed without replaying saved training; no VRAM issue.
Per user's review-and-continue authorization, Kick-Refine now changes only lift
bonus to+2 and multiplies by clamp(actual_clearance/.01,0,1). Other weights and
2cm amplitudes fixed. Exact-core scoring improves target-standing gap~37%, and
zero-clearance contact flicker earns no lift bonus.31 checks and transfer smoke
passed. Continuing500 additional iterations toward absolute2000 via
--kick-refine --kick-continue --resume-state, same experiment status directory.
Do not treat initial FL learning as full-amplitude/bilateral kick success.

The kick review and continuation are complete:500 introductory +500 refined
iterations,4096 environments,98.304M additional samples. Final model1999, absolute
2000 completed including the prior1000 walking iterations, is in status.json.
Final evaluation zero falls/32 timeouts. FL qualifies144/144 with1.97cm outward,
2.06cm clearance and all three supports; FR qualifies0/144, has no liftoff and
only1.02mm clearance despite1.88cm lateral extension. Both maintain support and
four-foot stance outside requests. The stronger height-aware lift bonus did not
solve FR lifting. This is a learned introductory FL kick, not bilateral/full-
amplitude kick success. No amplitude increase or extra training was started.
Final video:logs/conga_kick_intro_video_final/videos/go2-step-0.mp4; curves and
measurements in logs/conga_kick_intro_4096_experiment/analysis/ and eval_2000/.
Next review should address FR height/exploration, not assume more conditional
bonus weight or more iterations will solve it. Full details in docs/go2_conga.md.
