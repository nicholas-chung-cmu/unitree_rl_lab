# Unitree RL Lab project instructions

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
