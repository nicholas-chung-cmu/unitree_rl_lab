# Goal-conditioned Go2 conga skills

Implemented on `feature/hierarchical-conga` for the October 29, 2026 request.
The branch was clean before implementation. Legacy conga configurations, robot
assets and checkpoints are preserved. This is a new policy family, not a reward
change to an existing velocity-tracking policy. No long training was launched.

## Architecture and interfaces

`skills/core.py` is a simulator-independent, batched PyTorch implementation:

- `SkillExecutor.begin(ids, kind, selected, goal, duration, state)` dispatches
  goals. `selected` is a four-foot Boolean mask: a diagonal for STEP, one front
  foot for KICK, or the support configuration for STABILIZE. `goal` is body-frame
  delta-x, delta-y and delta-yaw relative to dispatch heading. Kick geometry is
  configurable per request through `lateral_displacement` and `foot_height`
  keyword arguments, with defaults in `SkillConfig`. Stage C requests use
  .08 m/.05 m/.25 s.
- `update(state, dt)` evaluates measurements once per control step. Status is
  latched until another dispatch. `success_event` and `failure_event` are one-shot
  signals. Reported duration stops at completion. Contact hysteresis is 5 N on,
  2 N off; touchdown confirmation is 40 ms. Initial contact is not counted as
  a completed STEP: both selected feet must actually become airborne.
- `State` contains base pose, feet, forces, foot velocities, tilt, angular speed
  and flat local ground height. Feet are sole positions; the adapter subtracts
  the .022 m Go2 spherical foot radius from world z. Ordered feet are FL/FR/RL/RR.
- `DancePlanner.set_velocity(batch)` caps XY norm at .5 m/s and yaw at the
  configurable limit (.8 rad/s initially). Commands affect the next dispatched
  goal. The low-level policy never sees a continuous velocity-tracking command.
- `DancePlanner.advance(executor, state, dt)` combines completion feedback and
  absolute musical deadlines. Integer simulation ticks multiplied by dt in
  float64 avoid accumulating phase increments. Physical episode age is separate.

`skills/adapter.py` supplies a `CommandTerm`, reward/termination functions and a
small `ManagerBasedRLEnv` subclass. It follows the installed Isaac Lab ordering:
physics, termination checks, rewards, resets, command dispatch, observations.
The termination/reward adapter updates the core once using `common_step_counter`,
snapshots event feedback before auto-reset, and dispatches new goals only after
rewards were computed. Resetting one environment leaves all other skill clocks
intact. RSL-RL's randomized initial episode ages do not affect the musical clock.

The policy retains stock Go2 joint-position actions: 12 joints, default-pose
relative actions at .25 rad scale. All six stages share **100 actor inputs, 112
critic inputs and 12 outputs**. Inputs include standard proprioception (including
base linear velocity), previous applied actions, skill one-hot, selected feet,
goal and remaining goal, foothold/trajectory errors, measured contacts, entry and
current raised-foot masks, duration, remaining time, progress, body height/vertical-velocity targets and preparation flag.
Critic observations additionally include joint effort. Legacy policies have a
different observation schema and must not be loaded into these tasks.

Dense rewards cover foot trajectory/placement, displacement, heading, selected
contact configuration and supporting feet. Integrated success/failure/transition
rewards are divided by control dt before the Isaac reward manager applies dt.
Inherited torque, limits, action-rate, slip, joint dynamics, orientation and
collision costs remain. Physical falls have their own event cost. Continuous
velocity tracking, stock air-time/variance and near-zero posture rewards are off.
Weights are initial untrained choices, not evidence that the gait is learnable.

## Completion and handoffs

STEP requires both selected feet to have been clear of the ground, then confirmed
four-foot contact, foothold error below .045 m, body displacement error below
.06 m, heading error below .12 rad, tilt below .30 rad, and roll/pitch angular
speed below 1.5 rad/s. Tolerances are configurable. Completion is allowed from
70% of the requested duration; early completion becomes a support-preserving
STABILIZE. No zero linear-velocity requirement is imposed.

A post-kick STEP begins directly with the front foot airborne and the opposite
rear foot supporting. Its trajectory starts at the actual current raised-foot
pose; the rear foot lifts during the same skill. Both feet land as one diagonal
STEP. Wrong-pair or invalid entry support is rejected. A contact crossing of the
previously raised foot before 60% progress is recorded and penalized; transition
success credit requires zero such touchdowns.

KICK uses a quintic outward extension with no retracting segment. It constrains
body-relative transverse position and ground-relative height; fore-aft placement
is free. Nominal transverse coordinates are calibrated from the initially settled
four-foot stance. Success requires the airborne endpoint within .015 m, the other
three feet confirmed, kicking-foot speed below .25 m/s, and a stable body, from
90% of target duration. A successful kick hands its actual terminal state directly
to the same goal-conditioned policy under a STEP command. In isolated Stage C,
a short three-foot STABILIZE follows, then the episode ends.

Timeout grace is .4 s. Training tasks terminate unsuccessful trials. The dance
planner can stabilize a late/failed skill with valid four-foot support and
resynchronize at the next phrase. A valid raised-front configuration after a
late kick receives the appropriate landing STEP. Invalid support/falls terminate
simulation rather than dispatching conflicting actions. This is not a validated
hardware recovery controller.

## Touchdown-based musical timing

The user's clarification makes the beats **landing targets**, not skill-dispatch
times. The planner sends each STEP command ahead of its landing beat. This removes
the earlier apparent third-step time conflict; the third STEP has a full beat to
execute, followed by a half-beat support interval before the kick.

Tempo is 90 BPM: beat = 2/3 s, phrase = 8/3 s, full alternating sequence = 16/3 s.
Kick start remains exactly 5/3 s into each phrase and duration remains .25 s.
The clock is independent of travel commands; zero commands request marching.

| Action | Nominal dispatch (s) | Touchdown/endpoint target (s) |
|---|---:|---:|
| STEP FL/RR, startup pre-roll | -2/3 | 0, beat 1 |
| STEP FR/RL | 0 | 2/3, beat 2 |
| STEP FL/RR | 2/3 | 4/3, beat 3 |
| STABILIZE in four-foot support | after third landing | 5/3 |
| KICK FR | 5/3 | 23/12, airborne endpoint |
| Post-kick STEP FR/RL | 23/12 | 8/3, next phrase beat 1 |
| STEP FL/RR | 8/3 | 10/3, next phrase beat 2 |
| STEP FR/RL | 10/3 | 4, next phrase beat 3 |
| KICK FL after support gap | 13/3 | 55/12, airborne endpoint |

`landing_offsets=(0,1,2)` specifies touchdown beats within the phrase;
`step_lead_beats=(1,1,1)` specifies advance dispatch. Initial support acquisition
is outside the musical clock, followed by a one-beat pre-roll. The first STEP
lands at musical time zero. The post-kick STEP starts directly at the actual kick
endpoint and targets the next phrase's first beat, giving approximately .75 s
instead of forcing an intermediate landing or waiting until the beat to begin.

Scheduling uses continuous deadlines evaluated on the 20 ms control grid, not a
rounded integer number of ticks per beat. Contact confirmation may delay the next
safe dispatch by a few samples. Duration goals use the remaining time to the
fixed touchdown deadline, so those sample delays do not accumulate musical drift.
A delay over .10 s causes resynchronization instead of compressed catch-up steps.
A late kick with less than half a beat left receives a full-beat recovery STEP,
then resynchronizes. Those recovery landings need not be on a musical beat.

Stages E/F require selected-foot **contact onset** within .06 s of the requested
landing time, subsequently confirmed by 40 ms contact filtering. Landing much
earlier and merely standing through the beat cannot satisfy this condition.
A dense completed-touchdown timing cost supplements success gating. The earlier
skill stages retain relaxed timing. Diagnostics distinguish touchdown timing,
completion/confirmation time and command-dispatch error.

Displacements equal capped velocity times the allocated STEP duration, with yaw
handled similarly. Kicks and holds contribute no requested distance. There is no
catch-up multiplier. Thus .5 m/s is the traveling goal cap, not a guaranteed
instantaneous physical-speed bound or phrase-average speed. Ordinary full-beat
steps and the .75 s post-kick landing interval still require dynamics and learned
policy validation. The idealized oracle proves sequencing only.

## Curriculum support and current status

| Stage/task suffix | Implemented support | Qualification still required |
|---|---|---|
| A `Step` | Isolated four-foot-entry STEP, random diagonals, conservative signed XY/yaw goals, .8–1.2 s durations; optional consecutive steps | Learn both diagonals, displacement/heading tracking and reliable touchdown |
| B `Raised-Step` | Mixed normal/procedurally raised entries; physics-only preparation and measured admission; same STEP observation/action interface | Preparation reliability and learned raised-entry landing |
| C `Kick` | Both front legs, full airborne endpoint, .25 s duration, three-foot terminal hold | Bilateral learned extension, endpoint speed and stability |
| D `Transition` | Same live policy executes STEP→KICK→STEP without resets; `steps_before_kick=3` expands the chain; explicit transition credit | Actual learned-kick terminal-state distribution and robust handoffs |
| E `Timing` | Clock/event planner, early stabilization, late recovery/resynchronization | Touchdown timing feasibility at 90 BPM and continuous handoffs |
| F `Dance` | Full signed travel/turning/marching command range; configurable synchronized height/vertical-velocity bounce cost | Repeated alternating sequences, robustness, gradual bounce introduction |

Stage A defaults to isolated successful steps. Set
`env.commands.skill.isolated_steps=False` for consecutive steps once isolated
performance qualifies. Stage D/E/F never reset between successful skills. Stage C
is an isolated task; resuming it alone can forget stepping. Checkpoint transfer
and subsequent joint training in D require retention evaluation on earlier tasks.
No automatic stage advancement or curriculum promotion is installed.

Stage B now prepares both front legs with a support-force controller through the
existing position actuators. Over two seconds it shifts the whole-body center of
mass toward the other three feet, lowers the base target to .28 m, and permits a
small mirrored lean (roll +/- .06 rad, pitch -.04 rad). After measured unloading
(<8 N for .10 s with the other three contacts confirmed), a 1.5 s smooth lift
reaches the randomized endpoint. Joint gravity compensation and Jacobian-transpose
foot forces are converted into stock PD position targets. There are no raised-state
root/joint writes or reward changes.

Height remains randomized .04–.06 m and outward extension .06–.09 m. Admission
requires the completed lift followed by a continuous 1 s stationary hold:
three confirmed support contacts, airborne selected front foot, height error
<.015 m, extension error <.025 m, tilt <.20 rad, full angular speed <.25 rad/s,
body speed <.06 m/s and every foot speed <.06 m/s. Any invalid sample resets the
hold timer. Preparation times out at 10 s. Policy actions remain ignored during
this observed preparation substate; initialization is not STEP success.

The October 29 repair passed two held-out seeds (9030/9031), each with 64 first
attempts balanced FL/FR and an extended three-second hold: 128/128 admissions,
64 on each side, no falls or resets. All 19,328 recorded post-lift hold samples
passed the physical criteria. Separate bilateral video probes passed at seed
9032. A 550-step, 16-environment mixed-entry reset/export smoke had 16 prepared
admissions and zero preparation failures/falls. All 31 hierarchical component
tests passed. No PPO learning was performed. See
[initializer repair report](../reports/go2_initializer_repair.md) for metrics/artifacts.

The earlier 420-step x8 probe (3 entries, 4 preparation timeouts, 1 fall) and
seed-9029 overnight probe (9/16 admissions) remain historical failed tests of the
previous initializer. The new results qualify scripted preparation within the
current simulation distribution; learned landing, actual policy KICK-to-STEP
transitions, wider dynamics randomization and hardware remain unverified.

Bounce is disabled by default (`bounce_amplitude=0`). Stage F can be manually
ramped toward .0075 m amplitude, yielding .015 m peak-to-peak at one oscillation
per beat. The policy observes explicit body-height and vertical-velocity goals derived
from the musical clock in every stage. Bounce is not ready for training before transition qualification.

## Verification and commands

Use only `/home/rml2/anaconda3/envs/nick_sillywalk/bin/python`. No packages or
shared Isaac Lab files were changed. The installed Go2 actuators, Isaac Lab
manager APIs, RSL-RL runner and exporters were inspected and used directly.

Verified evidence (October 29):

| Check | Result / artifact |
|---|---|
| New core and Isaac adapter contracts | 27 passed, including touchdown-on-beat scheduling, early-landing rejection, direct raised-leg handoff and one-shot event rewards; `logs/hierarchical_final_checks/` |
| Legacy numerical references | 10 passed unchanged; 31 tests when combined with the 21 new CPU core tests |
| Configurations | All six new task configurations validate |
| Step/PPO/export smoke | 16 environments, two optimizer updates with save/load between them, JIT parity and ONNX export, then 300 zero-action steps; zero falls and 48 correctly failed STEP trials; `logs/hierarchical_ppo_final_smoke/` |
| Normal training entry point | One PPO iteration, 16 environments, 384 samples; checkpoint and configuration export created in `logs/rsl_rl/go2_hierarchical/2026-10-29_01-28-13_architecture_smoke/` |
| Touchdown-based dance smoke | 550 × 8 zero-action steps, zero falls, 32 STEP timeouts recovered/resynchronized, zero STEP successes; `logs/hierarchical_dance_smoke/` |
| Isolated kick smoke | Final 120 × 8 zero-action steps, zero falls, 16 KICK timeouts, zero false successes; `logs/hierarchical_final_checks/` |
| Procedural raised-entry probe | 420 × 8 steps, 3 valid entries, 4 preparation timeouts, 1 fall, no successful landing policy; `logs/hierarchical_raised_final_smoke/` |

The preparation probe preceded adding two body-bounce goal observations; its
controller/admission procedure is unchanged. Early diagnostic folders are retained
as failed-prototype evidence. These results validate software paths and expose
preparation limits; none establishes a learned STEP, KICK or dance.

The regular training command was verified only with `--num_envs 16
--max_iterations 1 --run_name architecture_smoke`. No 200-iteration or larger
hierarchical experiment has been started. Existing URDF head-visual reference
warnings occur during simulator loading; bounded physics/export checks completed.

CPU contract tests (no simulator):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/rml2/anaconda3/envs/nick_sillywalk/bin/python -m pytest tests/test_hierarchical_skills.py -q -p no:cacheprovider
```

Simulator component/config checks and bounded PPO smoke:

```bash
/home/rml2/anaconda3/envs/nick_sillywalk/bin/python scripts/hierarchical_check.py --headless --num_envs 16 --steps 300 --test-components --ppo-iterations 2 --output logs/hierarchical_ppo_final_smoke
```

This requests exactly two small PPO iterations with a save/load between them,
then JIT parity and ONNX export, then a zero-action rollout. It is an integration
test, not an experiment intended to learn a skill. Observe `status.json` and
`summary.json`; simulator teardown can hide a Python error in the process code.

The evaluation script has nine separate scenarios: `four-foot`, `raised`, `kick`,
`transition`, `consecutive`, `march`, `moving-turn`, `in-place`, and `dance`.
For example, after obtaining a compatible trained hierarchical checkpoint:

```bash
/home/rml2/anaconda3/envs/nick_sillywalk/bin/python scripts/hierarchical_check.py --headless --scenario four-foot --num_envs 16 --steps 2000 --checkpoint /absolute/path/to/model.pt --output logs/hierarchical_eval_four_foot
```

Change the scenario and output directory to evaluate each case independently.
`--checkpoint` accepts an RSL-RL checkpoint, not an exported TorchScript file.
Without it, the script applies zero policy actions. The `raised` case forces
procedural preparation in all environments; regular Stage B training mixes 50%.

`trace.jsonl` captures pre-reset outcomes, contacts and confirmation ages,
touchdown counts/times, selected feet, raised state, support departures, pose,
placement/displacement/heading errors, slip and musical dispatch error.
`summary.json` groups STEP/KICK trials by normal/raised entry, reports success
rates and duration/errors, and separately reports actual-kick transition success.
Empty denominators are null, not successful zero-error trials. Falls and episode
ends are separate counts. The trace includes actual robot joint/action samples
for preparation diagnosis; episode-end joint/action samples can be post-reset,
whereas skill outcome measurements are always taken before reset.

## Recommended first experiment (not launched)

Start a fresh Stage A pilot: **1,024 environments, 200 PPO iterations, 24-step
rollouts**, conservative .08 m/s displacement sampling scale and .8–1.2 s duration,
with isolated steps. The budget is 4,915,200 samples. Review checkpoint evaluations
around 50/100/200 for both diagonals, actual swing/landing, falls, pose, slip and
completion rates. Do not interpret standing, low falls or initialized environments
as STEP success, and do not advance automatically. No claim is made that 200
iterations is enough to learn this skill.

Proposed launch command, requiring explicit approval for the experiment:

```bash
/home/rml2/anaconda3/envs/nick_sillywalk/bin/python scripts/rsl_rl/train.py --task Unitree-Go2-Hierarchical-Step --num_envs 1024 --max_iterations 200 --headless --run_name stage_a_pilot
```

This uses `logs/rsl_rl/go2_hierarchical/` with a separate timestamped run directory.
Resume only compatible hierarchical checkpoints using the existing training
script's `--resume --checkpoint_path /absolute/path/to/model.pt`. Stage-specific
run names or `agent.experiment_name` overrides keep future experiments separate.
There is no new long-run supervisor or automatic VRAM recovery in this architecture;
the historical experiment supervisors do not automatically apply to these tasks.

## Deployment limitation

The existing training wrapper and policy weight export work with the new schema.
Exported metadata explicitly records the skill contract, feature ordering and
`hardware_compatible: false`. A physical controller still needs state/contact
estimation, the planner, goal feature construction, calibrated foot geometry and
validated settling/recovery logic. Exporting a neural network does not supply
those pieces. No hardware test or full learned dance has been performed.

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
