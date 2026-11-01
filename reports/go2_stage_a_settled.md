# Stage A with settled reset entry — 2026-10-29

The user identified that the robot began stepping while recovering from its reset
drop. The old same-seed preflight dispatched at .32 s with mean body speed
.274 m/s and angular speed .592 rad/s. The corrected preflight dispatched at
2.74 s with mean body speed .0265 m/s and angular speed .0262 rad/s.

The initializer now applies stock default-pose actions until it has measured
continuous four-foot support and low motion for .5 s, with a minimum reset age
of 1 s. Limits are .05 m/s body and per-foot speed, .2 rad/s angular speed and
.15 rad tilt. Motion/contact loss resets the hold timer. Settling times out at
5 s; physical falls remain terminal. The robot can still spawn above the ground,
but the skill does not start during the drop. No state teleport or physical
age reset is introduced. The existing preparation observation also marks this
settling state. Task rewards are suppressed during settling; inherited physical
costs remain. Active STEP references/rewards and completion criteria are unchanged.

All 33 component tests passed, including motion rejection, hold restart and
recovery from the old short STABILIZE timeout. A 300-step x16 reset/export smoke
passed. A separate seed-9042 preflight admitted 64/64 balanced cases at 2.74 s.
The earlier progress message described the minimum gate as one second; the
observed preflight admissions were all 2.74 seconds.

Fresh training budget: 4,096 environments, 200 PPO iterations, 24-step rollouts,
seed 42, 19,660,800 samples. This matches the preceding aligned-contact run's
budget, but more samples are spent settling and fewer are active STEP attempts.
No automatic Stage B advancement or extra training is authorized.

Run: logs/hierarchical_stage_a_20261029_160817/. Live status is authoritative.
Training, all three checkpoint evaluations and both videos are complete.

## Results

| Policy / entry | Executor successes | Falls | Clean one-swing/contact trials |
|---|---:|---:|---:|
| Prior policy, old dropping entry | 607/607 | 0 | 0/607 |
| Prior policy, corrected settled entry | 99/155 (63.9%) | 0 | 0/155 |
| Fresh retrain, corrected settled entry | 0/128 | 0 | 0/128 |

The fresh retrain completed 200 iterations. Checkpoints50/100/199 each had0/128
successes and zero falls under the same held-out seed9041 evaluation. Final
FL/RR was0/63, FR/RL0/65. No selected-foot lift or touchdown was registered in
any completed trial; all timed out. Final mean slip was .00644 m/s, foot error
about .0792 m and body error about .0351 m. Reduced slip/repetition is not a
successful gait: the new policy stays planted and adjusts its body.

The old policy was separately evaluated under the new initialization with no
learning. It retained49/77 FL/RR and50/78 FR/RL successes, but failed the stricter
single-swing criterion throughout. There were no support-contact departures.
Repeated selected-foot airborne excursions occurred in66/77 FL/RR and74/78 FR/RL
trials (140/155 overall); repeated contact onsets occurred in151/155 trials.
An airborne excursion here means >8 mm sole clearance and no contact for at
least two sampled control frames. Thus these repetitions cannot all be dismissed
as one-frame contact noise. In the old entry evaluation, FR/RL support-contact
losses did not include a confirmed >8 mm airborne excursion; describe these as
contact losses, not proven large support-leg lifts.

The correction makes initialization reliable, and the comparison indicates that
entry state mattered, but reset impact was not the sole cause of repeated steps.
The old policy can still lift from rest, while the fresh short-budget run settles
into planted behavior. The extra settling time also reduces active training
attempts within the fixed total sample budget. This does not prove the fresh
policy could never learn with a different curriculum or longer budget.

## Two videos

Exactly two 10-second, 960x720, 50 fps clips were recorded from the fresh final
checkpoint. Each requests its diagonal at every isolated trial; reset/drop and
the measured waiting period remain visible. Both contain two completed STEP
attempts, with zero selected lifts/successes. Frames from settling and the skill
were inspected alongside contact traces. They show the corrected waiting period
and the new policy's planted behavior, not a successful clean step.

- [FL/RR diagonal](../logs/hierarchical_stage_a_20261029_160817/videos/FL/FL.mp4)
- [FR/RL diagonal](../logs/hierarchical_stage_a_20261029_160817/videos/FR/FR.mp4)

[Evaluation summary](../logs/hierarchical_stage_a_20261029_160817/evaluations.json)
contains all checkpoint results; eval_199/contact_analysis.json and
previous_policy_settled/contact_analysis.json contain the stricter contact audit.
The analysis is reproducible with scripts/analyze_hierarchical_steps.py.
Final checkpoint: logs/rsl_rl/go2_hierarchical/2026-10-29_16-08-31_stage_a_20261029_160817/model_199.pt.

Recommended next review: retain settled initialization; require exactly one
verified swing/landing per selected foot and support retention in the completion
contract. Consider the prior policy as a fine-tuning baseline, since it can lift
from rest, but do not accept its repeated swings as clean steps. A new reward or
training experiment is not installed or launched here. Stage B was not started.
