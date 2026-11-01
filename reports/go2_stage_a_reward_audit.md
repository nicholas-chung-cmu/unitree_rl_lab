# Stage A reward and exploration audit — 2026-10-29

No runtime rewards were changed and no training was started. Reproduce with
scripts/audit_hierarchical_stage_a.py in nick_sillywalk; detailed scores are in
logs/hierarchical_stage_a_reward_audit/audit.json.

## Findings

The final held-out evaluation kept selected feet in contact on all 25,865 STEP
samples. Selected sole clearance was median 1.13 mm, 95th percentile 1.57 mm,
maximum 1.67 mm. None of the 353 terminal trials registered a selected-foot lift.
This supports a planted-foot/body-shift diagnosis rather than a timing-only miss.

The reward offers easy partial credit. At the 4 cm height peak, a foot with
correct XY but zero clearance receives exp(-.04^2/.0025)=.527 of its placement
credit. During the requested swing, all-four-foot support still receives .5 of
the contact score, plus full support credit. Heading and body displacement can
improve without lifting. The integrated +2 success event requires both selected
feet to have lifted, landed, reached the endpoint and satisfied body/heading
criteria; it provides no intermediate unloading or lift credit.

There is also an explicit reference conflict: contact requests airborne selected
feet only for .15 < progress < .70. The sine-squared height reference is still
2.618 cm high at .70 and exceeds the executor's 8 mm lift threshold until .8524.
Thus a correct height-tracking trajectory loses contact reward during late swing.
Fix the target consistency before tuning strength. The .70 minimum success time
is an earliest completion gate, not by itself a touchdown deadline in Stage A.

Training exploration narrowed: mean action standard deviation decreased from
.998 to .295 (action units, before .25 joint-action scaling). Logged mean skill
successes peaked at .06955 but averaged only .000223 over the last 50 updates.
These logged metrics are episode aggregates, not held-out success estimates.
Training mean return nevertheless rose from -3.30 to +2.59. This is consistent
with converging to easier partial-credit behavior, but one 200-iteration seed
does not establish that longer training could never succeed.

Height tracking is not currently penalized by a fixed base-height term in Stage A;
bounce and indiscriminate vertical-velocity penalties are off. Orientation,
action-rate, joint acceleration/velocity/torque, slip and collision costs remain.
Their presence alone does not establish that they cause the failure. A raised
single-foot initializer also does not prove a two-foot diagonal swing is feasible:
the latter must balance dynamically on the other diagonal.

## Offline candidate comparison

A synthetic one-second .08 m STEP was scored at equal duration, with body/heading
tracking and support credit held equal. Contacts use a >2 mm sole-height proxy,
not simulated forces. Physical costs, initial stabilization, event bonuses and
actual termination lengths are excluded. These are incentive diagnostics, not
predicted PPO returns or dynamically feasible trajectories.

| Synthetic behavior | Existing dense terms | Add proposed lift term at +1 |
|---|---:|---:|
| Reference trajectory | 3.720 | 4.128 |
| Planted feet, body tracks | 2.808 | 2.808 |
| Only one selected foot follows | 3.264 | 3.468 |
| Twice target height | 3.421 | 3.421 |

The proposed term rewards reduction in selected-foot absolute height error
relative to a planted baseline, normalized by 4 cm and bounded [-1,1]. It is
linear below the target and gives no positive credit to planted feet; doubling
target height does not improve it. The provisional scoring uses the existing
swing window; align that window before any runtime implementation. Weight +.5
was also scored. Single-foot progress still earns partial credit intentionally;
monitor both feet and diagonals rather than claiming this removes all loopholes.

The current objective already ranks the ideal reference above standing. The
issue is plausibly reaching that behavior through exploration and intermediate
feedback, not a proof that standing globally maximizes the reward.

## Recommended sequence

1. Align contact targets with the existing height trajectory, retaining near-ground
   transition margins and touchdown timing. Add a reference/contact consistency
   test. Do not silently change the final musical touchdown schedule.
2. Evaluate that correction as one controlled factor. If lifting remains absent,
   test the explicit selected-foot height-progress term separately (+1 is a
   scored candidate, not an approved or validated weight).
3. Keep physical costs and PPO exploration settings fixed for the first comparison.
   Diagnose stochastic lift attempts before a separate exploration change.

Do not extend Stage A or advance to Stage B automatically. No live reward change
or new training was performed for this audit.
