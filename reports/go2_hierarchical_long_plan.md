# Calibrated long Stage A experiment — 2026-10-29

Authorized by the user after reviewing the failed short fine-tunes. Restart from
the original aligned-contact model_199.pt (200 completed updates), preserving
its learned lifting behavior. Budget: up to 1,000 additional PPO updates, 4,096
environments, 24-step rollouts, seed 42 (98,304,000 new samples if completed).
Keep settled entry, strict single-swing/support checks, clearance shaping,
clearance-gated placement and mild body-height control. No Stage B advancement.

## Sliding calibration

scripts/calibrate_hierarchical_slip.py scores equal-duration synthetic one-second
steps with 4/8/12 cm displacement, comparing reference, grounded sliding, body
shift with planted feet, and stationary behavior. The smallest tested sliding
coefficient that makes grounded sliding score no better than either planted
alternative while retaining the reference preference is **-4.0** (previous -2.0).
For the 8 cm case: reference 4.277, grounded slide 1.749, body shift 2.254,
standing 1.978. The former -.1 coefficient gave sliding 2.373 versus standing
1.978. Full results: logs/hierarchical_long_calibration/scores.json.

This is incentive ordering, not physics feasibility or a learnability guarantee.
The synthetic contact proxy uses 2 mm clearance and omits event rewards,
actuator costs and early terminations. The simulator baseline separately uses
actual reward-manager outputs and force-history-based sliding costs. Its mean
weighted rates over active STEP samples were: sliding -.666, displacement +.919,
placement +.406, clearance +.060, body-height -.033. Thus the sliding cost is
material relative to the progress terms. These rates are different from the
training logger's episode-normalized reward summaries.

Baseline artifacts: logs/hierarchical_long_calibration/baseline/. All 38 existing
component tests passed in that simulator run. Legacy task weights remain intact.

## Gated execution and learning rate

The supervisor is scripts/run_hierarchical_long.py. Train/evaluate sequentially
at exactly 50/100/200/500/1000 additional updates. Each segment resumes from a
private checkpoint copy with only its stored iteration advanced to the next
index. Original checkpoints are never overwritten. Optimizer/policy state is
retained; the saved optimizer LR initializes the adaptive scalar at each segment,
so evaluation boundaries do not reset it to .001. The original baseline saved LR
is .000260123. PPO retains its adaptive KL schedule and existing limits.

The full run's final checkpoint index is1199 if the budget completes, with prior
gates249/299/399/699. The checkpoint index is not the new-update budget.
Physical environments reset at segment boundaries; network/optimizer learning
continues. Fresh environment construction uses seed42 at each segment.

Deterministic held-out evaluation: seed9041,32 environments,1000 control steps.
Track both diagonals, verified lifts/landings, strict successes, falls, slip and
reward contributions. At200+ updates, low paired lift or zero completion triggers
an additional sampled-policy diagnostic using the checkpoint's learned Gaussian
noise. Stochastic results are exploration diagnostics, not acceptance successes.

Review stops:

- Deterministic fall count / completed STEP trials >10%.
- At100+ updates, success drops >20 percentage points after a gate with >=20%.
- At200+ updates, no verified lift at two deterministic gates and none in the
  current stochastic probe.
- At500+ updates, no completions in either probe and deterministic paired lifting
  below5% of completed trials.

Seven synthetic guard checks passed. A two-segment one-update smoke validates
exact checkpoint accounting, preserved optimizer/LR, and sampled evaluation.
Smoke completed successfully: logs/hierarchical_long_smoke_20261029_173528/. Stop on errors/deadline; only
the supervisor's own child is terminated. Four-hour overall limit. Two diagonal
videos are generated at the terminal checkpoint, including on a review stop.

Live run: logs/hierarchical_long_20261029_173703/status.json; tmux go2_stage_a_long. Launch is not completion.
