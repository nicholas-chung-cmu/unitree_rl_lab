# Walking-only validation — 2026-10-27

**Result: do not introduce the kick.** The 200-iteration experiment completed,
but the deterministic policy learned to remain planted rather than walk, march,
or turn. Longer episodes and increased reward did not establish a usable gait.
I recommend addressing the reward balance, then doing another bounded
walking-only experiment before increasing the training budget or adding bounce
and kick. One short run does not prove that more training alone could never work.

## Configuration and run

- Task: `Unitree-Go2-Conga-Walk-Validation`, a separate temporary configuration.
- Environment: `nick_sillywalk`; shared Isaac Lab source and the original
  `isaaclab` environment were not modified during this experiment.
- PPO: 256 environments, 200 iterations (indices 0–199), 48 steps per rollout,
  seed 42; 2,457,600 transitions, approximately 5 minutes 29 seconds of training.
- Commands: vx ±0.4 m/s, vy ±0.25 m/s, yaw ±1 rad/s, with zero commands and
  pure turning explicitly sampled. Episode limit: 20 seconds.
- Fixed 120 BPM clock and continuous diagonal gait retained. Each foot has a
  1 Hz touchdown target; diagonal swing opportunities occur at 2 Hz.
- Kick disabled; kick/bounce scales zero. Kick, body-height and reference
  vertical-velocity rewards removed. Contact, cadence, clearance, marching,
  velocity tracking, torque, limits, action rate, orientation, slip and collision
  terms retain their original weights. Policy/critic/action sizes: 55/70/12.
- Final conga reference, motion specification, and existing curriculum configs
  were not changed. No reward tuning was performed during or after the run.

Run: `logs/rsl_rl/go2_conga_walk_validation/2026-10-27_17-22-24_walk_only_200/`.
Final checkpoint: `model_199.pt` (200 completed iterations).
Training console output: `/tmp/conga_walk_train_200.log`.

## Training curves

Values below are averages of the first and last 20 logged iterations. These
describe exploratory training, including action/observation noise. Early episode
statistics also include the runner's randomized initial episode lengths.

| Metric | First 20 | Last 20 |
|---|---:|---:|
| Mean episode return | -8.04 | 39.26 |
| Mean episode length | 5.70 s | 19.33 s |
| XY velocity error | 0.294 m/s | 0.208 m/s |
| Travel-only velocity error | 0.253 m/s | 0.243 m/s |
| Yaw velocity error | 0.768 rad/s | 0.383 rad/s |
| Contact-target mean absolute error | 0.422 | 0.329 |
| Actual touchdown distance to nearest target | 0.246 s | 0.251 s |
| Actual touchdowns per foot | 4.81 Hz | 1.79 Hz |
| Reference touchdowns per foot | 1.03 Hz | 1.00 Hz |
| Stance-foot slip | 0.298 m/s | 0.060 m/s |

Reward and survival improve strongly, then largely flatten. Timing error does
not improve. In the last 20 iterations, weighted contact reward is 0.679,
clearance 0.179, cadence **-0.026**, and marching only **0.003**. The mean logged
timeout termination fraction is 95.95%; bad orientation accounts for the other
4.05%. These are averages of logged termination fractions, not pooled counts.

![Training curves](../logs/conga_walk_analysis/training_curves.png)

## Deterministic checkpoint evaluation

Evaluated `model_199.pt`, with observation corruption disabled and deterministic
actor outputs, for 2,000 control steps × 8 fixed-command environments: 40 seconds
per case, two complete 20-second episodes each. No learning occurred. JIT output
matched the loaded actor, and ONNX serialization passed.

Steady-state statistics exclude the first 2 seconds of every episode. Touchdowns
are matched one-to-one per foot to the analytic schedule within ±0.08 seconds;
boundary windows and reset rows are excluded. Each case has 144 scheduled events
in its scored windows. This test uses the reduced training command range, not
the eventual 1.25 m/s target.

| Command (vx, vy, yaw) | Mean XY error (m/s) | Mean yaw error (rad/s) | Missed / scheduled touchdowns |
|---|---:|---:|---:|
| (0, 0, 0) | 0.022 | 0.031 | 144 / 144 |
| (0.4, 0, 0) | 0.400 | 0.051 | 144 / 144 |
| (-0.4, 0, 0) | 0.401 | 0.023 | 144 / 144 |
| (0, 0.25, 0) | 0.250 | 0.045 | 144 / 144 |
| (0, -0.25, 0) | 0.250 | 0.019 | 144 / 144 |
| (0, 0, 1) | 0.045 | 0.995 | 144 / 144 |
| (0, 0, -1) | 0.062 | 0.999 | 144 / 144 |
| (0.3, 0.15, 0.5) | 0.334 | 0.497 | 144 / 144 |

There were **zero falls across 16 complete episodes**, but **all 1,152 scheduled
touchdowns were missed**. There was one extra, off-phase rear-left touchdown in
the negative-yaw case. No events matched, so matched timing error is undefined,
not zero. The other seven cases had zero scored touchdown events; the one
rear-left event corresponds to 0.028 Hz for that foot, far below its 1 Hz target.

All four feet remained in contact for 99.72–100% of steady-state samples.
Mean translational velocity components were within about 0.003 m/s of zero in
every case; mean yaw was within 0.007 rad/s of zero. The body sat at about
0.282 m rather than the 0.32 m nominal height (height reward was disabled).
The higher touchdown rates in training therefore do not demonstrate a learned
march: they occur under exploration, while deterministic behavior is planted.

![Contact traces](../logs/conga_walk_analysis/contact_timing.png)

![Velocity tracking](../logs/conga_walk_analysis/velocity_tracking.png)

The velocity plots omit episode warmup intervals; connecting lines over those
gaps do not represent measurements. Solid lines are actual velocities and dotted
lines are commanded values. Contact plots show actual contacts against dotted
soft reference targets over seconds 2–6 of the first episode.

## Interpretation and next experiment

Reward balance is a plausible cause of this stationary solution, supported by
both the rollout and a numerical check of the existing reward equations:

- Keeping all feet planted earns about **0.70** mean dense contact reward over
  a complete walking reference cycle. Feet at 1 mm clearance also earn about
  **0.173** weighted clearance reward without executing a swing.
- Remaining stationary under a 0.4 m/s command still earns about **0.791** of
  the 1.5 maximum linear-velocity reward because its tracking kernel is broad.
- Perfect scheduled touchdown events contribute only about **0.02 reward per
  simulated second** at the current weight and 0.02 s control timestep. The
  reward manager multiplies the sparse event term by the timestep, whereas
  dense rewards accumulate every step. Missing events earn zero rather than
  incurring an explicit missed-event cost.

These calculations identify weak incentives; they are not a controlled proof
of which individual change will fix learning. Before adding kick, I recommend:

1. Make required swings and touchdowns materially rewarding: normalize sparse
   touchdown credit for the timestep, add an explicit missed-event/swing cost,
   and assess clearance during the central part of swing so planted feet cannot
   collect much of that reward. Retain soft boundaries and the 120 BPM clock.
2. Tighten or rebalance moving-command velocity tracking so remaining still
   cannot earn such a large fraction of the tracking reward. Preserve zero-
   command marching and pure-yaw cases and retain the physical penalties.
3. Repeat a bounded walking experiment and evaluate deterministic checkpoints
   during training. Consider a constant nominal-height posture reward separately
   from bounce; the current ablation also permits a lower stance.

Additional walking-only training and validation are required. Do not advance
based on episode length or total reward alone: require sustained commanded
translation/turning, genuine zero-command marching, per-foot cadence near 1 Hz,
and high scheduled-touchdown recall with few extras. The 0.08 s event tolerance
is an initial diagnostic criterion, not a validated physical acceptance limit.
Multiple training seeds, visual gait review, robustness, and final kick
feasibility remain untested. No further training or reward changes were run.

## Reproduction and artifacts

```bash
conda activate nick_sillywalk
python scripts/conga_check.py --headless \
  --task Unitree-Go2-Conga-Walk-Validation --num_envs 8 --steps 2000 \
  --walking-commands \
  --rsl-checkpoint logs/rsl_rl/go2_conga_walk_validation/2026-10-27_17-22-24_walk_only_200/model_199.pt \
  --output logs/conga_walk_evaluation
MPLCONFIGDIR=/tmp/conga_mpl python scripts/analyze_conga_walk.py \
  --run logs/rsl_rl/go2_conga_walk_validation/2026-10-27_17-22-24_walk_only_200 \
  --rollout logs/conga_walk_evaluation/rollout.csv \
  --output logs/conga_walk_analysis
```

`logs/conga_walk_evaluation/` contains 16,000 CSV rows, deployment parameters,
and explicitly trained JIT/ONNX exports. `logs/conga_walk_analysis/` contains
the figures, `training_scalars.csv`, and `summary.json`. Logs/artifacts are local
and Git-ignored. The evaluator's `SIMULATION PASS` means its runtime assertions
passed; it does **not** mean the policy passed walking acceptance.
