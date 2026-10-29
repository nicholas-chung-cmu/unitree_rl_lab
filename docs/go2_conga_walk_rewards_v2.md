# Walking reward revision v2 — 2026-10-27

**Result: rejected.** The requested reward changes were implemented and tested,
then trained from scratch for 200 PPO iterations with 256 environments. This
revision replaced the previous planted-foot solution with early orientation
failures. It is not a usable walking policy and must not advance to bounce/kick.
No additional training or post-run reward tuning was performed.

## Implemented changes

Changes are confined to the temporary `Unitree-Go2-Conga-Walk-Validation` task.
The final conga specification, reference generator, existing curriculum tasks,
120 BPM clock, command ranges, observation/action layouts, and physical
penalties were preserved. Bounce and kick remain disabled.

| Term | Baseline | Revision v2 |
|---|---|---|
| Contact | +1 × (1 − soft contact error) | −3 × soft contact error; removes the constant credit |
| Swing clearance | +0.75 × height kernel | +2 × central-swing-weighted quality; requires actual lift and no contact |
| Missed swing | None | −2 × central-swing deficit; planted feet count as missed swings |
| Touchdown | +2 × sparse signed timing score | +1 × score/control timestep; positive credit requires ≥0.08 s prior air time |
| Marching | +0.5 × drift-gated clearance | Same weight/drift gate, using the stricter swing quality |
| XY tracking kernel width | 0.5 m/s | 0.2 m/s |
| Yaw tracking kernel width | 0.5 rad/s | 0.3 rad/s |
| Nominal posture | No height reward | +0.5, constant 0.32 m height, 0.04 m kernel width; no bounce |

The missed-swing penalty is dense, not a stateful count of missing touchdown
events. Scheduled touchdown misses are still measured independently in
evaluation. Sparse event normalization cancels RewardManager's multiplication
by the 0.02 s control timestep. Short contact chatter cannot earn positive
touchdown credit, while off-phase contacts retain their cost.

Implementation: `mdp/conga_walk_rewards.py` and
`robots/go2/conga_walk_validation_env_cfg.py`. Ten pure-reference tests and ten
adapter/configuration tests passed, including grounded-foot rejection, correct
swing credit, timestep-invariant integrated event reward, chatter rejection,
reset masking, and isolation from the final conga configuration. An 8-environment
GPU/PPO/export preflight passed before training.

## Controlled repeat

Used `nick_sillywalk`, seed 42, fresh initialization, 256 environments,
48 steps per rollout, and exactly 200 iterations: **2,457,600 transitions**.
No PPO hyperparameters or physical penalties were changed. Training took about
6 minutes 7 seconds. The original `isaaclab` environment and shared Isaac Lab
source were not modified.

Run: `logs/rsl_rl/go2_conga_walk_validation/2026-10-27_17-44-59_walk_rewards_v2_200/`.
Final checkpoint: `model_199.pt`. Reward/reference source snapshots are saved
with the run's `params/`. Console log: `/tmp/conga_walk_v2_train.log`.

| Last-20-iteration training metric | Baseline | Revision v2 |
|---|---:|---:|
| Average episode duration | 19.33 s | **0.355 s** |
| Mean logged timeout fraction | 95.95% | 0.0069% |
| Mean logged bad-orientation fraction | 4.05% | **99.9931%** |
| XY error | 0.208 m/s | 0.209 m/s |
| Travel-only XY error | 0.243 m/s | 0.169 m/s |
| Yaw error | 0.383 rad/s | 0.310 rad/s |
| Actual touchdown distance to nearest target | 0.251 s | 0.158 s |
| Observed reference touchdown rate | 1.000 Hz/foot | **0.030 Hz/foot** |

The lower timing/velocity errors in v2 are **not gait improvements**: episodes
usually terminate before the first scheduled touchdown at 0.38 s. The clock
frequency itself has not changed. Termination fractions are means of logged
fractions, not pooled counts. Revision v2's mean return rose from −43.81 in the
first 20 iterations to −0.354 in the last 20 as episodes became shorter. Returns
across revisions are not directly comparable because their reward definitions
differ. Reward improvement within v2 also does not imply improved behavior.

![Survival comparison](../logs/conga_walk_v2_analysis/baseline_comparison.png)

![Revision v2 training curves](../logs/conga_walk_v2_analysis/training_curves.png)

## Same fixed-command evaluation

Evaluated the deterministic final checkpoint with observation corruption off,
using the same eight commands and 2,000 steps per environment as the baseline:
40 seconds per case, 320 environment-seconds total. Exported TorchScript matched
the loaded actor; ONNX serialization passed. No learning occurred.

| Command (vx, vy, yaw) | Orientation-limit terminations | Completed episode duration |
|---|---:|---:|
| (0, 0, 0) | 105 | 0.38 s |
| (0.4, 0, 0) | 111 | 0.36 s |
| (-0.4, 0, 0) | 105 | 0.38 s |
| (0, 0.25, 0) | 105 | 0.38 s |
| (0, -0.25, 0) | 111 | 0.36 s |
| (0, 0, 1) | 105 | 0.38 s |
| (0, 0, -1) | 105 | 0.38 s |
| (0.3, 0.15, 0.5) | 105 | 0.38 s |

**852 early terminations**, versus zero in the original deterministic evaluation.
No episode reached the two-second steady-state warmup threshold. Consequently,
steady-state velocity, cadence, and missing/extra touchdown metrics are
unavailable, not zero or passing. The analysis script now explicitly preserves
short-episode durations and marks such metrics unavailable.

## Interpretation and next change needed

The revision removed planted-foot credit, but the balance introduced a more
serious escape: terminating before accumulating penalties. A numerical check
over the walking reference cycle gives planted feet approximately −0.900 contact
reward/s and −0.749 missed-swing reward/s, with zero clearance credit. Remaining
stationary under a 0.4 m/s command now earns only 0.0275 linear-tracking reward/s.
Even at perfect nominal height with correct zero yaw, these terms sum to about
−0.372 reward/s before other physical costs. Off-phase event penalties also
became much stronger immediately, including during initial exploration.

This is strong evidence of a termination incentive, though a multi-change,
single-seed experiment cannot isolate each term's causal contribution. The
revision should not be trained longer unchanged or promoted into the final task.
The next bounded experiment should first make non-timeout termination costly
and ensure early upright behavior is preferable to terminating, then introduce
the swing/off-phase penalties progressively. Keep grounded-foot rejection and
the fixed clock, and evaluate survival and actual scheduled contacts together.
An explicit termination cost and a carefully chosen survival baseline should be
checked numerically before another run; neither was added after this experiment.

## Reproduction

```bash
conda activate nick_sillywalk
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Conga-Walk-Validation \
  --num_envs 256 --max_iterations 200 --seed 42 --run_name walk_rewards_v2_200
python scripts/conga_check.py --headless --task Unitree-Go2-Conga-Walk-Validation \
  --num_envs 8 --steps 2000 --walking-commands \
  --rsl-checkpoint logs/rsl_rl/go2_conga_walk_validation/2026-10-27_17-44-59_walk_rewards_v2_200/model_199.pt \
  --output logs/conga_walk_v2_evaluation
MPLCONFIGDIR=/tmp/conga_mpl python scripts/analyze_conga_walk.py \
  --run logs/rsl_rl/go2_conga_walk_validation/2026-10-27_17-44-59_walk_rewards_v2_200 \
  --baseline-run logs/rsl_rl/go2_conga_walk_validation/2026-10-27_17-22-24_walk_only_200 \
  --rollout logs/conga_walk_v2_evaluation/rollout.csv --output logs/conga_walk_v2_analysis
```

Evaluation artifacts include 16,000 CSV rows, deployment parameters, and trained
exports. `logs/conga_walk_v2_analysis/` contains figures, `training_scalars.csv`,
and `summary.json`. `SIMULATION PASS` in the evaluator log means its runtime
assertions passed, not that the policy passed gait acceptance. All run artifacts
are local and Git-ignored. Baseline artifacts were retained separately.
