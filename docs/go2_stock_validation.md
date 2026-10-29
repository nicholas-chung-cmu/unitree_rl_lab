# Stock Go2 reproduction — 2026-10-27

The authorized 200-iteration stock run completed, but did not learn walking.
A compatible mature archived policy does walk in the same simulator. This is
not an equal-budget comparison: the mature run nominally used 4,000 times as
many training transitions.

## Training and compatibility

Used `nick_sillywalk`, `Unitree-Go2-Velocity`, seed 42, 256 environments,
200 iterations and the unchanged stock 24-step rollout: **1,228,800 transitions**.
No stock rewards, PPO hyperparameters, curriculum, randomization or terminations
were changed. The stock terrain configuration in this checkout generates flat
terrain only. Runtime was approximately 1 minute 32 seconds. The earlier conga
experiments used 48-step rollouts and therefore twice as many transitions.

Run: `logs/rsl_rl/unitree_go2_velocity/2026-10-27_18-53-18_stock_go2_200/`.
Final checkpoint: `model_199.pt`. Console log: `/tmp/stock_go2_train_200.log`.

The archived reference run is
`/home/rml2/Documents/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-08-26_13-31-03/`.
It trained with 4,096 environments on flat, rough, slope, box and stair terrain.
It is a local archive, not a verified official checkpoint release. Its saved
rewards, observations, actions, termination settings, commands, events,
curriculum, policy, PPO algorithm, seed, actuator settings and default pose
match the new run's corresponding settings. The URDFs are identical after
normalizing visual mesh directory names. Actor/critic/action sizes are 45/60/12.
The cached Isaac Lab Go2 policy has a different 48-input actor and was not used.

| Checkpoint | Iterations represented | Environments | Nominal transitions |
|---|---:|---:|---:|
| New `model_199.pt` | 200 | 256 | 1,228,800 |
| Archived `model_200.pt` | Approximately 201, using zero-based indexing | 4,096 | Approximately 19,759,104 |
| Archived `model_49999.pt` | 50,000 | 4,096 | 4,915,200,000 |

Counts are computed from the saved configuration and checkpoint indices, not
wall-clock time. No exactly equal-budget archived checkpoint was available.

## Results

The new run's mean episode duration fell from 2.41 s in the first 20 iterations
to **0.406 s** in the last 20. Its last-20 logged orientation termination fraction
was 100%; the XY command curriculum remained at ±0.1 m/s. Return increased from
−8.49 to −0.373 as episodes shortened, which is not walking progress.

Each policy was evaluated for 2,000 steps in 16 environments, 40 s per command:
eight commands from the prior walking comparison and eight initial-range
commands. Evaluation used identical flat ground, seed 42, no observation noise,
no material/mass randomization or pushes, and frozen commands/curriculum.
Normal reset events were retained. Post-reset rows were excluded and the first
2 s of each episode omitted from steady-state metrics. Exported JIT actions
matched the loaded actor for all three checkpoints.

| Policy | Early terminations, all 16 cases | Completed 20 s episodes | XY error, prior eight commands | Yaw error, prior eight commands |
|---|---:|---:|---:|---:|
| New stock run | **926** | 0 | Unavailable | Unavailable |
| Archived iteration 200 | 0 | 32 | 0.231 m/s | 0.217 rad/s |
| Mature archived policy | 0 | 32 | **0.049 m/s** | **0.036 rad/s** |

The new policy's completed episodes lasted 0.46–0.86 s, with no samples surviving
the warmup. The archived iteration-200 policy mostly stayed planted for linear
commands, although it responded to some yaw commands. The mature policy tracked
0.4 m/s forward with 0.029 m/s mean XY error and showed alternating FL/RR and
FR/RL contact patterns. Exactly two feet supported it in 55% of steady-state
forward samples; all of those two-foot patterns were diagonal.

Force-contact crossings give approximately **2.50 touchdowns/s per foot** in
that forward case, not the eventual conga target of 1 Hz per foot. At 0.1 m/s
the mature policy stayed planted in this evaluation. Thus ordinary stock gait
and low-speed behavior still need adaptation for the conga schedule and marching.

There is a diagnostic discrepancy: `compute_first_contact()` event counts do
not agree with crossings of the sampled force-contact signal for the mature
policy. Both are preserved in the analysis output. The gait frequency above
uses force crossings; sensor timing/tolerances need investigation before using
the sparse event counts to tune conga timing rewards. This observation does not
establish the cause of the earlier training failures.

![Training curves](../logs/stock_go2_analysis/training_curves.png)

![Contact comparison](../logs/stock_go2_analysis/contacts_comparison.png)

These results demonstrate that the compatible mature policy operates in this
simulator, but do not establish why the small-budget run failed. Environment
count, total samples, terrain diversity and training duration are confounded.
The appropriate next control is a longer stock experiment with explicit sample
budget and intermediate deterministic evaluations, rather than another reward
rewrite based only on this short run. No further training was started.

## Evidence for a longer test budget

The archived 4,096-environment run's TensorBoard log shows a command limit of
±0.2 m/s by iteration 125, ±0.3 by 209, ±0.5 by 334, and ±0.8 by 667; it was
at ±1.0 by iteration 1,000. These curriculum values alone do not prove walking.

A separate 500-step (10-second-per-command) spot-check of its `model_500.pt`
confirmed partial locomotion: mean forward velocity 0.183 m/s for a 0.4 m/s
command, backward velocity −0.318 m/s for −0.4 m/s, and yaw approximately
+1.094/−0.950 rad/s for ±1 rad/s commands. The moving-turn case tracked with
0.076 m/s XY error. Pure lateral commands still produced standing. None of the
16 cases terminated during the 10-second check; this is shorter than the main
evaluation and does not prove sustained stability. Artifacts are in
`logs/stock_go2_eval_archive_500_short/`.

This brackets the emergence of some walking between the inspected iteration-200
and iteration-500 archived checkpoints, not the precise first occurrence.
At 4,096 environments and 24 rollout steps, 500–1,000 iterations represent
approximately 49–98 million transitions. At 256 environments, the same sample
counts require 8,000–16,000 iterations, but changing batch size can alter learning
and does not guarantee equivalent results. A proposed next test is 1,000
iterations with 4,096 environments, stock settings, and intermediate evaluation.
That proposal was subsequently authorized and completed; see the results below.

## 4,096-environment capacity check

On 2026-10-27, a user-authorized 20-iteration stock PPO smoke test completed
successfully (exit 0) in `nick_sillywalk`, with 4,096 environments, 24 rollout
steps, and seed 42: 1,966,080 transitions. Total process time was 29.37 seconds.
GPU memory sampled approximately every 0.5 seconds peaked at 5,623 MiB of
16,303 MiB, including desktop usage, leaving 10,680 MiB of headroom. Sampled
GPU utilization peaked at 84%, temperature at 65°C. Later iterations ran at
approximately 0.78 seconds each. This supports using 4,096 environments for the
proposed headless stock experiment; it does not validate a learned gait or
guarantee the same memory usage with video rendering. The 1,000-iteration run
has not been started. No reward, environment-package, or shared Isaac Lab source
changes were made. Monitoring artifacts: `logs/stock_go2_capacity_4096/`
(`gpu.csv`, `summary.json`, `train.log`).

## Authorized 1,000-iteration stock experiment

Completed 2026-10-27 in `nick_sillywalk`, exit 0. Fresh initialization, seed 42,
4,096 environments, unchanged stock rewards/PPO, 24 rollout steps, 1,000
iterations: **98,304,000 transitions**, 80 times the earlier 200×256 run.
Training took 14 minutes 31 seconds including GPU contention from intermediate
evaluations. Run: `logs/rsl_rl/unitree_go2_velocity/2026-10-27_19-20-46_stock_go2_4096_1000/`;
final checkpoint `model_999.pt`. This is a separate run from the capacity smoke
test. Original Isaac Lab environment, shared source, and conga specification were
not changed.

Reproduction:

```bash
/home/rml2/anaconda3/envs/nick_sillywalk/bin/python scripts/rsl_rl/train.py \
  --headless --task Unitree-Go2-Velocity --num_envs 4096 \
  --max_iterations 1000 --seed 42 --run_name stock_go2_4096_1000
```

Mean return improved from −9.50 (first 20 iterations) to 27.33 (last 20).
Mean episode duration improved from 3.38 s to 19.79 s. Last-20 logged termination
fractions were 97.64% time limit and 2.36% bad orientation. Initial early
orientation failures recovered during training. The command curriculum reached
its full 1.0 m/s limit. These training statistics are not substitutes for rollout
evaluation.

Saved checkpoints 200, 500, and 999 were evaluated with the same 16 fixed-command
cases, 40 seconds per case, and 2-second post-reset warmup as the previous tests.
Each checkpoint had zero falls and 32 completed 20-second episodes. At iteration
200, translation remained planted. By 500, forward speed was 0.388 m/s for a
0.4 m/s command, with turning and diagonal support. Thus locomotion emerged
between the inspected 200 and 500 checkpoints; its exact onset was not measured.

Final checkpoint actual mean velocities:

| Command | Actual relevant velocity | Interpretation |
| --- | --- | --- |
| Forward +0.4 m/s | +0.355 m/s | Walks, modest underspeed |
| Backward −0.4 m/s | −0.367 m/s | Improved markedly from iteration 500 |
| Left +0.25 m/s | +0.003 m/s | Remains almost planted |
| Right −0.25 m/s | −0.130 m/s | Moves, but underspeed |
| Yaw +1 / −1 rad/s | +0.973 / −0.989 rad/s | Turns in place |
| Moving turn (0.3, 0.15, 0.5) | (0.279, 0.151, 0.519) | Tracks combined command |

Across the first eight cases, mean XY/yaw errors were:

| Checkpoint | XY error (m/s) | Yaw error (rad/s) |
| --- | --- | --- |
| New iteration 200 | 0.2293 | 0.2893 |
| New iteration 500 | 0.1141 | 0.0373 |
| New final (999) | 0.0681 | 0.0259 |
| Archived iteration 1,000 | 0.0896 | 0.0477 |

The archived checkpoint has 1,001 nominal iterations × 4,096 × 24 = 98,402,304
samples, approximately 0.1% more than the new final checkpoint. This is a much
closer budget comparison than the mature archive, but its mixed-terrain training
history differs. Both had zero falls here. One seed and one controlled evaluation
do not establish general superiority. New final errors on the eight initial-range
cases were 0.0610 m/s XY and 0.0165 rad/s yaw.

At +0.4 m/s, force contacts showed 90 touchdowns per foot over 35.96 scored
seconds (~2.50 Hz/foot); all observed two-foot support was diagonal. Four-foot
contact occupied 55% of scored time. Mean support-foot slip was 0.0344 m/s
forward and 0.0890 m/s backward. Sensor-event cadence agreed with force crossings
for this forward case, but the previously observed disagreement elsewhere remains
unresolved. This stock cadence is not the conga clock, and standing at zero
commands is expected for stock locomotion, not proof of conga marching.

**Recommendation:** retain stock rewards for now. The longer run established
basic locomotion without reward changes, but lateral tracking and low-speed
motion remain incomplete. Continue the walking baseline and evaluate those gaps
before introducing kick/bounce. The final conga phase/contact curriculum and
observation adapter still require separate validation; this stock policy is not
a completed conga policy. Evaluation disables observation noise and selected
randomization, so zero falls here does not establish robustness.

Artifacts: `logs/stock_go2_4096_eval_{200,500,final}/`,
`logs/stock_go2_eval_archive_1000/`, and `logs/stock_go2_4096_analysis/`.

![Longer-run training curves](../logs/stock_go2_4096_analysis/training_curves.png)

![Longer-run contact comparison](../logs/stock_go2_4096_analysis/contacts_comparison.png)

Forward video: [new 1,000-iteration policy](../logs/stock_go2_4096_video_final/videos/go2-step-0.mp4).

## Checkpoint recordings

Two approximately 10-second MP4 recordings use the same 0.4 m/s forward command
and following camera (960×720, 50 fps; 499 frames, 9.98 seconds each):

- [New 200-iteration policy](../logs/stock_go2_video_new/videos/go2-step-0.mp4)
- [Mature archived policy](../logs/stock_go2_video_reference/videos/go2-step-0.mp4)

Both files were decoded and sample frames inspected. The existing
`scripts/rsl_rl/play.py --video --video_length 500` supports general playback.
For a controlled command and output location, the added evaluator supports:

```bash
conda activate nick_sillywalk
python scripts/stock_go2_check.py --headless --video --steps 500 \
  --command 0.4 0 0 --checkpoint /absolute/path/model_N.pt \
  --output logs/my_go2_video
```

All evaluation CSVs, deployment metadata, hashes and summaries are in
`logs/stock_go2_eval_new`, `logs/stock_go2_eval_archive_200`, and
`logs/stock_go2_eval_archive_final`. Each has 32,000 rows. Analysis and the saved
configuration audit are in `logs/stock_go2_analysis/`. Artifacts are Git-ignored.
Final conga settings, the original `isaaclab` environment, and shared Isaac Lab
source were not modified.
