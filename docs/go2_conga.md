# Go2 conga task

## Latest agreed behavior: stand at neutral, restart on activation

The user now permits standing at zero operator command in both training and
final deployment. A transition from neutral to nonzero translation or yaw starts
the step pattern fresh; updates while already active do not restart it. Cadence
while active remains 120 BPM and independent of speed. Activation must use raw
operator intent, so the internal kick travel envelope cannot trigger a restart.
Deadband/hysteresis, safe settling when released, and rapid restart handling need
agreement before implementation. Motion-clock resets must not reset episode age.

This decision supersedes older continuous-zero-command marching requirements in
this document. Existing implementation descriptions, recorded experiments and
checkpoints retain their historical semantics; joystick restart support has NOT
been implemented or validated. The next discussed experiment is basic walking
with gentle duration shaping (0.20 s swing, 0.80 s per-foot stance), disabled at
neutral. Coefficients and tolerance bands remain proposals, and no new training
is authorized by this behavior agreement alone.


The motion contract is maintained in [AGENTS.md](../AGENTS.md). This is an initial
phase-conditioned PPO environment, not a trained or hardware-qualified gait.

The subsequent [stock Go2 control experiment](go2_stock_validation.md) includes
the 200-iteration training result, archived-policy comparison, and recorded
videos. Its stock rewards were not changed. See that report's budget differences
and contact-event diagnostic caveat before interpreting it as a conga baseline.

## Clone setup

Use the dedicated `nick_sillywalk` Conda environment on this machine. It is a
clone of `isaaclab`; keep all project package changes in `nick_sillywalk`.
The original `isaaclab` environment's editable `unitree_rl_lab` installation
has been restored to `/home/rml2/Documents/unitree_rl_lab/source/unitree_rl_lab`.
Install this checkout explicitly only in the dedicated environment: an existing
editable installation can otherwise import another clone.

```bash
conda activate nick_sillywalk
python -m pip install --no-deps --no-build-isolation -e source/unitree_rl_lab
git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_ros.git .local/unitree_ros
git -C .local/unitree_ros sparse-checkout set robots/go2_description
```

The Go2 assets are already downloaded for this checkout; skip cloning again.
`.local/` is ignored by Git. Robot configuration accepts `UNITREE_ROS_DIR` as
an environment override, then uses `.local/unitree_ros` when present, retaining
the original configured path as a fallback. The override must contain
`robots/go2_description/urdf/go2_description.urdf` and its meshes.
No shell activation hooks or unrelated robot asset paths were changed.
Isaac Lab remains linked to the existing `/home/rml2/IsaacLab` source checkout;
treat that shared source as read-only. Isaac Sim, Torch, and RSL-RL resolve
inside `/home/rml2/anaconda3/envs/nick_sillywalk`. Future dependency changes
must use that environment, never the original `isaaclab` environment.

## Architecture and timing

`conga_reference.py` is a pure batched Torch reference generator, independently
testable without Isaac Sim. `conga.py` adapts it to the existing uniform velocity
command, rewards, observations, and episode metrics. `conga_env_cfg.py` derives
from the existing Go2 task, retaining joint actions, actuator dynamics,
proprioception, velocity tracking, and stability penalties.

The clock is `episode_length_buf * step_dt`, with a 0.02 s control period and
0.005 s physics timestep. It resets independently per environment, remains
independent of command resampling and speed, and supports RSL-RL's initial random
episode lengths. Rewards and observations evaluate this same clock directly.
Isaac Lab updates commands **after** rewards, so storing phase only in a command
update would introduce stale reward timing. This implementation avoids that.
The relevant API is Isaac Lab's [command manager](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/api/lab/isaaclab.managers.html);
the installed `/home/rml3/IsaacLab` implementation was also inspected and tested.

The 0.25 s kick has its maximum at 1.375 s. At 50 Hz control, the reference is
sampled at adjacent times rather than exactly at this peak. The mathematical
trajectory and contact targets are continuous; discrete contact events have
approximately one control-step timing resolution.

Each phrase has three diagonal swings, one front-only kick, then recovery. The
recovery deliberately changes which diagonal leads the next phrase. A completely
uninterrupted two-diagonal trot would otherwise kick the same front leg every
2 seconds. The kick support pattern is a soft target: three contacts alone do
not prove that the center of mass lies inside the support triangle. The short
third swing and upper speed limit need learned-policy feasibility evaluation.
After kick touchdown, recovery contacts relax smoothly to neutral (0.5) over
1.50–1.60 s, allowing adaptive foot placement as travel resumes. They return to
support targets over 1.92–2.00 s before the next phrase. Neutral recovery contacts
receive neither swing-clearance credit nor touchdown timing penalties; this
avoids requesting motion while forcing four feet to remain planted.

Full kick displacement is 0.08 m outward from the default-pose front-foot lateral
coordinate, derived from the supplied URDF; body-frame x is not constrained.
Height means **sole clearance**, so the 0.022 m collision-sphere radius is removed
from the foot-link world height. Ground height is the environment origin's z on
the flat plane. This implementation must be extended with terrain sampling before
enabling rough terrain. Changing the robot's default joint pose or foot geometry
also requires updating `nominal_front_y` or `foot_radius`.

Translation and yaw both use a smooth travel envelope: decelerate from 1.10 to
1.25 s, pause through 1.50 s, accelerate through 1.70 s. Commands specify travel
speed, not phrase-average speed. XY vector magnitude is capped at 1.25 m/s before
the envelope, with no compensation for the pause. Pure yaw and zero commands
are explicitly sampled; zero commands retain the gait schedule.

## Observations, rewards, and export

The actor has 55 observations: the existing 45 followed by ten uncorrupted values:

1. sin/cos of the 4 s dance phase;
2. sin/cos of the 0.5 s beat phase;
3. travel multiplier;
4. capped raw vx, vy, yaw commands (so intent stays observable during a pause);
5. kick amplitude scale and bounce amplitude scale.

The ordinary three command observations contain the **enveloped** instantaneous
target. The critic has 70 observations. All five stages use these same dimensions
and the existing 12 joint-position actions.

Custom reward weights are configurable through `env.rewards.*`. Initial values:
contact 1.0, kick 1.0, body height 0.75, actual touchdown timing 2.0,
swing clearance 0.75, stationary marching 0.5. These are starting guesses.
Contact confidence decreases smoothly near expected liftoff/touchdown. The
touchdown reward uses actual contact-sensor events and a smooth timing kernel;
missing events receive no event credit and extra off-phase events incur cost.
Dense contact and clearance rewards disfavor standing still. Body vertical
velocity tracks the bounce derivative rather than zero. Velocity-dependent air
time, air-time variance, and stand-still joint-position costs are removed.

Existing training and playback export conventions remain in use. `deploy.yaml`
also contains reference parameters, foot ordering, phase feature ordering, and a
`requires_conga_observation_adapter` marker. The stock Go2 C++ controller does not
implement these ten features. Before sim-to-sim or hardware operation, implement
the same external clock, XY norm cap, travel envelope, feature order, and reset
behavior in the deployment observation adapter. Compare its outputs against
`generate_reference()` and `conga_features()` before enabling robot actions.
An ordinary Go2 velocity checkpoint is dimensionally incompatible; conga
curriculum checkpoints are compatible with each other.

## Incremental verification

Run from the repository root in the dedicated project environment:

```bash
conda activate nick_sillywalk
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_conga_reference.py
python scripts/list_envs.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python scripts/conga_check.py --headless --config-only
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python scripts/conga_check.py --headless --num_envs 8 --steps 220 --check-ppo-export
```

Disabling pytest plugin autoload avoids an unrelated ROS pytest plugin in this
machine's environment. `test_conga_mdp.py` needs the launched simulator's Python
modules and is run automatically by `conga_check.py`.

The smoke check validates all five registrations/configs, actual command/reward
adapters, foot name ordering, finite observations/rewards, speed cap, asynchronous
reset behavior, and deployment YAML. Optional PPO/export checking constructs the
existing RSL-RL runner, checks an actor forward pass, exports JIT/ONNX, and compares
the reloaded JIT actor against the runner. **It performs no learning updates.**
Generated untrained models are saved under `untrained_export`, never presented as
trained checkpoints. A zero-action rollout simply holds the default joint targets.

### Clone verification on rml2, 2026-10-27

After restoring the original `isaaclab` repository link and cloning that
environment into `nick_sillywalk`, verification was repeated in `nick_sillywalk`:
10 reference tests, 7 adapter tests, all five configurations, and an 8-environment
220-step full-conga GPU smoke test passed, including PPO/export checks, with
zero falls under zero actions and no learning updates. Artifacts are in
`logs/conga_nick_sillywalk_check/`; process output is in
`/tmp/conga_nick_sillywalk_check.log`. The original environment's `pip freeze
--all` output matched before and after the clone setup. Shared Isaac Lab source
was not modified.

Rechecked this checkout with Python 3.11.13, Isaac Sim 5.1.0, Isaac Lab 2.3.2,
RSL-RL 3.0.1, PyTorch 2.7.0+cu128, and the RTX 5070 Ti (16 GB).
The editable package now resolves to this repository. Official Go2 assets are
from `unitree_ros` commit `7919cba47a3f68888f3b084945b0ffbf3bb3763a`.

- All 10 reference tests and 7 adapter tests passed.
- All five task registrations and configuration/Hydra checks passed; the training
  CLI recognizes the conga stages.
- Robust-stage GPU simulation passed: 8 environments × 550 steps, finite
  observations/rewards, speed cap, partial resets, foot ordering/nominal geometry,
  randomization and pushes, with zero falls under zero actions.
- PPO construction, actor forward, TorchScript reload parity, ONNX serialization,
  and deployment configuration export passed. No learning updates were run.

Artifacts: `logs/conga_clone_check/` (4,400 CSV samples, deployment YAML and
explicitly untrained exports). Process logs: `/tmp/conga_clone_config.log` and
`/tmp/conga_clone_smoke.log`. Nonfatal URDF/USD head-visual warnings and RSL-RL
configuration compatibility warnings occurred. Visual appearance, ONNX runtime
parity, learned performance, and a 512-environment training workload remain
unverified. The simulation and export processes both exited successfully.

### Earlier verification on the source machine, 2026-10-27

Verified locally with Isaac Sim 5.1, the installed Isaac Lab checkout, RSL-RL
3.0.1, PyTorch 2.7.0+cu128, and an NVIDIA RTX 5080:

- 10 pure-reference tests and 7 controlled-state Isaac adapter tests passed.
- All five Gym registrations, configuration validation, Hydra registration,
  nested amplitude overrides, and configuration-instance isolation passed.
- Actual simulator foot ordering and nominal lateral FK matched the reference.
- Final robustness-stage smoke: 8 environments × 550 control steps (11 simulated
  seconds), finite observations/rewards, speed cap, partial resets, material/mass
  randomization and interval pushes. No falls occurred with zero actions; this
  establishes no dance performance because the robot was holding its default pose.
- Existing RSL-RL PPO construction and actor forward passed; TorchScript reload
  matched the runner output, and ONNX serialization passed. No optimizer updates
  were run. ONNX runtime parity and hardware adapter behavior remain untested.
- Black formatting, isort checks, flake8, and `git diff --check` passed on the changes.

Final artifacts are under `logs/conga_check_final/`: `rollout.csv` (4,400 samples),
`params/deploy.yaml`, and explicitly untrained JIT/ONNX exports. The detailed
process output is `/tmp/conga_final_check.log`. The simulator emitted URDF import
warnings and a Warp CUDA UUID lookup warning; the tested PhysX GPU rollout and
exports completed successfully. Other Warp-dependent workloads were not tested.

## Progressive training (requires approval before a long run)

Use the existing `scripts/rsl_rl/train.py`. Dance stages log under
`logs/rsl_rl/go2_conga`; basic preparation logs under `logs/rsl_rl/go2_conga_basic`.
Start with a small environment count and budget, inspect
the metrics and videos, then increase them as appropriate for the GPU.

```bash
# Stage 0: basic standing/walking, stock rewards/commands, conga phase inputs.
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Conga-Basic --num_envs 4096 --max_iterations 1000 --run_name basic_4096_1000

# Stage 1: fixed-cadence locomotion, marching and turning, reduced XY range.
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Conga-Cadence --num_envs 512 --max_iterations 1000 --run_name cadence

# Stage 2: resume a reviewed stage-1 checkpoint and introduce bounce.
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Conga-Bounce --num_envs 512 --max_iterations 1000 --resume --load_run '<cadence-run-directory>' --checkpoint model_1000.pt --run_name bounce

# Stage 3: resume bounce; start the kick at 25% lateral amplitude.
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Conga-Kick --num_envs 512 --max_iterations 1000 --resume --load_run '<bounce-run-directory>' --checkpoint model_1000.pt --run_name kick25

# Continue stage 3 from the latest reviewed checkpoint at 50%, then 75%, then 100%.
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Conga-Kick --num_envs 512 --max_iterations 1000 --resume --load_run '<kick-run-directory>' --checkpoint '<checkpoint-file>' env.commands.base_velocity.reference.kick_scale=0.5

# Stage 4: full dance and full commanded velocity range.
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Conga --num_envs 512 --max_iterations 1000 --resume --load_run '<kick100-run-directory>' --checkpoint '<checkpoint-file>' --run_name full

# Stage 5: restore the standard Go2 material/mass randomization and pushes.
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Conga-Robust --num_envs 512 --max_iterations 1000 --resume --load_run '<full-run-directory>' --checkpoint '<checkpoint-file>' --run_name robust
```

Replace placeholders and checkpoint filenames with actual reviewed outputs;
iteration budgets above are examples, not validated convergence thresholds.
The stage transitions are manual so each component can be checked before
introducing the next. There is no automatic success claim or iteration-triggered
curriculum advancement. Retain measurement of zero-command behavior at every
stage; do not advance solely because aggregate reward rises.

The basic stage was added at the user's request after the successful stock control.
It inherits the complete stock environment, including its flat mesh terrain,
command curriculum, physical penalties, and randomization. Only ten conga inputs
are appended: dance/beat sine and cosine, travel=1, three raw commands, and zero
kick/bounce scales. The 120 BPM clock runs independently of speed, resets per
environment, and is not rewarded yet. Zero commands permit standing. Actor/critic
dimensions are 55/70, with the same term order and 12 actions as later stages.
PPO uses 24 rollout steps to match the stock sample budget, rather than the later
stages' 48. No rejected walking-v2 rewards are used. A matching layout supports
checkpoint loading but does not establish successful transfer to timed stepping.
Basic-to-cadence transfer needs an explicit checkpoint path across the two
experiment directories and a reviewed cadence reward transition; the stage-1
command above is a standalone example, not automatic transfer.

### Basic-stage experiment, 2026-10-27

The user authorized **only basic standing/walking**, not advancement to cadence,
bounce, or kick. The fresh seed-42 run completed 1,000 iterations with 4,096
environments and 24 rollout steps: 98,304,000 transitions, matching the stock
control budget exactly. Training elapsed time was 15 minutes 15 seconds, including
GPU contention during checkpoint evaluations. Run directory:
`logs/rsl_rl/go2_conga_basic/2026-10-27_19-45-47_basic_4096_1000/`;
final checkpoint `model_999.pt`. No stock weights were transferred.

Verification before the long run: 10 reference tests and 12 adapter/config tests
passed, including exact stock configuration parity after removing the added phase
observations and reference metadata, phase/reset consistency, and Hydra registration.
Unrelated ROS pytest plugins required `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; no
packages were changed. The first 8-environment smoke process aborted during native
Isaac Sim initialization (`double free or corruption`), before training. A clean
retry completed two PPO iterations, followed by a 220-step rollout with 55/70
actor/critic dimensions, checkpoint loading, deployment metadata and TorchScript
parity checks. The crash cause is not established; the full run exited 0.

All 16 reward terms are inherited unchanged from stock: XY tracking +1.5 and yaw
tracking +0.75 (both std 0.5); vertical velocity −2; roll/pitch angular velocity
−0.05; joint velocity −0.001; acceleration −2.5e−7; torque −2e−4; action rate
−0.1; joint limits −10; energy −2e−5; orientation −2.5; joint-position cost −0.7
(stock standing multiplier 5); foot air time +0.1; air-time variance −1; foot slip
−0.1; undesired contacts −1. No conga contact, phase-timing, marching, bounce or
kick rewards were active. The reference clock remained observable but unrewarded.

First-20 versus last-20 training means: return −11.31 → 24.88, episode duration
4.06 → 19.64 seconds. Last-20 termination fractions were 96.04% time limit,
3.93% bad orientation, and 0.026% base contact. The command limit reached 1.0 m/s.
The stock control's last-20 return was 27.33 and episode duration 19.79 seconds;
the basic-stage training statistics are slightly worse despite some better
controlled-command results.

The same evaluator and commands used for stock were run on basic checkpoints
200, 500 and 999: 16 cases × 40 seconds, with two seconds excluded after each
reset for steady-state scoring. Each evaluation had **zero falls and 32 completed
20-second episodes**. Checkpoint 200 was mostly planted; checkpoint 500 walked
forward, backward, laterally and turned. This brackets walking emergence between
the inspected 200 and 500 checkpoints, as in stock; exact onset was not measured.

| Fixed command | Basic final actual velocity | Stock final actual velocity |
| --- | --- | --- |
| Forward +0.4 m/s | +0.377 m/s | +0.355 m/s |
| Backward −0.4 m/s | −0.308 m/s | −0.367 m/s |
| Left +0.25 m/s | +0.172 m/s | +0.003 m/s |
| Right −0.25 m/s | −0.155 m/s | −0.130 m/s |
| Turn +1 rad/s | +0.984 rad/s | +0.973 rad/s |
| Turn −1 rad/s | −0.969 rad/s | −0.989 rad/s |
| Moving turn (0.3, 0.15, 0.5) | (0.282, 0.134, 0.502) | (0.279, 0.151, 0.519) |

Across the first eight cases, basic final mean XY error was 0.0526 m/s versus
stock 0.0681; yaw error was 0.0270 rad/s versus 0.0259. On the initial-range
eight cases, basic XY/yaw errors were 0.0619/0.0253 versus stock 0.0610/0.0165.
These are comparable results, with tradeoffs rather than uniform improvement.
Changing input dimension also changes network initialization and stochastic
training, even at the same seed; this single run does not prove a benefit from
phase inputs.

At +0.4 m/s, all observed two-foot support was diagonal (42.3% of scored time);
four-foot contact occupied 44.1%. Force crossings counted 72/72/72/70 touchdowns
over 35.96 seconds, about 1.95–2.00 Hz per foot. This cadence is not evidence of
phase synchronization: no timing reward was active. Sensor first-contact event
rates again disagreed with force crossings, so that diagnostic issue remains
open before cadence/touchdown reward work. Forward support-foot slip was
0.0336 m/s (stock 0.0344); backward slip was **0.1486 m/s** (stock 0.0890), a
material weakness despite the absence of falls. At zero commands the policy
mostly stood, with about 0.0130 m/s XY velocity error; marching was not required.

**Conclusion:** the basic stage reproduced the expected broad stock walking
behavior with the future conga observation layout. No reward rewrite is justified
by this comparison alone. Retain the checkpoint; review backward slip, lateral
underspeed and low-speed tracking before advancing. Do not infer robustness from
the controlled tests, which disable observation noise and selected randomization.
No cadence, bounce or kick training was started, and the final conga motion targets
remain unchanged.

Artifacts: `logs/conga_basic_eval_{200,500,final}/`,
`logs/conga_basic_analysis/`, and source/config snapshots in the training run.
Reproduce evaluation with `scripts/stock_go2_check.py --headless --task
Unitree-Go2-Conga-Basic --checkpoint <checkpoint> --output <directory>`.

![Basic-stage training curves](../logs/conga_basic_analysis/training_curves.png)

![Basic versus stock contact patterns](../logs/conga_basic_analysis/contacts_comparison.png)

[Basic-stage walking video](../logs/conga_basic_video_final/videos/go2-step-0.mp4)
uses the same +0.4 m/s command and camera as the stock recording.

### Cadence reward calibration, 2026-10-27 — no training

The user agreed to review two dense phase terms before resuming basic into
cadence. No cadence configuration or checkpoint was changed. The existing CSVs
had contacts/velocities but lacked sole heights and individual stock rewards, so
the final basic checkpoint was reevaluated with `--reward-audit`. This bounded
16-case × 40-second rollout had zero falls and 32 timeouts. Forward velocity
samples reproduced the original recording exactly (maximum difference 0), and
all scored foot-contact samples agreed. No optimizer steps occurred.

The evaluator now optionally records world-z sole clearance above local ground
(foot radius 0.022 m) and the reward manager's weighted per-term rates before
`step_dt`. At every simulation step it asserts that the sum of recorded rates
times 0.02 equals the environment reward. Reset rows are excluded, as are the
first two seconds of each episode: 28,800 scored samples remain. This evaluates
the controlled policy trajectory, not training-time noisy/randomized states.

`scripts/score_cadence_rewards.py` scores these recordings offline with the
existing continuous-diagonal reference (kick/bounce off, alternating every
0.5 s, 1 Hz per foot, 3 cm peak sole clearance). Candidate contact score is
`1 - mean((2*target_contact-1)^2 * (contact-target_contact)^2)`. Candidate clearance
uses the existing `swing_quality` formula with actual unloading and lift required,
mid-swing emphasis, and 0.015 m Gaussian width. These functions are scored only;
they have not been installed in the cadence task.

All numbers below are reward rates per simulation second, before dt, unless
identified as unweighted scores. Multiply by 0.02 for a control-step contribution.

| Measured quantity | Mean across all 16 commands |
| --- | ---: |
| XY tracking reward | +1.47267 |
| Yaw tracking reward | +0.74601 |
| Joint-position penalty | −0.18407 |
| Stock air-time reward | −0.00259 |
| Air-time variance penalty | −0.03779 |
| All remaining physical penalties combined | −0.08689 |
| Current total stock reward rate | +1.90732 |
| Same trajectory after removing the three retiring terms | +2.13178 |
| Candidate contact score, unweighted | 0.64467 |
| Candidate clearance score, unweighted | 0.01123 |

The stock air-time term is signed despite its positive coefficient; the measured
short swings give negative credit. For the first zero-command case, the
joint-position cost alone is −0.54565 and air-time variance −0.05716. Removing
these conflicts is a significant change, not merely a housekeeping adjustment.
Retained vertical-velocity cost averages only −0.00403, supporting keeping its
stock −2 coefficient for this no-bounce stage.

| Foot-motion benchmark | Mean contact score | Mean clearance score |
| --- | ---: | ---: |
| Recorded forward walking (+0.4 m/s) | 0.61889 | 0.03357 |
| Recorded backward walking (−0.4 m/s) | 0.63778 | 0.00768 |
| Planted feet (synthetic) | 0.70000 | 0 |
| Reference-aligned feet (synthetic) | 1.00000 | 0.36318 |
| Wrong-diagonal reference (synthetic) | 0.39056 | approximately 0 |
| All feet unloaded at reference heights (synthetic) | 0.34000 | 0.36319 |

The synthetic rows replace only feet and are not feasible whole-body simulations
or estimates of their torque/slip costs. Contact matching alone prefers planted
feet to the current mistimed trot, although it prefers correctly timed feet to
both. Clearance complements contact matching; its small current average is not
a reason to blindly multiply its weight until it matches velocity reward.
Reference-perfect clearance is not exactly the mathematical maximum near low
reference heights: the lift factor divides by `max(target_height, 0.015)` and can
favor a slight height overshoot. Mid-swing target remains 3 cm. This limitation
must be retained in interpretation or separately revised before implementation.

A phase-offset scan still gives only 0.65444 maximum forward contact score versus
0.61889 at the actual clock; a simple phase shift cannot fix the roughly 2 Hz
measured foot cadence versus the 1 Hz target. Clearance width sensitivity for
forward motion is 0.02816 / 0.03357 / 0.03706 at widths 0.010 / 0.015 / 0.020 m;
the reported candidate retains 0.015 rather than silently changing the tolerance.

**Proposed coefficients for discussion, not applied:** start gently at contact
**+0.25**, clearance **+0.5**, with a reviewed ceiling of contact **+0.5**,
clearance **+1.0**. Fade out joint-position, stock air-time and air-time-variance
terms while introducing these terms; the ramp duration and progression checks
remain to be agreed. Keep the other stock terms, 24-step PPO rollout and training
randomization. Do not add touchdown-event, separate marching, height, bounce or
kick rewards in this step.

At +0.25/+0.5 the new terms contribute +0.16678 on the recorded trajectories;
at +0.5/+1.0 they contribute +0.33357. The reference-feet improvement over those
trajectories would be +0.26481 or +0.52961 respectively, approximately 12% or 24%
of measured velocity-tracking reward (+2.21868). At the upper pair, the
reference-feet versus planted-feet gap is +0.51318. These are scale-based choices,
not paper-derived constants, PPO advantages, or predictions that exploration will
find the gait. Static scores cannot establish the cost of actual 1 Hz marching,
learning stability, or robustness. The sensor-event/force-crossing discrepancy
remains unresolved, so no event-based touchdown reward is recommended yet.

Reproduction:

```bash
python scripts/stock_go2_check.py --headless --task Unitree-Go2-Conga-Basic \
  --reward-audit --checkpoint logs/rsl_rl/go2_conga_basic/2026-10-27_19-45-47_basic_4096_1000/model_999.pt \
  --output logs/conga_cadence_reward_audit
python scripts/score_cadence_rewards.py --rollout logs/conga_cadence_reward_audit \
  --output logs/conga_cadence_reward_scores
```

Use `nick_sillywalk` for both commands. Per-case distributions, all stock terms,
weight sweeps and phase/tolerance sensitivity are saved in
`logs/conga_cadence_reward_scores/summary.json`.

![Cadence reward scores and contact targets](../logs/conga_cadence_reward_scores/scores.png)

### Approved cadence starting values and speed cap

Following the scoring review, the user approved contact **+0.25**, clearance
**+0.5**, and a **0.5 m/s maximum XY command norm** to prioritize marching over
speed. These are prepared in `Unitree-Go2-Conga-Cadence-Start`, separate from
the legacy `Unitree-Go2-Conga-Cadence` configuration inherited by bounce/kick.
Use the new task for the reviewed transition once its remaining schedule is
agreed; the old cadence command above does not select these new settings.

Forward/backward sampling is ±0.5 m/s, lateral sampling remains ±0.25 m/s, and
combined XY commands are norm-capped at 0.5 m/s. This is a command limit, not a
hard bound on actual robot velocity. Yaw remains ±1 rad/s. The 120 BPM clock and
zero-command swing targets remain independent of speed. Kick/bounce are off.
Stock physical rewards/randomization and the basic stage's 24-step PPO rollout
are retained. Only the two scored dense phase rewards are added.

The user subsequently approved the fade and training. The three retiring terms
(joint position, air time, air-time variance) now scale by
`max(0, 1 - completed_cadence_control_steps / 4800)` before each control step.
This is 200 PPO iterations at 24 steps, independent of episode resets and the
basic checkpoint's iteration count. The cadence-only runner saves progress in
checkpoint `infos.cadence_stage`; later cadence resumes restore it and start at
the next iteration. Basic transfer restores policy/optimizer but starts the new
stage count at zero. Contact/clearance stay +0.25/+0.5, and higher coefficients
remain unapproved. Final conga maximum travel speed remains 1.25 m/s.

The authorized experiment targets 1,000 additional iterations with 4,096
environments, gated by evaluation after 50, 100 and 200 iterations. It may stop
early if stability deteriorates. Segmented resumes retain policy, optimizer and
fade progress but reset simulated worlds; this is not an uninterrupted trajectory.
The original Isaac Lab environment and shared source remain unchanged.

### Authorized cadence fade experiment — completed 2026-10-27

**Outcome: walking survived the fade, but cadence and coordinated marching were
not learned. Do not advance to bounce or kick.** The user-approved contact +0.25,
clearance +0.5 and 0.5 m/s XY norm cap stayed fixed. No unapproved weight increase
or additional reward was introduced.

The final basic `model_999.pt` was loaded with its optimizer. Cadence then trained
for 1,000 additional iterations × 4,096 environments × 24 rollout steps =
**98,304,000 additional transitions**. The stage's checkpoint count starts at zero
independently of basic training. Training was segmented at completed iterations
50, 100 and 200 for the agreed evaluations, then continued to 1,000. The final
checkpoint contains `infos.cadence_stage.control_steps=24000`; intermediate
checkpoints contained 1200, 2400 and 4800 steps. The fade was not restarted by
resume or episode resets. Simulated worlds were reset between training segments;
policy, optimizer and fade progress were retained.

Training runs under `logs/rsl_rl/go2_conga_cadence_start/`:

- `2026-10-27_20-27-29_cadence_fade_to50/model_49.pt`
- `2026-10-27_20-30-26_cadence_fade_to100/model_99.pt`
- `2026-10-27_20-33-22_cadence_fade_to200/model_199.pt`
- **`2026-10-27_20-37-17_cadence_fixed_to1000/model_999.pt`**

Combined reported training time was 20 minutes 53 seconds, including GPU
contention during the additional halfway evaluation, excluding sequential
evaluation/startup overhead. The combined TensorBoard record contains all 1,000
iteration indices without overlap. Last-20 means: return 40.99, episode duration
19.92 s, timeout fraction 98.58%, bad orientation 1.39%, base contact 0.028%.
The return increase is confounded by removing negative rewards and adding phase
rewards; it is not evidence of cadence success.

Every controlled evaluation (50, 100, 200, 501 completed iterations via
`model_500.pt`, and final 1,000) had **zero falls and 32 completed 20-second
episodes** across 16 commands × 40 seconds. Physics/noise randomization is
disabled in these tests as before, so this is not a robustness claim. The retiring
reward weights are frozen at the checkpoint's fade position during evaluation.

| Evaluation | XY error (m/s) | Contact score | Clearance score | Missing target touchdowns / 1,152 |
| --- | ---: | ---: | ---: | ---: |
| Basic checkpoint | 0.0526 | 0.6304 | 0.0156 | 909 |
| Cadence 50 | 0.0631 | 0.6520 | 0.0340 | 856 |
| Cadence 100 | 0.0447 | 0.6461 | 0.0372 | 832 |
| Cadence 200 | 0.0301 | 0.6454 | 0.0568 | 718 |
| Cadence 501 | 0.0311 | 0.6321 | 0.0965 | 704 |
| Cadence 1,000 | 0.0357 | 0.6316 | 0.1034 | 693 |

These aggregates use the first eight fixed commands. Contact/clearance are
unweighted scores. Touchdowns are force-contact crossings matched one-to-one
within ±80 ms of the scheduled per-foot touchdown; reset boundaries and the
first two seconds are excluded. These are diagnostic tolerance counts, not
event rewards. Final extra contacts numbered 1,570 across these eight tests.
The known first-contact sensor discrepancy remains open and that signal was not
used to score touchdown matching or reward cadence.

Final mean velocities: forward 0.401 m/s for +0.4; backward −0.392 for −0.4;
left +0.199 for +0.25; right −0.247 for −0.25; yaw +1.009/−0.959 rad/s for ±1;
moving turn (0.315, 0.151, 0.513) for (0.3, 0.15, 0.5). Mean yaw error across
the main eight cases was 0.0316 rad/s (basic 0.0270). Mean support-foot slip was
0.0393 m/s (basic 0.0499); backward slip remained 0.1036 m/s.

**Cadence failure:** forward force crossings were ~2.00 Hz on every foot, against
the requested **1 Hz per foot** (diagonals alternate every 0.5 s). In both
zero-command cases the rear-left foot repeatedly unloaded at ~2 Hz while the
other feet mostly remained supporting. Each missed 108/144 scheduled touchdowns.
Forward contact matching was 0.610 (basic 0.619), while clearance rose from 0.034
to 0.120. The policy improved clearance within its old fast gait rather than
learning the slower diagonal sequence. Partial foot lifting also earns clearance
credit without coordinated marching.

The fade schedule appears tolerable for preserving locomotion in this experiment.
The data do **not** justify calling cadence solved or simply advancing stages.
Next discussion should address contact/clearance balance and credit for completing
both feet of the scheduled diagonal, including the extra off-phase swings. A
stronger timing incentive may help, but no new coefficients or reward formulas
were applied after this experiment, and static/contact diagnostics should be
reviewed before authorizing another run.

Verification: 14 adapter/config/fade tests passed before training, including
50/100/200-iteration fractions, basic-to-cadence count reset, cadence resume
continuity, vector speed cap, and unchanged physical/cadence weights. A separate
8-environment two-iteration resume smoke test passed; its 384 transitions are not
part of the reported budget. All training and evaluation processes exited 0.
No original `isaaclab` packages or shared Isaac Lab source were modified.

Resume entry point (budget is additional iterations):

```bash
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Conga-Cadence-Start \
  --num_envs 4096 --max_iterations 50 --seed 42 --resume \
  --checkpoint_path /absolute/path/to/reviewed_checkpoint.pt --run_name cadence_segment
```

The bounded evaluator supports the cadence task with `--reward-audit`.
`scripts/analyze_cadence_gate.py` measures each rollout;
`scripts/analyze_cadence_experiment.py` combines the segmented training curves
and gate comparisons. Artifacts are `logs/conga_cadence_eval_{50,100,200,500,final}`
and `logs/conga_cadence_analysis/`; source snapshots and training logs accompany
each run.

![Full cadence training history](../logs/conga_cadence_analysis/training_curves.png)

Videos of the final checkpoint:
[zero-command behavior](../logs/conga_cadence_video_march/videos/go2-step-0.mp4),
[forward walking](../logs/conga_cadence_video_forward/videos/go2-step-0.mp4).

### Offline contact-weight and paired-diagonal comparison

After the failed cadence experiment, the user authorized offline comparison of
higher contact weights and rewards requiring both scheduled diagonal feet.
`scripts/compare_cadence_candidates.py` reads the saved basic audit and final
cadence rollouts, with the same two-second warmup/reset exclusions (28,800 samples
per rollout). No simulation, policy updates, or task/reward configuration changes
were made for this comparison. Results are in
`logs/conga_cadence_candidate_comparison/summary.json`.

Candidates:

- Existing mean contact matching, with weights 0.25, 0.5, 1.0 and 2.0; clearance
  stays at 0.5.
- **Paired clearance:** reference swing emphasis × the lesser clearance quality
  of the scheduled diagonal's two feet × contact indicators for both opposite
  support feet. It retains the existing 3 cm trajectory, lift requirement and
  0.015 m tolerance. Support means the existing force threshold is crossed, not
  a claim of sufficient support force or dynamic stability.
- **Whole-pattern contact:** product of all four soft per-foot contact-match
  scores instead of their mean. Correct swing and support must occur together.
- A 50/50 independent/paired-clearance blend, to examine denser feedback versus
  retained credit for partial movement.

All synthetic cases are reference-only foot-state substitutions over complete
four-second cycles, not feasible robot trajectories. Single-foot cases follow
the desired schedule for just that foot; they are distinct from the observed
rear-left ~2 Hz behavior. Included patterns are planted feet, each individual
foot, one front foot per beat, correct diagonals, wrong diagonals, all feet
unloaded, and double-frequency motion at several offsets. Sanity assertions
verify that strict paired clearance rejects planted/single-foot/wrong-diagonal/
all-unloaded states, stays below independent clearance, and rewards the complete
reference. Whole-pattern contact stays in [0,1].

**Increasing contact weight alone does not remove partial-step credit.** The
existing contact score is 0.700 for planted feet, 0.775 for one correctly timed
foot, 0.850 for one foot each beat, and 1.000 for complete reference diagonals.
Thus a stronger coefficient also strengthens credit for incomplete movement.

| Contact weight (clearance fixed +0.5) | Recorded final forward phase reward | Synthetic planted phase reward |
| --- | ---: | ---: |
| 0.25 | 0.2129 | 0.1750 |
| 0.50 | 0.3655 | 0.3500 |
| 1.00 | 0.6707 | 0.7000 |
| 2.00 | 1.2811 | 1.4000 |

These are only phase-reward contributions per second before dt. Velocity tracking
and physical costs still matter; the table does not predict that a policy would
necessarily stop walking. For the observed zero-command rear-foot lifting, phase
reward rises from 0.2125 at contact 0.25 to 0.7372 at contact 1.0, versus planted
0.1750/0.7000. Increasing contact alone barely changes its advantage over standing.

Strict paired clearance identifies the observed problem:

| Recorded final motion | Existing independent clearance | Strict paired clearance |
| --- | ---: | ---: |
| Zero command, case 0 | 0.07525 | 0.000003 |
| Zero command, case 8 | 0.07523 | 0 |
| Forward | 0.12050 | 0.07379 |
| Backward | 0.09816 | 0.05503 |
| Left | 0.08563 | 0.02670 |
| Right | 0.11342 | 0.05493 |

Across the main eight commands, independent/paired clearance averages were
0.01555/0.00899 for basic and 0.10344/0.04920 for final cadence. Moving final
cases retain positive strict-pair credit on about 12–17.5% of scored samples.
Zero-command cases have positive pair credit on 0–0.056% of samples, showing
both removal of the undesired reward and a potential exploration difficulty.

At unchanged coefficients contact 0.25 / clearance 0.5:

| Synthetic pattern | Current mean-contact + independent | Whole-pattern contact + paired |
| --- | ---: | ---: |
| Planted | 0.1750 | 0.1000 |
| One correctly timed foot | 0.2391 | 0.1000 |
| One foot each beat | 0.3033 | 0.1000 |
| Complete reference diagonals | 0.4316 | 0.4316 |
| All feet unloaded at reference heights | 0.2666 | 0 |

Paired clearance alone leaves some single-foot credit through the mean-contact
term (0.1938 versus planted 0.1750 for the single-foot synthetic case). Pairing
both contact and clearance removes that advantage in these benchmarks. The
50/50 clearance blend still gives single-foot synthetic phase reward 0.2164 and
observed zero-command clearance about 0.0376, so it reduces rather than resolves
the partial-step incentive.

**Recommendation for discussion:** prioritize reward structure over increasing
contact to 1 or 2. Whole-pattern contact plus paired clearance at the existing
0.25/0.5 coefficients is a better diagnostic candidate: it keeps the complete
reference score while removing partial-pattern credit in these tests. However,
its feedback is sparser, particularly at zero command, and offline rankings do
not demonstrate learnability or a successful transition. A bounded stochastic
rollout without learning could quantify whether the current policy's exploration
ever produces rewarded pairs before choosing an introduction schedule. No new
candidate is installed in the training task and no further training is authorized.

![Contact weight and diagonal-pair comparison](../logs/conga_cadence_candidate_comparison/comparison.png)

Reproduce with `python scripts/compare_cadence_candidates.py` in `nick_sillywalk`.
The JSON includes every command, candidate score distribution, nonzero-credit
fraction, checkpoint hash, and rescored total for the recorded trajectory only.
Existing phase terms and retired stock terms are excluded from the retained
reward subtotal to prevent double counting when rescoring the final rollout.

### Offline shorter-swing scoring

The user requested scoring a 0.20-second swing / 0.30-second four-foot-support
schedule. Only the offline scorer changed. No training, simulation, or active
task/reference edits were performed. The candidate preserves 120 BPM, alternates
diagonals every 0.5 s (1 Hz per foot), retains 3 cm sole clearance and the 0.04 s
soft contact ramps, and centers each swing at the original 0.22 s phase. Thus
windows change from [0.04, 0.40] to [0.12, 0.32] within each beat. Window duration
is not identical to actual airborne time; transitions are soft and actual motion
has not been simulated under the proposed target.

`scripts/compare_cadence_candidates.py --swing-duration .20` applies the variant
only when scoring saved data. Its default .36-second implementation was checked
numerically against the production reference. Counterfactual rejection/bounds
assertions passed for both durations. Both saved policies contribute 28,800
scored samples each, with no change to physical trajectories or velocity errors.

For the proposed whole-pattern contact + paired clearance structure, with
coefficients fixed at +0.25/+0.5:

| Quantity | 0.36 s swing / 0.14 s support | 0.20 s swing / 0.30 s support |
| --- | ---: | ---: |
| Synthetic planted phase reward | 0.1000 | 0.1800 |
| Synthetic single-foot phase reward | 0.1000 | 0.1800 |
| Synthetic complete-reference phase reward | 0.4316 | 0.3510 |
| Reference advantage over planted | 0.3316 | 0.1710 |
| Recorded final forward paired clearance | 0.07379 | 0.03810 |
| Recorded final forward nonzero pair-credit fraction | 17.5% | 12.72% |
| Recorded final main-eight paired clearance | 0.04919 | 0.02680 |

These are phase scores/rates before dt, not physical returns for synthetic
behaviors. Longer support intervals give planted feet more contact credit, while
shorter swing windows provide fewer opportunities to earn clearance credit. The
complete-reference advantage over standing shrinks by about 48% at unchanged
weights. Under the existing independent-foot structure, planted mean contact
also rises from 0.70 to 0.86. Higher contact scores after shortening the target
therefore must not be mistaken for improved cadence in the unchanged recordings.

The strict pair score continues to reject the observed zero-command rear-foot
lifting: effectively zero for both durations (shortened case 0 approximately
0.000006; case 8 zero). It consequently remains an exploration concern. Positive
pair feedback on moving final-policy cases ranges from 10% to 14% of samples
under the shortened midpoint-preserving window; basic-policy forward pair
feedback decreases from 15.44% to 7%.

Timing sensitivity was also tested at short-window centers 0.12 and 0.32 s.
Final main-eight paired clearance is 0.01560 / 0.02680 / 0.00034 at centers
0.12 / 0.22 / 0.32 s. Thus offline reward availability depends substantially on
alignment with the inherited gait. These observations do not select a learned
optimal phase or show that the policy can adapt to another phase.

Mechanical implication of the unchanged 3 cm sin-squared height pulse:
peak reference vertical foot speed increases from 0.262 to 0.471 m/s (1.8×),
and peak reference vertical acceleration from 4.57 to 14.80 m/s² (3.24×).
These follow analytically from `height*pi/duration` and
`2*height*pi²/duration²`. They describe the requested foot trajectory, not body
acceleration, measured joint loads, or a demonstrated feasibility limit.

**Interpretation:** shorter swings provide the intended longer four-foot-support
opportunity, but this audit cannot establish improved stability. Simply shortening
the window at fixed coefficients makes the standing alternative more competitive
and pair feedback sparser. Before training, discuss balancing the swing and
support reward contributions by their available duration so longer support does
not automatically increase standing credit, and assess the faster foot motion
with a bounded evaluation. No coefficient, phase-normalization, or training change
has been authorized or applied. Final conga motion targets remain unchanged.

Artifacts: `logs/conga_swing_duration_long/`, `logs/conga_swing_duration_short/`,
`logs/conga_swing_duration_short_early/`, `logs/conga_swing_duration_short_late/`;
each includes detailed JSON and a plot. Reproduce the sensitivity cases with
`--swing-duration .20 --swing-center .12` or `.32` and a distinct `--output`.

![Shorter swing schedule and reward comparison](../logs/conga_swing_duration_comparison/comparison.png)

Playback and bounded evaluation:

```bash
python scripts/rsl_rl/play.py --task Unitree-Go2-Conga --num_envs 8 --checkpoint '/absolute/path/model_N.pt'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python scripts/conga_check.py --headless --task Unitree-Go2-Conga --num_envs 8 --steps 2000 --checkpoint '/absolute/path/exported/policy.pt' --output logs/conga_evaluation
```

Match the evaluation task/stage to the trained checkpoint. For intermediate kick
amplitudes, change the evaluation config's reference scale to the trained value
before evaluating; `play.py` does not expose Hydra overrides.

## Revised paired cadence scheme (prepared, not trained)

The separate `Unitree-Go2-Conga-Cadence-Paired` task implements the revised
candidate. The previous Cadence-Start experiment and the final conga motion
remain unchanged. This candidate uses swings at [0.12, 0.32] seconds of each
0.5-second beat: 0.20 seconds of swing followed by 0.30 seconds of support.
Diagonals alternate at 120 BPM, giving each foot a 1 Hz target. Clearance remains
3 cm, with soft contact transitions. Translation remains capped at 0.5 m/s;
zero-command marching and turning remain enabled. Bounce and kick are disabled.

| Reward | Weight | Requirement |
| --- | ---: | --- |
| Four-foot support | +0.10 | All four feet contact during scheduled support |
| Paired step | +0.50 | Both scheduled feet clear the ground together, while the opposite pair supports |
| Phase matching | +0.50 | Match the complete four-foot contact pattern; 80% of its budget goes to swing and 20% to support |

Each underlying component is divided by its mean on a perfect reference cycle,
computed at the simulation control timestep. These policy-independent constants
give each component a fixed reference-cycle budget rather than giving longer
support intervals a larger budget. Instantaneous normalized scores can exceed
one. Paired-step credit uses the minimum clearance quality of the two scheduled
feet, multiplied by opposite-pair contact and reference swing activity. Phase
matching multiplies all four soft contact matches. This is dense phase matching,
not a measured-frequency or touchdown-event bonus; the 1 Hz frequency is enforced
indirectly by the periodic targets. No sparse sensor-event reward is introduced.

The synthetic short-swing cycle scores are 1.10 for the reference, 0.25 for
standing, 0.25 for single-foot stepping or the wrong diagonal, and zero for flight.
Three tested double-frequency offsets score 0.145–0.416. These are mean cadence
reward rates only, not full policy returns or guarantees of learned stability.
Rescoring the saved final Cadence-Start rollout gives 0.172 at zero command and
0.348 moving forward. Results and synthetic assertions are reproducible with
`scripts/score_paired_scheme.py`; output is
`logs/conga_paired_scheme/scores.json`.

The coefficients are engineering choices based on these normalized budgets and
counterexample scores, not values established by a paper or training sweep.
Velocity tracking and physical penalties are inherited. The same 200-iteration
fade of the three retired stock rewards is retained when starting from basic;
resuming a cadence checkpoint preserves its saved fade progress. The old mean
contact and individual clearance terms are replaced, not added on top.

Changing reward weights does not reduce the kinematic demands of the unchanged
3 cm, 0.20-second reference pulse: peak vertical foot speed is about 0.471 m/s
and acceleration 14.80 m/s². A lower height or longer swing would be a separate
trajectory decision. Strict paired credit also remains sparse on the existing
zero-command behavior, so learning must still be evaluated.

Verification: 10 reference tests and 15 adapter/configuration tests passed. A
220-step, 16-environment, no-learning rollout of the basic checkpoint completed
with zero falls; reward auditing and policy export checks passed. Artifacts are
in `logs/conga_paired_scheme/smoke`. This is a short integration smoke test, not
evidence that the new cadence has been learned. No new training was started.

Introducing mild phase/contact shaping during basic locomotion is worth a
controlled comparison: that policy already observes phase but previously had no
reward for using it. Joint gait and locomotion conditioning has precedent in
[Walk These Ways](https://proceedings.mlr.press/v205/margolis23a.html), but that
does not validate these weights or prove an advantage here. A gradual phase
reward ramp after initial balance competence is a reasonable experiment before
imposing the strict paired-clearance objective. Compare equal training budgets
and seeds, including falls, velocity tracking and zero-command stepping. This
early-phase experiment has not been implemented or started.

## Paired cadence training result (200 iterations)

The user authorized the phase-2 option from the basic checkpoint. Completed
200 additional PPO iterations, 4,096 environments, 24 steps per rollout, seed 42
(19,660,800 additional transitions). Training was split at 50, 100 and 200 for
evaluation, preserving policy, optimizer and fade progress. Worlds reset between
segments. The three new reward coefficients stayed fixed; the three retired
stock rewards reached zero at 4,800 control steps. No basic-stage retraining,
bounce, kick, shared Isaac checkout edits or environment package changes occurred.

Source checkpoint:
`logs/rsl_rl/go2_conga_basic/2026-10-27_19-45-47_basic_4096_1000/model_999.pt`.
Runs under `logs/rsl_rl/go2_conga_cadence_paired/`:

- `2026-10-27_21-35-21_paired_to50/model_49.pt`
- `2026-10-27_21-38-28_paired_to100/model_99.pt`
- `2026-10-27_21-41-43_paired_to200/model_199.pt`

Each gate evaluated 16 fixed-command cases for 40 simulated seconds, including
zero commands, forward/backward, lateral motion, pure turns and moving turns.
All gates had zero falls and 32 full 20-second episode timeouts. Training's last
20 iterations averaged return 47.765 and episode length 19.988 s, with 99.51%
logged timeouts and 0.49% orientation terminations. Return increases partly
reflect the fading penalties and cannot establish cadence success.

The first eight command cases, scored against the new short-swing reference:

| Checkpoint | XY error m/s | Yaw error rad/s | Slip m/s | Missing / expected touchdowns | Extra touchdowns |
| --- | ---: | ---: | ---: | ---: | ---: |
| Basic source | 0.0526 | 0.0270 | 0.0499 | 1004 / 1152 | 1755 |
| 50 | 0.0882 | 0.0424 | 0.0662 | 576 / 1152 | 1140 |
| 100 | 0.0769 | 0.0446 | 0.0759 | 576 / 1152 | 1203 |
| 200 | 0.0572 | 0.0301 | 0.0703 | 576 / 1152 | 1146 |

Touchdowns use force-contact crossings and one-to-one matching within 80 ms,
not the previously unreliable sensor first-contact count. The source rollout
was rescored with the new schedule; its paired-baseline metrics are saved
separately. `analyze_cadence_gate.py --paired` selects this schedule explicitly.

**The cadence objective was not achieved.** At zero command, final per-foot
rates [FL, FR, RL, RR] are [0, 2, 2, 0] Hz: FR/RL repeatedly lift while FL/RR
remain planted. Four-foot support occupies 71% of the scored time, but the
policy does not alternate diagonals. Forward motion remains approximately
2 Hz on all feet. Every first-eight case misses half its scheduled touchdowns,
unchanged between 50 and 200 iterations. Thus more support and better paired
clearance do not imply correct frequency or alternation.

Final actual forward/backward velocities are +0.356/-0.303 m/s for +/-0.4
commands; lateral velocities are +0.130/-0.171 m/s for +/-0.25 commands; pure
yaw is +0.990/-0.952 rad/s for +/-1 commands. Aggregate tracking mostly recovered
after the initial regression, but leftward tracking is below basic (+0.172),
and aggregate slip increased by about 41%.

The measured cadence reward-rate components (support, pair, phase), averaged
over the first eight cases, rose from (0.0798, 0.1700, 0.2874) at iteration 50
to (0.0864, 0.2354, 0.3006) at 200, totaling 0.6225 versus 1.10 for the synthetic
reference. The policy therefore found a stable partial-credit behavior. This
does not establish that the desired gait is infeasible, or that a longer run
could never work; it does not justify advancing to bounce/kick or declaring
successful frequency matching. Full foot speed/acceleration vectors were not
logged, so no reduction in those quantities is claimed.

Recommendation: stop this run at the agreed 200-iteration gate. The next
controlled experiment should test gentle phase/contact shaping during basic
learning, as discussed, rather than automatically extending this transfer.
That experiment requires a separate agreed schedule and authorization; no
additional training has started. Preserve this checkpoint for comparison.

Artifacts: `logs/conga_paired_eval_{50,100,200}/` hold rollouts, exported policies
and metrics. `logs/conga_paired_analysis/summary.json`, `training_curves.png`
and `contacts.png` hold the combined report and plots. The existing analysis
script now includes the three paired reward terms in its reward plot.

## Concurrent GPU capacity smoke (2026-10-28)

Two fresh, disposable PPO iterations of Cadence-Paired completed at both 1,024
and 4,096 environments in nick_sillywalk. These test memory/compute capacity,
not the proposed gentle early-phase reward scheme, which is still unimplemented.
One-second nvidia-smi samples recorded total GPU memory peaks of 7,979 MiB and
7,590 MiB respectively (16,303 MiB total), and utilization peaks of 94% and 72%.
Both processes exited successfully; the 2 GiB free-memory guard never triggered.
Logs and samples are in `logs/conga_gpu_capacity_smoke/`.

The pre-existing Isaac Lab process PID 2946976 disappeared during the test;
another Isaac Lab process appeared afterward and also exited. Its completion
reason is unknown from the available project logs. No external process was
signaled or modified. The changing concurrent workload prevents treating these
two memory/utilization measurements as a controlled scaling comparison, or
claiming that the external training was unaffected. Sampled capacity supports
a 4,096-environment experiment, but does not guarantee against future allocation
spikes, contention or driver failures. No long experiment was started.

## Early shaping from scratch (authorized 2026-10-28)

The user approved a fresh `Unitree-Go2-Conga-Early` run with 1,024 environments,
1,000 total PPO iterations and 24-step rollouts (24,576,000 transitions if
completed). Seed 42; no trained checkpoint is used for initialization. This is
a quarter of the original basic run's sample budget. The existing basic and
final conga configurations remain unchanged.

Early phase matching starts at +0.05. Between completed iterations 50 and 200,
it ramps linearly to +0.25 and paired clearance ramps from zero to +0.10. During
the same window, joint-position -0.7, air-time +0.1 and air-time variance -1.0
fade to zero. There is no separate four-foot support reward. All other stock
tracking and physical penalties remain unchanged. Phase matching uses the
existing normalized 80% swing / 20% support allocation; the reference remains
120 BPM, 0.20-second diagonal swings, 0.30-second support gaps and 3 cm clearance.
Bounce and kick are disabled. Observation/action layout stays compatible.

Translation ranges begin at +/-0.1 m/s and expand over iterations 50–200 to
vx +/-0.5 and vy +/-0.25; XY norm is always capped at 0.5 m/s. New ranges apply
at subsequent command resampling. Yaw remains +/-1 rad/s, with 20% zero-command
and 15% pure-turn sampling inherited from the conga command configuration.
This combines early shaping with slower commands; it does not isolate the
causal contribution of either change.

`early_training.py` checkpoints the schedule version and global control steps.
Resuming advances from the next iteration and preserves the completed ramp.
The supervisor `scripts/run_early_experiment.py` trains segments ending at
200/500/1000, runs 16 fixed-command 40-second evaluations at each gate and scores
the short-swing reference. It pauses for review if the last-50 training episode
duration drops below 65% of the preceding gate AND evaluation fall fraction
increases by more than 15 percentage points. This is an operational regression
guard, not a learned-gait acceptance criterion. Missing cadence improvement is
reported but does not itself stop this bounded exploratory budget.

GPU usage is sampled every two seconds; less than 2 GiB free terminates only
the supervisor's current child process group to release its GPU allocations.
At the user's subsequent request, memory recovery now waits until at least
8 GiB is free continuously for 60 seconds (a dip resets the timer), then resumes
from the latest readable early-stage checkpoint. The remaining iteration count
is computed from the checkpoint's completed iteration, keeping the target at
1,000. Up to roughly 50 unsaved iterations can be replayed; simulator worlds
reset on restart. Invalid/incomplete newest saves fall back to an earlier valid
save; if no save exists, the current gate restarts from its source checkpoint
or from scratch. Evaluation retries do not repeat completed training.

The original supervisor was already running when this change was authorized.
`watch_early_memory_resume.py` is attached without interrupting it and hands
control to the revised supervisor only if the original exits for VRAM pressure
or a logged CUDA out-of-memory error. Subsequent memory interruptions are handled
inside the revised supervisor. `memory_recovery.json` records the attachment;
`status.json` will show `waiting_for_vram` with the continuous-headroom timer
when recovery is needed. Unrelated errors and gait-regression review stops do
not auto-resume. Retried TensorBoard samples are merged by iteration with the
later attempt replacing replayed data. Continuous-headroom reset, corrupt-save
fallback and checkpoint iteration bounds were checked with mocked monitoring
and temporary checkpoints, without intentionally exhausting GPU memory.

This cannot guarantee no resource contention or catch instantaneous allocation
spikes. The other training process is not signaled. The task uses nick_sillywalk
only and does not modify shared Isaac Lab source or packages.

Verification before launch: 16 adapter/config/schedule tests passed, including
ramp boundaries, evaluation freeze and checkpoint resume. A disposable two-PPO-
iteration integration run and 220-step evaluation/export/reward audit completed.
These smoke checkpoints are excluded from the fresh run and establish no gait
quality. Live state, GPU samples and per-gate logs are under
`logs/conga_early_1024_experiment/`; `status.json` records whether training is
running, evaluating, stopped, paused for review or complete. No completion or
learning-success claim should be made without checking that state and results.

### Early-shaping run completed

The supervised 1,024-environment run completed all 1,000 iterations (24,576,000
transitions) without invoking VRAM recovery. All three fixed-command evaluations
had zero falls and 32 full-length timeouts. Last-50 training episode durations
at 200/500/1000 were 18.78/17.79/19.21 seconds. Final last-20 return was 33.69;
this is not directly comparable to runs with different reward schedules.

At 200, the policy was essentially planted. At 500, forward and lateral commands
were still mostly planted, but backward motion and turns had emerged. By 1,000,
forward/backward motion reached +0.445/-0.426 m/s for +/-0.4 commands and yaw
reached +0.974/-1.018 rad/s for +/-1 commands. Lateral motion remained weak:
+0.030/-0.104 m/s for +/-0.25 commands. First-eight-case XY errors decreased
0.215 → 0.202 → 0.093 m/s; final mean slip was 0.0555 m/s.

The cadence objective remains unmet. At zero command, FL/RR repeat at 2 Hz
while FR/RL stay planted (the opposite diagonal from the earlier transfer run).
Forward steps are approximately 2 Hz per foot. Final timing diagnostics miss
572 of 1,152 target touchdowns and count 1,246 excess crossings. Thus earlier
gentle shaping did not resolve diagonal alternation in this run. This has only
one quarter of the original basic run's samples and one seed; it does not isolate
early shaping from slower commands or establish that early shaping cannot work.

Final checkpoint:
`logs/rsl_rl/go2_conga_early/2026-10-28_00-56-55_early_1024_to1000/model_999.pt`.
Results and curves are in `logs/conga_early_1024_experiment/analysis/`; the live
status is `complete`. No further training or progression to bounce/kick has
been started or authorized.

### Offline alternating-contact bonus scoring

At the user's request, `scripts/score_alternation.py` scored a sequence bonus
without changing training rewards or running simulation. A valid beat requires
exactly one scheduled diagonal swing, both swing feet at least 1 cm clear for
two consecutive 20 ms samples, the opposite pair supporting, and at least two
consecutive all-four-contact samples before and after the swing. Swing boundaries
must lie within the 0.12–0.32 s window with 40 ms tolerance. Other sustained
missing contacts invalidate the beat; a single noisy sample is tolerated.
These are sampled-contact criteria, not substep timing guarantees.

Two consecutive valid beats earn one event; their clock targets necessarily
alternate diagonals. There is at most one event per beat, valued at 0.5, for
an asymptotic normalized reference rate of 1 per second. A candidate coefficient
of +0.10 gives at most +0.10 per second on a sustained reference. This is an
offline coefficient for comparison with the early task's +0.10 paired reward,
not an installed or trained coefficient. Missing beats and episode resets break
the sequence. No state crosses episode boundaries; warmup is 2 s.

Synthetic perfect, 20 ms early and 20 ms late references each achieve 35/35
eligible alternations in 18 scored seconds (+0.0972 per second at weight .10;
the finite segment needs an initial beat). Standing, either repeated diagonal,
wrong phase, five doubled-frequency offsets, single-foot steps, low clearance,
flight, one-sample chatter and absent four-foot support all earn zero. Reset
isolation and a finite per-beat bonus budget are checked by script assertions.

All six saved evaluations score **zero completed alternations out of 1,120
eligible transitions each** (16 cases, two episodes per case): basic source,
old final Cadence-Start, paired transfer at 200, and early shaping at 200/500/1000.
The paired-transfer rollout has 378 valid individual beats and final early
shaping has 249, but none occur consecutively with the required alternation.
Both final policies have 36 valid individual beats in the zero-command case;
they repeatedly use one diagonal, so the other 36 prevent sequence completion.
Lowering clearance to 5 mm or raising it to 15 mm, and requiring 40 or 60 ms
of consecutive samples, still gives zero bonuses for every saved rollout.

This distinguishes the intended sequence from the tested failure modes, but
reveals a sparse exploration problem. It does not establish physical feasibility
or that PPO can learn from this reward. Revise the earlier suggestion of outright
replacing paired clearance: initially retain dense phase/local stepping feedback
and consider the sequence term as an additional completion bonus after further
review. Do not assume increasing a zero-on-policy bonus weight supplies useful
feedback. No reward/configuration changes or training were made for this audit.
Results: `logs/conga_alternation_scoring/scores.json` and `summary.txt`.

### Last-liftoff history gate: offline rescore

The user's simpler proposal is audited in `scripts/score_liftoff_history.py`.
It multiplies the existing normalized paired-clearance score by a per-swing
eligibility flag: the last detected diagonal liftoff must have been the opposite
pair. The first event after reset seeds history without earning credit. Every
detected event updates history regardless of phase or clearance reward, and
eligibility stays latched for the swing. Each pair rearms after both feet contact
for two consecutive samples. Both takeoffs must occur within 40 ms of each
other, with two consecutive airborne samples. Ambiguous flight does not seed
either pair. This detector uses saved 50 Hz force contacts, not sensor
first-contact events, and cannot resolve substep transitions.

Unlike the strict previous audit, this gate does not require the previous swing
to satisfy phase/height targets, a full four-foot support interval, or consecutive
valid beat windows. Current-swing clearance, opposite-pair support and phase
weighting still come from the existing paired reward. Its reference normalizer
is unchanged: the gate only removes credit, with no compensating rescaling.

Synthetic mean normalized paired scores (before any coefficient):

| Pattern | Existing paired score | With history gate |
| --- | ---: | ---: |
| Correct reference | 1.000 | 0.967 |
| Either diagonal repeated | 0.500 | 0 |
| One-sample contact chatter | 0.500 | 0 |
| Standing, flight, wrong-phase reference | 0 | 0 |
| Five tested doubled-frequency offsets | 0.001–0.226 | 0.001–0.223 |

The small reference reduction is the causal debounce delay. Faster alternating
steps still qualify for history; existing phase/clearance timing discounts them
but does not eliminate all credit. The gate is an alternation mechanism, not
a standalone frequency constraint. Tests verify A,A,B,A eligibility as
false,false,true,true and verify resets erase history.

On the latest early-shaping checkpoint, across all 16 fixed-command cases,
mean normalized pair score falls from 0.1977 to 0.0441; nonzero-score timesteps
fall from 12.68% to 2.08%. This retains about 22.3% of the old paired credit,
versus zero sequence bonuses in the strict audit. At coefficient .10 these
means contribute 0.01977 versus 0.00441 reward per second. In the first zero-
command case, score 0.2169 becomes zero; forward 0.1800 becomes 0.1777 and
backward 0.3472 becomes 0.3402. Thus real alternating walking retains feedback
while the observed single-diagonal marching loses its bonus.

Across the six saved evaluations (basic, old final cadence, paired transfer 200,
early 200/500/1000), new mean normalized scores are respectively
0.01014, 0.07770, 0.12330, 0, 0, 0.04410. Stricter detection reduces feedback:
on the latest rollout, 40–60 ms debounce and 20–60 ms allowed separation yield
mean scores 0.01357–0.04410, with nonzero feedback on 0.62–2.08% of timesteps.

This candidate is less sparse than the strict sequence bonus and specifically
removes the repeated-pair incentive. It is worth considering as the paired-
reward modification, while retaining dense phase matching; these offline scores
do not establish learnability or correct cadence. No runtime reward was modified
and no training was started. Full results are in
`logs/conga_liftoff_history_scoring/scores.json`.

### Authorized history-gated fine-tune

The user approved 200 additional PPO iterations with 1,024 environments and
24-step rollouts, starting from early-shaping `model_999.pt`. The separate
`Unitree-Go2-Conga-History` task changes only paired reward eligibility using
the audited last-opposite-liftoff gate. Pair weight stays +0.10, phase stays
+0.25, the three faded stock terms stay zero, and command ranges, reference,
physical penalties and policy observation/action layout remain unchanged.
The task starts with the final settings even before checkpoint loading; no
curriculum is restarted. Policy and optimizer resume from the source checkpoint.

Runtime history is vectorized per environment and resets independently with
each episode. First liftoff seeds history, every detected pair updates it,
and eligibility remains latched during the swing. Both feet must lift within
40 ms, with two consecutive airborne samples and two contact samples before
rearming. The causal detector matches the offline audit. History is internal
reward state; no new policy inputs are added. Simulation worlds/history restart
at segment boundaries, as in previous segmented experiments.

Launch with `scripts/run_early_experiment.py --history-finetune`. Evaluation
gates are 50/100/200 additional iterations (absolute completed counts
1050/1100/1200), with 16 fixed-command 40-second rollouts. Checkpoints now save
every 10 iterations for this separate task, limiting memory-stop replay to
roughly 10 unsaved iterations. The existing supervisor guards VRAM below
2 GiB and waits for >=8 GiB free continuously for 60 seconds before resuming
the latest valid save toward the same absolute iteration target. Only its own
child processes are signaled. Stability deterioration pauses for review; no
bounce/kick advancement or budget extension is authorized.

Seventeen config/adapter tests passed, including offline/runtime gate agreement,
per-environment reset handling and checkpoint schedule checks. A disposable
two-iteration resume smoke passed; it is excluded from the authorized fine-tune
budget and source. Live status, GPU samples, logs and evaluations are under
`logs/conga_history_1024_experiment/`. Inspect status before claiming completion.

### History-gated fine-tune result

The authorized 200 additional iterations completed (absolute 1000–1199), adding
4,915,200 transitions. No VRAM recovery was needed; sampled free memory stayed
above 8,096 MiB. All three 16-case fixed-command evaluations had zero falls and
32 full-length timeouts. Final last-50 training episodes averaged 19.42 seconds;
training with exploration/randomization still had about 6.8% logged orientation
terminations over the final 20 iterations, so controlled evaluation success
does not establish robust stability.

The history gate worked as intended but did not teach alternating marching:
the final zero-command case repeats FL/RR at 2 Hz, with FR/RL planted. Its
measured paired reward is exactly zero after warmup, while phase reward remains
about 0.1472 per second. Forward movement remains around 2 Hz per foot. The
policy can therefore retain this behavior despite losing the paired bonus;
removing its bonus was insufficient in this finite experiment.

Across the first eight command cases, XY error changed from source 0.0931 to
0.0795 m/s and yaw error from 0.0620 to 0.0512 rad/s. Slip increased from
0.0555 to 0.0653 m/s (about 18%). Left/right velocities improved from
+0.030/-0.104 to +0.083/-0.195 m/s against +/-0.25 commands. Forward velocity
is +0.468 m/s against +0.4, so it overshoots. Final timing still misses
576/1152 scheduled touchdowns, with 1282 extras. The cadence/alternation
objective remains unmet; do not advance to bounce/kick or extend automatically.

Final checkpoint is recorded in `logs/conga_history_1024_experiment/status.json`
and ends in `model_1199.pt`. Curves and metrics are under that experiment's
`analysis/` folder. No further training has started.

### Offline explicit-pause reward scoring

The user retained the 0.20-second swing and requested scoring before further
reward implementation/training. `scripts/score_pause_rewards.py` keeps phase
+0.25 and history-gated paired clearance +0.10, and tests support coefficients
0/.05/.10 against missing-contact penalty magnitudes .10/.25/.50. The external
clock still supplies a .30-second pause between scheduled diagonal swings.
Pause weights ramp over the first/last .02 s of that interval and are zero
inside the .12–.32 swing window. Four-foot support and fraction of feet missing
contact are normalized by the fixed reference mean pause weight. No new event
counter or frequency-dependent bonus is introduced. Existing physical/velocity
terms are unchanged; these are offline candidates, not configured rewards.

At extra support +.05 and pause penalty -.25, synthetic cadence-only reference
score is .3967, standing .1250, repeated diagonal .2125, and the strongest of
five tested double-frequency offsets .1355. Extra support raises standing and
repeated-pair scores by the same .05 as the correct reference; for several
double-frequency patterns it cancels the penalty exactly. Therefore adding
another support bonus is not recommended from this audit; existing phase matching
already contains support credit.

With **no extra support bonus**, candidate pause penalty -.25 yields:

| Synthetic pattern | Existing cadence-only mean rate | With pause penalty |
| --- | ---: | ---: |
| Correct reference | .3467 | .3467 |
| Standing | .0750 | .0750 |
| Repeated diagonal on every beat | .1625 | .1625 |
| Best of five tested doubled-frequency offsets | .1355 | .0998 |

The best tested doubled-frequency score is .1212 at penalty -.10 and .0641 at
-.50. The reference incurs no pause penalty; +/-20 ms shifted references also
avoid pause penalties under this contact model. These coefficients are scored
engineering candidates, not demonstrated PPO optima.

On the latest history-gated rollout, the -.25-only addition reduces forward
cadence-only mean rate from .1630 to .1081. It subtracts about .0549 reward/s
from the audited full forward reward rate (approximately 2.2281 to 2.1732).
The zero-command repeated pair is unaffected: cadence-only .1472 remains .1472,
because it already keeps four feet down during the scheduled pause. Leftward
repeated-pair behavior is similarly unaffected. This candidate targets premature
steps during pauses, not missing the opposite diagonal during swing windows.

Full-rate estimates add the hypothetical pause term to logged rewards using
post-step contact snapshots; alignment is limited to one 20 ms control tick.
Synthetic values omit velocity and physical costs and establish only reward
preference, not feasibility or learned behavior. The audit supports a pause
penalty without another support bonus, but it does not establish a complete fix
for stationary alternation. No runtime reward changes or training occurred.
Results: `logs/conga_pause_scoring/scores.json`.

### Offline repeated-pair penalty combined with pause cost

`scripts/score_repeat_penalty.py` audits the user-requested repeated-step cost,
retaining phase +.25, history-gated clearance +.10, and the proposed pause
penalty -.25 without an extra support bonus. Debounced paired events use the
same offline detector. A repeated event is one whose diagonal equals the last
detected diagonal; first events after episode resets are exempt. History updates
on every detected event, whether rewarded or penalized. Reset/first-event
exemption and repeated-event rates are asserted in the script.

The grid is **.01/.025/.05/.10 integrated reward cost per repeated liftoff**, not
continuous Isaac reward-manager weights. At 50 Hz, an implementation using a
one-tick event signal would need weight -cost/.02 (e.g. -2.5 for cost .05), because
the manager multiplies reward rates by step_dt. No such runtime term is installed.

At event cost .05, synthetic cadence-only mean rates are:

| Pattern | Pause-only | Pause plus repeat cost |
| --- | ---: | ---: |
| Correct reference | .3467 | .3467 |
| Standing | .0750 | .0750 |
| Either repeated diagonal | .1625 | .0625 |
| Best of five fast alternating offsets | .0998 | .0998 |
| Single-foot swings | .0750 | .0750 |
| Repeated diagonal with 60 ms stagger | .1077 | .1077 |
| One-sample contact chatter | .1625 | .1625 |

Synthetic repeated-pair scores at costs .01/.025/.05/.10 are respectively
.1425/.1125/.0625/-.0375. Thus .05 is the smallest tested event cost that puts
the repeated-pair pattern below standing, while correct alternation remains
well above standing in cadence-only score. This does not prove that a moving
reference remains preferable after unknown physical costs are included.

In the latest rollout, the first zero-command case has two repeated liftoffs
per second: cadence-only .1472 becomes .0472 at cost .05. Forward walking has
zero detected repeated events and stays .1081 after its pause penalty. Backward
motion has 1.47 repeated events/s and drops .1241 → .0505; moving-turn behavior
also loses some credit. Full logged-reward estimates are retained in the output,
with the same post-step/control-tick alignment limitation as the pause audit.

The adversarial cases reveal an unresolved loophole: takeoffs staggered beyond
the detector's 40 ms tolerance and one-sample chatter evade repeated-event
detection. They still receive dense phase reward even though paired-clearance
credit is zero. Therefore .05/event is a useful candidate for suppressing the
observed repeated pair, not a training-ready complete solution. Increasing its
weight cannot fix events the detector never recognizes, and stopping entirely
is another possible response. Review detector coverage/continuous wrong-contact
costs before claiming the combined reward is robust. All work was offline;
no runtime configuration changed and no training started. Results are at
`logs/conga_repeat_penalty_scoring/scores.json`.

### Updated pause/repetition rewards (implemented, untrained)

The user authorized minimizing the identified loopholes and updating rewards,
without authorizing training. The separate `Unitree-Go2-Conga-Pause` task keeps
the latest History task's fixed settings and checkpoint-compatible layout.
Reference remains 0.20 s swing / 0.30 s pause, 3 cm clearance, 120 BPM and a
0.5 m/s XY command cap. No kick/bounce or final-conga specification changes.

| Term | Updated setting |
| --- | --- |
| Phase/contact matching | +0.25, unchanged |
| Paired clearance | +0.10, gated by confirmed opposite-pair history |
| Pause missing-contact fraction | -0.25, reference-pause normalized |
| Repeated contact departure | -0.025 integrated per foot; -0.05 for both diagonal feet |
| Extra four-foot support bonus | None |

Velocity tracking and all physical costs remain unchanged; the three retired
stock terms stay zero. Pause shaping is zero inside the swing window and uses
20 ms boundary ramps as in the offline audit. `repeated_departures` returns
count/step_dt, so its manager weight -.025 gives the stated integrated cost.

The new detector accounts for each foot's sampled contact departures immediately,
without requiring simultaneous diagonal takeoff. After a foot has left contact,
subsequent departures cost .025 until the opposite diagonal completes a confirmed
swing. Confirmation requires both opposite feet at least 5 mm clear for two
consecutive 20 ms samples while the other pair supports; a pair must first have
both feet in contact for two samples before rearming. A real staggered opposite
swing may qualify: there is no 40 ms takeoff-separation loophole. A single foot,
tiny clearance wiggle or flight cannot clear the repetition record. The first
departure of each foot after reset is exempt. Pair eligibility is latched from
the previous confirmed diagonal; first confirmation seeds history without pair
credit. All state is per environment and updates once per control tick.

`scripts/score_hardened_pause.py` uses the actual runtime core on synthetic
patterns and the latest saved rollout. Cadence-only mean scores are: correct
reference .3467, standing .0750, repeated pair .0625, 60 ms staggered repetition
.0077, 200 ms staggered repetition -.0768, repeated single-foot movement .0500,
one-sample chatter -.1875, and the tested tiny opposite-foot wiggle -.0087.
The best tested fast alternating offset retains .0998; its pause violation is
penalized but its valid alternation is not a repeated step. The correct reference
receives no repetition/pause cost. These checks close the tested bypasses, not
every possible reward exploit or the risk of learning to stand.

On the latest rollout, zero-command repeated-pair cadence credit becomes .0472,
forward .1074, backward .0152. Backward paired credit is now zero because those
opposite swings do not meet the confirmation requirements consistently. This
is a meaningful stricter-feedback tradeoff to monitor during any later training.
Instantaneous contact noise can incur departure costs even without height;
that avoids the chatter loophole but is a sensor-noise sensitivity, not proof
of a physically repeated step. Full physical/velocity-return preference remains
unverified for synthetic motions.

Eighteen configuration/adapter tests passed, including independent resets,
per-foot accounting, reward-manager dt scaling and same-tick caching. The
runtime-core scorer asserts clean reference behavior and costs for all tested
stagger/single-foot/chatter/low-clearance bypasses. Scores are saved in
`logs/conga_hardened_pause_scoring/scores.json`. A 220-step, 16-environment
no-learning evaluation of model_1199 passed reward auditing and policy export,
with zero falls (`logs/conga_pause_integration_eval/`). This verifies integration,
not learning under the new rewards. No new policy was trained.

### Authorized 4,096-environment pause-reward experiment

The user authorized training with 4,096 environments and delegated the duration
needed to assess learning. A bounded 1,000-additional-iteration experiment was
selected: 24-step rollouts, seed 42, up to 98,304,000 additional transitions.
It resumes the history-gated `model_1199.pt` policy and optimizer, keeping the
new Pause rewards fixed throughout. The source's existing 1,200 iterations are
not counted as new training; absolute final iteration count is 2,200.

`scripts/run_early_experiment.py --pause-finetune` supervises evaluation after
50/100/200/500/1000 new iterations (absolute completed counts
1250/1300/1400/1700/2200). Each gate runs the established 16 fixed-command cases
for 40 seconds, with reward auditing, exports and force-contact timing analysis.
Analysis now also reports per-foot airborne intervals, both-diagonal airborne
intervals and four-foot support durations (median, p10/p90, complete-interval
count). Censored intervals and intervals crossing reset/warmup boundaries are
excluded. These are contact-derived durations at 20 ms resolution, not full
kinematic foot-trajectory durations. The source rollout was rescored with the
same added metrics for comparison.

The prior stability regression guard remains: review if last-50 training
episode duration drops below 65% of the preceding gate AND evaluation fall
fraction rises by >15 percentage points. The initial baseline uses the source's
19.42 s training episode duration and zero controlled evaluation falls. VRAM
below 2 GiB stops only this experiment's child process; >=8 GiB continuously
free for 60 seconds enables checkpoint resume toward the original target.
Checkpoints save every 10 iterations. No other training is signaled or modified.

Before launch, the source's iteration/control-step metadata, 4,096 environment
count, 98.304M transition budget and five gates were verified. The previously
completed 18 checks, adversarial runtime-core scoring and bounded simulation/
export smoke validate the reward implementation, not learned performance.
Live state, GPU samples and results are in `logs/conga_pause_4096_experiment/`.
Do not assume completion or advance to bounce/kick without reviewing results.

### Pause-reward experiment completed

All 1,000 additional iterations at 4,096 environments completed (98,304,000
transitions), ending at model_2199.pt. All five evaluation gates had zero falls
and 32 full-length timeouts. Final last-20 training episode duration averaged
19.96 s, with 99.59% logged timeouts and 0.41% orientation terminations.

Locomotion improved, but target cadence did not emerge. In the first eight
fixed-command cases, source-to-final XY error decreased .0795 → .0229 m/s,
yaw error .0512 → .0236 rad/s, and slip .0653 → .0263 m/s. Forward velocity
finished at .393 m/s against a .4 command. Forward median four-foot support
interval increased .08 → .12 s, still below the .30 s target. Simultaneous
diagonal airborne intervals remain .14 s for FL/RR and .10 s for FR/RL;
all four feet step approximately 2 Hz rather than 1 Hz.

At zero command, FL/RR still repeat at 2 Hz while FR/RL stay planted. Its
median four-foot support interval is .36 s. Logged reward auditing confirms
paired reward zero, repeat cost -.100/s, pause cost zero, and phase credit
+.1625/s: the new costs are active, but the policy keeps the repeated gait.
All gates still miss 576/1152 target touchdowns across the first eight cases;
extra crossings rise from source 1282 to final 1584. Improved tracking/slip
therefore must not be interpreted as improved timing or alternation.

The experiment is complete, not still training. No extension or bounce/kick
progression has started. Checkpoint and full results are recorded under
`logs/conga_pause_4096_experiment/`; curves are in `analysis/training_curves.png`.
Do not extend unchanged merely because locomotion metrics improved: review why
the active phase/repetition incentives still permit the persistent gait.

### March-first candidate: scoring and bounded tests only

The user requested scoring/testing a fresh march-first approach after the pause
fine-tune retained its old gait. `Unitree-Go2-Conga-March` is isolated from all
previous tasks. Translation and yaw command ranges are exactly zero, standing
command sampling is 100%, pure-turn sampling is zero, and command/terrain-level
curricula are disabled. Zero commands do not stop the clock or swing targets.
The task retains 120 BPM, alternating .20-second swing / .30-second pause,
3 cm clearance, and the same 55/70 policy/critic inputs and 12 actions. It has
no automatic progression to traveling/bounce/kick. Existing flat-terrain physics
and training randomization remain inherited; the controlled policy probe disables
randomization in the same way as previous fixed-command evaluations.

Unlike the previous mild early shaping, these terms are active at full strength
from the first step, with no stock standing/air-time warmup:

| Term | Coefficient |
| --- | ---: |
| Normalized phase/contact matching | +1.00 |
| Normalized individual scheduled-foot clearance | +0.25 |
| Normalized paired-clearance completion | +0.25 |
| Scheduled swing feet remaining in contact | -0.50 |
| Missing contacts during pause | -0.25 |
| Repeated departure | -.025 integrated per foot |
| Zero-command translation / yaw tracking | +1.50 / +0.75, unchanged |

The three retired stock terms remain zero. All other physical penalties are
unchanged. Individual clearance provides partial progress before pair completion.
Paired completion uses the existing dense pair score without requiring previous
successful alternation; repetition costs still discourage repeatedly using one
pair. The missed-swing term uses only explicitly airborne contact targets
(zero at neutral/stance targets), normalized by the reference swing opportunity.
Both missed swings and premature departures now have explicit costs. Touchdown
timing is encouraged by return-to-support targets, not a new sparse reward using
the unresolved sensor first-contact events. Horizontal foot placement stays free.

`scripts/score_hardened_pause.py --march-first` uses the actual pure runtime
reward cores, with results in `logs/conga_march_scoring/scores.json`. Synthetic
cadence-only mean reward rates are:

| Pattern | Score |
| --- | ---: |
| Correct reference | 1.500 |
| Standing | -0.200 |
| Repeated diagonal | 0.550 |
| Best of five fast alternating offsets | 0.173 |
| Single scheduled foot | -0.0375 |
| 60 ms staggered repeated pair | 0.153 |
| 200 ms staggered repeated pair | -0.226 |
| One-sample chatter | 0.300 |

Single-foot progress exceeds standing despite incomplete motion, deliberately
providing an exploratory learning signal. Repetition and chatter still retain
partial credit; this is not a proof against all exploits or a learned gait.
Standing's negative cadence score does not mean negative total return: zero
velocity tracking still rewards staying balanced without drifting. A 3x3 offline
sensitivity grid (phase .5/1/1.5, missed-swing .25/.5/1) is included. Coefficients
are engineering candidates chosen for the stronger desired-motion preference,
not paper-derived optima or a successful training sweep.

Verification: 19 configuration/adapter tests passed, including zero-command
resampling with unchanged clock, partial clearance feedback, and no cost to
neutral contact targets. A 550-step x16-environment zero-action simulation
passed with zero falls, finite observations/rewards, per-step reward-accounting
checks, independent reset checks, PPO actor construction, JIT parity and ONNX
serialization (`logs/conga_march_smoke/`). No optimizer updates occurred.

A second 550-step x16 zero-command probe of the existing model_2199 checkpoint
passed with zero falls and export/reward checks
(`logs/conga_march_existing_policy_probe/`). It still repeats FL/RR at 2 Hz;
this is expected because its weights were not trained on the new objective.
Its first-case cadence reward rate .4743 closely matches the offline .4750
estimate, confirming the implemented reward is active during actual stepping.
This checkpoint is a test fixture, not initialization of a new training run.

The candidate is available for review and a separately authorized bounded fresh
learning experiment. No fresh training, long run, or curriculum advancement has
been started. Timing/stability must be demonstrated before introducing travel.

### Authorized fresh march-first training test

The user approved running the test after reviewing the march-first score/smoke
results. A fresh 4,096-environment, 1,000-iteration PPO run is supervised with
24-step rollouts and seed 42 (up to 98,304,000 transitions). It initializes
random policy/optimizer weights, not an earlier walking checkpoint. All velocity
commands remain zero, march rewards stay fixed from the first step, and no
travel/bounce/kick advancement occurs automatically.

`scripts/run_early_experiment.py --march-first` evaluates at completed iterations
50/100/200/500/1000. Each gate uses 16 zero-command cases for 40 seconds and
reports force-contact cadence, missing/excess touchdowns, per-foot and paired
airborne intervals, four-foot support durations, drift, slip, falls and reward
rates. The previous VRAM stop/wait/resume and sustained-regression guards remain
active. Checkpoints save every 25 iterations for this task; memory interruption
can replay unsaved work while retaining the 1,000-iteration target.

`MarchRunner` preserves iteration/control-step metadata on resume without calling
the early walking curriculum, so segment resumes cannot expand commands or
change reward weights. Twenty adapter/config/resume checks passed before launch,
including rejection of checkpoints from another stage and fixed zero-command
settings after resume. Earlier numerical scoring and simulation/export checks
also passed; no stable learned march is claimed from those checks.

Live status, GPU samples, logs and evaluation artifacts are under
`logs/conga_march_4096_experiment/`. Check `status.json` before reporting completion.

The march-first experiment subsequently completed its full 1,000-iteration
budget. Final controlled evaluation had zero falls and 32 timeouts, but did not
learn alternating marching: the first case lifted FL/RR only about .22 Hz with
FR/RL planted, missing 128/144 target touchdowns. Final last-50 training episode
duration averaged 19.92 seconds. The checkpoint is model_999.pt in
`logs/rsl_rl/go2_conga_march/2026-10-28_12-14-01_march_4096_to1000_retry1793204038167507147/`.
A requested 15-second zero-command video was recorded and verified readable at
`logs/conga_march_video_final/videos/go2-step-0.mp4`. No further training started.

### March reward symmetry and starting-diagonal audit

The user requested audits of the persistent FL/RR preference and the proposed
balanced-cycle reward before further changes. This work is read-only with
respect to training configuration/policy weights. New evaluation instrumentation
`stock_go2_check.py --phase-offset 0.5` temporarily shifts the march command
instance's reference and its policy phase observations together; physical resets
and episode timeouts are unchanged. Offset is recorded in evaluation metadata
and respected by contact timing analysis. No shared Isaac Lab/RSL source changed.

**Reset audit:** `train.py` calls RSL learn with randomized initial episode
lengths. RSL assigns each environment an integer in [0,1000) at the beginning
of each training segment, before obtaining observations. The 20-second range
contains equal phase opportunities for both diagonals (500 sampled ticks of
each beat parity). This is a one-time segment initialization, not reset-time
phase randomization. Isaac Lab's normal `_reset_idx` sets episode_length_buf
to zero, and conga references read that buffer directly. Therefore every normal
episode reset schedules FL/RR first. Current code does not alternate or randomly
choose the first diagonal on reset. This is a real initialization asymmetry,
not proof that it caused the learned preference.

**Reward audit:** `scripts/audit_march_symmetry.py` swaps feet FL↔FR and RL↔RR
and shifts the reference by half a second. It checks phase, pair, individual
clearance, missed-swing, pause and repetition components. Synthetic assertions
pass, and all 16 final recorded cases agree to <3e-14 maximum component error
after excluding invalid terminal rows. Thus these reward formulas do not
intrinsically favor FL/RR. Default joint positions are left/right mirrored and
the actuator gains are shared; this does not constitute an exhaustive URDF,
contact-physics or multi-seed symmetry proof. The phase inputs encode a 4-second
cycle plus the repeating beat, with no explicit first-diagonal flag; parity is
inferable from phase, but not separately exposed as a binary observation.

**Opposite-start policy probe:** The existing final model_999 policy was evaluated
for 2,000 steps in 16 zero-command environments with a .5-second reference/
observation offset. Both baseline and offset tests had zero falls and 32 full
timeouts. Mean foot rates [FL,FR,RL,RR] were baseline [.222,0,0,.222] Hz and
offset [.278,0,0,.278] Hz. It did not switch anatomical diagonals when FR/RL was
scheduled first. It missed 1984/2304 targets versus baseline 2048/2304. Merely
changing inference start phase does not repair the learned preference; varying
starts during learning remains an unperformed experiment.

**Balanced-credit scoring:** Candidate restores history eligibility on the .25
paired term, retains individual clearance, and adds a full-cycle bonus of
.25*min(A,B), where A/B are normalized paired quality averaged over each half
of a complete one-second cycle. A real implementation would pay the bonus only
after completing that cycle; reported values below are average reward rates.
No cycle bridges resets or incomplete recordings.

| Synthetic B opportunities correctly executed | Current march score | Candidate score |
| --- | ---: | ---: |
| 0%, repeating A instead | .5500 | .4250 |
| 25% | .7875 | .7542 |
| 50% | 1.0250 | 1.0834 |
| 100% | 1.5000 | 1.7418 |

The reference-over-repetition gap grows .9500→1.3168 (about 39%). Developing B
from 0% to 25% improves candidate score by .3292 instead of .2375. Standing stays
at -.2000. A-only on its own scheduled opportunities scores .6000 currently,
.4750 with the candidate; the balanced bonus is zero. In every actual final
rollout, the cycle bonus is also zero. The first case's cadence-only rate drops
-.02475→-.05103 because its existing pair credit is removed. This confirms a
stronger counterfactual preference for developing B, but also the sparse-feedback
tradeoff; it is not evidence of better PPO learning or full physical return.

Recommendation for a clean next test: balance which diagonal starts on **every
reset**, using a persistent per-episode 0/.5-second phase offset shared by all
reference consumers and observations, without shortening physical episodes or
changing clock frequency. Keep physical starting poses identical. Test this
from scratch against the fixed-start baseline; treat balanced-cycle credit as a
separate candidate rather than assume stacking changes will solve exploration.
Neither reset randomization nor balanced reward was installed, and no new
training was started. Audit artifacts are in `logs/conga_march_phase_audit/`
(`reward_audit.json` and `opposite_start/`).

## Evaluation and acceptance

### Temporary walking-only validation experiment

Two authorized 200-iteration runs are complete. **Both failed walking validation.**
The [baseline](go2_conga_walk_validation.md) stayed planted; the
[reward-v2 repeat](go2_conga_walk_rewards_v2.md) terminated after 0.36–0.38 s.
The current temporary configuration contains the rejected v2 coefficients for
review/reproduction, not a recommended training setup. Do not introduce kick.

`Unitree-Go2-Conga-Walk-Validation` is a separate experimental registration,
not a change to the final conga motion or the existing curriculum stages.
It inherits the reduced-range cadence stage: continuous 120 BPM diagonal steps,
zero-command marching, XY velocity tracking, and pure/combined yaw commands.
Kick is disabled, kick and bounce scales are zero, and kick, body-height and
reference vertical-velocity rewards are removed. The baseline retained the
existing walking reward weights. Revision v2 changes contact, touchdown timing,
swing clearance, marching, and tracking rewards, adds missed-swing penalties
and a constant-height posture reward, and retains the physical penalties.
Observation/action dimensions remain 55/70/12; details are in the v2 report.

The user authorized this bounded experiment on 2026-10-27:

```bash
conda activate nick_sillywalk
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Conga-Walk-Validation \
  --num_envs 256 --max_iterations 200 --seed 42 --run_name walk_only_200
```

Its separate experiment directory is `logs/rsl_rl/go2_conga_walk_validation`.
For evaluation, `scripts/conga_check.py --rsl-checkpoint <model_N.pt>` loads the
training checkpoint and exports it under `trained_export/` without learning.
Add `--walking-commands` to evaluate zero, forward, backward, lateral in both
directions, yaw in both directions, and a moving turn within the training ranges.
`scripts/analyze_conga_walk.py` extracts TensorBoard curves and fixed-command
CSV metrics, including one-to-one touchdown matching within 0.08 seconds,
missing/extra events, and per-foot cadence. The walking reference is 1 Hz per
foot (2 diagonal swings per second). Exclude each episode's first 2 seconds
from steady-state metrics, and do not match touchdowns across resets.

### Acceptance measurements

Training logs `Metrics/base_velocity/*` episode means, including velocity error,
travel-only error, mean commanded speed, contact error, actual/reference touchdown
rates, touchdown timing error, kick lateral/height error, bounce error, slip and
tilt. Standard termination logs distinguish falls from time limits. Diagnostic
accumulation skips reset states; the terminal sample is omitted because Isaac
Lab resets before updating command metrics. The reward still uses the terminal
pre-reset state. Metrics with no applicable events are zero, not proof of success.

`conga_check.py` writes per-environment, per-step `rollout.csv`, including actual
and reference contacts, actual touchdown flags and distance to reference events,
command speed/yaw, phase envelope, tracking errors, slipping, and falls. Terminal
rows retain fall flags and leave reset physical measurements blank.

For a trained policy, compare zero, 0.4, 1.25 m/s, backward, lateral, moving turns,
and pure-yaw cases (the eight evaluation environments cover these). Group by
command speed and yaw, align contact crossings by episode time, and match actual
touchdowns to expected crossings within an initial 0.08 s tolerance. Count missing
and extra events outside the unconstrained recovery region as well as timing error: a low timing error on a few events is
insufficient. Measure observed touchdown count per foot per simulated second;
the full conga reference averages 0.875 Hz per foot (three diagonal steps plus
one front kick per phrase), while the initial continuous-trot stage averages
1 Hz per foot. During traveling beats, diagonal step opportunities remain 2 Hz
at every speed. Recovery may add optional contacts, so separate those from the
mandatory scheduled events when comparing measured rates across speed bins.

Inspect kick errors only during kick windows, and inspect both legs separately.
Measure bounce peak-to-peak amplitude and 2 Hz phase alignment, not just mean
body-height error. Report velocity error during travel separately from the kick
pause; also check transitions, falls per episode, body tilt, and stance-foot slip.
Verify sustained motion visually, then repeat with several seeds, stage-5
randomization, and sim-to-sim testing. This initial implementation does not
establish that a Go2 can safely execute the complete requested motion.

### Balanced starting-diagonal march experiment

The user authorized a fresh comparison using `Unitree-Go2-Conga-March-Balanced`:
4,096 environments, 1,000 PPO iterations, 24-step rollouts (98,304,000 samples),
with evaluations at 50/100/200/500/1000. It uses the previous march task's exact
rewards, zero commands, 0.20-second swings and physical initialization. No
balanced-cycle bonus is installed. Final dance stages are unchanged.

`BalancedMarchCommand` initializes half the environments with each 0/0.5-second
phase offset using a private seeded shuffle, then toggles each environment's
starting diagonal on every reset. This balances successive episodes per
environment, including asynchronous resets; the instantaneous population need
not remain exactly half after asynchronous resets. All reference consumers and
phase observations use that offset. Command resampling leaves it unchanged,
and physical episode age and timeout length are unchanged. Evaluation CSVs log
per-row offsets so timing analysis handles resets correctly. Checkpoints mark
the phase mode to prevent accidental fixed/balanced training resume.

Configuration checks and 21 adapter tests passed, including exact reward
configuration equality, offset persistence and selective reset behavior.
The supervisor command is `scripts/run_early_experiment.py --march-balanced`;
state lives in `logs/conga_march_balanced_4096_experiment/status.json`.
Existing VRAM stop/recovery and deterioration gates apply. This is a fresh
learning experiment, not evidence that balanced starts resolve the learned bias.

Before launch, the 550-step × 16-environment simulation smoke passed with zero
falls, correct asynchronous reset phase, PPO construction, JIT parity and ONNX
serialization. It used zero actions and establishes no learned gait quality.
The authorized supervised run has been launched; consult its live state for
progress rather than assuming completion.

### Balanced-start march final results

The authorized run completed all 1,000 iterations (98.304M samples). Final
checkpoint: `logs/rsl_rl/go2_conga_march_balanced/2026-10-28_13-12-27_march_balanced_4096_to1000_retry1793207544575686666/model_999.pt`.
All five deterministic evaluation gates had zero falls and 32 full timeouts,
but all four feet remained planted in steady-state measurements at every gate.
The final 16-case evaluation missed all 2,304 scheduled touchdowns; measured
per-foot rates were [0,0,0,0] Hz and four-foot support fraction was 1.0.
Mean XY drift was 0.00456 m/s and slip 0.00202 m/s. In comparison, the preceding
fixed-start march run missed 2,048/2,304 targets and achieved only FL/RR steps
at about .222 Hz. Neither run achieved alternating marching.

Over the final 50 training iterations, mean episode duration was 19.901 s and
mean return 36.811. Logged reward rates were phase +.29471, paired completion
~1.8e-8, individual clearance +.000163, missed swing -.49560, translation
tracking +1.46287 and yaw tracking +.72011. Stable standing earns substantial
zero-velocity tracking reward while paying the missed-swing cost; the return
and episode length therefore do not establish progress on cadence. Training
exploration metrics must not be confused with deterministic contact crossings.

This seed did not benefit from reset balancing and exhibited less stepping than
the fixed-start baseline. A single comparison does not prove that balanced
starts generally hurt learning. The result supports reviewing how exploratory
foot lifts earn feedback and how movement costs interact with the march rewards
before extending training. The scored balanced-cycle bonus remains uninstalled:
it would give this planted policy no positive bonus. No new reward change,
training extension, or advancement is authorized. Curves and evaluation artifacts
are under `logs/conga_march_balanced_4096_experiment/analysis/` and `eval_1000/`.

Requested final-checkpoint video: `logs/conga_march_balanced_video_final/videos/go2-step-0.mp4`
(14.98 s, 50 fps, 960x720, zero commands, no learning). Sampled frames at
2/6/10/14 seconds show the same planted stance, consistent with force-contact
measurements. Recording completed without falls. This video is shorter than a
full episode; the separate 40-second evaluations cover episode resets.

## Gentle-duration basic walking experiment (authorized 2026-10-28)

`Unitree-Go2-Conga-Duration` starts fresh with 4,096 environments, 1,000 iterations,
24-step rollouts, and gates at 50/100/200/500/1000. It inherits stock basic physics,
posture and physical costs, tracking +1.5/+0.75, and stock command curriculum from
+/-0.1 m/s to vx +/-0.5, vy +/-0.25, with a hard 0.5 m/s XY vector cap. Yaw is
+/-1 rad/s. Stock air-time bonus and variance weights are zero. Two new fixed
weights are -.05 each for air and contact duration. No explicit phase/contact,
clearance, paired-step, repeat or missed-swing reward is present.

Duration costs use force contacts >1 N, averaged across feet. Each is bounded
[0,1], giving at most -.10/s combined. Error is zero inside [.16,.24] s airborne
and [.70,.90] s grounded; outside, cost is 1-exp(-(distance/scale)^2), with scales
.20 and .80 s respectively. Completed costs are held until the next completed
interval of the same kind and combined by maximum with ongoing overrun costs.
Thus event frequency does not directly multiply cost, and never lifting cannot
avoid ongoing stance cost. Initial partial intervals are excluded from completed
costs; overrun costs still apply. Neutral or reset clears history. Commands with
XY norm <=.05 m/s AND |yaw|<=.05 rad/s disable both penalties. This is a simulation
activation threshold, not a validated joystick hysteresis/settling controller.
Phase observation restarts on activation, freezes at zero at neutral, and
continues across active command changes; physical episode age is unaffected.
Deployment transitions remain unimplemented. No reward enforces absolute phase.

Exact-core synthetic scoring: target 0/s, fast .5-second cycle -.01128/s,
.4-second swing -.02562/s, active standing -.04995/s, neutral standing zero.
Saved basic walking contact replays were also scored (counterfactually active,
including recorded zero-command cases): typical .4 m/s forward cost -.00983/s.
These scores do not predict PPO learning. Results: `logs/conga_duration_scoring/`.
22 adapter tests passed; fresh two-iteration PPO, two-iteration checkpoint resume,
and 220-step evaluation/reward-accounting/export smoke passed. Checkpoint metadata
preserves command ranges and step count. The production run is fresh, not resumed
from these smoke checkpoints.

Supervisor: `scripts/run_early_experiment.py --duration-walking`; live state:
`logs/conga_duration_4096_experiment/status.json`. VRAM recovery and deterioration
gates remain active. No automatic stage advancement or extra iterations. Analyze
walking tracking, per-foot swing/stance distributions, diagonal coordination,
falls and slip. Legacy phase-match diagnostics are descriptive only: this task
does not enforce clock locking and zero-command steps are not required.

During monitoring, a resume defect was found: command managers copy their config,
so the first DurationRunner implementation saved initial env.cfg ranges rather
than live command-term ranges. Earlier evaluation-gate resumes therefore restarted
command expansion at +/-0.1; the 500 checkpoint had reached +/-0.5 before its
resume. The run was paused at about 600 iterations, and the latest model_600
metadata was repaired from its matching logged curriculum value (vx +/-0.3,
vy +/-0.25). Original checkpoint backup and repair record are retained under
`logs/conga_duration_4096_experiment/range_repair.json`. Saving/loading now uses
command_manager.get_term('base_velocity').cfg.ranges. Resume targets the original
1000 iterations, not an extended run. Earlier resets are a comparison limitation;
they are not retroactively corrected or concealed. Reward weights are unchanged.


### Gentle-duration walking: completed results

Completed all 1000 iterations, with final checkpoint
`logs/rsl_rl/go2_conga_duration/2026-10-28_13-53-41_duration_4096_to1000_retry1793210018592529433/model_999.pt`.
Final evaluation: zero falls, 32 full timeouts across 16 cases. Neutral stands.
Forward .4 m/s command tracks .375; backward -.4 tracks -.348; lateral +/- .25
tracks +.196/-.191; yaw +/-1 rad/s tracks +.977/-.986. Mixed [.3,.15,.5] command
tracks [.292,.134,.490]. These are controlled simulation measurements, not
hardware or robustness validation. Walking appeared by the 500-iteration gate.

Forward feet remain at 2 Hz, despite desired 1 Hz per foot. Median swing durations
in FL/FR/RL/RR order are [.12,.12,.16,.18] s; stance durations are
[.38,.38,.34,.32] s. Four-foot pause median .08 s versus target .30 s.
Two-contact support was diagonal whenever observed in the forward case. Rear
swings are inside the accepted air-time band; front swings are short and no
completed forward stance intervals fall in the .70-.90 s band. Neutral has no
completed swing intervals, as now intended. Final forward duration costs are
-.00121/s air and -.00852/s stance, versus +1.49077 XY and +.74671 yaw tracking.
These weights permitted useful walking but have not established desired timing.

Last-50 training mean episode duration: 19.936 s. Final-20 mean return: 32.833;
mean duration costs -.00628/-.00867 per second. Evaluation falls at gates
50/100/200/500/1000: 854/0/5/0/0. Keep the documented command-range resume defect
in mind when comparing learning speed or declaring reproduction. Its fix passed
23 adapter checks and live checkpoint metadata verification. No VRAM failure
occurred; one corrective pause resumed within the same 1000-iteration target.
The supervisor finished all analyses; no extra training or advancement started.
See `logs/conga_duration_4096_experiment/analysis/training_curves.png`,
`analysis/summary.json`, and `eval_1000/cadence_metrics.json` for full results.

### Stronger duration experiment

User authorized a new run with harsher penalties. `Unitree-Go2-Conga-Duration-Strong`
uses -.20 air and -.20 stance, exactly four times the preceding coefficients;
all other task settings match Duration. It starts fresh with 4096 environments,
1000 iterations and 24-step rollouts; gates 50/100/200/500/1000. Corrected
live command-range checkpoint persistence applies from launch. The baseline's
older resume defect remains a comparison limitation. No automatic advancement.

Scores using identical saved contacts: final baseline forward duration cost
-.009729/s becomes -.038915/s, and total physical+task rate 1.92406 becomes
1.89488/s. Synthetic target cost remains zero; fast timing -.04512/s,
long swing -.10248/s, active standing approximately -.19981/s, neutral zero.
Maximum combined duration cost is -.40/s. These counterfactual scores are not
predictions of PPO behavior. See `logs/conga_duration_strong_scoring/scores.json`.
24 adapter tests and fresh two-iteration PPO smoke passed. Separate task/runner
metadata preserves the gentle task and prevents accidentally cross-resuming.
Supervisor mode `--duration-strong`; live state and evaluations are under
`logs/conga_duration_strong_4096_experiment/`. Training was launched with unchanged
VRAM recovery and deterioration checks. Check live state before claiming results.


### Stronger duration experiment: completed results

Completed all 1000 iterations with 4096 environments and 24-step rollouts;
no memory recovery, interruptions, extensions or stage advancement. Final checkpoint:
`/home/rml2/Documents/nicholas_chung/unitree_rl_lab/logs/rsl_rl/go2_conga_duration_strong/2026-10-28_14-47-07_duration_strong_4096_to1000_retry1793213223697938773/model_999.pt`.
All results and curves: `logs/conga_duration_strong_4096_experiment/analysis/`.
Final controlled evaluation had zero falls and 32 timeouts. Neutral stands;
forward .4 m/s tracks .380, backward -.4 tracks -.375, lateral +/-.25 tracks
+.202/-.201, yaw +/-1 tracks +.986/-.972. Last-50 training episodes averaged
19.929 s. Evaluation falls at 50/100/200/500/1000 were 1836/0/0/0/0.

The desired slow gait remains unmet. Forward per-foot rates were approximately
[2.00,2.00,2.00,1.95] Hz. Median swing [FL,FR,RL,RR] was [.14,.12,.16,.16] s;
stance [.36,.38,.34,.32] s; four-foot pause .08 s. Compared with gentle weights,
forward speed .375 -> .380 m/s, pause .08 -> .08 s and stance essentially unchanged.
Fourfold coefficients did not produce the .20 s swing/.80 s stance/1 Hz objective.
Final forward duration costs were -.00282/s air and -.03419/s stance. Stronger
penalties did not eliminate the fast gait. This single-seed comparison also has
the documented baseline command-range resume confound; avoid causal claims that
isolate weight changes from that difference. No new training is authorized.

### Offline velocity-averaging audit

User requested empirical scoring of the hypothesis that instantaneous velocity
tracking favors early steps over longer support pauses. No runtime reward or
training changed. `scripts/score_velocity_averaging.py` replays 16 fixed-command
cases from each of the gentle, strong, stock-basic and planted final evaluations.
It recomputes the actual XY reward (weight1.5, std.5) and compares 0.5/1/2-second
rolling averages of vector velocity error BEFORE applying the exponential.
All variants score identical samples after two seconds of history, with separate
histories across resets. Logged original rewards agree within 1e-4. Files:
`logs/conga_velocity_average_scoring/scores.json` and `comparison.png`.

For the strong final forward case, instantaneous XY rate 1.49352 becomes 1.49607
with a two-second window: gain only .00255/s. Across its first eight cases,
gains are roughly .00000-.00607/s. Gentle forward gains .00473/s. Existing walks
already have small velocity ripple, so this does NOT establish that instantaneous
tracking caused the learned 2 Hz cadence or that averaging will fix it. No learned
slow-pause gait exists in these records for a direct equal-quality comparison.

Synthetic traces at mean forward .4 m/s show the mechanism: constant .4 earns
1.5/s in either scheme; sinusoidal +/- .1 earns 1.47045 instantaneous versus 1.5
averaged; +/- .3 earns 1.26307 versus 1.5. The instantaneous score is identical
at 1 and 2 Hz when the speed distribution is identical: it penalizes fluctuations,
not frequency itself. Full stop/go (0 for .3 s, 1 m/s for .2 s) earns .61672 versus
1.5; these bursts are illustrative and not physically validated trajectories.
Standing under a .4 command remains .79094 in both, and steady underspeed remains
penalized. Averaging does not reward simply going slowly at the wrong mean speed.

Loophole: .4 +/- .8 m/s oscillations, including backward excursions, also earn
1.5 under a full-cycle mean. An offline 80% averaged /20% instantaneous blend
retains some penalty (1.32124 rather than1.5 in that extreme case), but this does
not prove it eliminates exploits. One-second windows span a desired per-foot
cycle; two seconds span two. A half-second window can still penalize 1 Hz ripple
more than 2 Hz ripple and is a weaker candidate for the stated purpose.
Before any implementation, define command-transition history handling, neutral
response, and body/world-frame treatment on turns. Fixed-command body-frame
replay does not validate those transitions or net world displacement. A rolling
reward also introduces history that must be considered with the policy's
observation layout. Averaging is a plausible candidate, not an authorized change.

### Averaged XY tracking experiment (authorized)

User approved implementing the scored 80/20 blend and training, then working on
an independently timed kick. Separate task `Unitree-Go2-Conga-Duration-Average`
inherits Strong (-.20 air/-.20 stance), changing only XY tracking. It returns
.8*exp(-||mean(error)||^2/.25)+.2*exp(-||current_error||^2/.25), weight1.5;
errors are body-frame XY command minus velocity. The mean covers up to the last
2 seconds (100 control steps). Initial partial windows use their actual sample
count. All three command components changing by >1e-6 clears history, without
resetting the phase clock while already active. Neutral (XY<=.05 AND |yaw|<=.05)
uses fully instantaneous XY tracking and clears history each tick. Resets clear
only affected environments. Yaw reward remains instantaneous, weight .75.
This matches the scoring frame; it is not a net world-displacement objective.
The history is reward state, not a new policy input; 55/70/12 layout is preserved,
so the feedforward actor does not explicitly observe the full reward history.

26 adapter tests passed, including independent deque parity, selective reset,
command changes, buffer wraparound, neutral response and isolated config changes.
Fresh two-iteration PPO, two-iteration resume and 220-step evaluation/reward-sum/
export smoke passed. Neutral response and command changes are defined for this
simulation reward, not a validated deployment transition controller.

Fresh run: 4096 environments, 1000 iterations, 24-step PPO, gates at
50/100/200/500/1000; `scripts/run_early_experiment.py --duration-average`.
Live state/results: `logs/conga_duration_average_4096_experiment/`. Corrected live
command-range persistence, separate checkpoint mode, VRAM recovery and regression
checks apply. No automatic budget extension or kick training. Next work is an
isolated kick with simulation-time timing and stable support, evaluated separately
from walking cadence. Existing final kick timing/amplitudes are retained pending
review; runtime feasibility is not assumed from speed-independent references.


### Averaged tracking: completed results

All 1000 iterations/4096 environments completed with no interruptions or memory
recovery. Final checkpoint: `/home/rml2/Documents/nicholas_chung/unitree_rl_lab/logs/rsl_rl/go2_conga_duration_average/2026-10-28_23-16-04_duration_average_4096_to1000_retry1793243761040329112/model_999.pt`.
26 adapter checks and fresh/resume/evaluation smoke passed before launch.
Final evaluation: zero falls, 32 full timeouts. Neutral stands. Commands -> actual:
forward .4 -> .430 m/s, backward -.4 -> -.376, lateral +/-.25 -> +.040/-.035,
yaw +/-1 -> +.990/-.983 rad/s; mixed [.3,.15,.5] -> [.343,.125,.513].
Forward cadence remains approximately [2.00,1.95,2.00,2.00] Hz. Median forward
swing [FL,FR,RL,RR] [.16,.14,.18,.22] s, stance [.34,.36,.32,.28] s; four-foot
pause .04 s. Forward two-contact support was diagonal whenever observed.
Last-50 training episode duration 19.891 s. Last-20 return31.470 and duration
rates -.02560/-.03683. Evaluation falls at 50/100/200/500/1000:
1346/0/0/0/0. At 500, translation cases still stood; walking emerged later.

Against the instantaneous Strong final checkpoint, averaging did not reduce
forward cadence, shortened pauses .08 -> .04 s, and substantially regressed
lateral tracking (.202/-.201 -> .040/-.035). A single seed does not establish a
general causal result, but this test does not support promoting the blend.
Preserve both checkpoints; no additional training or reward tuning was started.
Full results: `logs/conga_duration_average_4096_experiment/analysis/` and
`eval_1000/cadence_metrics.json`.

User wants kick work next, with little dependence on gait speed. Existing final
kick already uses fixed simulation-time .25 s extension/retraction independent
of speed. The next proposed isolation is a single front-foot kick with three-foot
support and free fore-aft placement, beginning from stable stance, retaining the
specified .08 m lateral/.05 m clearance endpoint and checking touchdown/support.
This avoids requiring a successful 1 Hz gait before testing the kick itself.
A separate kick request/activation must be distinguished from neutral joystick
standing. Existing kick-stage configs also impose the full dance reference, so
simply resuming that stage is not an isolated kick experiment. No kick training
budget or new kick runtime behavior has been agreed or launched yet.

## Isolated front-kick test (user requested)

Separate `Unitree-Go2-Conga-Kick-Test` explicitly requests alternating FR/FL kicks
from stance on the existing 4-second clock, at 1.25-1.50 /3.25-3.50 s. It is a
periodic request test harness, not automatic kicking from a neutral joystick.
No traveling diagonal swings, recovery contact relaxation or bounce. Exactly one
front foot has a soft swing contact target; the other three target support.
Full target remains .08 m outward relative to nominal body-frame y and .05 m
sole clearance over .25 s, sin-squared pulse. Fore-aft placement remains free.
Translation/yaw targets are zero. Pure reference generation has no speed input.
The original final dance reference and existing configs are unchanged.

Candidate rewards retain existing contact1/kick1/height.75 and inherited
tracking/physical costs, removing cadence, clearance and marching terms. These
are evaluation candidates, not reviewed/validated kick learning coefficients.
Synthetic exact-core contact+kick average rates: target1.10493, half1.05036,
quarter1.03313, planted1.00139. The ideal-over-standing advantage is only .10355/s
averaged over a full cycle, excluding physical costs. This may be weak relative
to movement costs; do not infer trainability from the synthetic ordering.
Both-leg target endpoint/peak zero-velocity checks passed. 28 adapter/config tests
passed, including isolated support and speed independence. A zero-action
550-step x8 reset/PPO/JIT/ONNX smoke completed with zero falls, without learning.

A 2000-step x8 probe loaded the stronger instantaneous-duration walking model_999
from `logs/rsl_rl/go2_conga_duration_strong/2026-10-28_14-47-07_duration_strong_4096_to1000_retry1793213223697938773/model_999.pt`.
It performed no learning and had zero falls. Excluding the first two seconds of
each episode, 144 kick windows were scored (80 FL/64 FR):

| Measurement | FL | FR |
| --- | ---: | ---: |
| Windows with any kick-foot liftoff | 80/80 | 0/64 |
| Force-crossing touchdown within +/-80 ms of target | 80/80 | 0/64 |
| Lateral target MAE | .06194 m | .04857 m |
| Clearance target MAE | .01647 m | .02478 m |
| Peak observed sole clearance | .02488 m | .00141 m |
| Peak outward displacement from nominal y | -.01122 m | -.00348 m |
| Fraction with all other three feet in contact | .25 | .3333 |

Neither foot reached outward extension. FL liftoff/timely touchdown alone is
not a kick success: its support and lateral path failed. This is the existing
walking policy's response to new observations/requests, not a learned-kick result
or proof of infeasibility. No kick training was run, and no additional training
budget is authorized by this bounded test. Review dense kick exploration,
support incentives and amplitude introduction before launching kick learning.
Artifacts: `logs/conga_kick_test_smoke/`, `logs/conga_kick_test_walking_probe/`,
`logs/conga_kick_isolation_scoring/scores.json`; reproduce analysis with
`scripts/score_isolated_kick.py --rollout logs/conga_kick_test_walking_probe/rollout.csv`.

## Small-kick learning: accepted rewards and authorized experiment

User approved scoring and training with 4096 environments, enough iterations to
observe progress, followed by reward review and continuation. Bounded plan:
500 additional 24-step PPO iterations from Strong walking model999, review at
absolute iteration1500, then up to500 more (absolute2000) after reviewing data.
Evaluate at +50/+100/+200/+500 and after continuation. Fixed introductory targets
are .02 m outward/.02 m sole clearance, .25 s duration; these are separate
training amplitudes and do not replace the final .08/.05 m specification.

Task `Unitree-Go2-Conga-Kick-Intro` separates height +2 (std.015 m), lateral +2
(std.020 m), requested-foot airborne +.5 (weighted toward middle of kick), and
missing support -1 (fraction of required contacts). Height/lateral are weighted
by soft kick occupancy. Other three feet always require support; requested foot
is smoothly removed from support during the kick and all four target support
outside requests. This replaces mean contact and joint height/lateral reward;
existing height +.75, zero-velocity +1.5/+.75 and physical penalties are retained.
First requested leg is privately shuffled and toggled on each environment reset
using a0/2s offset, shared with phase observations. These are explicit autonomous
kick requests for training, not neutral joystick-triggered deployment behavior.
No bounce/traveling swings, no amplitude ramp during the first block.

Exact runtime-core synthetic mean rates (kick terms only, before physical costs):
target .44190, half .38486, quarter .33141, standing .25692; target plus wrong-foot
lift .40657, standing plus wrong-foot lift .22159. Target > partial > standing,
and support loss reduces credit; this does not prove PPO learnability or exclude
all physical exploits. Script: scripts/score_kick_intro.py. Components are shared
with runtime in mdp/kick_scores.py. Diagnostic qualified small-kick window requires
at one common sample: >=1cm outward, >=1cm clearance, requested foot airborne,
and other three in contact. This is a progress metric, not final-kick acceptance.
Also report per-leg errors, support fraction, liftoff and force-based touchdown
within80ms. Partial/initial windows and reset states must not be called success.

30 adapter tests, transfer/two-update PPO, matching checkpoint counters, and
220-step x8 reset/JIT/ONNX/reward-sum simulation checks passed; zero smoke falls.
Smoke checkpoint is not the production starting checkpoint. KickRunner transfers
Strong model999 at completed1000 and saves a distinct kick_intro_v1 resume mode.
Supervisor `--kick-intro`; state/results:
`logs/conga_kick_intro_4096_experiment/`. Corrected memory recovery applies;
only owned children are stopped. First block launches to1500. Following actual
review, `--kick-intro --kick-continue --resume-state` resumes toward2000. Review
may retain coefficients if learning improves; changes require scoring against the
new data. No full-amplitude, dance-stage or unlimited training is implied.

### Small-kick first-block review and continuation (2026-10-29)

First500 additional iterations completed. All four controlled evaluation gates
had zero falls/32 timeouts. At +50/+100/+200/+500, FL qualified144/144 windows,
FR0/144. At +500, FL mean peak outward .01954 m, clearance .02100 m, lateral/
height MAE .00076/.00133 m, three-foot support1.0 and144 timely touchdowns.
FR mean peak outward .01808 m but clearance only .00136 m, lateral/height MAE
.00146/.00926 m, support1.0 and zero liftoffs/touchdowns. Outside kicks, four-foot
support1.0. Mean XY body drift .03089 m/s. FL success is only for introductory
2cm amplitudes, not the full final kick. First-block checkpoint:
`logs/rsl_rl/go2_conga_kick_intro/2026-10-29_00-15-12_kick_intro_4096_to1500_retry1793247309056305750/model_1499.pt`.

An evaluation harness failure after the first50 iterations was corrected by
propagating PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, matching the passing manual check
environment. No rollout existed from the failed invocation. The saved checkpoint
was evaluated again without repeating training. This was not a VRAM failure.

Review identified FR clearance rather than lateral tracking/support as the stall.
The user authorized reevaluation and continuation. The selected refinement is
only airborne reward: weight+.5 -> +2, multiplied by clamp(actual_clearance/.01,0,1)
in addition to phase/contact gating. This preserves smooth exploratory feedback
but prevents zero-clearance contact flicker receiving the bonus. Height/lateral
weights2/2, support-1, physical costs and .02/.02 amplitudes remain unchanged.
Shared exact-core synthetic rates: target .51030 (was .44190), half .43957,
quarter .34706, standing .25692; zero-clearance flicker .25541, tiny1mm departure
.27322. Ideal-over-standing gap grows .18498 -> .25338 (about37%). Support loss
still reduces reward. These are preferences, not predictions of learning.
Recorded +500 contacts were rescored under candidate terms in review_1500/;
FR's planted behavior earns no new lift credit. Scripts/score_kick_intro.py
--refined measures this variant; no baseline artifacts were overwritten.

31 adapter checks and two-update transfer smoke passed. Separate task
`Unitree-Go2-Conga-Kick-Refine`/checkpoint mode kick_intro_air_v2 resumes model1499
for500 more iterations (absolute2000), without amplitude progression. Supervisor:
`--kick-refine --kick-continue --resume-state`, same experiment status directory.
Final evaluation follows at2000. Initial scoring, prior checkpoints and final
geometry remain preserved. Further work beyond this continuation is not implied.


### Small-kick continuation: final results (2026-10-29)

Both authorized blocks completed: 500 introductory +500 refined kick iterations,
4096 environments,24-step rollouts (98.304M additional samples beyond the walking
checkpoint). Final checkpoint:
`/home/rml2/Documents/nicholas_chung/unitree_rl_lab/logs/rsl_rl/go2_conga_kick_intro/2026-10-29_00-28-45_kick_intro_4096_to2000_retry1793248121965858208/model_1999.pt`.
Mode kick_intro_air_v2 and control_steps48000 were verified. Final controlled
40-second x16 evaluation: zero falls,32 timeouts; last50 training episodes20.0s.
There were no VRAM recoveries. The one initial evaluation-harness interruption
was repaired from the saved checkpoint without replaying completed training.

| Final small-kick measurement | FL | FR |
| --- | ---: | ---: |
| Qualified progress windows |144/144|0/144|
| Windows with liftoff |144/144|0/144|
| Timely force-crossing touchdowns |144/144|0/144|
| Mean peak outward |.01968m|.01878m|
| Mean peak sole clearance |.02064m|.00102m|
| Lateral path MAE |.00046m|.00082m|
| Height path MAE |.00120m|.00952m|
| Other three contacts during request |1.0|1.0|

Outside requests, four-contact fraction1.0. Mean XY drift .03085m/s and stance
slip .00509m/s. The supported FL kick improved slightly in tracking accuracy;
FR clearance stayed near the ground despite improved lateral tracking. The
refinement did not resolve FR liftoff. Body-relative lateral tracking alone must
not be called a kick. The learned FL motion is at2cm introductory amplitudes;
the final8cm outward/5cm clearance kick and moving transitions remain untrained.

Video `logs/conga_kick_intro_video_final/videos/go2-step-0.mp4` is14.98s,50fps,
960x720, no learning; it contains explicit periodic requests at zero travel.
Do not interpret this as neutral joystick auto-kicking in deployment. Curves and
both-block comparison are in `logs/conga_kick_intro_4096_experiment/analysis/`
(`kick_training_curves.png`,`summary.json`). Final measurements in eval_2000/.
No extra iterations or amplitude increases were started after the agreed review
and continuation. Further review should focus on the FR height/exploration
failure and check counterfactual supported right-foot lifts before another run;
raising a conditional lift bonus alone has not been sufficient in this trial.
