# Go2 conga task

The motion contract is maintained in [AGENTS.md](../AGENTS.md). This is an initial
phase-conditioned PPO environment, not a trained or hardware-qualified gait.

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

Run from the repository root in the installed Isaac Lab environment:

```bash
conda activate isaaclab
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

### Recorded verification, 2026-10-27

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

Use the existing `scripts/rsl_rl/train.py`. All stages log under
`logs/rsl_rl/go2_conga`. Start with a small environment count and budget, inspect
the metrics and videos, then increase them as appropriate for the GPU.

```bash
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

Playback and bounded evaluation:

```bash
python scripts/rsl_rl/play.py --task Unitree-Go2-Conga --num_envs 8 --checkpoint '/absolute/path/model_N.pt'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python scripts/conga_check.py --headless --task Unitree-Go2-Conga --num_envs 8 --steps 2000 --checkpoint '/absolute/path/exported/policy.pt' --output logs/conga_evaluation
```

Match the evaluation task/stage to the trained checkpoint. For intermediate kick
amplitudes, change the evaluation config's reference scale to the trained value
before evaluating; `play.py` does not expose Hydra overrides.

## Evaluation and acceptance

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
