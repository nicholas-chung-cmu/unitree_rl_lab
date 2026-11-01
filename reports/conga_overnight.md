# Go2 hierarchical overnight experiment — stopped at validation

Date: October 29, 2026, America/New_York.

**No overnight training was launched.** The requested four-hour experiment was
conditional on successful environment validation. A fresh, balanced raised-entry
preflight admitted 9 of 16 requested initializations; seven preparations timed out.
The pipeline stopped before Phase 2, following the user's instruction to stop on
failed validation. There is no overnight tmux session or training child process.

This was a strict initializer-readiness check requiring every sampled first attempt
to produce a valid entry. That validation criterion was chosen conservatively for
unattended execution; it is separate from the user's 80%/70% learned-policy gates.
The nine admitted states were valid. The failure is unreliable state preparation,
not evidence that the STEP policy failed nine valid landing trials.

## Experiments attempted and actual duration

| Work | Configuration | Actual outcome |
|---|---|---|
| Repository/GPU inspection | `feature/hierarchical-conga`; RTX 5070 Ti | About 14.2 GiB free; no competing training process. Existing uncommitted implementation preserved. |
| CPU numerical/contract regression | 21 hierarchical core tests + 10 legacy reference tests | 31 passed in 1.35 s. Covers event-based steps, both diagonal contracts, airborne kick completion and timing. |
| Raised-entry simulation preflight | 16 environments, seed 9029, 8 requested FL and 8 requested FR entries | Failed initializer readiness; 367 control steps, 14.08 s process wall time including simulator startup. |
| Substantial PPO training | None | **0 iterations, 0 training seconds.** |
| Reward/curriculum variations | None | No training variations attempted. |
| Kick/transition policy evaluation | Not reached | Prerequisite validation failed. |

No four-hour job, memory-recovery loop or hyperparameter search was started.
The original legacy policies and their checkpoints were not changed. No shared
Isaac Lab checkout or package environment was modified.

## Configuration and validation

Task: `Unitree-Go2-Hierarchical-Raised-Step`, using the existing Stage B configuration
with `raised_fraction=1.0` for this probe. No rewards or preparation-controller
parameters were changed. The test assigned front legs deterministically by
environment parity so both diagonals were represented. It assessed exactly one
initialization attempt per environment; later automatic resets did not add trials.

- 16 parallel environments, seed 9029; 20 ms control interval.
- Zero learned-policy actions; the existing physical preparation controller acts
  during its preparation substate through the stock joint-position actuators.
- Target raised-foot sole clearance sampled in .04–.06 m; outward displacement
  sampled in .06–.09 m.
- Preparation moves over several seconds, with admission from 3.8 s and a
  seven-second preparation timeout.
- Admission requires three confirmed supports, the selected front foot airborne,
  endpoint tolerances, tilt below .2 rad and roll/pitch angular speed below
  .5 rad/s, continuously valid for .1 s.
- The audit independently checks finite joint states, hard joint limits with a
  .02 rad numerical tolerance, contact configuration, selected diagonal membership,
  raised-foot clearance, and body stability at actual STEP dispatch.
- All observations and rewards were checked for NaNs/Infs throughout the probe.

The supplied four-hour budget was not consumed on training after this failure.
The short validation probe took roughly 14 seconds of process wall time; this is
not a claim of exactly measured GPU-active compute time.

## Results by starting configuration

| Requested entry | Attempts | Valid admissions | Admission rate | Preparation timeouts | Preparation falls |
|---|---:|---:|---:|---:|---:|
| FL raised → STEP(FL,RR) | 8 | 4 | 50.0% | 4 | 0 |
| FR raised → STEP(FR,RL) | 8 | 5 | 62.5% | 3 | 0 |
| Total | 16 | 9 | 56.25% | 7 | 0 |

All nine admissions passed the measured joint, contact and selected-pair checks.
No invalid configuration was admitted. Both raised-leg cases preserved the front
foot airborne at STEP dispatch; no intermediate touchdown was inserted.
The seven failures did not satisfy the combined preparation-admission condition
before timeout. The current record does not identify one single failing component
for every timeout; detailed endpoint/support/controller diagnostics are the next
investigation, not a proven root-cause conclusion.

![Initialization outcomes, not policy learning](conga_overnight_initialization.png)

The plot reports initialization outcomes only. There are **no new learning curves**,
because no PPO experiment was started. Existing earlier smoke updates are not
presented as overnight learning progress.

### Timing statistics

For accepted initializations, time from episode start to STEP dispatch:

| Entry | Mean | Range |
|---|---:|---:|
| FL raised | 6.150 s | 5.860–6.820 s |
| FR raised | 6.164 s | 5.840–7.320 s |

These are slow preparation times, **not STEP completion times**. No learned STEP
completion-duration, placement-error or displacement-error distribution was
measured tonight. Those policy statistics remain unavailable rather than being
reported as zero. The simulator recorded 13 resets across all environments while
waiting for the first attempt in every environment to resolve; that reset count
includes post-admission activity and is not a count of preparation failures/falls.

## Validation coverage and advancement gates

- Current registrations/configuration and the installed code were inspected.
  The tested Stage B configuration validated and the simulator initialized/reset.
- Joint/contact audits, finite observations/rewards, both diagonal memberships,
  and both raised-front-leg cases were exercised in this fresh probe.
- Fresh CPU tests verified that STEP success requires actual swing/landing events
  and KICK success requires an airborne endpoint.
- Earlier implementation checks already covered all six configs, normal-entry
  simulation and small PPO/save/load/export. They are historical evidence in
  `docs/go2_hierarchical.md`, not a newly completed overnight preflight.
- Fresh four-foot simulation and the small PPO preflight were not rerun after
  the raised-initializer check failed. No success is claimed for the complete
  nine-item Phase 1 checklist.

| Gate | Result |
|---|---|
| Phase 1 initializer readiness | **Failed: 7/16 preparation timeouts** |
| Held-out four-foot STEP success ≈80% | Not evaluated |
| Held-out raised-entry STEP success ≈70% | Not evaluated |
| Bilateral full-amplitude airborne KICK | Not evaluated |
| STEP→KICK→STEP without reset | Not evaluated |
| Full dance readiness | Not reached; no full dance training |

Initialization admission rates must not be compared with the learned raised-entry
STEP success gate: they measure different operations.

## Artifacts, provenance and checkpoints

Experiment root: `logs/conga_overnight_20261029/`.

- `status.json`: `stopped_validation`; no training process or tmux session.
- `preflight_raised/result.json`: per-attempt outcomes and admitted physical states.
- `preflight_raised/env.yaml`: exact environment configuration.
- `preflight_raised/params/deploy.yaml`: action/observation metadata; not hardware compatibility.
- `preflight_raised/process.log`: complete simulator output.
- `initialization_attempts.csv`: all 16 first-attempt results.
- `source_snapshot.tar.gz`, `source_manifest.json`, `tracked_changes.patch`,
  `git_status.txt`: effective source/provenance, including uncommitted source.

Git HEAD: `16f90730eec367c353f355dc0cdaffaea988c3ed`.
Branch: `feature/hierarchical-conga`. The working tree was already dirty with the
new architecture; no commit, reset or cleanup was performed.

**No new policy checkpoint was generated.** Prior bounded smoke artifacts remain
at `logs/hierarchical_ppo_final_smoke/` and
`logs/rsl_rl/go2_hierarchical/2026-10-29_01-28-13_architecture_smoke/model_0.pt`.
They are not learned-skill checkpoints or overnight experiment results.

## Recommended next experiment and verified command

Improve the physical raised-entry preparation procedure, then rerun this balanced
admission audit on both legs and additional seeds. Instrument the failed endpoint,
support and residual-motion conditions before changing controller parameters.
Use one controlled change at a time. Once all required validation passes, launch
the authorized bounded training sequence with the held-out policy gates; do not
substitute initialization admission for learned skill success.

The preflight command was executed successfully as a diagnostic, and correctly
reported a **failed readiness result**:

```bash
/home/rml2/anaconda3/envs/nick_sillywalk/bin/python scripts/hierarchical_preflight.py \
  --headless --stage B --num_envs 16 --seed 9029 \
  --output logs/conga_overnight_20261029/preflight_raised
```

Use a new output directory for another probe to preserve this record. The helper
also supports `--stage A` and `--stage C`, but those modes were not exercised in
this overnight attempt. Check `result.json`, since simulator teardown can hide
a Python failure in the process exit status.

To check the outcome later:

```bash
cat logs/conga_overnight_20261029/status.json
cat reports/conga_overnight.md
```

There is no running overnight session to attach to. The conditional tmux launch
was not reached because preflight failed.
