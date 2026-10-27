# Unitree RL Lab project instructions

## Finalized Go2 conga motion specification

Implement a phase-conditioned PPO task on flat ground using the existing Go2
actuator, action, proprioception, PPO, and export infrastructure. This specification
defines the requested reference motion; physical feasibility and learned stability
remain to be established in simulation. Keep this file synchronized with changes.

- External simulation-time clock: 120 BPM, 0.5 s/beat, 2 s/phrase, 4 s/full
  alternating sequence. Speed commands must never affect clock frequency. Each
  environment resets its own clock; command resampling does not reset it.
- Foot order everywhere: FL, FR, RL, RR. Body coordinates: x forward, y left;
  vertical clearance uses world z above the flat local ground, corrected for the
  0.022 m spherical foot radius in the supplied Go2 URDF.
- Three traveling diagonal swings per phrase: [0.04, 0.40], [0.54, 0.90],
  [1.02, 1.23] s. First phrase: FL/RR, FR/RL, FL/RR; next phrase swaps
  diagonals. The shortened third swing allows touchdown before the kick.
- Kick FR during [1.25, 1.50] s of the first phrase; FL during the same interval
  of the next phrase. Thus the kick front leg belongs to the next diagonal swing.
  Only that front foot swings during the kick; the other three are encouraged to
  support the body. Beat 4 [1.50, 2.00] is recovery, including a deliberate
  diagonal phase reset for alternating kicks. After kick touchdown, contact
  targets relax smoothly to neutral 0.5 by 1.60 s and return to support targets
  over [1.92, 2.00] s. Recovery foot placement is free so resumed travel does not
  conflict with forced four-foot stance. Neutral contacts receive no clearance
  reward or touchdown timing penalty. Contact targets are soft near ordinary
  liftoff/touchdown too, not kinematic constraints.
- The kick uses a sin-squared extension/retraction pulse, zero velocity at start,
  maximum extension, and end. Full amplitude is 0.08 m outward from the nominal
  front-foot body-frame y position; sole clearance peaks at 0.05 m. The lateral
  target is body-relative and vertical clearance ground-relative. Fore-aft foot
  position is unconstrained. Nominal y comes from the default-pose URDF geometry.
- Commands support backward/forward, lateral motion, moving turns, and turns in
  place. Cap the XY command vector norm at 1.25 m/s, not each component separately.
  Smoothly reduce translation AND yaw over [1.10, 1.25] s, hold zero through
  the kick, and restore over [1.50, 1.70] s. No catch-up speed compensation.
  Report traveling speed separately from phrase-average speed.
- Zero commands still require stepping and synchronized bounce; they do not imply
  standing still. The clock and contact schedule continue independently.
- Body bounce: sinusoidal 0.015 m peak-to-peak (0.0075 m amplitude), 2 Hz;
  nominal base height 0.32 m. Track height and reference vertical velocity rather
  than suppressing all vertical velocity. Initial reward weights are unvalidated.
- Preserve torque, joint limit, action rate, orientation, slip, and collision
  penalties. Replace velocity-dependent air time, symmetric air-time variance,
  near-zero joint-position penalties, and indiscriminate vertical-velocity cost.
- Curriculum stages: continuous fixed-cadence diagonal stepping including zero
  commands and yaw; add bounce; add kick with progressive amplitude; full command
  range and full dance; domain randomization and robustness evaluation. Keep the
  policy observation/action layout identical across stages for checkpoint resume.

## Verification and operating rules

Work incrementally: verify the numerical references, then rewards/commands and
configuration/registration, then simulation/reset/export smoke tests. Do not start
a long training run without explicit user approval. Do not claim a stable learned
gait from a reference test or an untrained simulation rollout. Preserve pre-existing
user edits, especially robot asset paths in `assets/robots/unitree.py`.

Evaluate actual touchdown timing and missing/excess contacts, cadence by commanded
speed, velocity error, kick lateral/height error, bounce error, falls, and slipping.
The short third step, a 0.25 s kick, and 1.25 m/s travel may be demanding; retain
these targets and evaluate feasibility rather than silently changing them.
Stock deployment needs the new clock/phase observations and travel envelope;
exporting weights alone does not make the existing hardware controller compatible.

Implementation, curriculum commands, evaluation procedures, and the latest
verification record live in `docs/go2_conga.md`. As of 2026-10-27, reference and
adapter tests, configuration/Hydra checks, GPU smoke simulation, and PPO/export
checks passed. No learning updates or long training run have been performed.
