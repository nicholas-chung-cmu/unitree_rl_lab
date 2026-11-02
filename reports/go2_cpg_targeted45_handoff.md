# Overnight handoff — targeted robustness session

No overnight work is launched or scheduled. The user will authorize the next
session separately. The current session must stop by05:37:12UTC on2026-11-02.

## Starting point

Use the outcome and selected bundle in `go2_cpg_targeted45.md`. The strongest
current checkpoint is the timing-prioritized branch at300 additional updates,
`logs/cpg_targeted45_20261102/timing_train2/training/model_299.pt`. Keep the prior
robust75@250 export as the fallback. Do not assume the latest checkpoint is best.
The current actor has75 inputs; this session did not change the deployable
observation contract or enable hardware.

## Evidence to preserve

Matched nominal gates cover eight40s commands, including backward as a separate
diagnostic. Matched MuJoCo transfer suites include neutral90s, forward60s, a
non-backward demo120s, armature.015, mass+10%, gains-10%, sensor noise,20ms
sensor delay and20ms action delay. Stronger combined Isaac uncertainty uses
32envs,60s, command changes every6s and seed9321. Original settings, snapshots,
traces and endpoint/support diagnoses are in `logs/cpg_targeted45_20261102/`.

## Suggested next experiment

1. Prioritize non-backward motion and reliable support after a kick. Diagnose
   support-force margin before adding a dense stance-support reward. A replay
   of precision600 found an opposite front foot lifted2.7mm for40ms just after
   the post-kick landing. Do not mask this by relaxing the rejection threshold.
2. Reconcile independent audit windows with the controller completion tick. The
   same trace has66/66 physical-window steps but62/66 controller completions;
   support loss crosses40ms at about67ms after the beat, just outside the
   offline60ms window. Preserve old counts and add a separately versioned audit.
3. Preserve foot clearance while reducing drift. The selected neutral test's
   median step peak is4.9cm versus5.6cm at the starting policy. Nominal target
   remains6cm. Distinguish body stabilization from shortened/lower swings.
4. Keep targeted randomization and low learning rates. Compare intermediate
   checkpoints with held-out dynamics/latency and both kick parities. Replicate
   promising settings across training seeds before claiming general superiority.
5. Keep the selected reference gains unchanged unless a separately qualified
   experiment supports a change. A frozen scan found only modest drift benefit
   from support damping6→14; reducing the kick shift3cm→2/1cm did little.
6. Preserve travel-only velocity accounting; kick/hold time must not cause
   catch-up speed. Restrict any initial hardware-demo envelope explicitly;
   backward remains unqualified.

## Hardware boundary

No physical robot was connected. Actuator/contact calibration, measured latency,
total PD-plus-feedforward torque limiting and airborne fault recovery remain.
Keep hardware mode blocked. No further training should start until the next
session is authorized and given its own wall-clock budget.

## Runner housekeeping

The outer Gazebo tool session reported exit143 even though all three trials
wrote successful results and the inner completion artifact. Final verification
found no remaining owned Gazebo processes. Preserve this cleanup anomaly and
review process-group shutdown before reusing the runner overnight; do not
confuse it with a robot motion failure. See the session `final_status.json`.
