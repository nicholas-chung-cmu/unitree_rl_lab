# Go2 CPG hardware test integration

The complete operator guide is in the conga bundle/repository at
`docs/go2_cpg_hardware.md`; qualification evidence is in
`reports/go2_cpg_hardware6h.md`. This integration is an opt-in test candidate,
not hardware-qualified. No real robot was connected during development.

Build `quad_msgs`, `quad_utils` and `robot_driver` natively in a separate overlay.
The custom `CpgSensorFrame` carries a single coherent LowState sample. The shadow
adapter consumes it without motor output. The active adapter feeds `cpg/target`
only after real calibration/timing attestations and explicit READY.

`quad_cpg_hardware.py` requires the complete conga root, selection, profile,
policy Python and `activate_hardware:=true`. It never arms automatically and
checks the worker before vendor initialization. Launching the driver releases
Unitree Sport mode; start supported/seated with one motor-command owner.

For this branch: SIT=0, READY=1, SIT_TO_READY=2, READY_TO_SIT=3, SAFETY=4.
Do not use the generic tutorial's mode 2 as a walking request. Neutral input
keeps dancing. A requests deliberate settling/standing; B/SAFETY latches damping.
Damping is not a guaranteed upright recovery, and a fault requires restart.

The existing learned walking controller and its policy remain available as the
rollback path. Hardware-specific changes are guarded by `controller=cpg` except
for the additive coherent raw-sensor topic and same-packet force retrieval.
