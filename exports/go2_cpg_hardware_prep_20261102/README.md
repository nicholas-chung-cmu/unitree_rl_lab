# Go2 conga hardware test candidate

Start with **docs/go2_cpg_hardware.md**. The complete scheduler/reference/IK/
residual runtime is included; replacing a walking ONNX actor alone will not work.
This bundle has explicit hardware activation gates, but has NOT been qualified
on a physical robot. The template profile cannot activate hardware.

The worker runs on CPU with torch, numpy, onnxruntime and PyYAML. ROS stays in
its system Python environment. Build the Quad-SDK integration natively on the
robot in a separate overlay, preserving the existing walking installation.
quad_sdk/working_tree.patch and base.json describe the desktop source changes;
quad_sdk/source contains review copies. Do not copy desktop build binaries.

Read reports/go2_cpg_hardware6h.md for measured results, rejected training
checkpoints, simulation videos and remaining real-robot checks. Evidence is
provided below evidence/ and the selected log paths. Full traces remain in the
source workspace. manifest.json hashes every bundled file.

No physical motor commands were sent while preparing this bundle.
