# Go2 CPG deployment candidate

This contains the complete scheduler/reference/IK/residual controller, not just
the ONNX actor. It is a simulation and Quad-SDK ROS 2 **shadow** candidate.
Strict MuJoCo kick timing still fails; hardware operation is not qualified.

Use Python with torch, numpy, onnxruntime and PyYAML. MuJoCo is additionally
needed for simulation; imageio/imageio-ffmpeg for video. ROS runs in its own
Python environment and starts the policy worker with --policy-python.

From this folder:

```bash
python scripts/cpg_mujoco.py --selection assets/selection.json \
  --model /path/to/unitree_mujoco/unitree_robots/go2/go2.xml \
  --viewer --seconds 120 --output logs/view
```

After sourcing ROS 2 and your Quad-SDK quad_msgs workspace:

```bash
python3 scripts/cpg_quad_ros2.py --selection assets/selection.json \
  --policy-python /path/to/policy/python --robot-ns /robot_1
```

The ROS adapter publishes only /cpg_sim/leg_command_array. It does not install
a hardware LegController mode. Read docs/go2_cpg_mujoco.md and the included
qualification report for mappings, calibration requirements, and results.
manifest.json records every bundled file's SHA-256 hash.
