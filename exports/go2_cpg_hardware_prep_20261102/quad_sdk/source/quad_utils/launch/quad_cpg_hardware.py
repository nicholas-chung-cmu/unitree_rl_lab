"""Explicit opt-in Go2 CPG hardware launch. Does not arm or request standing."""
import json
from pathlib import Path
import sys
import tempfile
import subprocess
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,ExecuteProcess,IncludeLaunchDescription,OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import yaml


def start(context):
    root=Path(LaunchConfiguration('conga_root').perform(context)).expanduser().resolve()
    profile_path=Path(LaunchConfiguration('profile').perform(context)).expanduser().resolve()
    selection=Path(LaunchConfiguration('selection').perform(context)).expanduser().resolve()
    if LaunchConfiguration('activate_hardware').perform(context).lower()!='true':
        raise RuntimeError('Hardware launch requires activate_hardware:=true. For read-only shadow use cpg_quad_hardware.py without --active.')
    sys.path.insert(0,str(root/'scripts'))
    from cpg_quad_hardware import validate_profile
    profile=json.loads(profile_path.read_text());sel=json.loads(selection.read_text());validate_profile(profile,sel,True)
    # Gate all checks BEFORE creating any driver (its vendor interface releases sport mode).
    from hashlib import sha256
    actor=Path(sel['onnx_export']);actor=actor if actor.is_absolute() else selection.parent/actor
    if sha256(actor.read_bytes()).hexdigest()!=sel['onnx_sha256']:raise ValueError('Policy checksum mismatch')
    policy_python=LaunchConfiguration('policy_python').perform(context)
    trace=Path(LaunchConfiguration('trace').perform(context)).expanduser().resolve()
    if trace.exists():raise ValueError('Trace exists; use a fresh path')
    trace.parent.mkdir(parents=True,exist_ok=True)
    # Prove that ROS imports, the custom message, policy dependencies and worker
    # handshake work BEFORE starting the vendor interface / releasing Sport.
    subprocess.run(['/usr/bin/python3','-c',
        'import rclpy; from quad_msgs.msg import CpgSensorFrame; from geometry_msgs.msg import Twist'],check=True,timeout=15)
    with tempfile.TemporaryDirectory(prefix='cpg_preflight_') as tmp:
        subprocess.run(['/usr/bin/python3',str(root/'scripts/cpg_preflight.py'),
            '--selection',str(selection),'--policy-python',policy_python,
            '--output',str(Path(tmp)/'preflight.json')],check=True,timeout=55)
    with tempfile.NamedTemporaryFile(dir=trace.parent,prefix='.cpg_trace_probe_'):
        pass
    ns=LaunchConfiguration('namespace').perform(context).strip('/')
    if not ns or '/' in ns:raise ValueError('Use a single robot namespace, e.g. robot_1')
    with tempfile.NamedTemporaryFile(mode='w',prefix='cpg_hardware_',suffix='.yaml',delete=False) as f:
        yaml.safe_dump({'/**/robot_driver':{'ros__parameters':{'cpg.hardware_enabled':True,'cpg.contact_calibrated':True,'cpg.timing_verified':True}}},f)
        params=f.name
    return [
        ExecuteProcess(cmd=['/usr/bin/python3','-u',str(root/'scripts/cpg_quad_hardware.py'),'--selection',str(selection),'--profile',str(profile_path),'--policy-python',policy_python,'--robot-ns','/'+ns,'--trace',str(trace),'--active'],output='screen'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(Path(get_package_share_directory('quad_utils'))/'launch/robot_driver.py')),
            launch_arguments={'robot_type':'go2','controller':'cpg','is_hardware':'true','mocap':'false','use_sim_time':'false','namespace':ns,'extra_params':params}.items())]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('conga_root'),DeclareLaunchArgument('selection'),DeclareLaunchArgument('profile'),
        DeclareLaunchArgument('policy_python'),DeclareLaunchArgument('activate_hardware',default_value='false'),
        DeclareLaunchArgument('namespace',default_value='robot_1'),DeclareLaunchArgument('trace',default_value='/tmp/cpg_hardware_trace.jsonl'),
        OpaqueFunction(function=start)])
