"""Create a relocatable, checksum-recorded CPG simulation/Quad-SDK shadow bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import importlib.metadata
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection',type=Path,default=ROOT/'logs/conga_cpg_session_20261101/selected/dance_bounce.json')
    parser.add_argument('--report',default='reports/go2_cpg_mujoco.md')
    parser.add_argument('--hardware-prep',action='store_true',help='Include opt-in hardware adapter, guides and native Quad-SDK sources; does not qualify hardware')
    parser.add_argument('--quad-sdk',type=Path)
    args=parser.parse_args()
    if args.hardware_prep and not args.quad_sdk:parser.error('--hardware-prep requires --quad-sdk')
    args.output.mkdir(parents=True,exist_ok=False)
    files=['deploy/cpg/__init__.py','deploy/cpg/runtime.py','deploy/cpg/inputs.py','deploy/cpg/quad_sdk.py','deploy/cpg/safety.py',
           'deploy/cpg/wireless_receiver.cpp','deploy/cpg/CMakeLists.txt',
           'scripts/cpg_mujoco.py','scripts/cpg_quad_ros2.py','scripts/cpg_worker.py',
           'scripts/analyze_cpg_contacts.py','scripts/audit_cpg_kicks.py','scripts/cpg_tracking_metrics.py',
           'docs/go2_cpg_mujoco.md','docs/go2_cpg_robustness.md',args.report,
           '.local/unitree_ros/robots/go2_description/urdf/go2_description.urdf']
    prefix='source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/'
    files += [prefix+'cpg/'+name+'.py' for name in ('core','dance','controller','sensing')]
    files += [prefix+'skills/preparation.py']
    if args.hardware_prep:
        files += ['deploy/cpg/lowstate_receiver.cpp','deploy/cpg/hardware_profile.template.json',
            'scripts/cpg_quad_hardware.py','scripts/cpg_wireless_ros2.py','scripts/cpg_contact_calibration.py',
            'scripts/cpg_preflight.py','scripts/analyze_cpg_contacts_v2.py','docs/go2_cpg_hardware.md']
    if (ROOT/'LICENSE').exists():files.append('LICENSE')
    figure=Path(args.report).with_suffix('.png')
    if (ROOT/figure).exists():files.append(str(figure))
    for name in files:
        destination=args.output/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
    selection=json.loads(args.selection.read_text())
    for key in ('onnx_export','jit_export','deploy_metadata','settings','checkpoint'):
        if key in selection and not Path(selection[key]).is_absolute():
            selection[key]=str(args.selection.resolve().parent/selection[key])
    if hashlib.sha256(Path(selection['onnx_export']).read_bytes()).hexdigest()!=selection['onnx_sha256']:
        raise ValueError('Selected policy checksum mismatch')
    assets=args.output/'assets';assets.mkdir()
    for key,name in [('onnx_export','policy.onnx'),('jit_export','policy.pt'),('deploy_metadata','deploy.yaml'),('settings','settings.json')]:
        shutil.copy2(selection[key],assets/name)
    selected={key:selection[key] for key in ('onnx_sha256','jit_sha256','checkpoint_sha256','actor_critic_actions','equivariant_actor','joystick_limits','observation_dim','actuator_profile','command_min','command_max','forward_at_max_yaw','sampled_torque_cap_nm') if key in selection}
    selected['jit_sha256']=hashlib.sha256((assets/'policy.pt').read_bytes()).hexdigest()
    if 'checkpoint' in selection:
        selected['checkpoint_sha256']=hashlib.sha256(Path(selection['checkpoint']).read_bytes()).hexdigest()
    selected.update(onnx_export='policy.onnx',jit_export='policy.pt',deploy_metadata='deploy.yaml',settings='settings.json',
                    hardware_compatible=False,qualification=selection.get('qualification','Simulation and ROS2 shadow candidate; see included qualification report for measured limits'))
    (assets/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    (args.output/'README.md').write_text('''# Go2 CPG deployment candidate

This contains the complete scheduler/reference/IK/residual controller, not just
the ONNX actor. It is a simulation and Quad-SDK ROS 2 **shadow** candidate.
Hardware operation is not qualified. See the included report for motion results.

Use Python with torch, numpy, onnxruntime and PyYAML. MuJoCo is additionally
needed for simulation; imageio/imageio-ffmpeg for video. ROS runs in its own
Python environment and starts the policy worker with --policy-python.

From this folder:

```bash
python scripts/cpg_mujoco.py --selection assets/selection.json \\
  --model /path/to/unitree_mujoco/unitree_robots/go2/go2.xml \\
  --viewer --seconds 120 --output logs/view
```

After sourcing ROS 2 and your Quad-SDK quad_msgs workspace:

```bash
python3 scripts/cpg_quad_ros2.py --selection assets/selection.json \\
  --policy-python /path/to/policy/python --robot-ns /robot_1
```

The ROS adapter publishes only /cpg_sim/leg_command_array. It does not install
a hardware LegController mode. Read docs/go2_cpg_mujoco.md and the included
qualification report for mappings, calibration requirements, and results.
manifest.json records every bundled file's SHA-256 hash.
''')
    readme=args.output/'README.md'
    readme.write_text(readme.read_text().replace('# Go2 CPG deployment candidate',
        '# Go2 CPG deployment candidate\n\n'+selected['qualification']))
    if args.hardware_prep:
        (args.output/'README.md').write_text('''# Go2 conga hardware test candidate

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
''')
        quad=args.quad_sdk.resolve();qout=args.output/'quad_sdk';qout.mkdir()
        modified=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=quad,text=True).splitlines()
        new=[name for name in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=quad,text=True).splitlines()
             if 'cpg' in name.lower()]
        patch=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=quad)
        for name in new:
            part=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',name],cwd=quad,capture_output=True)
            if part.returncode not in (0,1):raise RuntimeError(part.stderr.decode())
            patch+=part.stdout
        (qout/'working_tree.patch').write_bytes(patch)
        for name in sorted(set(modified+new)):
            if (quad/name).is_file():
                dst=qout/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(quad/name,dst)
        base={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=quad,text=True).strip(),
              'branch':subprocess.check_output(['git','branch','--show-current'],cwd=quad,text=True).strip(),
              'submodules':subprocess.check_output(['git','submodule','status'],cwd=quad,text=True),
              'files':sorted(set(modified+new)),
              'scope':'Complete working-tree integration, including earlier session changes. Review against the robot branch; git apply --check before applying. No binaries.'}
        (qout/'base.json').write_text(json.dumps(base,indent=2)+'\n')
        evidence=args.output/'evidence';evidence.mkdir()
        deps={'python':sys.version,'architecture':platform.machine(),'packages':{name:importlib.metadata.version(name) for name in ('torch','numpy','onnxruntime','PyYAML','mujoco','imageio','imageio-ffmpeg')}}
        (evidence/'host_dependencies.json').write_text(json.dumps(deps,indent=2)+'\n')
        session=ROOT/'logs/cpg_hardware6h_20261102'
        for name in ['preflight_packets.json','desktop_preflight.json','summary.json','final_validation.json']:
            if (session/name).exists():shutil.copy2(session/name,evidence/name)
        # Preserve paths linked by the report, without multi-GB physics traces.
        for source in session.rglob('*'):
            relative=source.relative_to(session)
            if not source.is_file() or any(p.endswith('_source') or p in ('before','calibration_fixture') for p in relative.parts):continue
            if 'INVALID_FOR_HARDWARE' in source.name:continue
            if source.suffix not in ('.json','.mp4') or source.name=='preflight_packets.json':continue
            if source.name in ('settings.json','physics.json') and 'training' in relative.parts:continue
            dst=args.output/'logs/cpg_hardware6h_20261102'/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
    manifest={str(path.relative_to(args.output)):hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(args.output.rglob('*')) if path.is_file()}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'bundle':str(args.output),'files':len(manifest),'policy_sha256':selected['onnx_sha256']},indent=2))


if __name__=='__main__':main()
