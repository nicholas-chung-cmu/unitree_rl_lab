#!/usr/bin/env python3
"""Run the portable CPG dance controller against the official Go2 MuJoCo model.

This is a local simulation transport. It has no robot command publisher.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from deploy.cpg.runtime import DanceRuntime, SensorFrame, SDK_NAMES, FOOT_NAMES, rotation, load_selection
from deploy.cpg.inputs import WirelessInput, scripted_sticks, stick_velocity


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('/home/rml2/unitree/unitree_mujoco/unitree_robots/go2/go2.xml'))
    parser.add_argument('--selection', type=Path, default=ROOT/'logs/conga_cpg_session_20261101/selected/dance_bounce.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=40.)
    parser.add_argument('--input', choices=['zero', 'fixed', 'scripted', 'wireless'], default='zero')
    parser.add_argument('--velocity',type=float,nargs=3,default=[0.,0.,0.])
    parser.add_argument('--command-limits',type=float,nargs=3,help='Full-stick vx/vy/yaw; default from qualified selection or .05/.04/.10')
    parser.add_argument('--wireless-state', type=Path, default=Path('/tmp/go2_wireless.json'))
    parser.add_argument('--first-front', choices=['FL', 'FR'], default='FL')
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--camera',choices=['fixed','follow'],default='fixed')
    parser.add_argument('--diagnostic-truth', action='store_true', help='Explicitly use true foot height/speed for feedback; not deployment qualification')
    parser.add_argument('--nominal-only',action='store_true',help='Diagnostic: zero residual actions to test nominal trajectories')
    return parser.parse_args()


def make_model(source, output):
    # Preserve the official robot dynamics/contacts. Use a flat floor rather than
    # its demo scene's obstacles. Never modify the external checkout.
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find('compiler')
    compiler.set('meshdir', str((source.parent/compiler.get('meshdir', '')).resolve()))
    root.find('option').set('timestep', '.002')
    world = root.find('worldbody')
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='540')
    assets=root.find('asset')
    if assets is None:assets=ET.SubElement(root,'asset')
    ET.SubElement(assets,'texture',name='travel_grid',type='2d',builtin='checker',rgb1='.20 .25 .30',rgb2='.32 .37 .42',width='256',height='256')
    ET.SubElement(assets,'material',name='travel_grid',texture='travel_grid',texrepeat='5 5',texuniform='true')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 .1', material='travel_grid', friction='.8 .02 .01')
    ET.SubElement(world, 'light', pos='0 0 3', dir='0 0 -1', directional='true', ambient='.25 .25 .25', diffuse='.9 .9 .9')
    tree.write(output)


def main(args):
    if args.seconds <= 0 or not np.isfinite(args.seconds):
        raise ValueError('Duration must be positive and finite')
    if args.input == 'wireless' and not args.viewer:
        raise ValueError('Live wireless input requires --viewer for real-time pacing')
    args.output.mkdir(parents=True, exist_ok=False)
    args._created_output = True
    def write(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2)+'\n')
    write('status.json', {'status': 'running'})
    torch.set_num_threads(1)
    selection = load_selection(args.selection)
    runtime = DanceRuntime(selection['deploy_metadata'], selection['onnx_export'], int(args.first_front == 'FR'))
    if args.nominal_only:
        from types import SimpleNamespace
        runtime.policy=SimpleNamespace(run=lambda *unused:[np.zeros((1,12),dtype=np.float32)])
    if args.command_limits is None:args.command_limits=selection.get('joystick_limits',[.05,.04,.10])
    limits=np.asarray(args.command_limits)
    if not np.isfinite(limits).all() or (limits<0).any() or max(limits[:2])>runtime.controller.cfg.gait.max_speed+1e-6 or limits[2]>runtime.controller.cfg.gait.max_yaw+1e-6:
        raise ValueError('Joystick limits exceed selected controller command bounds')
    if args.input=='fixed' and (np.linalg.norm(args.velocity[:2])>runtime.controller.cfg.gait.max_speed+1e-6 or abs(args.velocity[2])>runtime.controller.cfg.gait.max_yaw+1e-6):
        raise ValueError('Fixed evaluation command exceeds the selected controller command bounds')
    if args.video and not args.viewer:
        os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    make_model(args.model.resolve(), args.output/'scene.xml')
    model = mujoco.MjModel.from_xml_path(str(args.output/'scene.xml'))
    data = mujoco.MjData(model)
    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in SDK_NAMES])
    if (joint_ids < 0).any():
        raise ValueError('Model missing Go2 joints')
    qids = model.jnt_qposadr[joint_ids]
    vids = model.jnt_dofadr[joint_ids]
    actuator_ids = np.array([np.flatnonzero(model.actuator_trnid[:, 0] == j)[0] for j in joint_ids])
    foot_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, leg+'_foot') for leg in FOOT_NAMES])
    geom_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg) for leg in FOOT_NAMES])
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    default_sdk = np.empty(12)
    default_sdk[runtime.map] = runtime.defaults
    data.qpos[:3] = [0, 0, .325]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[qids] = default_sdk
    mujoco.mj_forward(model, data)
    kp, kd = np.empty(12), np.empty(12)
    kp[runtime.map], kd[runtime.map] = runtime.kp, runtime.kd
    contacts = np.zeros(4)
    truth_contact = np.zeros(4,dtype=bool)
    force = np.zeros(6)
    body_velocity = np.zeros(6)
    wireless = WirelessInput(args.wireless_state,limits=args.command_limits)
    viewer = renderer = writer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.65, 135, -20
    camera.lookat[:]=[0.,0.,.25]
    if args.camera=='fixed':camera.distance=2.0
    if args.camera=='fixed' and args.input=='fixed':
        # Keep a translating robot in view without hiding motion by following it.
        extent=np.linalg.norm(args.velocity[:2])*args.seconds
        camera.distance=max(2.,2.+.65*extent)
        camera.lookat[:2]=np.asarray(args.velocity[:2])*args.seconds*.4
    if args.video:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(model, height=540, width=960)
        writer = imageio.get_writer(args.output/'dance.mp4', fps=50, macro_block_size=2)
    records = []
    inference_times = []
    wall_start = time.monotonic()
    fault = None
    stopped = None
    write('settings.json', {**vars(args), 'model': str(args.model), 'selection': str(args.selection),
          'output': str(args.output), 'wireless_state': str(args.wireless_state),
          'sensor_feedback': 'truth diagnostic' if args.diagnostic_truth else 'IMU/encoder/support-foot estimate',
          'robot_model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
          'onnx_sha256': selection['onnx_sha256'], 'mujoco_version': mujoco.__version__,
          'control_dt': runtime.dt, 'physics_dt': model.opt.timestep,
          'physics': 'official robot damping/friction/inertia; Go2HV torque-speed envelope; flat floor',
          'hardware_transport': False})
    try:
        for tick in range(round(args.seconds/runtime.dt)):
            if viewer and not viewer.is_running():
                stopped = 'viewer closed'
                break
            # mj_step integrates qpos/qd last. Refresh derived poses, IMU and
            # contacts so all sensor fields represent the same control instant.
            mujoco.mj_forward(model, data)
            contacts[:] = 0
            for i in range(data.ncon):
                contact = data.contact[i]
                for foot, geom in enumerate(geom_ids):
                    if geom in (contact.geom1, contact.geom2):
                        mujoco.mj_contactForce(model, data, i, force)
                        contacts[foot] += np.linalg.norm(force[:3])
            truth_contact=np.where(truth_contact,contacts>2.,contacts>=5.)
            positions = data.xpos[foot_ids].copy()
            speeds = np.zeros((4, 3))
            for i, body in enumerate(foot_ids):
                # BODY uses the inertial-frame origin (xipos), which differs
                # even for these massless foot bodies. XBODY matches xpos/FK.
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, body_velocity, 0)
                speeds[i] = body_velocity[3:]
            frame = SensorFrame(float(data.time), data.qpos[qids].copy(), data.qvel[vids].copy(),
                                data.sensor('imu_quat').data.copy(), data.sensor('imu_gyro').data.copy(), contacts.copy())
            velocity = (wireless.read() if args.input == 'wireless' else
                        stick_velocity(*scripted_sticks(data.time),limits=args.command_limits) if args.input == 'scripted' else
                        np.asarray(args.velocity) if args.input=='fixed' else np.zeros(3))
            if wireless.stop:
                stopped = 'wireless B stop'
                break
            start = time.perf_counter()
            with torch.inference_mode():
                observation = runtime.observe(frame, velocity, (positions, speeds) if args.diagnostic_truth else None)
                target = runtime.act(observation)
            inference_times.append(time.perf_counter()-start)
            c = runtime.controller
            record = {k: v.numpy().copy() for k, v in c.last_records.items()}
            # Audits always use independent physical truth, including estimate mode.
            record['truth_contact']=truth_contact[None].copy()
            record['estimated_sole'] = record['sole'].copy()
            record['sole'] = positions[None, :, 2]-.022
            record['estimated_foot_velocity'] = runtime.robot.data.body_lin_vel_w.numpy().copy()
            rot = rotation(frame.quaternion)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, base, body_velocity, 0)
            record.update(height=np.array([data.xpos[base, 2]]), gravity=observation[:, 3:6].copy(),
                          angular_velocity=frame.gyro[None], joints=frame.q[[SDK_NAMES.index(n) for n in
                          [f'{leg}_{j}_joint' for leg in FOOT_NAMES for j in ('hip','thigh','calf')]]][None],
                          foot_velocity=speeds[None], feet_body=((positions-data.xpos[base])@rot)[None],
                          body_velocity=body_velocity[None, 3:].copy(), body_velocity_local=(rot.T@body_velocity[3:])[None],
                          body_position=data.xpos[base][None].copy(),
                          reference_joints=c.q_ref[:, c.joint_ids].numpy().copy(),
                          operator_velocity=velocity[None], planned_velocity=c.velocity.numpy().copy(),
                          target_velocity=getattr(c,'travel_velocity',c.velocity).numpy().copy(),
                          slip=np.array([(np.linalg.norm(speeds[:, :2], axis=-1)*c.events.contact[0].numpy()).sum()/max(1,c.events.contact.sum().item())]),
                          fall=np.array([False]), reset=np.array([False]),
                          observation=observation.copy(), action=runtime.last_action.copy(), target_sdk=target[None],
                          sensor_q=frame.q[None], sensor_qd=frame.qd[None], sensor_quaternion=frame.quaternion[None],
                          sensor_force=frame.foot_force[None], sensor_time=np.array([frame.time]))
            records.append(record)
            if args.camera=='follow':camera.lookat[:] = data.xpos[base]
            if viewer:
                viewer.cam.lookat[:] = camera.lookat
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = camera.distance, camera.azimuth, camera.elevation
                viewer.sync()
            if renderer:
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
            for _ in range(round(runtime.dt/model.opt.timestep)):
                q, qd = data.qpos[qids], data.qvel[vids]
                torque = kp*(target-q)-kd*qd
                bound = np.where(torque*qd > 0, 20.2, 23.4)*np.clip((30-np.abs(qd))/(30-13.5), 0, 1)
                data.ctrl[actuator_ids] = np.clip(torque, -bound, bound)
                mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or data.xpos[base, 2] < .16:
                fault = 'Physical fall/nonfinite state'
                records[-1]['fall'][:] = True
                break
            if viewer:
                time.sleep(max(0., wall_start+(tick+1)*runtime.dt-time.monotonic()))
    except (ValueError, RuntimeError) as error:
        fault = str(error)
        if records and 'tilt' in fault.lower():
            records[-1]['fall'][:] = True
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
        if viewer: viewer.close()
    if records:
        np.savez_compressed(args.output/'traces.npz', **{k: np.stack([r[k] for r in records]) for k in records[0]}, groups=np.array([0]))
        from cpg_tracking_metrics import tracking_metrics
        write('tracking_audit.json',tracking_metrics({k:np.stack([r[k] for r in records]) for k in records[0]}))
    result = {'status': 'fault' if fault else 'stopped' if stopped else 'complete', 'fault': fault, 'stop_reason': stopped,
              'seconds': float(data.time), 'requested_seconds': args.seconds, 'samples': len(records),
              'wall_seconds': time.monotonic()-wall_start,
              'falls': sum(bool(r['fall'][0]) for r in records),
              'metrics': {k: float(v.sum()) for k, v in runtime.controller.metrics.items()},
              'input_status': wireless.status if args.input == 'wireless' else args.input,
              'control_p50_ms': float(np.median(inference_times)*1000) if inference_times else None,
              'control_p99_ms': float(np.percentile(inference_times,99)*1000) if inference_times else None,
              'feedback': 'diagnostic_truth' if args.diagnostic_truth else 'estimated', 'hardware_tested': False}
    write('result.json', result)
    write('status.json', result)
    print(json.dumps(result, indent=2))
    return 1 if fault else 0


if __name__ == '__main__':
    args = arguments()
    try:
        sys.exit(main(args))
    except Exception as error:
        if getattr(args, '_created_output', False):
            (args.output/'error.json').write_text(json.dumps({'error': str(error), 'traceback': traceback.format_exc()}, indent=2))
            (args.output/'status.json').write_text(json.dumps({'status':'error','error':str(error)}, indent=2))
        raise
