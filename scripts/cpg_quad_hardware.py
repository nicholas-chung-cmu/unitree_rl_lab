#!/usr/bin/env python3
"""Go2 CPG raw-sensor adapter. Shadow by default; explicit calibrated activation.

Never publishes Unitree DDS directly. Active targets pass through robot_driver.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deploy.cpg.quad_sdk import decode_state,encode_command
from deploy.cpg.safety import CommandEnvelope,ForceCalibration,SensorGate,JsonLineReader,ControlClockGate


def validate_profile(profile,selection,active):
    if profile.get('robot_type')!='go2':raise ValueError('Go2 only')
    if active:
        for key in ('hardware_enabled','contact_calibrated','timing_verified'):
            if profile.get(key) is not True:raise ValueError('Hardware gate: '+key+' must be verified')
        if profile.get('policy_sha256')!=selection['onnx_sha256']:
            raise ValueError('Hardware profile must identify the tested policy hash')
    if selection.get('sampled_torque_cap_nm',18.)!=18.:raise ValueError('Hardware CPG command sink uses the fixed 18Nm sampled limit')
    envelope=CommandEnvelope(selection)
    if envelope.lower[:2]!=[0,0] or envelope.upper[1]!=0 or envelope.upper[0]>.12 or max(abs(envelope.lower[2]),abs(envelope.upper[2]))>.15:
        raise ValueError('Restricted hardware profile: forward 0..0.12, lateral/backward zero, yaw <=0.15')
    return envelope,ForceCalibration(profile['force_calibration'],require_measured=active)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--policy-python',required=True)
    p.add_argument('--robot-ns',default='/robot_1')
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--active',action='store_true')
    a,ros_args=p.parse_known_args()
    if a.trace.exists():p.error('Trace exists; select a new filename')
    selected=json.loads(a.selection.read_text());profile=json.loads(a.profile.read_text())
    envelope,calibration=validate_profile(profile,selected,a.active)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy
    sensor_qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST)
    from sensor_msgs.msg import JointState,Imu
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String,UInt8,Bool
    from quad_msgs.msg import FootContact,LegCommandArray,LegCommand,MotorCommand,CpgSensorFrame
    rclpy.init(args=ros_args)
    class Bridge(Node):
        def __init__(self):
            super().__init__('cpg_hardware_adapter')
            if self.get_parameter('use_sim_time').value:raise ValueError('Hardware/shadow adapter requires wall time')
            ns=a.robot_ns.rstrip('/');self.ns=ns
            a.trace.parent.mkdir(parents=True,exist_ok=True);self.trace=a.trace.open('x',buffering=1)
            self.worker=subprocess.Popen([a.policy_python,str(ROOT/'scripts/cpg_worker.py'),'--selection',str(a.selection)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)
            self.reader=JsonLineReader(self.worker.stdout)
            try:
                response=self.reader.read(30)
                if response.get('ready') is not True or response.get('dt')!=.02:raise ValueError('Invalid worker contract')
            except Exception:
                self.close_worker();self.trace.close();raise
            self.latest={};self.velocity=[0.,0.,0.];self.command_time=-math.inf
            self.gate=SensorGate();self.clock_gate=ControlClockGate();self.mode=0;self.started=False;self.fault=None;self.previous=None;self.tick=0
            self.mode_time=-math.inf;self.begin=time.monotonic();self.stand_requested=False
            self.pub=self.create_publisher(LegCommandArray,ns+'/cpg/target' if a.active else '/cpg_sim/leg_command_array',1)
            self.status=self.create_publisher(String,ns+'/cpg/status',1)
            self.fault_pub=self.create_publisher(String,ns+'/cpg/fault' if a.active else '/cpg_sim/fault',1)
            self.source_tick=None
            self.create_subscription(CpgSensorFrame,ns+'/cpg/sensors',self.sensors,sensor_qos)
            if a.active:self.create_subscription(String,ns+'/cpg/fault',lambda m:self.fail(m.data),1)
            self.create_subscription(Twist,ns+'/cmd_vel',self.command,1)
            self.create_subscription(UInt8,ns+'/state/control_mode',self.mode_callback,1)
            self.create_subscription(Bool,ns+'/cpg/stand',lambda m:setattr(self,'stand_requested',self.stand_requested or m.data),1)
            self.create_timer(.02,self.update)
            self.get_logger().info('CPG '+('ACTIVE targets, waiting for READY' if a.active else 'SHADOW ONLY, no motor output'))
        def sensors(self,m):
            now=time.monotonic()
            if self.source_tick is not None:
                delta=(m.source_tick-self.source_tick)&0xffffffff
                if delta==0:return # repeated source sample cannot refresh reception
                if delta>=0x80000000:self.fail('Regressing MCU tick');return
            self.source_tick=m.source_tick
            self.latest={k:(value,now) for k,value in [('joints',m.joints),('imu',m.imu),('contact',m.contact)]}
            stamp=self.stamp(m)
            if any(self.stamp(value)!=stamp for value,_ in self.latest.values()):self.fail('Inconsistent atomic sensor headers')

        def mode_callback(self,m):
            self.mode=m.data;self.mode_time=time.monotonic()
            if a.active and self.started and self.mode!=1:self.fail('Left READY; restart required')
        def command(self,m):
            try:self.velocity=envelope.apply([m.linear.x,m.linear.y,m.angular.z]);self.command_time=time.monotonic()
            except ValueError as e:self.fail(str(e))
        @staticmethod
        def stamp(m):return m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        def fail(self,reason):
            if self.fault:return
            self.fault=reason;self.fault_pub.publish(String(data=reason));self.status.publish(String(data=json.dumps({'fault':reason,'active':a.active})))
            self.trace.write(json.dumps({'fault':reason,'wall':time.monotonic(),'tick':self.tick})+'\n')
            self.get_logger().error('CPG latched: '+reason)
        def update(self):
            if self.fault:return
            now=time.monotonic()
            if a.active and self.mode!=1:return
            try:
                if a.active and now-self.mode_time>.1:raise ValueError('Stale driver mode')
                if len(self.latest)!=3:
                    if self.started or now-self.begin>15:raise ValueError('Missing raw sensors')
                    return
                stamps={k:(self.stamp(m),received) for k,(m,received) in self.latest.items()}
                self.gate.check(stamps,self.get_clock().now().nanoseconds*1e-9,now)
                phase_error=self.clock_gate.check(now,self.tick)
                if self.previous is not None and not .01<=now-self.previous<=.04:raise ValueError('Missed 50Hz control deadline')
                forces=calibration.apply(self.latest['contact'][0].foot_force_raw)
                sensor=decode_state(self.latest['joints'][0],self.latest['imu'][0],SimpleNamespace(foot_force_raw=forces),self.tick*.02)
                velocity=self.velocity if now-self.command_time<.25 else [0.,0.,0.]
                request={'sensor':sensor,'velocity':velocity,'telemetry':True,'request_stand':self.stand_requested}
                # Small packets; worker consumes exactly one per control tick.
                self.worker.stdin.write((json.dumps(request)+'\n').encode())
                reply=self.reader.read(.015)
                if 'fault' in reply:raise ValueError(reply['fault'])
                end=time.monotonic()
                if end-now>.020:raise ValueError('End-to-end control computation exceeded 20ms')
                # Timestamp the source sample, not the later publication time.
                stamp=self.latest['joints'][0].header.stamp
                self.pub.publish(encode_command(reply['target_sdk'],reply['kp_sdk'],reply['kd_sdk'],LegCommandArray,LegCommand,MotorCommand,stamp))
                self.trace.write(json.dumps(dict(tick=self.tick,wall=now,worker_ms=(end-now)*1000,clock_error_ms=phase_error*1000,source_stamps={k:v[0] for k,v in stamps.items()},sensor=sensor,velocity=velocity,active=a.active,**reply))+'\n')
                self.previous=now;self.started=True;self.tick+=1
                if self.tick%50==0:self.status.publish(String(data=json.dumps({'tick':self.tick,'active':a.active,'ready':reply['ready'],'standing':reply.get('standing',False),'stand_requested':reply.get('stand_requested',False)})))
            except Exception as e:self.fail(str(e))
        def close_worker(self):
            if self.worker.poll() is None:
                self.worker.terminate()
                try:self.worker.wait(2)
                except subprocess.TimeoutExpired:self.worker.kill();self.worker.wait()
    node=Bridge()
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    except RuntimeError:
        if rclpy.ok():raise
    finally:
        if a.active and rclpy.ok():node.fail('Adapter shutdown')
        node.close_worker();node.trace.close();node.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
