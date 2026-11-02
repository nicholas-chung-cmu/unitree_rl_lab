#!/usr/bin/env python3
"""Quad-SDK ROS 2 message adapter for simulation/shadow evaluation.

Output is restricted to /cpg_sim/. This does not activate a robot_driver mode,
publish Unitree commands, or disable the robot's existing controller.
"""
import argparse
import json
import math
from pathlib import Path
import select
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deploy.cpg.quad_sdk import decode_state,encode_command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-python',default='/home/rml2/anaconda3/envs/nick_sillywalk/bin/python')
    parser.add_argument('--selection',type=Path,default=ROOT/'logs/conga_cpg_session_20261101/selected/dance_bounce.json')
    parser.add_argument('--robot-ns',default='/robot_1')
    parser.add_argument('--output-topic',default='/cpg_sim/leg_command_array')
    parser.add_argument('--force-scale',type=float,default=1.)
    parser.add_argument('--force-offset',type=float,default=0.)
    parser.add_argument('--command-limits',type=float,nargs=3)
    args,ros_args=parser.parse_known_args()
    if args.command_limits is None:args.command_limits=json.loads(args.selection.read_text()).get('joystick_limits',[.05,.04,.1])
    if not all(math.isfinite(x) and x>=0 for x in args.command_limits):parser.error('Invalid command limits')
    if not args.output_topic.startswith('/cpg_sim/'):
        parser.error('This qualification adapter only publishes under /cpg_sim/')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState,Imu
    from geometry_msgs.msg import Twist
    from quad_msgs.msg import FootContact,LegCommandArray,LegCommand,MotorCommand
    rclpy.init(args=ros_args)

    class Bridge(Node):
        def __init__(self):
            super().__init__('go2_cpg_sim_adapter')
            self.worker=subprocess.Popen([args.policy_python,str(ROOT/'scripts/cpg_worker.py'),'--selection',str(args.selection)],
                                         stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
            self.latest={};self.velocity=[0.,0.,0.];self.velocity_time=0.
            self.tick=0;self.fault=None;self.started=False;self.previous_wall=None
            if not select.select([self.worker.stdout],[],[],20)[0] or not json.loads(self.worker.stdout.readline()).get('ready'):
                self.close_worker();raise RuntimeError('CPG worker did not initialize')
            self.pub=self.create_publisher(LegCommandArray,args.output_topic,1)
            for name,kind,topic in [('joints',JointState,'state/joints'),('imu',Imu,'state/imu'),('contact',FootContact,'state/foot_contact')]:
                self.create_subscription(kind,args.robot_ns.rstrip('/')+'/'+topic,
                    lambda message,key=name:self.latest.update({key:(message,time.monotonic())}),qos_profile_sensor_data)
            self.create_subscription(Twist,args.robot_ns.rstrip('/')+'/cmd_vel',self.command,1)
            self.create_timer(.02,self.update)
            self.get_logger().info('CPG shadow output: '+args.output_topic+'; no hardware controller activated')

        def command(self,message):
            velocity=[message.linear.x,message.linear.y,message.angular.z]
            if not all(math.isfinite(v) for v in velocity):
                self.fail('Nonfinite velocity command');return
            self.velocity=velocity
            self.velocity_time=time.monotonic()

        def update(self):
            if self.fault:return
            now=time.monotonic()
            if len(self.latest)<3:return
            if any(now-received>.05 for _,received in self.latest.values()):
                if self.started:self.fail('Stale sensor stream')
                return
            if self.previous_wall is not None and now-self.previous_wall>.04:
                self.fail('Missed control deadline');return
            stamps=[m.header.stamp.sec+m.header.stamp.nanosec*1e-9 for m,_ in self.latest.values()]
            ros_now=self.get_clock().now().nanoseconds*1e-9
            if any(not -.01 <= ros_now-stamp <= .05 for stamp in stamps):
                self.fail('Stale or future sensor timestamp');return
            if max(stamps)-min(stamps)>.01:
                self.fail('Unsynchronized joint/IMU/contact messages');return
            try:
                sensor=decode_state(*(self.latest[k][0] for k in ('joints','imu','contact')),self.tick*.02,args.force_scale,args.force_offset)
                velocity=self.velocity if now-self.velocity_time<.25 else [0.,0.,0.]
                # Same bounded commands as the tested joystick path.
                velocity=[max(-limit,min(limit,float(v))) for v,limit in zip(velocity,args.command_limits)]
                self.worker.stdin.write(json.dumps({'sensor':sensor,'velocity':velocity})+'\n');self.worker.stdin.flush()
                if not select.select([self.worker.stdout],[],[],.02)[0]:
                    raise RuntimeError('Policy worker missed 20 ms deadline')
                reply=json.loads(self.worker.stdout.readline())
                if 'fault' in reply:raise RuntimeError(reply['fault'])
                command=encode_command(reply['target_sdk'],reply['kp_sdk'],reply['kd_sdk'],LegCommandArray,LegCommand,MotorCommand,self.get_clock().now().to_msg())
                self.pub.publish(command)
                self.tick+=1;self.started=True;self.previous_wall=now
            except Exception as error:self.fail(str(error))

        def fail(self,reason):
            self.fault=reason
            self.get_logger().error('CPG output stopped and fault latched: '+reason)

        def close_worker(self):
            if self.worker.poll() is None:
                self.worker.terminate()
                try:self.worker.wait(timeout=2)
                except subprocess.TimeoutExpired:self.worker.kill();self.worker.wait()

    node=Bridge()
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:node.close_worker();node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
