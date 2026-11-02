#!/usr/bin/env python3
"""Gazebo-only CPG sensor bridge. Ground truth is logged, never fed to the actor.

Requires robot_driver controller=cpg, is_hardware=false, and the portable
75-observation runtime. Commands go through robot_driver's READY gate.
"""
import argparse
import json
import math
from pathlib import Path
import select
import subprocess
import sys
import time
from types import SimpleNamespace



def toe_vertical_force(message, toe):
    """Gazebo contact wrench on the toe, independent of collision ordering."""
    force = 0.
    for contact in message.contacts:
        if f'toe{toe}::' in contact.collision1.name:
            force += sum(w.body_1_wrench.force.z for w in contact.wrenches)
        elif f'toe{toe}::' in contact.collision2.name:
            force += sum(w.body_2_wrench.force.z for w in contact.wrenches)
    if not math.isfinite(force):
        raise ValueError('Nonfinite contact force')
    return abs(force)


def contact_forces_at(contacts, now):
    # Gazebo emits no per-toe updates during flight; expire old forces.
    return [contacts[i][0] if -.01 <= now-contacts[i][1] <= .01 else 0.
            for i in range(4)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--conga-root', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--policy-python', required=True)
    p.add_argument('--robot-ns', default='/robot_1')
    p.add_argument('--trace', type=Path, required=True)
    args, ros_args = p.parse_known_args()
    sys.path.insert(0, str(args.conga_root))
    from deploy.cpg.quad_sdk import decode_state, encode_command
    from deploy.cpg.safety import CommandEnvelope
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState, Imu
    from geometry_msgs.msg import Twist
    from std_msgs.msg import UInt8, String, Bool
    from ros_gz_interfaces.msg import Contacts
    from quad_msgs.msg import LegCommandArray, LegCommand, MotorCommand, RobotState
    rclpy.init(args=ros_args)

    class Bridge(Node):
        def __init__(self):
            super().__init__('cpg_gazebo')
            if not self.get_parameter('use_sim_time').value:
                raise RuntimeError('Gazebo adapter requires use_sim_time=true')
            self.worker = subprocess.Popen(
                [args.policy_python, str(args.conga_root/'scripts/cpg_worker.py'),
                 '--selection', str(args.selection)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, text=True, bufsize=1)
            if not select.select([self.worker.stdout], [], [], 20)[0]:
                self.close_worker(); raise RuntimeError('Policy initialization timeout')
            if not json.loads(self.worker.stdout.readline()).get('ready'):
                self.close_worker(); raise RuntimeError('Policy initialization failed')
            self.envelope=CommandEnvelope(json.loads(args.selection.read_text()))
            self.limits = json.loads(args.selection.read_text()).get('joystick_limits', [.05,.04,.1])
            if len(self.limits)!=3 or not all(math.isfinite(x) and x>=0 for x in self.limits):
                self.close_worker(); raise ValueError('Invalid command limits')
            self.latest = {}; self.contacts = {}; self.truth = None
            self.mode = 0; self.velocity = [0.,0.,0.]; self.command_time = -math.inf
            self.stand_requested=False
            self.tick = 0; self.previous = None; self.started = False; self.fault = None
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            self.trace = args.trace.open('w', buffering=1)
            ns = args.robot_ns.rstrip('/')
            self.pub = self.create_publisher(LegCommandArray, ns+'/cpg/target', 1)
            self.status = self.create_publisher(String, ns+'/cpg/status', 1)
            self.create_subscription(JointState, ns+'/joint_states', lambda m:self.latest.update(joints=m), qos_profile_sensor_data)
            self.create_subscription(Imu, ns+'/imu', lambda m:self.latest.update(imu=m), qos_profile_sensor_data)
            for i in range(4):
                self.create_subscription(Contacts, ns+f'/gazebo/toe{i}_contact_states',
                    lambda m,i=i:self.contact(i,m), qos_profile_sensor_data)
            self.create_subscription(UInt8, ns+'/state/control_mode', self.mode_callback, 1)
            self.create_subscription(Bool,ns+'/cpg/stand',lambda m:setattr(self,'stand_requested',self.stand_requested or m.data),1)
            self.create_subscription(String,ns+'/cpg/fault',lambda m:self.fail(m.data),1)
            self.create_subscription(Twist, ns+'/cmd_vel', self.command, 1)
            self.create_subscription(RobotState, ns+'/state/ground_truth', lambda m:setattr(self,'truth',m), qos_profile_sensor_data)
            self.create_timer(.02, self.update)
            self.get_logger().info('75-input CPG ready; waiting for robot_driver READY (mode 1)')

        def mode_callback(self, msg):
            self.mode = msg.data
            if self.started and self.mode != 1 and not self.fault:
                self.fail('Left READY; restart CPG runtime before rearming')

        def now(self): return self.get_clock().now().nanoseconds*1e-9

        def command(self, msg):
            values = [msg.linear.x, msg.linear.y, msg.angular.z]
            if not all(math.isfinite(v) for v in values):
                self.fail('Nonfinite command'); return
            self.velocity = self.envelope.apply(values)
            self.command_time = self.now()

        def contact(self, i, msg):
            try:
                self.contacts[i] = (toe_vertical_force(msg, i), self.stamp(msg))
            except ValueError as error:
                self.fail(str(error))

        @staticmethod
        def stamp(msg): return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9

        def fail(self, reason):
            if self.fault: return
            self.fault = reason
            self.get_logger().error('CPG fault latched: '+reason)
            self.status.publish(String(data=json.dumps({'fault':reason})))
            self.trace.write(json.dumps({'time':self.now(),'fault':reason})+'\n')

        def update(self):
            if self.fault or self.mode != 1: return
            now = self.now()
            if len(self.latest)!=2 or len(self.contacts)!=4: return
            # Gazebo contact sensors publish events while touching; an airborne
            # foot can stop publishing. Expire each force after 10 ms, while
            # requiring a live contact stream from at least one support foot.
            stamps = [self.stamp(m) for m in self.latest.values()]+[max(x[1] for x in self.contacts.values())]
            if any(not -.01 <= now-s <= .05 for s in stamps) or max(stamps)-min(stamps)>.025:
                if self.started: self.fail('Stale or unsynchronized Gazebo sensors: '+str(dict(zip(['joints','imu','latest_contact'],[now-s for s in stamps]))))
                return
            if self.previous is not None and not .01 <= now-self.previous <= .04:
                self.fail('Missed simulation control tick'); return
            try:
                force = SimpleNamespace(foot_force_raw=contact_forces_at(self.contacts, now))
                sensor = decode_state(self.latest['joints'], self.latest['imu'], force, self.tick*.02)
                velocity = self.velocity if now-self.command_time < .25 else [0.,0.,0.]
                request = {'sensor':sensor, 'velocity':velocity, 'telemetry':True,'request_stand':self.stand_requested}
                start = time.monotonic()
                self.worker.stdin.write(json.dumps(request)+'\n'); self.worker.stdin.flush()
                if not select.select([self.worker.stdout],[],[],.05)[0]:
                    raise RuntimeError('Policy worker exceeded 50 ms wall deadline (simulation)')
                reply = json.loads(self.worker.stdout.readline())
                if 'fault' in reply: raise RuntimeError(reply['fault'])
                self.pub.publish(encode_command(reply['target_sdk'],reply['kp_sdk'],reply['kd_sdk'],
                    LegCommandArray,LegCommand,MotorCommand,self.get_clock().now().to_msg()))
                row = dict(time=now, tick=self.tick, sensor=sensor, velocity=velocity,
                           contact_age=[now-self.contacts[i][1] for i in range(4)],
                           worker_ms=1000*(time.monotonic()-start), **reply)
                if self.truth:
                    b=self.truth.body
                    row['truth_position']=[b.pose.position.x,b.pose.position.y,b.pose.position.z]
                    row['truth_velocity']=[b.twist.linear.x,b.twist.linear.y,b.twist.linear.z]
                    row['truth_stamp']=self.stamp(self.truth)
                    row['truth_feet']=[[f.position.x,f.position.y,f.position.z] for f in self.truth.feet.feet]
                    row['truth_foot_velocities']=[[f.velocity.x,f.velocity.y,f.velocity.z] for f in self.truth.feet.feet]
                self.trace.write(json.dumps(row)+'\n')
                self.tick+=1; self.previous=now; self.started=True
                if self.tick%50==0:
                    self.status.publish(String(data=json.dumps({'ticks':self.tick,'ready':reply['ready'],'dance_time':reply['dance_time'],'standing':reply.get('standing',False),'stand_requested':reply.get('stand_requested',False)})))
            except Exception as e: self.fail(str(e))

        def close_worker(self):
            if self.worker.poll() is None:
                self.worker.terminate()
                try: self.worker.wait(timeout=2)
                except subprocess.TimeoutExpired: self.worker.kill(); self.worker.wait()

    node=Bridge()
    try: rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException): pass
    finally: node.close_worker(); node.trace.close(); node.destroy_node(); rclpy.try_shutdown()


if __name__=='__main__': main()
