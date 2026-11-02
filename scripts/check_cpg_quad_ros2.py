"""Local ROS 2 replay test using real Quad-SDK message types and the policy worker."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deploy.cpg.quad_sdk import QUAD_TO_SDK,QUAD_NAMES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--selection',type=Path)
    args=parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='77' or os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE')!='LOCALHOST':
        raise RuntimeError('Replay must use isolated domain 77 and LOCALHOST discovery')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState,Imu
    from geometry_msgs.msg import Twist
    from quad_msgs.msg import FootContact,LegCommandArray
    packets=json.loads((args.directory/'ros_replay_packets.json').read_text())
    rclpy.init()
    node=Node('cpg_adapter_replay')
    pubs=[node.create_publisher(kind,'/cpg_replay/'+topic,qos_profile_sensor_data) for kind,topic in
          [(JointState,'state/joints'),(Imu,'state/imu'),(FootContact,'state/foot_contact')]]
    cmd=node.create_publisher(Twist,'/cpg_replay/cmd_vel',1)
    replies=[]
    node.create_subscription(LegCommandArray,'/cpg_sim/replay_commands',replies.append,1)
    log=open(args.directory/'ros_bridge.log','w')
    bridge_args=[sys.executable,str(ROOT/'scripts/cpg_quad_ros2.py'),'--robot-ns','/cpg_replay',
                 '--output-topic','/cpg_sim/replay_commands']
    if args.selection:bridge_args.extend(['--selection',str(args.selection.resolve())])
    bridge=subprocess.Popen(bridge_args,stdout=log,stderr=subprocess.STDOUT)
    try:
        for i,packet in enumerate(packets):
            s=packet['sensor'];deadline=time.monotonic()+15 if i==0 else time.monotonic()+1
            before=len(replies)
            while len(replies)==before:
                if bridge.poll() is not None:raise RuntimeError('ROS bridge exited; see ros_bridge.log')
                if time.monotonic()>deadline:raise RuntimeError(f'No ROS controller output for packet {i}')
                stamp=node.get_clock().now().to_msg()
                joints=JointState();joints.header.stamp=stamp;joints.name=list(QUAD_NAMES)
                joints.position=[s['q'][k] for k in QUAD_TO_SDK];joints.velocity=[s['qd'][k] for k in QUAD_TO_SDK]
                imu=Imu();imu.header.stamp=stamp
                imu.orientation.w,imu.orientation.x,imu.orientation.y,imu.orientation.z=s['quaternion']
                imu.angular_velocity.x,imu.angular_velocity.y,imu.angular_velocity.z=s['gyro']
                contact=FootContact();contact.header.stamp=stamp
                contact.foot_force_raw=[round(s['foot_force'][k]) for k in (0,2,1,3)]
                contact.contact_states=[v>5 for v in contact.foot_force_raw]
                for publisher,message in zip(pubs,(joints,imu,contact)):publisher.publish(message)
                velocity=Twist();velocity.linear.x,velocity.linear.y,velocity.angular.z=packet['velocity'];cmd.publish(velocity)
                rclpy.spin_once(node,timeout_sec=.005)
            reply=replies[-1]
            assert len(reply.leg_commands)==4
            for leg in reply.leg_commands:
                assert len(leg.motor_commands)==3
                for motor in leg.motor_commands:
                    assert math.isfinite(motor.pos_setpoint) and -3<motor.pos_setpoint<5
                    assert motor.kp==25 and motor.kd==.5 and motor.torque_ff==0
        # A lost sensor stream must latch the adapter off; fresh messages alone
        # cannot silently restart it. This is distinct from neutral joystick.
        end=time.monotonic()+.2
        while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.01)
        stopped=len(replies)
        for _ in range(10):
            for publisher,message in zip(pubs,(joints,imu,contact)):publisher.publish(message)
            rclpy.spin_once(node,timeout_sec=.01)
        assert len(replies)==stopped,'Sensor loss failed to latch output off'
        result={'status':'passed','replayed_sensor_packets':len(packets),'command_messages':len(replies),
                'actual_quad_msgs':True,'stale_sensor_latch':True,'hardware_connected':False,
                'dds_domain':77,'discovery':'LOCALHOST'}
    finally:
        bridge.send_signal(signal.SIGINT)
        try:bridge.wait(timeout=4)
        except subprocess.TimeoutExpired:bridge.kill();bridge.wait()
        log.close();node.destroy_node();rclpy.try_shutdown()
    assert bridge.returncode==0,f'ROS bridge exited with {bridge.returncode}'
    result['bridge_exit_code']=bridge.returncode
    (args.directory/'ros_replay_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
