#!/usr/bin/env python3
"""Read local Unitree wireless snapshots and publish restricted ROS cmd_vel.

No automatic stand/arming. B latches a CPG fault and requests driver SAFETY.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deploy.cpg.inputs import WirelessInput
from deploy.cpg.safety import CommandEnvelope

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--robot-ns',default='/robot_1');a,ros=p.parse_known_args()
 sel=json.loads(a.selection.read_text());envelope=CommandEnvelope(sel);wireless=WirelessInput(a.input,limits=sel['joystick_limits'])
 import rclpy
 from rclpy.node import Node
 from geometry_msgs.msg import Twist
 from std_msgs.msg import UInt8,String,Bool
 rclpy.init(args=ros);node=Node('cpg_wireless_commands');ns=a.robot_ns.rstrip('/')
 stand=node.create_publisher(Bool,ns+'/cpg/stand',1)
 cmd=node.create_publisher(Twist,ns+'/cmd_vel',1);mode=node.create_publisher(UInt8,ns+'/control/mode',1);fault=node.create_publisher(String,ns+'/cpg/fault',1)
 def tick():
  values=envelope.apply(wireless.read().tolist())
  if wireless.stand:stand.publish(Bool(data=True))
  if wireless.stop:
   values=[0.,0.,0.];fault.publish(String(data='Unitree remote B pressed'));mode.publish(UInt8(data=4))
  message=Twist();message.linear.x,message.linear.y,message.angular.z=values;cmd.publish(message)
 node.create_timer(.02,tick)
 try:rclpy.spin(node)
 except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
 finally:
  if rclpy.ok():cmd.publish(Twist())
  node.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
