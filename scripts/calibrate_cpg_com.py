"""Compute nominal whole-body CoM from the exact local Go2 URDF; no simulation."""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--urdf', type=Path, default=Path('.local/unitree_ros/robots/go2_description/urdf/go2_description.urdf'))
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = ET.parse(a.urdf).getroot()
links = {x.get('name'): x for x in root.findall('link')}
joints = root.findall('joint')
children = {j.find('child').get('link') for j in joints}
roots = set(links)-children
assert len(roots) == 1
base = roots.pop()
poses = {base: np.eye(4)}


def origin(element):
    result = np.eye(4)
    if element is None:
        return result
    result[:3, 3] = list(map(float, element.get('xyz', '0 0 0').split()))
    r, p, y = map(float, element.get('rpy', '0 0 0').split())
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    result[:3, :3] = rz @ ry @ rx
    return result


angle = math.acos(.30/.426)
for _ in links:
    for joint in joints:
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        if parent not in poses or child in poses:
            continue
        name = joint.get('name')
        q = angle if '_thigh_joint' in name else -2*angle if '_calf_joint' in name else 0.
        rotation = np.eye(4)
        if q:
            axis = np.array(list(map(float, joint.find('axis').get('xyz').split())))
            skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            rotation[:3, :3] += np.sin(q)*skew+(1-np.cos(q))*(skew@skew)
        poses[child] = poses[parent] @ origin(joint.find('origin')) @ rotation
assert set(poses) == set(links), 'URDF link tree is incomplete'
mass = 0.
moment = np.zeros(3)
for name, link in links.items():
    inertial = link.find('inertial')
    if inertial is None:
        continue
    value = float(inertial.find('mass').get('value'))
    position = (poses[name] @ origin(inertial.find('origin')))[:3, 3]
    mass += value
    moment += value*position
result = dict(urdf=str(a.urdf.resolve()), base_link=base, mass_kg=mass,
              nominal_leg_height_m=.30, hip_angle_rad=0., thigh_angle_rad=angle,
              calf_angle_rad=-2*angle, com_body_m=(moment/mass).tolist())
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
