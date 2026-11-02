#!/usr/bin/env python3
"""Draft contact-evidence calibration from read-only unloaded/standing recordings.

This calibrates CONTACT CLASSIFICATION, not Newtons or actual support load.
All hardware enable/verification flags remain false for operator review.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def load(path):
 rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
 if len(rows)<200:raise ValueError('Need >=200 source samples per recording')
 ticks=np.array([r['tick'] for r in rows],dtype=np.int64)
 if ((ticks<0)|(ticks>0xffffffff)).any():raise ValueError('Invalid MCU tick')
 delta=np.diff(ticks)&0xffffffff
 if (delta>=0x80000000).any():raise ValueError('Regressing MCU tick; do not combine recordings across reboot')
 keep=np.r_[True,delta!=0];rows=[r for r,k in zip(rows,keep) if k]
 if len(rows)<200:raise ValueError('Too few advancing MCU samples')
 forces=np.array([r['foot_force_sdk'] for r in rows],dtype=float)[:,[1,3,0,2]]
 if forces.shape[1]!=4 or not np.isfinite(forces).all():raise ValueError('Invalid forces')
 return rows,forces

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--air',type=Path,required=True);p.add_argument('--standing',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--policy-sha256',required=True);a=p.parse_args()
 if a.output.exists():p.error('Output exists')
 air,fa=load(a.air);stand,fs=load(a.standing);offset=np.percentile(fa,99,axis=0);loaded=np.percentile(fs,10,axis=0);gap=loaded-offset
 if (gap<5).any():raise ValueError('Insufficient unloaded/loaded separation; inspect recordings and sensors')
 scale=20/gap
 profile=dict(robot_type='go2',hardware_enabled=False,contact_calibrated=False,timing_verified=False,policy_sha256=a.policy_sha256,force_calibration=dict(order='FL,RL,FR,RR',measured=True,units='contact_evidence',method='unloaded_p99_to_loaded_p10_20_units_v1',offset=offset.tolist(),scale=scale.tolist()),notes='DRAFT: measured counts mapped to contact evidence, not N. Verify each actual airborne/planted foot independently before setting contact_calibrated. Timing and hardware enable remain false.',calibration_evidence=dict(air=str(a.air.resolve()),standing=str(a.standing.resolve()),air_sha256=hashlib.sha256(a.air.read_bytes()).hexdigest(),standing_sha256=hashlib.sha256(a.standing.read_bytes()).hexdigest(),samples=[len(air),len(stand)],raw_on_threshold=(offset+5/scale).tolist(),raw_off_threshold=(offset+2/scale).tolist(),standing_gyro_mean=np.mean([r['gyro'] for r in stand],axis=0).tolist()))
 a.output.write_text(json.dumps(profile,indent=2)+'\n');print(json.dumps(profile,indent=2))
if __name__=='__main__':main()
