"""Bounded JSON-lines process boundary between ROS Python and policy Python.

stdin contains SensorFrame dictionaries plus velocity. stdout contains ready,
position targets/gains, or a latched fault. No sockets or hardware transport.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from deploy.cpg.runtime import DanceRuntime,SensorFrame,load_selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    selection=load_selection(args.selection)
    runtime=DanceRuntime(selection['deploy_metadata'],selection['onnx_export'])
    kp,kd=np.empty(12),np.empty(12)
    kp[runtime.map],kd[runtime.map]=runtime.kp,runtime.kd
    print(json.dumps({'ready':True,'dt':runtime.dt}),flush=True)
    for line in sys.stdin:
        try:
            if len(line)>65536:raise ValueError('Oversized sensor request')
            packet=json.loads(line)
            sensor=packet['sensor']
            frame=SensorFrame(time=float(sensor['time']),**{k:np.asarray(sensor[k],dtype=float) for k in ('q','qd','quaternion','gyro','foot_force')})
            with torch.inference_mode():
                observation=runtime.observe(frame,packet['velocity'])
                target=runtime.act(observation)
            reply={'target_sdk':target.tolist(),'kp_sdk':kp.tolist(),'kd_sdk':kd.tolist(),
                   'dance_time':runtime.controller.time.item(),'ready':runtime.controller.ready.item()}
            if packet.get('telemetry'):
                c=runtime.controller
                reply['metrics']={k:v.detach().cpu().tolist() for k,v in c.metrics.items()}
                reply['expected_contact']=c.expected[0].tolist()
                reply['measured_contact']=c.events.contact[0].tolist()
                reply['records']={k:v.detach().cpu().tolist() for k,v in c.last_records.items()}
                if hasattr(c,'dance'):
                    reply['dance_kind']=c.dance.kind.item()
                    reply['kick_target_time']=c.dance.target_time.item()
                    reply['kick_success']=c.kick_success.item()
                    reply['kick_failure']=c.kick_failure.item()
            print(json.dumps(reply),flush=True)
        except Exception as error:
            print(json.dumps({'fault':str(error)}),flush=True)
            return 1
    return 0


if __name__=='__main__':sys.exit(main())
