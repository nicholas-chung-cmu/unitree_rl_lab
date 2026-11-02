#!/usr/bin/env python3
"""Read-only host/policy preflight. Does not initialize ROS or Unitree DDS."""
import argparse,json,os,platform,subprocess,sys,time,hashlib,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deploy.cpg.safety import JsonLineReader

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True);p.add_argument('--policy-python',default=sys.executable);p.add_argument('--output',type=Path,required=True);p.add_argument('--packets',type=Path);a=p.parse_args()
 if a.output.exists():p.error('Output exists')
 report={'hardware_access':False,'machine':platform.machine(),'platform':platform.platform(),'python':sys.version,'ros_distro':os.environ.get('ROS_DISTRO'),'os_release':Path('/etc/os-release').read_text() if Path('/etc/os-release').exists() else None,'selection':str(a.selection.resolve())}
 process=subprocess.Popen([a.policy_python,str(ROOT/'scripts/cpg_worker.py'),'--selection',str(a.selection)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)
 try:
  reader=JsonLineReader(process.stdout);start=time.monotonic();ready=reader.read(45);report['worker_start_seconds']=time.monotonic()-start
  if ready.get('ready') is not True or ready.get('dt')!=.02:raise ValueError('Invalid worker handshake')
  report['worker_ready']=True
  if a.packets:
   packets=json.loads(a.packets.read_text());durations=[]
   for packet in packets:
    start=time.monotonic();process.stdin.write((json.dumps(packet)+'\n').encode());reply=reader.read(.5);durations.append((time.monotonic()-start)*1000)
    if 'fault' in reply:raise ValueError(reply['fault'])
   durations.sort();report['benchmark']={'packets':len(durations),'median_ms':statistics.median(durations),'p99_ms':durations[min(len(durations)-1,int(len(durations)*.99))],'max_ms':max(durations),'all_under_15ms':max(durations)<15,'description':'Offline worker replay on this host, not full ROS/motor timing qualification'}
   if max(durations)>=15:raise ValueError('Offline worker exceeded the 15ms response budget; onboard timing is not verified')
  report['status']='passed'
 except Exception as e:report.update(status='failed',error=str(e))
 finally:
  process.terminate()
  try:process.wait(3)
  except subprocess.TimeoutExpired:process.kill();process.wait()
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0 if report['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
