"""Gated 1000-update Stage A experiment. Checkpoints are copied, never overwritten."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/rml2/anaconda3/envs/nick_sillywalk/bin/python'
BASELINE=ROOT/'logs/rsl_rl/go2_hierarchical/2026-10-29_15-49-35_stage_a_20261029_154924/model_199.pt'


def gate_reason(current, sampled, history, updates):
    if current['falls']/max(1,current['trials'])>.10:
        return 'Deterministic fall rate exceeds 10%'
    if updates>=100 and history:
        prior=history[-1]
        if prior['success_rate']>=.2 and current['success_rate']<prior['success_rate']-.2:
            return 'Deterministic success dropped more than 20 percentage points'
    if updates>=200 and history and not current['any_lift_trials'] and not history[-1]['any_lift_trials'] and not sampled['any_lift_trials']:
        return 'No verified lift at two deterministic gates or current stochastic probe'
    if updates>=500 and not current['successes'] and not sampled['successes'] and current['paired_lift_trials']/max(1,current['trials'])<.05:
        return 'No completion in either probe and paired lifting below 5% at 500+ updates'
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    import torch
    stamp=time.strftime('%Y%m%d_%H%M%S')
    out=ROOT/'logs'/('hierarchical_long_'+('smoke_' if args.smoke else '')+stamp)
    out.mkdir()
    deadline=time.time()+4*3600
    gates=(1,2) if args.smoke else (50,100,200,500,1000)
    environments=16 if args.smoke else 4096
    eval_envs=8 if args.smoke else 32
    eval_steps=250 if args.smoke else 1000
    state=dict(status='starting',output=str(out),budget=gates[-1],completed_updates=0,
               environments=environments,seed=42,deadline=deadline,baseline=str(BASELINE),
               gates=list(gates),stage_b_started=False,smoke=args.smoke)
    def update(**kw):
        state.update(kw)
        temp=out/'status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(out/'status.json')
    def run(name, argv):
        update(status='running',phase=name)
        with (out/(name+'.log')).open('w') as log:
            p=subprocess.Popen([PYTHON]+argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            update(child_pid=p.pid)
            try: rc=p.wait(timeout=max(1,deadline-time.time()))
            except BaseException:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL);p.wait()
                raise
        update(child_pid=None)
        if rc:raise RuntimeError(f'{name} exited {rc}')
    def evaluate(checkpoint, label, stochastic=False):
        dest=out/label
        flags=['--stochastic'] if stochastic else []
        run(label,['scripts/hierarchical_check.py','--headless','--scenario','four-foot',
            '--num_envs',str(eval_envs),'--steps',str(eval_steps),'--seed','9041',
            '--checkpoint',str(checkpoint),'--audit-rewards','--output',str(dest)]+flags)
        assert json.loads((dest/'status.json').read_text())['status']=='passed'
        run(label+'_contacts',['scripts/analyze_hierarchical_steps.py',str(dest)])
        rows=[json.loads(x) for x in (dest/'trace.jsonl').read_text().splitlines()]
        events=[r for r in rows if r['event'] and r['kind']==0]
        summary=json.loads((dest/'summary.json').read_text())
        result=dict(trials=len(events),successes=sum(r['status']==1 for r in events),falls=summary['falls'],
            any_lift_trials=sum(any(r['lifted']) for r in events),
            paired_lift_trials=sum(all(l or not s for l,s in zip(r['lifted'],r['selected'])) for r in events),
            mean_slip=summary['groups']['STEP/four-foot']['mean_slip'],by_pair={})
        result['success_rate']=result['successes']/max(1,result['trials'])
        for label,index in [('FL_RR',0),('FR_RL',1)]:
            pair=[r for r in events if r['selected'][index]]
            result['by_pair'][label]=dict(trials=len(pair),successes=sum(r['status']==1 for r in pair))
        if not args.smoke:assert result['trials']>0,'No completed evaluation trials'
        return result
    try:
        snapshot=out/'source';snapshot.mkdir()
        for f in [*ROOT.glob('source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/skills/*.py'),
                  ROOT/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/hierarchical_env_cfg.py',
                  ROOT/'scripts/rsl_rl/train.py',ROOT/'scripts/hierarchical_check.py',
                  ROOT/'scripts/analyze_hierarchical_steps.py',Path(__file__)]:
            shutil.copy2(f,snapshot/f.name)
        shutil.copy2(ROOT/'logs/hierarchical_long_calibration/scores.json',out/'calibration.json')
        run('preflight',['scripts/hierarchical_preflight.py','--headless','--stage','A','--num_envs','16',
            '--seed','9040','--output',str(out/'preflight')])
        assert json.loads((out/'preflight/result.json').read_text())['status']=='passed'
        checkpoint=BASELINE
        history=[evaluate(checkpoint,'eval_baseline')]
        completed=0
        results={}
        reason=None
        for gate in gates:
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            # Saved iter denotes the last executed update. Resume at the next
            # index to avoid replay, while preserving optimizer and policy state.
            start=int(saved['iter'])+1
            saved['iter']=start
            learning_rate=float(saved['optimizer_state_dict']['param_groups'][0]['lr'])
            assert all(torch.isfinite(t).all() for t in saved['model_state_dict'].values())
            resume=out/f'resume_{completed}.pt';torch.save(saved,resume)
            updates=gate-completed
            name=f'long_{stamp}_{gate}'
            update(segment_updates=updates,initial_learning_rate=learning_rate,resume_checkpoint=str(checkpoint))
            run(f'train_{gate}',['scripts/rsl_rl/train.py','--headless','--task','Unitree-Go2-Hierarchical-Step',
                '--num_envs',str(environments),'--max_iterations',str(updates),'--seed','42',
                '--run_name',name,'--resume','--checkpoint_path',str(resume),
                f'agent.algorithm.learning_rate={learning_rate}'])
            dirs=list((ROOT/'logs/rsl_rl/go2_hierarchical').glob('*_'+name))
            assert len(dirs)==1
            checkpoint=dirs[0]/f'model_{start+updates-1}.pt'
            assert checkpoint.exists(),'Training did not produce its final checkpoint'
            final=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert all(torch.isfinite(t).all() for t in final['model_state_dict'].values())
            completed=gate
            update(completed_updates=completed,latest_checkpoint=str(checkpoint))
            current=evaluate(checkpoint,f'eval_{gate}')
            sampled=None
            if (args.smoke and gate==2) or (gate>=200 and (current['paired_lift_trials']/max(1,current['trials'])<.05 or not current['successes'])):
                sampled=evaluate(checkpoint,f'eval_{gate}_stochastic',True)
            reason=None if args.smoke else gate_reason(current,sampled or current,history,gate)
            history.append(current)
            results[str(gate)]=dict(deterministic=current,stochastic=sampled,review_reason=reason)
            (out/'evaluations.json').write_text(json.dumps(results,indent=2)+'\n')
            if reason and gate<gates[-1]:break
        if not args.smoke:
            for foot in ['FL','FR']:
                dest=out/'videos'/foot
                run('video_'+foot,['scripts/hierarchical_check.py','--headless','--enable_cameras',
                    '--scenario','four-foot','--num_envs','1','--steps','500','--video-foot',foot,
                    '--checkpoint',str(checkpoint),'--output',str(dest)])
                assert json.loads((dest/'status.json').read_text())['status']=='passed'
        update(status='review_stop' if reason and completed<gates[-1] else 'complete',phase='finished',
               review_reason=reason,latest_checkpoint=str(checkpoint),completed_updates=completed)
    except BaseException:
        update(status='stopped',error=traceback.format_exc(),child_pid=None)
        raise


if __name__=='__main__':main()
