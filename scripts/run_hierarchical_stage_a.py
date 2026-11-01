"""Bounded Stage A pilot; persistent single GPU child, no automatic advancement."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--resume-checkpoint', type=Path)
parser.add_argument('--record-videos', action='store_true')
parser.add_argument('--num-envs', type=int, default=1024)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
PYTHON = '/home/rml2/anaconda3/envs/nick_sillywalk/bin/python'
STAMP = time.strftime('%Y%m%d_%H%M%S')
OUT = ROOT/'logs'/('hierarchical_stage_a_'+STAMP)
OUT.mkdir()
DEADLINE = time.time()+4*3600
RUN_NAME = 'stage_a_'+STAMP
state = dict(status='starting', stage='A', environments=args.num_envs, iterations=200,
             rollout_steps=24, seed=42, deadline=DEADLINE, output=str(OUT))
def update(**kw):
    state.update(kw)
    (OUT/'status.json').write_text(json.dumps(state, indent=2)+'\n')
def run(name, args):
    update(status='running', phase=name)
    with (OUT/(name+'.log')).open('w') as log:
        p = subprocess.Popen([PYTHON]+args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        update(child_pid=p.pid)
        try:
            rc = p.wait(timeout=max(1, DEADLINE-time.time()))
        except BaseException:
            os.killpg(p.pid, signal.SIGTERM)
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise
    if rc:
        raise RuntimeError(f'{name} exited {rc}; inspect its log')
try:
    snapshot = OUT/'source'; snapshot.mkdir()
    for f in [*ROOT.glob('source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/skills/*.py'),
              ROOT/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/hierarchical_env_cfg.py',
              ROOT/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/hierarchical_ppo_cfg.py',
              ROOT/'scripts/hierarchical_check.py', ROOT/'scripts/analyze_hierarchical_steps.py', Path(__file__)]:
        shutil.copy2(f, snapshot/f.name)
    run('preflight', ['scripts/hierarchical_preflight.py', '--headless', '--stage', 'A',
                     '--num_envs', '16', '--seed', '9040', '--output', str(OUT/'preflight')])
    preflight = json.loads((OUT/'preflight/result.json').read_text())
    assert preflight['status']=='passed', 'Stage A preflight failed'
    import torch
    start_index = 0
    resume_args = []
    if args.resume_checkpoint:
        source = args.resume_checkpoint.resolve()
        start_index = int(torch.load(source, map_location='cpu', weights_only=False)['iter'])
        resume_args = ['--resume', '--checkpoint_path', str(source)]
        update(resume_checkpoint=str(source), resume_index=start_index)
    final_index = start_index+199
    eval_indices = (50,100,199) if not args.resume_checkpoint else (
        ((start_index+49+49)//50)*50, ((start_index+99+49)//50)*50, final_index)
    update(final_index=final_index, evaluation_indices=list(eval_indices),
           evaluation_new_updates=[i-start_index+1 for i in eval_indices])
    run('training', ['scripts/rsl_rl/train.py' , '--headless', '--task', 'Unitree-Go2-Hierarchical-Step',
                     '--num_envs', str(args.num_envs), '--max_iterations', '200', '--seed', '42', '--run_name', RUN_NAME]+resume_args)
    dirs = list((ROOT/'logs/rsl_rl/go2_hierarchical').glob('*_'+RUN_NAME))
    assert len(dirs)==1, f'Unexpected training directories {dirs}'
    train = dirs[0]
    assert (train/f'model_{final_index}.pt').exists(), 'Final checkpoint absent; training may have failed'
    update(training_directory=str(train), training_complete=True)
    import torch
    checkpoint = torch.load(train/f'model_{final_index}.pt', map_location='cpu', weights_only=False)
    assert all(torch.isfinite(t).all() for t in checkpoint['model_state_dict'].values()), 'Nonfinite weights'
    results = {}
    for iteration in eval_indices:
        dest = OUT/f'eval_{iteration}'
        run(f'eval_{iteration}', ['scripts/hierarchical_check.py', '--headless', '--scenario', 'four-foot',
            '--num_envs', '32', '--steps', '1000', '--seed', '9041', '--checkpoint', str(train/f'model_{iteration}.pt'),
            '--output', str(dest)])
        assert json.loads((dest/'status.json').read_text())['status']=='passed', 'Evaluation failed'
        run(f'contact_analysis_{iteration}', ['scripts/analyze_hierarchical_steps.py', str(dest)])
        summary = json.loads((dest/'summary.json').read_text())
        by_pair = {}
        rows = [json.loads(line) for line in (dest/'trace.jsonl').read_text().splitlines()]
        for label, index in [('FL_RR',0), ('FR_RL',1)]:
            trials = [r for r in rows if r['event'] and r['kind']==0 and r['selected'][index]]
            by_pair[label] = dict(trials=len(trials), successes=sum(r['status']==1 for r in trials))
        summary['by_pair'] = by_pair
        results[str(iteration)] = summary
        (OUT/'evaluations.json').write_text(json.dumps(results, indent=2)+'\n')
    if args.record_videos:
        for foot in ('FL', 'FR'):
            dest = OUT/'videos'/foot
            run('video_'+foot, ['scripts/hierarchical_check.py', '--headless', '--enable_cameras',
                '--scenario', 'four-foot', '--num_envs', '1', '--steps', '500', '--seed', '9041',
                '--video-foot', foot, '--checkpoint', str(train/f'model_{final_index}.pt'), '--output', str(dest)])
            assert json.loads((dest/'status.json').read_text())['status']=='passed', 'Video failed'
    update(status='complete', phase='finished' , child_pid=None, final_checkpoint=str(train/f'model_{final_index}.pt'),
           stage_b_started=False, evaluations=str(OUT/'evaluations.json'))
except BaseException:
    update(status='stopped', error=traceback.format_exc(), child_pid=None)
    raise
