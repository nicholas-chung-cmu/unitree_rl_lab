"""Supervise the authorized 1024-env early-shaping experiment and evaluation gates.

Run only with explicit training authorization. Signals only child process groups
created here. GPU samples are whole-device measurements, not process attribution.
"""
import json
import os
import pickle
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
KICK_REFINE = '--kick-refine' in sys.argv
KICK = '--kick-intro' in sys.argv or KICK_REFINE
KICK_CONTINUE = '--kick-continue' in sys.argv
HISTORY = '--history-finetune' in sys.argv
PAUSE = '--pause-finetune' in sys.argv
MARCH = '--march-first' in sys.argv
AVERAGE = '--duration-average' in sys.argv
STRONG = '--duration-strong' in sys.argv
DURATION = '--duration-walking' in sys.argv
BALANCED = '--march-balanced' in sys.argv
if sum((HISTORY,PAUSE,MARCH,BALANCED,DURATION,STRONG,AVERAGE,KICK))>1:
    raise ValueError('Choose only one experiment mode')
TASK = 'Unitree-Go2-Conga-History' if HISTORY else 'Unitree-Go2-Conga-Early'
PREFIX = 'history_1024' if HISTORY else 'early_1024'
START = 1000 if HISTORY else 0
GATES = (1050,1100,1200) if HISTORY else (200,500,1000)
INITIAL = ROOT / 'logs/rsl_rl/go2_conga_early/2026-10-28_00-56-55_early_1024_to1000/model_999.pt' if HISTORY else None
OUT = ROOT / ('logs/conga_history_1024_experiment' if HISTORY else 'logs/conga_early_1024_experiment')
RUNS = ROOT / ('logs/rsl_rl/go2_conga_history' if HISTORY else 'logs/rsl_rl/go2_conga_early')
NUM_ENVS = 1024
if PAUSE:
    TASK = 'Unitree-Go2-Conga-Pause'
    PREFIX = 'pause_4096'
    NUM_ENVS = 4096
    START = 1200
    GATES = (1250,1300,1400,1700,2200)
    INITIAL = ROOT / 'logs/rsl_rl/go2_conga_history/2026-10-28_01-43-59_history_1024_to1200_retry1793166235728426334/model_1199.pt'
    OUT = ROOT / 'logs/conga_pause_4096_experiment'
    RUNS = ROOT / 'logs/rsl_rl/go2_conga_pause'
if MARCH or BALANCED or DURATION or STRONG or AVERAGE:
    TASK = 'Unitree-Go2-Conga-March'
    PREFIX = 'march_4096'
    NUM_ENVS = 4096
    START = 0
    GATES = (50,100,200,500,1000)
    INITIAL = None
    OUT = ROOT / 'logs/conga_march_4096_experiment'
    RUNS = ROOT / 'logs/rsl_rl/go2_conga_march'
if BALANCED:
    TASK = 'Unitree-Go2-Conga-March-Balanced'
    PREFIX = 'march_balanced_4096'
    OUT = ROOT / 'logs/conga_march_balanced_4096_experiment'
    RUNS = ROOT / 'logs/rsl_rl/go2_conga_march_balanced'
if DURATION:
    TASK = 'Unitree-Go2-Conga-Duration'
    PREFIX = 'duration_4096'
    OUT = ROOT / 'logs/conga_duration_4096_experiment'
    RUNS = ROOT / 'logs/rsl_rl/go2_conga_duration'
if STRONG:
    TASK = 'Unitree-Go2-Conga-Duration-Strong'
    PREFIX = 'duration_strong_4096'
    OUT = ROOT / 'logs/conga_duration_strong_4096_experiment'
    RUNS = ROOT / 'logs/rsl_rl/go2_conga_duration_strong'
if AVERAGE:
    TASK = 'Unitree-Go2-Conga-Duration-Average'
    PREFIX = 'duration_average_4096'
    OUT = ROOT / 'logs/conga_duration_average_4096_experiment'
    RUNS = ROOT / 'logs/rsl_rl/go2_conga_duration_average'
if KICK:
    TASK = 'Unitree-Go2-Conga-Kick-Refine' if KICK_REFINE else 'Unitree-Go2-Conga-Kick-Intro'
    PREFIX = 'kick_intro_4096'
    NUM_ENVS = 4096
    START = 1000
    GATES = (2000,) if KICK_CONTINUE else (1050,1100,1200,1500)
    INITIAL = ROOT / 'logs/rsl_rl/go2_conga_duration_strong/2026-10-28_14-47-07_duration_strong_4096_to1000_retry1793213223697938773/model_999.pt'
    OUT = ROOT / 'logs/conga_kick_intro_4096_experiment'
    RUNS = ROOT / 'logs/rsl_rl/go2_conga_kick_intro'
PYTHON = '/home/rml2/anaconda3/envs/nick_sillywalk/bin/python'
OUT.mkdir(parents=True, exist_ok=True)
STATE = {'status': 'starting', 'gates': [], 'pid': os.getpid(), 'starting_iteration':START, 'final_iteration':GATES[-1], 'num_envs':NUM_ENVS}
if '--resume-state' in sys.argv:
    STATE.update(json.loads((OUT / 'status.json').read_text()))
    STATE['pid'] = os.getpid()
    STATE['final_iteration'] = GATES[-1]


class MemoryPressure(RuntimeError):
    pass


def wait_for_memory():
    # The child is gone: require enough free memory to recreate its allocations,
    # plus headroom, continuously for one minute (hysteresis).
    status(status='waiting_for_vram', child_pid=None, free_required_MiB=8192, stable_required_seconds=60)
    since = None
    while True:
        sample = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=10)
        free = int(sample.stdout.strip()) if sample.returncode == 0 else 0
        now = time.monotonic()
        since = (since if since is not None else now) if free >= 8192 else None
        stable = now-since if since is not None else 0
        status(free_MiB=free, stable_seconds=round(stable, 1))
        if stable >= 60:
            return
        time.sleep(5)


def newest_checkpoint(directory, limit):
    import torch
    candidates = sorted(directory.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[-1]), reverse=True)
    for path in candidates:
        try:
            saved = torch.load(path, map_location='cpu', weights_only=False)
            completed = saved['iter']+1
            stage = saved['infos']['march_stage' if MARCH or BALANCED or DURATION or STRONG or AVERAGE or KICK else 'early_stage']
            if 0 < completed <= limit and stage['control_steps'] == completed*24 and stage['schedule_version'] == 1:
                return path, completed
        except (OSError, RuntimeError, EOFError, KeyError, ValueError, pickle.UnpicklingError):
            continue  # interruption may have left the newest save incomplete
    return None, 0


def status(**updates):
    STATE.update(updates)
    temporary = OUT / 'status.tmp'
    temporary.write_text(json.dumps(STATE, indent=2)+'\n')
    temporary.replace(OUT / 'status.json')


def run(arguments, label):
    with (OUT / f'{label}.log').open('w') as log, (OUT / 'gpu.jsonl').open('a') as gpu:
        p = subprocess.Popen([PYTHON, *arguments], cwd=ROOT, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True,
                             env={**os.environ, 'PYTHONUNBUFFERED':'1', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'})
        status(active=label, child_pid=p.pid)
        try:
            while p.poll() is None:
                sample = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                if sample.returncode:
                    raise RuntimeError('GPU monitor failed; stopping our child process')
                used, free, utilization = map(int, sample.stdout.strip().split(','))
                gpu.write(json.dumps({'time':time.time(), 'stage':label, 'used_MiB':used,
                                      'free_MiB':free, 'utilization':utilization})+'\n')
                gpu.flush()
                if free < 2048:
                    raise MemoryPressure('Free GPU memory below 2 GiB; stopping our child process')
                time.sleep(2)
            if p.returncode:
                text = (OUT / f'{label}.log').read_text(errors='replace').lower()
                if 'cuda out of memory' in text or 'cuda error: out of memory' in text:
                    raise MemoryPressure(f'{label}: CUDA out of memory')
                raise RuntimeError(f'{label} exited with code {p.returncode}; see its log')
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()


def run_with_memory_retry(arguments, label):
    attempt = 0
    while True:
        attempt += 1
        try:
            run(arguments, f'{label}_attempt{attempt}_{time.time_ns()}')
            return
        except MemoryPressure as exc:
            status(reason=str(exc))
            wait_for_memory()
            status(status='evaluating')


def main():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    checkpoint = Path(STATE['checkpoint']) if STATE.get('checkpoint') else INITIAL
    previous = STATE['gates'][-1]['iteration'] if STATE['gates'] else START
    if '--resume-state' in sys.argv:
        wait_for_memory()
    for gate in GATES:
        if gate <= previous:
            continue
        # Recover the current authorized gate, including the original supervisor's
        # run if it stopped under VRAM pressure. Never select integration smoke runs.
        completed = previous
        directory = checkpoint.parent if checkpoint else None
        for candidate in RUNS.glob(f'*_{PREFIX}_to{gate}*'):
            recovered, count = newest_checkpoint(candidate, gate)
            if count > completed:
                checkpoint, completed, directory = recovered, count, candidate
        attempt = 0
        while completed < gate:
            attempt += 1
            name = f'{PREFIX}_to{gate}_retry{time.time_ns()}'
            status(status='training', target_iteration=gate, completed_iterations=completed,
                   checkpoint=str(checkpoint) if checkpoint else None, reason=None)
            before = set(RUNS.glob(f'*_{name}'))
            args = ['scripts/rsl_rl/train.py', '--headless', '--task', TASK,
                    '--num_envs',str(NUM_ENVS),'--max_iterations',str(gate-completed), '--seed','42','--run_name',name]
            if checkpoint:
                args += ['--resume','--checkpoint_path',str(checkpoint)]
            pressure = False
            try:
                run(args, f'train_{gate}_{attempt}_{time.time_ns()}')
            except MemoryPressure as exc:
                pressure = True
                status(reason=str(exc))
            created = set(RUNS.glob(f'*_{name}'))-before
            if len(created) > 1 or (not created and not pressure):
                raise RuntimeError('Cannot uniquely identify the new training directory')
            if created:
                current_directory = created.pop()
                recovered, count = newest_checkpoint(current_directory, gate)
                if recovered and count > completed:
                    checkpoint, completed, directory = recovered, count, current_directory
            if pressure:
                status(checkpoint=str(checkpoint) if checkpoint else None, completed_iterations=completed)
                wait_for_memory()
            elif completed != gate:
                raise RuntimeError('Training exited without the expected gate checkpoint')
        if checkpoint is None:
            raise RuntimeError('Missing checkpoint for evaluation')
        status(status='evaluating', checkpoint=str(checkpoint), completed_iterations=gate)
        evaluation = OUT / f'eval_{gate}'
        if KICK:
            run_with_memory_retry(['scripts/conga_check.py','--headless','--task',TASK,
                '--num_envs','16','--steps','2000','--rsl-checkpoint',str(checkpoint),
                '--output',str(evaluation)],f'eval_{gate}')
            run_with_memory_retry(['scripts/score_kick_intro.py','--rollout',str(evaluation/'rollout.csv')]+(['--refined'] if KICK_REFINE else []),f'analysis_{gate}')
        else:
            run_with_memory_retry(['scripts/stock_go2_check.py','--headless','--task',TASK,
                 '--checkpoint',str(checkpoint),'--steps','2000','--reward-audit','--output',str(evaluation)], f'eval_{gate}')
            run_with_memory_retry(['scripts/analyze_cadence_gate.py',str(evaluation),'--paired'], f'analysis_{gate}')
        metrics = json.loads((evaluation / 'cadence_metrics.json').read_text())
        # A retry may contain fewer than 50 iterations; merge by iteration,
        # replacing rolled-back observations with the resumed attempt's values.
        merged = {}
        gate_runs = sorted(RUNS.glob(f'*_{PREFIX}_to{gate}*'))
        for gate_run in gate_runs:
            events = EventAccumulator(str(gate_run)).Reload()
            if 'Train/mean_episode_length' in events.Tags()['scalars']:
                for point in events.Scalars('Train/mean_episode_length'):
                    if point.step < gate:
                        merged[point.step] = point
        lengths = [v for _, v in sorted(merged.items())][-50:]
        mean_length = sum(v.value for v in lengths)/len(lengths)*.02
        fall_fraction = metrics['falls']/max(1, metrics['falls']+metrics['timeouts'])
        result = {'iteration':gate, 'run':str(directory), 'runs':[str(r) for r in gate_runs], 'evaluation':str(evaluation),
                  'falls':metrics['falls'], 'timeouts':metrics['timeouts'],
                  'fall_fraction':fall_fraction, 'last50_episode_seconds':mean_length}
        prior = STATE['gates'][-1] if STATE['gates'] else ({'last50_episode_seconds':19.4199 if PAUSE else 19.2089,'fall_fraction':0.} if HISTORY or PAUSE else None)
        STATE['gates'].append(result)
        # Require both sustained training shortening and worsening evaluation
        # to distinguish deterioration from the expected difficult fresh start.
        if prior and mean_length < .65*prior['last50_episode_seconds'] and fall_fraction > prior['fall_fraction']+.15:
            status(status='paused_for_review', active=None, child_pid=None,
                   reason='Training episode length and evaluation falls deteriorated across gates')
            return
        previous = gate
    if KICK:
        status(status='complete', completed_iterations=GATES[-1], active=None, child_pid=None)
        return
    run_with_memory_retry(['scripts/analyze_cadence_experiment.py','--latest-overlaps','--runs',*[r for g in STATE['gates'] for r in g.get('runs', [g['run']])],
         '--evaluations',*[g['evaluation'] for g in STATE['gates']], '--output',str(OUT/'analysis')], 'combined_analysis')
    status(status='complete', completed_iterations=GATES[-1], active=None, child_pid=None)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        status(status='stopped', reason=str(exc), active=None, child_pid=None)
        raise
