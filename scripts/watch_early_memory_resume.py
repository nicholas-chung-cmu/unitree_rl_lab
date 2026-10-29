"""Attach memory-resume recovery to the already-running original supervisor."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs/conga_early_1024_experiment'
state = json.loads((OUT / 'status.json').read_text())
pid = state['pid']
proc = Path(f'/proc/{pid}/stat')
identity = proc.read_text().split()[21] if proc.exists() else None
(OUT / 'memory_recovery.json').write_text(json.dumps({
    'watcher_pid':os.getpid(), 'supervisor_pid':pid, 'status':'armed',
    'resume_free_MiB':8192, 'resume_stable_seconds':60}, indent=2))
while proc.exists():
    try:
        fields = proc.read_text().split()
    except FileNotFoundError:
        break
    if fields[21] != identity or fields[2] == 'Z':
        break
    time.sleep(2)
state = json.loads((OUT / 'status.json').read_text())
reason = state.get('reason', '')
recover = state['status'] == 'stopped' and 'Free GPU memory below 2 GiB' in reason
if state['status'] == 'stopped' and not recover:
    # Original supervisor predates explicit OOM classification; inspect only
    # the log named in its failure, not unrelated earlier smoke logs.
    label = reason.split(' exited with code ')[0]
    log = OUT / f'{label}.log'
    if log.is_file():
        content = log.read_text(errors='replace').lower()
        recover = 'cuda out of memory' in content or 'cuda error: out of memory' in content
if recover:
    (OUT / 'memory_recovery.json').write_text(json.dumps({'status':'handed_to_resumable_supervisor'}))
    os.chdir(ROOT)
    os.execv('/home/rml2/anaconda3/envs/nick_sillywalk/bin/python', [
        '/home/rml2/anaconda3/envs/nick_sillywalk/bin/python',
        str(ROOT / 'scripts/run_early_experiment.py'), '--resume-state'])
else:
    (OUT / 'memory_recovery.json').write_text(json.dumps({
        'status':'finished_without_memory_recovery', 'supervisor_status':state['status'], 'reason':reason}))
