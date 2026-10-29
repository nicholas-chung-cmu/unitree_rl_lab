"""Offline alternating-contact candidate audit; no task or policy changes.

Require one scheduled diagonal swing per beat, >=40 ms of paired clearance,
and >=40 ms four-foot support before and after it. Reward adjacent valid beats.
At 50 Hz this uses two consecutive samples; timing is resolution-limited.
"""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FEET = ['FL','FR','RL','RR']


def blocks(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def score(t, contact, height, done=None, clearance=.01, debounce=2, tolerance=.04):
    """Per-episode fixed clock, no state carried over resets or missing samples."""
    done = np.zeros(len(t), bool) if done is None else done.astype(bool)
    breaks = np.r_[True, (np.diff(t) < .001) | (np.diff(t) > .021) | done[:-1]]
    segment = np.cumsum(breaks)
    beats = []
    for seg in np.unique(segment):
        ids = np.flatnonzero((segment == seg) & ~done)
        if not len(ids): continue
        # Float32 episode clocks can lie a few microseconds below a beat edge.
        beat_ids = np.floor((t[ids]+1e-5)/.5).astype(int)
        for beat in np.unique(beat_ids):
            use = ids[beat_ids == beat]
            phase = t[use]-beat*.5
            # Score complete beats after warmup. Incomplete final/reset beats
            # cannot produce a bonus or bridge an alternation sequence.
            if beat < 4 or len(use) != 25 or phase.min() > .021 or phase.max() < .459:
                continue
            c = contact[use] > .5
            h = height[use]
            all_four = c.all(-1)
            expected = [0,3] if beat % 2 == 0 else [1,2]
            other = [1,2] if beat % 2 == 0 else [0,3]
            pair = (~c[:,expected]).all(-1) & c[:,other].all(-1)
            events = [(a,b) for a,b in blocks(pair) if b-a >= debounce]
            valid = False
            if len(events) == 1:
                a,b = events[0]
                in_window = phase[a] >= .12-tolerance-1e-5 and phase[b-1] <= .32+tolerance+1e-5
                clear = (h[a:b,expected] >= clearance).all(-1)
                lifted = any(y-x >= debounce for x,y in blocks(clear))
                before = any(y-x >= debounce for x,y in blocks(all_four[:a]))
                after = any(y-x >= debounce for x,y in blocks(all_four[b:]))
                # Reject extra sustained missing contacts, including wrong-pair
                # steps, flight or single-foot dragging; tolerate one noisy tick.
                unexpected = ~all_four.copy()
                unexpected[a:b] = False
                extra = any(y-x >= debounce for x,y in blocks(unexpected))
                valid = in_window and lifted and before and after and not extra
            beats.append({'segment':int(seg),'beat':int(beat),'valid':bool(valid)})
    transitions = 0
    opportunities = 0
    for first, second in zip(beats, beats[1:]):
        if second['segment'] == first['segment'] and second['beat'] == first['beat']+1:
            opportunities += 1
            transitions += first['valid'] and second['valid']
    # One event per adjacent beat maximum, worth .5: reference tends to 1/sec.
    seconds = len(beats)*.5
    return {'complete_beats':len(beats),'valid_beats':sum(b['valid'] for b in beats),
            'opportunities':opportunities,'alternations':int(transitions),
            'success_fraction':transitions/opportunities if opportunities else 0.,
            'normalized_rate':.5*transitions/seconds if seconds else 0.,
            'weighted_rate_at_0.10':.05*transitions/seconds if seconds else 0.}


def main():
    spec = importlib.util.spec_from_file_location('alternation_ref',ROOT/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/conga_reference.py')
    refmod = importlib.util.module_from_spec(spec);sys.modules[spec.name]=refmod;spec.loader.exec_module(refmod)
    cfg = refmod.CongaReferenceCfg(kick_enabled=False,kick_scale=0,bounce_scale=0,
                                 cadence_swing_start=.12,cadence_swing_end=.32)
    t = np.arange(1000)*.02
    def reference(time):
        r=refmod.generate_reference(torch.tensor(time,dtype=torch.float64),cfg)
        return (r.contact.numpy()>=.5-1e-6).astype(float),r.foot_height.numpy()
    c,h = reference(t)
    patterns = {'correct':(c,h),'standing':(np.ones_like(c),np.zeros_like(h)),
                'wrong_phase':(c[:,[1,0,3,2]],h[:,[1,0,3,2]]),
                'flight':(np.zeros_like(c),np.ones_like(h)*.03),
                'too_low':(c,h*.1)}
    for diag in ([0,3],[1,2]):
        cc=np.ones_like(c);hh=np.zeros_like(h)
        active=(c<.5).any(-1)
        for foot in diag:
            cc[:,foot]=~active;hh[:,foot]=h.max(-1)
        patterns['repeat_'+str(diag)]=(cc,hh)
    for offset in (0,.125,.25,.375,.5):
        patterns['double_frequency_'+str(offset)]=reference(2*t+offset)
    for f in range(4):
        cc=np.ones_like(c);hh=np.zeros_like(h);cc[:,f]=c[:,f];hh[:,f]=h[:,f]
        patterns['single_'+FEET[f]]=(cc,hh)
    chatter_c=np.ones_like(c);chatter_h=np.zeros_like(h)
    chatter_c[::2]=c[::2];chatter_h[::2]=h[::2]
    patterns['one_tick_chatter']=(chatter_c,chatter_h)
    nc=c.copy();nh=h.copy()
    support=(c>.5).all(-1)
    nc[support,0]=0;nh[support,0]=.02
    patterns['no_four_foot_support']=(nc,nh)
    patterns['correct_20ms_late']=reference(t-.02)
    patterns['correct_20ms_early']=reference(t+.02)
    synthetic={name:score(t,cs,hs) for name,(cs,hs) in patterns.items()}
    assert synthetic['correct']['success_fraction']==1
    assert all(v['success_fraction']==1 for k,v in synthetic.items() if k.startswith('correct'))
    assert all(v['alternations']==0 for k,v in synthetic.items() if not k.startswith('correct'))
    reset=np.zeros(len(t),bool);reset[499]=True
    reset_time=np.r_[t[:500],t[:500]]
    rc,rh=reference(reset_time)
    reset_result=score(reset_time,rc,rh,reset)
    assert reset_result['success_fraction']==1 and reset_result['opportunities']==reset_result['complete_beats']-2
    rollouts=[]
    for directory in ['logs/conga_cadence_reward_audit','logs/conga_cadence_eval_final',
                      'logs/conga_paired_eval_200',*[f'logs/conga_early_1024_experiment/eval_{n}' for n in (200,500,1000)]]:
        d=np.genfromtxt(ROOT/directory/'rollout.csv',delimiter=',',names=True)
        meta=json.loads((ROOT/directory/'summary.json').read_text())
        cases=[]
        for case in meta['cases']:
            r=d[d['env']==case['case']]
            contact=np.stack([r['contact_'+f] for f in FEET],-1)
            height=np.stack([r['clearance_'+f] for f in FEET],-1)
            args=(r['episode_time_s'],contact,height,r['done'])
            nominal=score(*args)
            sensitivity={}
            for clearance in (.005,.01,.015):
                for debounce in (2,3):
                    sensitivity[f'{clearance}m_{debounce*20}ms']=score(*args,clearance=clearance,debounce=debounce)
            cases.append({'case':case['case'],'command':case['command'],**nominal,'sensitivity':sensitivity})
        rollouts.append({'directory':directory,'cases':cases})
    out={'definition':{'clearance_m':.01,'debounce_samples':2,'dt':.02,'timing_tolerance_s':.04,
        'event_value':.5,'candidate_weight':.10,'warmup_s':2},'synthetic':synthetic,
        'reset_check':reset_result,'rollouts':rollouts}
    dest=ROOT/'logs/conga_alternation_scoring';dest.mkdir(exist_ok=True)
    (dest/'scores.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Synthetic:',{k:v['alternations'] for k,v in synthetic.items()})
    for run in rollouts:
        print(run['directory'], 'valid beats',sum(c['valid_beats'] for c in run['cases']),
              'alternations',sum(c['alternations'] for c in run['cases']),
              'opportunities',sum(c['opportunities'] for c in run['cases']))


if __name__=='__main__':main()
