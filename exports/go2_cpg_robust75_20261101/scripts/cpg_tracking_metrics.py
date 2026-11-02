"""Travel-only velocity audit. Kick/hold/recovery and reset samples are excluded."""
import argparse,json
from pathlib import Path
import numpy as np


def tracking_metrics(x):
    active=x['travel_active'].astype(bool)&x['ready'].astype(bool)&~x['reset'].astype(bool)
    target=x['tracking_velocity']
    requested=x['operator_velocity']
    measured=np.concatenate((x['body_velocity_local'][...,:2],x['angular_velocity'][...,2:3]),-1)
    steady=active&(np.linalg.norm(target[...,:2]-requested[...,:2],axis=-1)<.005)&(np.abs(target[...,2]-requested[...,2])<.01)
    err=measured-target
    cases={}
    rounded=np.round(requested,3)
    for command in np.unique(rounded.reshape(-1,3),axis=0):
        mask=steady&(rounded==command).all(-1)
        if not mask.any():continue
        actual=measured[mask].mean(0);desired=target[mask].mean(0)
        cases[str(command.tolist())]={'samples':int(mask.sum()),'travel_seconds':float(mask.sum()*.02),
             'actual_mean':actual.tolist(),'planned_mean':desired.tolist(),
             'xy_bias_m_s':float(np.linalg.norm(actual[:2]-desired[:2])),
             'yaw_bias_rad_s':float(abs(actual[2]-desired[2]))}
    nonzero=steady&(np.linalg.norm(target[...,:2],axis=-1)>.01)
    gains=(measured[...,:2]*target[...,:2]).sum(-1)/np.maximum((target[...,:2]**2).sum(-1),1e-8)
    return {'definition':'Only measured-ready SUPPORT/STEP travel intervals; kick, hold, recovery and reset excluded. No catch-up speed.',
            'active_samples':int(active.sum()),'excluded_samples':int(active.size-active.sum()),'steady_samples':int(steady.sum()),
            'xy_rmse_m_s':float(np.sqrt(np.mean((err[...,:2][active]**2).sum(-1)))) if active.any() else None,
            'yaw_rmse_rad_s':float(np.sqrt(np.mean(err[...,2][active]**2))) if active.any() else None,
            'nonzero_directional_gain':float(gains[nonzero].mean()) if nonzero.any() else None,
            'mean_command_xy_bias_m_s':float(np.mean([v['xy_bias_m_s'] for v in cases.values()])) if cases else None,
            'mean_command_yaw_bias_rad_s':float(np.mean([v['yaw_bias_rad_s'] for v in cases.values()])) if cases else None,
            'cases':cases}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    result=tracking_metrics(dict(np.load(a.directory/'traces.npz')))
    (a.directory/'tracking_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
