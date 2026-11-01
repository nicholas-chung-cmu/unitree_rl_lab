"""Measure actual body-height response at the fixed 90 BPM beat frequency.

Fit height = offset + drift*time + a*sin(beat phase) + b*cos(beat phase)
after the first four musical seconds. This separates the requested-frequency
component from total height variation; it does not attribute all motion to bounce.
Compare a bounce-enabled run with its no-bounce checkpoint baseline.
"""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('--amplitude', type=float, default=.0075)
a = p.parse_args()
x = np.load(a.directory / 'traces.npz')
rows = []
for env in range(x['time'].shape[1]):
    valid = x['ready'][:, env].astype(bool) & (x['time'][:, env] > 4) & ~x['reset'][:, env].astype(bool)
    t, height = x['time'][valid, env].astype(float), x['height'][valid, env].astype(float)
    if len(t) < 100:
        continue
    phase = 2 * np.pi * t / (2 / 3)
    design = np.stack((np.ones_like(t), t - t.mean(), np.sin(phase), np.cos(phase)), -1)
    coef = np.linalg.lstsq(design, height, rcond=None)[0]
    rows.append(dict(env=env, group=int(x['groups'][env]),
                     height_offset_m=float(coef[0]),
                     in_phase_amplitude_m=float(coef[2]),
                     quadrature_amplitude_m=float(coef[3]),
                     harmonic_peak_to_peak_m=float(2 * np.hypot(coef[2], coef[3])),
                     phase_lead_rad=float(np.arctan2(coef[3], coef[2])),
                     target_rmse_m=float(np.sqrt(np.mean((height - .30 - a.amplitude * np.sin(phase)) ** 2))),
                     residual_rmse_m=float(np.sqrt(np.mean((height - design @ coef) ** 2)))))
keys = [k for k in rows[0] if k not in ('env', 'group')] if rows else []
out = dict(method=__doc__, target_peak_to_peak_m=2*a.amplitude,
           median={k: float(np.median([r[k] for r in rows])) for k in keys},
           groups={str(g): {k: float(np.median([r[k] for r in rows if r['group'] == g])) for k in keys}
                   for g in sorted({r['group'] for r in rows})}, environments=rows)
(a.directory / 'bounce_audit.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({k: v for k, v in out.items() if k != 'environments'}, indent=2))
