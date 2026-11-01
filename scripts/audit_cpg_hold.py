"""Audit the first kick per environment over a full stationary 0.75 s hold.

The initial kick must reach the measured endpoint within the relaxed 0.20 s
admission window. Starting there, require 0.75 s of continued HOLD, no confirmed
touchdown or support departure, and >=80% strict endpoint-quality samples.
Also report the more demanding uninterrupted-valid duration. Later retries do not
replace a failed first attempt. This is an isolated hold test, not musical timing.
"""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('--duration', type=float, default=.75)
a = p.parse_args()
x = dict(np.load(a.directory / 'traces.npz'))
dt = .02
frames = int(np.ceil(a.duration / dt))
result = {leg: dict(attempts=0, endpoint=0, qualified_hold=0,
                   continuous_strict_hold=0, valid_fractions=[]) for leg in ('FL', 'FR')}
for env in range(x['time'].shape[1]):
    starts = np.flatnonzero(x['kick_active'][:, env] &
                           ~np.r_[False, x['kick_active'][:-1, env]])
    if not len(starts):
        continue
    start = starts[0]
    leg = int(x['kick_leg'][start, env])
    row = result[('FL', 'FR')[leg]]
    row['attempts'] += 1
    boundary = starts[1] if len(starts) > 1 else len(x['time'])
    endpoints = np.flatnonzero(x['kick_success'][start:boundary, env])
    if not len(endpoints):
        continue
    row['endpoint'] += 1
    first = start + endpoints[0] + 1  # HOLD begins after endpoint feedback
    last = first + frames
    if last > boundary or last > len(x['time']):
        continue
    valid = x['hold_valid'][first:last, env]
    held = x['hold_active'][first:last, env].all()
    contacts = x['contact'][first - 1:last, env]
    selected_grounded = (contacts[1:, leg] & contacts[:-1, leg]).any()
    supports = [j for j in range(4) if j != leg]
    support_lost = (~contacts[1:, supports] & ~contacts[:-1, supports]).any()
    reset = x['reset'][first:last, env].any()
    fraction = float(valid.mean())
    row['valid_fractions'].append(fraction)
    retained = held and not selected_grounded and not support_lost and not reset
    row['qualified_hold'] += int(retained and fraction >= .8)
    row['continuous_strict_hold'] += int(retained and valid.all())
for row in result.values():
    fractions = row.pop('valid_fractions')
    row['median_valid_fraction'] = float(np.median(fractions)) if fractions else None
    row['qualified_fraction'] = row['qualified_hold'] / max(row['attempts'], 1)
out = dict(method=__doc__, requested_duration_s=a.duration,
           sampled_duration_s=frames * dt, legs=result)
(a.directory / 'hold_audit.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
