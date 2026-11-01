"""Plot measured support and sole clearance against the musical contact schedule."""
import argparse
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cpg_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('--dance', action='store_true')
p.add_argument('--env', type=int, default=0)
p.add_argument('--seconds', type=float, default=8.)
p.add_argument('--bounce-amplitude', type=float, help='Overlay commanded body height in meters')
a = p.parse_args()
x = dict(np.load(a.directory / 'traces.npz'))
time = x['time'][:, a.env]
resets = np.flatnonzero(np.diff(time) < -.1)
end = resets[0] + 1 if len(resets) else len(time)
mask = (time[:end] >= 0) & (time[:end] <= a.seconds) & x['ready'][:end, a.env]
indices = np.flatnonzero(mask)
t = time[indices]
fig, axes = plt.subplots(5, 1, figsize=(13, 9), sharex=True)
for foot, ax in enumerate(axes[:4]):
    ax.step(t, x['expected'][indices, a.env, foot], where='post',
            color='black', linestyle='--', label='Scheduled contact')
    ax.step(t, .82*x['contact'][indices, a.env, foot], where='post',
            color='#2574b8', label='Measured contact (offset for visibility)')
    ax.set_ylim(-.12, 1.15)
    ax.set_ylabel(('FL', 'FR', 'RL', 'RR')[foot] + ' support')
    height = ax.twinx()
    height.plot(t, 100*x['sole'][indices, a.env, foot], color='#d86a16',
                alpha=.8, linewidth=1, label='Measured sole clearance')
    height.set_ylim(-1, 9)
    height.set_ylabel('Clearance (cm)', color='#d86a16')
    if foot == 0:
        lines, labels = ax.get_legend_handles_labels()
        extra, extra_labels = height.get_legend_handles_labels()
        ax.legend(lines+extra, labels+extra_labels, loc='upper right', fontsize=8)
axes[4].plot(t, 100*x['height'][indices, a.env], color='#387b45')
if a.bounce_amplitude is not None:
    axes[4].plot(t, 30+100*a.bounce_amplitude*np.sin(2*np.pi*t/(2/3)),
                 color='black', linestyle='--', label='Scheduled body height')
    axes[4].legend(fontsize=8)
axes[4].set_ylabel('Body height (cm)')
axes[4].set_xlabel('Musical/controller time (s), after measured settling')
for ax in axes:
    ax.grid(alpha=.15)
    if a.dance:
        for phrase in range(int(a.seconds/(8/3))+1):
            base = phrase*8/3
            ax.axvspan(base+5/3, base+5/3+.25, color='#d86a16', alpha=.10)
            ax.axvspan(base+5/3+.25, base+8/3-.2, color='#8d67ad', alpha=.08)
            for slot in range(3):
                ax.axvline(base+slot*2/3, color='black', alpha=.13, linewidth=.8)
    else:
        for target in np.arange(2/3, a.seconds, 2/3):
            ax.axvline(target, color='black', alpha=.13, linewidth=.8)
axes[0].set_title(a.directory.name + (' — orange: kick, purple: airborne hold' if a.dance else ' — vertical lines: touchdown beats'))
axes[-1].set_xlim(0, a.seconds)
fig.tight_layout()
fig.savefig(a.directory / 'contact_detail.png', dpi=150)
plt.close(fig)
