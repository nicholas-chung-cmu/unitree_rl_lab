"""Plot training diagnostics separately from the held-out selection gates."""
import os
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
os.environ.setdefault('MPLCONFIGDIR','/tmp/cpg_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('logs/conga_cpg_session_20261101')
groups={
 'walking':[('walk_a_200','A: uncompensated'),('walk_b_200','B: compensated'),
            ('walk_b_500','B: continuation'),('walk_c_velocity_200','C: velocity'),
            ('walk_k_travel_200','K: travel'),('walk_m_yaw_travel_200','M: yaw')],
 'dance':[('dance_d_200','D: kick'),('dance_d_500','D: continuation'),
          ('dance_e_symmetry_200','E: mirror loss'),('dance_f_hold_200','F: hold reward'),
          ('dance_g_damping_200','G: damping'),('dance_h_hold_200','H: extended hold'),
          ('dance_i_mixed_hold_200','I: mixed hold'),('dance_l_angular_reward_200','L: angular reward'),
          ('dance_p_support_200','P: support geometry'),('dance_q_equivariant_300','Q: exact symmetry'),
          ('dance_r_travel_300','R: traveling dance')],
 'bounce':[('bounce_s_small_200','S: 5mm peak-to-peak reference'),
           ('bounce_t_medium_200','T: 10mm + height tracking'),
           ('bounce_u_full_300','U: 15mm reference')]
}
for group,experiments in groups.items():
 fig,axes=plt.subplots(2,2,figsize=(14,9))
 for name,label in experiments:
  if not (root/name/'result.json').exists():continue
  e=EventAccumulator(str(root/name/'training'));e.Reload()
  def values(tag):
   rows=e.Scalars(tag);return np.array([r.step for r in rows]),np.array([r.value for r in rows])
  step,correct=values('Metrics/cpg/correct')
  _,attempts=values('Metrics/cpg/attempts');_,skips=values('Metrics/cpg/skipped')
  axes[0,0].plot(step,correct/np.maximum(attempts+skips,1),label=label)
  for ax,tag in [(axes[0,1],'Episode_Termination/bad_orientation'),(axes[1,0],'Loss/learning_rate'),(axes[1,1],'Policy/mean_noise_std')]:
   x,y=values(tag);ax.plot(x,y,label=label)
 for ax,title in zip(axes.flat,['Training correct / (attempted + skipped)','Training orientation termination fraction','Adaptive PPO learning rate','Gaussian action standard deviation']):
  ax.set_title(title);ax.set_xlabel('Stored update index (branches may overlap)');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8,ncol=2);axes[1,0].set_yscale('log')
 fig.tight_layout();fig.savefig(root/('learning_curves_'+group+'.png'));plt.close(fig)
