"""Measured arrival/time-scale normalization; no change to operational stop results."""

from pathlib import Path

import sys,json

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from paths import ROOT, REF, VAL, VIEWER
HERE=REF

sys.path.insert(0,str(ROOT/'src'))

from kinetics_diagnostics import fit_tail

OUT=HERE/'normalization_results';OUT.mkdir(exist_ok=True)

INJ=2184.728/60

rates=pd.read_csv(HERE/'plateau_results/rate_and_plateau_summary.csv');rates=rates[rates.source=='Centroid + vendor TIR'].copy()

fronts=pd.read_csv(HERE/'serial_results/front_times.csv');fronts=fronts[fronts.edge=='Biotin rise']

plt.rcParams.update({'font.size':9,'legend.fontsize':8,'pdf.fonttype':42})

f,axs=plt.subplots(2,2,figsize=(10,6.2),layout='constrained')

colors=['#276fbf','#758bd1','#008777','#bd7622']

rows=[]

for color,(_,r) in zip(colors,rates.iterrows()):

    lid=r.channel.split()[0];fr=fronts[fronts.channel==r.channel].iloc[0]

    d=pd.read_csv(HERE/f'plateau_results/{lid}_centroid_derivatives.csv')

    t=d.time_min.to_numpy();y=d.corrected_response_mdeg.to_numpy()

    mask=(t>=r.start_min)&(t<=58)

    z=(y[mask]-r.baseline_mdeg)/r.tail_amplitude_mdeg

    for ax,x in [(axs[0,0],t[mask]-INJ),(axs[0,1],t[mask]-fr.t50_min),(axs[1,0],(t[mask]-r.start_min)/r.tau_min)]:

        ax.plot(x,z,color=color,lw=.85,label=lid)

    # Equal local-age sensitivity fit: the same 1–9 min after TIR midpoint.

    equal=fit_tail(t,y,fr.t50_min+1,fr.t50_min+9)

    rows.append(dict(readout=lid,physical_channel=1 if lid in ('L1','L2') else 2,

        tau_min=r.tau_min,kobs_per_s=r.kobs_per_s,

        arrival50_after_injection_min=fr.t50_min-INJ,

        fit_start_after_injection_min=r.start_min-INJ,

        relaxation99_from_tail_start_min=np.log(100)*r.tau_min,

        T99_after_injection_min=r.t99_experiment_min-INJ,

        T99_after_arrival50_min=r.t99_experiment_min-fr.t50_min,

        T99_dimensionless=(r.t99_experiment_min-r.start_min)/r.tau_min,

        equal_local_age_tau_min=equal['tau_min'],equal_local_age_rmse_mdeg=equal['rmse_mdeg'],equal_local_age_parameter_boundary=equal['parameter_boundary'],equal_local_age_amplitude_mdeg=equal['tail_amplitude_mdeg']))

    export=d[mask].copy();export['T_after_injection_min']=t[mask]-INJ;export['s_after_arrival50_min']=t[mask]-fr.t50_min

    export['u_tail_age_over_tau']=(t[mask]-r.start_min)/r.tau_min;export['normalized_tail_response']=z

    export.to_csv(OUT/f'{lid}_normalized.csv',index=False)

axs[0,0].set(title='A. Operator clock',xlabel='Time since Biotin injection (min)')

axs[0,1].set(title='B. Arrival aligned',xlabel='Time since local TIR midpoint (min)')

axs[1,0].set(title='C. Scaled by each fitted time constant',xlabel='Tail age / fitted tau')

u=np.linspace(0,7.5,200);axs[1,0].plot(u,1-np.exp(-u),'k--',lw=1,label='1 - exp(-u)')

axs[1,0].axvline(np.log(100),color='gray',ls=':',lw=1)

for ax in [axs[0,0],axs[0,1],axs[1,0]]:

    ax.set(ylabel='Normalized tail response (R - b) / A',ylim=(-.2,1.55));ax.grid(alpha=.2);ax.legend(ncol=2)

summary=pd.DataFrame(rows);x=np.arange(4)

axs[1,1].bar(x,summary.fit_start_after_injection_min,label='Tail start after injection',color='#b4c6d8')

axs[1,1].bar(x,summary.relaxation99_from_tail_start_min,bottom=summary.fit_start_after_injection_min,label='4.605 x tau',color='#287f78')

axs[1,1].set(xticks=x,xticklabels=summary.readout,title='D. Forecast decomposition',ylabel='Predicted T99 after injection (min)')

axs[1,1].legend();axs[1,1].grid(axis='y',alpha=.2)

f.savefig(OUT/'normalized_kinetics.pdf');f.savefig(OUT/'normalized_kinetics.png',dpi=180);plt.close(f)

summary.to_csv(OUT/'normalization_summary.csv',index=False)

pairs=[]

for a,b in [(0,1),(2,3)]:

    ra,rb=rows[a],rows[b]

    pairs.append(dict(pair=ra['readout']+'/'+rb['readout'],relative_tau_difference_percent=100*abs(ra['tau_min']-rb['tau_min'])/np.mean([ra['tau_min'],rb['tau_min']]),relative_k_difference_percent=100*abs(ra['kobs_per_s']-rb['kobs_per_s'])/np.mean([ra['kobs_per_s'],rb['kobs_per_s']]),T99_difference_min=abs(ra['T99_after_injection_min']-rb['T99_after_injection_min'])))

report=dict(rows=rows,pairs=pairs,interpretation='Arrival alignment does not remove the fitted between-channel timescale difference. Scaling each fit by its own tau forces its model T99 to ln(100); this is an identity, not validation. Equal-local-age fits are retrospective sensitivity checks using 1–9 min after each TIR midpoint, not replacement forecasts or stop decisions.')

(OUT/'analysis.json').write_text(json.dumps(report,indent=2))

assert np.allclose(summary.T99_dimensionless,np.log(100))

assert np.allclose(summary.T99_after_injection_min,summary.fit_start_after_injection_min+summary.relaxation99_from_tail_start_min)

print(summary.to_string(index=False));print(json.dumps(pairs,indent=2))

