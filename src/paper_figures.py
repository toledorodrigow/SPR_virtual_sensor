"""Publication figures from measured data and saved experimental fits."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import ROOT, REF, VAL, VIEWER
HERE=REF
VIEWER=VIEWER
sys.path.insert(0,str(VIEWER))
from data_access import read_spr2,attach_binary,auto_find_companion_bin
OUT=HERE/'paper_figures';OUT.mkdir(exist_ok=True)
INJECTION=2184.728/60
XLABEL='Time since Biotin injection (min)'
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42})
colors=['#276fbf','#758bd1','#008777','#bd7622']
labels=['L1: channel 1, 670 nm','L2: channel 1, 785 nm','L3: channel 2, 670 nm','L5: channel 2, 785 nm']
ids=[1,2,3,5]
frames={i:pd.read_csv(HERE/f'plateau_results/L{i}_centroid_derivatives.csv') for i in ids}
source=ROOT/'data/raw/reference.spr2'
ds=read_spr2(source);attach_binary(ds,auto_find_companion_bin(source))
f,axs=plt.subplots(2,1,figsize=(10,5.5),sharex=True,layout='constrained')
for i,col,label in zip(ids,colors,labels):
    g=ds.binary_table[ds.binary_table.channel_id==i-1]
    t=g.time_min.to_numpy();q=g.TirAngle.to_numpy()*1000
    axs[0].plot(t-INJECTION,q-np.mean(q[(t>=0)&(t<=2)]),color=col,lw=1,label=label)
    d=frames[i];axs[1].plot(d.time_min-INJECTION,d.corrected_response_mdeg,color=col,lw=1)
for ax in axs:
    ax.axvline(3.808083-INJECTION,color='black',ls='--',lw=.8)
    ax.axvline(0,color='black',ls='--',lw=.8)
    ax.axvline(76.435183-INJECTION,color='#b32134',ls='-.',lw=1)
    ax.axvspan(59-INJECTION,81-INJECTION,color='#ddd5cd',alpha=.4)
    ax.grid(alpha=.2)
axs[0].legend(ncol=2,loc='lower left')
axs[0].set(ylabel='TIR change (mdeg)',title='Measured solution exchange and corrected response')
axs[1].set(xlabel=XLABEL,ylabel='Corrected response (mdeg)',xlim=(-INJECTION,113-INJECTION))
f.savefig(OUT/'full_record.pdf');plt.close(f)
f,axs=plt.subplots(2,2,figsize=(10,5.5),layout='constrained',sharex=True)
for ax,i,label in zip(axs.flat,ids,labels):
    d=frames[i];d=d[(d.time_min>=46)&(d.time_min<=58)]
    t=d.time_min.to_numpy()-INJECTION
    ax.plot(t,d.dSPR,color='#276fbf',label='SPR slope')
    ax.plot(t,d.dWeightedTIR,color='#bd7622',label='S x TIR slope')
    ax.plot(t,d.dCorrected,color='#008777',label='Corrected slope')
    ax.fill_between(t,d.CorrectedLow95.to_numpy(),d.CorrectedHigh95.to_numpy(),color='#008777',alpha=.17)
    for v in [-.5,.5]:ax.axhline(v,color='gray',ls=':',lw=.8)
    ax.set(title=label,xlabel=XLABEL,ylabel='Slope (mdeg/min)')
    ax.grid(alpha=.2)
axs[0,0].legend()
f.savefig(OUT/'slopes.pdf');plt.close(f)
rates=pd.read_csv(HERE/'plateau_results/rate_and_plateau_summary.csv')
f,axs=plt.subplots(2,2,figsize=(10,5.5),layout='constrained')
for ax,i,label in zip(axs.flat,ids,labels):
    r=rates[(rates.channel.str.startswith(f'L{i} '))&(rates.source=='Centroid + vendor TIR')].iloc[0]
    d=frames[i];d=d[(d.time_min>=r.start_min)&(d.time_min<=58)]
    ax.plot(d.time_min-INJECTION,d.corrected_response_mdeg,color='#276fbf',lw=.9,label='Measured corrected')
    for a,b,style in [(r.start_min,52,'-'),(52,58,'--')]:
        t=np.linspace(a,b,150)
        y=r.baseline_mdeg+r.tail_amplitude_mdeg*(1-np.exp(-r.kobs_per_s*(t-r.start_min)*60))
        ax.plot(t-INJECTION,y,color='#bd7622',ls=style,label='Prefix fit' if style=='-' else 'Continuation')
    ax.axvline(52-INJECTION,color='black',ls=':',lw=.8)
    ax.axhline(r.Rinf_mdeg,color='gray',ls=':',lw=.8)
    ax.set(title=f'{label}; tau = {r.tau_min:.2f} min',xlabel=XLABEL,ylabel='Corrected response (mdeg)')
    ax.grid(alpha=.2)
axs[0,0].legend()
f.savefig(OUT/'tail_fits.pdf');plt.close(f)
print('Created three vector publication figures from experimental exports.')

# Front plots retain the same fitted data and use the operator injection clock.
import json
from serial_delivery import EDGES
fronts=json.loads((HERE/'serial_results/analysis.json').read_text())['fronts']
f,axs=plt.subplots(2,2,figsize=(10,5.5),layout='constrained')
for ax,(name,lo,hi,sign,event) in zip(axs.flat,EDGES):
    for i,col,label in zip(ids,colors,labels):
        cid=i-1;g=ds.binary_table[ds.binary_table.channel_id==cid]
        t=g.time_min.to_numpy();y=g.TirAngle.to_numpy()*1000
        r=next(r for r in fronts if r['channel_id']==cid and r['edge']==name)
        b,m,A,t50,w=r['parameters'];mask=(t>=lo)&(t<=hi)
        z=(y[mask]-y[mask][0]-b-m*(t[mask]-r['center_min']))/(sign*A)
        ax.plot(t[mask]-INJECTION,z,color=col,lw=1,label=label.split(':')[0])
        ax.axvline(t50-INJECTION,color=col,ls=':',lw=.8)
    if event is not None:ax.axvline(event-INJECTION,color='black',ls='--',lw=.8)
    ax.set(title=name,xlabel=XLABEL,ylabel='Normalized TIR exchange',ylim=(-.15,1.15))
    ax.grid(alpha=.2);ax.legend(ncol=2)
f.savefig(OUT/'serial_fronts.pdf');plt.close(f)
