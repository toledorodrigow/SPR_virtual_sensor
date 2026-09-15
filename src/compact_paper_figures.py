"""Main-paper panels selected by physical channel; all readouts remain in supplement."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import ROOT, REF, VAL, VIEWER
HERE=REF
VIEWER=VIEWER
OUT=HERE/'paper_figures';INJ=2184.728/60
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'legend.fontsize':8,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(10,5.6),layout='constrained')
for cid,col in zip([1,2,3,5],['#276fbf','#758bd1','#008777','#bd7622']):
    a=pd.read_csv(ROOT/f'data/reference/processed/L{cid}_Weighted_centroid_Vendor_TIR_(hybrid).csv')
    d=pd.read_csv(HERE/f'plateau_results/L{cid}_centroid_derivatives.csv')
    t=a.time_min.to_numpy();q=a.TIRAngleEmpirical.to_numpy()*1000
    axs[0,0].plot(t-INJ,q-np.mean(q[(t>=0)&(t<=2)]),color=col,lw=1,label=f'L{cid}')
    axs[0,1].plot(d.time_min-INJ,d.corrected_response_mdeg,color=col,lw=1)
for ax,title,ylabel in zip(axs[0],['A. Reference: solution exchange','B. Reference: corrected response'],['TIR change (mdeg)','Corrected R (mdeg)']):
    ax.set(title=title,ylabel=ylabel,xlabel='Time after injection (min)',xlim=(-36.4,76))
    ax.axvline(0,color='k',ls='--',lw=.8);ax.axvline(40.023,color='#b32134',ls='-.',lw=.8)
axs[0,0].legend(ncol=4,fontsize=8)
for ax,cid,title in zip(axs[1],[1,3],['C. Channel 1: L1 slopes','D. Channel 2: L3 slopes']):
    d=pd.read_csv(HERE/f'plateau_results/L{cid}_centroid_derivatives.csv');d=d[d.time_min.between(46,58)];t=d.time_min.to_numpy()-INJ
    for field,c,label in [('dSPR','#276fbf','SPR'),('dWeightedTIR','#bd7622','S x TIR'),('dCorrected','#008777','Corrected')]:ax.plot(t,d[field],color=c,lw=1,label=label)
    ax.fill_between(t,d.CorrectedLow95.to_numpy(),d.CorrectedHigh95.to_numpy(),color='#008777',alpha=.17)
    for val in [-.5,.5]:ax.axhline(val,ls=':',color='gray',lw=.8)
    ax.set(title=title,ylabel='Slope (mdeg/min)',xlabel='Time after injection (min)')
    ax.legend(ncol=3,fontsize=8)
for ax in axs.flat:ax.grid(alpha=.2)
fig.savefig(OUT/'main_reference.pdf');plt.close(fig)
short=VAL
import json
report=json.loads((short/'analysis.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(10,5.6),layout='constrained')
for row,cid in enumerate([1,3]):
    front=next(f for f in report['fronts'] if f['channel']==f'L{cid}')
    a=pd.read_csv(ROOT/f'data/validation/processed/L{cid}_processed.csv');b=pd.read_csv(HERE/f'plateau_results/L{cid}_centroid_derivatives.csv')
    t=a.T_after_injection_min.to_numpy();y=a.corrected_response_mdeg.to_numpy();ot=b.time_min.to_numpy()-INJ;oy=b.corrected_response_mdeg.to_numpy()
    ax,bx=axs[row]
    ax.plot(ot,oy,color='#777777',lw=1,label='Reference');ax.plot(t,y,color='#087e8b',lw=1,label='25-min run')
    ax.axvline(23.92,ls=':',color='#b35612',lw=.8,label='Reference target (23.92 min)')
    ax.axvline(25.1358,ls=':',color='k',lw=.8,label='Validation command end')
    ax.set(xlim=(0,40),title=f'{chr(65+row*2)}. L{cid}: association',xlabel='Time after injection (min)',ylabel='Corrected R (mdeg)')
    bx.plot(ot-front['reference_rinse50_after_injection_min'],oy,color='#777777',lw=1,label='Reference')
    bx.plot(t-front['rinse50_after_injection_min'],y,color='#087e8b',lw=1,label='25-min run')
    bx.axvspan(15,20,color='#dca329',alpha=.15);bx.set(xlim=(5,24),title=f'{chr(66+row*2)}. L{cid}: retained response',xlabel='Time after local rinse midpoint (min)',ylabel='Corrected R (mdeg)')
    for axis in [ax,bx]:axis.legend(fontsize=7);axis.grid(alpha=.2)
fig.savefig(OUT/'main_short.pdf');plt.close(fig)
