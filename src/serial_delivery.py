"""Event-anchored retrospective TIR-front timing, with physical labels unresolved."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import least_squares, minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import ROOT, REF, VAL, VIEWER
HERE=REF
VIEWER=VIEWER
sys.path.insert(0,str(VIEWER))
from data_access import read_spr2,attach_binary,auto_find_companion_bin
OUT=HERE/'serial_results'
EDGES=[('PBS rise',3.5,17.,1,228.485/60),('PBS decline',29.,39.,-1,None),
       ('Biotin rise',38.5,47.,1,2184.728/60),('Water-associated decline',59.,81.,-1,None)]


def front_fit(t,y,lo,hi,sign):
    mask=(t>=lo)&(t<=hi)
    x=t[mask];z=y[mask]-y[mask][0]
    center=(lo+hi)/2
    def model(p,tt):
        b,m,A,t50,w=p
        return b+m*(tt-center)+sign*A*expit((tt-t50)/w)
    fits=[]
    for width in (.2,.7,1.5):
        for fraction in (.35,.65):
            p=[float(z[0]),0,min(250,max(10,float(np.ptp(z)))),lo+(hi-lo)*fraction,width]
            r=least_squares(lambda p:model(p,x)-z,p,
                bounds=([-300,-10,1,lo,.03],[300,10,300,hi,5]),x_scale='jac',max_nfev=1000)
            if r.success:fits.append(r)
    if not fits:raise ValueError('Front fit failed')
    r=min(fits,key=lambda r:np.sum(r.fun*r.fun))
    b,m,A,t50,w=r.x
    return dict(t10_min=float(t50-np.log(9)*w),t50_min=float(t50),t90_min=float(t50+np.log(9)*w),
        width10_90_min=float(2*np.log(9)*w),amplitude_mdeg=float(A),drift_mdeg_min=float(m),
        rmse_mdeg=float(np.sqrt(np.mean(r.fun*r.fun))),parameters=r.x.tolist(),center_min=center)


def shifted_fit(t,early,late,lo,hi):
    """Late(t) ~= gain*early(t-delay)+offset, no temporal resampling of data."""
    mask=(t>=lo+4)&(t<=hi)
    tx=t[mask];y=late[mask]
    def objective(delay):
        ref=np.interp(tx-delay,t,early)
        X=np.column_stack((ref,np.ones(len(ref))))
        p=np.linalg.lstsq(X,y,rcond=None)[0]
        return float(np.mean((y-X@p)**2))
    r=minimize_scalar(objective,bounds=(-3,4),method='bounded')
    return dict(delay_min=float(r.x),rmse_mdeg=float(np.sqrt(r.fun)))


def main():
    OUT.mkdir(exist_ok=True)
    p=ROOT/'data/raw/reference.spr2'
    ds=read_spr2(p);attach_binary(ds,auto_find_companion_bin(p))
    channels={c:g.sort_values('time_min') for c,g in ds.binary_table.groupby('channel_id')}
    rows=[];comparisons=[]
    for name,lo,hi,sign,event in EDGES:
        fits={}
        for cid,g in channels.items():
            f=front_fit(g.time_min.to_numpy(),g.TirAngle.to_numpy()*1000,lo,hi,sign)
            fits[cid]=f
            rows.append(dict(edge=name,channel=ds.channel_label(cid),channel_id=int(cid),
                             command_min=event,command_to_t10_min=None if event is None else f['t10_min']-event,
                             command_to_t50_min=None if event is None else f['t50_min']-event,**f))
        for early,late in ((2,0),(4,1),(0,1),(2,4),(2,5),(2,6)):
            t=channels[late].time_min.to_numpy()
            yy=channels[late].TirAngle.to_numpy()*1000
            ee=np.interp(t,channels[early].time_min,channels[early].TirAngle)*1000
            shift=shifted_fit(t,ee,yy,lo,hi)
            delay=fits[late]['t50_min']-fits[early]['t50_min']
            comparisons.append(dict(edge=name,early_trace=ds.channel_label(early),late_trace=ds.channel_label(late),
                t50_delay_min=delay,apparent_volume_uL=10*delay,shift_fit_delay_min=shift['delay_min'],
                shift_fit_rmse_mdeg=shift['rmse_mdeg']))
    pd.DataFrame(rows).drop(columns='parameters').to_csv(OUT/'front_times.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'inter_trace_delays.csv',index=False)
    scan_times=np.sort(ds.binary_table.time_min.unique())
    report=dict(source='data/raw/reference.bin',scan_count=len(scan_times),spacing_s=float(np.median(np.diff(scan_times))*60),
        serial_mode_recorded=any('Serial' in str(e.get('flowchannels','')) for e in ds.events),
        header=ds.header,events=ds.events,fronts=rows,comparisons=comparisons,
        physical_channel_mapping={'1': [0, 1], '2': [2, 4, 5, 6]},
        mapping_source='Experimenter confirmation, 2026-09-14: L1/L2 are physical channel 1; remaining active readouts are channel 2.',
        interpretation='Physical channel 2 responds before channel 1. TIR fronts measure RI exchange, not direct molecule concentration. Cross-event consistency supports relative serial delay; no single t10 is exact molecular arrival.',
        calibration='Existing vendor TIR used to avoid fitting empirical SPR-angle scale to the full trace for this timing calculation.')
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    colors={0:'#276fbf',1:'#58a2dc',2:'#007f74',4:'#59ad80',5:'#a56eb7',6:'#c47424'}
    for ax,(name,lo,hi,sign,event) in zip(axs.flat,EDGES):
        for cid in (0,1,2,4):
            g=channels[cid];t=g.time_min.to_numpy();y=g.TirAngle.to_numpy()*1000
            f=next(r for r in rows if r['channel_id']==cid and r['edge']==name)
            b,m,A,t50,w=f['parameters'];mask=(t>=lo)&(t<=hi)
            normalized=(y[mask]-y[mask][0]-b-m*(t[mask]-f['center_min']))/(sign*A)
            ax.plot(t[mask],normalized,color=colors[cid],lw=1,label=ds.channel_label(cid))
            ax.axvline(t50,color=colors[cid],lw=.7,ls=':')
        if event is not None:ax.axvline(event,color='black',ls='--',lw=.8,label='Recorded command')
        ax.set(title=name,xlabel='Experiment time (min)',ylabel='Normalized TIR exchange',ylim=(-.15,1.15))
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Repeated solution fronts resolve an inter-group delay\nDotted lines: fitted 50% exchange times; normalization removes fitted linear drift')
    fig.savefig(OUT/'serial_fronts.png',dpi=180);fig.savefig(OUT/'serial_fronts.pdf');plt.close(fig)
    f,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
    pairs=[r for r in comparisons if r['early_trace'].startswith(('L3','L5')) and r['late_trace'].startswith(('L1','L2'))]
    for i,pair in enumerate((('L3 670nm','L1 670nm'),('L5 785nm','L2 785nm'))):
        vals=[r for r in pairs if (r['early_trace'],r['late_trace'])==pair]
        ax.plot(np.arange(4),[v['t50_delay_min'] for v in vals],'-o',label=f'{pair[0]} → {pair[1]}')
    ax.set(xticks=np.arange(4),xticklabels=[e[0].replace(' ','\n') for e in EDGES],ylabel='Later minus earlier t50 (min)')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    f.savefig(OUT/'delay_consistency.pdf');f.savefig(OUT/'delay_consistency.png',dpi=180);plt.close(f)
    print(pd.DataFrame(comparisons).to_string(index=False),flush=True)


if __name__=='__main__':main()
