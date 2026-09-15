"""Corrected objective: near-plateau exposure followed by a matched water rinse.

The user confirmed the falling portion is water cleaning. It is not a missed
association asymptote. No shortened-run retained endpoint is manufactured here.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import ROOT, REF, VAL, VIEWER
HERE=REF
VIEWER=VIEWER
sys.path[:0]=[str(VIEWER),str(HERE)]
from kinetics_diagnostics import derivative_table, plateau_flags, fit_tail, bootstrap_tail, tail_model
from run_analysis import causal_arrival, baseline_zero
from data_access import read_spr2, attach_binary, auto_find_companion_bin, load_bulk_sensitivity_profile

OUT=HERE/'plateau_results'
INJECTION=2184.728/60
FIT_CUTOFF=52.
ASSOCIATION_LIMIT=58.  # conservative pre-decline interval, pending exact rinse time


def first_time(t,mask):
    idx=np.flatnonzero(mask)
    return None if not len(idx) else float(t[idx[0]])


def run():
    OUT.mkdir(exist_ok=True)
    ds=read_spr2(ROOT/'data/raw/reference.spr2')
    attach_binary(ds,auto_find_companion_bin(ds.spr2_path))
    load_bulk_sensitivity_profile(ds,ROOT/'data/settings/bulk_sensitivity.json')
    summary=[]; prefix_rows=[]; sensitivity_rows=[]; plot_data={}
    for cid in (0,1,2,4):
        vendor=ds.binary_table[ds.binary_table.channel_id==cid]
        tv=vendor.time_min.to_numpy()
        qv=vendor.TirAngle.to_numpy()*1000
        pv=vendor.PeakMinAngle.to_numpy()*1000
        S=ds.bulk_sensitivity[cid]
        arrival=float(tv[causal_arrival(tv*60,qv/1000,INJECTION*60)])
        frame=pd.read_csv(ROOT/f'data/reference/processed/L{cid+1}_Weighted_centroid_Vendor_TIR_(hybrid).csv')
        for source,t,p,q in [('Vendor',tv,pv,qv),('Centroid + vendor TIR',frame.time_min.to_numpy(),frame.SPRAngleEmpirical.to_numpy()*1000,frame.TIRAngleEmpirical.to_numpy()*1000)]:
            response=baseline_zero(t*60,p-S*q,INJECTION*60)
            derivatives=derivative_table(t,p,q,S,window_min=2.)
            eq,steady=plateau_flags(derivatives,response,0.,arrival+2,ASSOCIATION_LIMIT)
            derivatives['equivalent_slope']=eq
            derivatives['sustained_plateau']=steady
            derivatives['corrected_response_mdeg']=response
            slug=f'L{cid+1}_'+('centroid' if source!='Vendor' else 'vendor')
            derivatives.to_csv(OUT/f'{slug}_derivatives.csv',index=False)
            boot=bootstrap_tail(t,response,arrival,FIT_CUTOFF,repetitions=120,seed=1309+cid)
            candidates=[]
            for cutoff in np.arange(np.ceil(arrival+4),ASSOCIATION_LIMIT+.01,1.):
                try:
                    fit=fit_tail(t,response,arrival,float(cutoff))
                    i=np.searchsorted(t,cutoff,side='right')-1
                    candidates.append(dict(channel=ds.channel_label(cid),source=source,cutoff_min=float(cutoff),
                        plateau_pass=bool(steady[i]),**fit))
                except ValueError: continue
            dual=None
            for i,r in enumerate(candidates):
                stable=False
                if i>=2:
                    ks=[f['kobs_per_s'] for f in candidates[i-2:i+1]]
                    stable=(max(ks)-min(ks))/np.median(ks)<=.2
                r['rate_stable_20pct']=bool(stable)
                r['joint_exploratory_gate']=bool(r['plateau_pass'] and stable and not r['parameter_boundary']
                      and r['end_min']>=r['t99_experiment_min'] and r['rmse_mdeg']<=2.)
                if dual is None and r['joint_exploratory_gate']: dual=r['end_min']
            prefix_rows+=candidates
            for margin in (.2,.5,1.):
                _,passing=plateau_flags(derivatives,response,0.,arrival+2,ASSOCIATION_LIMIT,slope_margin=margin)
                sensitivity_rows.append(dict(channel=ds.channel_label(cid),source=source,slope_margin_mdeg_min=margin,
                                             first_sustained_min=first_time(t,passing)))
            # One fixed early plateau comparison window, not a searched best window.
            selected=derivatives[derivatives.time_min.between(50,54)]
            row=dict(channel=ds.channel_label(cid),source=source,S_bulk=S,arrival_detection_min=arrival,
                first_derivative_plateau_min=first_time(t,steady),joint_exploratory_trigger_min=dual,
                mean_SPR_slope_50_54=float(selected.dSPR.mean()),
                mean_weighted_TIR_slope_50_54=float(selected.dWeightedTIR.mean()),
                mean_corrected_slope_50_54=float(selected.dCorrected.mean()),
                derivative_identity_max_error=float(derivatives.identity_error.abs().max()),
                **boot)
            summary.append(row)
            plot_data[slug]=(t,response,derivatives,boot,candidates)
            print(row['channel'],source,'k',boot['kobs_per_s'],'tau min',boot['tau_min'],
                  'plateau',row['first_derivative_plateau_min'],'joint',dual,flush=True)
    report=dict(objective='Shorten Biotin exposure near the association plateau, then use the same water rinse and compare retained response with the full exposure.',
        user_clarification='Water cleaning causes the later falling response. Earlier association-versus-post-rinse failure interpretation is withdrawn.',
        injection_command_min=INJECTION,fit_cutoff_min=FIT_CUTOFF,provisional_association_limit_min=ASSOCIATION_LIMIT,
        settings=dict(derivative_window_min=2,slope_equivalence_margin_mdeg_min=.5,persistence_min=3,
            minimum_rise_mdeg=10,confidence='Approximate 95% Newey-West local-slope interval; not an anytime guarantee',
            joint_gate='sustained slope equivalence AND fitted 99% tail time reached AND k stable within 20% over 3 updates AND fit RMSE <=2 mdeg',
            bootstrap_repetitions=120,bootstrap_block_samples=8,declared_rate='apparent association-tail kobs, not intrinsic kon or koff'),
        summaries=summary,prefixes=prefix_rows,threshold_sensitivity=sensitivity_rows)
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(summary).drop(columns=['parameters']).to_csv(OUT/'rate_and_plateau_summary.csv',index=False)
    pd.DataFrame(prefix_rows).drop(columns=['parameters']).to_csv(OUT/'prefix_k_tau.csv',index=False)
    pd.DataFrame(sensitivity_rows).to_csv(OUT/'slope_threshold_sensitivity.csv',index=False)
    figures(plot_data,summary)
    write_report(report)
    print('DONE',OUT,flush=True)


def figures(data,summary):
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    rf,ra=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for cid,ax,bx in zip((0,1,2,4),axes.flat,ra.flat):
        t,response,d,fit,prefix=data[f'L{cid+1}_centroid']
        ax.plot(t,d.dSPR,color='#c47424',lw=1,label='dSPR/dt')
        ax.plot(t,d.dWeightedTIR,color='#8c62aa',lw=1,label='S × dTIR/dt')
        ax.plot(t,d.dCorrected,color='#007f74',lw=1.4,label='dCorrected/dt')
        ax.fill_between(t,d.CorrectedLow95,d.CorrectedHigh95,color='#007f74',alpha=.15,label='Corrected slope: approximate 95% band')
        ax.axhspan(-.5,.5,color='gray',alpha=.15,label='Declared plateau band ±0.5 mdeg/min')
        ax.axhline(0,color='gray',lw=.6)
        ax.set(xlim=(46,58),ylim=(-3,10),xlabel='Experiment time (min)',ylabel='Trailing 2-min slope (mdeg/min)',title=f'L{cid+1}')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
        use=(t>=fit['start_min'])&(t<=ASSOCIATION_LIMIT)
        bx.plot(t[use],response[use],color='.65',lw=1,label='Observed association tail')
        bx.plot(t[use],tail_model(fit['parameters'],(t[use]-fit['start_min'])*60),color='#007f74',label='Fit from data through 52 min')
        bx.axvline(FIT_CUTOFF,color='black',ls='--',lw=1,label='Fit cutoff (later data held out)')
        bx.axhline(fit['Rinf_mdeg'],color='#c47424',ls=':',lw=1,label='Fitted tail asymptote')
        bx.set(title=f"L{cid+1}: kobs={fit['kobs_per_s']:.4g} s⁻¹; τ={fit['tau_min']:.2f} min",xlabel='Experiment time (min)',ylabel='Corrected response (mdeg)')
        bx.grid(alpha=.2); bx.legend(fontsize=7)
    fig.suptitle('Does bulk compensation reveal a plateau?\nSlope calculation uses past scans only; constant S; upstream angle calibration remains retrospective')
    rf.suptitle('Association predictions, excluding the later water rinse\nApparent single-exponential tail; kobs and τ are not independently identified kon / koff')
    fig.savefig(OUT/'derivative_compensation.png',dpi=150)
    rf.savefig(OUT/'association_k_tau.png',dpi=150)
    plt.close(fig);plt.close(rf)
    pf,pa=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for cid,ax in zip((0,1,2,4),pa.flat):
        for source,color in [('vendor','#c47424'),('centroid','#007f74')]:
            _,_,_,_,prefix=data[f'L{cid+1}_{source}']
            ax.plot([p['end_min'] for p in prefix],[p['tau_min'] for p in prefix],'-o',ms=3,color=color,label=source)
        ax.set(title=f'L{cid+1}',xlabel='Last observed time (min)',ylabel='Apparent τ (min)')
        ax.grid(alpha=.2);ax.legend()
    pf.suptitle('Rate-estimate stability as association data accumulate\nOnly past samples are fitted at each cutoff')
    pf.savefig(OUT/'tau_prefix_stability.png',dpi=150);plt.close(pf)


def write_report(report):
    pass # numerical exports above are the reproducibility record

if __name__=='__main__':run()
