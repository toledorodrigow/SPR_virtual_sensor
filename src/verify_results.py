"""Recompute main-table quantities and verify saved numerical provenance."""
import json
import numpy as np
import pandas as pd
from paths import ROOT,REF
from data_access import read_spr2
from kinetics_diagnostics import fit_tail,derivative_table
from run_analysis import baseline_zero

def run():
    ds=read_spr2('reference');rows=[]
    rates=pd.read_csv(ROOT/'results/reference/plateau_results/rate_and_plateau_summary.csv')
    for cid in (0,1,2,4):
        lid=f'L{cid+1}'
        a=pd.read_csv(ROOT/f'data/reference/processed/{lid}_Weighted_centroid_Vendor_TIR_(hybrid).csv')
        t=a.time_min.to_numpy();p=a.SPRAngleEmpirical.to_numpy()*1000;q=a.TIRAngleEmpirical.to_numpy()*1000;S=ds.bulk_sensitivity[cid]
        y=baseline_zero(t*60,p-S*q,2184.728)
        d=derivative_table(t,p,q,S)
        saved=pd.read_csv(ROOT/f'results/reference/plateau_results/{lid}_centroid_derivatives.csv')
        np.testing.assert_allclose(y,saved.corrected_response_mdeg,atol=1e-8)
        np.testing.assert_allclose(d.dCorrected,saved.dCorrected,atol=1e-8,equal_nan=True)
        r=rates[(rates.channel.str.startswith(lid+' '))&(rates.source=='Centroid + vendor TIR')].iloc[0]
        f=fit_tail(t,y,float(r.arrival_detection_min),52.)
        np.testing.assert_allclose(f['kobs_per_s'],r.kobs_per_s,rtol=1e-6)
        # A causal prefix must not change when only future observations change.
        modified=y.copy();modified[t>52]+=10000
        g=fit_tail(t,modified,float(r.arrival_detection_min),52.)
        np.testing.assert_allclose(f['parameters'],g['parameters'],rtol=1e-10)
        v=ds.binary_table[ds.binary_table.channel_id==cid]
        def noise(tt,yy):
            delta=np.diff(np.asarray(yy)[(tt>=10)&(tt<=30)])
            return 1.4826*np.median(np.abs(delta-np.median(delta)))/np.sqrt(2)
        vn=noise(v.time_min.to_numpy(),(v.PeakMinAngle-S*v.TirAngle).to_numpy()*1000)
        cn=noise(t,y)
        rows.append(dict(readout=lid,vendor_noise_mdeg=vn,centroid_noise_mdeg=cn,noise_reduction_pct=100*(1-cn/vn),
            kobs_per_s=f['kobs_per_s'],tau_min=f['tau_min'],T99_after_injection_min=f['t99_experiment_min']-2184.728/60,
            mean_corrected_slope=d.dCorrected[(t>=50)&(t<=54)].mean()))
    table=pd.DataFrame(rows);table.to_csv(REF.parent/'main_tables_recomputed.csv',index=False)
    stats=json.loads((ROOT/'results/reference/runtime_results/benchmark.json').read_text())
    latency=pd.read_csv(ROOT/'results/reference/runtime_results/latency.csv')
    np.testing.assert_allclose(latency.latency_ms.mean(),stats['latency_ms']['mean'])
    np.testing.assert_allclose(latency.latency_ms.max(),stats['latency_ms']['maximum'])
    counts=pd.read_csv(ROOT/'results/reference/runtime_results/operation_counts.csv')
    np.testing.assert_allclose(counts.basic_residual_flops.mean(),stats['basic_residual_flops']['mean'])
    return table

if __name__=='__main__':print(run().to_string(index=False))
