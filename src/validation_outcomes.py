"""Recompute measured delivery and endpoints. No validation kinetic model."""
import json
import numpy as np
import pandas as pd
from paths import ROOT,REF,VAL
from data_access import read_spr2
from serial_delivery import front_fit
from run_analysis import baseline_zero

def run():
    ds=read_spr2('validation');oldfront=pd.read_csv(REF/'serial_results/front_times.csv')
    injection=float(next(e['elapsed_s'] for e in ds.events if 'biotin' in str(e.get('sample','')).lower()))/60
    fronts=[];endpoints=[]
    for cid in (0,1,2,4):
        a=pd.read_csv(ROOT/f'data/validation/processed/L{cid+1}_processed.csv')
        t=a.time_min.to_numpy();p=a.SPRAngleEmpirical.to_numpy()*1000;q=a.TIRAngleEmpirical.to_numpy()*1000
        S=ds.bulk_sensitivity[cid];y=baseline_zero(t*60,p-S*q,injection*60)
        np.testing.assert_allclose(y,a.corrected_response_mdeg,atol=1e-8)
        rise=front_fit(t,q,injection+38.5-2184.728/60,injection+47-2184.728/60,1)
        fall=front_fit(t,q,injection+59-2184.728/60,injection+81-2184.728/60,-1)
        oldrise=oldfront[(oldfront.channel_id==cid)&(oldfront.edge=='Biotin rise')].iloc[0]
        oldfall=oldfront[(oldfront.channel_id==cid)&(oldfront.edge=='Water-associated decline')].iloc[0]
        fronts.append(dict(channel=f'L{cid+1}',rise50_after_injection_min=rise['t50_min']-injection,
            rinse50_after_injection_min=fall['t50_min']-injection,midpoint_exposure_min=fall['t50_min']-rise['t50_min'],
            reference_midpoint_exposure_min=oldfall.t50_min-oldrise.t50_min,
            reference_rinse50_after_injection_min=oldfall.t50_min-2184.728/60,rising=rise,falling=fall))
        old=pd.read_csv(REF/f'plateau_results/L{cid+1}_centroid_derivatives.csv')
        oldt=old.time_min.to_numpy();oldy=old.corrected_response_mdeg.to_numpy()
        vp=ds.binary_table[ds.binary_table.channel_id==cid].PeakMinAngle.to_numpy()*1000
        vy=baseline_zero(t*60,vp-S*q,injection*60)
        oldv=pd.read_csv(REF/f'plateau_results/L{cid+1}_vendor_derivatives.csv').corrected_response_mdeg.to_numpy()
        oldraw=pd.read_csv(ROOT/f'data/reference/processed/L{cid+1}_Weighted_centroid_Vendor_TIR_(hybrid).csv')
        oldpq=(oldraw.SPRAngleEmpirical-S*oldraw.TIRAngleEmpirical).to_numpy()*1000
        for method,short,ref in [('centroid',y,oldy),('vendor',vy,oldv),
            ('centroid_initial_water',p-S*q-np.mean((p-S*q)[(t>=.5)&(t<=2)]),oldpq-np.mean(oldpq[(oldt>=.5)&(oldt<=2)]))]:
            for lo,hi in ((10,15),(15,20),(20,24)):
                aa=(t-fall['t50_min']>=lo)&(t-fall['t50_min']<hi)
                bb=(oldt-oldfall.t50_min>=lo)&(oldt-oldfall.t50_min<hi)
                x=float(np.mean(short[aa]));z=float(np.mean(ref[bb]))
                endpoints.append(dict(channel=f'L{cid+1}',method=method,rinse_age_start_min=lo,rinse_age_end_min=hi,
                    short_mean_mdeg=x,reference_mean_mdeg=z,difference_mdeg=x-z,short_over_reference=x/z,
                    short_within_window_sd=float(np.std(short[aa],ddof=1)),reference_within_window_sd=float(np.std(ref[bb],ddof=1)),
                    short_samples=int(aa.sum()),reference_samples=int(bb.sum())))
    report=json.loads((ROOT/'results/validation/analysis.json').read_text())
    report.update(fronts=fronts,endpoints=endpoints)
    (VAL/'analysis.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame(endpoints).to_csv(VAL/'endpoint_comparison.csv',index=False)
    pd.DataFrame(fronts).drop(columns=['rising','falling']).to_csv(VAL/'exposure_comparison.csv',index=False)
    expected=pd.read_csv(ROOT/'results/validation/endpoint_comparison.csv')
    actual=pd.DataFrame(endpoints)
    np.testing.assert_allclose(actual.short_mean_mdeg,expected.short_mean_mdeg,atol=1e-7)
    np.testing.assert_allclose(actual.reference_mean_mdeg,expected.reference_mean_mdeg,atol=1e-7)
    return actual

if __name__=='__main__':print(run().to_string(index=False))
