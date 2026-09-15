"""Export reported times on the operator clock: Biotin injection = 0 min."""
from pathlib import Path
import csv,json,math
from paths import ROOT, REF, VAL, VIEWER
HERE=REF
INJECTION=2184.728/60
OUT=HERE/'injection_time_results';OUT.mkdir(exist_ok=True)
def read(name):
    with (HERE/'plateau_results'/name).open(newline='') as f:return list(csv.DictReader(f))
slopes=read('slope_threshold_sensitivity.csv')
rows=[]
for r in read('rate_and_plateau_summary.csv'):
    if r['source']!='Centroid + vendor TIR':continue
    low,hi=json.loads(r['t99_experiment_min_bootstrap95'])
    sr=next(x for x in slopes if x['channel']==r['channel'] and x['source']==r['source'] and float(x['slope_margin_mdeg_min'])==1)
    crossing=float(sr['first_sustained_min'])-INJECTION if sr['first_sustained_min'] else None
    row=dict(readout=r['channel'],physical_channel=1 if r['channel'].startswith(('L1 ','L2 ')) else 2,
       kobs_per_s=float(r['kobs_per_s']),tau_min=float(r['tau_min']),
       fit_cutoff_after_injection_min=52-INJECTION,
       forecast_T99_after_injection_min=float(r['t99_experiment_min'])-INJECTION,
       T99_conditional95_low_min=low-INJECTION,T99_conditional95_high_min=hi-INJECTION,
       slope_only_margin1_crossing_after_injection_min=crossing,
       strict_joint_gate='No accepted switch within analyzed prefix',
       analysis_cutoff_after_injection_min=58-INJECTION)
    assert math.isclose(row['forecast_T99_after_injection_min'],float(r['start_min'])-INJECTION+math.log(100)*row['tau_min'])
    rows.append(row)
with (OUT/'timing_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
pilot=math.ceil(max(r['forecast_T99_after_injection_min'] for r in rows))
assert pilot==24
print('Injection-relative timing CSV exported.')
