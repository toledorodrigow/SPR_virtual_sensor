"""Replay the published four-readout tail estimator; no synthetic signals.

Runtime is measured separately from callback instrumentation and tracemalloc.
Residual arithmetic counts exclude solver algebra and transcendental internals.
"""
from pathlib import Path
import sys, json, time, platform, os, tracemalloc, gc
import numpy as np
import pandas as pd
import scipy

from paths import ROOT, REF, VAL, VIEWER
HERE=REF
VIEWER=VIEWER
sys.path.insert(0, str(VIEWER))
import kinetics_diagnostics as kd

OUT = HERE/'runtime_results'
OUT.mkdir(exist_ok=True)
summary = pd.read_csv(HERE/'plateau_results/rate_and_plateau_summary.csv')
streams = []
for cid in (1,2,3,5):
    meta = summary[(summary.channel.str.startswith(f'L{cid} ')) & (summary.source=='Centroid + vendor TIR')].iloc[0]
    f = pd.read_csv(ROOT/f'data/reference/processed/L{cid}_Weighted_centroid_Vendor_TIR_(hybrid).csv')
    d = pd.read_csv(HERE/f'plateau_results/L{cid}_centroid_derivatives.csv')
    t = f.time_min.to_numpy()
    assert np.allclose(t, d.time_min)
    streams.append(dict(cid=cid,t=t,p=f.SPRAngleEmpirical.to_numpy()*1000,
        q=f.TIRAngleEmpirical.to_numpy()*1000,y=d.corrected_response_mdeg.to_numpy(),
        S=float(meta.S_bulk),start=float(meta.arrival_detection_min)))

eligible = [i for i,t in enumerate(streams[0]['t']) if t >= max(s['start'] for s in streams)+4 and t<=58]
# All four channels eligible; each update sees only its acquired prefix.
def update(i, collect=False):
    fits=[]
    for s in streams:
        t=s['t'][:i+1]; p=s['p'][:i+1]; q=s['q'][:i+1]
        j=np.searchsorted(t,t[-1]-2)
        a=kd.slope_with_hac(t[j:],p[j:])
        b=kd.slope_with_hac(t[j:],q[j:])
        c,h=kd.slope_with_hac(t[j:],p[j:]-s['S']*q[j:])
        fit=kd.fit_tail(t,s['y'][:i+1],s['start'],t[-1])
        # Pointwise gate inputs; persistence/history bookkeeping is excluded.
        fit['slope_equivalent']=bool(c-h>=-.5 and c+h<=.5)
        fits.append(fit)
    return fits

update(eligible[0]) # warm libraries and solver
rows=[]
for repeat in range(3):
    for i in eligible:
        tic=time.perf_counter_ns(); fits=update(i); elapsed=(time.perf_counter_ns()-tic)/1e6
        rows.append(dict(repeat=repeat,index=i,time_min=float(streams[0]['t'][i]),latency_ms=elapsed))
    print('Timed replay',repeat+1,flush=True)
pd.DataFrame(rows).to_csv(OUT/'latency.csv',index=False)

original=kd.least_squares
counts=[]
def counted(fun,x0,**kw):
    counter={'calls':0,'point_evaluations':0}
    def wrapped(p):
        value=fun(p); counter['calls']+=1;counter['point_evaluations']+=len(value)
        return value
    r=original(wrapped,x0,**kw)
    counter.update(nfev=r.nfev,njev=r.njev,success=bool(r.success))
    counts.append(counter)
    return r
kd.least_squares=counted
oprows=[]
for i in eligible:
    counts.clear(); fits=update(i)
    assert len(counts)==16
    points=sum(c['point_evaluations'] for c in counts)
    oprows.append(dict(index=i,residual_calls=sum(c['calls'] for c in counts),
        residual_point_evaluations=points,basic_residual_flops=4*points,
        expm1_calls=points,scalar_negations=points+sum(c['calls'] for c in counts),
        nfev=sum(c['nfev'] for c in counts),njev=sum(c['njev'] for c in counts),
        maximum_fit_samples=max(f['samples'] for f in fits)))
kd.least_squares=original
pd.DataFrame(oprows).to_csv(OUT/'operation_counts.csv',index=False)
gc.collect();tracemalloc.start(); update(eligible[-1]); current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
# Check exact published-cutoff results, outside timing.
verification=[]
for s in streams:
    f=kd.fit_tail(s['t'],s['y'],s['start'],52.)
    m=summary[(summary.channel.str.startswith(f"L{s['cid']} ")) & (summary.source=='Centroid + vendor TIR')].iloc[0]
    assert np.isclose(f['kobs_per_s'],m.kobs_per_s,rtol=1e-7)
    verification.append(f['kobs_per_s'])
cpu=platform.processor() or platform.machine()
lat=np.array([r['latency_ms'] for r in rows]);op=pd.DataFrame(oprows)
result=dict(cpu=cpu,os=platform.platform(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
    repetitions=3,updates_per_replay=len(eligible),readouts_per_update=4,
    first_time_after_injection_min=float(streams[0]['t'][eligible[0]]-2184.728/60),
    last_time_after_injection_min=float(streams[0]['t'][eligible[-1]]-2184.728/60),
    latency_ms=dict(mean=float(lat.mean()),median=float(np.median(lat)),p95=float(np.quantile(lat,.95)),maximum=float(lat.max())),
    median_scan_spacing_s=float(np.median(np.diff(streams[0]['t']))*60),
    minimum_scan_spacing_s=float(np.min(np.diff(streams[0]['t']))*60),
    basic_residual_flops=dict(mean=float(op.basic_residual_flops.mean()),maximum=int(op.basic_residual_flops.max())),
    expm1_calls=dict(mean=float(op.expm1_calls.mean()),maximum=int(op.expm1_calls.max())),
    residual_calls=dict(mean=float(op.residual_calls.mean()),maximum=int(op.residual_calls.max())),
    maximum_fit_samples=int(op.maximum_fit_samples.max()),
    final_prefix_tracemalloc_peak_bytes=peak,
    maximum_tail_time_response_payload_bytes=int(sum(16*np.count_nonzero((s['t']>=s['start'])&(s['t']<=58)) for s in streams)),
    published_rate_verification=verification,
    scope='Four sequential tail fits and three trailing 2-min HAC slopes per readout, at each eligible scan. Loaded preprocessed real signals and fixed calibration; no angular extraction, file IO, GUI, bootstrap, arrival detection, persistence/history bookkeeping or model C(t) inversion. Growing prefixes; unchanged four-start SciPy fit_tail.',
    operation_convention='Per residual point: two multiplies plus two adds/subtracts = 4 basic FLOPs; one expm1 separately. Negation separately. Includes finite-difference residual callbacks. Excludes solver linear algebra, derivatives, array management and all other arithmetic. Not a hardware FLOP counter or total FLOP bound.',
    memory_scope='tracemalloc peak incremental allocation for the final four-readout update after preload, separate from timings; not process RSS or full native workspace; input payload reported separately.')
(OUT/'benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
