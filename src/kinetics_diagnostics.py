"""Association-tail and causal derivative diagnostics in mdeg and minutes.

These are exploratory diagnostics, not a calibrated sequential certificate.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import t as student_t


def slope_with_hac(x, y):
    """OLS slope plus Newey-West standard error (lag floor(sqrt(n)))."""
    n = len(x)
    X = np.column_stack((np.ones(n), x-x.mean()))
    inv = np.linalg.inv(X.T@X)
    beta = inv@X.T@y
    residual = y-X@beta
    score = X*residual[:,None]
    meat = score.T@score
    lag = min(int(np.sqrt(n)), n-3)
    for j in range(1,lag+1):
        cross = score[j:].T@score[:-j]
        meat += (1-j/(lag+1))*(cross+cross.T)
    cov = inv@meat@inv*n/(n-2)
    se = np.sqrt(max(float(cov[1,1]),0))
    return float(beta[1]), float(student_t.ppf(.975,n-2)*se)


def derivative_table(time_min, spr_mdeg, tir_mdeg, sensitivity, window_min=2.):
    """Same linear operator for SPR, TIR and SPR-S*TIR; no future samples.

    S must be constant. Direct residual calculation preserves paired covariance.
    The estimate is the average slope over the trailing window, not an exact
    instantaneous derivative. Missing samples/gaps invalidate their windows.
    """
    t,p,q = map(lambda a:np.asarray(a,float),(time_min,spr_mdeg,tir_mdeg))
    if t.ndim!=1 or len(t)!=len(p) or len(t)!=len(q) or len(t)<8 or np.any(~np.isfinite(t)) or np.any(np.diff(t)<=0):
        raise ValueError('Need at least 8 aligned samples with increasing finite times.')
    if not np.isfinite(sensitivity) or not np.isfinite(window_min) or window_min<=0:
        raise ValueError('Sensitivity and positive window must be finite.')
    result=[]
    spacing=float(np.median(np.diff(t)))
    for i, end in enumerate(t):
        j=np.searchsorted(t,end-window_min)
        x=t[j:i+1]
        row=dict(time_min=end,window_start_min=float(x[0]),samples=len(x))
        if len(x)<8 or x[-1]-x[0]<window_min*.85 or np.max(np.diff(x))>2.5*spacing or not np.all(np.isfinite(p[j:i+1])) or not np.all(np.isfinite(q[j:i+1])):
            result.append(row); continue
        a,ha=slope_with_hac(x,p[j:i+1])
        b,hb=slope_with_hac(x,q[j:i+1])
        c,hc=slope_with_hac(x,p[j:i+1]-sensitivity*q[j:i+1])
        row.update(dSPR=a,dTIR=b,dWeightedTIR=sensitivity*b,dCorrected=c,
                   SPRHalfWidth95=ha,TIRHalfWidth95=hb,CorrectedHalfWidth95=hc,
                   CorrectedLow95=c-hc,CorrectedHigh95=c+hc,
                   identity_error=c-(a-sensitivity*b))
        result.append(row)
    df=pd.DataFrame(result)
    for col in ('dSPR','dTIR','dWeightedTIR','dCorrected','SPRHalfWidth95','TIRHalfWidth95','CorrectedHalfWidth95','CorrectedLow95','CorrectedHigh95','identity_error'):
        if col not in df: df[col]=np.nan
    return df


def plateau_flags(table, response, baseline, start_min, end_min,
                  slope_margin=.5, persistence_min=3., minimum_signal=10.):
    """95% descriptive slope band must be INSIDE +/- margin, not just cross zero.

    Fixed exploratory defaults, not optimized against the late endpoint. Flags
    end at the declared water-rinse boundary. Repeated intervals are descriptive.
    """
    t=table.time_min.to_numpy()
    active=(t>=start_min)&(t<end_min)&(np.asarray(response)-baseline>=minimum_signal)
    equivalent=active & (table.CorrectedLow95.to_numpy()>=-slope_margin) & (table.CorrectedHigh95.to_numpy()<=slope_margin)
    sustained=np.zeros(len(t),bool)
    begin=None
    spacing=float(np.median(np.diff(t)))
    for i,ok in enumerate(equivalent):
        if not ok or (i and t[i]-t[i-1]>2.5*spacing): begin=None
        if ok:
            if begin is None: begin=t[i]
            sustained[i]=t[i]-begin>=persistence_min
    return equivalent,sustained


def tail_model(p, elapsed_s):
    b,A,k=p
    return b+A*(-np.expm1(-k*np.asarray(elapsed_s)))


def fit_tail(time_min, response_mdeg, start_min, end_min):
    t,y=np.asarray(time_min,float),np.asarray(response_mdeg,float)
    use=(t>=start_min)&(t<=end_min)&np.isfinite(y)
    tt,yy=t[use],y[use]
    if len(tt)<20 or tt[-1]-tt[0]<2:
        raise ValueError('Select at least 20 association samples spanning at least 2 minutes.')
    x=(tt-tt[0])*60
    scale=max(float(np.ptp(yy)),1.)
    candidates=[]
    for rate in (.0005,.002,.01,.05):
        r=least_squares(lambda p:tail_model(p,x)-yy,[yy[0],scale,rate],
                        bounds=([-2000.,0.,1e-6],[2000.,2000.,.5]),
                        x_scale='jac',max_nfev=1500)
        if r.success: candidates.append(r)
    if not candidates: raise ValueError('Association-tail fit did not converge.')
    best=min(candidates,key=lambda r:np.sum(r.fun**2))
    b,A,k=best.x
    fitted=tail_model(best.x,x)
    residual=yy-fitted
    bound=bool(k<=1.01e-6 or k>=.499 or A>=1999 or A<.1)
    return dict(start_min=float(tt[0]),end_min=float(tt[-1]),samples=len(tt),
                baseline_mdeg=float(b),tail_amplitude_mdeg=float(A),Rinf_mdeg=float(b+A),
                kobs_per_s=float(k),tau_s=float(1/k),tau_min=float(1/k/60),
                t95_experiment_min=float(tt[0]-np.log(.05)/k/60),
                t99_experiment_min=float(tt[0]-np.log(.01)/k/60),
                remaining_tail_mdeg_at_cutoff=float(A*np.exp(-k*x[-1])),
                rmse_mdeg=float(np.sqrt(np.mean(residual**2))),
                residual_lag1=float(np.corrcoef(residual[:-1],residual[1:])[0,1]) if np.std(residual)>1e-12 else 0.,
                parameter_boundary=bound,parameters=best.x.tolist())


def bootstrap_tail(t,y,start,end,repetitions=120,seed=1309,block_samples=8):
    fit=fit_tail(t,y,start,end)
    t,y=np.asarray(t,float),np.asarray(y,float)
    use=(t>=start)&(t<=end)&np.isfinite(y)
    tt,yy=t[use],y[use]
    pred=tail_model(fit['parameters'],(tt-fit['start_min'])*60)
    residual=yy-pred
    residual-=residual.mean()
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(repetitions):
        starts=rng.integers(0,len(tt),int(np.ceil(len(tt)/block_samples)))
        idx=np.concatenate([(np.arange(block_samples)+s)%len(tt) for s in starts])[:len(tt)]
        try:
            f=fit_tail(tt,pred+residual[idx],start,end)
            samples.append([f['kobs_per_s'],f['tau_min'],f['t99_experiment_min'],f['Rinf_mdeg']])
        except ValueError: pass
    if len(samples)<repetitions*.8: raise ValueError('Too many bootstrap fit failures.')
    intervals=np.quantile(samples,[.025,.975],axis=0)
    for j,name in enumerate(('kobs_per_s','tau_min','t99_experiment_min','Rinf_mdeg')):
        fit[name+'_bootstrap95']=intervals[:,j].tolist()
    fit.update(bootstrap_success=len(samples),bootstrap_block_samples=block_samples,
               bootstrap_note='Conditional moving-block residual intervals; excludes calibration, delivery and model-choice uncertainty.')
    return fit
