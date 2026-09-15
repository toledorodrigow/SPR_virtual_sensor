import numpy as np

def causal_arrival(t, tir, injection):
    """First three TIR points >5 mdeg above pre-command baseline, using past only."""
    pre = (t >= injection-60) & (t < injection)
    if pre.sum() < 3: raise ValueError('Insufficient pre-command baseline')
    level = np.mean(tir[pre])+.005
    streak = 0
    for i in np.flatnonzero(t >= injection):
        streak = streak+1 if np.isfinite(tir[i]) and tir[i] > level else 0
        if streak == 3: return int(i)
    raise ValueError('No sustained TIR rise detected')

def baseline_zero(t, y, injection):
    pre = (t >= injection-60) & (t < injection) & np.isfinite(y)
    if pre.sum() < 3: raise ValueError('Insufficient baseline')
    return y-float(np.mean(y[pre]))