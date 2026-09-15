"""Per-scan angular estimators. No temporal filtering or future-scan templates."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
from scipy.integrate import trapezoid


@dataclass(frozen=True)
class ProcessingConfig:
    spr_lo: float
    spr_hi: float
    tir_lo: float
    tir_hi: float
    smooth_points: int = 21
    fit_points: int = 21
    centroid_fraction: float = 0.3
    spr_method: str = 'Weighted centroid'
    tir_method: str = 'Derivative peak'
    baseline_start: float = 0.0
    baseline_end: float = 2.0

    def validate(self):
        if not all(np.isfinite(v) for v in (self.spr_lo, self.spr_hi, self.tir_lo,
                   self.tir_hi, self.centroid_fraction, self.baseline_start, self.baseline_end)):
            raise ValueError('Settings must be finite.')
        if not self.tir_lo < self.tir_hi < self.spr_lo < self.spr_hi:
            raise ValueError('Use separate increasing TIR and SPR motor-coordinate ranges (TIR before SPR).')
        if self.smooth_points < 5 or self.smooth_points % 2 != 1:
            raise ValueError('Angular smoothing needs an odd number of points, at least 5.')
        if self.fit_points < 5 or self.fit_points % 2 != 1:
            raise ValueError('Local fit needs an odd number of points, at least 5.')
        if not 0.05 <= self.centroid_fraction <= 0.8:
            raise ValueError('Centroid fraction must be between 0.05 and 0.8.')
        if self.baseline_end < self.baseline_start:
            raise ValueError('Baseline end must follow its start.')
        if self.spr_method not in ('Weighted centroid', 'Local quadratic') or self.tir_method not in ('Derivative peak', 'Derivative centroid', 'Edge template shift', 'Vendor TIR (hybrid)'):
            raise ValueError('Unknown estimator.')


def ordered_curve(ds, index, cid):
    x, _ = ds.x_axis(index, cid, 'motor')
    y = np.asarray(ds.scans[index].channels[cid], dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(x) < 7 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(np.diff(x) <= 0):
        raise ValueError('Invalid or insufficient curve samples')
    if not np.allclose(np.diff(x), np.median(np.diff(x)), rtol=1e-5):
        raise ValueError('Angular smoothing requires uniformly spaced motor samples')
    return x, y


def suggest_config(ds, cid):
    """Initial scan only; user must inspect and fix the regions before analysis."""
    index = next(i for i, s in enumerate(ds.scans) if cid in s.channels)
    x, y = ordered_curve(ds, index, cid)
    if len(x) < 250: raise ValueError('Automatic regions need at least 250 samples; enter regions manually.')
    z = savgol_filter(y, 21, 3)
    k = int(np.argmin(z))
    d = np.gradient(z, x)
    stop = max(22, k - 60)
    j = 21 + int(np.argmax(d[21:stop]))
    dx = float(np.median(np.diff(x)))
    return ProcessingConfig(float(x[max(0, k-100)]), float(x[min(len(x)-1, k+100)]),
                            float(x[max(0, j-45)]), float(x[min(k-101, j+45)]))


def quadratic_vertex(x, y, k, points, minimum):
    h = points // 2
    if k < h or k + h >= len(x):
        raise ValueError('Feature too close to ROI boundary for local fit')
    xx, yy = x[k-h:k+h+1], y[k-h:k+h+1]
    a, b, _ = np.polyfit(xx-x[k], yy, 2)
    if (a <= 0 if minimum else a >= 0) or abs(a) < 1e-16:
        raise ValueError('Local fit has wrong curvature')
    v = x[k] - b/(2*a)
    if not xx[0] < v < xx[-1]:
        raise ValueError('Fit vertex is outside local support')
    return float(v)


def lobe_centroid(x, y, k, level, minimum=True):
    """Integrate a contiguous lobe with interpolated zero-weight boundaries."""
    w = level-y if minimum else y-level
    left = right = k
    while left > 0 and w[left-1] > 0: left -= 1
    while right < len(x)-1 and w[right+1] > 0: right += 1
    if left == 0 or right == len(x)-1 or right-left < 2:
        raise ValueError('Centroid lobe truncated by ROI or undersampled')
    xl = x[left-1] + (x[left]-x[left-1]) * (-w[left-1])/(w[left]-w[left-1])
    xr = x[right] + (x[right+1]-x[right]) * w[right]/(w[right]-w[right+1])
    xx = np.r_[xl, x[left:right+1], xr]
    ww = np.r_[0., w[left:right+1], 0.]
    area = trapezoid(ww, xx)
    if area <= 0: raise ValueError('No positive feature area')
    return float(trapezoid(ww*xx, xx)/area), float(xr-xl)


def make_template(ds, cid, cfg):
    index = next(i for i, scan in enumerate(ds.scans) if cid in scan.channels)
    x, y = ordered_curve(ds, index, cid)
    from dataclasses import replace
    r, z, _ = extract_curve(x, y, replace(cfg, tir_method='Derivative peak'))
    if not np.isfinite(r['TIRMotor']): raise ValueError('First-scan TIR template is invalid')
    return x, z, r['TIRMotor']


def template_shift(x, z, cfg, template):
    if template is None: raise ValueError('First-scan edge template is required')
    tx, tz, anchor = template
    bound = (cfg.tir_hi-cfg.tir_lo)*.2
    mask = (x >= cfg.tir_lo+bound) & (x <= cfg.tir_hi-bound)
    xx, yy = x[mask], z[mask]
    if len(xx) < 10 or xx[0]-bound < tx[0] or xx[-1]+bound > tx[-1]:
        raise ValueError('Insufficient common template support')
    def fit(shift):
        ref = np.interp(xx-shift, tx, tz)
        design = np.column_stack((ref, np.ones(len(xx))))
        coef = np.linalg.lstsq(design, yy, rcond=None)[0]
        return float(np.mean((yy-design@coef)**2)), coef
    optimum = minimize_scalar(lambda s: fit(s)[0], bounds=(-bound, bound), method='bounded', options={'xatol':1e-5})
    mse, coef = fit(optimum.x)
    if not optimum.success or abs(optimum.x) > .98*bound or coef[0] <= 0:
        raise ValueError('Template fit failed or shift reached search limit')
    normalized = np.sqrt(mse)/max(np.ptp(yy), 1e-12)
    if normalized > .05: raise ValueError('TIR shape mismatch exceeds 5% of edge range')
    return float(anchor+optimum.x), float(normalized)


def extract_curve(x, y, cfg, template=None):
    if len(x) < cfg.smooth_points: raise ValueError('Curve shorter than smoothing window')
    z = savgol_filter(y, cfg.smooth_points, 3)
    derivative = savgol_filter(y, cfg.smooth_points, 3, deriv=1, delta=float(np.median(np.diff(x))))
    result = {}
    for kind, lo, hi in [('SPR', cfg.spr_lo, cfg.spr_hi), ('TIR', cfg.tir_lo, cfg.tir_hi)]:
        mask = (x >= lo) & (x <= hi)
        xx, zz = x[mask], z[mask]
        try:
            if len(xx) < max(cfg.fit_points+2, 7): raise ValueError('ROI has too few samples')
            if lo < x[0] or hi > x[-1]: raise ValueError('Scan does not cover ROI')
            if kind == 'SPR':
                k = int(np.argmin(zz))
                depth = min(zz[0], zz[-1])-zz[k]
                if depth <= 0: raise ValueError('No enclosed SPR dip')
                result['DipDepth'] = float(depth)
                if cfg.spr_method == 'Weighted centroid':
                    value, width = lobe_centroid(xx, zz, k, zz[k]+cfg.centroid_fraction*depth)
                    result['CentroidWidthMotor'] = width
                else:
                    value = quadratic_vertex(xx, zz, k, cfg.fit_points, True)
                result['RawMinimumMotor'] = float(xx[np.argmin(y[mask])])
            else:
                dd = derivative[mask]
                k = int(np.argmax(dd))
                floor = max(float(dd[0]), float(dd[-1]), 0.)
                if dd[k] <= floor: raise ValueError('No enclosed rising TIR edge')
                result['TIREdgeSlope'] = float(dd[k])
                if cfg.tir_method == 'Edge template shift':
                    value, residual = template_shift(x, z, cfg, template)
                    result['TIRTemplateRelativeRMSE'] = residual
                elif cfg.tir_method in ('Derivative peak', 'Vendor TIR (hybrid)'):
                    value = quadratic_vertex(xx, dd, k, cfg.fit_points, False)
                else:
                    value, _ = lobe_centroid(xx, dd, k, floor+0.5*(dd[k]-floor), False)
            result[kind+'Motor'] = value
            result[kind+'Quality'] = 'ok'
        except ValueError as e:
            result[kind+'Motor'] = np.nan
            result[kind+'Quality'] = str(e)
    return result, z, derivative


def process_channel(ds, cid, cfg):
    cfg.validate()
    template = make_template(ds, cid, cfg) if cfg.tir_method == 'Edge template shift' else None
    rows = []
    for i, scan in enumerate(ds.scans):
        if cid not in scan.channels: continue
        row = dict(scan_index=i, channel_id=cid, time_min=scan.rtime_ms/60000., direction=scan.direction)
        try:
            x, y = ordered_curve(ds, i, cid)
            result, _, _ = extract_curve(x, y, cfg, template)
            row.update(result)
        except ValueError as e:
            row.update(SPRMotor=np.nan, TIRMotor=np.nan, SPRQuality=str(e), TIRQuality=str(e))
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty: raise ValueError('No scans for this channel')
    slope, intercept, rmse = ds.angle_calibration.get(cid, (np.nan, np.nan, np.nan))
    calibrated = np.isfinite(slope) and slope > 0
    if cfg.tir_method == 'Vendor TIR (hybrid)':
        if ds.binary_table is None or not calibrated:
            raise ValueError('Hybrid TIR requires a companion BIN and positive empirical angle calibration.')
        vendor = ds.binary_table[ds.binary_table.channel_id == cid].set_index('scan_index')
        if vendor.index.has_duplicates: raise ValueError('Duplicate vendor scan identifiers')
        match = vendor.reindex(table.scan_index)
        angle = match.TirAngle.to_numpy(float)
        aligned = np.abs(match.time_min.to_numpy(float)-table.time_min.to_numpy(float)) < 1/60
        table['TIRMotor'] = np.where(aligned & np.isfinite(angle), (angle-intercept)/slope, np.nan)
        table['TIRQuality'] = np.where(table.TIRMotor.notna(), 'ok', 'Missing or time-mismatched vendor TIR')
    window = table.time_min.between(cfg.baseline_start, cfg.baseline_end)
    valid = window & table.SPRMotor.notna() & table.TIRMotor.notna()
    if not valid.any(): raise ValueError('No valid SPR/TIR pair in baseline interval; inspect ROIs and baseline times.')
    for kind in ('SPR', 'TIR'):
        table['Delta'+kind+'Motor'] = table[kind+'Motor'] - table.loc[valid, kind+'Motor'].mean()
        table[kind+'AngleEmpirical'] = table[kind+'Motor']*slope+intercept if calibrated else np.nan
        table['Delta'+kind+'Angle'] = table['Delta'+kind+'Motor']*slope if calibrated else np.nan
    sensitivity = ds.bulk_sensitivity.get(cid, np.nan)
    table['BulkSensitivity'] = sensitivity
    table['PureKineticsCandidate'] = table.DeltaSPRAngle - sensitivity*table.DeltaTIRAngle
    table['AngleCalibrationRMSEDeg'] = rmse
    return table


def noise_metrics(table, start, end):
    """Descriptive short-term noise, not an accuracy or kinetic-fidelity score."""
    w = table[table.time_min.between(start, end)]
    result = {'samples': len(w)}
    for field in ('DeltaSPRAngle', 'DeltaTIRAngle', 'PureKineticsCandidate'):
        v = w[field].to_numpy(float)
        delta = np.diff(v)
        delta = delta[np.isfinite(delta)]
        result[field] = float(1.4826*np.median(np.abs(delta-np.median(delta)))/np.sqrt(2)) if len(delta) >= 5 else None
    return result


def export_processing(path, ds, cid, cfg, table):
    path = Path(path)
    table.to_csv(path, index=False)
    metadata = dict(source=str(ds.spr2_path.resolve()), channel_id=cid, settings=asdict(cfg),
                    angle_calibration=ds.angle_calibration.get(cid), bulk_sensitivity=ds.bulk_sensitivity.get(cid),
                    bulk_sensitivity_source=ds.bulk_sensitivity_source,
                    temporal_filter='none', template_source='first available channel scan; gain and offset fitted per scan',
                    baseline_uses_future_until_min=cfg.baseline_end,
                    warning='Empirical angle axis fitted using full-record vendor minima. TIR absolute angle is extrapolated. Bulk coefficient requires validation for these estimators; not an early-stop validation.',
                    baseline_noise=noise_metrics(table, cfg.baseline_start, cfg.baseline_end))
    path.with_suffix('.settings.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
