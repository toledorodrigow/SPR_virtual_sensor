"""Optional reconstruction from a user-supplied, unbundled angular .spr2 file."""
from pathlib import Path
import sys,json,argparse
from dataclasses import replace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from spr_reader import read_spr2,attach_binary,load_bulk_sensitivity_profile
from curve_processing import ProcessingConfig,process_channel
from run_analysis import baseline_zero

def run(part,path):
    ds=read_spr2(path);attach_binary(ds,ROOT/f'data/raw/{part}.bin')
    load_bulk_sensitivity_profile(ds,ROOT/'data/settings/bulk_sensitivity.json')
    injection=float(next(e['elapsed_s'] for e in ds.events if 'biotin' in str(e.get('sample','')).lower()))/60
    dest=ROOT/f'outputs/raw_extraction/{part}';dest.mkdir(parents=True,exist_ok=True)
    for cid in (0,1,2,4):
        stem=f'L{cid+1}_Weighted_centroid_Vendor_TIR_(hybrid)'
        settings=json.loads((ROOT/f'data/settings/{stem}.settings.json').read_text())
        ds.angle_calibration[cid]=tuple(settings['angle_calibration'])
        cfg=ProcessingConfig(**settings['settings'])
        if part=='validation':cfg=replace(cfg,baseline_start=injection-1,baseline_end=injection)
        table=process_channel(ds,cid,cfg)
        expected=pd.read_csv(ROOT/(f'data/reference/processed/{stem}.csv' if part=='reference' else f'data/validation/processed/L{cid+1}_processed.csv'))
        for field in ['SPRAngleEmpirical','TIRAngleEmpirical']:
            np.testing.assert_allclose(table[field],expected[field],atol=1e-9,rtol=1e-10)
        table.to_csv(dest/f'L{cid+1}_processed.csv',index=False)
        print(part,cid+1,len(table),'angular extraction matches bundled CSV')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('experiment',choices=['reference','validation']);p.add_argument('spr2',type=Path)
    a=p.parse_args();run(a.experiment,a.spr2)
