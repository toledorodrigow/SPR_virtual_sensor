"""Load archived processed binary readouts, not unavailable angular scans.

These compatibility functions serve downstream scripts only. For full angular
extraction use spr_reader.read_spr2 on an actual XML .spr2 record.
"""
from types import SimpleNamespace
import json
from paths import ROOT
from spr_reader import read_companion_bin

def read_spr2(path):
    part='validation' if 'validation' in str(path) else 'reference'
    metadata=json.loads((ROOT/f'data/{part}/experiment.json').read_text())
    profile=json.loads((ROOT/'data/settings/bulk_sensitivity.json').read_text())
    labels=profile['channel_labels']
    ds=SimpleNamespace(spr2_path=path,header=metadata.get('header',{}),events=metadata['events'],
        binary_table=read_companion_bin(ROOT/f'data/raw/{part}.bin'),
        bulk_sensitivity={int(k):v for k,v in profile['bulk_sensitivity'].items()},
        channel_label=lambda cid:labels[str(cid)])
    return ds

def attach_binary(ds,path):
    pass # already loaded explicitly by the processed-data adapter

def auto_find_companion_bin(path):
    return ROOT/'data/raw/reference.bin'

def load_bulk_sensitivity_profile(ds,path):
    return ds.bulk_sensitivity
