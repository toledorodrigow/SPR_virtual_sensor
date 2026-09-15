"""Portable paths; published exports are immutable, reruns go to outputs/."""
from pathlib import Path
import os,shutil
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=Path(os.environ.get('SPR_OUTPUT_DIR',str(ROOT/'outputs'))).resolve()
REF=OUTPUT/'reference';VAL=OUTPUT/'validation';VIEWER=ROOT/'src'
for name,dest in [('reference',REF),('validation',VAL)]:
    if not dest.exists():shutil.copytree(ROOT/'results'/name,dest)
(REF/'paper_figures').mkdir(exist_ok=True)
