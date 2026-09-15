"""Run all manuscript plots, numerical checks, or optional expensive refits."""
from pathlib import Path
import sys,subprocess,argparse,shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from paths import REF,VAL

def execute(name):
    subprocess.run([sys.executable,str(ROOT/'src'/name)],check=True,cwd=ROOT)

def figures():
    for name in ['paper_figures.py','normalized_analysis.py','compact_paper_figures.py','supplement_only_figures.py']:execute(name)
    dest=REF.parent/'figures';dest.mkdir(exist_ok=True)
    for name in ['main_reference.pdf','main_short.pdf','serial_fronts.pdf','tail_fits.pdf','supplement_slopes_785.pdf','supplement_short_785.pdf']:
        shutil.copy2(REF/'paper_figures'/name,dest/name)
    shutil.copy2(REF/'normalization_results/normalized_kinetics.pdf',dest/'normalized_kinetics.pdf')
    print('Seven empirical figures regenerated; Fig. 1 is editable TikZ in manuscript/paper.tex.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true',help='Recompute reference bootstrap/prefix results (several minutes).')
    p.add_argument('--benchmark',action='store_true',help='Measure current hardware; not expected to reproduce published latency.')
    a=p.parse_args()
    if a.full:execute('association_plateau.py')
    execute('serial_delivery.py');execute('validation_outcomes.py');figures()
    execute('verify_results.py')
    if a.benchmark:execute('benchmark_streaming.py')
