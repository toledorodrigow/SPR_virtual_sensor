"""Complementary panels only; no refitting or changes to experimental exports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import ROOT, REF, VAL, VIEWER
HERE=REF
OUT = HERE / 'paper_figures'
INJ = 2184.728 / 60
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11,
                     'legend.fontsize': 8, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), layout='constrained')
for ax, cid, letter in zip(axes, [2, 5], ['A', 'B']):
    d = pd.read_csv(HERE / f'plateau_results/L{cid}_centroid_derivatives.csv')
    d = d[d.time_min.between(46, 58)]
    t = d.time_min.to_numpy() - INJ
    for field, color, label in [('dSPR', '#276fbf', 'SPR'),
                               ('dWeightedTIR', '#bd7622', 'S x TIR'),
                               ('dCorrected', '#008777', 'Corrected')]:
        ax.plot(t, d[field], color=color, lw=1, label=label)
    ax.fill_between(t, d.CorrectedLow95.to_numpy(), d.CorrectedHigh95.to_numpy(),
                    color='#008777', alpha=.17)
    for val in [-.5, .5]:
        ax.axhline(val, ls=':', color='gray', lw=.8)
    ax.set(title=f'{letter}. L{cid}: 785 nm, channel {1 if cid == 2 else 2}',
           xlabel='Time after injection (min)', ylabel='Slope (mdeg/min)')
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=.2)
fig.savefig(OUT / 'supplement_slopes_785.pdf')
plt.close(fig)

# Reuse the published comparison plotting logic for only the two wavelengths
# absent from the main paper. Saved fit parameters and baselines are unchanged.
source = (ROOT / 'src/compact_paper_figures.py').read_text(encoding='utf-8')
comparison = source[source.index("short=VAL"):]
comparison = comparison.replace('enumerate([1,3])', 'enumerate([2,5])')
comparison = comparison.replace("'main_short.pdf'", "'supplement_short_785.pdf'")
comparison = comparison.replace("label='25-min run'", "label='Independent repeat'")
exec(compile(comparison, str(ROOT / 'src/compact_paper_figures.py'), 'exec'))
print('Created complementary 785-nm slope and repeat-comparison figures.')
