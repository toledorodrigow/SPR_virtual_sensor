# Complete manuscript result map

| Item | Figure / numerical export | Reproduction source |
|---|---|---|
| Main Fig. 1 | `figures/model_diagram.tex` | Original TikZ from manuscript |
| Main Fig. 2 | `main_reference.pdf` | `compact_paper_figures.py` |
| Main Fig. 3 | `main_short.pdf` | `compact_paper_figures.py`; validation measured CSVs only |
| Supplement Fig. S1 | `serial_fronts.pdf` | `serial_delivery.py`, `paper_figures.py` |
| Supplement Fig. S2 | `tail_fits.pdf` | `association_plateau.py`, `paper_figures.py` |
| Supplement Fig. S3 | `supplement_slopes_785.pdf` | `supplement_only_figures.py` |
| Supplement Fig. S4 | `normalized_kinetics.pdf` | `normalized_analysis.py` |
| Supplement Fig. S5 | `supplement_short_785.pdf` | `supplement_only_figures.py` |
| Main Table I | noise and slopes | `verify_results.py`; original processing method CSVs |
| Main Table II | `reference/plateau_results/rate_and_plateau_summary.csv` | `association_plateau.py`, `kinetics_diagnostics.py` (120 bootstrap draws) |
| Main Table III | `validation/endpoint_comparison.csv`, `exposure_comparison.csv` | `validation_outcomes.py` |
| Main Table IV | `reference/runtime_results/benchmark.json`, latency and operation CSVs | `benchmark_streaming.py`; historical measurements preserved |
| Supplement Table S1 | `reference/serial_results/inter_trace_delays.csv` | `serial_delivery.py` |
| Supplement Table S2 | `reference/normalization_results/normalization_summary.csv` | `normalized_analysis.py` |
| Supplement Tables S3–S5 | `validation/endpoint_comparison.csv` filtered by method and rinse window | `validation_outcomes.py` |
| Reference strict and alternative slope gates | `reference/plateau_results/prefix_k_tau.csv`, `slope_threshold_sensitivity.csv` | `association_plateau.py` |
| Equal-local-age rate sensitivity and paired-wavelength differences | `reference/normalization_results/analysis.json` | `normalized_analysis.py` |
| Model derivations and definitions | main/supplement manuscript source | Analytical expressions; no synthetic-data experiment |

Numerical-export paths above are relative to `results/`; figure filenames are
relative to `figures/`. New runs write corresponding results beneath `outputs/`.
The notebook executes the standard path and displays all eight figures, including
the conceptual TikZ diagram. Bootstrap and hardware profiling are optional cells.
