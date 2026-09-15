# SPR virtual sensor: experimental reproducibility

Code, processed experimental data, and an executed notebook for **SPR Virtual
Sensor for Early Stopping: Arrival-Aware Kinetic Forecasting and Experimental
Bulk-Corrected Response Analysis**.

The reference experiment supplies all kinetic fits and stop diagnostics. The
approximately 25-minute experiment supplies measured validation outcomes only.
There is no kinetic refit or stopping replay of the validation experiment.

## Start here

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your OS.
python -m pip install -r requirements.txt
jupyter lab notebooks/SPR_reproducibility.ipynb
```

The notebook explains the processing, clocks, model, derivatives, validation
endpoints, normalization, and computational accounting in sequence. Run all
cells for the standard reproduction using bundled CSVs and saved reference
bootstrap results. Code is implemented in readable modules in `src/`; notebook
cells show the central functions and invoke the same implementation.

```bash
python scripts/reproduce.py
python scripts/reproduce.py --full
python scripts/reproduce.py --benchmark
```

The default run regenerates all seven empirical manuscript figures, refits TIR
fronts, recomputes validation endpoints, checks the four reference point fits,
derivatives and baseline noise, and verifies published runtime summaries against
their individual measurements. `--full` also reruns reference prefix fits and
120-draw bootstrap calculations. `--benchmark` measures the current computer;
its elapsed times and memory use need not match the publication's laptop.
`results/` contains the preserved manuscript exports; all reruns go to `outputs/`.

## Contents

| Folder | Contents |
|---|---|
| `notebooks/` | Executed step-by-step notebook |
| `src/` | Portable numerical algorithms and figure generators |
| `scripts/` | Reproduction and optional raw-extraction entry points |
| `data/reference/processed/` | Original reference processing exports and method comparisons |
| `data/validation/processed/` | Four validation corrected-response exports |
| `data/raw/` | Small original binary vendor kinetics records |
| `data/settings/` | Original processing windows, angular maps and bulk sensitivities |
| `results/` | Frozen numerical results used in the article and supplement |
| `figures/` | Published empirical figures and the editable model diagram |
| `manuscript/` | Manuscript source snapshots for result mapping (not a full LaTeX project) |

See `RESULTS_MAP.md` for every figure and table, and `DATA_DICTIONARY.md` for units
and channel mapping. `MANIFEST_SHA256.csv` records input and source checksums.

## Large angular scans

The primary workflow intentionally uses processed CSVs. Original angular `.spr2`
files are excluded at the experimenter's request because of their size. The
small `.bin` files contain processed vendor kinetics, not reflectance curves.
The CSV-only workflow does **not** regenerate angular extraction from raw scans.
The exact extraction implementation and saved settings are included, so a reader
with the original records can run:

```bash
python scripts/process_raw.py reference /path/to/reference.spr2
python scripts/process_raw.py validation /path/to/validation.spr2
```

This checks the reconstructed angles against the bundled CSVs. Reference angular
calibration was derived retrospectively and transferred to the validation run;
causality of the downstream prefix fit does not make that calibration prospective.

## Interpretation and provenance

Two measured serial-flow experiments used 1 mM Sulfo-NHS-SS-Biotin in PBS 1X,
10 microlitres/min and water rinsing, on different sensors. L1/L2 share physical
channel 1; L3/L5 share physical channel 2. Wavelengths are not independent sensor
replicates. Apparent tail rates are not intrinsic association/dissociation rates.
The validation duration follows the reference maximum point forecast; positive
retention is not itself proof of identical endpoints. No synthetic experimental
results or validation-run model outputs are included.

The existing numerical functions were retained, while filesystem paths and data
loading were adapted for this standalone package. `data_access.py` explicitly
loads the binary sensorgrams and archived metadata; it does not fabricate angular
scans. The new validation entry point contains only front and endpoint calculations.

Proceedings citation: Rodrigo T. Araújo, Arthur A. Melo and Antonio M. N. Lima,
"SPR virtual sensor: real-time kinetic estimation and active control via nonlinear
least squares," 2026 IEEE Sensors Applications Symposium (SAS), IEEE, pp. 1–6.
The journal manuscript has the additional author Mateus S. Marques.
