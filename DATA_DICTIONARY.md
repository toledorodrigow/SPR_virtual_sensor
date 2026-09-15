# Data and units

- `time_min`: minutes since acquisition start. Reference injection is
  2184.728 seconds (36.4121333 minutes); displayed times subtract this origin.
- `T_after_injection_min`: validation time since its recorded Biotin injection.
- `SPRAngleEmpirical`, `TIRAngleEmpirical`: degrees. Multiply by 1000 for mdeg.
- `corrected_response_mdeg`: baseline-zeroed SPR minus S times TIR, in mdeg.
- `S_bulk`: dimensionless angular sensitivity ratio, fixed per readout.
- `dSPR`, `dWeightedTIR`, `dCorrected`: trailing two-minute slopes in mdeg/min.
- `CorrectedLow95`, `CorrectedHigh95`: approximate HAC slope confidence limits.
- `kobs_per_s`: apparent fitted rate in inverse seconds; `tau_min` = 1/(60 kobs).
- `t99_experiment_min`: forecast on acquisition clock; subtract injection time.
- Endpoint CSV windows are minutes after each local TIR rinse midpoint.
- `short_*` fields denote measured validation-run outcomes, not new model fits.
- Source `channel_id` is zero-based: 0=L1, 1=L2, 2=L3, 4=L5, 5=L6, 6=L7.
- L1/L3: 670 nm; L2/L5: 785 nm; L6: 850 nm; L7: 980 nm.
- Noise is 1.4826 MAD of successive baseline differences divided by sqrt(2),
  evaluated from 10 through 30 minutes of the reference acquisition.
- Runtime latency is milliseconds; allocation and payload values are bytes.
  Residual arithmetic and exponential calls are counted separately, not as a
  complete hardware FLOP count.

Archived event/header JSON preserves instrument labels for provenance. These
labels are data, not additional assumptions about chemical arrival. Original
instrument records may contain acquisition operator names and device metadata.
