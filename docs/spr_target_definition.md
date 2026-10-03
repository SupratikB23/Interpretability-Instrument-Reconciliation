# SPR Target Definition (preregistered)

This file fixes the signal-to-perturbation-ratio (SPR) definitions, the
preregistered primary estimator, and the cross-instrument transfer decision
rule. Code: `src/spr/estimator.py`, `src/spr/transfer.py`.

## 1. Within-instrument SPR

For readout direction `v` (L2-normalized before projection), FP16 activations
`x_t`, and quantization-induced perturbation `Δ_t = x_t^quant - x_t^fp16`:

- **Signal statistic (primary):** `peak_abs_projection`
  `S_v = max_t |<v, x_t>|`
- **Perturbation statistic (fixed):** `mean_abs_directional_delta`
  `P_v = mean_t |<v, Δ_t>|`
- **Estimator:** `SPR_v = S_v / max(P_v, eps)` with `eps = 1e-8`
  (`estimate_spr_streaming`; activations are consumed in aligned chunks and
  never materialized as one token-by-hidden matrix).

Alternative signal statistics (`mean_abs_projection`, `rms_projection`) exist
for sensitivity analysis only; the preregistered estimator is
`peak_abs_projection/mean_abs_directional_delta`. Experiment profiles must set
`spr.preregistered_estimator` to exactly
`<signal_statistic>/<perturbation_statistic>` (config validation enforces the
identity).

## 2. Damage targets

Transfer targets are normalized damage in `[0, 1]` per instrument
(`src/spr/transfer.py`):

- SAE: `1 - clip(c_j, 0, 1)` from per-feature correlations.
- Probes: `clip((AUC_fp16 - AUC_quant) / scale, 0, 1)` with preregistered
  `scale` (config).
- Circuits: `1 - clip(stability_or_overlap, 0, 1)`.
- Patching: `clip(|effect_quant - effect_fp16| / scale, 0, 1)`.

## 3. Cross-instrument transfer test

**Design:** fit a model on the SOURCE instrument's `(SPR, damage)` pairs only;
score it, unchanged, on the TARGET instrument. Transfer is the target-scored
performance of the source-fitted mapping — never the target's own in-sample
correlation (that would be circular).

**Preregistered primary statistic:** calibration/error based,
because rank metrics (AUC, Spearman) are insensitive to calibration:

- binary target pairs: **negative log loss** of source-fit probabilities on the
  target (`neg_log_loss`);
- continuous target pairs: **negative RMSE** of source-fit predictions on the
  target (`neg_rmse`).

AUC and Spearman are still computed (`TransferResult.metrics`) as secondary
diagnostics only; they never decide `passes`.

**Decision rule:** a pair passes iff BOTH hold:

1. margin + significance vs. the permutation null (target outcomes shuffled,
   source fit unchanged): `observed > null_mean + margin` and
   `z >= z_threshold` (config `spr_transfer`: margin `0.05`, `z_threshold 2.0`,
   `n_null_shuffles 1000`);
2. the source-fit model **beats the constant source-prevalence/source-mean
   baseline** on the target (`beats_constant_baseline`). Rationale: a model
   with perfect target ranking but source-mismatched calibration beats shuffled
   targets while remaining useless; the constant-baseline requirement catches
   it.

**Baselines:** when callers supply bit-width, signal, and perturbation features
(`InstrumentData`), the preregistered baselines `constant`, `bit_width_only`,
`signal_only`, `perturbation_only` are fitted on the source and reported beside
the main `spr` model (`evaluate_transfer_models`), with grouped bootstrap CIs
resampling independent units (prompts/cells/features, never tokens).

## 4. What counts as evidence

- Reproduction claims require the reproduction gates (see
  `docs/scientific_contract.md`) to pass first.
- SPR transfer evidence is the set of ordered pairs passing the decision rule
  above, reported with observed/null statistics, z-scores, and secondary
  diagnostics; failures are reported, not hidden.
