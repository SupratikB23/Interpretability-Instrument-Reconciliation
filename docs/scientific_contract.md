# R1 Scientific Contract

This document is the binding contract for what each run mode may claim, what
provenance is required before it may run, and where its artifacts live. Code
enforces this contract (`src/sae/protocol.py`, `src/config.py`,
`src/r1/run.py`); the text here explains the intent.

## 1. Experiment modes

| Mode | Meaning | May claim reproduction? | Artifact root |
|---|---|---|---|
| `reproduction` | Re-run a published result on the same model/checkpoint/corpus/tokenization as the reference | Yes, if the gate passes | `results/reproduction/` |
| `harmonized` | R1's own configured comparison substrate (common corpus, RTN sweep) across instruments | No — directional/comparative claims only | `results/harmonized/` |
| `smoke` | Tiny-budget pipeline validation | Never | `results/smoke/` |

Mode is part of every artifact path and every result row. A harmonized or
smoke artifact can never satisfy a reproduction gate: `run_gates.py` filters
probe cells to `mode == "reproduction"` and fails loudly if none exist, and SAE
reproduction requires `ReferenceProvenance` (below).

## 2. Provenance requirements

### Reproduction (`ReferenceProvenance`)

All of the following must be resolved (non-empty) before a reproduction SAE run
executes; `validate_protocol` raises and lists every unresolved field:

- `source_reference` — the reference repository/commit or paper section the run
  reproduces.
- `corpus_id`, `corpus_config`, `corpus_split`, `corpus_revision` — the exact
  activation corpus.
- `tokenization_protocol`, `tokenizer_id`, `tokenizer_revision`, `bos_policy`,
  `sequence_packing` — the exact token stream construction.
- `sae_release`, `sae_id`, `sae_revision` — the exact frozen SAE checkpoint.

Guessing or substituting plausible values is forbidden: an unresolved field is
a hard stop, not a warning.

### Harmonized (`HarmonizedProvenance`)

Requires the configured common-corpus identity (`corpus_id`, `corpus_split`,
`tokenization_protocol`, `tokenizer_id`, `bos_policy`, `sequence_packing`);
revision fields may be null. Supplying reference reproduction provenance to a
harmonized run is rejected so artifacts cannot be mislabeled.

## 3. Artifact layout

```
results/
  reproduction/
    sae/       survival_rates.csv, correlations_int6.npz, perplexity.csv,
               metadata.json, quantization_audit_int<b>.json
    probes/    auc_calibration.csv, metadata.json
    audit/     quantization_audit_int<b>.json
  harmonized/
    sae/       (same as above)
    probes/    (same as above)
    circuits/  edge_stability.csv, stability_configs.csv, metadata.json
    patching/  nie_pie.csv, metadata.json
    spr/       within_instrument.json, transfer_results.json,
               instruments/<instrument>.npz
  smoke/       (any of the above; never scientifically valid)
```

Gates read the reproduction family only (`configs/quant_rtn_sweep.yaml`
`results:` block names every path).

## 4. Run metadata

Every SAE sweep persists `metadata.json` with one entry per condition
(`src/sae/metadata.py::RunMetadata`): mode, full provenance, model/SAE
identifiers and revisions, hook, token budget, sequence length, perplexity
window/stride, bits, seed, the complete quantization audit, device, dtype,
config path, and repository root. Probe sweeps persist an equivalent
`metadata.json` (`src/probes/run.py::_write_metadata`).

## 5. Gates

1. **SAE INT6 reproduction gate** — survived/degraded/damaged within ±3pp of
   51.30 / 48.70 / 0.00 (`src/sae/gates.py`).
2. **Probe big-drop reproduction gate** — strict semantics:
   observed big-drop rate must fall **inside** [0.0083, 0.0333]; eligible-cell
   count/coverage minimums and zero operational failures (post-update AUC <
   0.70) are enforced. The legacy 10x order-of-magnitude band is not supported
   anywhere (`src/probes/gates.py`).

Gates are real assertions run by `python run_gates.py` or
`python -m r1.run gate-sae|gate-probes`; missing artifacts fail loudly.

## 6. Budgets

Every experiment profile declares `budgets.dry_run/smoke/full` with
`max_examples` and `max_tokens`; validation requires `smoke.max_tokens <
full.max_tokens`. Smoke runs use the smoke budget and route to `results/smoke/`.

## 7. Seeds

Every stochastic step takes an explicit seed; experiment profiles declare the
seed list (`seeds:`) and the runner defaults to its first element unless
`--seed` overrides it.
