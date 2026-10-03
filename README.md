# Which Interpretability Instrument Should You Trust?

**Reconciling contradictory verdicts on quantization damage across linear probes,
sparse autoencoders, circuits and activation patching, using the
signal-to-perturbation ratio (SPR).**

Interpretability instruments are increasingly used to check that a quantized model
still behaves like its full-precision source, yet they disagree. On Gemma-2-2B,
frozen linear probes report quantization as nearly harmless, while a frozen sparse
autoencoder (SAE) finds about half of its active features degraded at six bits, even
though perplexity barely moves.

This repository contains the full pipeline used to:

1. **Measure the divergence across four instruments** (probes, SAE features,
   CIRCUS-style circuits, NIE/PIE activation patching) under one fixed quantization
   substrate.
2. **Test a single geometric account of it**: an instrument's robustness should be
   predictable from the signal-to-perturbation ratio along the direction(s) it
   reads out, computed from FP16 statistics alone.

---

## Table of contents

- [Background](#background)
- [The SPR hypothesis](#the-spr-hypothesis)
- [The four instruments](#the-four-instruments)
- [Experimental substrate](#experimental-substrate)
- [Key findings](#key-findings)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation protocol](#evaluation-protocol)
- [Limitations](#limitations)
- [References](#references)

---

## Background

| Source | Instrument | Verdict | Key number |
|---|---|---|---|
| Duan 2026a, arXiv:2606.15980 | Frozen linear probes | Nearly harmless | 0.83–3.33% big-drop rate, 0% operational failures across 840 quantized cells |
| Duan 2026b, arXiv:2606.03002 | Frozen Gemma Scope SAE (layer 12, d_sae = 16,384) | Serious | INT6: only 51.3% of active features survive, while perplexity changes by +3.99% |

Feature survival at INT6 is predictable from FP16-only statistics (AUC 0.971 on
Gemma-2-2B), with peak activation the strongest single predictor. That observation
is the empirical seed for the SPR hypothesis below.

## The SPR hypothesis

> An interpretability instrument's robustness to a perturbation is predictable from
> the signal-to-perturbation ratio along the direction(s) it reads out.

```
SPR_i(C) = |<v_i, mu_i>| / E_t[ || Delta_C(t) ||_{v_i} ]
```

- `v_i`: the instrument's readout direction (probe normal, SAE decoder row, ...)
- `mu_i`: its typical FP16 signal along that direction
- `Delta_C(t)`: the activation perturbation that compression operator `C` induces at token `t`

**Intuition.** A probe reads one supervised, high-amplitude direction, so its SPR is
high and it is robust. An SAE tiles the whole residual stream, including many
low-amplitude features whose SPR is low, which produces the large degraded band.

**Falsifiable claims.**
- *Within an instrument:* higher SPR means less damage.
- *Across instruments:* an SPR model fit on one instrument predicts damage on another.

A null result on the second claim is still informative: it would show that the
instruments diverge for reasons specific to each one.

## The four instruments

| Arm | Module | Protocol source | Output |
|---|---|---|---|
| Linear probes | `src/probes/` | Duan 2606.15980, re-run under RTN | ΔAUC per bit width, big-drop and operational-failure rates |
| SAE features | `src/sae/` | Duan 2606.03002 QDM pipeline, same thresholds | Per-feature correlation c_j and survived / degraded / damaged shares |
| Circuits | `src/circuits/` | CIRCUS stability ensembles (Parekh, arXiv:2603.00523) | Edge inclusion-frequency shifts, never single-threshold circuits |
| Activation patching | `src/patching/` | NIE/PIE decomposition (Vaidyanathan et al., arXiv:2606.27510) | ΔNIE and ΔPIE reported separately, disagreements flagged |

SAE survival taxonomy (reused exactly from Duan 2606.03002): **survived** c_j > 0.9,
**degraded** 0.5 ≤ c_j ≤ 0.9, **damaged** c_j < 0.5, scored only on features with
FP16 firing rate above 0.001. Correlations use a streaming sufficient-statistics
estimator with O(d_sae) memory.

## Experimental substrate

Held constant across all four arms:

| Component | Choice |
|---|---|
| Primary model | Gemma-2-2B (Pythia-70M-deduped as a robustness model) |
| Compression | Round-to-nearest (RTN), per-output-channel, bit widths {8, 7, 6, 5, 4} |
| Quantized modules | Attention Q/K/V/O and MLP gate/up/down projections; embeddings, unembeddings, norms and biases untouched |
| Read-out | Layer-12 residual stream |
| SAE | Gemma Scope, layer 12, width 16k (canonical L0) via SAELens |
| Corpora | WikiText-2-raw (perplexity and SAE reproduction), FineWeb-Edu (harmonized default) |

The probe arm is re-run under the same RTN sweep as the SAE arm, so that the
instrument comparison is not confounded with a comparison of quantization methods.

## Key findings

All numbers come from runs of this pipeline on one 8 GB consumer GPU.

**1. Reproduction and the BOS effect.** With flat 256-token blocks and no
beginning-of-sequence (BOS) token, the SAE keeps **54.84%** of 7,141 active features
at INT6 (published: 51.30% of 7,144), and the whole survival curve tracks the
published one. A 2×2 isolation crossing tokenization with forward-pass BOS shows that
the BOS token alone moves INT6 survival by about 44 points (≈98.7% with BOS),
while tokenization moves it by at most 0.5 points.

**2. Four-instrument divergence (no-BOS substrate).**

| bits | SAE non-survival % | probe big-drop % | ΔPPL % |
|---|---|---|---|
| 8 | 0.28 | 0.0 | +4.64 |
| 7 | 10.64 | 0.0 | −1.77 |
| **6** | **45.16** | **0.0** | **+2.01** |
| **5** | **70.03** | **0.0** | **+4.76** |
| 4 | 86.81 | 33.3 | +55.11 |

The preregistered divergence gate (SAE non-survival ≥ 25%, probe big-drop ≤ 10%)
fires at INT6 and INT5, and at no bit width when BOS is present.

**3. SPR.**

| instrument | SPR points | Spearman ρ (SPR vs damage) |
|---|---|---|
| SAE | 35,705 | −0.83 |
| probe | 75 | −0.75 |
| circuits | 323,295 | −0.90 |
| patching | 5 | −0.82 (not significant) |

- **Within instruments:** SPR predicts damage in all four, on both substrates.
- **Across instruments:** calibrated predictions transfer for 0 of 12 ordered pairs
  with raw SPR, and 3 of 12 with per-instrument standardized SPR.
- **Natural intervention:** removing BOS halves median INT6 SPR (14.4 → 7.3) on
  6,307 shared features, and the per-feature SPR drop predicts the survival drop
  (partial ρ = 0.40).

## Repository layout

```
├── README.md
├── pyproject.toml
├── requirements.txt
├── run_gates.py                  # reproduction-gate runner (exit 0 = all pass)
├── r1/                           # shim so `python -m r1.run` works from the root
├── configs/
│   ├── model_gemma2_2b.yaml      # primary model, SAE checkpoint, corpora
│   ├── model_pythia70m.yaml      # secondary / robustness model
│   ├── quant_rtn_sweep.yaml      # bit widths, gates, taxonomy, CIRCUS and SPR settings
│   ├── reproduction/             # Duan-protocol profiles, incl. BOS / tokenization isolation
│   ├── harmonized/               # common-substrate profiles (with BOS, no-BOS flat stream)
│   └── smoke/                    # tiny end-to-end pipeline check
├── docs/
│   ├── scientific_contract.md    # run modes, provenance, artifact layout, gates
│   └── spr_target_definition.md  # preregistered SPR estimator and transfer rule
├── scripts/
│   ├── build_measurement_bank.py # shared prompt/token pool for all instruments
│   ├── export_derived.py         # compact CSV/JSON exports for plotting
│   └── plot_*.py                 # figures (damage curves, ΔPPL, SPR within / transfer)
└── src/
    ├── config.py                 # YAML + argparse config loader
    ├── manifest.py               # run manifests and provenance
    ├── compression/              # RTN per-output-channel quantization
    ├── probes/                   # probe training and evaluation
    ├── sae/                      # QDM pipeline, streaming correlation, survival taxonomy
    ├── circuits/                 # CIRCUS stability ensembles over attribution graphs
    ├── patching/                 # NIE / PIE activation patching
    ├── spr/                      # SPR estimator, transfer tests, BOS intervention
    └── r1/                       # staged experiment runner
```

Every module that reproduces a published method cites the arXiv ID and
section/equation it follows. Each arm ships `test_*.py` files covering its logic.

## Installation

```bash
pip install -r requirements.txt

# Anthropic's circuit-tracer is not on PyPI
pip install git+https://github.com/safety-research/circuit-tracer.git

# Gemma-2-2B and two probe datasets are gated on Hugging Face.
# Accept the terms for google/gemma-2-2b, walledai/XSTest and allenai/wildguardmix, then:
huggingface-cli login
```

Requirements: Python ≥ 3.10, PyTorch, and a CUDA GPU for real model runs. An 8 GB
card (e.g. RTX 3070) is enough: the SAE sweep keeps one model copy on the GPU and
the FP16 state on CPU. Keep `transformers` on 4.57.x for circuit-tracer
compatibility.

## Usage

All protocol choices live in `configs/*.yaml`. Nothing is hardcoded.

```bash
python -m pytest                       # test suite
python -m r1.run verify-env            # dependency and GPU report
python -m r1.run audit                 # RTN module selection check on the real model
python -m r1.run smoke-all --config configs/smoke/tiny.yaml   # tiny end-to-end run

# SAE arm (reproduction) and gate 1
python -m r1.run reproduce-sae --config configs/reproduction/sae_gemma2_2b_duan_exact.yaml
python -m r1.run gate-sae

# Probe arm and gate 2
python -m r1.run validate-probe-datasets --config configs/reproduction/probes_gemma2_2b_it.yaml
python -m r1.run reproduce-probes --config configs/reproduction/probes_gemma2_2b_it.yaml
python -m r1.run gate-probes

# Common substrate and divergence gate
python -m r1.run harmonized-sae    --config configs/harmonized/common_gemma2_2b_flat_stream.yaml
python -m r1.run harmonized-probes --config configs/harmonized/common_gemma2_2b_flat_stream.yaml
python -m r1.run divergence-gate

# Circuits and patching
python -m r1.run circuits --config configs/harmonized/common_gemma2_2b_flat_stream.yaml
python -m r1.run patching --config configs/harmonized/common_gemma2_2b_flat_stream.yaml

# Cross-instrument SPR
python -m r1.run spr-build             # validate the four SPR artifacts
python -m r1.run spr-within            # within-instrument association
python -m r1.run spr-transfer          # fit on instrument A, test on instrument B
python -m r1.run spr-loio              # leave-one-instrument-out
```

**Recommended order:** SAE → gate 1 → probes → gate 2 → circuits → patching → SPR.
A Pythia-70M-deduped pilot (`configs/model_pythia70m.yaml`) is a cheap way to check
the pipeline before the Gemma-2-2B run. Outputs are written to `results/`, which is
not tracked, and figures are regenerated from it with `scripts/export_derived.py`
followed by `scripts/plot_*.py`.

## Evaluation protocol

All criteria are fixed before running (see `docs/spr_target_definition.md`).

- **Reproduction gate 1 (SAE):** INT6 survival within ±3 pp of 51.30% / 48.70% / 0.00%.
  On the no-BOS protocol the run gives 54.84%, which is 0.54 pp outside the window.
  The gate is reported as failing rather than relaxed.
- **Reproduction gate 2 (probes):** big-drop rate inside the 0.83–3.33% reference
  interval, with zero operational failures.
- **Within-instrument SPR:** the Spearman correlation must beat a permutation null.
- **Cross-instrument transfer:** a pair passes only if its RMSE beats a shuffled
  null by more than 0.05 with z ≥ 2 *and* beats a constant baseline. A
  bit-width-only baseline is reported alongside every result.

## Limitations

- SAE survival describes this Gemma Scope dictionary, not the model's "true"
  features (Leask et al. 2025). Results are stated as rankings or as conditioned
  levels.
- Patching currently has five SPR points (one site, the full residual stream), so
  its correlation is underpowered and NIE equals PIE by construction.
- Probes are scored on monitor text and the SAE on WikiText, so part of the
  instrument gap may be a corpus gap. Separating the two is ongoing work.
- Results cover one model family and one compression operator (RTN). Pruning and a
  second architecture are planned.

## References

1. Duan, E. (2026). *Do Activation Monitors Survive Model Updates? Benchmarking, Predicting, and Repairing Activation-Monitor Staleness.* arXiv:2606.15980.
2. Duan, E. (2026). *Perplexity Can Miss SAE Feature Damage Under Quantization.* arXiv:2606.03002.
3. Borobia, H., Seguí-Mas, E., & Tormo-Carbó, G. (2026). *How Pruning Reshapes Features.* arXiv:2603.25325.
4. Parekh, S. (2026). *CIRCUS: Circuit Consensus under Uncertainty via Stability Ensembles.* arXiv:2603.00523.
5. Vaidyanathan, S., et al. (2026). *The Curse of Multiple Mediators.* arXiv:2606.27510.
6. Leask, P., et al. (2025). *Sparse Autoencoders Do Not Find Canonical Units of Analysis.* ICLR 2025.
7. Lieberum, T., et al. (2024). *Gemma Scope.* arXiv:2408.05147.
8. Ameisen, E., Lindsey, J., et al. (2025). *Circuit Tracing: Revealing Computational Graphs in Language Models.* Transformer Circuits Thread.
9. Wang, K., et al. (2022). *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small.* arXiv:2211.00593.
10. Sun, M., et al. (2024). *A Simple and Effective Pruning Approach for Large Language Models.* ICLR 2024.
