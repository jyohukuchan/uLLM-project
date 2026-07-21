# Importance-score C5 gradient extension

## Scope and invariants

- Starting point: git `532d488b` (C0/C1/C4/C6 two-model decision was NO-GO).
- Add only C5 Taylor/Fisher measurements. Existing score formulas, values, labels,
  quantizers, kernels, and runtime dispatch remain unchanged.
- GPU execution is restricted to `HIP_VISIBLE_DEVICES=1`. The
  `ullm-openai.service` lifecycle is outside this work and is not touched.
- Qwen is the development model. Score formulas, Monte Carlo count, reporting
  thresholds, and implementation hashes are frozen before any new Gemma C5
  score/label join.

## Implementation checkpoint

- Added an independent, label-blind `D_fisher` freezer. It selects 128 raw
  records (approximately 16k tokens at sequence length 128) from the same local
  source pool after excluding every frozen base-split record ID and normalized
  content hash. It emits four deterministic shards and does not rewrite the
  existing corpus manifest or split files.
- Extended the formal report with five C5 rows. Taylor-quant and self-Fisher are
  winner-eligible; Taylor deletion L1/squared and empirical Fisher are secondary.
- Preserved the six v0.1 scores as their existing Benjamini-Hochberg family and
  placed conditional C5 candidates in a separate family, preventing retroactive
  changes to old adjusted p-values.
- Added exact legacy-metric invariance checks against the sealed v0.1 metrics.
- Added a label-free prejoin extension that verifies and byte-preserves the old
  score rows, then appends C5 values with exact tensor and four-shard coverage.
- CPU test checkpoint: `65 passed` across the C4 regression, C5 formula/runner,
  prejoin extension, active lockbox view, and formal statistics suites.

GPU smoke tests and full-run timings will be appended after the collector and
lockbox chain pass CPU validation.
