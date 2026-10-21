# Result tables backing the paper

Summary tables only. The full artefacts (per-encoder probe results, cached
features, per-image predictions, checkpoints) stay under `$DATA_ROOT` and are
gitignored -- they are large and derived. These are committed so the reported
numbers travel with the code that produced them.

All accuracies are **test-split top-1**, from linear probes on a frozen encoder.
Splits are 75/15/10 stratified on species, seed 42, frozen in `labels.parquet`.

| file | contents |
|---|---|
| `headline_gap.csv` | The paper's main table: gap in percentage points at the final chunk of each configuration, per granularity. Positive = joint better. |
| `baseline_accuracy.csv` | Absolute accuracy of the two reference encoders (joint from-scratch, and off-the-shelf Meta DINOv2). The denominator for every relative figure. |
| `gap_table.csv` | Every encoder x granularity, with absolute metrics and gap against the joint baseline. 148 rows. |
| `mechanism.csv` | Per-encoder taxonomic error structure: mean hierarchical distance, share of species errors contained within the correct genus/family, and linear CKA against the joint encoder. |
| `stream_coverage.csv` | Per-chunk image and species counts for both stream orderings -- quantifies the coverage shift the seasonal arm introduces. |
| `lr_sweep.csv` | Abandoned adaptation experiment: 4 learning rates spanning 100x, none of which beat the un-adapted encoder. |
| `aug_test.csv` | Abandoned adaptation experiment: softened photometric augmentation and a backbone freeze. |

## Reading the gap

The gap is `joint - incremental` at the **final chunk** of a configuration, which
is the end state of the incremental stream. Earlier chunks have much larger gaps
simply because they have seen less data; the per-chunk trajectory in
`gap_table.csv` shows the recovery.

## Caveats that belong with the numbers

- **Single seed.** Each figure is a point estimate. The claim rests on the
  monotone trend across N in {2,4,8} replicated in two independent orderings,
  not on any one value.
- **Upper bound.** `checkpoint=` restores model weights only, so the learning
  rate schedule and warmup restart at every chunk boundary. The measured gap
  therefore includes a schedule-restart component and overstates the pure
  data-ordering effect.
- **Species is not the worst level.** Genus and family gaps exceed species. The
  supported claim is fine-grained versus coarse, not a gradient down the
  hierarchy.
- **The two orderings agree closely**, so the effect is driven by fragmentation
  of the stream rather than by distribution shift.
- `lr_sweep.csv` and `aug_test.csv` describe a **rejected** approach (adapting
  Meta's pretrained weights). They are kept as evidence for why the encoder is
  trained from scratch, not as results about the gap.
