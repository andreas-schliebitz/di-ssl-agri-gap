# Mind the Gap: Joint vs. Data-Incremental Self-Supervised Pretraining for Plant Classification

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

Code, configurations and result tables for the paper of the same name. The manuscript
source is in [`docs/paper/main.tex`](docs/paper/main.tex); see
[CITATION.cff](CITATION.cff) for how to cite this work.

Agricultural image collections grow continuously, which makes pretraining a visual
encoder *incrementally* on arriving data attractive compared with repeating *joint*
pretraining over the whole accumulated corpus. This repository measures what that
convenience costs and shows that the answer depends almost entirely on **how
fine-grained the downstream labels are**. Because a single aggregate probe accuracy
(the convention in the continual self-supervised literature) cannot express that
distinction, it conceals the effect.

## Headline result

DINOv2 ViT-S/14 pretrained from random initialisation on iNatAg-mini, jointly and as
chained streams of *N* segments, then frozen and evaluated by linear probes at four
nested taxonomic levels. Both arms spend **219,000 optimiser steps** and see every
image **100 times**, so the comparison isolates presentation order from compute.

Gap to joint pretraining in accuracy points at the final segment, positive means joint
is better ([`docs/results/headline_gap.csv`](docs/results/headline_gap.csv)):

| Ordering | *N* | species (2,958) | genus (1,171) | family (215) | crop/weed (2) |
|---|--:|--:|--:|--:|--:|
| i.i.d.   | 2 | 1.79 | 2.06 | 1.74 | 0.73 |
| i.i.d.   | 4 | 3.20 | 4.52 | 4.35 | 0.94 |
| i.i.d.   | 8 | **7.06** | **8.84** | **8.23** | **1.81** |
| seasonal | 2 | 1.32 | 2.17 | 2.16 | 0.99 |
| seasonal | 4 | 3.33 | 4.15 | 4.55 | 1.35 |
| seasonal | 8 | 7.22 | 9.06 | 8.55 | 2.33 |

At eight segments the penalty is **7.06 points at species level against 1.81 for crop
versus weed**, which is 18.6 % against 2.4 % relative to the joint baseline, a factor of
roughly eight. It grows monotonically with segment count under both orderings. An
unweighted mean over the four levels reports 6.49 points, overstating the crop/weed
penalty 3.6-fold and understating the genus penalty 1.4-fold.

Three findings qualify the naive reading of that table:

- **Species is not the worst-affected level.** Genus (8.84) and family (8.23) exceed it.
  The supported claim is a division between fine-grained and coarse targets, not a
  gradient tracking class count.
- **Seasonal ordering behaves almost identically to the coverage-preserving one**
  (7.22 against 7.06 at species), which locates the cause in *fragmentation* of the
  stream rather than in distribution shift within it.
- **The mechanism is representational drift.** Mean hierarchical error distance rises
  1.495 to 1.745, containment of species errors within the correct genus falls 0.190 to
  0.145 and linear CKA against the joint encoder decreases 0.667 to 0.607.

## Repository layout

```
di_ssl_agri_gap/          the pipeline, one module per stage
  enums.py                Granularity, Arm, Ordering, Split
  splits.py               seeded stratified splitting, returns index arrays
  taxonomy.py             the four-level taxonomy, AgML family join
  utils.py                settings from .env, seeding, loggers, provenance
  prepare.py              stage 1-2   extract 2,958 archives, verify, delete
  manifest.py             stage 3     labels.parquet, the single source of truth
  resize.py               stage 4     re-encode to 256px short side
  stage_shm.py            stage 5     copy to /dev/shm, build train-only view
  pretrain.py             stage 6     joint pretraining
  embed.py                stage 7     frozen feature extraction, once per encoder
  probe.py                stage 8     four linear probes on cached features
  stream.py               stage 9     segment assignment, both orderings
  incremental.py          stage 10    chained pretraining over segments
  analysis.py             stage 11    gap tables and the three drift metrics
  lr_sweep.py             ablation    learning-rate sweep for weight adaptation
  aug_test.py             ablation    softened photometric augmentation
  audit.py                completeness audit and MLflow resync
  supervise.py            watchdog for multi-day runs
  main.py                 orchestration entry point
tests/                    93 tests, including the invariants listed below
docs/
  paper/main.tex          the manuscript (LNI class, biblatex + biber)
  paper/bibliography.bib  29 cited references, each verified against Crossref or arXiv
  results/*.csv           the tables backing every number in the paper
  abstract.md             extended abstract
  lightlytrain-findings.md  library behaviours established empirically, see below
```

## Installation

Python 3.12. The training stack pins torch 2.9.1 / torchvision 0.24.1 (cu128 wheels),
verified against a 535.261.03 driver (CUDA 12.2).

```bash
python3.12 -m venv venv && source venv/bin/activate
./install.sh          # editable install, then freezes requirements.lock
```

`install.sh` refuses to run outside an active virtual environment. It writes
`requirements.lock` from `pip freeze`, which is what pins the environment here:
Poetry declares the dependency ranges in `pyproject.toml` but does not manage this venv.

Then configure paths:

```bash
cp .env.example .env && chmod 600 .env
```

`INATAG_ZIP_DIR` points at the read-only iNatAg-mini snapshot, `DATA_ROOT` at a working
directory with roughly 300 GB free and `SHM_ROOT` at a `tmpfs` mount.

**MLflow is optional.** `USE_MLFLOW` defaults to `false`, in which case runs log to
`logs/` and CSV plus Parquet only. Every probe result is written to
`$DATA_ROOT/results/` *before* anything is sent to a tracking server, so a network
failure at hour 200 of a 400-hour matrix cannot lose a measurement.

## Data

iNatAg-mini, the reduced release of iNatAg capping each species at 200 images. Obtain
the HuggingFace snapshot of 2,958 per-species zip archives (237 GB compressed, 254.1 GB
extracted) and point `INATAG_ZIP_DIR` at it.

Counts below were read first-hand from all 2,958 archives rather than taken from the
dataset card, which is why a few of them differ from it:

- **560,846 images**, against 560,844 in the dataset README and 560,849 summed from AgML
  metadata. `.jpeg` 359,563 / `.jpg` 200,147 / `.png` 1,136. Crop 369,718 / weed 191,128.
  Per species: minimum 50, median 200, maximum 200.
- **`annotations.json` is a superset of the delivered images**: 641,951 records for
  560,846 files, with 100 species carrying more records than images. The join must be
  keyed on `photo_id`; anything positional or count-based is silently wrong. 560,841
  images match a record, the 5 that do not are dropped.
- **27 images (0.005 %) carry truncated JPEG payloads** upstream from iNaturalist.
  Extraction verifies every file against its archive's central directory, so the
  corruption is not introduced here. They are decoded with missing scanlines filled grey
  rather than dropped, because removing rows would shift every index in the frozen split.
  Listed in `$DATA_ROOT/truncated_images.txt`.
- **Split**: 420,630 / 84,126 / 56,085 (75/15/10), stratified on species, seed 42. Since
  the hierarchy is strictly nested, all 2,958 species, 1,171 genera, 215 families and
  both crop/weed classes appear in *every* part, so one split serves all four probes.
- **`abrus_pecatorius`** is the only species AgML files under family `Unknown`. It is a
  misspelling of *Abrus precatorius* and is remapped to Fabaceae, the family its
  genus-mate carries, which keeps the family probe a clean 215-way problem.
- Family labels exist **nowhere** in the directory structure. AgML's bundled metadata is
  their only source, which is why `agml` is a hard dependency rather than a convenience.

## Reproducing the paper

Stages are separate subcommands rather than one pipeline, because the expensive ones run
for days and need to be launched, monitored and resumed independently. Subcommands that
delegate to a stage module take their own flags after `--`.

```bash
python -m di_ssl_agri_gap.main --help
```

### One-time data preparation (about 25 minutes on 96 cores)

```bash
python -m di_ssl_agri_gap.main prepare  -- --workers 64 --delete-zips
python -m di_ssl_agri_gap.main manifest -- --workers 64
python -m di_ssl_agri_gap.main resize   -- --workers 96
python -m di_ssl_agri_gap.main stage
```

`--delete-zips` removes the source archives and fires only if every archive verified
clean. It is a separate flag precisely so it cannot happen by accident; extraction and
verification are never fused with deletion.

`stage` copies the resized tree into `/dev/shm` and builds `img256_train/` as a hardlink
view containing **only** train-split images. That view is what pretraining reads and is
the mechanism enforcing that validation and test images never enter an encoder, even
as unlabelled pixels. Set equality against the manifest is asserted in the tests.

### Phase 1, joint baseline (41.9 h on 2x A100-40GB)

```bash
python -m di_ssl_agri_gap.main pretrain -- --epochs 100 --batch-size 192 --devices 2

python -m di_ssl_agri_gap.main evaluate \
    --encoder-id joint_vits14scratch_ep100_seed42 \
    --checkpoint out/joint_vits14scratch_ep100_seed42/exported_models/exported_last.pt
```

### Phase 2, data-incremental arms (about 280 h)

```bash
python -m di_ssl_agri_gap.main incremental --ordering iid      --n-chunks 2 4 8
python -m di_ssl_agri_gap.main incremental --ordering seasonal --n-chunks 2 4 8
```

Each configuration writes `out/chain_<ordering>_n<N>_seed42.json` recording realised
steps per segment against the joint budget. Segment 0 starts from the same random
initialisation as the joint arm, which is what makes the gap attributable.

### Evaluation and analysis (about 10 h for 30 encoders)

```bash
python -m di_ssl_agri_gap.main evaluate-all   # resumable, skips finished encoders
python -m di_ssl_agri_gap.main analyse
```

`evaluate-all` extracts features once per encoder and fits four probes on the cache,
sweeping the probe learning rate over six values, selecting on validation and reporting
test. It isolates failures so one bad encoder cannot cost the other 29 their evaluation.
`analyse` writes `gap_table.parquet` and `mechanism.parquet` under `$DATA_ROOT/results/`;
the committed snapshots in `docs/results/` are their CSV mirrors.

### The rejected ablations

Both are kept because they justify a design decision, not because they measure the gap:

```bash
python -m di_ssl_agri_gap.lr_sweep --epochs 10
python -m di_ssl_agri_gap.aug_test --epochs 10
```

### Building the manuscript

`lni.cls` v2.0 loads biblatex with the biber backend, so this is **not** BibTeX and there
is no `.bst`:

```bash
export PATH=/cluster/apps/texlive/2026/bin/x86_64-linux:$PATH
cd docs/paper && pdflatex main && biber main && pdflatex main && pdflatex main
```

Both figures are pgfplots with coordinates inlined from `docs/results/*.csv`, so there
are no external asset files to resolve. DOIs are present in `bibliography.bib` for
archival value but do not render, because `doi=false` is an LNI style default.

## Results

[`docs/results/`](docs/results/) holds the summary tables, each a direct input to a
number in the paper. Full artefacts (per-encoder probe results, cached features,
per-image predictions, checkpoints) stay under `$DATA_ROOT` and are gitignored: they are
large and derived. See [`docs/results/README.md`](docs/results/README.md) for the caveats
that belong with the numbers.

| file | contents |
|---|---|
| `headline_gap.csv` | Gap in points at the final segment of each configuration, per granularity. The paper's main claim. |
| `baseline_accuracy.csv` | Absolute accuracy of the joint from-scratch and released DINOv2 encoders. |
| `gap_table.csv` | Every encoder times granularity, absolute metrics and gap against the joint baseline. 148 rows. |
| `mechanism.csv` | Per-encoder taxonomic error structure and linear CKA against the joint encoder. |
| `stream_coverage.csv` | Per-segment image and species counts for both orderings. |
| `lr_sweep.csv`, `aug_test.csv` | The rejected adaptation experiments. |

## Verification

```bash
pytest        # 93 tests
```

The tests that matter for trusting the results are the invariants, not the unit tests:

- **Split integrity**: every species, genus and family survives every split;
  re-running with the same seed reproduces the split exactly.
- **No leakage**: `img256_train/` is set-equal to the manifest's train rows.
- **Partition invariants**: segments are disjoint, their union is exactly the train
  split, sizes differ by at most *N* images and under i.i.d. ordering every segment
  contains all 2,958 species with per-class counts differing by at most one. This is a
  regression test against the partitioner bias described below.
- **Hemisphere flip** moves exactly the `latitude < 0` rows and is an involution.
- **Compute matching**: realised total steps per chain against the joint budget.
- **Self-containment**: no imports from sibling repositories, no `git =` or `path =`
  dependency sources in `pyproject.toml`.
- **Gap computation** resolves to a single joint reference encoder rather than averaging
  over several, which an earlier version silently did.

## Compute budget

Measured on 2x A100-40GB, ViT-S/14 with the DINOv2 multi-crop (2x224 plus 8x98):

| Stage | Measured |
|---|--:|
| Extract and verify 2,958 archives | 6 min |
| Resize to 256px | 12 min |
| Stage to `/dev/shm` and build train view | 3.5 min |
| **Joint pretraining**, 219,000 steps | **41.9 h** |
| **Incremental**, 28 runs across 6 chains | **~280 h** |
| Adaptation ablations, 7 runs | 56.7 h |
| Evaluation, 30 encoders | ~10 h |
| **Total** | **~16 days of 2-GPU time** |

Wall-clock for the incremental arm was 307 h, of which about 28 h was one segment
restarted after an NCCL collective timeout. Training is GPU-bound rather than data-bound:
96 % GPU utilisation with the CPU 96 % idle and nearly all dataloader workers asleep.
LightlyTrain's `data_wait` counter reads about 32 % but adding workers does not help.

## Design decisions worth knowing

Each of these was settled by measurement and each changes how the results should be
read. [`docs/lightlytrain-findings.md`](docs/lightlytrain-findings.md) documents the
library behaviours in full.

**The encoder is trained from scratch, not adapted from released DINOv2 weights.**
Adaptation was attempted across four learning rates spanning two orders of magnitude and
two augmentation variants: **no configuration matched the un-adapted encoder** at
species level: the best reached 40.40 % against 54.45 %. A 100-epoch run at the reference
rate reached 39.81 %, placing it among the short runs rather than above them, so the
ceiling is near 40 % rather than a matter of undertraining. 560k images at roughly 190
per species are too few for this objective to improve on a representation distilled from
142M. A joint baseline weaker than doing nothing would make the gap uninterpretable, so
`MODEL = "dinov2/vits14-notpretrained"`. The released encoder becomes context in the
results table rather than a headroom bound.

**Not DINOv3.** LightlyTrain 0.17.0 cannot train a DINOv3 architecture with an
EMA-teacher objective: `_methods/dinov2/dinov2.py` requires a `DINOv2ViTModelWrapper`, so
`dinov3/*` is reachable only through distillation from a *frozen* Meta teacher. A frozen
teacher is a fixed target, which would mute exactly the representation drift this study
measures.

**Continuum's default partitioner is not used.** `InstanceIncremental` supplies the
scenario semantics, but its `random_seed` does not influence the partition (the random
state is constructed after `_setup` runs and `_split_dataset` never shuffles) and all
per-class remainders land in task 0, leaving it roughly 22 % oversized at *N*=8. That
would break compute matching and hand the first segment an advantage, so the segment
assignment is computed here and passed in through Continuum's documented custom-task-id
branch.

**Seasonal ordering is by month of year, not by absolute date.** `observed_on` is
99.61 % populated but severely front-loaded: 91.5 % of the corpus falls in 2020 to 2025
and 50.9 % in 2024 alone. Equal-count date blocks would put five of eight segments inside
a single year, so the arm would degenerate into i.i.d. after the first boundary.
Month-of-year is near-uniform by comparison and southern-hemisphere months are shifted
by six so northern and southern seasons do not cancel. The 1,645 training images without
a usable date are excluded from this arm only.

The ordering deliberately breaks per-segment class coverage, which is the shift it exists
to introduce ([`docs/results/stream_coverage.csv`](docs/results/stream_coverage.csv)):

| *N* | seasonal species coverage | i.i.d. |
|--:|---|---|
| 2 | 99.3 to 99.6 % | 100 % |
| 4 | 95.2 to 98.1 % | 100 % |
| 8 | 87.4 to 94.1 % | 100 % |

At *N*=8 the first segment sees 2,584 of 2,958 species, so 374 are entirely absent.

**Global batch is 192 (96 per GPU), not 256.** The iBOT loss masks patches with a
variable ratio, so the `log_softmax` allocation in `dinov2_loss.forward_masked` is
variable-sized per batch. At 128 per GPU that is about 94 % of the card and an OOM becomes
a coin flip on the mask draw: one 100-epoch run survived, a later identically configured
run died at step 0. Any batch size is defensible provided both arms share it.

**Macro accuracy is not top-1.** `torchmetrics.MulticlassAccuracy` defaults to
`average="macro"`. Over 2,958 classes of 50 to 200 images that differs substantially from
top-1 while the iNatAg reference figures are micro accuracy, so both are reported at every
granularity.

**Determinism is scoped.** Linear probing runs under full determinism
(`seed_rng(strict=True)`). Pretraining does not, because several DINOv2 multi-crop and
attention kernels have no deterministic implementation; it is seed-controlled but not
bitwise reproducible; applying strict determinism to it crashes the run.

**The iNatAg reference figures are not comparable.** The 79.40 / 89.83 / 94.07 % numbers
from arXiv:2503.20068 are fine-tuned Swin-L plus LoRA with geospatial inputs, on a
2,059-species / 2.06M-image subset of *full* iNatAg, not linear-probed and not
iNatAg-mini. They indicate the shape of the granularity curve, not a number these probes
can be measured against.

## Limitations

- **Single seed.** Every value is a point estimate. The evidence carrying the claim is
  the monotone trend across *N* in {2, 4, 8} replicated under two independent orderings,
  not any individual measurement. This matters most at *N*=2, where the gaps are small.
- **The gap is an upper bound.** Chaining passes the full training checkpoint, so the
  momentum teacher and both projection heads carry across a boundary, but the optimiser
  state and learning-rate schedule do not. Warmup and the cosine cycle therefore restart
  at every segment and the measured gap contains a schedule-restart component alongside
  the ordering effect. This is what a practitioner resuming pretraining without
  reconstructing the schedule obtains, which is why it is what gets reported.
- **Scale.** 560k images is below the point at which this objective is competitive with
  released weights, so absolute accuracies are modest and the gaps could behave
  differently at the full 4.7M-image release.
- **One architecture, one objective, one dataset.** The asymmetry is established for this
  setting rather than shown to be general.

## Authorship and self-containment

This project contains no code from any other repository by the authors: nothing imported,
nothing copied, nothing adapted. Every module is written from scratch against public APIs
(Lightning, torchmetrics, scikit-learn, LightlyTrain, Continuum, AgML) and every
dependency is a version-pinned public PyPI package with no `git` or `path` sources, so
the artefact is reproducible and redistributable from this repository alone. Enforced by
`tests/test_selfcontained.py`.

## Citation

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff),
which GitHub renders as a "Cite this repository" button. Its `preferred-citation`
entry is the paper; the venue fields there are provisional until acceptance.

## License

Apache License 2.0, see [LICENSE](LICENSE). iNatAg-mini and AgML carry their own licences,
which this repository does not redistribute or supersede.
