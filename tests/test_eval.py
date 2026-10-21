"""Feature extraction, linear probing and the mechanism metrics.

The alignment test is the important one here: `embed.py` writes feature rows in
`sorted(rel_path_256)` order and `probe.py` reads labels in
`sort_values("rel_path_256")` order. If those ever disagree, every label is
attached to the wrong feature vector and the results are silently meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from di_ssl_agri_gap.analysis import (
    hierarchical_error_metrics,
    linear_cka,
    parse_encoder_id,
)
from di_ssl_agri_gap.enums import Granularity, Split
from di_ssl_agri_gap.probe import evaluate, train_probe
from di_ssl_agri_gap.utils import load_settings

# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("split", list(Split))
def test_feature_and_label_orders_agree(split):
    cfg = load_settings()
    if not cfg.manifest_path.exists():
        pytest.skip("manifest not built yet")
    frame = pd.read_parquet(cfg.manifest_path, columns=["rel_path_256", "split"])
    subset = frame[frame["split"] == split.value]
    embed_order = sorted(subset["rel_path_256"].tolist())  # embed.py
    probe_order = subset.sort_values("rel_path_256")[
        "rel_path_256"
    ].tolist()  # probe.py
    assert embed_order == probe_order


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #


@pytest.fixture
def separable() -> tuple[torch.Tensor, torch.Tensor, int]:
    torch.manual_seed(0)
    n, d, k = 2000, 128, 10
    centres = torch.randn(k, d) * 3
    y = torch.randint(0, k, (n,))
    return centres[y] + torch.randn(n, d), y, k


def test_probe_learns(separable):
    x, y, k = separable
    model = train_probe(x, y, k, learning_rate=1e-2, epochs=15, batch_size=256, seed=42)
    metrics, _ = evaluate(model, x, y, k)
    assert metrics["accuracy_micro"] > 0.9


def test_probe_is_reproducible(separable):
    """Same seed must give bitwise-identical weights.

    Regression test: seeding only the shuffle generator left `LinearProbe`'s
    weight initialisation drawing from the unseeded global RNG.
    """
    x, y, k = separable
    a = train_probe(x, y, k, 1e-2, 5, 256, seed=42)
    b = train_probe(x, y, k, 1e-2, 5, 256, seed=42)
    c = train_probe(x, y, k, 1e-2, 5, 256, seed=7)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert not all(torch.equal(p, q) for p, q in zip(a.parameters(), c.parameters()))


def test_micro_and_macro_accuracy_diverge_under_imbalance():
    """The metric subtlety the whole study depends on getting right.

    torchmetrics defaults `MulticlassAccuracy` to macro. On a skewed, genuinely
    hard problem that is nothing like top-1, so both must be reported.
    """
    torch.manual_seed(0)
    d, k = 32, 10
    # Heavily skewed support and overlapping classes, so rare classes are lost.
    counts = [1800] + [20] * (k - 1)
    y = torch.cat([torch.full((c,), i) for i, c in enumerate(counts)])
    centres = torch.randn(k, d) * 0.4
    x = centres[y] + torch.randn(len(y), d)

    model = train_probe(x, y, k, 1e-2, 20, 256, seed=42)
    metrics, _ = evaluate(model, x, y, k)
    assert metrics["accuracy_micro"] - metrics["accuracy_macro"] > 0.2, metrics


# --------------------------------------------------------------------------- #
# Mechanism metrics
# --------------------------------------------------------------------------- #


@pytest.fixture
def toy_taxonomy() -> pd.DataFrame:
    # species 0,1 -> genus 0 -> family 0 ; species 2 -> genus 1 -> family 0
    # species 3   -> genus 2 -> family 1
    return pd.DataFrame(
        {
            "species_id": [0, 1, 2, 3],
            "genus_id": [0, 0, 1, 2],
            "family_id": [0, 0, 0, 1],
        }
    )


def test_hierarchical_distance_scores_each_tier(toy_taxonomy):
    preds = pd.DataFrame(
        {
            "y_true": [0, 0, 0, 0],
            # correct | same genus | same family, other genus | other family
            "y_pred": [0, 1, 2, 3],
        }
    )
    m = hierarchical_error_metrics(preds, toy_taxonomy)
    assert m["species_accuracy"] == 0.25
    assert m["mean_hierarchical_distance"] == pytest.approx((0 + 1 + 2 + 3) / 4)
    assert m["contained_in_genus"] == pytest.approx(1 / 3)
    assert m["contained_in_family"] == pytest.approx(2 / 3)
    assert m["escaped_family"] == pytest.approx(1 / 3)
    assert m["n_errors"] == 3


def test_perfect_predictions_have_zero_distance(toy_taxonomy):
    preds = pd.DataFrame({"y_true": [0, 1, 2, 3], "y_pred": [0, 1, 2, 3]})
    m = hierarchical_error_metrics(preds, toy_taxonomy)
    assert m["species_accuracy"] == 1.0
    assert m["mean_hierarchical_distance"] == 0.0
    assert m["n_errors"] == 0
    assert np.isnan(m["contained_in_genus"])


def test_cka_is_one_for_identical_and_invariant_to_rotation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 64)).astype(np.float32)
    assert linear_cka(x, x, device="cpu") == pytest.approx(1.0, abs=1e-4)

    # Linear CKA is invariant to orthogonal transforms -- drift must reflect a
    # genuine change in representational structure, not a change of basis.
    q, _ = np.linalg.qr(rng.normal(size=(64, 64)))
    assert linear_cka(x, x @ q.astype(np.float32), device="cpu") == pytest.approx(
        1.0, abs=1e-3
    )


def test_cka_drops_for_unrelated_features():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 64)).astype(np.float32)
    b = rng.normal(size=(500, 64)).astype(np.float32)
    assert linear_cka(a, b, device="cpu") < 0.5


# --------------------------------------------------------------------------- #
# Encoder-id round trip
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("encoder_id", "expected"),
    [
        # Only the canonical reference counts as the joint arm. Everything else
        # beginning with "joint" is a pre-training ablation (the abandoned
        # adaptation LR sweep, the augmentation tests). Matching on the prefix
        # pulled 8 encoders into the baseline and, because `pivot_table`
        # aggregates with mean, measured every gap against their average.
        (
            "joint_vits14scratch_ep100_seed42",
            {"arm": "joint", "n_chunks": 1, "chunk_idx": 0},
        ),
        ("joint_vits14_ep100_seed42", {"arm": "ablation"}),
        ("joint_vits14_ep100_seed42_ep10_lr0.004", {"arm": "ablation"}),
        ("offtheshelf_vits14", {"arm": "offtheshelf", "n_chunks": 0, "chunk_idx": -1}),
        (
            "iid_n4_chunk2_vits14_seed42",
            {"arm": "incremental", "ordering": "iid", "n_chunks": 4, "chunk_idx": 2},
        ),
        (
            "seasonal_n8_chunk7_vits14_seed42",
            {
                "arm": "incremental",
                "ordering": "seasonal",
                "n_chunks": 8,
                "chunk_idx": 7,
            },
        ),
    ],
)
def test_encoder_id_parses(encoder_id, expected):
    parsed = parse_encoder_id(encoder_id)
    for key, value in expected.items():
        assert parsed[key] == value


def test_granularity_label_columns_exist_in_manifest():
    cfg = load_settings()
    if not cfg.manifest_path.exists():
        pytest.skip("manifest not built yet")
    columns = pd.read_parquet(cfg.manifest_path).columns
    for granularity in Granularity:
        assert granularity.label_column in columns


# --------------------------------------------------------------------------- #
# Global-state hygiene
# --------------------------------------------------------------------------- #


def test_strict_determinism_does_not_leak_into_pretraining():
    """`seed_rng(strict=True)` must be undoable by `seed_rng(strict=False)`.

    Regression test. `probe_encoder` enables strict determinism globally and
    never restores it. In a driver that interleaves probing and pre-training in
    one process (the LR sweep; phase 2 if probes run between chunks), the next
    pre-training run inherited it and died at step 0, because DINOv2's multi-crop
    has no deterministic `upsample_bicubic2d_aa_backward`. `run_pretrain` now
    establishes its own state instead of trusting what came before.
    """
    from di_ssl_agri_gap.utils import seed_rng

    try:
        seed_rng(42, strict=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()

        seed_rng(42, strict=False)
        # warn_only keeps nondeterministic kernels usable instead of raising.
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(False)


def test_run_pretrain_resets_determinism_before_training():
    """The reset must live in `run_pretrain`, not in its callers."""
    import inspect

    from di_ssl_agri_gap.pretrain import run_pretrain

    source = inspect.getsource(run_pretrain)
    assert "seed_rng(spec.seed, strict=False)" in source


def test_gap_baseline_must_be_a_single_encoder():
    """A fanned-out join would silently average several encoders into the baseline.

    Regression test for the bug that measured every gap against the mean of 8
    "joint*" encoders, one of which scored 10.78% at species level.
    """
    import pandas as pd
    import pytest as _pytest

    from di_ssl_agri_gap.analysis import JOINT_REFERENCE, build_gap_table

    rows = []
    for enc in (JOINT_REFERENCE, "iid_n2_chunk0_vits14scratch_seed42"):
        for gran in ("species", "cropweed"):
            rows.append(
                {
                    "encoder_id": enc,
                    "granularity": gran,
                    "split": "test",
                    "accuracy_micro": 0.5,
                    "f1_macro": 0.5,
                }
            )
    frame = pd.DataFrame(rows)
    out = build_gap_table(frame)
    assert len(out) == len(frame)  # no fan-out
    assert (out["gap_top1"] == 0.0).all()  # identical inputs -> zero gap

    with _pytest.raises(ValueError, match="no probe results"):
        build_gap_table(frame, reference="does_not_exist")
