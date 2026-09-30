"""Class balancing for the training half of a cross-validation fold.

The cohort is 90.3% negative for mortality (9.66% prevalence, measured on the
`mean` aggregation's 46,032 records), and until now nothing corrected for that at
training time.

Three strategies, all resampling the *training* rows of one fold only:

    oversample   duplicate real minority rows until the classes match
    undersample  drop real majority rows until the classes match
    smote        interpolate new minority rows between real neighbours

`oversample` and `smote` produce training sets of identical size (2 x n_majority),
so comparing them isolates exactly one question — duplicate an existing patient,
or synthesise a new one.

Pure numpy and sklearn on purpose: no cupy import, so this module can be imported
and self-checked on a login node without `rerun/_cpu_backend.py`. The caller is
responsible for handing in host-side arrays and moving the result back to the
device.

## The zero problem

`RecordEHR.to_tensor()` does `np.nan_to_num(..., nan=0.0)`, so a 0 in the
168-vector means "never measured", not "measured as zero". 39% of the matrix is
those structural zeros (temperature alone is only 23% observed, and 10% of
patients are ~88% missing). Textbook SMOTE breaks on this twice:

1. Interpolating a real heart rate against a structural 0 fabricates a value that
   was never measured and is not physiologically meaningful — 80 bpm blended with
   "missing" gives 40 bpm.
2. Less obvious and worse: plain Euclidean kNN over zero-filled vectors mostly
   measures *missingness-pattern overlap* rather than physiological similarity, so
   the neighbour graph is wrong before any interpolation happens.

`smote_masked` below fixes both. See its docstring for the three rules.

## Provenance

Every strategy returns a `provenance` vector parallel to the balanced rows:
`>= 0` is the source record's position in the arm's row order, `-1` marks a
synthetic row. That is what lets the caller prove no test-fold record — and no
generated point — ever reaches evaluation. The structural guarantee is stronger
than the vector (a synthetic row has no position, so it cannot be written into
`oof_scores`), but the vector is what makes it auditable.

Run `python Managers/balancing_manager.py` for the self-check.
"""
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

# Running this file directly for the self-check puts Managers/ on sys.path rather
# than the repo root, so the absolute import below would fail. Imported as a
# module (the normal path) this is a no-op.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Managers.partition_manager import PARTITION_SEED

# Index into this tuple is the strategy id folded into the RNG seed, so the
# ordering is load-bearing: changing it changes every resample draw. 'none' is
# index 0 to match the positional convention the plotting helpers assume, where
# arm 0 is always the baseline.
STRATEGIES = ("none", "oversample", "undersample", "smote")

# A minority row with almost nothing measured is a useless SMOTE parent, and a
# pair with zero overlap makes nan_euclidean return NaN outright. Such rows stay
# in the training set; they just cannot be drawn as a parent.
MIN_OBSERVED_CELLS = 5

# Neighbours considered per minority point. Clamped down when the donor pool is
# smaller than this.
DEFAULT_K = 5


class BalancedFold(NamedTuple):
    """One fold's training set after balancing.

    `x` and `y` are what the forest is fitted on. `provenance` is parallel to
    them: entry >= 0 is the source record's position in the arm's row order,
    entry == -1 is synthetic. `parents` is (n_synthetic, 2), the two real
    positions each synthetic row was interpolated between, so a generated point
    can always be traced back to real patients.
    """
    x: np.ndarray
    y: np.ndarray
    provenance: np.ndarray
    parents: np.ndarray
    stats: Dict[str, object]


def balance_seed(repeat: int, fold: int, strategy: str) -> np.random.SeedSequence:
    """RNG seed for one (repeat, fold, strategy).

    Deliberately *not* `forest_seed(repeat, fold)` reused verbatim: the forest
    already consumes that for its bootstrap, and drawing the resample from the
    same stream would correlate which rows get duplicated with which rows each
    tree happens to bag. Folding the strategy index in as well gives each
    strategy an independent stream, so `oversample` and `smote` do not pick the
    same base rows and then get compared as if that were a coincidence.
    """
    return np.random.SeedSequence([PARTITION_SEED, repeat, fold, STRATEGIES.index(strategy)])


def smote_masked(
    minority: np.ndarray,
    n_new: int,
    rng: np.random.Generator,
    k: int = DEFAULT_K,
    min_observed: int = MIN_OBSERVED_CELLS,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """SMOTE that understands 0 as "not measured".

    Returns `(synthetic, parents_local)` where `parents_local` indexes into
    `minority`, or None when the donor pool is too small to interpolate at all
    (fewer than two rows clear `min_observed`). The caller falls back to plain
    oversampling in that case rather than failing the fold.

    Three rules, each fixing a specific way textbook SMOTE misreads this matrix:

    1. **Neighbours are found over co-observed cells only.** 0 is mapped to NaN
       and the search runs under `metric='nan_euclidean'`, which sums squared
       differences across the dimensions both rows observe and rescales by that
       count. Without this the neighbour graph is dominated by whether two
       patients happen to share a measurement schedule.

    2. **The synthetic patient inherits the base parent's measurement schedule.**
       `out` starts as a copy of A, so the generated row has exactly A's
       missingness mask — it never invents a measurement A lacked, and never
       drops one A had. This matters beyond plausibility: missingness here is
       class-correlated (positives are 68.3% observed against 60.1% for
       negatives, because sicker patients get measured more), so the mask is
       itself predictive. Inheriting it keeps that signal intact instead of
       smearing it across classes.

    3. **Interpolation happens only where both parents observed.** Where A
       measured and B did not, A's real value survives untouched; where A did not
       measure, the cell stays 0. So no cell is ever a blend of a reading and a
       structural zero.

    The neighbour search is brute-force (nan_euclidean has no tree support), but
    sklearn chunks `kneighbors` internally so the full pairwise matrix is never
    materialised.
    """
    if n_new <= 0:
        return np.empty((0, minority.shape[1]), dtype=minority.dtype), np.empty((0, 2), dtype=np.int64)

    observed = minority != 0
    donors = np.where(observed.sum(axis=1) >= min_observed)[0]

    if donors.size < 2:
        return None

    donor_rows = minority[donors]
    # NaN is the missing marker nan_euclidean expects; 0 would be read as a value.
    donor_nan = np.where(donor_rows == 0, np.nan, donor_rows)

    k_effective = int(min(k, donors.size - 1))
    finder = NearestNeighbors(n_neighbors=k_effective + 1, metric="nan_euclidean")
    finder.fit(donor_nan)
    # Column 0 of kneighbors is the point itself; drop it so a row is never its
    # own mate (which would make lam irrelevant and emit an exact duplicate).
    neighbours = finder.kneighbors(donor_nan, return_distance=False)[:, 1:]

    base_local = rng.integers(0, donors.size, n_new)
    mate_local = neighbours[base_local, rng.integers(0, k_effective, n_new)]

    a = donor_rows[base_local]
    b = donor_rows[mate_local]

    lam = rng.random((n_new, 1)).astype(minority.dtype, copy=False)
    both_observed = (a != 0) & (b != 0)

    synthetic = a.copy()
    np.copyto(synthetic, a + lam * (b - a), where=both_observed)

    if not np.isfinite(synthetic).all():
        raise RuntimeError("masked SMOTE produced a non-finite value")

    parents_local = np.stack([donors[base_local], donors[mate_local]], axis=1).astype(np.int64)
    return synthetic, parents_local


def balance_training_fold(
    strategy: str,
    features: np.ndarray,
    labels: np.ndarray,
    train_positions: np.ndarray,
    repeat: int,
    fold: int,
) -> BalancedFold:
    """Resample one fold's training rows to a 1:1 class ratio.

    `features` and `labels` are the arm's *whole* matrices; `train_positions`
    selects this fold's training side out of them. Nothing outside
    `train_positions` is ever read, which is the property that keeps the held-out
    fold clean — the test rows are not passed in at all, so they cannot be
    resampled by accident.

    Degenerate folds are returned unbalanced rather than raising: an empty or
    single-class training side has no minority to grow and no majority to shrink.
    `high invalid data` keeps zero records under the three deviation
    aggregations, so this path is exercised for real.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")

    train_positions = np.asarray(train_positions, dtype=np.int64)
    y_train = np.asarray(labels, dtype=np.int64)[train_positions]

    classes, counts = np.unique(y_train, return_counts=True)
    stats: Dict[str, object] = {
        "strategy": strategy,
        "n_train_in": int(train_positions.size),
        "class_counts_in": {int(c): int(n) for c, n in zip(classes, counts)},
        "smote_fallback": False,
    }

    def unchanged(reason: str) -> BalancedFold:
        stats.update({
            "n_real": int(train_positions.size), "n_synthetic": 0,
            "n_train_out": int(train_positions.size), "degenerate_reason": reason,
            "class_counts_out": {int(c): int(n) for c, n in zip(classes, counts)},
        })
        return BalancedFold(
            x=np.asarray(features)[train_positions], y=y_train,
            provenance=train_positions.copy(),
            parents=np.empty((0, 2), dtype=np.int64), stats=stats,
        )

    if strategy == "none":
        return unchanged("strategy is 'none'")
    if classes.size < 2:
        return unchanged("single-class training fold")

    minority_class = int(classes[int(np.argmin(counts))])
    majority_class = int(classes[int(np.argmax(counts))])
    minority_positions = train_positions[y_train == minority_class]
    majority_positions = train_positions[y_train == majority_class]
    deficit = majority_positions.size - minority_positions.size

    rng = np.random.default_rng(balance_seed(repeat, fold, strategy))
    features = np.asarray(features)
    parents = np.empty((0, 2), dtype=np.int64)

    if strategy == "undersample":
        # Without replacement: the point is to discard majority rows, and drawing
        # with replacement would quietly duplicate the survivors instead.
        kept_majority = rng.choice(majority_positions, size=minority_positions.size, replace=False)
        provenance = np.concatenate([minority_positions, kept_majority])
        x_balanced = features[provenance]
        y_balanced = np.asarray(labels, dtype=np.int64)[provenance]
        synthetic_count = 0

    elif strategy == "oversample":
        extra = rng.choice(minority_positions, size=deficit, replace=True)
        provenance = np.concatenate([train_positions, extra])
        x_balanced = features[provenance]
        y_balanced = np.asarray(labels, dtype=np.int64)[provenance]
        synthetic_count = 0

    else:  # smote
        generated = smote_masked(features[minority_positions], deficit, rng)

        if generated is None:
            # Too few usable parents. Falling back keeps the arm comparable in
            # size to `oversample` instead of silently leaving the fold skewed.
            stats["smote_fallback"] = True
            extra = rng.choice(minority_positions, size=deficit, replace=True)
            provenance = np.concatenate([train_positions, extra])
            x_balanced = features[provenance]
            y_balanced = np.asarray(labels, dtype=np.int64)[provenance]
            synthetic_count = 0
        else:
            synthetic, parents_local = generated
            x_balanced = np.concatenate([features[train_positions], synthetic])
            y_balanced = np.concatenate([
                y_train, np.full(synthetic.shape[0], minority_class, dtype=np.int64),
            ])
            provenance = np.concatenate([
                train_positions, np.full(synthetic.shape[0], -1, dtype=np.int64),
            ])
            # Report parents as arm-row positions, not indices into the minority
            # slice, so a synthetic row is traceable without the caller having to
            # reconstruct which rows were minority.
            parents = minority_positions[parents_local]
            synthetic_count = int(synthetic.shape[0])

    out_classes, out_counts = np.unique(y_balanced, return_counts=True)
    stats.update({
        "minority_class": minority_class, "majority_class": majority_class,
        "n_real": int((provenance >= 0).sum()), "n_synthetic": synthetic_count,
        "n_train_out": int(y_balanced.size),
        "class_counts_out": {int(c): int(n) for c, n in zip(out_classes, out_counts)},
    })

    return BalancedFold(
        x=x_balanced, y=y_balanced, provenance=provenance, parents=parents, stats=stats,
    )


def assert_no_leakage(
    balanced: BalancedFold, train_positions: np.ndarray, test_positions: np.ndarray
) -> None:
    """Prove this fold's training set touches no held-out record.

    Cheap enough (two set operations on a vector of a few tens of thousands) to
    run on every fold rather than behind a debug flag, and it is the assertion
    the whole provenance scheme exists to make possible. Raises rather than
    returning a bool: a leak makes every number downstream meaningless, so there
    is nothing sensible to do but stop.
    """
    real = balanced.provenance[balanced.provenance >= 0]

    intruders = np.intersect1d(real, np.asarray(test_positions, dtype=np.int64))
    if intruders.size:
        raise RuntimeError(
            f"{intruders.size:,} held-out records appear in the training set "
            f"(first few: {intruders[:5].tolist()})"
        )

    if not np.isin(real, np.asarray(train_positions, dtype=np.int64)).all():
        raise RuntimeError("training set contains a record from outside this fold's training side")

    if balanced.parents.size and not np.isin(balanced.parents.ravel(), train_positions).all():
        raise RuntimeError("a synthetic row was interpolated from a record outside the training side")

    if int((balanced.provenance == -1).sum()) != int(balanced.stats["n_synthetic"]):
        raise RuntimeError("provenance disagrees with the reported synthetic count")


def self_check() -> bool:
    """Standalone verification, matching the repo idiom (partition_manager.py:194)."""
    ok = True

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  {'PASS' if condition else 'FAIL'}  {name}{' — ' + detail if detail else ''}")
        ok = ok and condition

    print("balancing_manager self-check")

    rng = np.random.default_rng(0)
    n, d = 4000, 168
    features = rng.normal(50, 10, size=(n, d)).astype(np.float32)
    # Reproduce the real missingness rate (~39% of cells are structural zeros).
    features[rng.random((n, d)) < 0.39] = 0.0
    labels = (rng.random(n) < 0.0966).astype(np.int64)
    train_positions = np.sort(rng.choice(n, size=int(n * 0.8), replace=False))
    test_positions = np.setdiff1d(np.arange(n), train_positions)

    for strategy in ("oversample", "undersample", "smote"):
        balanced = balance_training_fold(strategy, features, labels, train_positions, 0, 0)
        counts = balanced.stats["class_counts_out"]
        check(f"{strategy}: classes are balanced", len(set(counts.values())) == 1, str(counts))
        check(
            f"{strategy}: provenance length matches rows",
            balanced.provenance.size == balanced.y.size == balanced.x.shape[0],
        )
        check(
            f"{strategy}: real rows all come from the training side",
            np.isin(balanced.provenance[balanced.provenance >= 0], train_positions).all(),
        )
        try:
            assert_no_leakage(balanced, train_positions, test_positions)
            check(f"{strategy}: leakage assertion passes", True)
        except RuntimeError as error:
            check(f"{strategy}: leakage assertion passes", False, str(error))

        repeated = balance_training_fold(strategy, features, labels, train_positions, 0, 0)
        check(
            f"{strategy}: same seed reproduces byte-identical output",
            np.array_equal(balanced.x, repeated.x) and np.array_equal(balanced.provenance, repeated.provenance),
        )
        different = balance_training_fold(strategy, features, labels, train_positions, 1, 0)
        # Compare the rows as well as the provenance: SMOTE's provenance is
        # `train_positions` followed by a run of -1 regardless of the draw, so
        # provenance alone cannot detect a stuck RNG for that strategy.
        check(
            f"{strategy}: a different fold draws differently",
            not (np.array_equal(balanced.provenance, different.provenance)
                 and np.array_equal(balanced.x, different.x)),
        )

    # Only SMOTE may invent rows; the other two must be pure record selection.
    for strategy in ("oversample", "undersample"):
        balanced = balance_training_fold(strategy, features, labels, train_positions, 0, 0)
        check(f"{strategy}: creates nothing synthetic", int(balanced.stats["n_synthetic"]) == 0)
        check(
            f"{strategy}: every row is a verbatim copy of its source record",
            np.array_equal(balanced.x, features[balanced.provenance]),
        )

    smote = balance_training_fold("smote", features, labels, train_positions, 0, 0)
    synthetic_rows = smote.provenance == -1
    generated = smote.x[synthetic_rows]
    base, mate = smote.parents[:, 0], smote.parents[:, 1]
    base_rows, mate_rows = features[base], features[mate]

    check("smote: generated the exact deficit", int(smote.stats["n_synthetic"]) == generated.shape[0])
    check("smote: real rows are left untouched",
          np.array_equal(smote.x[~synthetic_rows], features[train_positions]))
    check("smote: no non-finite values", bool(np.isfinite(generated).all()))
    check("smote: synthetic mask equals the base parent's mask",
          np.array_equal(generated != 0, base_rows != 0))
    check("smote: never fills a cell the base parent had missing",
          bool((generated[base_rows == 0] == 0).all()))

    # Where both parents measured, the value must be a convex combination.
    both = (base_rows != 0) & (mate_rows != 0)
    low = np.minimum(base_rows, mate_rows)
    high = np.maximum(base_rows, mate_rows)
    within = (generated[both] >= low[both] - 1e-3) & (generated[both] <= high[both] + 1e-3)
    check("smote: co-observed cells lie between the two parents", bool(within.all()),
          f"{int((~within).sum())} violations of {int(both.sum()):,}")

    # Where only the base measured, its real value must survive unblended.
    base_only = (base_rows != 0) & (mate_rows == 0)
    check("smote: base-only cells keep the parent's real value",
          bool(np.allclose(generated[base_only], base_rows[base_only])),
          f"{int(base_only.sum()):,} cells")

    check("smote: parents are always real training records",
          bool(np.isin(smote.parents.ravel(), train_positions).all()))
    check("smote: a row is never its own mate", bool((base != mate).all()))

    # Degenerate inputs must no-op instead of raising — `high invalid data` keeps
    # zero records under the three deviation aggregations.
    empty = balance_training_fold("smote", features, labels, np.array([], dtype=np.int64), 0, 0)
    check("degenerate: empty fold returns empty", empty.y.size == 0)

    single = np.where(labels == 0)[0][:500]
    single_class = balance_training_fold("smote", features, labels, single, 0, 0)
    check("degenerate: single-class fold is returned unbalanced", single_class.y.size == single.size)

    tiny_labels = labels.copy()
    tiny_labels[:] = 0
    tiny_labels[train_positions[0]] = 1
    lone = balance_training_fold("smote", features, tiny_labels, train_positions, 0, 0)
    check("degenerate: one-record minority falls back to oversampling",
          bool(lone.stats["smote_fallback"]) and int(lone.stats["n_synthetic"]) == 0)

    print("OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if self_check() else 1)
