"""Fold assignment for the cross-validated filter sweep.

Every arm of the sweep must be scored on the same partition of the same patients,
or filter-vs-raw comparisons measure the shuffle as much as the filter. The
pre-2026-08-11 evaluation path could not do that: `DatasetEHR.split` draws
`torch.randperm(len(self))` per dataset object (ehr_dataset.py:70), and `len`
differs between arms, so each arm landed on an unrelated partition. It also
cached on `split_weights` alone, so the four "seeds" all reused the first draw —
the spread across them measured forest randomness, not sampling variability.

This module replaces that with one rule:

    the partition is drawn ONCE on the raw cohort, keyed by admission_id, and
    every arm inherits the fold label of each record it retains.

Nothing about an arm — its size, its position in the sweep, the ambient torch RNG
state — can move it. That is what makes arms paired by construction, which is
also what a correct McNemar needs.

Stratified on the target label because mortality prevalence is under 10% and an
unstratified fifth of an aggressive arm can contain no positives at all.

Run `python Managers/partition_manager.py` for the self-check.
"""
import hashlib
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

# Fixed and arbitrary. The point is that it is written down and never derived
# from the arm, the label, or iteration order — change it and every cached CV
# result becomes incomparable, so don't, casually.
PARTITION_SEED = 20260811

N_SPLITS = 5
N_REPEATS = 4

# Outer test fold is 1/N_SPLITS = 20%. The remaining 80% splits 7:1 into fit and
# inner-validation, giving 70/10/20 overall — the same nesting as the sibling
# hour-scaling experiment (hour_scaling_experiment/v2/splits.py:83).
INNER_VALIDATION_SHARE = 1.0 / 8.0


def fingerprint(admission_ids: np.ndarray, labels: np.ndarray) -> str:
    """Short digest of (cohort, labels), for spotting a stale cached partition.

    Adapted from hour_scaling_experiment/v2/splits.py:47. Hashing the ids as well
    as the labels — the sibling version hashes labels alone, which cannot tell
    two different cohorts with the same class balance apart.
    """
    hasher = hashlib.sha1()
    hasher.update(np.ascontiguousarray(np.asarray(admission_ids, dtype=np.int64)))
    hasher.update(np.ascontiguousarray(np.asarray(labels, dtype=np.int8)))
    return hasher.hexdigest()[:16]


def forest_seed(repeat: int, fold: int) -> int:
    """Estimator seed, deliberately distinct from the partition seed.

    Conflating the two is what made the old sweep's four seeds look like four
    partitions when they were one. Here the partition depends only on `repeat`
    and the forest depends on both, so they cannot be confused again.
    """
    return PARTITION_SEED + 1000 * repeat + fold


def build_fold_assignment(
    admission_ids: np.ndarray,
    labels: np.ndarray,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
) -> np.ndarray:
    """Assign every raw-cohort record a fold, once per repeat.

    Returns an (n_repeats, n_records) int8 array aligned to `admission_ids`
    order, where entry [r, i] is the outer fold record i belongs to in repeat r.

    Each repeat is a fresh stratified shuffle seeded `PARTITION_SEED + r`, so the
    repeats are genuinely independent partitions rather than the same one four
    times. That difference is asserted in `self_check`.
    """
    admission_ids = np.asarray(admission_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    if admission_ids.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{admission_ids.shape[0]:,} ids against {labels.shape[0]:,} labels"
        )
    if len(np.unique(admission_ids)) != len(admission_ids):
        raise ValueError("admission ids must be unique to key a partition on them")

    assignment = np.full((n_repeats, len(admission_ids)), -1, dtype=np.int8)

    for repeat in range(n_repeats):
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=PARTITION_SEED + repeat
        )
        for fold, (_, test_index) in enumerate(splitter.split(admission_ids, labels)):
            assignment[repeat, test_index] = fold

    if np.any(assignment < 0):
        raise RuntimeError("StratifiedKFold left records unassigned")

    return assignment


def project_to_arm(
    arm_admission_ids: np.ndarray,
    raw_admission_ids: np.ndarray,
    raw_assignment: np.ndarray,
) -> np.ndarray:
    """Inherit the raw cohort's fold labels for the records this arm retains.

    The arm never draws its own partition — that is the whole design. Returns an
    (n_repeats, len(arm_admission_ids)) array aligned to the arm's own order.

    Raises if the arm holds an id the raw cohort does not, which would mean the
    two are not views of the same underlying cohort.
    """
    arm_admission_ids = np.asarray(arm_admission_ids, dtype=np.int64)
    raw_admission_ids = np.asarray(raw_admission_ids, dtype=np.int64)

    order = np.argsort(raw_admission_ids)
    sorted_ids = raw_admission_ids[order]
    position = np.searchsorted(sorted_ids, arm_admission_ids)

    if np.any(position >= len(sorted_ids)) or not np.all(
        sorted_ids[np.clip(position, 0, len(sorted_ids) - 1)] == arm_admission_ids
    ):
        missing = int(np.sum(
            sorted_ids[np.clip(position, 0, len(sorted_ids) - 1)] != arm_admission_ids
        ))
        raise ValueError(
            f"{missing:,} of this arm's admission ids are absent from the raw cohort"
        )

    return raw_assignment[:, order[position]]


def inner_split(
    training_positions: np.ndarray, labels: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve a stratified inner-validation slice out of one outer fold's training set.

    The threshold has to be chosen somewhere disjoint from the fold being scored,
    otherwise the operating point leaks into the number it is supposed to be
    honest about. This is that somewhere.

    Falls back to using the training set itself when it is too small or
    single-class to split — an aggressive arm can leave a fold with almost
    nothing. The caller sees this via an empty validation array and
    `pick_threshold` degrades to 0.5 on its own.
    """
    training_positions = np.asarray(training_positions)
    fold_labels = labels[training_positions]

    positives = int(np.sum(fold_labels == 1))
    negatives = int(np.sum(fold_labels == 0))
    minimum = max(2, int(np.ceil(1.0 / INNER_VALIDATION_SHARE)))

    if positives < minimum or negatives < minimum:
        return training_positions, np.array([], dtype=training_positions.dtype)

    fit_positions, validation_positions = train_test_split(
        training_positions,
        test_size=INNER_VALIDATION_SHARE,
        random_state=seed,
        stratify=fold_labels,
    )
    return fit_positions, validation_positions


def fold_prevalences(assignment: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Worst-case prevalence drift of any fold against the population.

    Reported into the diagnostics sidecar so a reader can see whether an arm's
    inherited folds stayed balanced after it dropped records — stratification is
    exact on the raw cohort but only approximate on a subset of it.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return {"population_prevalence": float("nan"), "max_fold_drift": float("nan")}

    population = float(np.mean(labels))
    drift = 0.0
    for repeat in range(assignment.shape[0]):
        for fold in np.unique(assignment[repeat]):
            mask = assignment[repeat] == fold
            if mask.sum():
                drift = max(drift, abs(float(np.mean(labels[mask])) - population))

    return {"population_prevalence": population, "max_fold_drift": drift}


def self_check() -> bool:
    """Standalone verification, matching the repo idiom (v2/splits.py:105)."""
    rng = np.random.default_rng(0)
    n = 46032
    ids = np.arange(1000, 1000 + n, dtype=np.int64)
    labels = (rng.random(n) < 0.0997).astype(np.int64)

    ok = True

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  {'PASS' if condition else 'FAIL'}  {name}{' — ' + detail if detail else ''}")
        ok = ok and condition

    print("partition_manager self-check")
    assignment = build_fold_assignment(ids, labels)

    check("shape", assignment.shape == (N_REPEATS, n), str(assignment.shape))

    # Every record in exactly one fold per repeat, folds cover the cohort.
    complete = all(
        np.array_equal(np.sort(np.unique(assignment[r])), np.arange(N_SPLITS))
        and np.bincount(assignment[r], minlength=N_SPLITS).sum() == n
        for r in range(N_REPEATS)
    )
    check("folds partition the cohort exactly", complete)

    # Stratification held.
    drift = fold_prevalences(assignment, labels)["max_fold_drift"]
    check("max fold prevalence drift < 0.005", drift < 0.005, f"{drift:.5f}")

    # THE regression test for the seed bug: repeats must be different partitions.
    agreements = [
        float(np.mean(assignment[a] == assignment[b]))
        for a in range(N_REPEATS)
        for b in range(a + 1, N_REPEATS)
    ]
    worst = max(agreements)
    check(
        "repeats are independent partitions (pairwise agreement ~1/5, not 1.0)",
        worst < 0.35,
        f"max pairwise agreement {worst:.3f}",
    )

    # Determinism.
    check("rebuild is byte-identical", np.array_equal(assignment, build_fold_assignment(ids, labels)))

    # Projection onto a shrunken arm.
    keep = rng.choice(n, size=20323, replace=False)
    keep.sort()
    arm_ids = ids[keep]
    projected = project_to_arm(arm_ids, ids, assignment)
    check("projection preserves each record's fold", np.array_equal(projected, assignment[:, keep]))

    try:
        project_to_arm(np.array([-1], dtype=np.int64), ids, assignment)
        check("projection rejects unknown ids", False)
    except ValueError:
        check("projection rejects unknown ids", True)

    # Inner split is disjoint and stratified.
    training = np.where(assignment[0] != 0)[0]
    fit, validation = inner_split(training, labels, forest_seed(0, 0))
    check("inner split is disjoint", len(np.intersect1d(fit, validation)) == 0)
    check("inner split covers the training set", len(fit) + len(validation) == len(training))
    check(
        "inner validation is ~1/8 of training",
        abs(len(validation) / len(training) - INNER_VALIDATION_SHARE) < 0.01,
        f"{len(validation) / len(training):.4f}",
    )
    check("forest seeds are distinct", len({forest_seed(r, f) for r in range(N_REPEATS)
                                            for f in range(N_SPLITS)}) == N_REPEATS * N_SPLITS)
    check("fingerprint is stable", fingerprint(ids, labels) == fingerprint(ids, labels))
    check("fingerprint separates cohorts", fingerprint(ids, labels) != fingerprint(ids + 1, labels))

    print("OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if self_check() else 1)
