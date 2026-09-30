"""Training and scoring for the filter-impact experiments.

Everything here runs on GPU via cuML — a filter sweep is 12 filters × 5
aggregations × 2 labels × several seeds, which is a lot of random forests. cuML's
API deliberately mirrors sklearn's, so this reads like sklearn code, but the
arrays are cupy and have to be moved back with cp.asnumpy() before anything
sklearn- or scipy-based touches them. That's the one recurring gotcha in this file.
"""

import math

import cupy as cp
import numpy as np
import torch
from typing import Any, Dict, List, Tuple
from cuml.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score
from cuml.model_selection import KFold
from scipy.stats import mode
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score, recall_score,
    roc_auc_score, roc_curve,
)
from statsmodels.stats.contingency_tables import mcnemar
from tqdm.notebook import tqdm
from tabulate import tabulate
from Entities.ehr_dataset import DatasetEHR
from Managers.dataset_manager import flatten_subset_to_cupy
from Managers.balancing_manager import (
    STRATEGIES, assert_no_leakage, balance_training_fold,
)
from Managers.partition_manager import (
    INNER_VALIDATION_SHARE, N_SPLITS, build_fold_assignment, fingerprint,
    fold_prevalences, forest_seed, inner_split, project_to_arm,
)



def cross_validate_model(timeseries: cp.ndarray, labels: cp.ndarray, seed: int, splits: int = 4) -> float:
    """K-fold accuracy on the training split, averaged over folds.

    Empty folds score 0.0 rather than NaN or an exception. That's a compromise:
    aggressive filters can shrink a dataset enough that a fold comes out empty,
    and a 0 drags the mean down in a visible way instead of poisoning it with
    NaN. Worth remembering when reading a suspiciously low CV score — it may mean
    "folds were empty", not "the model is bad".
    """
    k_fold = KFold(n_splits=splits, shuffle=True, random_state=seed)
    fold_scores = []

    for training_index, validation_index in k_fold.split(timeseries):
        x_train = timeseries[training_index]
        x_validation = timeseries[validation_index]
        y_train = labels[training_index]
        y_validation = labels[validation_index]

        if x_train.size == 0 or x_validation.size == 0:
            fold_scores.append(0.0)
            continue

        model = RandomForestClassifier(n_estimators=300, random_state=seed)
        model.fit(x_train, y_train)
        predictions = model.predict(x_validation)
        fold_scores.append(accuracy_score(y_validation, predictions))

    return float(cp.mean(cp.array(fold_scores)))


def pick_threshold(y_validation: np.ndarray, scores: np.ndarray) -> float:
    """Choose a decision threshold by maximising Youden's J on validation.

    Youden's J (equivalently balanced accuracy) weights sensitivity and
    specificity equally, so unlike the implicit 0.5 it doesn't hand the decision
    to the majority class — which matters here, where mortality prevalence is
    under 15% and a 0.5 threshold buys accuracy by predicting almost nothing
    positive.

    Selected on validation only. Never on test, never on training, so the
    reported operating point is honest.

    Falls back to 0.5 when validation is empty or single-class. That isn't
    hypothetical: the split is unstratified, and an aggressive arm like
    'high invalid data' keeps a small enough fraction of records that its 10%
    validation subset can contain no positives at all. 0.5 reproduces the old
    behaviour rather than raising.
    """
    if y_validation.size == 0:
        return 0.5
    positives = int(y_validation.sum())
    if positives == 0 or positives == y_validation.size:
        return 0.5

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_validation, scores)
    youden = true_positive_rate - false_positive_rate
    threshold = thresholds[int(np.argmax(youden))]

    # roc_curve's first threshold is +inf by construction; guard against picking it.
    if not np.isfinite(threshold):
        finite = thresholds[np.isfinite(thresholds)]
        threshold = float(finite.max()) if finite.size else 0.5

    return float(threshold)


def _positive_scores(model: Any, features: cp.ndarray) -> np.ndarray:
    """P(class = 1) as a host-side numpy vector.

    predict_proba's columns follow model.classes_, so the positive class isn't
    unconditionally column 1 — an arm whose training split happens to be
    single-class has only one column. Resolved by looking classes_ up rather
    than assuming.

    cuML returns device arrays and roc_curve/f1_score are host-side, so
    everything comes back through cp.asnumpy here rather than at each call site.
    """
    if features.size == 0:
        return np.array([])

    probabilities = np.asarray(cp.asnumpy(model.predict_proba(features)))
    classes = np.asarray(cp.asnumpy(model.classes_)).astype(int).tolist()

    if 1 not in classes:
        # Training split saw only one class; it was the negative one.
        return np.zeros(probabilities.shape[0])

    return probabilities[:, classes.index(1)]


def evaluate_dataset_label_impact(dataset: Any, label_index: int, seed: int) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Dict[str, float]]:
    """Train one forest on one dataset and report CV score, test score, and
    predictions on both splits.

    Splits 80/10/10 and keeps all three parts separate. Validation is never
    scored as test data and never trained on — its one job is to choose the
    decision threshold (see pick_threshold), which is then applied unchanged to
    the training and test splits. Test is therefore a true held-out 10%.

    This used to concatenate validation onto test for an effective 80/20, on the
    reasoning that nothing looked at validation on its own so holding it out
    only cost evaluation sample. That reasoning no longer applies now that the
    threshold is selected on it, and the merge made the operating point
    unchoosable. The cost is real though: test is half the size it was, so test
    metrics and the McNemar comparisons are noisier than in pre-2026-08-10
    results.

    Returns (cv_score, test_score, test_predictions, test_labels,
    train_predictions, train_labels, threshold, diagnostics). The training-split
    predictions exist so the caller can compute a training F1; expect them to
    look overfit — a random forest scores near-perfectly on its own training
    data.

    Predictions come back as numpy because that's what McNemar and sklearn's F1
    need downstream.
    """
    train_sub, val_sub, test_sub = dataset.split(0.8, 0.1, 0.1)

    x_train, y_train = flatten_subset_to_cupy(train_sub, label_index)
    x_test, y_test = flatten_subset_to_cupy(test_sub, label_index)
    x_validation, y_validation = flatten_subset_to_cupy(val_sub, label_index)

    y_train_host = cp.asnumpy(y_train).astype(int) if y_train.size else np.array([], dtype=int)
    y_test_host = cp.asnumpy(y_test).astype(int) if y_test.size else np.array([], dtype=int)
    y_validation_host = cp.asnumpy(y_validation).astype(int) if y_validation.size else np.array([], dtype=int)

    diagnostics = {
        'n_train': int(y_train_host.size),
        'n_validation': int(y_validation_host.size),
        'n_test': int(y_test_host.size),
        'validation_prevalence': float(y_validation_host.mean()) if y_validation_host.size else float('nan'),
        'test_prevalence': float(y_test_host.mean()) if y_test_host.size else float('nan'),
    }

    if x_train.size == 0 or x_test.size == 0:
        diagnostics.update({'threshold': 0.5, 'validation_accuracy': float('nan'),
                            'validation_f1_macro': float('nan'), 'test_positive_rate': float('nan')})
        return (0.0, 0.0, np.zeros(len(y_test_host)), y_test_host,
                np.zeros(len(y_train_host)), y_train_host, 0.5, diagnostics)

    cross_validation_score = cross_validate_model(x_train, y_train, seed)

    final_model = RandomForestClassifier(n_estimators=300, random_state=seed)
    final_model.fit(x_train, y_train)

    validation_scores = _positive_scores(final_model, x_validation)
    threshold = pick_threshold(y_validation_host, validation_scores)

    # Both splits are thresholded at the same validation-selected operating
    # point, rather than going through .predict()'s implicit 0.5, so the
    # training F1 and the test F1 describe the same classifier
    test_predictions = (_positive_scores(final_model, x_test) >= threshold).astype(int)
    train_predictions = (_positive_scores(final_model, x_train) >= threshold).astype(int)
    test_score = float(np.mean(test_predictions == y_test_host))

    validation_predictions = (validation_scores >= threshold).astype(int)
    diagnostics.update({
        'threshold': float(threshold),
        'validation_accuracy': (float(np.mean(validation_predictions == y_validation_host))
                                if y_validation_host.size else float('nan')),
        'validation_f1_macro': (float(f1_score(y_validation_host, validation_predictions, average='macro'))
                                if y_validation_host.size else float('nan')),
        'test_positive_rate': float(test_predictions.mean()),
    })

    return (cross_validation_score, test_score, test_predictions, y_test_host,
            train_predictions, y_train_host, float(threshold), diagnostics)


def calculate_mcnemar_test(baseline_predictions: List[np.ndarray], filtered_predictions: List[np.ndarray]) -> Tuple[float, float]:
    """Is the filtered model's disagreement with the baseline more than noise?

    Takes a majority vote across seeds first, then compares the two consensus
    prediction vectors. Voting rather than pooling because a single seed's forest
    is noisy enough that McNemar would mostly be measuring seed variance; the
    majority vote asks the more useful question of whether the *filter* changes
    what the model consistently predicts.

    exact=True runs the binomial test rather than the chi-square approximation.
    Slower, but the discordant counts here are often small enough that the
    approximation misbehaves.

    Note this compares predictions against each other, not against ground truth —
    it detects that the two models disagree, not which one is right. Read it
    alongside the accuracy numbers, never on its own.

    One assumption that will bite you if it's ever violated: both prediction
    vectors have to be the same length and in the same patient order, which holds
    only because every dataset in a sweep is split with the same seed.
    """
    baseline_predictions = np.stack(baseline_predictions, axis=0)
    filter_predictions = np.stack(filtered_predictions, axis=0)

    base_majority = mode(baseline_predictions, axis=0, keepdims=True).mode[0]
    filter_majority = mode(filter_predictions, axis=0, keepdims=True).mode[0]

    table = np.zeros((2, 2), dtype=int)
    for baseline_prediction, filter_prediction in zip(base_majority.astype(int), filter_majority.astype(int)):
        table[baseline_prediction, filter_prediction] += 1

    result = mcnemar(table, exact=True)
    return float(result.statistic), float(result.pvalue)


def evaluate_filter_impact(raw_dataset: DatasetEHR, filtered_datasets: Dict[str, DatasetEHR], target_label: str,
                           seeds: List[int], return_diagnostics: bool = False) -> tuple:
    """Score every filtered dataset against the unfiltered baseline.

    Returns five parallel lists, all indexed the same way with 'raw' at position
    0 — the plotting code relies on that and slices [1:] to drop the
    baseline-vs-itself comparison.

    The five-list shape is load-bearing and must not grow: the cross-pipeline
    comparison notebook unpacks the pickled result with a strict
    `_acc, _f1, _auc, _, _mnm = pickle.load(...)`, which raises on six. Hence
    `return_diagnostics`, off by default — set it and you get a sixth element,
    a per-arm dict of thresholds and split sizes, without changing what the
    cached pickles look like.

    Careful with the return values: `training_averages` is the cross-validation
    score, not training accuracy. `training_f1s` is scored on the training split
    itself, so expect it near 1.0 — a random forest memorises its training data;
    it's there to show overfit headroom, not model quality.

    Since 2026-08-10 the test metrics are on a true held-out 10% and use a
    threshold chosen on the validation split, not the implicit 0.5. Absolute
    scores are therefore not comparable to results cached before that date;
    within-run filter-vs-raw deltas are.
    """
    label_index = raw_dataset.label_index_map[target_label]

    ordered_names = ['raw'] + list(filtered_datasets.keys())
    all_datasets = {'raw': raw_dataset, **filtered_datasets}

    # Seeding torch per seed *before* the inner loop is what makes the comparison
    # fair: every dataset gets the identical train/test split for a given seed,
    # so differences come from the filter rather than from the shuffle.
    #
    # Worth knowing, now that the threshold is selected on validation: DatasetEHR.split
    # caches on split_weights, so the second and later seeds reuse the partition the
    # first one drew. Every seed therefore picks its threshold on the *same* validation
    # records, and only the forest's own randomness varies across seeds.
    raw_results = {name: [] for name in ordered_names}
    for seed in seeds:
        torch.manual_seed(seed)
        for name in ordered_names:
            dataset = all_datasets[name]
            raw_results[name].append(evaluate_dataset_label_impact(dataset, label_index, seed))

    training_averages = []
    testing_averages = []
    mcnemar_results = []
    training_f1s = []
    testing_f1s = []

    baseline_preds = [t[2] for t in raw_results['raw']]

    for name in ordered_names:
        trials = raw_results[name]
        training_averages.append(float(np.mean([t[0] for t in trials])))
        testing_averages.append(float(np.mean([t[1] for t in trials])))
        # macro averaging so the minority class counts as much as the majority —
        # mortality prevalence is low enough that weighted F1 would mostly be
        # reporting how well we predict survival.
        train_f1_list = [f1_score(t[5], t[4], average='macro') for t in trials if len(t[4]) > 0 and len(t[5]) > 0]
        test_f1_list = [f1_score(t[3], t[2], average='macro') for t in trials if len(t[2]) > 0 and len(t[3]) > 0]
        training_f1s.append(float(np.mean(train_f1_list)) if train_f1_list else 0.0)
        testing_f1s.append(float(np.mean(test_f1_list)) if test_f1_list else 0.0)

        # 'raw' gets compared against itself here, giving p=1 at index 0.
        # Harmless, and it keeps all five lists index-aligned.
        current_preds = [t[2] for t in trials]
        stat, p_val = calculate_mcnemar_test(baseline_preds, current_preds)
        mcnemar_results.append((stat, p_val))

    results = (training_averages, testing_averages, training_f1s, testing_f1s, mcnemar_results)

    if not return_diagnostics:
        return results

    diagnostics = {}
    for name in ordered_names:
        trials = raw_results[name]
        # Split sizes are identical across seeds (the split is cached), so the
        # first trial's are representative; thresholds are not, so keep them all.
        per_seed = [t[7] for t in trials]
        diagnostics[name] = {
            'thresholds': {seed: t[6] for seed, t in zip(seeds, trials)},
            'n_train': per_seed[0]['n_train'],
            'n_validation': per_seed[0]['n_validation'],
            'n_test': per_seed[0]['n_test'],
            'validation_prevalence': per_seed[0]['validation_prevalence'],
            'test_prevalence': per_seed[0]['test_prevalence'],
            'validation_accuracy': float(np.mean([d['validation_accuracy'] for d in per_seed])),
            'validation_f1_macro': float(np.mean([d['validation_f1_macro'] for d in per_seed])),
            'test_positive_rate': float(np.mean([d['test_positive_rate'] for d in per_seed])),
        }

    return results + (diagnostics,)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validated evaluation (2026-08-11)
#
# Everything above scores a single fixed 10% holdout. Everything below scores
# every record out-of-fold under four independent stratified partitions, and
# pairs arms on admission_id so McNemar is actually a paired test. The two paths
# coexist on purpose: the legacy one keeps the pre-2026-08-11 pickles and figures
# reproducible, and its results are NOT comparable to these.
# ─────────────────────────────────────────────────────────────────────────────


def _mean_of_finite(values: List[float]) -> float:
    """Mean over the finite entries, NaN if there are none.

    A degenerate arm produces all-NaN metric lists, and np.nanmean warns once per
    call on those — enough noise in a 13-arm sweep to bury the real output.
    """
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float('nan')


def _std_of_finite(values: List[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.std(finite)) if finite else float('nan')


def _flatten_whole_dataset(dataset: Any, label_index: int) -> Tuple[Any, np.ndarray]:
    """Flatten an entire dataset once, rather than per fold.

    The legacy path re-flattens for every split of every seed. With 20 fits per
    arm that is 20 redundant passes over ~46k records, and flattening is not
    free — it walks the record list building a 168-vector each. Doing it once and
    indexing rows is the single biggest saving in the CV path.
    """
    subset = torch.utils.data.Subset(dataset, list(range(len(dataset.data))))
    features, labels = flatten_subset_to_cupy(subset, label_index)
    if features.size == 0:
        return features, np.array([], dtype=int)
    return features, cp.asnumpy(labels).astype(int)


def evaluate_dataset_label_cv(
    dataset: Any,
    admission_ids: np.ndarray,
    label_index: int,
    fold_assignment: np.ndarray,
) -> Dict[str, Any]:
    """Repeated stratified k-fold with out-of-fold predictions for one arm.

    `fold_assignment` is (n_repeats, n_records), inherited from the raw cohort by
    partition_manager.project_to_arm — this function never draws a partition of
    its own, which is what keeps every arm scored on the same patients.

    Per outer fold: hold the fold out, split the remaining 80% stratified into
    fit (7/8) and inner-validation (1/8), fit one forest, choose the threshold on
    inner-validation with the existing `pick_threshold` (Youden's J, unchanged),
    and apply it to the held-out fold. The threshold is never chosen on data it
    is then scored against.

    Combining repeats averages the out-of-fold *scores* and thresholds them once,
    rather than majority-voting labels. Voting was what the legacy path did, and
    with an even number of trials `scipy.stats.mode` breaks ties toward the
    smaller label — a silent bias toward the negative class on a 10%-prevalence
    problem. Averaging has no tie to break.

    Records whose fold could not be fitted (an aggressive arm can empty one) keep
    a NaN score and are excluded from every metric; `coverage` reports how many.
    """
    n_records = len(dataset.data)
    admission_ids = np.asarray(admission_ids, dtype=np.int64)

    if n_records != len(admission_ids):
        raise ValueError(
            f"{n_records:,} records against {len(admission_ids):,} admission ids"
        )

    n_repeats = fold_assignment.shape[0]
    features, labels = _flatten_whole_dataset(dataset, label_index)

    oof_scores = np.full((n_repeats, n_records), np.nan, dtype=np.float64)
    oof_predictions = np.full((n_repeats, n_records), -1, dtype=np.int8)
    thresholds: Dict[str, float] = {}
    train_accuracies: List[float] = []
    train_f1s: List[float] = []

    # No early return for the empty-arm case. `high invalid data` genuinely keeps
    # zero records under the three deviation aggregations, and a special-cased
    # return here shipped a diagnostics dict with a different key set than the
    # normal path — which the caller then indexed into and died on. The loops and
    # reducers below all handle n_records == 0 on their own, so letting the empty
    # arm walk the same path is both shorter and impossible to drift.

    for repeat in range(n_repeats):
        for fold in range(N_SPLITS):
            test_positions = np.where(fold_assignment[repeat] == fold)[0]
            train_positions = np.where(fold_assignment[repeat] != fold)[0]

            if test_positions.size == 0 or train_positions.size == 0:
                continue

            seed = forest_seed(repeat, fold)
            fit_positions, validation_positions = inner_split(train_positions, labels, seed)

            y_fit = labels[fit_positions]
            if fit_positions.size == 0 or len(np.unique(y_fit)) < 2:
                # Single-class fit split: a forest trained on it predicts one
                # class for everything and the threshold is meaningless. Leave
                # the fold uncovered rather than emitting a fake prediction.
                continue

            model = RandomForestClassifier(n_estimators=300, random_state=seed)
            model.fit(features[cp.asarray(fit_positions)], cp.asarray(y_fit))

            if validation_positions.size:
                validation_scores = _positive_scores(
                    model, features[cp.asarray(validation_positions)]
                )
                threshold = pick_threshold(labels[validation_positions], validation_scores)
            else:
                threshold = 0.5

            test_scores = _positive_scores(model, features[cp.asarray(test_positions)])
            oof_scores[repeat, test_positions] = test_scores
            oof_predictions[repeat, test_positions] = (test_scores >= threshold).astype(np.int8)
            thresholds[f"r{repeat}f{fold}"] = float(threshold)

            fit_scores = _positive_scores(model, features[cp.asarray(fit_positions)])
            fit_predictions = (fit_scores >= threshold).astype(int)
            train_accuracies.append(float(np.mean(fit_predictions == y_fit)))
            train_f1s.append(float(f1_score(y_fit, fit_predictions, average='macro')))

    # Per-repeat metrics over that repeat's full out-of-fold vector. The spread
    # across these is the honest error bar: four independent partitions, not four
    # forests on one partition.
    per_repeat = []
    for repeat in range(n_repeats):
        covered = oof_predictions[repeat] >= 0
        if not covered.any():
            per_repeat.append({'accuracy': float('nan'), 'f1_macro': float('nan'), 'n': 0})
            continue
        y_true = labels[covered]
        y_pred = oof_predictions[repeat][covered]
        per_repeat.append({
            'accuracy': float(np.mean(y_pred == y_true)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
            'n': int(covered.sum()),
        })

    # Explicit mask rather than a bare nanmean: an arm can leave records with no
    # score at all, and np.nanmean on an all-NaN row warns per row — thousands of
    # lines in a job log for a degenerate arm.
    covered_mask = ~np.isnan(oof_scores)
    covered_count = covered_mask.sum(axis=0)
    consensus_scores = np.full(n_records, np.nan, dtype=np.float64)
    has_any = covered_count > 0
    consensus_scores[has_any] = (
        np.nansum(oof_scores[:, has_any], axis=0) / covered_count[has_any]
    )
    mean_threshold = float(np.mean(list(thresholds.values()))) if thresholds else 0.5
    consensus_covered = ~np.isnan(consensus_scores)
    consensus_predictions = np.full(n_records, -1, dtype=np.int8)
    consensus_predictions[consensus_covered] = (
        consensus_scores[consensus_covered] >= mean_threshold
    ).astype(np.int8)

    diagnostics = {
        'n_records': int(n_records),
        'coverage': float(np.mean(consensus_covered)) if n_records else 0.0,
        'n_folds_fitted': len(thresholds),
        'mean_threshold': mean_threshold,
        'accuracy_mean': _mean_of_finite([r['accuracy'] for r in per_repeat]),
        'accuracy_std': _std_of_finite([r['accuracy'] for r in per_repeat]),
        'f1_macro_mean': _mean_of_finite([r['f1_macro'] for r in per_repeat]),
        'f1_macro_std': _std_of_finite([r['f1_macro'] for r in per_repeat]),
        'predicted_positive_rate': (
            float(np.mean(consensus_predictions[consensus_covered]))
            if consensus_covered.any() else float('nan')
        ),
        'degenerate': not bool(thresholds),
    }
    diagnostics.update(fold_prevalences(fold_assignment, labels))

    return {
        'admission_ids': admission_ids,
        'y': labels,
        'oof_scores': oof_scores,
        'oof_predictions': oof_predictions,
        'consensus_scores': consensus_scores,
        'consensus_predictions': consensus_predictions,
        'per_repeat': per_repeat,
        'thresholds': thresholds,
        'train_accuracy': float(np.mean(train_accuracies)) if train_accuracies else float('nan'),
        'train_f1_macro': float(np.mean(train_f1s)) if train_f1s else float('nan'),
        'diagnostics': diagnostics,
    }


def calculate_mcnemar_paired(
    baseline_predictions: np.ndarray,
    baseline_ids: np.ndarray,
    arm_predictions: np.ndarray,
    arm_ids: np.ndarray,
    labels_by_id: Dict[int, int],
) -> Tuple[float, float, Dict[str, Any]]:
    """McNemar over the records both arms actually hold, scored against truth.

    Two things this fixes in `calculate_mcnemar_test`:

    1. **It pairs.** The old version zipped two prediction vectors positionally.
       For a record-dropping arm those vectors are different lengths *and*
       different patients, and `zip` truncated to the shorter without a word —
       `high invalid data` contributed 138 unrelated pairs against raw's 4,604.
       Here both vectors are aligned to the sorted intersection of their
       admission ids, and a length disagreement raises.
    2. **Ground truth enters.** The old table was
       `table[baseline_prediction, arm_prediction]`, which measures whether the
       two models *disagree*, not whether either is *right* — it cannot tell a
       filter that helped from one that hurt. This tabulates correctness, which
       is what "did the filter improve the classifier" needs. Same formulation as
       `mcnemar_counts` in comparison/pipeline_comparison.ipynb and
       hour_scaling_experiment/v2/metrics.py:189.

    Because the CV path gives the baseline out-of-fold coverage of the whole
    cohort, its prediction already exists for every record any arm retains — no
    replaying the baseline model on the arm's records is needed.

    Returns (statistic, pvalue, detail). A pair count of zero yields
    (nan, nan) and a flag rather than the old silent all-zero table.
    """
    baseline_ids = np.asarray(baseline_ids, dtype=np.int64)
    arm_ids = np.asarray(arm_ids, dtype=np.int64)

    # Only records both arms actually predicted (coverage gaps are marked -1).
    baseline_ok = np.asarray(baseline_predictions) >= 0
    arm_ok = np.asarray(arm_predictions) >= 0

    shared = np.intersect1d(baseline_ids[baseline_ok], arm_ids[arm_ok])
    detail = {'n_paired': int(shared.size), 'n01': 0, 'n10': 0, 'degenerate': False}

    if shared.size == 0:
        detail['degenerate'] = True
        return float('nan'), float('nan'), detail

    baseline_lookup = {int(i): p for i, p in zip(baseline_ids, baseline_predictions)}
    arm_lookup = {int(i): p for i, p in zip(arm_ids, arm_predictions)}

    y_true = np.array([labels_by_id[int(i)] for i in shared], dtype=int)
    baseline_aligned = np.array([baseline_lookup[int(i)] for i in shared], dtype=int)
    arm_aligned = np.array([arm_lookup[int(i)] for i in shared], dtype=int)

    if not (len(y_true) == len(baseline_aligned) == len(arm_aligned)):
        raise ValueError(
            f"unpaired vectors after alignment: {len(y_true)}/"
            f"{len(baseline_aligned)}/{len(arm_aligned)}"
        )

    baseline_correct = baseline_aligned == y_true
    arm_correct = arm_aligned == y_true

    n01 = int(np.sum(~baseline_correct & arm_correct))   # arm fixed it
    n10 = int(np.sum(baseline_correct & ~arm_correct))   # arm broke it
    detail.update({
        'n01': n01, 'n10': n10,
        'n_both_correct': int(np.sum(baseline_correct & arm_correct)),
        'n_both_wrong': int(np.sum(~baseline_correct & ~arm_correct)),
        'baseline_accuracy_on_paired': float(np.mean(baseline_correct)),
        'arm_accuracy_on_paired': float(np.mean(arm_correct)),
    })

    table = np.array([
        [detail['n_both_correct'], n10],
        [n01, detail['n_both_wrong']],
    ], dtype=int)

    if n01 + n10 == 0:
        # Identical predictions everywhere — the exact test is degenerate but the
        # answer is unambiguous: no evidence of a difference.
        return 0.0, 1.0, detail

    result = mcnemar(table, exact=True)
    return float(result.statistic), float(result.pvalue), detail


def evaluate_filter_impact_cv(
    raw_dataset: Any,
    filtered_datasets: Dict[str, Any],
    target_label: str,
    return_diagnostics: bool = False,
    oof_output_path: str = None,
) -> tuple:
    """Cross-validated counterpart to `evaluate_filter_impact`.

    Every dataset passed in must already carry `admission_ids` (call
    `DatasetEHR.load_admission_ids`, populated by rerun/regen_admission_ids.py).
    Without them there is no way to pair arms, which is the entire point.

    Returns the **same five positions** as the legacy function so the existing
    table and plot helpers work unchanged, but the semantics differ and the two
    are not comparable:

        [0] mean in-fold training accuracy, at the fold's own threshold
        [1] out-of-fold accuracy, mean across repeats
        [2] mean in-fold training macro F1, same threshold
        [3] out-of-fold macro F1, mean across repeats
        [4] (statistic, pvalue) from the paired, correctness-based McNemar

    Positions 0 and 1 now describe one classifier at one operating point. In the
    legacy tuple position 0 was a 4-fold CV score taken at the implicit 0.5 while
    position 1 was a holdout score at the Youden threshold, so the two columns
    sitting next to each other described different classifiers.

    The partition is drawn once on the raw cohort and inherited by every arm, so
    `raw` at index 0 is a genuine paired baseline rather than a coincidence of
    equal lengths.

    `oof_output_path` writes the out-of-fold scores to a .npz. Worth doing: it
    makes any other metric or threshold rule re-derivable without refitting.
    """
    label_index = raw_dataset.label_index_map[target_label]

    ordered_names = ['raw'] + list(filtered_datasets.keys())
    all_datasets = {'raw': raw_dataset, **filtered_datasets}

    for name, dataset in all_datasets.items():
        if getattr(dataset, 'admission_ids', None) is None:
            raise ValueError(
                f"arm '{name}' has no admission_ids; run rerun/regen_admission_ids.py "
                "and load the sidecar before calling this."
            )

    raw_ids = np.asarray(raw_dataset.admission_ids, dtype=np.int64)
    _, raw_labels = _flatten_whole_dataset(raw_dataset, label_index)

    assignment = build_fold_assignment(raw_ids, raw_labels)
    labels_by_id = {int(i): int(y) for i, y in zip(raw_ids, raw_labels)}
    partition_fingerprint = fingerprint(raw_ids, raw_labels)

    arm_results = {}
    for name in ordered_names:
        dataset = all_datasets[name]
        arm_ids = np.asarray(dataset.admission_ids, dtype=np.int64)
        arm_assignment = project_to_arm(arm_ids, raw_ids, assignment)
        arm_results[name] = evaluate_dataset_label_cv(
            dataset, arm_ids, label_index, arm_assignment
        )

    training_averages, testing_averages = [], []
    training_f1s, testing_f1s, mcnemar_results = [], [], []
    diagnostics = {}

    baseline = arm_results['raw']

    for name in ordered_names:
        result = arm_results[name]
        training_averages.append(result['train_accuracy'])
        testing_averages.append(result['diagnostics']['accuracy_mean'])
        training_f1s.append(result['train_f1_macro'])
        testing_f1s.append(result['diagnostics']['f1_macro_mean'])

        statistic, p_value, detail = calculate_mcnemar_paired(
            baseline['consensus_predictions'], baseline['admission_ids'],
            result['consensus_predictions'], result['admission_ids'],
            labels_by_id,
        )
        mcnemar_results.append((statistic, p_value))

        diagnostics[name] = {
            **result['diagnostics'],
            'per_repeat': result['per_repeat'],
            'thresholds': result['thresholds'],
            'mcnemar': detail,
            'partition_fingerprint': partition_fingerprint,
        }

    results = (training_averages, testing_averages, training_f1s, testing_f1s, mcnemar_results)

    if oof_output_path is not None:
        payload = {'raw_admission_ids': raw_ids, 'raw_labels': raw_labels,
                   'fold_assignment': assignment}
        for name in ordered_names:
            slug = name.replace(' ', '_').lower()
            payload[f'{slug}__admission_ids'] = arm_results[name]['admission_ids']
            payload[f'{slug}__oof_scores'] = arm_results[name]['oof_scores'].astype(np.float32)
            payload[f'{slug}__consensus_predictions'] = arm_results[name]['consensus_predictions']
        np.savez_compressed(oof_output_path, **payload)

    if not return_diagnostics:
        return results

    return results + (diagnostics,)


def compute_tensor_centroid(tensor: torch.Tensor):
    """Per-vital mean for one patient, ignoring hours with no measurement.

    Masked rather than using torch.nanmean because 0 has to count as missing too
    — RecordEHR.to_tensor() zero-fills NaNs, so by the time we see a tensor the
    gaps are zeros and a plain mean would drag every vital toward 0.

    A vital with no valid readings at all divides by zero and yields NaN. That's
    intentional and callers filter it out; substituting 0 would put the patient
    at the origin and skew the dataset centroid.
    """
    invalid = torch.isnan(tensor) | (tensor == 0)
    mask = ~invalid

    column_sums = torch.where(mask, tensor, 0.0).sum(dim=0)
    column_counts = mask.sum(dim=0)

    return (column_sums / column_counts).tolist()

def compute_dataset_centroid(dataset: DatasetEHR, label: str=None, polarity: str='pos'):
    """Average the per-patient centroids over one class, returning both the mean
    and the individual points behind it.

    Both are returned because the plots need both: a bar chart of how far the
    centroid moved, and a violin showing the spread it moved within. A shift
    that's tiny next to the spread isn't interesting, and you can't tell that
    from the centroid alone.

    Passing label=None sets label_index to -1, which the checks below read as
    "no filtering, take everything" — not as "index the last label".

    Careful: `label` gets reassigned inside the loop below, so it no longer holds
    the caller's argument after this point. label_index is captured first, which
    is why that doesn't break anything.
    """
    label_index = dataset.label_index_map[label] if label is not None else -1

    assert polarity in ['pos', 'neg']
    polarity_index = 1 if polarity == 'pos' else 0

    tensor_centroids = []
    tensor_labels = []
    for entry in dataset.data:
        tensor = entry[0]
        label = entry[1]
        tensor_centroids.append(compute_tensor_centroid(tensor))
        tensor_labels.append(label)

    if len(tensor_centroids) == 0:
        return [], []

    num_vitals = len(tensor_centroids[0])
    # Per-vital counters rather than one divisor, since each patient may be NaN
    # for a different subset of vitals — a shared denominator would understate
    # any vital that's usually present.
    dataset_centroid_sums = [0 for _ in range(num_vitals)]
    dataset_centroid_counts = [0 for _ in range(num_vitals)]
    for i in range(len(tensor_centroids)):
        if tensor_labels[i][label_index] == polarity_index or label_index == -1:
            for j in range(num_vitals):
                if not math.isnan(tensor_centroids[i][j]):
                    dataset_centroid_sums[j] += tensor_centroids[i][j]
                    dataset_centroid_counts[j] += 1

    # Same membership test as above, so the violin plot shows exactly the
    # sub-population the centroid was computed from. Note these still include
    # NaNs — the plotting code drops them per-vital.
    filtered_points = [
        tensor_centroids[i]
        for i in range(len(tensor_centroids))
        if tensor_labels[i][label_index] == polarity_index or label_index == -1
    ]
    return [dataset_centroid_sums[i] / dataset_centroid_counts[i] for i in range(num_vitals)], filtered_points

# ─────────────────────────────────────────────────────────────────────────────
# Class-balanced evaluation (stage H, 2026-08-20)
#
# Everything above corrects class imbalance at the *decision* layer: the forest is
# fitted on the natural distribution and `pick_threshold` then moves the operating
# point off 0.5. This path corrects it at the *training data* layer instead —
# resample the training fold to 1:1, fit, and score at a fixed 0.5. That is the
# other half of bug_register.py E-11, and the two halves are deliberately not
# combined here: stacking a resampler and a tuned threshold would leave no way to
# say which one produced the change.
#
# Consequences of dropping threshold selection, both intended:
#   * there is no inner-validation split at all, so a fold trains on the full 80%
#     rather than 7/8 of it;
#   * accuracy is read at 0.5 on a test fold that keeps its natural prevalence.
#
# These results are NOT comparable to the `_cv` pickles above, which are scored at
# each fold's own Youden threshold. They are written to their own filenames.
# ─────────────────────────────────────────────────────────────────────────────

FIXED_THRESHOLD = 0.5


def _score_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """The metric block reported for one out-of-fold vector.

    Accuracy alone is close to useless at 9.7% prevalence — predicting all-negative
    scores 0.903 — so it travels with four companions. Balanced accuracy and macro
    F1 weight the minority class up; AUROC and AUPRC are computed from the raw
    scores and so are independent of the 0.5 cut, which makes them the only
    numbers here that compare strategies without the operating point confounding
    the answer. AUPRC specifically is what E-11's fix field asks for: at this
    prevalence a PR curve separates useful models far better than a ROC does.

    Single-class inputs make the ranking metrics undefined; they come back NaN
    rather than raising, and `_mean_of_finite` upstream drops them.
    """
    if y_true.size == 0:
        return {k: float('nan') for k in
                ('accuracy', 'f1_macro', 'balanced_accuracy', 'recall', 'auroc', 'auprc')}

    both_classes = len(np.unique(y_true)) > 1
    return {
        'accuracy': float(np.mean(y_pred == y_true)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)) if both_classes else float('nan'),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'auroc': float(roc_auc_score(y_true, scores)) if both_classes else float('nan'),
        'auprc': float(average_precision_score(y_true, scores)) if both_classes else float('nan'),
    }


def evaluate_dataset_label_cv_balanced(
    dataset: Any,
    admission_ids: np.ndarray,
    label_index: int,
    fold_assignment: np.ndarray,
    strategy: str,
) -> Dict[str, Any]:
    """Repeated stratified k-fold for one arm under one balancing strategy.

    Structurally the same as `evaluate_dataset_label_cv` — same partition, same
    forest, same out-of-fold bookkeeping — with two deliberate differences:
    `inner_split`/`pick_threshold` are not called at all, and the training rows
    are resampled by `balance_training_fold` before the fit.

    **Why the held-out fold cannot be contaminated.** The resampled matrix is a
    fold-local variable; test features are always read out of the original
    `features` matrix by position (`features[test_positions]`), never out of it.
    A synthetic row therefore has no position in the arm's row order and is not
    representable in `oof_scores`/`oof_predictions` at all. `assert_no_leakage`
    then checks the weaker but auditable property — that no real row in the
    training set came from the held-out fold — on every fold, not behind a flag.

    Note the training set here is the full 80% of the cohort, where the threshold-
    selecting path trains on 7/8 of that. Fitting is correspondingly slower, and
    a `_cv` number is not a fair baseline for one of these. See
    `baseline_from_cached_scores` for the comparable baseline.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")

    n_records = len(dataset.data)
    admission_ids = np.asarray(admission_ids, dtype=np.int64)

    if n_records != len(admission_ids):
        raise ValueError(f"{n_records:,} records against {len(admission_ids):,} admission ids")

    n_repeats = fold_assignment.shape[0]
    features, labels = _flatten_whole_dataset(dataset, label_index)

    # One host copy for the resampler, which is pure numpy by design. Under the
    # CPU backend cp.asnumpy is np.asarray and this is free; on a real GPU it is
    # one device→host transfer per arm rather than per fold.
    features_host = np.asarray(cp.asnumpy(features)) if n_records else np.empty((0, 0))

    oof_scores = np.full((n_repeats, n_records), np.nan, dtype=np.float64)
    oof_predictions = np.full((n_repeats, n_records), -1, dtype=np.int8)
    train_accuracies: List[float] = []
    train_f1s: List[float] = []
    balance_log: Dict[str, Dict[str, Any]] = {}
    n_folds_fitted = 0

    for repeat in range(n_repeats):
        for fold in range(N_SPLITS):
            test_positions = np.where(fold_assignment[repeat] == fold)[0]
            train_positions = np.where(fold_assignment[repeat] != fold)[0]

            if test_positions.size == 0 or train_positions.size == 0:
                continue

            # Same guard as the threshold path: a single-class training side
            # gives a forest that predicts one label everywhere, and there is
            # nothing for a resampler to balance either.
            if len(np.unique(labels[train_positions])) < 2:
                continue

            balanced = balance_training_fold(
                strategy, features_host, labels, train_positions, repeat, fold
            )
            assert_no_leakage(balanced, train_positions, test_positions)
            balance_log[f"r{repeat}f{fold}"] = balanced.stats

            seed = forest_seed(repeat, fold)
            model = RandomForestClassifier(n_estimators=300, random_state=seed)
            model.fit(cp.asarray(balanced.x), cp.asarray(balanced.y))

            test_scores = _positive_scores(model, features[cp.asarray(test_positions)])
            oof_scores[repeat, test_positions] = test_scores
            oof_predictions[repeat, test_positions] = (test_scores >= FIXED_THRESHOLD).astype(np.int8)
            n_folds_fitted += 1

            # Scored on the *original* training rows, not the resampled ones, so
            # this stays comparable across strategies — an oversampled fold would
            # otherwise be graded on a set containing each minority patient
            # several times.
            fit_scores = _positive_scores(model, features[cp.asarray(train_positions)])
            fit_predictions = (fit_scores >= FIXED_THRESHOLD).astype(int)
            train_accuracies.append(float(np.mean(fit_predictions == labels[train_positions])))
            train_f1s.append(float(f1_score(labels[train_positions], fit_predictions, average='macro')))

    per_repeat = []
    for repeat in range(n_repeats):
        covered = oof_predictions[repeat] >= 0
        if not covered.any():
            per_repeat.append({**_score_metrics(np.array([]), np.array([]), np.array([])), 'n': 0})
            continue
        per_repeat.append({
            **_score_metrics(labels[covered], oof_predictions[repeat][covered], oof_scores[repeat][covered]),
            'n': int(covered.sum()),
        })

    covered_mask = ~np.isnan(oof_scores)
    covered_count = covered_mask.sum(axis=0)
    consensus_scores = np.full(n_records, np.nan, dtype=np.float64)
    has_any = covered_count > 0
    consensus_scores[has_any] = np.nansum(oof_scores[:, has_any], axis=0) / covered_count[has_any]

    consensus_covered = ~np.isnan(consensus_scores)
    consensus_predictions = np.full(n_records, -1, dtype=np.int8)
    consensus_predictions[consensus_covered] = (
        consensus_scores[consensus_covered] >= FIXED_THRESHOLD
    ).astype(np.int8)

    synthetic_total = sum(int(s['n_synthetic']) for s in balance_log.values())
    diagnostics = {
        'strategy': strategy,
        'threshold': FIXED_THRESHOLD,
        'n_records': int(n_records),
        'coverage': float(np.mean(consensus_covered)) if n_records else 0.0,
        'n_folds_fitted': n_folds_fitted,
        'n_synthetic_total': synthetic_total,
        'smote_fallback_folds': sum(1 for s in balance_log.values() if s['smote_fallback']),
        'mean_train_rows': (float(np.mean([s['n_train_out'] for s in balance_log.values()]))
                            if balance_log else float('nan')),
        'predicted_positive_rate': (float(np.mean(consensus_predictions[consensus_covered]))
                                    if consensus_covered.any() else float('nan')),
        'baseline_trained_on_fraction': 1.0 - 1.0 / N_SPLITS,
        'degenerate': n_folds_fitted == 0,
    }
    for metric in ('accuracy', 'f1_macro', 'balanced_accuracy', 'recall', 'auroc', 'auprc'):
        diagnostics[f'{metric}_mean'] = _mean_of_finite([r[metric] for r in per_repeat])
        diagnostics[f'{metric}_std'] = _std_of_finite([r[metric] for r in per_repeat])
    diagnostics.update(fold_prevalences(fold_assignment, labels))

    return {
        'admission_ids': admission_ids,
        'y': labels,
        'oof_scores': oof_scores,
        'oof_predictions': oof_predictions,
        'consensus_scores': consensus_scores,
        'consensus_predictions': consensus_predictions,
        'per_repeat': per_repeat,
        'balance_log': balance_log,
        'train_accuracy': float(np.mean(train_accuracies)) if train_accuracies else float('nan'),
        'train_f1_macro': float(np.mean(train_f1s)) if train_f1s else float('nan'),
        'diagnostics': diagnostics,
    }


def baseline_from_cached_scores(
    npz_path: str, arm: str, fold_assignment: np.ndarray
) -> Dict[str, Any]:
    """Re-derive the unbalanced baseline from stage F's cached out-of-fold scores.

    Stage F already fitted these forests and wrote every record's out-of-fold
    *score* to `<label>_oof_scores_cv.npz`. Scores do not depend on the operating
    point, so the no-balancing arm can be re-read at a fixed 0.5 without fitting
    anything — which is the whole reason the baseline costs nothing here.

    What is *not* reusable is the cached accuracy in `<label>_filter_impact_cv.pkl`:
    that was taken at each fold's Youden threshold and would flatter the baseline
    against a 0.5-thresholded balanced arm. Reading the scores and re-thresholding
    is the correction.

    One caveat the caller must carry, and which is written into the returned
    diagnostics as `baseline_trained_on_fraction`: these forests were fitted on
    `fit_positions`, i.e. 7/8 of the 80% training fold, because stage F carved off
    an inner-validation slice. The balanced arms train on the whole 80%. The
    baseline is therefore fitted on ~12% less data, which biases the comparison in
    favour of the balancing strategies. `regen_balance_cv.py --refit-baseline`
    removes the caveat by refitting instead; it costs 20 fits per (aggregation,
    label).

    Training-split metrics cannot be recovered from the npz and come back NaN.
    """
    slug = arm.replace(' ', '_').lower()
    with np.load(npz_path) as payload:
        required = f'{slug}__oof_scores'
        if required not in payload:
            raise KeyError(
                f"{required!r} not in {npz_path}; stage F must have run for this "
                f"(aggregation, label) before the baseline can be reused"
            )
        oof_scores = payload[required].astype(np.float64)
        admission_ids = payload[f'{slug}__admission_ids'].astype(np.int64)
        labels = payload['raw_labels'].astype(np.int64)
        raw_ids = payload['raw_admission_ids'].astype(np.int64)

    # The npz stores raw's labels only, so an arm that dropped records needs its
    # own labels looked up by id rather than assumed positionally.
    if not np.array_equal(admission_ids, raw_ids):
        lookup = {int(i): int(y) for i, y in zip(raw_ids, labels)}
        labels = np.array([lookup[int(i)] for i in admission_ids], dtype=np.int64)

    n_records = admission_ids.size
    oof_predictions = np.where(np.isnan(oof_scores), -1,
                               (oof_scores >= FIXED_THRESHOLD).astype(np.int8)).astype(np.int8)

    per_repeat = []
    for repeat in range(oof_scores.shape[0]):
        covered = oof_predictions[repeat] >= 0
        if not covered.any():
            per_repeat.append({**_score_metrics(np.array([]), np.array([]), np.array([])), 'n': 0})
            continue
        per_repeat.append({
            **_score_metrics(labels[covered], oof_predictions[repeat][covered], oof_scores[repeat][covered]),
            'n': int(covered.sum()),
        })

    covered_mask = ~np.isnan(oof_scores)
    covered_count = covered_mask.sum(axis=0)
    consensus_scores = np.full(n_records, np.nan, dtype=np.float64)
    has_any = covered_count > 0
    consensus_scores[has_any] = np.nansum(oof_scores[:, has_any], axis=0) / covered_count[has_any]

    consensus_covered = ~np.isnan(consensus_scores)
    consensus_predictions = np.full(n_records, -1, dtype=np.int8)
    consensus_predictions[consensus_covered] = (
        consensus_scores[consensus_covered] >= FIXED_THRESHOLD
    ).astype(np.int8)

    diagnostics = {
        'strategy': 'none',
        'threshold': FIXED_THRESHOLD,
        'n_records': int(n_records),
        'coverage': float(np.mean(consensus_covered)) if n_records else 0.0,
        'n_folds_fitted': 0,
        'n_synthetic_total': 0,
        'smote_fallback_folds': 0,
        'mean_train_rows': float('nan'),
        'predicted_positive_rate': (float(np.mean(consensus_predictions[consensus_covered]))
                                    if consensus_covered.any() else float('nan')),
        # 7/8 of the 80% training fold — see the docstring.
        'baseline_trained_on_fraction': (1.0 - 1.0 / N_SPLITS) * (1.0 - INNER_VALIDATION_SHARE),
        'reused_from_cache': True,
        'source_npz': str(npz_path),
        'degenerate': False,
    }
    for metric in ('accuracy', 'f1_macro', 'balanced_accuracy', 'recall', 'auroc', 'auprc'):
        diagnostics[f'{metric}_mean'] = _mean_of_finite([r[metric] for r in per_repeat])
        diagnostics[f'{metric}_std'] = _std_of_finite([r[metric] for r in per_repeat])
    diagnostics.update(fold_prevalences(fold_assignment, labels))

    return {
        'admission_ids': admission_ids,
        'y': labels,
        'oof_scores': oof_scores,
        'oof_predictions': oof_predictions,
        'consensus_scores': consensus_scores,
        'consensus_predictions': consensus_predictions,
        'per_repeat': per_repeat,
        'balance_log': {},
        'train_accuracy': float('nan'),
        'train_f1_macro': float('nan'),
        'diagnostics': diagnostics,
    }


def evaluate_balance_impact_cv(
    raw_dataset: Any,
    datasets: Dict[str, Any],
    target_label: str,
    strategies: Tuple[str, ...] = STRATEGIES,
    baseline_npz_path: str = None,
    refit_baseline: bool = False,
    return_diagnostics: bool = False,
    oof_output_path: str = None,
) -> tuple:
    """Score every (arm, strategy) pair against the unbalanced baseline.

    `datasets` is the arm map to sweep — `{'raw': raw_dataset}` today, the full
    13-arm map later; nothing in here assumes the smaller case. `raw_dataset` is
    separate because the partition is always drawn on the raw cohort and inherited
    by every arm via `project_to_arm`, exactly as in `evaluate_filter_impact_cv`.
    That is what makes the arms paired, and it is also what lets stage F's cached
    scores be dropped in as the baseline: same PARTITION_SEED, same folds.

    Returns the **same five positions** as the two older sweeps so the existing
    table and plot helpers work unchanged:

        [0] mean in-fold training accuracy   (NaN for a cache-reused baseline)
        [1] out-of-fold accuracy at 0.5, mean across repeats
        [2] mean in-fold training macro F1   (NaN for a cache-reused baseline)
        [3] out-of-fold macro F1 at 0.5, mean across repeats
        [4] (statistic, pvalue) from the paired, correctness-based McNemar

    Index 0 is `<first arm> / none`, the baseline every other entry is compared
    against — the same positional convention `filter_impact_plot` and the
    comparison notebook rely on. The richer metrics (balanced accuracy, recall,
    AUROC, AUPRC) do not fit in five positions and live in the diagnostics
    sidecar; the tuple shape is frozen because
    `comparison/pipeline_comparison.ipynb` unpacks it with a strict five-target
    assignment that raises on six.

    `baseline_npz_path` points at stage F's `<label>_oof_scores_cv.npz`. With
    `refit_baseline=True` the 'none' arm is refitted at 0.5 with no inner
    validation split instead, which costs 5 x n_repeats fits per arm but removes
    the training-set-size confound described in `baseline_from_cached_scores`.
    """
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"unknown strategies {sorted(unknown)}; expected from {STRATEGIES}")
    if 'none' in strategies and baseline_npz_path is None and not refit_baseline:
        raise ValueError(
            "the 'none' arm needs either baseline_npz_path (reuse stage F's cached "
            "scores) or refit_baseline=True"
        )

    label_index = raw_dataset.label_index_map[target_label]

    for name, dataset in {'raw': raw_dataset, **datasets}.items():
        if getattr(dataset, 'admission_ids', None) is None:
            raise ValueError(
                f"arm '{name}' has no admission_ids; run rerun/regen_admission_ids.py "
                "and load the sidecar before calling this."
            )

    raw_ids = np.asarray(raw_dataset.admission_ids, dtype=np.int64)
    _, raw_labels = _flatten_whole_dataset(raw_dataset, label_index)

    assignment = build_fold_assignment(raw_ids, raw_labels)
    labels_by_id = {int(i): int(y) for i, y in zip(raw_ids, raw_labels)}
    partition_fingerprint = fingerprint(raw_ids, raw_labels)

    # Flat (arm, strategy) ordering, arms outermost, so index 0 is the first
    # arm's 'none' and the layout generalises unchanged from 1 arm to 13.
    ordered_pairs = [(arm, strategy) for arm in datasets for strategy in strategies]
    results_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for arm, strategy in ordered_pairs:
        dataset = datasets[arm]
        arm_ids = np.asarray(dataset.admission_ids, dtype=np.int64)
        arm_assignment = project_to_arm(arm_ids, raw_ids, assignment)

        if strategy == 'none' and not refit_baseline:
            results_by_pair[(arm, strategy)] = baseline_from_cached_scores(
                baseline_npz_path, arm, arm_assignment
            )
        else:
            results_by_pair[(arm, strategy)] = evaluate_dataset_label_cv_balanced(
                dataset, arm_ids, label_index, arm_assignment, strategy
            )

    training_averages, testing_averages = [], []
    training_f1s, testing_f1s, mcnemar_results = [], [], []
    diagnostics = {}

    baseline = results_by_pair[ordered_pairs[0]]

    for pair in ordered_pairs:
        result = results_by_pair[pair]
        name = f"{pair[0]} / {pair[1]}"

        training_averages.append(result['train_accuracy'])
        testing_averages.append(result['diagnostics']['accuracy_mean'])
        training_f1s.append(result['train_f1_macro'])
        testing_f1s.append(result['diagnostics']['f1_macro_mean'])

        statistic, p_value, detail = calculate_mcnemar_paired(
            baseline['consensus_predictions'], baseline['admission_ids'],
            result['consensus_predictions'], result['admission_ids'],
            labels_by_id,
        )
        mcnemar_results.append((statistic, p_value))

        diagnostics[name] = {
            **result['diagnostics'],
            'arm': pair[0],
            'per_repeat': result['per_repeat'],
            'balance_log': result['balance_log'],
            'mcnemar': detail,
            'partition_fingerprint': partition_fingerprint,
        }

    results = (training_averages, testing_averages, training_f1s, testing_f1s, mcnemar_results)

    if oof_output_path is not None:
        payload = {'raw_admission_ids': raw_ids, 'raw_labels': raw_labels,
                   'fold_assignment': assignment}
        for pair in ordered_pairs:
            slug = f"{pair[0]}__{pair[1]}".replace(' ', '_').lower()
            result = results_by_pair[pair]
            payload[f'{slug}__admission_ids'] = result['admission_ids']
            payload[f'{slug}__oof_scores'] = result['oof_scores'].astype(np.float32)
            payload[f'{slug}__consensus_predictions'] = result['consensus_predictions']
        np.savez_compressed(oof_output_path, **payload)

    if not return_diagnostics:
        return results

    return results + (diagnostics,)
