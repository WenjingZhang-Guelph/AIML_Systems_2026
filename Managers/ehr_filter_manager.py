"""The eleven filters, in the order they're meant to run.

Split into two groups that are NOT interchangeable, which is the single most
important thing to know about this file:

  1-7   per-vital outlier removal. These operate on the raw list-per-cell
        timeseries and prune individual measurements.
  8-11  structural filters. These assume cells are scalars and use isna() and
        == 0 to spot gaps, so they only make sense on an already-aggregated
        record.

Experiments/apply_dataset_filter.py enforces this with an `index < 7` check.
Run a structural filter on un-aggregated data and it won't crash — it'll just
quietly find no missing data anywhere, because a list is never NaN.

FILTERS at the bottom is an OrderedDict and the order is load-bearing:
downstream code zips results against filter names positionally, and cached
results under Data/<dataset>/<aggregation>/ are keyed by it.
"""

from typing import List, Tuple, OrderedDict, Dict, Optional
import numpy as np
from Entities.ehr_filter_change_tracker import FilterChangeTrackerEHR
from Entities.ehr_record import RecordEHR
from copy import deepcopy
import pandas as pd
from pandas import DataFrame
from scipy.interpolate import UnivariateSpline
from sklearn.impute import KNNImputer


def _remove_outliers(records: List[RecordEHR], filter_name: str, column_index: int, inclusive_range: Tuple[float, float]) -> Tuple[List[RecordEHR], List[FilterChangeTrackerEHR]]:
    """Drop measurements outside a vital's plausible range, keeping the rest.

    Note this prunes *values*, not records — a heart rate of 900 disappears and
    the hour keeps its other readings. That's the point: a single fat-fingered
    entry shouldn't cost us a whole patient.

    Each vital is addressed by `column_index`, a hardcoded 0-6 that has to match
    the column order the processing scripts wrote. Renaming or reordering the
    VITALS dict silently applies heart-rate bounds to blood pressure.
    """
    filtered_records = []
    changes = []

    for record in records:
        timeseries = record.timeseries.copy()
        column_values = timeseries.iloc[:, column_index].copy()

        pop_count = 0
        for index, cell in column_values.items():
            if cell is None:
                continue

            cell = list(cell)  # don't mutate the original record's list in place

            # Backwards, because we're popping by index while iterating.
            for i in range(len(cell) - 1, -1, -1):
                if cell[i] is None:
                    continue
                if not (inclusive_range[0] <= cell[i] <= inclusive_range[1]):
                    cell.pop(i)
                    pop_count += 1

            column_values.iloc[index] = cell

        timeseries.iloc[:, column_index] = column_values

        new_record = deepcopy(record)
        new_record.timeseries = timeseries
        filtered_records.append(new_record)

        record_change = FilterChangeTrackerEHR(filter_name, record.admission_id)
        record_change.value_level += pop_count
        changes.append(record_change)

    return filtered_records, changes


# Seven near-identical wrappers rather than one parameterised filter, because
# every filter in FILTERS has to share the same (records, vitals) signature so
# the experiment runner can call them uniformly.
def heart_rate_filter(records: List[RecordEHR], vitals: Dict[str, Tuple[Tuple[int, int], str]]) -> Tuple[List[RecordEHR], List[FilterChangeTrackerEHR]]:
    return _remove_outliers(records=records, filter_name='heart rate', column_index=0, inclusive_range=vitals['heart rate'][0])

def systolic_blood_pressure_filter(records: List[RecordEHR], vitals: Dict[str, Tuple[Tuple[int, int], str]]) -> Tuple[List[RecordEHR], List[FilterChangeTrackerEHR]]:
    return _remove_outliers(records=records, filter_name='systolic blood pressure', column_index=1, inclusive_range=vitals['systolic blood pressure'][0])

def diastolic_blood_pressure_filter(records: List[RecordEHR], vitals: Dict[str, Tuple[Tuple[int, int], str]]) -> Tuple[List[RecordEHR], List[FilterChangeTrackerEHR]]:
    return _remove_outliers(records=records, filter_name='diastolic blood pressure', column_index=2, inclusive_range=vitals['diastolic blood pressure'][0])

def mean_blood_pressure_filter(records: List[RecordEHR], vitals: Dict[str, Tuple[Tuple[int, int], str]]) -> Tuple[List[RecordEHR], List[FilterChangeTrackerEHR]]:
    return _remove_outliers(records=records, filter_name='mean blood pressure', column_index=3, inclusive_range=vitals['mean blood pressure'][0])

def respiration_rate_filter(records: List[RecordEHR], vitals: Dict[str, Tuple[Tuple[int, int], str]]) -> Tuple[List[RecordEHR], List[FilterChangeTrackerEHR]]:
    return _remove_outliers(records=records, filter_name='respiration rate', column_index=4, inclusive_range=vitals['respiration rate'][0])

def temperature_filter(records: List[RecordEHR], vitals: Dict[str, Tuple[Tuple[int, int], str]]) -> Tuple[List[RecordEHR], List[FilterChangeTrackerEHR]]:
    return _remove_outliers(records=records, filter_name='temperature', column_index=5, inclusive_range=vitals['temperature'][0])

def oxygen_saturation_filter(records: List[RecordEHR], vitals: Dict[str, Tuple[Tuple[int, int], str]]) -> Tuple[List[RecordEHR], List[FilterChangeTrackerEHR]]:
    return _remove_outliers(records=records, filter_name='oxygen saturation', column_index=6, inclusive_range=vitals['oxygen saturation'][0])

def fill_missing_data_filter(records: List[RecordEHR], vitals: Dict[str, Tuple[Tuple[int, int], str]]) -> Tuple[List[RecordEHR], List[FilterChangeTrackerEHR]]:
    """KNN-impute gaps in each vital's 24-hour trace.

    Post-aggregate only — `.astype(float)` on a column of lists throws.

    Imputes one column at a time rather than all seven together, so a patient's
    heart rate is filled from their own neighbouring hours and never from their
    blood pressure. Cross-vital imputation would leak correlations we're
    specifically trying to measure.

    The `>= 2 known values` guard is there because imputing a trace with one real
    reading just smears that value across 24 hours, which looks like data but isn't.
    Only in-range readings count toward the guard — a trace whose real values are
    all physiologically impossible has nothing trustworthy to impute from.
    """
    processed_records = []
    changes = []

    for record in records:
        df_copy = record.timeseries.copy()

        for column_name in df_copy.columns:
            column_data = df_copy[column_name].values.astype(float)

            valid_range = vitals[column_name][0]
            known_indices = np.where(~np.isnan(column_data))[0]
            known_indices = known_indices[(column_data[known_indices] >= valid_range[0]) & (column_data[known_indices] <= valid_range[1])]

            missing_indices = np.where(np.isnan(column_data))[0]

            if len(missing_indices) > 0 and len(known_indices) >= 2:
                imputer = KNNImputer(n_neighbors=3)

                # Reshaped to a single column so KNNImputer treats each hour as
                # its own sample — neighbours are then nearby hours, which is
                # what we want for a time series.
                col_2d = column_data.reshape(-1, 1)
                filled = imputer.fit_transform(col_2d).flatten()
                df_copy.iloc[:, df_copy.columns.get_loc(column_name)] = filled

                record_change = FilterChangeTrackerEHR('fill missing data', record.admission_id)
                record_change.feature_level += len(missing_indices)
                changes.append(record_change)

        new_record = RecordEHR(
            admission_id=record.admission_id,
            timeseries=df_copy,
            icu=record.icu,
            mortality=record.mortality
        )
        processed_records.append(new_record)

    return processed_records, changes

def long_missing_segment_filter(records: List[RecordEHR], vitals: Optional[Dict[str, Tuple[Tuple[int, int], str]]]) -> Tuple[List[RecordEHR], List[FilterChangeTrackerEHR]]:
    """Throw out records where any single vital goes dark for 10+ straight hours.

    Unlike the outlier filters this drops whole records, because there's no
    sensible way to repair a 10-hour hole — imputing across it invents most of
    the trace.

    Zero counts as missing throughout. Not strictly true (a heart rate really can
    be 0), but a charted 0 in this data is overwhelmingly a placeholder, and it's
    the same convention the other structural filters use.

    Post-aggregate only: on raw list-per-cell data, isna() is False everywhere
    and nothing is ever dropped.

    The `len(df) < 10` early-out is dead code in practice — every record is
    exactly 24 rows by construction. Harmless, and it'd matter if the window
    ever became configurable.
    """
    kept_records = []
    changes = []

    for record in records:
        df = record.timeseries

        if len(df) < 10:
            kept_records.append(record)
            continue

        drop_record = False

        for col in df.columns:
            is_missing = df[col].isna() | (df[col] == 0)

            if not is_missing.any():
                continue

            # Standard trick for finding consecutive runs: the cumsum of
            # "value changed from the previous row" gives each run a unique id,
            # then summing the boolean within each group counts run length.
            groups = (is_missing != is_missing.shift()).cumsum()

            run_lengths = is_missing.groupby(groups).sum()

            # First offending vital is enough — break out rather than checking
            # the rest, the record is already going.
            if not run_lengths.empty and run_lengths.max() >= 10:
                record_change = FilterChangeTrackerEHR('long missing segment', record.admission_id)
                record_change.sample_level += 1
                changes.append(record_change)

                drop_record = True
                break

        if not drop_record:
            kept_records.append(record)

    return kept_records, changes


def long_gap_filter(records: List[RecordEHR], vitals: Optional[Dict[str, Tuple[Tuple[int, int], str]]]) -> Tuple[List[RecordEHR], List[FilterChangeTrackerEHR]]:
    """Drop records with 2+ consecutive hours of nothing charted at all.

    The stricter cousin of long_missing_segment_filter: that one asks whether any
    single vital went quiet, this one asks whether *everything* did. Two
    consecutive blank hours usually means the patient left the unit — off to
    imaging, theatre, discharged — so the trace isn't 24 hours of ICU stay at all.

    Hence the much lower threshold. One blank hour is routine; two in a row isn't.
    """
    kept_records = []
    changes = []

    for record in records:
        df = record.timeseries

        if len(df) < 2:
            kept_records.append(record)
            continue

        # all(axis=1) — every vital has to be missing for the hour to count as blank.
        is_empty_row = (df.isna() | (df == 0)).all(axis=1)

        row_groups = (is_empty_row != is_empty_row.shift()).cumsum()

        empty_run_lengths = is_empty_row.groupby(row_groups).sum()

        if not empty_run_lengths.empty and empty_run_lengths.max() >= 2:
            record_change = FilterChangeTrackerEHR('long gap', record.admission_id)
            record_change.sample_level += 1
            changes.append(record_change)

            continue

        kept_records.append(record)

    return kept_records, changes


def high_invalid_data_filter(records: List[RecordEHR], vitals: Optional[Dict[str, Tuple[Tuple[int, int], str]]], threshold: float = 0.10) -> Tuple[List[RecordEHR], List[FilterChangeTrackerEHR]]:
    """Drop records more than 10% empty across the whole 24×7 grid.

    Where the two gap filters look for *shape* — a run of blank hours — this one
    just counts holes wherever they fall. Scattered missingness is what it
    catches.

    10% of 168 cells is about 17, which is aggressive: plenty of clinically
    normal records get cut because temperature is only charted every few hours.
    That's deliberate. This is the harshest filter in the set, and seeing how
    much accuracy survives it is a large part of what the filter-impact
    experiments are asking.
    """
    kept_records = []
    changes = []

    for record in records:
        df = record.timeseries

        total_cells = df.size

        invalid_mask = df.isna() | (df == 0)
        invalid_cells = invalid_mask.values.sum()

        invalid_ratio = invalid_cells / total_cells

        if invalid_ratio < threshold:
            kept_records.append(record)

        else:
            record_change = FilterChangeTrackerEHR('high invalid data', record.admission_id)
            record_change.sample_level += 1
            changes.append(record_change)

    return kept_records, changes

def all_vitals_filter(records: List[RecordEHR], vitals: Dict[str, Tuple[Tuple[int, int], str]]) -> Tuple[List[RecordEHR], List[FilterChangeTrackerEHR]]:
    """All seven outlier filters back to back — the "clean everything" baseline.

    Not in FILTERS, since the experiment runner treats it as a separate arm: the
    individual filters answer "what does cleaning heart rate alone buy us", this
    answers "what does cleaning everything buy us". Order doesn't matter here,
    each one only touches its own column.
    """
    records, heart_rate_changes = heart_rate_filter(records, vitals)
    records, systolic_blood_pressure_changes = systolic_blood_pressure_filter(records, vitals)
    records, diastolic_blood_pressure_changes = diastolic_blood_pressure_filter(records, vitals)
    records, mean_blood_pressure_changes = mean_blood_pressure_filter(records, vitals)
    records, respiration_rate_changes = respiration_rate_filter(records, vitals)
    records, temperature_changes = temperature_filter(records, vitals)
    records, oxygen_saturation_changes = oxygen_saturation_filter(records, vitals)

    return records, (
        heart_rate_changes +
        systolic_blood_pressure_changes +
        diastolic_blood_pressure_changes +
        mean_blood_pressure_changes +
        respiration_rate_changes +
        temperature_changes +
        oxygen_saturation_changes
    )

def combination_filter(records: List[RecordEHR], vitals: Dict[str, Tuple[Tuple[int, int], str]], filter1, filter2, method: str = 'mean') -> Tuple[List[RecordEHR], List[FilterChangeTrackerEHR]]:
    """Chain a pre-aggregate filter and a post-aggregate one: f1 → aggregate → f2.

    Caller's job to get that right — filter1 must be one of the seven outlier
    filters and filter2 one of the four structural ones. Passing them the other
    way round doesn't raise, it just does nothing useful.

    Careful: `method` is accepted but never used — aggregate_filter is called
    without it, so this always aggregates by mean regardless of what you pass.
    """
    records1, changes1 = filter1(records, vitals)
    aggregated_records = aggregate_filter(records1)
    records2, changes2 = filter2(aggregated_records, vitals)
    return records2, changes1 + changes2

########################################################################################################################
# POST PROCESSING
########################################################################################################################


def aggregate(dataframe: DataFrame, method: str = "mean") -> DataFrame:
    """Collapse each cell's list of measurements down to one number.

    This is the step that turns a record from "lists everywhere" into something
    the structural filters and the models can actually use. Five methods, because
    which summary you pick is itself a preprocessing choice we're measuring —
    mean and median describe the level, the three deviation measures describe how
    much the vital moved within the hour, which for something like heart rate
    variability may carry more signal than the level does.

    Two things to watch:

    np.std defaults to ddof=0 (population). The comparison notebook on the
    MIMIC_Extract side uses pandas .std(), which defaults to ddof=1 (sample). On
    an hour with two or three readings that difference is not negligible, so a
    "standard deviation" comparison between the pipelines isn't quite
    apples-to-apples.

    An unrecognised method silently falls back to mean rather than raising, so a
    typo in a method name gets you plausible wrong numbers instead of an error.
    """
    new_dataframe = pd.DataFrame(np.nan, index=range(24), columns=dataframe.columns)

    ops = {
        "mean": np.mean,
        "median": np.median,
        "standard deviation": np.std,
        "mean deviation": lambda x: np.mean(np.abs(np.array(x) - np.mean(x))),
        "maximum deviation": lambda x: np.max(np.abs(np.array(x) - np.mean(x)))
    }
    func = ops.get(method, np.mean)

    for i, row_label in enumerate(dataframe.index):
        for j, col_label in enumerate(dataframe.columns):
            # Cells arrive as lists, arrays, None, or (if something already
            # aggregated this frame) bare scalars. The scalar branch makes
            # aggregate() idempotent, which matters because combination_filter
            # can end up calling it on already-aggregated records.
            cell_value = dataframe.at[row_label, col_label]
            if cell_value is None:
                average_cell_value = np.nan
            elif isinstance(cell_value, list) or isinstance(cell_value, np.ndarray):
                if len(cell_value) > 0:
                    numeric_values = [v for v in cell_value if v is not None]
                    average_cell_value = float(func(numeric_values)) if numeric_values else np.nan
                else:
                    average_cell_value = np.nan
            else:
                average_cell_value = cell_value
            new_dataframe.at[row_label, col_label] = average_cell_value

    return new_dataframe

def aggregate_filter(records: List[RecordEHR], method: str = "mean") -> List[RecordEHR]:
    filtered_records = []

    for record in records:
        new_timeseries = aggregate(record.timeseries, method=method)
        new_record = RecordEHR(record.admission_id, new_timeseries, record.icu, record.mortality)
        filtered_records.append(new_record)

    return filtered_records

# Order is part of the data contract — see the module docstring. Appending is
# safe; reordering or inserting invalidates every cached result downstream of
# Data/<dataset>/<aggregation>/, and mislabels every filter-impact plot in the
# gallery, which reads these names positionally.
FILTERS = OrderedDict([
    ('heart rate', heart_rate_filter),
    ('systolic blood pressure', systolic_blood_pressure_filter),
    ('diastolic blood pressure', diastolic_blood_pressure_filter),
    ('mean blood pressure', mean_blood_pressure_filter),
    ('respiration rate', respiration_rate_filter),
    ('temperature', temperature_filter),
    ('oxygen saturation', oxygen_saturation_filter),
    ('fill missing data', fill_missing_data_filter),
    ('long missing segment', long_missing_segment_filter),
    ('long gap', long_gap_filter),
    ('high invalid data', high_invalid_data_filter),
])