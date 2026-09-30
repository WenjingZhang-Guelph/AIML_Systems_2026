import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import sys
    from pathlib import Path
    sys.path.append(str(Path().resolve().parent))
    return Path, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fill Missing Data Filter Analysis

    A debugging notebook, kept for the record rather than as part of the pipeline.

    The `fill missing data` filter was producing wildly inflated centroids — far
    outside any plausible physiological range — and this is the investigation into
    why. The culprit turned out to be spline extrapolation: `UnivariateSpline`
    happily shoots off to huge values past the last known point, so a vital with a
    few readings early in the stay got extrapolated into nonsense for the
    remaining hours.

    Swapping to KNN imputation fixes it, since KNN can only ever produce values
    interpolated between observations it actually saw. `fill_missing_data_filter_v2`
    below is that version, and it's what now lives in
    `Managers/ehr_filter_manager.py` — so the two are the same code, and this
    notebook is the paper trail.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define Project Root
    """)
    return


@app.cell
def _():
    from Managers.path_manager import get_project_root 
    PROJECT_ROOT = get_project_root()
    return (PROJECT_ROOT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading Data
    """)
    return


@app.cell
def _():
    from Managers.dataset_manager import load_raw_dataset, load_filter_dataset

    raw_mean_dataset = load_raw_dataset('mimic-iii', "mean")
    filtered_mean_dataset = load_filter_dataset('mimic-iii', "mean", "fill missing data")[1]
    return filtered_mean_dataset, raw_mean_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing Data
    Since both datasets are the same size it means that the fill missing data filter is not deleting values.
    """)
    return


@app.cell
def _(filtered_mean_dataset, raw_mean_dataset):
    print(f'Raw Dataset Num Items: {len(raw_mean_dataset.data)}')
    print(f'Filtered Dataset Num Items: {len(filtered_mean_dataset.data)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the raw dataset the values look normal as expected, however after applying the fill missing data filter, the values syrocket towards the bottom left of the timeseries (the heart rate vital near the end of obervation window). The values even go negative in some spots. This is a clear indication that something is wrong with the fill missing data filter.
    """)
    return


@app.cell
def _(mo, raw_mean_dataset):
    import torch
    import pandas as pd
    import numpy as np
    from Managers.visualization_manager_v2 import heatmap, surface_plot

    raw_dataset_average_tensor = torch.stack([raw_mean_dataset[x][0] for x in range(len(raw_mean_dataset.data))]).mean(dim=0)
    raw_dataset_average_dataframe = pd.DataFrame(raw_dataset_average_tensor.numpy())

    mo.hstack([
        surface_plot(raw_dataset_average_dataframe, title='MIMIC-III Mean Aggregated Surface Plot', x_title='Vital', y_title='Hour', background_colour='#ffffff'),
        heatmap(raw_dataset_average_dataframe, title='MIMIC-III Mean Aggregated Heatmap', background_colour='#ffffff'),
    ])
    return heatmap, np, pd, surface_plot, torch


@app.cell
def _(filtered_mean_dataset, heatmap, mo, pd, surface_plot, torch):
    filtered_dataset_average_tensor = torch.stack([filtered_mean_dataset[x][0] for x in range(len(filtered_mean_dataset.data))]).mean(dim=0)
    filtered_dataset_average_dataframe = pd.DataFrame(filtered_dataset_average_tensor.numpy())

    mo.hstack([
        surface_plot(filtered_dataset_average_dataframe, title='MIMIC-III Mean Aggregated Surface Plot', x_title='Vital', y_title='Hour', background_colour='#ffffff'),
        heatmap(filtered_dataset_average_dataframe, title='MIMIC-III Mean Aggregated Heatmap', background_colour='#ffffff'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fixing Fill Missing Data Filter
    Using KNN imputation we get a very similar shape to the raw dataset.
    """)
    return


@app.cell
def _(PROJECT_ROOT, Path):
    from Managers.serialization_manager import load_data
    TRANSFORMED_EHR_DATA = load_data(Path(str(PROJECT_ROOT / 'Data' / 'mimic-iii' / 'processed_record_ehr.pkl')))
    return (TRANSFORMED_EHR_DATA,)


@app.cell
def _(mo, pd):
    VITALS = {
        'heart rate': [(1, 600), 'bpm'],
        'systolic blood pressure': [(1, 400), 'mmHg'],
        'diastolic blood pressure': [(1, 300), 'mmHg'],
        'mean blood pressure': [(1, 300), 'mmHg'],
        'respiration rate': [(1, 70), 'breaths/min'],
        'temperature': [(21, 50), 'C'],
        'oxygen saturation': [(1, 100), '%']
    }

    mo.ui.table(pd.DataFrame.from_dict(VITALS, orient="index", columns=["range", "unit"]), selection=None)
    return (VITALS,)


@app.cell
def _(np):
    from typing import List, Tuple, OrderedDict, Dict, Optional
    from Entities.ehr_filter_change_tracker import FilterChangeTrackerEHR
    from Entities.ehr_record import RecordEHR
    from copy import deepcopy
    from pandas import DataFrame
    from scipy.interpolate import UnivariateSpline
    from sklearn.impute import KNNImputer

    def fill_missing_data_filter_v2(records: List[RecordEHR], vitals: Dict[str, Tuple[Tuple[int, int], str]]) -> Tuple[List[RecordEHR], List[FilterChangeTrackerEHR]]:
        processed_records = []
        changes = []

        for record in records:
            df_copy = record.timeseries.copy()

            # One column at a time, so a vital is only ever imputed from its own
            # history — never from a different vital.
            for column_name in df_copy.columns:
                column_data = df_copy[column_name].values.astype(float)

                valid_range = vitals[column_name][0]
                known_indices = np.where(~np.isnan(column_data))[0]
                # Only in-range readings count toward the >= 2 guard below — a
                # trace whose real values are all physiologically impossible has
                # nothing trustworthy to impute from.
                known_indices = known_indices[(column_data[known_indices] >= valid_range[0]) & (column_data[known_indices] <= valid_range[1])]

                missing_indices = np.where(np.isnan(column_data))[0]

                # Refuse to impute from a single observation — that just paints
                # one value across the whole day and calls it data.
                if len(missing_indices) > 0 and len(known_indices) >= 2:
                    # KNN rather than the spline this replaced. The key property
                    # is that it interpolates and cannot extrapolate, so it can't
                    # invent the out-of-range values that started this whole
                    # investigation.
                    imputer = KNNImputer(n_neighbors=3)

                    # Single column, so each hour is its own sample and the
                    # "neighbours" are nearby hours.
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

    return (fill_missing_data_filter_v2,)


@app.cell
def _(TRANSFORMED_EHR_DATA, VITALS, fill_missing_data_filter_v2):
    from Managers.dataset_manager import create_filter_dataset

    filtered_mean_dataset_v2 = create_filter_dataset('mimic-iii', TRANSFORMED_EHR_DATA, 'mean', 'Fill Missing Data V2', fill_missing_data_filter_v2, VITALS, False)[1]
    return (filtered_mean_dataset_v2,)


@app.cell
def _(filtered_mean_dataset_v2, heatmap, mo, pd, surface_plot, torch):
    filtered_dataset_v2_average_tensor = torch.stack([filtered_mean_dataset_v2[x][0] for x in range(len(filtered_mean_dataset_v2.data))]).mean(dim=0)
    filtered_dataset_v2_average_dataframe = pd.DataFrame(filtered_dataset_v2_average_tensor.numpy())

    mo.hstack([
        surface_plot(filtered_dataset_v2_average_dataframe, title='MIMIC-III Mean Aggregated Surface Plot', x_title='Vital', y_title='Hour', background_colour='#ffffff'),
        heatmap(filtered_dataset_v2_average_dataframe, title='MIMIC-III Mean Aggregated Heatmap', background_colour='#ffffff'),
    ])
    return


if __name__ == "__main__":
    app.run()
