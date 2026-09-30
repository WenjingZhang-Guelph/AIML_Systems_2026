"""Same job as the MIMIC scripts, but eICU's schema makes it much less work.

Two differences that matter if you're comparing datasets rather than just
running this:

eICU keys everything on patientunitstayid — an ICU stay, not a hospital
admission. So a record here is not the same unit of analysis as a MIMIC-III
record, even though both end up as a RecordEHR with an `admission_id`.

And `hour` counts from *unit* admission, because that's what eICU's offset
columns are relative to. MIMIC-III counts from hospital admission. Neither is
wrong, but the two pipelines' "first 24 hours" cover different clinical windows,
which is worth remembering before reading anything into a cross-dataset
comparison.
"""

import os
import pathlib
import pickle
import shutil

import pandas as pd
import time

from tqdm import tqdm

from Entities.Utilities.database_connection import DatabaseConnection
from Entities.ehr_record import RecordEHR
from Managers.path_manager import get_project_root

# vitalperiodic stores each vital in its own column, so no item-ID lookup table
# to maintain — just a rename. Nice change from MIMIC.
VITAL_FEATURES = {
    'heart rate': 'heartrate',
    'systolic blood pressure': 'systemicsystolic',
    'diastolic blood pressure': 'systemicdiastolic',
    'mean blood pressure': 'systemicmean',
    'respiration rate': 'respiration',
    'temperature': 'temperature',
    'oxygen saturation': 'sao2'
}

DATABASE = DatabaseConnection('eicu', 'eicu.env')
CONNECTION = DATABASE.get_connection()
CURSOR = CONNECTION.cursor()

PROJECT_ROOT = get_project_root()

def fetch_all_timeseries():
    CURSOR.execute("""
    SELECT vp.patientunitstayid, observationoffset, 
           heartrate, systemicsystolic, systemicdiastolic,
           systemicmean, respiration, temperature, sao2 
    FROM vitalperiodic vp JOIN patient p ON vp.patientunitstayid = p.patientunitstayid;

    """)
    rows = CURSOR.fetchall()

    dataframe = pd.DataFrame(rows, columns=[
        'patientunitstayid', 'observationoffset', 'heartrate',
        'systemicsystolic', 'systemicdiastolic', 'systemicmean',
        'respiration', 'temperature', 'sao2'
    ])

    # observationoffset is "minutes from unit admit time"
    # (https://eicu-crd.mit.edu/eicutables/vitalperiodic/), so integer-dividing
    # by 60 gives the hour directly. No windowing here — anything past hour 23
    # gets dropped later by the reindex in build_vitals_timeseries.
    dataframe['hour'] = (dataframe['observationoffset'] // 60).astype(int)

    # Wide-to-long so the rest of this file matches the MIMIC scripts' shape,
    # which is what lets build_vitals_timeseries be near-identical across all three.
    dataframe = dataframe.melt(
        id_vars=['patientunitstayid', 'hour'],
        var_name='vital',
        value_name='value'
    )

    dataframe['vital'] = dataframe['vital'].map({v: k for k, v in VITAL_FEATURES.items()})

    return dataframe

def build_vitals_timeseries(dataframe: pd.DataFrame):
    grouped = dataframe.groupby(['patientunitstayid', 'hour', 'vital'])['value'].agg(list).reset_index()

    timeseries = {}
    for patientunitstayid, group in grouped.groupby('patientunitstayid'):
        matrix = group.pivot(index='hour', columns='vital', values='value')
        matrix = matrix.reindex(index=range(24), columns=VITAL_FEATURES.keys(), fill_value=[])
        timeseries[patientunitstayid] = matrix

    return timeseries

def get_admission_ids():
    CURSOR.execute("SELECT DISTINCT (patientunitstayid) FROM patient;")
    return [row[0] for row in CURSOR.fetchall()]

def get_icu_labels():
    CURSOR.execute("SELECT DISTINCT(patientunitstayid) FROM patient WHERE unitdischargeoffset > 3 * 24 * 60;")
    rows = CURSOR.fetchall()

    all_patients = set(get_admission_ids())
    icu_patients = set(row[0] for row in rows)

    dataframe = pd.DataFrame({
        'patientunitstayid': list(all_patients),
        'icu_label': [1 if pid in icu_patients else 0 for pid in all_patients]
    })

    return dataframe


def get_mortality_labels():
    CURSOR.execute("SELECT DISTINCT(patientunitstayid) FROM patient WHERE hospitaldischargestatus = 'Expired';")
    rows = CURSOR.fetchall()

    all_patients = set(get_admission_ids())
    mortality_patients = set(row[0] for row in rows)

    dataframe = pd.DataFrame({
        'patientunitstayid': list(all_patients),
        'mortality_label': [1 if pid in mortality_patients else 0 for pid in all_patients]
    })

    return dataframe


if __name__ == '__main__':
    CURSOR.execute('SET SEARCH_PATH TO eicu_crd;')

    print('Processing eICU data...')

    start = time.time()

    timeseries_grouped = fetch_all_timeseries()
    timeseries = build_vitals_timeseries(timeseries_grouped)

    icu_labels = get_icu_labels()
    mortality_labels = get_mortality_labels()

    end = time.time()

    print(f'\tTime Elapsed: {end - start}')

    print('Saving processed data to disk...')

    start = time.time()

    pathlib.Path(PROJECT_ROOT / 'Data').mkdir(exist_ok=True)

    if pathlib.Path(PROJECT_ROOT / 'Data' / 'eicu').exists():
        shutil.rmtree(PROJECT_ROOT / 'Data' / 'eicu')

    pathlib.Path(PROJECT_ROOT / 'Data' / 'eicu' / 'timeseries').mkdir(parents=True, exist_ok=True)

    for patientunitstayid, ts in timeseries.items():
        ts.to_feather(PROJECT_ROOT / 'Data' / 'eicu' / 'timeseries' / f'{patientunitstayid}.feather')

    icu_labels.to_feather(PROJECT_ROOT / 'Data' / 'eicu' / 'icu_labels.feather')
    mortality_labels.to_feather(PROJECT_ROOT / 'Data' / 'eicu' / 'mortality_labels.feather')

    end = time.time()

    print(f'\tTime Elapsed: {end - start}')

    print('Combining timeseries and labels into RecordEHR objects...')

    start = time.time()

    if pathlib.Path.exists(PROJECT_ROOT / 'Data' / 'eicu' / 'processed_record_ehr.pkl'):
        pathlib.Path.unlink(PROJECT_ROOT / 'Data' / 'eicu' / 'processed_record_ehr.pkl')

    # icu_labels.set_index('patientunitstayid', inplace=True)
    # mortality_labels.set_index('patientunitstayid', inplace=True)

    records = []
    incomplete_data = []
    for item in tqdm(os.listdir(PROJECT_ROOT / 'Data' / 'eicu' / 'timeseries')):
        admission_id = int(item[:item.find('.')])
        record = pd.read_feather(PROJECT_ROOT / 'Data' / 'eicu' / 'timeseries' / item)

        try:
            icu = icu_labels[icu_labels['patientunitstayid'] == admission_id].values[0][1]
            mortality = mortality_labels[mortality_labels['patientunitstayid'] == admission_id].values[0][1]
            records.append(RecordEHR(admission_id, record, icu, mortality))
        except IndexError:
            incomplete_data.append(admission_id)

    end = time.time()

    print(f'\tTime Elapsed: {end - start}')

    print('Saving RecordEHR objects to disk...')

    start = time.time()

    with open(PROJECT_ROOT / 'Data' / 'eicu' / 'processed_record_ehr.pkl', 'wb') as file:
        pickle.dump(records, file)

    end = time.time()

    print(f'\tTime Elapsed: {end - start}')

    print('Example RecordEHR objects:')

    for patientunitstayid in list(timeseries.keys())[:10]:
        print(timeseries[patientunitstayid].to_markdown())
        print(f'ICU Label: {icu_labels[icu_labels["patientunitstayid"] == patientunitstayid].iloc[0]["icu_label"]}')
        print(
            f'Mortality Label: {mortality_labels[mortality_labels["patientunitstayid"] == patientunitstayid].iloc[0]["mortality_label"]}')

