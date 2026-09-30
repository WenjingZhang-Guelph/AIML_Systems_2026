"""MIMIC-IV version of the MIMIC-III processing script. Same output shape.

Mostly a copy, with the differences you'd expect from the schema change: tables
are split across mimiciv_icu and mimiciv_hosp rather than one flat schema, and
the item-ID lists are far shorter because MIMIC-IV is MetaVision-only — no
CareVue-era duplicates to chase.
"""

import pathlib
import shutil

from Entities.Utilities.database_connection import DatabaseConnection
from Entities.ehr_record import RecordEHR
from Managers.path_manager import get_project_root
import pandas as pd
import time
from tqdm import tqdm
import os
import pickle

# From the MIMIC-IV concepts SQL:
# https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts_postgres/measurement/vitalsign.sql
VITAL_FEATURES = {
    "heart rate": (220045,),
    "systolic blood pressure": (220179, 220050, 225309),
    "diastolic blood pressure": (220180, 220051, 225310),
    "mean blood pressure": (220052, 220181, 225312),
    "respiration rate": (220210, 224690),
    "temperature": (223761, 223762),
    "oxygen saturation": (220277,)
}

DATABASE = DatabaseConnection('postgres', 'mimic-iv.env')
CONNECTION = DATABASE.get_connection()
CURSOR = CONNECTION.cursor()

PROJECT_ROOT = get_project_root()

def fetch_all_timeseries():
    all_item_ids = [i for ids in VITAL_FEATURES.values() for i in ids]
    item_ids_str = ", ".join(str(i) for i in all_item_ids)

    CURSOR.execute(f"""
        SELECT ce.hadm_id, ce.charttime, ce.itemid, ce.valuenum, a.admittime
        FROM mimiciv_icu.chartevents ce
        JOIN mimiciv_hosp.admissions a ON ce.hadm_id = a.hadm_id
        WHERE ce.itemid IN ({item_ids_str})
          AND ce.valuenum IS NOT NULL
    """)
    # No `error` column check here, unlike MIMIC-III — MIMIC-IV dropped that
    # column, so filtering on non-null valuenum is the closest equivalent.
    rows = CURSOR.fetchall()

    dataframe = pd.DataFrame(rows, columns=['hadm_id', 'charttime', 'itemid', 'valuenum', 'admittime'])
    dataframe['charttime'] = pd.to_datetime(dataframe['charttime'])
    dataframe['admittime'] = pd.to_datetime(dataframe['admittime'])

    # Hours run from hospital admission, matching mimic-iii-processing.py.
    # Negative offsets (pre-admission ED charting) are dropped rather than clamped.
    dataframe['hour'] = ((dataframe['charttime'] - dataframe['admittime']).dt.total_seconds() // 3600).astype(int)
    dataframe = dataframe[(dataframe['hour'] >= 0) & (dataframe['hour'] < 24)]

    vital_map = {id_: vital for vital, ids in VITAL_FEATURES.items() for id_ in ids}
    dataframe['vital'] = dataframe['itemid'].map(vital_map)

    # 223761 is the Fahrenheit temperature item, 223762 the Celsius one. Same
    # bimodal-distribution trap as MIMIC-III, just one fewer ID to convert.
    temperature_f_ids = (223761,)
    temp_mask = (dataframe['vital'] == 'temperature') & (dataframe['itemid'].isin(temperature_f_ids))
    dataframe.loc[temp_mask, 'valuenum'] = (dataframe.loc[temp_mask, 'valuenum'] - 32) / 1.8

    return dataframe


def build_vitals_timeseries(dataframe: pd.DataFrame):
    grouped = dataframe.groupby(['hadm_id', 'hour', 'vital'])['valuenum'].agg(list).reset_index()
    timeseries = {}

    for hadm_id, group in grouped.groupby('hadm_id'):
        matrix = group.pivot(index='hour', columns='vital', values='valuenum')
        matrix = matrix.reindex(index=range(24), columns=VITAL_FEATURES.keys(), fill_value=[])
        timeseries[hadm_id] = matrix

    return timeseries


def get_icu_labels():
    CURSOR.execute("SELECT hadm_id, intime, outtime FROM mimiciv_icu.icustays;")
    rows = CURSOR.fetchall()
    icu_data = pd.DataFrame(rows, columns=['hadm_id', 'intime', 'outtime'])
    icu_data['intime'] = pd.to_datetime(icu_data['intime'])
    icu_data['outtime'] = pd.to_datetime(icu_data['outtime'])

    icu_labels = []

    # Aggregate ICU stays per hadm_id
    for hadm_id, group in icu_data.groupby('hadm_id'):
        total_days = ((group['outtime'] - group['intime']).dt.total_seconds() / (24 * 3600)).sum()
        icu_labels.append({'hadm_id': hadm_id, 'icu_label': int(total_days > 3)})

    return pd.DataFrame(icu_labels)


def get_mortality_labels():
    CURSOR.execute("SELECT hadm_id, deathtime FROM mimiciv_hosp.admissions;")
    rows = CURSOR.fetchall()
    dataframe = pd.DataFrame(rows, columns=['hadm_id', 'deathtime'])
    dataframe['deathtime'] = pd.to_datetime(dataframe['deathtime'])
    dataframe['mortality_label'] = dataframe['deathtime'].notna().astype(int)

    return dataframe[['hadm_id', 'mortality_label']]


if __name__ == '__main__':
    CURSOR.execute('SET search_path TO mimiciv_icu;')

    print('Processing MIMIC-IV Data...')

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

    if pathlib.Path(PROJECT_ROOT / 'Data' / 'mimic-iv').exists():
        shutil.rmtree(PROJECT_ROOT / 'Data' / 'mimic-iv')

    pathlib.Path(PROJECT_ROOT / 'Data' / 'mimic-iv' / 'timeseries').mkdir(parents=True, exist_ok=True)

    for hadm_id, ts in timeseries.items():
        ts.to_feather(PROJECT_ROOT / 'Data' / 'mimic-iv' / 'timeseries' / f'{hadm_id}.feather')

    icu_labels.to_feather(PROJECT_ROOT / 'Data' / 'mimic-iv' / 'icu_labels.feather')
    mortality_labels.to_feather(PROJECT_ROOT / 'Data' / 'mimic-iv' / 'mortality_labels.feather')

    valid_stays = set(icu_labels['hadm_id'])
    timeseries = {k: v for k, v in timeseries.items() if k in valid_stays}

    end = time.time()

    print(f'\tTime Elapsed: {end - start}')

    print('Combining timeseries and labels into RecordEHR objects...')

    start = time.time()

    if pathlib.Path.exists(PROJECT_ROOT / 'Data' / 'mimic-iv' / 'processed_record_ehr.pkl'):
        pathlib.Path.unlink(PROJECT_ROOT / 'Data' / 'mimic-iv' / 'processed_record_ehr.pkl')

    # icu_labels.set_index('hadm_id', inplace=True)
    # mortality_labels.set_index('hadm_id', inplace=True)

    records = []
    incomplete_data = []
    for item in tqdm(os.listdir(PROJECT_ROOT / 'Data' / 'mimic-iv' / 'timeseries')):
        admission_id = int(item[:item.find('.')])
        record = pd.read_feather(PROJECT_ROOT / 'Data' / 'mimic-iv' / 'timeseries' / item)

        try:
            icu = icu_labels[icu_labels['hadm_id'] == admission_id].values[0][1]
            mortality = mortality_labels[mortality_labels['hadm_id'] == admission_id].values[0][1]
            records.append(RecordEHR(admission_id, record, icu, mortality))
        except IndexError:
            incomplete_data.append(admission_id)

    end = time.time()

    print(f'\tTime Elapsed: {end - start}')

    print('Saving RecordEHR objects to disk...')

    start = time.time()

    with open(PROJECT_ROOT / 'Data' / 'mimic-iv' / 'processed_record_ehr.pkl', 'wb') as file:
        pickle.dump(records, file)

    end = time.time()

    print(f'\tTime Elapsed: {end - start}')

    print('Example RecordEHR objects:')

    for hadm_id in list(timeseries.keys())[:10]:
        print(timeseries[hadm_id].to_markdown())
        print(f'ICU Label: {icu_labels[icu_labels["hadm_id"] == hadm_id].iloc[0]["icu_label"]}')
        print(f'Mortality Label: {mortality_labels[mortality_labels["hadm_id"] == hadm_id].iloc[0]["mortality_label"]}')


