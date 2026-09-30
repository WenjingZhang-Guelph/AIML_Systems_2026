"""Pull the seven bedside vitals for every MIMIC-III admission's first 24 hours.

Run this once per dataset; it writes Data/mimic-iii/processed_record_ehr.pkl,
which everything downstream reads. Needs a live Postgres with MIMIC-III loaded
(see Docker/initialize_mimic-iii.sh) and creds in Secrets/mimic-iii.env.

Takes a while and holds the full chartevents result set in memory — this is not
a login-node job.
"""

import pathlib
import time
from Entities.Utilities.database_connection import DatabaseConnection
import pandas as pd
import shutil

from Entities.ehr_record import RecordEHR
from Managers.path_manager import get_project_root
from tqdm import tqdm
import os
import pickle

# Item IDs lifted from the official MIMIC-III concepts SQL, so our definition of
# "heart rate" matches everybody else's:
# https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts_postgres/firstday/vitals_first_day.sql
#
# Each vital has several IDs because MIMIC-III spans two chart systems
# (CareVue's low numbers, MetaVision's 22xxxx range) and the same measurement
# got a different ID in each. Miss one and you silently lose a chunk of the cohort.
VITAL_FEATURES = {
    "heart rate": (211, 220045),
    "systolic blood pressure": (51, 442, 455, 6701, 220179, 220050),
    "diastolic blood pressure": (8368, 8440, 8441, 8555, 220180, 220051),
    "mean blood pressure": (456, 52, 6702, 443, 220052, 220181, 225312),
    "respiration rate": (615, 618, 220210, 224690),
    "temperature": (223761, 678, 223762, 676),
    "oxygen saturation": (646, 220277)
}

DATABASE = DatabaseConnection('mimic', 'mimic-iii.env')
CONNECTION = DATABASE.get_connection()
CURSOR = CONNECTION.cursor()

PROJECT_ROOT = get_project_root()

def fetch_all_timeseries():
    """One big query for every vital across every admission, binned to hour 0-23.

    Deliberately a single query rather than per-patient: chartevents is ~330M
    rows, and round-tripping per admission would take days.
    """
    all_item_ids = [i for ids in VITAL_FEATURES.values() for i in ids]
    item_ids_str = ", ".join(str(i) for i in all_item_ids)

    CURSOR.execute(f"""
        SELECT ce.hadm_id, ce.charttime, ce.itemid, ce.valuenum, a.admittime
        FROM chartevents ce
        JOIN admissions a ON ce.hadm_id = a.hadm_id
        WHERE ce.itemid IN ({item_ids_str})
          AND (ce.error IS NULL OR ce.error = 0)
    """)
    # `error IS NULL OR error = 0` rather than just `= 0` — the column is only
    # populated for MetaVision rows, so testing equality alone throws away every
    # CareVue measurement.
    rows = CURSOR.fetchall()

    dataframe = pd.DataFrame(rows, columns=['hadm_id', 'charttime', 'itemid', 'valuenum', 'admittime'])
    dataframe['charttime'] = pd.to_datetime(dataframe['charttime'])
    dataframe['admittime'] = pd.to_datetime(dataframe['admittime'])

    # Hours are measured from hospital admission, not ICU admission — worth
    # flagging because MIMIC_Extract anchors on ICU intime instead, so the two
    # pipelines' "hour 0" are not the same moment for a patient who spent time on
    # a ward first.
    #
    # Negative hours are real: charttime occasionally predates admittime, usually
    # ED measurements backfilled after the fact. Dropped rather than clamped to 0,
    # since we can't tell how long before admission they were taken.
    dataframe['hour'] = ((dataframe['charttime'] - dataframe['admittime']).dt.total_seconds() // 3600).astype(int)
    dataframe = dataframe[(dataframe['hour'] >= 0) & (dataframe['hour'] < 24)]

    vital_map = {id_: vital for vital, ids in VITAL_FEATURES.items() for id_ in ids}
    dataframe['vital'] = dataframe['itemid'].map(vital_map)

    # Temperature is the one vital MIMIC charts in two units — these two item IDs
    # are Fahrenheit, the other two Celsius, and nothing in the data marks which
    # is which. Without this conversion you get a bimodal temperature
    # distribution with peaks around 37 and 98, and every downstream range filter
    # behaves bizarrely.
    temperature_f_ids = (223761, 678)
    temperature_f_mask = (dataframe['vital'] == 'temperature') & (dataframe['itemid'].isin(temperature_f_ids))
    dataframe.loc[temperature_f_mask, 'valuenum'] = (dataframe.loc[temperature_f_mask, 'valuenum'] - 32) / 1.8

    return dataframe


def build_vitals_timeseries(dataframe: pd.DataFrame):
    """Reshape the flat query result into one 24×7 frame per admission.

    `.agg(list)` is the important bit and it's intentional — every measurement in
    an hour is kept as a list rather than averaged down to a number. Collapsing
    here would destroy exactly the within-hour variation the outlier filters need
    to look at, and it would hardcode a choice of summary statistic that's meant
    to be swappable later. See RecordEHR's docstring for what that means for
    consumers.
    """
    grouped = dataframe.groupby(['hadm_id', 'hour', 'vital'])['valuenum'].agg(list).reset_index()
    timeseries = {}

    for hadm_id, group in grouped.groupby('hadm_id'):
        matrix = group.pivot(index='hour', columns='vital', values='valuenum')
        # reindex forces all 24 hours and all 7 columns to exist even when a
        # patient has no data for them, so every record comes out the same shape
        # and the tensors stack. Empty cells are [] rather than NaN to stay
        # consistent with the populated ones.
        matrix = matrix.reindex(index=range(24), columns=VITAL_FEATURES.keys(), fill_value=[])
        timeseries[hadm_id] = matrix

    return timeseries


def get_admission_ids():
    CURSOR.execute("SELECT DISTINCT hadm_id FROM admissions ORDER BY hadm_id;")
    return [row[0] for row in CURSOR.fetchall()]

def get_icu_labels():
    """Label an admission 1 if its total ICU time exceeds three days.

    Two decisions baked in here that matter for any comparison against
    MIMIC_Extract:

    Overlapping and adjacent ICU stays get merged before summing, so a patient
    bounced between units doesn't get double-counted. That's what the sorted
    walk with current_start/current_end is doing.

    The threshold is 3 days, on *summed* time across all stays — not longest
    single stay, and not the ICU-LOS definition MIMIC_Extract uses. Don't assume
    the two label columns mean the same thing.
    """
    CURSOR.execute("SELECT hadm_id, intime, outtime FROM icustays;")
    rows = CURSOR.fetchall()
    icu_data = pd.DataFrame(rows, columns=['hadm_id', 'intime', 'outtime'])
    icu_data['intime'] = pd.to_datetime(icu_data['intime'])
    icu_data['outtime'] = pd.to_datetime(icu_data['outtime'])

    icu_labels = []

    for hadm_id, group in icu_data.groupby('hadm_id'):
        group_sorted = group.sort_values('intime').reset_index(drop=True)
        total_days = 0
        current_start, current_end = group_sorted.loc[0, ['intime', 'outtime']]
        for i in range(1, len(group_sorted)):
            start, end = group_sorted.loc[i, ['intime', 'outtime']]
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                total_days += (current_end - current_start).total_seconds() / (24*3600)
                current_start, current_end = start, end
        total_days += (current_end - current_start).total_seconds() / (24*3600)

        icu_labels.append({'hadm_id': hadm_id, 'icu_label': int(total_days > 3)})

    return pd.DataFrame(icu_labels)

def get_mortality_labels():
    """In-hospital mortality, straight off whether deathtime is populated.

    `deathtime` in admissions is only set for deaths that happened during that
    admission, so this is in-hospital mortality — not 30-day, not
    ICU-specific. A patient who died a week after discharge counts as 0.
    """
    CURSOR.execute("SELECT hadm_id, deathtime FROM admissions;")
    rows = CURSOR.fetchall()
    dataframe = pd.DataFrame(rows, columns=['hadm_id', 'deathtime'])
    dataframe['deathtime'] = pd.to_datetime(dataframe['deathtime'])

    dataframe['mortality_label'] = dataframe['deathtime'].notna().astype(int)
    return dataframe[['hadm_id', 'mortality_label']]

if __name__ == '__main__':
    # Everything lives in the mimiciii schema, not public — without this every
    # query fails on "relation does not exist".
    CURSOR.execute('SET search_path TO mimiciii;')

    print('Processing MIMIC-III data...')

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

    # Wiping the whole dataset directory rather than overwriting, so a re-run
    # after changing the vital set can't leave stale per-admission feather files
    # behind for admissions that no longer qualify. Note this also blows away the
    # cached aggregations and centroids under Data/mimic-iii/<aggregation>/,
    # which are expensive to regenerate.
    if pathlib.Path(PROJECT_ROOT / 'Data' / 'mimic-iii').exists():
        shutil.rmtree(PROJECT_ROOT / 'Data' / 'mimic-iii')

    pathlib.Path(PROJECT_ROOT / 'Data' / 'mimic-iii' / 'timeseries').mkdir(parents=True, exist_ok=True)

    for hadm_id, ts in timeseries.items():
        ts.to_feather(PROJECT_ROOT / 'Data' / 'mimic-iii' / 'timeseries' / f'{hadm_id}.feather')

    icu_labels.to_feather(PROJECT_ROOT / 'Data' / 'mimic-iii' / 'icu_labels.feather')
    mortality_labels.to_feather(PROJECT_ROOT / 'Data' / 'mimic-iii' / 'mortality_labels.feather')

    end = time.time()

    print(f'\tTime Elapsed: {end - start}')

    print('Combining timeseries and labels into RecordEHR objects...')

    start = time.time()

    if pathlib.Path.exists(PROJECT_ROOT / 'Data' / 'mimic-iii' / 'processed_record_ehr.pkl'):
        pathlib.Path.unlink(PROJECT_ROOT / 'Data' / 'mimic-iii' / 'processed_record_ehr.pkl')

    # Reading the feathers back off disk instead of using the in-memory dict is
    # intentional — it means a crash during the save step doesn't force a full
    # re-query, and it verifies the round-trip actually worked.
    #
    # An admission with vitals but no matching label row (IndexError below) is
    # dropped entirely. Those are admissions absent from icustays or
    # admissions, and there's no sensible label to invent for them.
    records = []
    incomplete_data = []
    for item in tqdm(os.listdir(PROJECT_ROOT / 'Data' / 'mimic-iii' / 'timeseries')):
        admission_id = int(item[:item.find('.')])
        record = pd.read_feather(PROJECT_ROOT / 'Data' / 'mimic-iii' / 'timeseries' / item)

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

    with open(PROJECT_ROOT / 'Data' / 'mimic-iii' / 'processed_record_ehr.pkl', 'wb') as file:
        pickle.dump(records, file)

    end = time.time()

    print(f'\tTime Elapsed: {end - start}')

    print('Example RecordEHR objects:')

    for hadm_id in list(timeseries.keys())[:10]:
        print(timeseries[hadm_id].to_markdown())
        print(f'ICU Label: {icu_labels[icu_labels["hadm_id"] == hadm_id].iloc[0]["icu_label"]}')
        print(f'Mortality Label: {mortality_labels[mortality_labels["hadm_id"] == hadm_id].iloc[0]["mortality_label"]}')




