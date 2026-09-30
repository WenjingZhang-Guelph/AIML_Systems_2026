"""Build and cache every filtered dataset for one (dataset, aggregation) pair.

    python Experiments/apply_dataset_filter.py mimic-iii mean

Run this once per aggregation method before the analysis notebooks — they expect
the caches under Data/<dataset>/<aggregation>/ to already exist. Splitting it out
as a script rather than doing it in the notebook means the expensive part can go
through a job scheduler and be resumed: anything already cached is skipped.
"""
import sys

from tqdm import tqdm
from Managers.dataset_manager import create_raw_dataset, load_raw_dataset, raw_dataset_exists, create_filter_dataset
from Managers.dataset_manager import filtered_dataset_exists, load_filter_dataset
from Managers.ehr_filter_manager import FILTERS
from Managers.path_manager import get_project_root
from Managers.serialization_manager import load_data

PROJECT_ROOT = get_project_root()
DATASET_NAME = sys.argv[1]
aggregation_method = sys.argv[2]

# Hard physiological ranges for the seven vitals, keyed by the repo-wide lowercase
# names. `_remove_outliers` tests these as a CLOSED interval (`lo <= v <= hi`), so
# the upper bound is a value the filter keeps, not the first one it rejects. 

VITALS = {
    'heart rate': [(1, 600), 'bpm'],
    'systolic blood pressure': [(1, 400), 'mmHg'],
    'diastolic blood pressure': [(1, 300), 'mmHg'],
    'mean blood pressure': [(1, 300), 'mmHg'],
    'respiration rate': [(1, 70), 'breaths/min'],
    'temperature': [(21, 50), 'C'],
    'oxygen saturation': [(1, 100), '%']
}

PROCESSED_RECORD_EHR = load_data(str(PROJECT_ROOT / 'Data' / DATASET_NAME / 'processed_record_ehr.pkl'))

records = None

RAW_DATASETS = dict()

if raw_dataset_exists(DATASET_NAME, aggregation_method):
    RAW_DATASETS[aggregation_method] = load_raw_dataset(DATASET_NAME, aggregation_method)
else:
    RAW_DATASETS[aggregation_method] = create_raw_dataset(DATASET_NAME, PROCESSED_RECORD_EHR, aggregation_method)

FILTERED_DATASETS = dict()

k = 0
for filter_name, filter_function in tqdm(FILTERS.items(), desc=f'Applying filters using {aggregation_method}'):
    print(aggregation_method, filter_name, filter_function)
    if aggregation_method not in FILTERED_DATASETS:
        FILTERED_DATASETS[aggregation_method] = dict()

    # The first seven entries in FILTERS are the per-vital outlier filters, which
    # need the raw list-per-cell data and so must run before aggregation. The
    # remaining four are structural and need scalars. This is why FILTERS being
    # an OrderedDict matters — the boundary is a position, not a property of the
    # filter, so reordering FILTERS silently sends filters down the wrong path
    # (where they don't crash, they just stop doing anything).
    pre_aggregate = True if k < 7 else False

    if filtered_dataset_exists(DATASET_NAME, aggregation_method, filter_name):
        FILTERED_DATASETS[aggregation_method][filter_name] = load_filter_dataset(DATASET_NAME, aggregation_method, filter_name)
    else:
        FILTERED_DATASETS[aggregation_method][filter_name] = create_filter_dataset(DATASET_NAME, PROCESSED_RECORD_EHR, aggregation_method, filter_name, filter_function, VITALS, pre_aggregate)

    k += 1