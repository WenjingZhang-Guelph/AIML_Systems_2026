"""Turns RecordEHR lists into tensor datasets, and caches them on disk.

The caching is the whole point. Building a filtered dataset means aggregating
~50k records and running a filter over each — minutes of work, and the filter
sweep asks for the same combination repeatedly across labels and seeds. So
everything is keyed by (dataset, aggregation method, filter name) under
Data/<dataset>/<aggregation>/ and the create/load/exists trio lets callers skip
the rebuild.

Nothing here invalidates the cache. Change a filter's behaviour or a vital's
range and the stale pickle is still sitting there and will still be used — you
have to delete Data/<dataset>/<aggregation>/ by hand.
"""

import gc
from typing import List, Tuple, Any, Dict
import cupy as cp
from torch.utils.data import Dataset

from Entities.ehr_dataset import DatasetEHR
from Entities.ehr_filter_change_tracker import FilterChangeTrackerEHR
from Entities.ehr_record import RecordEHR
from Managers.ehr_filter_manager import aggregate_filter, combination_filter
from Managers.path_manager import get_project_root

PROJECT_ROOT = get_project_root()

def create_raw_dataset(dataset_name: str, records: List[RecordEHR], aggregation_method: str) -> DatasetEHR:
    """The unfiltered baseline: aggregate, tensorise, cache.

    Still goes through aggregate_filter despite being "raw" — raw means no
    *filter*, not no aggregation. The lists have to collapse to scalars before
    to_tensor() will work at all.
    """
    data_directory = PROJECT_ROOT / 'Data' / dataset_name / aggregation_method
    data_directory.mkdir(parents=True, exist_ok=True)

    raw_dataset = DatasetEHR()
    raw_dataset_path = data_directory / 'raw_dataset_ehr.pkl'

    flattened_records = aggregate_filter(records, aggregation_method)
    flattened_tensor_records = [x.to_tensor() for x in flattened_records]
    raw_dataset.create(flattened_tensor_records)
    raw_dataset.save(str(raw_dataset_path))
    return raw_dataset

def load_raw_dataset(dataset_name: str, aggregation_method: str) -> DatasetEHR:
    # Define paths
    data_directory = PROJECT_ROOT / 'Data' / dataset_name / aggregation_method
    raw_dataset_path = data_directory / 'raw_dataset_ehr.pkl'

    # Create empty dataset
    raw_dataset = DatasetEHR()

    # Populate dataset
    raw_dataset.load(str(raw_dataset_path))
    return raw_dataset

def raw_dataset_exists(dataset_name: str, aggregation_method: str) -> bool:
    data_directory = PROJECT_ROOT / 'Data' / dataset_name / aggregation_method
    raw_dataset_path = data_directory / 'raw_dataset_ehr.pkl'
    if raw_dataset_path.exists():
        return True
    return False

def create_filter_dataset(dataset_name: str, records: List[RecordEHR], aggregation_method: str, filter_name: str, filter_function, vitals: Dict[str, Tuple[Tuple[int, int], str]], pre_aggregate=False):
    """Apply one filter and cache the result, plus a log of what it changed.

    `pre_aggregate` decides which side of the aggregation step the filter runs
    on, and getting it wrong doesn't error — it produces a filter that quietly
    does nothing. The seven per-vital outlier filters need pre_aggregate=True
    (they work on the raw lists); the four structural ones need False (they need
    scalars to test with isna()). See ehr_filter_manager's module docstring.

    Careful with the return value: the third element is always None. It's
    initialised and never assigned from `changes`, so the tracker data only
    survives in the JSON log — read that back with load_filter_dataset if you
    need it.
    """
    data_directory = PROJECT_ROOT / 'Data' / dataset_name / aggregation_method
    log_directory = PROJECT_ROOT / 'Logs' / dataset_name / aggregation_method
    data_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)

    filter_name_clean = filter_name.replace(' ', '_').lower()
    filtered_dataset_path = data_directory / f'{filter_name_clean}_filtered_dataset_ehr.pkl'
    filtered_log_path = log_directory / f'{filter_name_clean}_filter_log.json'

    filtered_dataset = DatasetEHR()
    filtered_changes = None

    if pre_aggregate:
        filtered_records, changes = filter_function(records, vitals)
        filtered_records = aggregate_filter(filtered_records, aggregation_method)
    else:
        filtered_records = aggregate_filter(records, aggregation_method)
        filtered_records, changes = filter_function(filtered_records, vitals)

    FilterChangeTrackerEHR.save_all(changes, str(filtered_log_path))
    tensor_data = [x.to_tensor() for x in filtered_records]
    filtered_dataset.create(tensor_data)
    filtered_dataset.save(str(filtered_dataset_path))

    # Manual gc because the sweep builds these back-to-back and each one holds
    # two full copies of the dataset. Without it, peak RSS climbs across the
    # sweep until the job gets OOM-killed partway through.
    del filtered_records, tensor_data
    gc.collect()

    return filter_name_clean, filtered_dataset, filtered_changes

def create_combination_filter_dataset(dataset_name: str, records: List[RecordEHR], aggregation_method: str, filter_name: str, filter_function, vitals: Dict[str, Tuple[Tuple[int, int], str]]):
    # Ensure directory exists
    data_directory = PROJECT_ROOT / 'Data' / dataset_name / aggregation_method
    log_directory = PROJECT_ROOT / 'Logs' / dataset_name / aggregation_method
    data_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)

    filter_name_clean = filter_name.replace(' ', '_').lower()
    filtered_dataset_path = data_directory / f'{filter_name_clean}_filtered_dataset_ehr.pkl'
    filtered_log_path = log_directory / f'{filter_name_clean}_filter_log.json'

    filtered_dataset = DatasetEHR()

    # `filter_function` is a 2-tuple here, not a callable like everywhere else in
    # this module — the assert is load-bearing documentation as much as a check.
    assert len(filter_function) == 2

    filter1, filter2 = filter_function
    filtered_records, changes = combination_filter(records, vitals, filter1, filter2, aggregation_method)
    filtered_changes = changes

    FilterChangeTrackerEHR.save_all(changes, str(filtered_log_path))
    tensor_data = [x.to_tensor() for x in filtered_records]
    filtered_dataset.create(tensor_data)
    filtered_dataset.save(str(filtered_dataset_path))

    # Explicit cleanup
    del filtered_records, tensor_data
    gc.collect()

    return filter_name_clean, filtered_dataset, filtered_changes

def load_filter_dataset(dataset_name: str, aggregation_method: str, filter_name: str):
    data_directory = PROJECT_ROOT / 'Data' / dataset_name / aggregation_method
    log_directory = PROJECT_ROOT / 'Logs' / dataset_name / aggregation_method

    filter_name_clean = filter_name.replace(' ', '_').lower()
    filtered_dataset_path = data_directory / f'{filter_name_clean}_filtered_dataset_ehr.pkl'
    filtered_log_path = log_directory / f'{filter_name_clean}_filter_log.json'

    filtered_dataset = DatasetEHR()
    filtered_changes = None

    filtered_dataset.load(str(filtered_dataset_path))
    filtered_changes = FilterChangeTrackerEHR.load_all(str(filtered_log_path))

    return filter_name_clean, filtered_dataset, filtered_changes

def filtered_dataset_exists(dataset_name: str, aggregation_method: str, filter_name: str) -> bool:
    data_directory = PROJECT_ROOT / 'Data' / dataset_name / aggregation_method

    filter_name_clean = filter_name.replace(' ', '_').lower()

    filtered_dataset_path = data_directory / f'{filter_name_clean}_filtered_dataset_ehr.pkl'

    return filtered_dataset_path.exists()

def flatten_subset_to_cupy(subset: Dataset, label_index: int) -> Tuple[cp.ndarray, cp.ndarray]:
    """torch Subset → GPU arrays cuML can train on.

    Flattens each 24×7 matrix to a 168-vector, since the random forest has no
    notion of time or of which column is which — it just wants a feature vector.
    Hour-of-day information is lost by doing this, which is fine here and would
    not be for a sequence model.

    Reaches into `subset.dataset.data` directly rather than iterating the Subset,
    because indexing a Subset goes through DatasetEHR.__getitem__ per element and
    that's noticeably slower over ~50k records.
    """
    if len(subset) == 0:
        return cp.array([]), cp.array([])

    data = [subset.dataset.data[i] for i in subset.indices]
    timeseries = cp.stack([cp.asarray(x[0].flatten().numpy()) for x in data])
    labels = cp.array([x[1][label_index].item() for x in data])

    return timeseries, labels
