import pandas as pd
import numpy as np
import torch

class RecordEHR:
    """One hospital admission's first 24 hours.

    The thing that trips everyone up: `timeseries` is a 24×7 DataFrame (hour ×
    vital) whose cells hold *lists*, not floats. A patient might get charted six
    heart rates and zero temperatures in a given hour, and the processing scripts
    keep all of them rather than collapsing early — the filters need the raw
    values, and which summary statistic to use is a decision made later by the
    aggregation step. Empty cells are `[]`, not NaN.

    Which means `to_tensor()` only works on an already-aggregated record. Call it
    on a freshly-processed one and pandas will choke trying to cast lists to float.
    """
    admission_id: int
    timeseries: pd.DataFrame
    icu: int
    mortality: int

    def __init__(self, admission_id: int, timeseries: pd.DataFrame, icu: int, mortality: int):
        self.admission_id = admission_id
        self.timeseries = timeseries
        self.icu = icu
        self.mortality = mortality

    def __hash__(self):
        # hadm_id is already unique per admission, so it's a free hash. Note this
        # gives no __eq__, so two records for the same admission hash alike but
        # don't compare equal — fine for the set/dict-keying we actually do.
        return self.admission_id

    def to_tensor(self):
        # Zero-filling missing values is a real modelling choice, not just
        # defensive coding: it means "no measurement" and "measured as 0" become
        # indistinguishable to the model. Defensible here only because the
        # vitals are all strictly positive in practice, so 0 is out-of-range
        # enough to act as its own signal.
        timeseries_array = self.timeseries.to_numpy(dtype=float)
        timeseries_array = np.nan_to_num(timeseries_array, nan=0.0)
        timeseries_tensor = torch.tensor(timeseries_array, dtype=torch.float32)
        # [icu, mortality] in that order, everywhere in the codebase — the
        # evaluation code indexes labels positionally, so swapping them silently
        # trains against the wrong target.
        label_tensor = torch.tensor([self.icu, self.mortality], dtype=torch.long)
        return timeseries_tensor, label_tensor