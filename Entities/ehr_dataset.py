from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from Managers.path_manager import get_project_root
import os

class DatasetEHR:
    name: str
    label_index_map: Dict[str, int]
    training_set: Dataset
    validation_set: Dataset
    testing_set: Dataset
    split_weights: Tuple[float, float, float]
    data: List[Tuple[torch.Tensor, torch.Tensor]]
    # Parallel to `data`, or None when the sidecar was never loaded. See
    # load_admission_ids — everything that pairs arms by patient needs this.
    admission_ids: Optional[np.ndarray]


    def __init__(self):
        self.name = 'ehr_dataset'
        # Mirrors the tensor layout in RecordEHR.to_tensor — keep them in sync.
        self.label_index_map = {'icu': 0, 'mortality': 1}
        self.training_set = Dataset()
        self.validation_set = Dataset()
        self.testing_set = Dataset()
        self.split_weights = (0, 0, 0)
        self.data = []
        self.admission_ids = None
        # 24 hours × 7 vitals. Fixed by construction upstream, stored here so
        # model code doesn't have to reach into a sample to find the shape.
        self.num_rows = 24
        self.num_columns = 7

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Two calling conventions. `ds[i]` is the normal torch one and returns
        # both labels; `ds[i, 0]` picks a single label out, which is what the
        # single-target classifiers want so they don't have to slice it
        # themselves on every batch. Use label_index_map rather than a bare 0/1.
        if isinstance(index, tuple) and len(index) == 2:
            idx, label_index = index
            timeseries, labels = self.data[idx]
            return timeseries, labels[label_index]
        else:
            return self.data[index]

    def create(self, data: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.data = data

    def split(self, training_weight, validation_weight, testing_weight) -> Tuple[Dataset, Dataset, Dataset]:
        """Three-way random split, cached so repeat calls give the same subsets.

        The caching is the point — asking for the same weights twice returns the
        identical split rather than reshuffling, so comparing two models over one
        dataset object is actually a fair comparison. Ask for *different* weights
        and you get a fresh shuffle, which quietly invalidates anything you'd
        already measured.

        Two caveats worth knowing. The split is unseeded, so results aren't
        reproducible across runs unless you seed torch yourself. And it's not
        stratified — mortality prevalence here is under 15%, so a small subset can
        end up with barely any positives.
        """
        assert abs(training_weight + validation_weight + testing_weight - 1.0) < 1e-6, (
            "Training, validation, and testing weights for splitting dataset must sum to 1.0"
        )

        if self.split_weights != (training_weight, validation_weight, testing_weight):
            indices = torch.randperm(len(self)).tolist()

            total_samples = len(self)
            num_training_samples = int(training_weight * total_samples)
            num_validation_samples = int(validation_weight * total_samples)

            # Testing takes whatever's left rather than its own int() slice, so
            # the rounding remainder lands there instead of being dropped.
            training_indices = indices[:num_training_samples]
            validation_indices = indices[num_training_samples:num_training_samples + num_validation_samples]
            testing_indices = indices[num_training_samples + num_validation_samples:]

            training_set = torch.utils.data.Subset(self, training_indices)
            validation_set = torch.utils.data.Subset(self, validation_indices)
            testing_set = torch.utils.data.Subset(self, testing_indices)

            self.training_set = training_set
            self.validation_set = validation_set
            self.testing_set = testing_set

            self.split_weights = (training_weight, validation_weight, testing_weight)

        return self.training_set, self.validation_set, self.testing_set

    def load(self, file_path: str) -> None:
        # Only self.data round-trips — the splits don't, so you have to call
        # split() again after loading and you'll get a different shuffle.
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file '{file_path}' not found.")
        self.data = torch.load(path)

    def save(self, file_path: str) -> None:
        torch.save(self.data, file_path)

    def load_admission_ids(self, file_path: str) -> None:
        """Attach the record keys written by rerun/regen_admission_ids.py.

        Kept out of the dataset pickle on purpose. `save` writes a bare
        `list[(Tensor, Tensor)]` via torch.save, and torch 2.12 loads with
        weights_only=True by default — adding a non-tensor field would break
        every existing reader of every cached pickle. So the ids live in a
        sibling .npy and are opt-in.

        Populating this is what makes an arm's predictions joinable against
        another arm's; without it a record's only identity is its list position,
        which differs between arms because the filters drop different records.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Admission id file '{file_path}' not found — run "
                "rerun/regen_admission_ids.py for this aggregation first."
            )
        ids = np.load(path)
        if len(ids) != len(self.data):
            raise ValueError(
                f"'{file_path}' holds {len(ids):,} ids but the dataset has "
                f"{len(self.data):,} records; they are not the same arm."
            )
        self.admission_ids = ids