from pathlib import Path
from typing import Optional, Any
import pandas as pd
import pickle

from Managers.path_manager import get_project_root

PROJECT_ROOT = get_project_root()

def load_data(file_path: str) -> Any:
    """Read a .pkl or .feather, picking the reader off the extension.

    Uses raw pickle.load rather than pd.read_pickle because these files hold
    lists of RecordEHR objects, not DataFrames, and pandas' wrapper is fussier
    about what it will unpickle.

    Unknown extensions return None instead of raising, which is a trap — a typo
    in a filename gets you a silent None rather than an error.
    """
    file_path = Path(file_path)

    if file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    elif file_path.suffix == '.feather':
        return pd.read_feather(file_path)

    else:
        return None


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Counterpart to load_data. Also silently no-ops on an unknown extension."""
    file_path = Path(file_path)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.suffix == '.pkl':
        pd.to_pickle(df, file_path)

    # Feather can't store the list-per-cell timeseries — use .pkl for those.
    elif file_path.suffix == '.feather':
        df.to_feather(file_path)

def save_object(obj: Any, file_path: str) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), 'wb') as f:
        pickle.dump(obj, f)

def load_object(file_path: str) -> Any:
    path = Path(file_path)
    if not path.exists():
        return None
    with path.open('rb') as f:
        return pickle.load(f)