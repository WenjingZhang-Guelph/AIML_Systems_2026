"""Single source of truth for where the project lives on disk.

Everything else builds paths off get_project_root() rather than hardcoding or
using cwd, so scripts behave the same whether they're run from the repo root, a
notebook, or a SLURM job with an arbitrary working directory.

Derived from this file's own location, which means moving path_manager.py to a
different directory depth silently repoints the whole project.
"""
from pathlib import Path

PROJECT_ROOT = None

def get_project_root() -> Path:
    global PROJECT_ROOT

    if PROJECT_ROOT is None:
        PROJECT_ROOT = Path(__file__).resolve().parent.parent

    return PROJECT_ROOT

if __name__ == '__main__':
    print(get_project_root())