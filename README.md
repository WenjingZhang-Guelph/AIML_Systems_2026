<p align="center">
  <img src="Assets/Images/icon.png" alt="Icon" WIDTH="256" height="256">
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-090D11?style=for-the-badge&logo=python&logoColor=38b178" alt="Python" style="border: none; outline: none;">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/pytorch-090D11?style=for-the-badge&logo=pytorch&logoColor=38b178" alt="PyTorch" style="border: none; outline: none;">
  </a>
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/scikit--learn-090D11?style=for-the-badge&logo=scikit-learn&logoColor=38b178" alt="scikit-learn" style="border: none; outline: none;">
  </a>
  <a href="https://www.docker.com/">
    <img src="https://img.shields.io/badge/docker-090D11?style=for-the-badge&logo=docker&logoColor=38b178" alt="Docker" style="border: none; outline: none;">
  </a>
</p>

<p align="center">
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code_style-black-38b178.svg?style=for-the-badge" alt="black" style="width: auto; height: 20px;">
  </a>
</p>


<h1 align="center">Electronic Health Record (EHR) Dataset Processing</h1>

<p>
Exploring EHR datasets and the impact of preprocessing in the context of synthetic dataset generation.
</p>

## What this actually does

Pulls seven bedside vitals — heart rate, systolic / diastolic / mean blood pressure, respiration rate, temperature, oxygen saturation — for the first 24 hours of each ICU stay, then asks a specific question: **does cleaning the data actually help the model, or does it just change the population?**

That's why the whole pipeline is built around applying filters one at a time and measuring the result. Eleven filters, five ways of aggregating each hour, two prediction targets (ICU stay > 3 days, in-hospital mortality). Every combination gets a random forest and a McNemar test against the unfiltered baseline. A filter that improves accuracy while shifting the class centroids a long way is suspicious — it may have thrown out the hard cases rather than the noise, and the centroid plots exist to catch exactly that.

Supports MIMIC-III, MIMIC-IV and eICU. They are not directly comparable: eICU keys on ICU stay while the MIMIC scripts key on hospital admission, and their 24-hour windows start from different events. See the module docstrings in `Processing/`.

## Getting oriented

Read in this order if you're new to the code:

| File | Why |
|---|---|
| `Processing/mimic-iii-processing.py` | Where the data comes from, and every cohort/label decision baked into it |
| `Entities/ehr_record.py` | The core data structure — read the docstring, the list-per-cell layout surprises everyone |
| `Managers/ehr_filter_manager.py` | The eleven filters and the pre/post-aggregation split that governs them |
| `Managers/evaluation_manager.py` | How anything gets scored |
| `Experiments/notebook.py` | The Marimo notebook that ties it together |

## Running it

Two separate environments, and neither is optional. The database containers:

```bash
docker build -t mimic-iii-postgres -f Docker/Dockerfile-mimic-iii Docker/
docker run --name Mimic-III -p 5432:5432 -v /path/to/mimiciii:/dataset \
    -e POSTGRES_PASSWORD=postgres --shm-size=8g -d mimic-iii-postgres
```

Each dataset needs its own container on its own port, and credentials in `Secrets/<dataset>.env`. `eicu.env` is checked in; the two MIMIC ones you create yourself, same four keys.

The compute environment is CUDA — cuDF, cuML and PyTorch do the heavy lifting, and `requirements.txt` only covers the CPU-side database and dataframe dependencies. There is no pip-only path that gives you a working environment; use `Docker/Dockerfile` or replicate what it installs.

Then, per dataset:

```bash
python Processing/mimic-iii-processing.py          # DB → Data/mimic-iii/processed_record_ehr.pkl
python Experiments/apply_dataset_filter.py mimic-iii mean   # build + cache filtered datasets
marimo edit Experiments/notebook.py                # the analysis
```

Repeat `apply_dataset_filter.py` for each aggregation method you want. It caches under `Data/<dataset>/<aggregation>/` and skips anything already built, so it's safe to re-run and safe to interrupt.


