#!/usr/bin/env bash
set -e

cd /mimic-code/mimic-iv/buildmimic/postgres

psql -U postgres -f create.sql
psql -U postgres -v ON_ERROR_STOP=1 -v mimic_data_dir=/dataset -f load.sql