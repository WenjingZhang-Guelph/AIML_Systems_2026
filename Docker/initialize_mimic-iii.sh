#!/usr/bin/env bash
set -e

cd /mimic-code/mimic-iii/buildmimic/postgres

make create-user mimic datadir=/dataset
make mimic-gz datadir="/dataet"

#psql -U postgres -d mimic -c "SET search_path TO mimiciii;"