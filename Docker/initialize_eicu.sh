#!/usr/bin/env bash
set -e

cd /eicu-code/build-db/postgres

make eicu-check datadir=/dataset/
make initialize DBUSER=eicu DBPASS=eicu DBNAME=eicu DBSCHEMA=eicu_crd
make eicu datadir=/dataset/