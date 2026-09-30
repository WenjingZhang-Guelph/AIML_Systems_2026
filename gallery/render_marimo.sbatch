#!/bin/bash
#SBATCH --job-name=ehr_render_marimo
#SBATCH --output=/home/ccampb47/work/clinical_preprocessing_toolbox/gallery/logs/render_marimo_%j.out
#SBATCH --error=/home/ccampb47/work/clinical_preprocessing_toolbox/gallery/logs/render_marimo_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

mkdir -p /home/ccampb47/work/clinical_preprocessing_toolbox/gallery/logs

export MPLBACKEND=Agg

# Use the MIMIC_Extract venv which has matplotlib, scipy, numpy, pandas, marimo
source /home/ccampb47/work/clinical_preprocessing_toolbox/pipelines/MIMIC_Extract/.venv_stats/bin/activate

cd /home/ccampb47/work/clinical_preprocessing_toolbox/gallery

python render_marimo_mimic_iii.py --section "${SECTION:-all}"

echo "Render job complete."
