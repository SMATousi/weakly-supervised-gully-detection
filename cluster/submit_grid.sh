#!/usr/bin/env bash
# usage: cluster/submit_grid.sh <exp> "<pseudo specs, e.g. snorkel_x mv_x:hard>" "<heads>" "<seeds>"
# One job per call; call it once per pseudo-label set to run sets in parallel on whatever GPUs are free.
set -euo pipefail
exp=$1; pseudo=$2; heads=$3; seeds=$4
job="gully-grid-$(echo "$exp" | tr '._' '--' | tr 'A-Z' 'a-z' | cut -c1-20)-$(printf '%s' "$pseudo$heads$seeds" | md5sum | cut -c1-8)"
sed -e "s/__JOB__/$job/" -e "s|__PSEUDO__|$pseudo|" -e "s|__HEADS__|$heads|" -e "s|__SEEDS__|$seeds|" -e "s|__EXP__|$exp|" \
    "$(dirname "$0")/train_grid_job.yaml" | kubectl apply -f -
echo "$job"
