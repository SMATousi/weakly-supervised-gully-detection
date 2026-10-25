#!/usr/bin/env bash
# usage: cluster/submit_robust.sh <pseudo> "<modes>" "<heads>" "<seeds>" ["<extra run_robust args>"]
set -euo pipefail
pseudo=$1; modes=$2; heads=$3; seeds=$4; extra=${5:-}
job="gully-e13-$(echo "$pseudo" | tr '._' '--' | cut -c1-22)-$(printf '%s|%s|%s|%s|%s' "$pseudo" "$modes" "$heads" "$seeds" "$extra" | md5sum | cut -c1-8)"
sed -e "s/__JOB__/$job/" -e "s|__PSEUDO__|$pseudo|" -e "s|__MODES__|$modes|" -e "s|__HEADS__|$heads|" \
    -e "s|__SEEDS__|$seeds|" -e "s|__EXTRA__|$extra|" "$(dirname "$0")/robust_job.yaml" | kubectl apply -f -
echo "$job"
