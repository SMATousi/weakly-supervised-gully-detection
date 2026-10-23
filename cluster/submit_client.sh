#!/usr/bin/env bash
# usage: cluster/submit_client.sh <config-name> "<sets e.g. labeled pool>" [extra run_labeler args]
set -euo pipefail
cfg=$1; sets=$2; extra=${3:-}
job="gully-cli-$(echo "$cfg" | tr '._' '--' | cut -c1-40)-$(echo "$sets" | tr ' ' '-')"
sed -e "s/__JOB__/$job/" -e "s/__CONFIG__/$cfg/" -e "s/__SETS__/$sets/" -e "s|__EXTRA__|$extra|" \
    "$(dirname "$0")/labeler_client_job.template.yaml" | kubectl apply -f -
echo "$job"
