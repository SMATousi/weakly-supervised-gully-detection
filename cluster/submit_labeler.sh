#!/usr/bin/env bash
# usage: cluster/submit_labeler.sh <config-name> <gpus> <min-gpu-mem-MiB> <sets, e.g. "labeled pool" (model loaded once)> [extra args]
# Job memory follows the namespace rule (limit <= 1.2 x CPU limit).
set -euo pipefail
cfg=$1; gpus=$2; minmem=$3; set_=$4; extra=${5:-}
job="gully-lab-$(echo "$cfg" | tr '._' '--' | cut -c1-40)-$(echo "$set_" | tr ' ' '-')"
cpu=$((4 * gpus)); mem=$((cpu * 6 / 5)); cpureq=$cpu; memreq=$mem   # requests = limits (namespace ratio rule)
sed -e "s/__JOB__/$job/" -e "s/__CONFIG__/$cfg/" -e "s/__SET__/$set_/" -e "s/__EXTRA__/$extra/" \
    -e "s/__GPUS__/$gpus/" -e "s/__MINMEM__/$minmem/" -e "s/__CPU__/$cpu/" -e "s/__MEM__/$mem/" \
    -e "s/__CPUREQ__/$cpureq/" -e "s/__MEMREQ__/$memreq/" "$(dirname "$0")/labeler_job.template.yaml" | kubectl apply -f -
echo "$job"
