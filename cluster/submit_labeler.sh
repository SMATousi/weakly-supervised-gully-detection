#!/usr/bin/env bash
# usage: cluster/submit_labeler.sh <config-name> <gpus> <min-gpu-mem-MiB> <sets, e.g. "labeled pool" (model loaded once)> [extra args]
# Job memory follows the namespace rule (limit <= 1.2 x CPU limit).
set -euo pipefail
cfg=$1; gpus=$2; minmem=$3; set_=$4; extra=${5:-}
# A100 nodes advertise their GPUs as nvidia.com/a100 (nvidia.com/gpu is 0 there): big-memory jobs must request that.
if [ "$minmem" -ge 38000 ]; then gpures=a100; else gpures=gpu; fi
base="gully-lab-$(echo "$cfg" | tr '._' '--' | cut -c1-28)"
job="$base-$(printf '%s|%s' "$set_" "$extra" | md5sum | cut -c1-8)"   # unique per (sets, extra), <= 63 chars
cpu=$((5 * gpus)); [ $cpu -lt 6 ] && cpu=6;   # vLLM needs ~7 GiB host RAM per GPU (7 GiB was OOM-killed at TP=2)
mem=$((cpu * 6 / 5)); cpureq=$cpu; memreq=$mem   # requests = limits (namespace ratio rule)
sed -e "s/__JOB__/$job/" -e "s/__CONFIG__/$cfg/" -e "s/__SET__/$set_/" -e "s|__EXTRA__|$extra|" \
    -e "s/__GPURES__/$gpures/" -e "s/__GPUS__/$gpus/" -e "s/__MINMEM__/$minmem/" -e "s/__CPU__/$cpu/" -e "s/__MEM__/$mem/" \
    -e "s/__CPUREQ__/$cpureq/" -e "s/__MEMREQ__/$memreq/" "$(dirname "$0")/labeler_job.template.yaml" | kubectl apply -f -
echo "$job"
