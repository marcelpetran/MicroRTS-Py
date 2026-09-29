#!/usr/bin/env bash
# Pull final-epoch checkpoints from the cluster into the local repo.
#
# For every run folder in ~/MicroRTS-Py/models/ on the cluster EXCEPT id 0,
# copies only the three epoch-$EPOCH checkpoints:
#   classic_qnet_ep$EPOCH.pth, om_inference_ep$EPOCH.pth, om_qnet_ep$EPOCH.pth
# into <repo>/models/<same id>/ locally. Folders without epoch-$EPOCH
# checkpoints (older experiments) are skipped, so the script is safe to
# rerun as new MAP_3 runs finish.
#
# Usage:
#   ./scripts/pull_models_from_cluster.sh
#   EPOCH=60 REMOTE=petram21@login3.rci.cvut.cz ./scripts/pull_models_from_cluster.sh
#
# Requires passwordless ssh to the cluster (already set up).

set -euo pipefail

REMOTE="${REMOTE:-petram21@login3.rci.cvut.cz}"
REMOTE_MODELS_DIR="${REMOTE_MODELS_DIR:-MicroRTS-Py/models}"  # relative to remote $HOME
EPOCH="${EPOCH:-60}"

LOCAL_MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$LOCAL_MODELS_DIR"

echo "Listing run folders on $REMOTE:$REMOTE_MODELS_DIR (skipping id 0) ..."
IDS="$(ssh "$REMOTE" "ls -1d $REMOTE_MODELS_DIR/*/ 2>/dev/null | xargs -n 1 basename | grep -v '^0$' | sort")"
if [ -z "$IDS" ]; then
    echo "No run folders found."
    exit 1
fi
n=$(echo "$IDS" | wc -l | tr -d ' ')
echo "Found $n run folders."

copied=0
skipped=0
for id in $IDS; do
    # Which of the three epoch-$EPOCH files exist for this folder?
    existing="$(ssh "$REMOTE" "ls $REMOTE_MODELS_DIR/$id/classic_qnet_ep$EPOCH.pth $REMOTE_MODELS_DIR/$id/om_inference_ep$EPOCH.pth $REMOTE_MODELS_DIR/$id/om_qnet_ep$EPOCH.pth 2>/dev/null || true")"
    if [ -z "$existing" ]; then
        echo "  [skip] $id (no epoch-$EPOCH checkpoints)"
        skipped=$((skipped + 1))
        continue
    fi
    dst="$LOCAL_MODELS_DIR/$id"
    mkdir -p "$dst"
    srcs=()
    for f in $existing; do
        srcs+=("$REMOTE:$f")
    done
    echo "  [copy] $id: $(echo "$existing" | xargs -n 1 basename | tr '\n' ' ')"
    scp -q "${srcs[@]}" "$dst/"
    copied=$((copied + 1))
done

echo "Done: $copied folder(s) synced, $skipped skipped -> $LOCAL_MODELS_DIR"
