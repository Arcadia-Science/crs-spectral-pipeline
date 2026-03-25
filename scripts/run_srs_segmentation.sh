#!/usr/bin/env bash
set -uo pipefail

SCRIPT="$(cd "$(dirname "$0")" && pwd)/srs_segmentation.py"

LIF_PATHS=(
    "The_Borg/Tara/2026-03-20/ED666_BY4741_unlabeled.lif"
    "The_Borg/Tara/2026-03-20/SP286_BY4741_unlabeled.lif"
)
IMAGE_NAMES=(
    "Lambda_3100_2800_SRS_CARS_Nav1"
    "Lambda_3100_2800_SRS_CARS_Nav1"
)

# Join arrays into comma-separated strings for the Modal entrypoint.
lif_paths_csv=$(IFS=,; echo "${LIF_PATHS[*]}")
image_names_csv=$(IFS=,; echo "${IMAGE_NAMES[*]}")

echo "━━━  Running SRS segmentation  ━━━"
echo "LIF files: ${#LIF_PATHS[@]}"
echo

modal run "$SCRIPT" \
    --lif-paths "$lif_paths_csv" \
    --image-names "$image_names_csv" \
    "$@"
