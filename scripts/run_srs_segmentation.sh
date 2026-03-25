#!/usr/bin/env bash
set -uo pipefail

SCRIPT="$(cd "$(dirname "$0")" && pwd)/srs_segmentation.py"

LIF_PATHS=(
    "The_Borg/Tara/2026-03-05/Chlorella_WT_Bristol_slide1.lif"
    "The_Borg/Tara/2026-03-05/Chlorella_WT_Bristol_slide2.lif"
    "The_Borg/Tara/2026-03-09/Chlorella_WT_TAP_slide1.lif"
    "The_Borg/Tara/2026-03-09/Chlorella_WT_TAP_slide2.lif"
)
IMAGE_NAMES=(
    "Lambda_SRS_CARS_1800_600_1pt0_Nav1"
    "Lambda_SRS_CARS_1800_600_1pt0_Nav1"
    "Lambda_1800_600_1pt0_Nav2"
    "Lambda_1800_600_1pt0_Nav1"
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
