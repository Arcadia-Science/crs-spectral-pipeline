#!/usr/bin/env bash
set -uo pipefail

NAS_ROOT="$HOME/arc_nas_mirror"
SCRIPT="$(cd "$(dirname "$0")" && pwd)/extract_cell_spectra.py"

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

passed=0
failed=0
failed_names=()

for i in "${!LIF_PATHS[@]}"; do
    lif_path="$NAS_ROOT/${LIF_PATHS[$i]}"
    image_name="${IMAGE_NAMES[$i]}"
    label="$(basename "${LIF_PATHS[$i]}" .lif) / $image_name"

    echo "━━━  $label  ━━━"

    if python "$SCRIPT" "$lif_path" "$image_name" "$@"; then
        ((passed++))
    else
        ((failed++))
        failed_names+=("$label")
    fi
    echo
done

echo "═══  Done: $passed passed, $failed failed  ═══"
for name in "${failed_names[@]}"; do
    echo "  FAILED: $name"
done

exit "$failed"
