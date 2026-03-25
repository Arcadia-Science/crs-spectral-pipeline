# scripts/

Processing pipeline for extracting single-cell SRS (stimulated Raman scattering) spectra from Leica LIF lambda-scan images. The pipeline has two stages: **segmentation** and **spectral extraction**.

## Pipeline overview

```
LIF file
  │
  ▼
srs_segmentation.py          ← Stage 1: segment cells (runs on Modal with GPU)
  │  outputs: per-FOV mask TIFFs, QC overlays, measurement CSVs
  ▼
extract_cell_spectra.py       ← Stage 2: track cells & extract spectra (runs locally)
     outputs: per-FOV spectral CSVs, interactive Plotly HTML plots, optional MP4 animations
```

## Scripts

| File | Description |
|---|---|
| `srs_segmentation.py` | Splits multi-FOV LIF series into per-FOV TIFF stacks, preprocesses them (Gaussian blur, CLAHE, DoG background subtraction), and runs Cellpose segmentation. Splitting runs on a single Modal container; segmentation fans out across up to 10 GPU containers via `starmap`. |
| `run_srs_segmentation.sh` | Wrapper that defines the LIF paths and image names for a batch, joins them into CSV strings, and calls `modal run srs_segmentation.py`. Extra CLI flags are forwarded. |
| `extract_cell_spectra.py` | Loads pre-computed masks, tracks cells across lambda frames with [btrack](https://github.com/quantumjot/btrack), extracts mean SRS intensity per cell per wavenumber, and writes tidy CSVs and interactive Plotly figures. |
| `run_extract_cell_spectra.sh` | Wrapper that loops over configured LIF/image-name pairs and calls `extract_cell_spectra.py` for each, forwarding extra CLI flags and reporting pass/fail counts. |
| `napari_utils.py` | `NapariAnimator` dataclass that creates MP4 animations of tracked cells over SRS image stacks using napari, with wavenumber and scale-bar overlays. |
| `btrack_patches.py` | Monkey-patch for btrack's GLPK-based track optimiser so it accepts `"feasible"` solutions (not just `"optimal"`), which arise when `mip_gap` or `tm_lim` options cause the solver to stop early. |
| `btrack_config.json` | Motion and hypothesis model configuration for btrack's Kalman-filter tracker. |

## Commands used to process the publication data

### Stage 1 — Segmentation

```bash
bash scripts/run_srs_segmentation.sh --fov-range 0,9 --channel-names "E-CARS"
```

### Stage 2 — Spectral extraction

```bash
bash scripts/run_extract_cell_spectra.sh --animate --fps 5 --patch-btrack
```

## Notes

- **Lambda vs. wavenumber indexing:** Lambda indices and Raman-shift wavenumbers run in opposite directions. As the lambda index increases, the wavenumber (cm⁻¹) decreases. The `--lambda-range` argument operates on the lambda index.
- **btrack patching:** Pass `--patch-btrack` to `extract_cell_spectra.py` to accept feasible (non-proven-optimal) GLPK solutions, which is almost always fine for cell-tracking problems.
- **Modal:** `srs_segmentation.py` is executed via [Modal](https://modal.com/) (`modal run`) for cloud GPU access. The wrapper scripts handle argument formatting.
