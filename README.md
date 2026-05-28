# crs-spectral-pipeline

Pipeline for extracting single-cell coherent Raman scattering (CRS) spectra from hyperspectral microscopy data. Performs cell segmentation (Cellpose on cloud GPU), cell tracking across wavenumber frames (btrack), and per-cell spectral extraction.

## Installation and setup

This repository uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage software environments and installations. Run the following commands to create the development environment:

```bash
git clone https://github.com/Arcadia-Science/crs-spectral-pipeline.git
cd crs-spectral-pipeline

uv venv --python 3.12
source .venv/bin/activate
uv sync
uv pip install pyqt6 "napari[all]" napari-animation
```

## Pipeline overview

```
Raw LIF files (S3)
       │
       ▼
Stage 1: Cell segmentation (Modal GPU)
       │   Cellpose on preprocessed per-FOV TIFF stacks → mask TIFFs + QC outputs
       ▼
Stage 2: Cell tracking + spectral extraction (local)
       │   btrack across wavenumber frames → per-cell mean SRS intensity
       ▼
Outputs: per-FOV spectral CSVs, interactive Plotly HTML, optional Napari MP4 animations
```

**Input:** Leica `.lif` lambda-scan images (SRS/CARS channels), multi-FOV, multi-wavelength.

**Output:** Tidy CSVs with one row per cell per wavenumber, plus interactive spectral plots.

See `[scripts/README.md](scripts/README.md)` for full pipeline documentation including commands, script descriptions, and technical notes.

## Key dependencies


| Package                                                                                 | Role                                          |
| --------------------------------------------------------------------------------------- | --------------------------------------------- |
| [arcadia-microscopy-tools](https://github.com/Arcadia-Science/arcadia-microscopy-tools) | LIF I/O, preprocessing, Cellpose segmentation |
| [Modal](https://modal.com/)                                                             | Serverless GPU compute for segmentation       |
| [btrack](https://github.com/quantumjot/btrack)                                          | Bayesian multi-object cell tracking           |
| [Plotly](https://plotly.com/python/)                                                    | Interactive spectral visualizations           |
| [Napari](https://napari.org/)                                                           | Optional animation of tracked cells           |


## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide--credit-for-contributions.md).