"""Extract per-cell SRS spectra from a segmented LIF image series.

For each FOV (M-index), this script:
    1. Loads pre-computed segmentation masks.
    2. Tracks cells across lambda frames with btrack.
    3. Extracts the mean SRS intensity per cell per wavenumber.
    4. Writes a CSV and an interactive HTML figure.

Usage:
    python scripts/extract_cell_spectra.py \
        /path/to/file.lif "Lambda_SRS_CARS_1800_600_1pt0_Nav1"
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import cast

import btrack
import numpy as np
import pandas as pd
import plotly.express as px
import skimage as ski
from arcadia_microscopy_tools import MicroscopyImage
from arcadia_microscopy_tools.leica import calculate_raman_shift
from arcadia_microscopy_tools.masks import SegmentationMask
from btrack.btypes import Tracklet
from btrack.io import segmentation_to_objects
from natsort import natsorted
from plotly.graph_objects import Figure
from tqdm import tqdm

BTRACK_CONFIG = Path(__file__).resolve().parent / "btrack_config.json"

BACKGROUND_ID = -1
MIN_TRACK_FRAMES = 5


def _track_cells(label_images: np.ndarray, config_path: Path = BTRACK_CONFIG) -> list[Tracklet]:
    """Run btrack on a stack of label images and return optimised tracks."""
    objects = segmentation_to_objects(label_images)
    with btrack.BayesianTracker(verbose=False) as tracker:
        tracker.configure(config_path)
        tracker.append(objects)
        tracker.track()
        tracker.optimize()
        return tracker.tracks


def _tracks_to_dataframe(
    tracks: list[Tracklet], min_frames: int = MIN_TRACK_FRAMES
) -> pd.DataFrame:
    """Convert tracklets to a DataFrame, keeping only long-enough tracks."""
    dfs = [pd.DataFrame(t.to_dict()) for t in tracks if len(t) >= min_frames]
    if not dfs:
        raise ValueError(f"No tracks with at least {min_frames} frames found")
    return pd.concat(dfs, ignore_index=True)


def _filter_complete_tracks(
    df: pd.DataFrame,
    t_min: int | None = None,
    t_max: int | None = None,
) -> pd.DataFrame:
    """Keep only tracks that span every frame in [t_min, t_max].

    Defaults to the full temporal range present in the data.
    """
    t_col = df["t"].to_numpy()
    if t_min is None:
        t_min = int(t_col.min())
    if t_max is None:
        t_max = int(t_col.max())
    expected_frames = t_max - t_min + 1
    df = cast(pd.DataFrame, df[(df["t"] >= t_min) & (df["t"] <= t_max)])
    counts = cast(pd.Series, df.groupby("ID")["t"].nunique())
    complete_ids = cast(pd.Series, counts[counts == expected_frames]).index.tolist()
    return cast(pd.DataFrame, df.loc[df["ID"].isin(complete_ids)])


def _extract_cell_spectra(
    df: pd.DataFrame,
    masks: list[SegmentationMask],
    srs_intensities: np.ndarray,
    wavenumbers: np.ndarray,
) -> pd.DataFrame:
    """Populate each tracked row with wavenumber and mean SRS intensity.

    Rows whose centroid falls on background (label == 0) are dropped.
    """
    labels: list[int] = []
    wn_values: list[float] = []
    intensities: list[float] = []
    valid_idx: list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting spectra"):
        frame_i = int(cast(float, row["t"]))
        y_idx = round(cast(float, row["y"]))
        x_idx = round(cast(float, row["x"]))
        label = int(masks[frame_i].label_image[y_idx, x_idx])

        if label == 0:
            continue

        cell_mask = masks[frame_i].label_image == label
        valid_idx.append(idx)
        labels.append(label)
        wn_values.append(float(wavenumbers[frame_i]))
        intensities.append(float(srs_intensities[frame_i][cell_mask].mean()))

    n_skipped = len(df) - len(valid_idx)
    if n_skipped:
        print(f"  Warning: {n_skipped}/{len(df)} centroids fell on background, skipped")

    result = df.loc[valid_idx].copy()
    result["label"] = labels
    result["wavenumber_cm1"] = wn_values
    result["mean_srs_intensity"] = intensities
    return result


def _background_spectra(
    masks: list[SegmentationMask],
    srs_intensities: np.ndarray,
    wavenumbers: np.ndarray,
) -> pd.DataFrame:
    """Compute mean background intensity (label == 0) per frame."""
    records = []
    for frame_i, intensities in enumerate(srs_intensities):
        bg_mask = masks[frame_i].label_image == 0
        records.append(
            {
                "ID": BACKGROUND_ID,
                "label": 0,
                "t": frame_i,
                "wavenumber_cm1": wavenumbers[frame_i],
                "mean_srs_intensity": float(intensities[bg_mask].mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def _make_figure(df: pd.DataFrame, title: str) -> Figure:
    """Create an interactive Plotly line chart of per-cell SRS spectra."""
    fig = px.line(df, x="wavenumber_cm1", y="mean_srs_intensity", color="ID")
    fig.update_layout(title=title, template="plotly_white")
    return fig


def process_fov(
    image: MicroscopyImage,
    m_index: int,
    masks_path: Path,
    wavenumbers: np.ndarray,
    output_dir: Path,
    dataset_label: str,
) -> None:
    """Full extraction pipeline for a single FOV (M-index)."""
    masks_raw = ski.io.imread(masks_path)
    masks = [SegmentationMask(m) for m in masks_raw]
    label_stack = np.array([m.label_image for m in masks])

    srs_intensities = image.get_intensities_from_channel("SRS")[m_index]

    tracks = _track_cells(label_stack)
    df = _tracks_to_dataframe(tracks)
    df = _filter_complete_tracks(df)
    df = _extract_cell_spectra(df, masks, srs_intensities, wavenumbers)
    df_bg = _background_spectra(masks, srs_intensities, wavenumbers)
    _btrack_only_cols = ["x", "y", "z", "parent", "root", "state", "generation", "dummy"]
    df_spectra = pd.concat([df, df_bg]).drop(columns=_btrack_only_cols, errors="ignore")

    stem = f"{dataset_label}_M{m_index:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_spectra.to_csv(output_dir / f"{stem}_spectra.csv", index=False)

    fig = _make_figure(df_spectra, f"{dataset_label} M{m_index:03d} cell spectra")
    fig.write_html(output_dir / f"{stem}_spectra.html")
    print(f"  Saved {stem}_spectra.csv and {stem}_spectra.html")


def main(lif_path: Path, image_name: str) -> None:
    processed_dir = lif_path.parent / "processed" / lif_path.stem / image_name
    masks_paths = natsorted(processed_dir.glob("*masks.tiff"))
    if not masks_paths:
        raise FileNotFoundError(f"No mask TIFFs found in {processed_dir}")

    image = MicroscopyImage.from_lif_path(lif_path, image_name=image_name)

    channel_index = [ch.name for ch in image.channels].index("SRS")
    wavelengths_nm = image.metadata.instrument.channel_metadata_list[
        channel_index
    ].measured.w_values_nm
    if wavelengths_nm is None:
        raise ValueError("Wavelengths not found in metadata")
    wavenumbers = np.asarray(calculate_raman_shift(wavelengths_nm))

    dataset_label = f"{lif_path.parent.name}_{lif_path.stem}"
    output_dir = processed_dir / "spectra"

    for masks_path in masks_paths:
        m_match = re.search(r"M(\d+)", masks_path.stem)
        if m_match is None:
            print(f"  Skipping {masks_path.name}: could not parse M-index")
            continue
        m_index = int(m_match.group(1))
        print(f"Processing FOV M{m_index:03d} ({masks_path.name})...")
        try:
            process_fov(image, m_index, masks_path, wavenumbers, output_dir, dataset_label)
        except Exception as e:
            print(f"  Skipping FOV M{m_index:03d}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lif_path", type=Path, help="Path to the .lif file")
    parser.add_argument("image_name", type=str, help="Image series name inside the LIF")
    args = parser.parse_args()
    main(args.lif_path, args.image_name)
