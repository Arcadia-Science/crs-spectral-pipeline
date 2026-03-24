"""Extract per-cell SRS spectra from a segmented LIF image series.

For each FOV (M-index), this script:
    1. Loads pre-computed segmentation masks (from ``srs_segmentation.py``).
    2. Tracks cells across lambda frames with btrack, linking the same cell
       across consecutive wavenumber acquisitions.
    3. Extracts the mean SRS intensity per cell per wavenumber, producing
       one spectral curve per tracked cell.
    4. Writes a tidy CSV (one row per cell-wavenumber pair) and an
       interactive Plotly HTML figure.

Outputs (written to ``<processed_dir>/spectra/``):
    - ``SRS_M<nnn>_spectra.csv``  — columns: ID, t, wavenumber_cm1, mean_srs_intensity, …
    - ``SRS_M<nnn>_spectra.html`` — interactive line chart (one trace per cell + background)

Usage:
    python scripts/extract_cell_spectra.py \\
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

# Columns that btrack adds to its tracklet DataFrames but are not part of the
# final spectral output.  Dropped right before writing the CSV.
_BTRACK_INTERNAL_COLS = ["x", "y", "z", "parent", "root", "state", "generation", "dummy"]

# GLPK options for btrack optimiser: set time limit (tm_lim, ms) and
# acceptable MIP gap for optimisation
GLPK_OPTIONS: dict[str, object] = {"tm_lim": 120_000, "mip_gap": 0.003}


def _track_cells(label_images: np.ndarray, config_path: Path = BTRACK_CONFIG) -> list[Tracklet]:
    """Run btrack on a stack of label images and return optimised tracks.

    Args:
        label_images: Integer label stack of shape (T, Y, X) where each 2-D
            slice is a segmentation mask (0 = background).
        config_path: Path to the btrack JSON configuration (motion / hypothesis
            models).

    Returns:
        Optimised tracklets linking cell labels across frames.
    """
    objects = segmentation_to_objects(label_images)
    with btrack.BayesianTracker(verbose=False) as tracker:
        tracker.configure(config_path)
        tracker.append(objects)
        tracker.track()
        # Pass GLPK options explicitly because tracker.optimize() with no args
        # passes an empty dict, inadvertently overriding btrack's default
        # tm_lim=60s with no time limit at all.
        tracker.optimise(options=GLPK_OPTIONS)
        return tracker.tracks


def _tracks_to_dataframe(
    tracks: list[Tracklet],
    min_frames: int = MIN_TRACK_FRAMES,
) -> pd.DataFrame:
    """Convert tracklets to a tidy DataFrame, discarding short tracks.

    Args:
        tracks: Raw tracklets from ``_track_cells``.
        min_frames: Minimum number of frames a track must span to be retained.
            Short tracks are usually segmentation artefacts or edge cells.

    Returns:
        One row per (track, frame) pair. Columns include ``ID``, ``t``, ``x``,
        ``y``, and various btrack internals.
    """
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

    Cells that disappear mid-acquisition (due to segmentation dropouts or
    drifting out of the FOV) produce incomplete spectra that complicate
    downstream analysis, so they are discarded here.

    Args:
        df: Track DataFrame from ``_tracks_to_dataframe``.
        t_min: Start of the temporal window to require coverage over.
            Defaults to the minimum frame in the data.
        t_max: End of the temporal window to require coverage over.
            Defaults to the maximum frame in the data.
    """
    t_col = df["t"].to_numpy()
    if t_min is None:
        t_min = int(t_col.min())
    if t_max is None:
        t_max = int(t_col.max())

    expected_frames = t_max - t_min + 1
    in_window = df[(df["t"] >= t_min) & (df["t"] <= t_max)]
    frames_per_id = cast(pd.Series, in_window.groupby("ID")["t"].nunique())
    complete_ids = cast(pd.Series, frames_per_id[frames_per_id == expected_frames]).index.tolist()
    id_col = cast(pd.Series, in_window["ID"])
    return cast(pd.DataFrame, in_window.loc[id_col.isin(complete_ids)])


def _extract_cell_spectra(
    df: pd.DataFrame,
    masks: list[SegmentationMask],
    srs_intensities: np.ndarray,
    wavenumbers: np.ndarray,
) -> pd.DataFrame:
    """Compute the mean SRS intensity for each tracked cell at each wavenumber.

    For every (cell, frame) row in *df*, the tracked centroid is used to look
    up which segmentation label the cell belongs to, and then the mean SRS
    intensity over that label's pixels is recorded.

    Rows whose centroid falls on background (label == 0) or outside the image
    bounds are dropped — both situations indicate tracking artefacts.

    Args:
        df: Filtered track DataFrame (one row per cell per frame).
        masks: Per-frame segmentation masks, indexed by frame number.
        srs_intensities: SRS intensity stack of shape (T, Y, X).
        wavenumbers: Wavenumber (cm-1) for each frame, same length as *masks*.

    Returns:
        Subset of *df* augmented with ``label``, ``wavenumber_cm1``, and
        ``mean_srs_intensity`` columns.
    """
    img_h, img_w = masks[0].label_image.shape

    labels: list[int] = []
    wn_values: list[float] = []
    intensities: list[float] = []
    valid_idx: list[int] = []
    n_oob = 0  # out of bounds counter

    for row in tqdm(df.itertuples(), total=len(df), desc="Extracting spectra"):
        frame_i = int(row.t)  # pyright: ignore[reportAttributeAccessIssue]
        y_idx = round(row.y)  # pyright: ignore[reportAttributeAccessIssue]
        x_idx = round(row.x)  # pyright: ignore[reportAttributeAccessIssue]

        # Centroids can be extrapolated outside the image by the tracker.
        if not (0 <= y_idx < img_h and 0 <= x_idx < img_w):
            n_oob += 1
            continue

        label = int(masks[frame_i].label_image[y_idx, x_idx])
        if label == 0:
            continue

        cell_mask = masks[frame_i].label_image == label
        valid_idx.append(row.Index)  # pyright: ignore[reportAttributeAccessIssue]
        labels.append(label)
        wn_values.append(float(wavenumbers[frame_i]))
        intensities.append(float(srs_intensities[frame_i][cell_mask].mean()))

    n_skipped = len(df) - len(valid_idx)
    if n_oob:
        print(f"  Warning: {n_oob}/{len(df)} centroids were out of bounds, skipped")
    if n_skipped - n_oob > 0:
        print(f"  Warning: {n_skipped - n_oob}/{len(df)} centroids fell on background, skipped")

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
    """Compute the mean background SRS intensity (label == 0) per frame.

    The background spectrum is appended to the cell spectra DataFrame with
    ``ID = BACKGROUND_ID`` so it can be plotted alongside cell traces for
    reference (e.g. to gauge signal-to-background contrast).
    """
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
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Mean SRS intensity (a.u.)",
    )
    return fig


def _save_tracking_animation(
    srs_intensities: np.ndarray,
    tracks: list[Tracklet],
    label_stack: np.ndarray,
    output_dir: Path,
    m_index: int,
    fps: float,
) -> None:
    """Render a napari tracking animation for a single FOV."""
    from cellpose.utils import masks_to_outlines
    from napari_utils import NapariAnimator  # lazy import; napari is optional

    cell_outlines = masks_to_outlines(label_stack)

    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / f"SRS_M{m_index:03d}_tracking.mp4"
    animator = NapariAnimator(
        foreground=srs_intensities,
        tracks=tracks,
        outlines=cell_outlines,
        fps=fps,
    )
    animator.save_animation(mp4_path)
    print(f"  Saved {mp4_path.name}")


def process_fov(
    image: MicroscopyImage,
    m_index: int,
    masks_path: Path,
    wavenumbers: np.ndarray,
    output_dir: Path,
    dataset_label: str,
    *,
    animate: bool = False,
    fps: float = 10,
) -> None:
    """Run the full spectral-extraction pipeline for a single FOV.

    Steps: load masks -> track cells -> (optionally animate) -> extract spectra
    -> append background -> write CSV + HTML.

    Args:
        image: The opened LIF image (all FOVs).
        m_index: Which FOV (M-index) to process.
        masks_path: Path to the per-frame segmentation mask TIFF stack for
            this FOV.
        wavenumbers: Wavenumber values (cm-1), one per lambda frame.
        output_dir: Directory for CSV / HTML outputs (created if absent).
        dataset_label: Human-readable prefix for the plot title.
        animate: If True, also render a napari tracking animation MP4.
        fps: Playback FPS for the animation (ignored when *animate* is False).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    masks_raw = ski.io.imread(masks_path)
    masks = [SegmentationMask(m) for m in masks_raw]
    label_stack = np.array([m.label_image for m in masks])

    srs_intensities = image.get_intensities_from_channel("SRS")[m_index]

    tracks = _track_cells(label_stack)

    if animate:
        _save_tracking_animation(
            srs_intensities,
            tracks,
            label_stack,
            output_dir.parent,
            m_index,
            fps,
        )

    df = _tracks_to_dataframe(tracks)
    df = _filter_complete_tracks(df)
    df = _extract_cell_spectra(df, masks, srs_intensities, wavenumbers)
    df_bg = _background_spectra(masks, srs_intensities, wavenumbers)
    df_spectra = pd.concat([df, df_bg]).drop(columns=_BTRACK_INTERNAL_COLS, errors="ignore")

    stem = f"SRS_M{m_index:03d}"
    df_spectra.to_csv(output_dir / f"{stem}_spectra.csv", index=False)

    fig = _make_figure(df_spectra, f"{dataset_label} M{m_index:03d} cell spectra")
    fig.write_html(output_dir / f"{stem}_spectra.html")
    print(f"  Saved {stem}_spectra.csv and {stem}_spectra.html")


def main(
    lif_path: Path,
    image_name: str,
    *,
    animate: bool = False,
    fps: float = 10,
) -> None:
    """Entry point: discover mask TIFFs, resolve wavenumbers, and loop over FOVs."""
    processed_dir = lif_path.parent / "processed" / lif_path.stem / image_name
    masks_paths = natsorted(processed_dir.glob("*masks.tiff"))
    if not masks_paths:
        raise FileNotFoundError(f"No mask TIFFs found in {processed_dir}")

    image = MicroscopyImage.from_lif_path(lif_path, image_name=image_name)

    channel_names = [ch.name for ch in image.channels]
    if "SRS" not in channel_names:
        raise ValueError(f"'SRS' channel not found in image — available channels: {channel_names}")
    channel_index = channel_names.index("SRS")

    wavelengths_nm = image.metadata.instrument.channel_metadata_list[
        channel_index
    ].measured.w_values_nm
    if wavelengths_nm is None:
        raise ValueError(
            f"Wavelengths (w_values_nm) not found in metadata for channel "
            f"'{channel_names[channel_index]}' (index {channel_index})"
        )
    wavenumbers = np.asarray(calculate_raman_shift(wavelengths_nm))

    dataset_label = f"{lif_path.parent.name}_{lif_path.stem}"
    output_dir = processed_dir / "spectra"

    passed = 0
    failed_fovs: list[str] = []

    for masks_path in masks_paths:
        m_match = re.search(r"M(\d+)", masks_path.stem)
        if m_match is None:
            print(f"  Skipping {masks_path.name}: could not parse M-index")
            continue
        m_index = int(m_match.group(1))
        print(f"Processing FOV M{m_index:03d} ({masks_path.name})...")
        try:
            process_fov(
                image,
                m_index,
                masks_path,
                wavenumbers,
                output_dir,
                dataset_label,
                animate=animate,
                fps=fps,
            )
            passed += 1
        except Exception as e:
            print(f"  Skipping FOV M{m_index:03d}: {type(e).__name__}: {e}")
            failed_fovs.append(f"M{m_index:03d}")

    total = passed + len(failed_fovs)
    print(f"\nDone: {passed}/{total} FOVs succeeded")
    if failed_fovs:
        print(f"  Failed: {', '.join(failed_fovs)}")


if __name__ == "__main__":
    from btrack_patches import patch_optimiser

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lif_path", type=Path, help="Path to the .lif file")
    parser.add_argument("image_name", type=str, help="Image series name inside the LIF")
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Save a napari tracking animation (MP4) for each FOV",
    )
    parser.add_argument(
        "--fps", type=float, default=10, help="Animation playback FPS (default: 10)"
    )
    parser.add_argument(
        "--patch-btrack",
        action="store_true",
        help="Patch btrack's optimiser to accept 'feasible' GLPK solutions "
        "(required when using mip_gap or tm_lim options that cause early termination)",
    )
    args = parser.parse_args()

    if args.patch_btrack:
        patch_optimiser()

    main(args.lif_path, args.image_name, animate=args.animate, fps=args.fps)
