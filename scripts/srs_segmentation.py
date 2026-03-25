"""Segment cells in SRS lambda-scan LIF images using Modal for parallelism.

For each configured LIF/image-name pair, this script:
    1. Splits a multi-FOV, multi-lambda LIF image series into per-FOV TIFF
       stacks (one stack per channel per FOV).
    2. Preprocesses each stack (Gaussian blur, adaptive histogram equalisation,
       difference-of-Gaussians background subtraction).
    3. Runs Cellpose segmentation on every frame, saving per-frame QC overlays,
       cell-property CSVs, and a full mask TIFF stack.

Splitting runs on a single Modal container; segmentation fans out across up
to 10 GPU containers via ``starmap``.

Outputs (written to ``<lif_parent>/processed/<lif_stem>/<image_name>/``):
    - ``<channel>_M<nnn>.tiff``         — per-FOV intensity stack
    - ``<channel>_M<nnn>_masks.tiff``   — per-FOV label-image stack
    - ``overlays/<stem>_<i>_overlay.jpg`` — QC overlay per frame
    - ``measurements/<stem>_<i>_measurements.csv`` — cell properties per frame

Usage:
    modal run scripts/srs_segmentation.py \\
        --lif-paths "path/to/file1.lif,path/to/file2.lif" \\
        --image-names "Lambda_SRS_CARS_Nav1,Lambda_SRS_CARS_Nav1" \\
        --fov-range 0,9 --lambda-range 5,95 --channel-names "E-CARS"
"""

from __future__ import annotations
import shutil
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path

import modal
import numpy as np

app = modal.App("srs-navigator-segmentation")

s3_mount = modal.CloudBucketMount(
    bucket_name="arcadia-microscopy-data",
    secret=modal.Secret.from_name("aws-s3-access-secret"),
)

dependencies = [
    "arcadia-microscopy-tools[all]>=0.3.1",
]
image = modal.Image.debian_slim().uv_pip_install(dependencies)


def _write_via_temp(dest: str | Path, suffix: str, writer: Callable[[str], None]) -> None:
    """Write an artifact via a temporary local file.

    This avoids partial writes on mounted storage by writing locally first,
    then copying the completed file to `dest`.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        writer(tmp.name)
        shutil.copy(tmp.name, dest)
        Path(tmp.name).unlink()


@app.function(image=image, volumes={"/s3": s3_mount})
def split_lif_file(
    lif_path: str,
    image_name: str,
    fov_range: tuple[int, int] | None = None,
    lambda_range: tuple[int, int] | None = None,
    channel_names: list[str] | None = None,
) -> tuple[list[str], float]:
    """Split a LIF image series into per-FOV TIFF stacks.

    The function loads one image series, optionally filters channels, applies
    optional FOV and lambda slicing, and writes one TIFF stack per FOV index.

    Returns:
        A tuple of (written_tiff_paths, pixel_size_um).
    """
    from arcadia_microscopy_tools import MicroscopyImage
    from skimage.io import imsave

    full_lif_path = Path(f"/s3/{lif_path}")
    image = MicroscopyImage.from_lif_path(full_lif_path, image_name=image_name)

    if channel_names is not None:
        channels = [ch for ch in image.channels if ch.name in channel_names]
    else:
        channels = list(image.channels)
    print(f"Found channels: {[ch.name for ch in channels]}")

    channel_index = image.channels.index(channels[0])
    pixel_size_um = image.metadata.instrument.channel_metadata_list[
        channel_index
    ].resolution.xy_step_um
    print(f"Pixel size: {pixel_size_um} um/pixel")

    lif_parent = full_lif_path.parent
    output_dir = lif_parent / "processed" / full_lif_path.stem / image_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_paths: list[str] = []

    for channel in channels:
        intensities = image.get_intensities_from_channel(channel)
        print(f"Channel '{channel.name}' raw shape: {intensities.shape}")

        # Apply optional crop windows while preserving full-range defaults.
        fov_slice = slice(*fov_range) if fov_range is not None else slice(None)
        l_slice = slice(*lambda_range) if lambda_range is not None else slice(None)
        cropped = intensities[fov_slice, l_slice, :, :]

        fov_start = fov_range[0] if fov_range is not None else 0
        print(f"Channel '{channel.name}' cropped shape: {cropped.shape}")

        for fov_idx in range(cropped.shape[0]):
            stack = cropped[fov_idx]
            filename = f"{channel.name}_M{fov_start + fov_idx:03d}.tiff"
            output_path = output_dir / filename

            _write_via_temp(
                output_path,
                ".tiff",
                lambda p, arr=stack: imsave(p, arr, check_contrast=False),
            )

            tiff_paths.append(str(output_path))
            print(f"Saved {filename} with shape {stack.shape}")

    print(f"\nSplit complete: {len(tiff_paths)} TIFF stacks saved to {output_dir}")
    return tiff_paths, pixel_size_um


@app.function(image=image, gpu="T4", volumes={"/s3": s3_mount}, max_containers=10, timeout=1800)
def segment_stack(s3_path: str, pixel_size_um: float) -> tuple[str, int | None, str | None]:
    """Segment all frames in a stack and persist QC/measurement outputs.

    Workflow:
        1) Load and preprocess each frame.
        2) Run segmentation per frame.
        3) Save per-frame overlay JPGs and measurement CSVs.
        4) Save the full mask stack as TIFF.

    Returns:
        (input_stack_path, total_cells, error_message), where `error_message`
        is None on success.
    """
    import pandas as pd
    import skimage as ski
    from arcadia_microscopy_tools import Channel, overlay_channels
    from arcadia_microscopy_tools.masks import SegmentationMask
    from arcadia_microscopy_tools.model import SegmentationModel
    from arcadia_microscopy_tools.operations import subtract_background_dog
    from arcadia_microscopy_tools.pipeline import ImageOperation, PipelineParallelized
    from arcadia_pycolor import HexCode
    from cellpose.utils import masks_to_outlines
    from skimage.io import imread, imsave

    try:
        stack_path = Path(s3_path)
        stack = imread(s3_path)
        print(f"Loaded {stack_path.name}: shape {stack.shape}, dtype {stack.dtype}")

        # Light denoise + local contrast + background suppression.
        preprocess = PipelineParallelized(
            [
                ImageOperation(ski.filters.gaussian, sigma=1.2),
                ImageOperation(ski.exposure.equalize_adapthist, clip_limit=0.01),
                ImageOperation(subtract_background_dog, percentile=10),
            ],
            preserve_dtype=False,
        )
        preprocessed = preprocess(stack)

        model = SegmentationModel()
        masks: list[np.ndarray] = []
        total_cells = 0
        stem = stack_path.stem

        for i, frame in enumerate(preprocessed):
            mask = model.segment(frame)
            masks.append(mask)

            if mask.max() == 0:
                print(f"  Frame {i}/{len(preprocessed)}: 0 cells (skipping QC outputs)")
                continue

            segmentation_mask = SegmentationMask(mask)
            num_cells = segmentation_mask.num_cells
            total_cells += num_cells

            # Convert label boundaries into a human-readable QC overlay.
            mask_channel = Channel("mask", color=HexCode("", "#ffff00"))
            mask_outlines = masks_to_outlines(segmentation_mask.label_image)
            overlay_rgb_float = overlay_channels(
                frame, channel_intensities={mask_channel: mask_outlines}
            )
            overlay_rgb_uint8 = ski.exposure.rescale_intensity(
                overlay_rgb_float,
                out_range=np.uint8,  # pyright: ignore[reportArgumentType]
            )
            output_dir = stack_path.parent / "overlays"
            output_dir.mkdir(parents=True, exist_ok=True)
            jpg_path = output_dir / f"{stem}_{i}_overlay.jpg"
            _write_via_temp(
                jpg_path,
                ".jpg",
                lambda p, img=overlay_rgb_uint8: imsave(p, img, check_contrast=False),
            )
            print(f"Saved overlay to {jpg_path}")

            # Convert cell properties to microns and save as CSV.
            cell_properties_um = segmentation_mask.convert_properties_to_microns(
                pixel_size_um=pixel_size_um
            )
            dataframe = pd.DataFrame.from_dict(cell_properties_um)
            output_dir = stack_path.parent / "measurements"
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / f"{stem}_{i}_measurements.csv"
            _write_via_temp(
                csv_path,
                ".csv",
                lambda p, df=dataframe: df.to_csv(p, index=False),
            )
            print(f"Saved cell properties to {csv_path}")

            print(f"  Frame {i}/{len(preprocessed)}: {num_cells} cells")

        mask_stack = np.stack(masks).astype(np.uint16)
        mask_path = stack_path.parent / f"{stem}_masks.tiff"
        _write_via_temp(
            mask_path,
            ".tiff",
            lambda p: imsave(p, mask_stack, check_contrast=False),
        )

        print(
            f"Saved {mask_path.name}: shape {mask_stack.shape}, "
            f"total cells across frames: {total_cells}"
        )
        return s3_path, total_cells, None
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"segment_stack failed for {s3_path}: {msg}")
        traceback.print_exc()
        return s3_path, None, msg


@app.function(image=image, volumes={"/s3": s3_mount}, timeout=3600)
def run_pipeline(
    lif_path_image_name_mapping: dict[str, str],
    fov_range: tuple[int, int] | None,
    lambda_range: tuple[int, int] | None,
    channel_names: list[str] | None,
) -> None:
    """Run split + segmentation over all configured LIF/image pairs."""
    for lif_path, image_name in lif_path_image_name_mapping.items():
        try:
            print(
                f"Splitting LIF: FOV={fov_range}, Lambda={lambda_range}, "
                f"channels={channel_names}"
            )
            tiff_paths, pixel_size_um = split_lif_file.remote(
                lif_path, image_name, fov_range, lambda_range, channel_names
            )
            print(f"\nSegmenting {len(tiff_paths)} stacks in parallel...")

            for path, cell_count, err in segment_stack.starmap(
                [(path, pixel_size_um) for path in tiff_paths]
            ):
                name = Path(path).name
                if err is not None:
                    print(f"  {name}: FAILED — {err}")
                else:
                    print(f"  {name}: {cell_count} total cells across all frames")
        except Exception as e:
            print(f"Failed LIF {lif_path} ({image_name}): {type(e).__name__}: {e}")
            traceback.print_exc()


def _parse_range(value: str) -> tuple[int, int]:
    """Parse a 'start,stop' string into a two-element integer tuple."""
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError(f"Expected 'start,stop', got '{value}'")
    return int(parts[0]), int(parts[1])


@app.local_entrypoint()
def main(
    lif_paths: str,
    image_names: str,
    fov_range: str | None = None,
    lambda_range: str | None = None,
    channel_names: str | None = None,
) -> None:
    """Orchestrate LIF splitting and stack segmentation runs.

    For each configured LIF/image pair, this entrypoint first writes split TIFF
    stacks, then dispatches segmentation across stacks with Modal starmap.

    Args:
        lif_paths: Comma-separated S3-relative LIF paths.
        image_names: Comma-separated image series names (same order as lif_paths).
        fov_range: Optional 'start,stop' field-of-view index range (e.g. '0,9').
        lambda_range: Optional 'start,stop' lambda index range.  Note that
            lambda indices run inversely to wavenumber (cm-1).
        channel_names: Optional comma-separated channel names to keep.
    """
    lif_list = [p.strip() for p in lif_paths.split(",")]
    name_list = [n.strip() for n in image_names.split(",")]
    if len(lif_list) != len(name_list):
        raise ValueError(
            f"Mismatched counts: {len(lif_list)} LIF paths vs {len(name_list)} image names"
        )

    lif_path_image_name_mapping = dict(zip(lif_list, name_list, strict=True))
    parsed_fov = _parse_range(fov_range) if fov_range is not None else None
    parsed_lambda = _parse_range(lambda_range) if lambda_range is not None else None
    parsed_channels = (
        [c.strip() for c in channel_names.split(",")] if channel_names is not None else None
    )

    run_pipeline.remote(
        lif_path_image_name_mapping,
        parsed_fov,
        parsed_lambda,
        parsed_channels,
    )
