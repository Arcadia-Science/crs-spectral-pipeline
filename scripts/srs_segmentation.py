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
    m_range: tuple[int, int] | None = None,
    lambda_range: tuple[int, int] | None = None,
    channel_names: list[str] | None = None,
) -> tuple[list[str], float]:
    """Split a LIF image series into per-M TIFF stacks.

    The function loads one image series, optionally filters channels, applies
    optional M and lambda slicing, and writes one TIFF stack per M index.

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
        m_slice = slice(*m_range) if m_range is not None else slice(None)
        l_slice = slice(*lambda_range) if lambda_range is not None else slice(None)
        cropped = intensities[m_slice, l_slice, :, :]

        m_start = m_range[0] if m_range is not None else 0
        print(f"Channel '{channel.name}' cropped shape: {cropped.shape}")

        for m_idx in range(cropped.shape[0]):
            stack = cropped[m_idx]
            filename = f"{channel.name}_M{m_start + m_idx:03d}.tiff"
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


@app.function(image=image, gpu="T4", volumes={"/s3": s3_mount}, max_containers=10)
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

        preprocess = PipelineParallelized(
            [
                # Light denoise + local contrast + background suppression.
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
            jpg_path = stack_path.with_name(f"{stem}_{i}_overlay.jpg")
            _write_via_temp(
                jpg_path,
                ".jpg",
                lambda p, img=overlay_rgb_uint8: imsave(p, img, check_contrast=False),
            )
            print(f"Saved overlay to {jpg_path}")

            cell_properties_um = segmentation_mask.convert_properties_to_microns(
                pixel_size_um=pixel_size_um
            )
            dataframe = pd.DataFrame.from_dict(cell_properties_um)
            csv_path = stack_path.with_name(f"{stem}_{i}_measurements.csv")
            _write_via_temp(
                csv_path,
                ".csv",
                lambda p, df=dataframe: df.to_csv(p, index=False),
            )
            print(f"Saved cell properties to {csv_path}")

            # Keep masks in memory to emit a single multi-frame TIFF artifact.
            masks.append(mask)
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


@app.function(image=image, volumes={"/s3": s3_mount})
def run_pipeline(
    lif_path_image_name_mapping: dict[str, str],
    m_range: tuple[int, int] | None,
    lambda_range: tuple[int, int] | None,
    channel_names: list[str] | None,
) -> None:
    """Run split + segmentation over all configured LIF/image pairs."""
    for lif_path, image_name in lif_path_image_name_mapping.items():
        try:
            print(f"Splitting LIF: M={m_range}, Lambda={lambda_range}, channels={channel_names}")
            tiff_paths, pixel_size_um = split_lif_file.remote(
                lif_path, image_name, m_range, lambda_range, channel_names
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


@app.local_entrypoint()
def main(detach: bool = False) -> None:
    """Orchestrate LIF splitting and stack segmentation runs.

    For each configured LIF/image pair, this entrypoint first writes split TIFF
    stacks, then dispatches segmentation across stacks with Modal starmap.

    Set `detach=True` to submit orchestration as a background Modal job and
    return immediately.
    """
    lif_path_image_name_mapping: dict[str, str] = {
        "The_Borg/Tara/2026-03-05/Chlorella_WT_Bristol_slide1.lif": (
            "Lambda_SRS_CARS_1800_600_1pt0_Nav1"
        ),
        "The_Borg/Tara/2026-03-05/Chlorella_WT_Bristol_slide2.lif": (
            "Lambda_SRS_CARS_1800_600_1pt0_Nav1"
        ),
        "The_Borg/Tara/2026-03-09/Chlorella_WT_TAP_slide1.lif": "Lambda_1800_600_1pt0_Nav2",
        "The_Borg/Tara/2026-03-09/Chlorella_WT_TAP_slide2.lif": "Lambda_1800_600_1pt0_Nav1",
    }

    m_range = (0, 9)  # least common denominator M range
    lambda_range = None  # None for full lambda range
    channel_names = ["E-CARS"]

    if detach:
        call = run_pipeline.spawn(
            lif_path_image_name_mapping,
            m_range,
            lambda_range,
            channel_names,
        )
        print("Submitted detached pipeline job.")
        print(f"Function call id: {call.object_id}")
        return

    run_pipeline.remote(
        lif_path_image_name_mapping,
        m_range,
        lambda_range,
        channel_names,
    )
