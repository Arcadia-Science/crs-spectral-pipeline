"""Napari animation utilities for visualizing cell tracking across SRS frames.

Creates MP4 animations showing image stacks with cell outline and track overlays,
useful for quality-checking segmentation and tracking results.

Adapted from https://github.com/Arcadia-Science/burt-lancaster
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import napari  # type: ignore[import-not-found]
import numpy as np
from btrack.btypes import Tracklet
from btrack.utils import tracks_to_napari
from napari_animation import Animation  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

DEFAULT_FPS = 10


@dataclass
class NapariAnimator:
    """Create napari animations of tracked cells over image stacks.

    Args:
        foreground: 3D array (T, Y, X) of image intensities.
        tracks: btrack Tracklets to overlay. If empty/None, no tracks are shown.
        outlines: Optional 3D array (T, Y, X) of cell outline images.
        fps: Playback frames per second.
    """

    foreground: np.ndarray
    tracks: list[Tracklet] | None
    outlines: np.ndarray | None = None
    fps: float = DEFAULT_FPS
    wavenumbers: np.ndarray | None = None
    pixel_size_um: float | None = None

    @property
    def num_frames(self) -> int:
        return len(self.foreground)

    @property
    def window_height_px(self) -> int:
        return self.foreground.shape[-2]

    @property
    def window_width_px(self) -> int:
        return self.foreground.shape[-1]

    def __post_init__(self) -> None:
        if not self.tracks:
            logger.warning("No tracks provided — animation will have no track overlays")
            return

        max_frame = max(track.t[-1] for track in self.tracks)
        min_frame = min(track.t[0] for track in self.tracks)

        if min_frame < 0:
            raise ValueError(f"Track contains negative frame index: {min_frame}")
        if max_frame >= self.num_frames:
            raise ValueError(
                f"Track frame index {max_frame} exceeds foreground frame count {self.num_frames}"
            )

    @cached_property
    def napari_tracks(self) -> np.ndarray | None:
        if not self.tracks:
            return None
        _napari_tracks, _properties, _graph = tracks_to_napari(self.tracks)
        return _napari_tracks

    def _configure_scale_bar(self, viewer: napari.Viewer) -> None:
        """Enable a physical scale bar when pixel_size_um is known."""
        if self.pixel_size_um is None:
            return
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = "um"
        viewer.scale_bar.font_size = 12

    def _configure_wavenumber_overlay(self, viewer: napari.Viewer) -> None:
        """Set up a text overlay that updates with the current wavenumber."""
        if self.wavenumbers is None:
            return

        wavenumbers = self.wavenumbers

        def _update_text(event: object) -> None:
            frame_idx = int(viewer.dims.current_step[0])
            viewer.text_overlay.text = f"{wavenumbers[frame_idx]:.0f} cm\u207b\u00b9"

        viewer.text_overlay.visible = True
        viewer.text_overlay.font_size = 12
        viewer.text_overlay.position = "top_left"
        viewer.text_overlay.color = "white"
        viewer.dims.events.current_step.connect(_update_text)
        _update_text(None)

    def save_animation(self, mp4_path: Path) -> None:
        """Create and save a napari animation to an MP4 file."""
        viewer = napari.Viewer(show=True)
        try:
            scale = (1, self.pixel_size_um, self.pixel_size_um) if self.pixel_size_um else None

            viewer.add_image(self.foreground, name="SRS", scale=scale)

            if self.outlines is not None:
                viewer.add_image(
                    self.outlines,
                    name="cell outlines",
                    blending="additive",
                    opacity=0.7,
                    colormap="green",
                    scale=scale,
                )

            if self.napari_tracks is not None:
                viewer.add_tracks(self.napari_tracks, name="tracks", scale=scale)

            self._configure_scale_bar(viewer)
            self._configure_wavenumber_overlay(viewer)

            # Resize the window so the canvas matches the image pixel dimensions.
            viewer.window.resize(self.window_width_px, self.window_height_px)
            initial_canvas = viewer._canvas_size[::-1]
            width_diff = self.window_width_px - initial_canvas[0]
            height_diff = self.window_height_px - initial_canvas[1]
            viewer.window.resize(
                self.window_width_px + width_diff,
                self.window_height_px + height_diff,
            )

            viewer.reset_view(margin=0)

            animation = Animation(viewer)
            current_step = viewer.dims.current_step

            viewer.dims.current_step = (0, *current_step[1:])
            animation.capture_keyframe()

            viewer.dims.current_step = (self.num_frames - 1, *current_step[1:])
            animation.capture_keyframe(steps=self.num_frames)

            animation.animate(mp4_path, fps=round(self.fps), canvas_only=True)
        finally:
            viewer.close()
