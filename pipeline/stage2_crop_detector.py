"""Pure pixel-domain crop detection for Stage 2.

This module only proposes crop rectangles.  It never writes image pixels or
talks to Siril; the stage runner remains responsible for applying a candidate
through the normal guarded crop command.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


CropRect = Tuple[int, int, int, int]


@dataclass(frozen=True)
class NativeCropDetection:
    """Result of the first-party contour crop detector."""

    rect: Optional[CropRect]
    accepted: bool
    reason: str
    method: str = "native_contour"
    evidence: Optional[Dict[str, Any]] = None

    def as_report(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "method": self.method,
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "candidate": (
                {
                    "x": int(self.rect[0]),
                    "y": int(self.rect[1]),
                    "width": int(self.rect[2]),
                    "height": int(self.rect[3]),
                }
                if self.rect is not None
                else None
            ),
        }
        if self.evidence:
            report["evidence"] = dict(self.evidence)
        return report


def _as_rgb_float(image: np.ndarray) -> np.ndarray:
    """Return an HWC RGB float32 working copy without changing source pixels."""

    source = np.asarray(image)
    if source.ndim == 2:
        source = source[:, :, None]
    elif source.ndim != 3:
        raise ValueError("Stage 2 contour detector expects a 2D or 3D image")

    if source.shape[0] in (1, 3, 4) and source.shape[-1] not in (1, 3, 4):
        source = np.moveaxis(source, 0, -1)
    if source.shape[-1] == 1:
        source = np.repeat(source, 3, axis=-1)
    elif source.shape[-1] >= 3:
        source = source[:, :, :3]
    else:
        raise ValueError("Stage 2 contour detector cannot determine image channels")

    if np.issubdtype(source.dtype, np.integer):
        scale = float(np.iinfo(source.dtype).max)
        return source.astype(np.float32) / max(scale, 1.0)
    return source.astype(np.float32, copy=False)


def _gray_image(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    ).astype(np.float32)


def _background_level(gray: np.ndarray, rect: Optional[CropRect] = None) -> float:
    height, width = gray.shape
    if rect is None:
        x, y, rect_width, rect_height = 0, 0, width, height
    else:
        x, y, rect_width, rect_height = rect
    x0 = max(0, x + int(rect_width * 0.25))
    x1 = min(width, x + max(int(rect_width * 0.75), int(rect_width * 0.25) + 1))
    y0 = max(0, y + int(rect_height * 0.25))
    y1 = min(height, y + max(int(rect_height * 0.75), int(rect_height * 0.25) + 1))
    sample = gray[y0:y1, x0:x1]
    finite = sample[np.isfinite(sample)]
    if finite.size < 64:
        finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return 0.0
    lower = finite[finite <= float(np.quantile(finite, 0.65))]
    if lower.size >= 64:
        finite = lower
    return max(0.0, float(np.median(finite)))


def _mask_resize_factor(width: int, height: int) -> float:
    minimum = min(width, height)
    if minimum >= 4000:
        return 0.10
    if minimum >= 2500:
        return 0.20
    if minimum >= 1500:
        return 0.50
    return 1.0


def largest_rectangle_in_mask(mask: np.ndarray) -> Optional[CropRect]:
    """Find the deterministic largest all-valid rectangle in a binary mask."""

    valid = np.asarray(mask, dtype=bool)
    if valid.ndim != 2 or valid.size == 0 or not np.any(valid):
        return None
    height, width = valid.shape
    heights = np.zeros(width, dtype=np.int32)
    best: Optional[CropRect] = None
    best_area = 0

    for row in range(height):
        heights = np.where(valid[row], heights + 1, 0)
        stack: list[Tuple[int, int]] = []
        for column in range(width + 1):
            current_height = int(heights[column]) if column < width else 0
            start = column
            while stack and stack[-1][1] > current_height:
                start_index, bar_height = stack.pop()
                candidate_width = column - start_index
                area = bar_height * candidate_width
                candidate = (
                    start_index,
                    row - bar_height + 1,
                    candidate_width,
                    bar_height,
                )
                if area > best_area or (
                    area == best_area and best is not None and candidate < best
                ):
                    best_area = area
                    best = candidate
                start = start_index
            if not stack or stack[-1][1] < current_height:
                stack.append((start, current_height))
    return best


def _scalar_lir_rect(values: Any) -> Optional[CropRect]:
    try:
        flattened = np.asarray(values).reshape(-1)
        if flattened.size < 4:
            return None
        rect = tuple(int(round(float(flattened[index]))) for index in range(4))
    except (TypeError, ValueError, OverflowError):
        return None
    if rect[2] <= 0 or rect[3] <= 0:
        return None
    return rect  # type: ignore[return-value]


def _scale_rect_inward(
    rect: CropRect,
    scale: float,
    width: int,
    height: int,
) -> Optional[CropRect]:
    x, y, rect_width, rect_height = rect
    left = max(0, int(math.ceil(x / scale)))
    top = max(0, int(math.ceil(y / scale)))
    right = min(width, int(math.floor((x + rect_width) / scale)))
    bottom = min(height, int(math.floor((y + rect_height) / scale)))

    # Keep CFA-safe coordinates for any later consumer without expanding the
    # candidate into pixels the detector did not mark as valid.
    if left % 2:
        left += 1
    if top % 2:
        top += 1
    if (right - left) % 2:
        right -= 1
    if (bottom - top) % 2:
        bottom -= 1
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


def crop_black_boundary_evidence(
    image: np.ndarray,
    rect: CropRect,
) -> Tuple[bool, Dict[str, Any]]:
    """Require fill/near-black evidence in every boundary a crop removes."""

    rgb = _as_rgb_float(image)
    gray = _gray_image(rgb)
    height, width = gray.shape
    x, y, rect_width, rect_height = (int(value) for value in rect)
    right = x + rect_width
    bottom = y + rect_height
    if x < 0 or y < 0 or right > width or bottom > height:
        return False, {"accepted": False, "reason": "candidate_out_of_bounds"}
    if x == 0 and y == 0 and right == width and bottom == height:
        return False, {"accepted": False, "reason": "candidate_keeps_full_frame"}

    background = _background_level(gray, rect)
    near_black_threshold = max(1.0e-7, min(0.02, background * 0.30))
    fill_threshold = max(1.0e-8, min(0.005, background * 0.08))
    finite = np.isfinite(gray)
    near_black = (~finite) | (gray <= near_black_threshold)
    fill_like = (~finite) | (gray <= fill_threshold)
    removed = np.ones((height, width), dtype=bool)
    removed[y:bottom, x:right] = False

    side_masks: Dict[str, np.ndarray] = {}
    if x > 0:
        side_masks["left"] = np.s_[:, :x]
    if right < width:
        side_masks["right"] = np.s_[:, right:]
    if y > 0:
        side_masks["top"] = np.s_[:y, x:right]
    if bottom < height:
        side_masks["bottom"] = np.s_[bottom:, x:right]

    side_bad_ratios: Dict[str, float] = {}
    side_fill_ratios: Dict[str, float] = {}
    for name, index in side_masks.items():
        side_bad_ratios[name] = float(np.mean(near_black[index]))
        side_fill_ratios[name] = float(np.mean(fill_like[index]))

    removed_count = int(np.count_nonzero(removed))
    overall_bad_ratio = (
        float(np.count_nonzero(near_black & removed) / removed_count)
        if removed_count
        else 0.0
    )
    overall_fill_ratio = (
        float(np.count_nonzero(fill_like & removed) / removed_count)
        if removed_count
        else 0.0
    )
    minimum_side_bad_ratio = min(side_bad_ratios.values(), default=0.0)
    minimum_side_fill_ratio = min(side_fill_ratios.values(), default=0.0)
    accepted = bool(
        side_bad_ratios
        and overall_bad_ratio >= 0.55
        and minimum_side_bad_ratio >= 0.35
        and overall_fill_ratio >= 0.35
        and minimum_side_fill_ratio >= 0.15
    )
    evidence: Dict[str, Any] = {
        "accepted": accepted,
        "background_level": background,
        "near_black_threshold": near_black_threshold,
        "fill_threshold": fill_threshold,
        "removed_pixel_count": removed_count,
        "removed_near_black_ratio": overall_bad_ratio,
        "removed_fill_ratio": overall_fill_ratio,
        "side_near_black_ratio": side_bad_ratios,
        "side_fill_ratio": side_fill_ratios,
        "reason": "near_black_boundary_confirmed" if accepted else "insufficient_near_black_boundary_evidence",
    }
    return accepted, evidence


def _robust_difference_noise(values: np.ndarray) -> float:
    horizontal = values[:, 1:] - values[:, :-1]
    vertical = values[1:, :] - values[:-1, :]
    differences = np.concatenate((horizontal.reshape(-1), vertical.reshape(-1)))
    differences = differences[np.isfinite(differences)]
    if differences.size < 64:
        return 0.0
    median = float(np.median(differences))
    mad = float(np.median(np.abs(differences - median)))
    return mad / (0.6745 * math.sqrt(2.0))


def _edge_connected_mask(hard: np.ndarray, grow: np.ndarray) -> np.ndarray:
    rows, columns = hard.shape
    connected = np.zeros_like(hard, dtype=bool)
    stack: list[Tuple[int, int]] = []
    for row in range(rows):
        for column in (0, columns - 1):
            if hard[row, column] and not connected[row, column]:
                connected[row, column] = True
                stack.append((row, column))
    for column in range(columns):
        for row in (0, rows - 1):
            if hard[row, column] and not connected[row, column]:
                connected[row, column] = True
                stack.append((row, column))
    while stack:
        row, column = stack.pop()
        for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate_row = row + row_delta
            candidate_column = column + column_delta
            if (
                0 <= candidate_row < rows
                and 0 <= candidate_column < columns
                and grow[candidate_row, candidate_column]
                and not connected[candidate_row, candidate_column]
            ):
                connected[candidate_row, candidate_column] = True
                stack.append((candidate_row, candidate_column))
    return connected


def detect_field_rotation_crop(
    image: np.ndarray,
    *,
    noise_ratio_min: float = 1.35,
    chroma_ratio_min: float = 1.20,
) -> NativeCropDetection:
    """Detect edge-connected low-coverage wedges from noise/chroma evidence.

    Alt-azimuth field rotation can leave every output pixel non-zero while the
    outer wedges contain far fewer contributing frames.  The resulting local
    luminance and chroma noise rises sharply.  Requiring both signals and an
    edge-connected component keeps stars, nebulosity and isolated noisy tiles
    from authorizing a crop.
    """

    try:
        rgb = _as_rgb_float(image)
    except ValueError as error:
        return NativeCropDetection(
            None,
            False,
            str(error),
            method="native_field_rotation",
        )
    height, width, _channels = rgb.shape
    if min(width, height) < 192:
        return NativeCropDetection(
            None,
            False,
            "image_too_small_for_coverage_grid",
            method="native_field_rotation",
        )

    gray = _gray_image(rgb)
    tile_size = max(24, int(round(min(width, height) / 36.0)))
    tile_size = max(8, (tile_size // 4) * 4)
    rows = int(round(height / float(tile_size)))
    columns = int(round(width / float(tile_size)))
    if rows < 12 or columns < 12:
        return NativeCropDetection(
            None,
            False,
            "coverage_grid_too_small",
            method="native_field_rotation",
        )

    # Equal-width edges include the real outermost pixels even when image
    # dimensions are not exact multiples of the nominal tile size.
    y_edges = np.linspace(0, height, rows + 1, dtype=np.int32)
    x_edges = np.linspace(0, width, columns + 1, dtype=np.int32)
    noise = np.zeros((rows, columns), dtype=np.float32)
    chroma = np.zeros_like(noise)
    for row in range(rows):
        y0 = int(y_edges[row])
        y1 = int(y_edges[row + 1])
        for column in range(columns):
            x0 = int(x_edges[column])
            x1 = int(x_edges[column + 1])
            gray_tile = gray[y0:y1, x0:x1]
            rgb_tile = rgb[y0:y1, x0:x1, :]
            noise[row, column] = _robust_difference_noise(gray_tile)
            red_green = rgb_tile[:, :, 0] - rgb_tile[:, :, 1]
            blue_green = rgb_tile[:, :, 2] - rgb_tile[:, :, 1]
            color_noise = np.sqrt(red_green * red_green + blue_green * blue_green)
            finite_color = color_noise[np.isfinite(color_noise)]
            chroma[row, column] = (
                float(np.median(finite_color)) if finite_color.size else 0.0
            )

    reference_y = max(2, rows // 4)
    reference_x = max(2, columns // 4)
    reference = (
        slice(reference_y, rows - reference_y),
        slice(reference_x, columns - reference_x),
    )
    reference_noise = float(np.median(noise[reference]))
    reference_chroma = float(np.median(chroma[reference]))
    if reference_noise <= 1.0e-8 or reference_chroma <= 1.0e-8:
        return NativeCropDetection(
            None,
            False,
            "coverage_reference_unavailable",
            method="native_field_rotation",
            evidence={
                "reference_noise": reference_noise,
                "reference_chroma": reference_chroma,
            },
        )

    noise_ratio_min = max(1.15, min(3.0, float(noise_ratio_min)))
    chroma_ratio_min = max(1.10, min(3.0, float(chroma_ratio_min)))
    grow_noise_ratio = 1.0 + (noise_ratio_min - 1.0) * 0.34
    grow_chroma_ratio = 1.0 + (chroma_ratio_min - 1.0) * 0.40
    noise_ratio = noise / reference_noise
    chroma_ratio = chroma / reference_chroma
    hard = (noise_ratio >= noise_ratio_min) & (
        chroma_ratio >= chroma_ratio_min
    )
    grow = (noise_ratio >= grow_noise_ratio) & (
        chroma_ratio >= grow_chroma_ratio
    )
    edge_connected = _edge_connected_mask(hard, grow)
    connected_count = int(np.count_nonzero(edge_connected))
    minimum_connected = max(4, int(math.ceil(rows * columns * 0.01)))

    boundary_noise = np.concatenate(
        (noise_ratio[0], noise_ratio[-1], noise_ratio[:, 0], noise_ratio[:, -1])
    )
    boundary_chroma = np.concatenate(
        (
            chroma_ratio[0],
            chroma_ratio[-1],
            chroma_ratio[:, 0],
            chroma_ratio[:, -1],
        )
    )
    evidence: Dict[str, Any] = {
        "tile_size": tile_size,
        "grid_rows": rows,
        "grid_columns": columns,
        "reference_noise": reference_noise,
        "reference_chroma": reference_chroma,
        "noise_ratio_min": noise_ratio_min,
        "chroma_ratio_min": chroma_ratio_min,
        "grow_noise_ratio": grow_noise_ratio,
        "grow_chroma_ratio": grow_chroma_ratio,
        "hard_tile_count": int(np.count_nonzero(hard)),
        "edge_connected_tile_count": connected_count,
        "edge_connected_ratio": connected_count / float(rows * columns),
        "boundary_noise_ratio_p90": float(np.quantile(boundary_noise, 0.90)),
        "boundary_chroma_ratio_p90": float(np.quantile(boundary_chroma, 0.90)),
        "boundary_noise_ratio_max": float(np.max(boundary_noise)),
        "boundary_chroma_ratio_max": float(np.max(boundary_chroma)),
    }
    if connected_count < minimum_connected:
        evidence["minimum_edge_connected_tile_count"] = minimum_connected
        return NativeCropDetection(
            None,
            False,
            "no_significant_edge_connected_coverage_anomaly",
            method="native_field_rotation",
            evidence=evidence,
        )

    tile_rect = largest_rectangle_in_mask(~edge_connected)
    if tile_rect is None:
        return NativeCropDetection(
            None,
            False,
            "no_coverage_interior_rectangle",
            method="native_field_rotation",
            evidence=evidence,
        )
    tile_x, tile_y, tile_width, tile_height = tile_rect
    left = int(x_edges[tile_x])
    top = int(y_edges[tile_y])
    right = int(x_edges[tile_x + tile_width])
    bottom = int(y_edges[tile_y + tile_height])
    rect = _scale_rect_inward(
        (
            left,
            top,
            right - left,
            bottom - top,
        ),
        1.0,
        width,
        height,
    )
    if rect is None:
        return NativeCropDetection(
            None,
            False,
            "invalid_coverage_rectangle",
            method="native_field_rotation",
            evidence=evidence,
        )
    x, y, rect_width, rect_height = rect
    retained_ratio = (rect_width * rect_height) / float(width * height)
    evidence["candidate_retained_ratio"] = retained_ratio
    evidence["rectangle_source"] = "edge_connected_coverage_mask"
    return NativeCropDetection(
        rect,
        True,
        "edge_connected_field_rotation_confirmed",
        method="native_field_rotation",
        evidence=evidence,
    )


def detect_native_contour_crop(
    image: np.ndarray,
    *,
    cv2_module: Any = None,
    lir_module: Any = None,
) -> NativeCropDetection:
    """Propose a safe contour/LIR crop for a canonical image array."""

    try:
        rgb = _as_rgb_float(image)
    except ValueError as error:
        return NativeCropDetection(None, False, str(error), evidence={})
    height, width, _channels = rgb.shape
    if min(width, height) < 32:
        return NativeCropDetection(None, False, "image_too_small", evidence={})

    if cv2_module is None:
        try:
            import cv2 as cv2_module  # type: ignore[no-redef]
        except (ImportError, OSError) as error:
            return NativeCropDetection(
                None,
                False,
                "opencv_unavailable",
                evidence={"dependency_error": str(error)},
            )
    if lir_module is None:
        try:
            import largestinteriorrectangle as lir_module  # type: ignore[no-redef]
        except (ImportError, OSError):
            lir_module = None

    gray = _gray_image(rgb)
    background = _background_level(gray)
    threshold = max(1.0e-7, min(0.02, background * 0.30))
    valid = (np.isfinite(gray) & (gray > threshold)).astype(np.uint8) * 255
    scale = _mask_resize_factor(width, height)
    try:
        kernel = np.ones((5, 5), dtype=np.uint8)
        valid = cv2_module.morphologyEx(
            valid,
            cv2_module.MORPH_CLOSE,
            kernel,
        )
        if scale < 1.0:
            valid = cv2_module.resize(
                valid,
                (0, 0),
                fx=scale,
                fy=scale,
                interpolation=cv2_module.INTER_NEAREST,
            )
        contours, _hierarchy = cv2_module.findContours(
            valid,
            cv2_module.RETR_EXTERNAL,
            cv2_module.CHAIN_APPROX_SIMPLE,
        )
    except Exception as error:  # Optional native dependency boundary.
        return NativeCropDetection(
            None,
            False,
            "opencv_detection_failed",
            evidence={"dependency_error": str(error)},
        )
    if not contours:
        return NativeCropDetection(
            None,
            False,
            "no_valid_footprint_contour",
            evidence={"background_level": background, "mask_threshold": threshold},
        )

    try:
        contour = max(contours, key=cv2_module.contourArea)
        contour_mask = np.zeros_like(valid, dtype=np.uint8)
        cv2_module.drawContours(
            contour_mask,
            [contour],
            -1,
            1,
            thickness=cv2_module.FILLED,
        )
    except Exception as error:  # Optional native dependency boundary.
        return NativeCropDetection(
            None,
            False,
            "opencv_contour_failed",
            evidence={"dependency_error": str(error)},
        )
    footprint_ratio = float(np.mean(contour_mask > 0))
    if footprint_ratio < 0.25:
        return NativeCropDetection(
            None,
            False,
            "valid_footprint_too_small",
            evidence={
                "background_level": background,
                "mask_threshold": threshold,
                "footprint_ratio": footprint_ratio,
            },
        )

    rectangle_source = "largestinteriorrectangle"
    mask_rect: Optional[CropRect] = None
    if lir_module is not None:
        try:
            contour_array = np.asarray([contour[:, 0, :]])
            mask_rect = _scalar_lir_rect(lir_module.lir(contour_array))
        except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
            mask_rect = None
    if mask_rect is not None:
        x, y, rect_width, rect_height = mask_rect
        inside = contour_mask[y : y + rect_height, x : x + rect_width]
        if (
            inside.shape != (rect_height, rect_width)
            or inside.size == 0
            or float(np.mean(inside > 0)) < 0.995
        ):
            mask_rect = None
    if mask_rect is None:
        rectangle_source = "binary_mask_fallback"
        mask_rect = largest_rectangle_in_mask(contour_mask)
    if mask_rect is None:
        return NativeCropDetection(
            None,
            False,
            "no_interior_rectangle",
            evidence={
                "background_level": background,
                "mask_threshold": threshold,
                "footprint_ratio": footprint_ratio,
                "rectangle_source": rectangle_source,
            },
        )

    mask_x, mask_y, mask_width, mask_height = mask_rect
    if (
        mask_x == 0
        and mask_y == 0
        and mask_width == valid.shape[1]
        and mask_height == valid.shape[0]
    ):
        return NativeCropDetection(
            None,
            False,
            "full_frame_is_valid",
            evidence={
                "background_level": background,
                "mask_threshold": threshold,
                "footprint_ratio": footprint_ratio,
                "rectangle_source": rectangle_source,
            },
        )

    rect = _scale_rect_inward(mask_rect, scale, width, height)
    if rect is None:
        return NativeCropDetection(None, False, "invalid_scaled_rectangle")
    x, y, rect_width, rect_height = rect
    if x == 0 and y == 0 and rect_width == width and rect_height == height:
        return NativeCropDetection(
            None,
            False,
            "full_frame_is_valid",
            evidence={
                "background_level": background,
                "mask_threshold": threshold,
                "footprint_ratio": footprint_ratio,
                "rectangle_source": rectangle_source,
            },
        )

    accepted, boundary_evidence = crop_black_boundary_evidence(rgb, rect)
    evidence = {
        "background_level": background,
        "mask_threshold": threshold,
        "footprint_ratio": footprint_ratio,
        "rectangle_source": rectangle_source,
        **boundary_evidence,
    }
    if not accepted:
        return NativeCropDetection(
            rect,
            False,
            "insufficient_near_black_boundary_evidence",
            evidence=evidence,
        )
    return NativeCropDetection(
        rect,
        True,
        "near_black_boundary_confirmed",
        evidence=evidence,
    )


__all__ = [
    "CropRect",
    "NativeCropDetection",
    "crop_black_boundary_evidence",
    "detect_field_rotation_crop",
    "detect_native_contour_crop",
    "largest_rectangle_in_mask",
]
