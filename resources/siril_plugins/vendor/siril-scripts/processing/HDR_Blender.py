# (c) Adrian Knagg-Baugh 2025
# Inverse Variance Maximum Likelihood Estimate (MLE) HDR Blender
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This script blends multiple images with different exposure lengths.
The images must be registered, and to enable an optimum statistical
blend the FITS header card LIVETIME or EXPTIME must be present (if
the images have been stacked in Siril, this will be true).

(Note that this script does not do tone mapping - it does not take a HDR
image and generate a tone mapped LDR image. It generates a HDR image
from multiple LDR images (if astrophotographic images can ever be called
LDR!) of different exposure lengths.

The use case is when dealing with structures with extreme dynamic range,
such as M42, where setting the exposure low enough to avoid saturating
Iota Orionis results in only a few ADU to capture the nebulosity, so it
is desirable to blend short exposures of the bright stars and the core
and Trapezium area with longer exposures of the nebulosity.

The script uses an inverse variance maximum likelihood model. It builds
a model of the read noise and shot noise using the bgnoise and median
values compute by Siril for each image, assigning weight to the
likelihood that the pixel in each image represents the true pixel value.
If a pixel exceeds the saturation level it is given a zero weight: it
is possible to set a saturation margin below the saturation level within
which the weight will taper off smoothly (cosine taper) in order to
account for sensor nonlinearity approaching saturation: if the
saturation margin is set to 0, no taper will be applied and there will
be a hard cutoff at the saturation point.

Images are converted from flux (total ADU) to radiance (ADU / s) for
blending purposes and the images are blended according to the weight
maps. Pixels where zero weight contributes from any input image (e.g.
bright star cores that are saturated in all inputs) are assigned the
saturation radiance value.

By default the output is rescaled to the Siril float range [0,1] but
it is possible to save raw radiance values if desired: in this case the
pixel values will represent radiance in ADU / s. It is optionally
possible to save the weight maps for each image, normalized so that
the sum of the weight maps adds up to a unity image, for inspection.
"""

# Version Log
VERSION = "1.0.2"
# 1.0.0: AKB - initial release
# 1.0.1: Added low memory mode and exposure calibration
# 1.0.2: Added psutil in dep

import sys
import os
import json
from pathlib import Path

import sirilpy as s
s.ensure_installed("psutil", "PyQt6")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QGroupBox, QLabel,
    QDoubleSpinBox, QSpinBox, QCheckBox, QFileDialog, QMessageBox,
    QProgressBar, QTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
import numpy as np
import psutil

def get_memory_info():
    """Get available system memory in bytes"""
    memory = psutil.virtual_memory()
    return memory.available

def estimate_memory_usage(image_shapes, num_images, return_weights=False):
    """Estimate memory usage for HDR processing in bytes"""
    if not image_shapes:
        return 0

    # Assume all images have the same shape (validated elsewhere)
    sample_shape = image_shapes[0]
    pixels_per_image = np.prod(sample_shape)

    # Memory usage breakdown:
    # 1. Input images stack (float32): num_images * pixels * 4 bytes
    # 2. Radiance values (float32): pixels * 4 bytes
    # 3. Variance values (float32): pixels * 4 bytes
    # 4. Weights (float32): num_images * pixels * 4 bytes
    # 5. Working arrays (temp calculations): ~2x the above for safety

    base_usage = (
        num_images * pixels_per_image * 4 +  # images stack
        pixels_per_image * 4 +               # radiance
        pixels_per_image * 4 +               # variance
        num_images * pixels_per_image * 4    # weights
    )

    if return_weights:
        base_usage += num_images * pixels_per_image * 4  # normalized weights output

    # Add safety factor for temporary arrays during computation
    total_usage = base_usage * 2

    return total_usage

def calibrate_exposure_matching(images, exptimes, overlap_percentile=70):
    """
    Match exposures using rings around saturated areas in the longest exposure.
    This approach specifically targets the boundary regions where discontinuities occur.
    """
    import scipy.ndimage as ndimage
    from scipy.ndimage import binary_dilation, binary_erosion

    calibrated_images = []
    calibration_factors = []

    # Find the longest exposure (reference image)
    ref_idx = np.argmax(exptimes)
    ref_image = images[ref_idx]
    ref_exptime = exptimes[ref_idx]

    print(f"Using image {ref_idx+1} (exposure {ref_exptime:.2f}s) as reference")

    # Auto-detect saturation level for reference image
    ref_max = np.max(ref_image)
    if ref_max <= 1.0:
        sat_threshold = 0.95
    elif ref_max <= 65535.0:
        if ref_max > 32000:
            sat_threshold = 60000.0
        elif ref_max > 16000:
            sat_threshold = 15000.0
        else:
            sat_threshold = ref_max * 0.9
    else:
        sat_threshold = 60000.0

    print(f"Using saturation threshold: {sat_threshold:.1f}")

    # Create saturation mask for reference image
    saturated_mask = ref_image >= sat_threshold

    if not np.any(saturated_mask):
        print("No saturated pixels found in reference image - using simple overlap method")
        # Fallback to simple overlap method
        for i, (img, exptime) in enumerate(zip(images, exptimes)):
            if i == ref_idx:
                calibrated_images.append(img)
                calibration_factors.append(1.0)
                continue

            # Use non-saturated pixels for calibration
            ref_radiance = ref_image / ref_exptime
            curr_radiance = img / exptime

            # Use pixels in middle intensity range
            ref_low = np.percentile(ref_image, 20)
            ref_high = np.percentile(ref_image, 80)
            curr_low = np.percentile(img, 20)
            curr_high = np.percentile(img, 80)

            overlap_mask = (ref_image > ref_low) & (ref_image < ref_high) & \
                          (img > curr_low) & (img < curr_high)

            if np.sum(overlap_mask) > 100:
                valid_ratios = ref_radiance[overlap_mask] / curr_radiance[overlap_mask]
                valid_ratios = valid_ratios[np.isfinite(valid_ratios) & (valid_ratios > 0)]

                if len(valid_ratios) > 50:
                    ratio = np.median(valid_ratios)
                    calibrated_images.append(img * ratio)
                    calibration_factors.append(ratio)
                else:
                    calibrated_images.append(img)
                    calibration_factors.append(1.0)
            else:
                calibrated_images.append(img)
                calibration_factors.append(1.0)

        return calibrated_images, calibration_factors

    # Create ring structure around saturated areas
    # Ring 1: pixels immediately adjacent to saturated areas
    # Ring 2: pixels 2-4 pixels away from saturated areas
    # Ring 3: pixels 5-8 pixels away from saturated areas

    # Dilate saturated mask to create rings
    struct = np.ones((3, 3))  # 8-connected structure

    # Ring 1: 1-pixel dilation minus original
    dilated_1 = binary_dilation(saturated_mask, structure=struct, iterations=1)
    ring_1 = dilated_1 & ~saturated_mask

    # Ring 2: 2-4 pixel dilation
    dilated_4 = binary_dilation(saturated_mask, structure=struct, iterations=4)
    ring_2 = dilated_4 & ~dilated_1

    # Ring 3: 5-8 pixel dilation
    dilated_8 = binary_dilation(saturated_mask, structure=struct, iterations=8)
    ring_3 = dilated_8 & ~dilated_4

    print(f"Ring pixels - Ring 1: {np.sum(ring_1)}, Ring 2: {np.sum(ring_2)}, Ring 3: {np.sum(ring_3)}")

    # Process each image
    for i, (img, exptime) in enumerate(zip(images, exptimes)):
        if i == ref_idx:
            # Reference image - no calibration needed
            calibrated_images.append(img)
            calibration_factors.append(1.0)
            continue

        # Convert to radiance
        ref_radiance = ref_image / ref_exptime
        curr_radiance = img / exptime

        # Try to find calibration factor using rings, starting with the innermost
        ratio = None
        ring_used = None

        for ring_num, ring_mask in enumerate([ring_1, ring_2, ring_3], 1):
            if not np.any(ring_mask):
                continue

            # Additional filtering: avoid pixels that are saturated in current image
            curr_max = np.max(img)
            if curr_max <= 1.0:
                curr_sat_threshold = 0.95
            elif curr_max <= 65535.0:
                curr_sat_threshold = curr_max * 0.9
            else:
                curr_sat_threshold = 60000.0

            # Also avoid very dim pixels (noise floor)
            ref_min_threshold = np.percentile(ref_image[ring_mask], 10)
            curr_min_threshold = np.percentile(img[ring_mask], 10)

            # Valid pixels: in ring, not saturated in either image, above noise floor
            valid_mask = ring_mask & \
                        (ref_image < sat_threshold) & \
                        (img < curr_sat_threshold) & \
                        (ref_image > ref_min_threshold) & \
                        (img > curr_min_threshold)

            if np.sum(valid_mask) < 50:  # Need sufficient pixels
                continue

            # Calculate ratios
            ref_values = ref_radiance[valid_mask]
            curr_values = curr_radiance[valid_mask]

            # Remove outliers and invalid ratios
            ratios = ref_values / curr_values
            valid_ratios = ratios[np.isfinite(ratios) & (ratios > 0) & (ratios < 10)]

            if len(valid_ratios) < 30:
                continue

            # Use robust statistics
            # Remove extreme outliers (beyond 2.5 sigma from median)
            median_ratio = np.median(valid_ratios)
            mad = np.median(np.abs(valid_ratios - median_ratio))
            robust_std = 1.4826 * mad  # Convert MAD to approximate std

            if robust_std > 0:
                outlier_mask = np.abs(valid_ratios - median_ratio) < (2.5 * robust_std)
                if np.sum(outlier_mask) > 20:
                    valid_ratios = valid_ratios[outlier_mask]

            # Final ratio using trimmed mean (remove top and bottom 10%)
            if len(valid_ratios) > 20:
                sorted_ratios = np.sort(valid_ratios)
                trim_count = max(1, len(sorted_ratios) // 10)
                trimmed_ratios = sorted_ratios[trim_count:-trim_count]
                ratio = np.mean(trimmed_ratios)
                ring_used = ring_num
                break

        # Apply calibration or fallback
        if ratio is not None:
            calibrated_img = img * ratio
            calibrated_images.append(calibrated_img)
            calibration_factors.append(ratio)
            print(f"Calibrated image {i+1} with factor {ratio:.3f} using ring {ring_used}")
        else:
            # Fallback: use global overlap method
            print(f"Ring calibration failed for image {i+1}, using global method")

            # Find non-saturated overlap region
            overlap_mask = (ref_image < sat_threshold * 0.8) & \
                          (img < np.max(img) * 0.8) & \
                          (ref_image > np.percentile(ref_image, 10)) & \
                          (img > np.percentile(img, 10))

            if np.sum(overlap_mask) > 100:
                valid_ratios = ref_radiance[overlap_mask] / curr_radiance[overlap_mask]
                valid_ratios = valid_ratios[np.isfinite(valid_ratios) & (valid_ratios > 0)]

                if len(valid_ratios) > 50:
                    ratio = np.median(valid_ratios)
                    calibrated_images.append(img * ratio)
                    calibration_factors.append(ratio)
                    print(f"Fallback calibration for image {i+1} with factor {ratio:.3f}")
                else:
                    calibrated_images.append(img)
                    calibration_factors.append(1.0)
                    print(f"No calibration applied to image {i+1}")
            else:
                calibrated_images.append(img)
                calibration_factors.append(1.0)
                print(f"No calibration applied to image {i+1}")

    return calibrated_images, calibration_factors

def hdr_merge_inverse_variance(
    images,
    exptimes,
    bg_noises=None,
    bg_medians=None,
    sat_level=None,
    sat_margin=0.0,
    gain=1.0,
    eps=1e-12,
    return_weights=False,
    calibrate_exposures=True,
):
    # Apply exposure calibration if requested
    if calibrate_exposures and len(images) > 1:
        images, calibration_factors = calibrate_exposure_matching(images, exptimes)

    imgs_stack = np.stack(images, axis=0).astype(np.float32)
    exptimes = np.array(exptimes, dtype=np.float32)
    exptime_expand = (slice(None),) + (None,) * (imgs_stack.ndim - 1)
    r_hat = imgs_stack / exptimes[exptime_expand]
    def align_bg(bg_arr, imgs_ndim):
        arr = np.array(bg_arr, dtype=np.float32)
        if imgs_ndim == 4:
            if arr.ndim == 1:
                return arr[:, None, None, None]
            if arr.ndim == 2:
                return arr[:, :, None, None]
            return arr.reshape((arr.shape[0], -1, 1, 1))
        if imgs_ndim == 3:
            if arr.ndim == 1:
                return arr[:, None, None]
            return arr.reshape((arr.shape[0], 1, 1))
        return arr
    if bg_noises is not None and bg_medians is not None:
        bg_noises_aligned = align_bg(bg_noises, imgs_stack.ndim)
        bg_medians_aligned = align_bg(bg_medians, imgs_stack.ndim)
        var_signal = np.maximum(imgs_stack - bg_medians_aligned, 0.0) / gain
        var_read = (bg_noises_aligned ** 2)
        var = var_signal + var_read
    elif bg_noises is not None:
        bg_noises_aligned = align_bg(bg_noises, imgs_stack.ndim)
        var_signal = imgs_stack / gain
        var_read = (bg_noises_aligned ** 2)
        var = var_signal + var_read
    else:
        var = imgs_stack / gain
    var_r = var / (exptimes[exptime_expand] ** 2) + eps
    if sat_level is None:
        R = np.ones_like(imgs_stack, dtype=np.float32)
    else:
        S = imgs_stack
        if sat_margin <= 0.0:
            R = np.where(S >= sat_level, 0.0, 1.0).astype(np.float32)
        else:
            R = np.ones_like(S, dtype=np.float32)
            high = S >= sat_level
            low = S < (sat_level - sat_margin)
            mid = (~high) & (~low)
            R[high] = 0.0
            R[mid] = 0.5 * (1.0 + np.cos(
                np.pi * (S[mid] - (sat_level - sat_margin)) / sat_margin
            ))
    weights = R / var_r
    num = np.sum(weights * r_hat, axis=0)
    den = np.sum(weights, axis=0)
    radiance = np.where(den > 0, num / den, 0.0)
    variance = np.where(den > 0, 1.0 / den, np.inf)

    if sat_level is not None:
        sat_radiance = sat_level / np.max(exptimes)
        radiance = np.where(np.isinf(variance), sat_radiance, radiance)
    if return_weights:
        den_k = np.sum(weights, axis=0, keepdims=True)
        norm_weights = np.where(den_k > 0, weights / den_k, 0.0)
        return (
            radiance.astype(np.float32),
            variance.astype(np.float32),
            norm_weights.astype(np.float32),
        )
    else:
        return radiance.astype(np.float32), variance.astype(np.float32)

def hdr_merge_inverse_variance_low_memory(
    file_paths,
    siril_interface,
    exptimes,
    bg_noises=None,
    bg_medians=None,
    sat_level=None,
    sat_margin=0.0,
    gain=1.0,
    eps=1e-12,
    return_weights=False,
    progress_callback=None,
    calibrate_exposures=True,
):
    """
    Low memory version that processes images one at a time.
    First pass: load images and perform calibration if requested
    Second pass: compute per-pixel weights and accumulate weighted sums.
    Third pass: compute final result and optionally weights.
    """
    num_images = len(file_paths)
    exptimes = np.array(exptimes, dtype=np.float32)

    # Load first image to get dimensions
    first_img = siril_interface.load_image_from_file(file_paths[0])
    sample_shape = first_img.data.shape

    # Step 1: Load images for calibration if requested
    calibration_factors = None
    if calibrate_exposures and num_images > 1:
        if progress_callback:
            progress_callback(5)

        # Load all images for calibration (this is unavoidable for proper calibration)
        temp_images = []
        for i, file_path in enumerate(file_paths):
            img_obj = siril_interface.load_image_from_file(file_path)
            img_data = img_obj.data.astype(np.float32)

            # Handle different data types
            if img_obj.data.dtype == np.uint16:
                temp_images.append(img_data)
            elif np.issubdtype(img_obj.data.dtype, np.floating):
                max_val = np.max(img_data)
                if max_val > 1.0:
                    temp_images.append((img_data / max_val) * 65535.0)
                else:
                    temp_images.append(img_data * 65535.0)

        # Perform calibration
        _, calibration_factors = calibrate_exposure_matching(temp_images, exptimes)
        del temp_images  # Free memory
    else:
        calibration_factors = [1.0] * num_images

    # Initialize accumulators
    weighted_sum = np.zeros(sample_shape, dtype=np.float32)
    weight_sum = np.zeros(sample_shape, dtype=np.float32)

    # Store individual weights if requested (this still uses significant memory)
    individual_weights = None
    if return_weights:
        individual_weights = np.zeros((num_images,) + sample_shape, dtype=np.float32)

    def align_bg(bg_arr, imgs_ndim, img_idx=None):
        """Align background arrays for a single image"""
        if bg_arr is None:
            return None

        if img_idx is not None:
            # Extract single image's background values
            if isinstance(bg_arr[0], list):
                arr = np.array(bg_arr[img_idx], dtype=np.float32)
            else:
                arr = np.array([bg_arr[img_idx]], dtype=np.float32)
        else:
            arr = np.array(bg_arr, dtype=np.float32)

        if imgs_ndim == 3:  # Color image
            if arr.ndim == 1 and len(arr) > 1:
                return arr.reshape((-1, 1, 1))
            else:
                return arr.reshape((1, 1))
        else:  # Grayscale
            return arr

    # First pass: accumulate weighted sums
    for i, file_path in enumerate(file_paths):
        if progress_callback:
            progress_callback(int(10 + (i / num_images) * 40))

        # Load and process single image
        img_obj = siril_interface.load_image_from_file(file_path)
        img_data = img_obj.data.astype(np.float32)

        # Handle different data types (same logic as original)
        if img_obj.data.dtype == np.uint16:
            pass  # already converted above
        elif np.issubdtype(img_obj.data.dtype, np.floating):
            max_val = np.max(img_data)
            if max_val > 1.0:
                img_data = (img_data / max_val) * 65535.0
            else:
                img_data = img_data * 65535.0

        # Apply calibration factor
        img_data = img_data * calibration_factors[i]

        # Convert to radiance
        exptime = exptimes[i]
        r_hat = img_data / exptime

        # Compute variance for this image
        if bg_noises is not None and bg_medians is not None:
            bg_noise = align_bg(bg_noises, img_data.ndim, i)
            bg_median = align_bg(bg_medians, img_data.ndim, i)

            var_signal = np.maximum(img_data - bg_median, 0.0) / gain
            var_read = bg_noise ** 2
            var = var_signal + var_read
        elif bg_noises is not None:
            bg_noise = align_bg(bg_noises, img_data.ndim, i)
            var_signal = img_data / gain
            var_read = bg_noise ** 2
            var = var_signal + var_read
        else:
            var = img_data / gain

        var_r = var / (exptime ** 2) + eps

        # Compute reliability mask
        if sat_level is None:
            R = np.ones_like(img_data, dtype=np.float32)
        else:
            S = img_data
            if sat_margin <= 0.0:
                R = np.where(S >= sat_level, 0.0, 1.0).astype(np.float32)
            else:
                R = np.ones_like(S, dtype=np.float32)
                high = S >= sat_level
                low = S < (sat_level - sat_margin)
                mid = (~high) & (~low)
                R[high] = 0.0
                R[mid] = 0.5 * (1.0 + np.cos(
                    np.pi * (S[mid] - (sat_level - sat_margin)) / sat_margin
                ))

        # Compute weights for this image
        weights = R / var_r

        # Accumulate
        weighted_sum += weights * r_hat
        weight_sum += weights

        # Store individual weights if needed
        if return_weights:
            individual_weights[i] = weights

        # Clean up
        del img_data, r_hat, var, var_r, R, weights

    # Compute final radiance
    radiance = np.where(weight_sum > 0, weighted_sum / weight_sum, 0.0)
    variance = np.where(weight_sum > 0, 1.0 / weight_sum, np.inf)

    # Handle saturation
    if sat_level is not None:
        sat_radiance = sat_level / np.max(exptimes)
        radiance = np.where(np.isinf(variance), sat_radiance, radiance)

    if return_weights:
        # Normalize individual weights
        weight_sum_expanded = np.expand_dims(weight_sum, 0)
        norm_weights = np.where(weight_sum_expanded > 0,
                               individual_weights / weight_sum_expanded, 0.0)
        return (
            radiance.astype(np.float32),
            variance.astype(np.float32),
            norm_weights.astype(np.float32),
        )
    else:
        return radiance.astype(np.float32), variance.astype(np.float32)

def auto_detect_saturation_level(images):
    max_values = [np.max(img) for img in images]
    overall_max = max(max_values)
    if overall_max <= 1.0:
        return 1.0
    elif overall_max <= 65535.0:
        if overall_max > 32000:
            return 65535.0
        elif overall_max > 16000:
            return 16383.0
        elif overall_max > 4000:
            return 4095.0
        else:
            return overall_max * 0.98
    else:
        return 65535.0

def save_weight_maps(weights, current_dir):
    """Save individual weight maps as FITS files"""
    try:
        import astropy.io.fits as fits

        # weights shape is (N, H, W) or (N, C, H, W)
        n_images = weights.shape[0]

        for i in range(n_images):
            weight_map = weights[i]
            filename = os.path.join(current_dir, f"weight_{i+1}.fits")

            # Create FITS HDU
            hdu = fits.PrimaryHDU(weight_map.astype(np.float32))
            hdu.header['COMMENT'] = f'Weight map for image {i+1}'

            # Save the file
            hdul = fits.HDUList([hdu])
            hdul.writeto(filename, overwrite=True)

    except ImportError:
        print("WARNING: astropy not available, cannot save weight maps")
    except Exception as e:
        print(f"ERROR: Failed to save weight maps: {str(e)}")

class HDRProcessingThread(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, files, parameters, siril):
        super().__init__()
        self.files = files
        self.parameters = parameters
        self.siril = siril

    def run(self):
        try:
            # Validate files exist
            for filename in self.files:
                if not os.path.exists(filename):
                    self.finished_signal.emit(False, f"File not found: {filename}")
                    return

            # Load image metadata to determine memory requirements
            self.status.emit("Analyzing memory requirements...")
            image_shapes = []
            exptimes = []
            bg_noises = []
            bg_medians = []
            image_types = []

            total_files = len(self.files)

            for i, filename in enumerate(self.files):
                self.status.emit(f"Analyzing {os.path.basename(filename)}...")
                self.progress.emit(int((i / total_files) * 20))

                try:
                    img = self.siril.load_image_from_file(filename)
                except Exception as e:
                    self.finished_signal.emit(False, f"Failed to load {filename}: {str(e)}")
                    return

                image_shapes.append(img.data.shape)

                # Get exposure time and normalization info
                original_exptime = getattr(img.keywords, 'livetime', None)
                if original_exptime is None:
                    original_exptime = getattr(img.keywords, 'exposure', 1.0)
                if original_exptime is None or original_exptime <= 0:
                    original_exptime = 1.0

                effective_exptime = float(original_exptime)

                # Handle normalization for float images
                if np.issubdtype(img.data.dtype, np.floating):
                    max_val = np.max(img.data)
                    if max_val > 1.0:
                        normalization_factor = max_val
                        effective_exptime = original_exptime * normalization_factor
                        self.status.emit(f"Will normalize image {i+1} (max={max_val:.3f})")

                exptimes.append(effective_exptime)

                # Extract background statistics
                if hasattr(img, 'stats') and img.stats:
                    if img.data.ndim == 3:
                        try:
                            C = img.data.shape[0]
                            noise_vals = []
                            median_vals = []
                            for c in range(C):
                                if c < len(img.stats):
                                    noise_vals.append(img.stats[c].bgnoise)
                                    median_vals.append(img.stats[c].median)
                                else:
                                    noise_vals.append(1.0)
                                    median_vals.append(0.0)
                            bg_noises.append(noise_vals)
                            bg_medians.append(median_vals)
                        except (AttributeError, IndexError):
                            C = img.data.shape[0]
                            bg_noises.append([1.0] * C)
                            bg_medians.append([0.0] * C)
                    else:
                        try:
                            bg_noises.append(img.stats[0].bgnoise)
                            bg_medians.append(img.stats[0].median)
                        except (AttributeError, IndexError):
                            bg_noises.append(1.0)
                            bg_medians.append(0.0)
                else:
                    if img.data.ndim == 3:
                        C = img.data.shape[0]
                        bg_noises.append([1.0] * C)
                        bg_medians.append([0.0] * C)
                    else:
                        bg_noises.append(1.0)
                        bg_medians.append(0.0)

            # Validate dimensions and exposure times
            if len(set(tuple(shape) for shape in image_shapes)) > 1:
                self.finished_signal.emit(False, "All images must have the same dimensions")
                return

            if len(set(exptimes)) == 1:
                self.finished_signal.emit(False, "Images must have different exposure times for HDR merge")
                return

            # Check memory requirements
            available_memory = get_memory_info()
            max_allowed_memory = int(available_memory * 0.85)  # 85% of available memory
            estimated_memory = estimate_memory_usage(image_shapes, len(self.files),
                                                   self.parameters['save_weights'])

            use_low_memory = estimated_memory > max_allowed_memory

            if use_low_memory:
                self.status.emit(f"Using low memory mode (need {estimated_memory/1024/1024/1024:.1f}GB, "
                               f"available {max_allowed_memory/1024/1024/1024:.1f}GB)")
            else:
                self.status.emit(f"Using standard mode (need {estimated_memory/1024/1024/1024:.1f}GB, "
                               f"available {max_allowed_memory/1024/1024/1024:.1f}GB)")

            # Set up processing parameters
            sat_level = self.parameters['sat_level']
            if sat_level == 0.0:
                # For auto-detection in low memory mode, we need to scan through images
                if use_low_memory:
                    max_values = []
                    for filename in self.files:
                        img = self.siril.load_image_from_file(filename)
                        max_values.append(np.max(img.data))
                    sat_level = auto_detect_saturation_level([np.array([mv]) for mv in max_values])
                else:
                    # Load all images for auto-detection
                    images = []
                    for filename in self.files:
                        img = self.siril.load_image_from_file(filename)
                        images.append(img.data.astype(np.float32))
                    sat_level = auto_detect_saturation_level(images)

                self.status.emit(f"Auto-detected saturation level: {sat_level:.1f}")

            # Progress callback for low memory mode
            def progress_cb(pct):
                self.progress.emit(20 + int(pct * 0.6))  # Map to 20-80% range

            # Perform HDR merge
            kwargs = {
                'exptimes': exptimes,
                'gain': self.parameters['gain'],
                'eps': 1e-12,
                'return_weights': self.parameters['save_weights']
            }

            if self.parameters['use_noise_model']:
                kwargs['bg_noises'] = bg_noises
                kwargs['bg_medians'] = bg_medians

            if sat_level is not None and sat_level > 0:
                kwargs['sat_level'] = sat_level
                kwargs['sat_margin'] = self.parameters['sat_margin']

            self.status.emit("Processing HDR merge...")

            if use_low_memory:
                kwargs['file_paths'] = self.files
                kwargs['siril_interface'] = self.siril
                kwargs['progress_callback'] = progress_cb
                kwargs['calibrate_exposures'] = self.parameters.get('calibrate_exposures', True)

                if self.parameters['save_weights']:
                    result, variance, weights = hdr_merge_inverse_variance_low_memory(**kwargs)
                    # Save weight maps
                    current_dir = os.getcwd()
                    self.status.emit("Saving weight maps...")
                    save_weight_maps(weights, current_dir)
                else:
                    result, variance = hdr_merge_inverse_variance_low_memory(**kwargs)
            else:
                # Use original high memory mode
                self.progress.emit(40)
                images = []
                for i, filename in enumerate(self.files):
                    img = self.siril.load_image_from_file(filename)
                    data = img.data

                    if data.dtype == np.uint16:
                        images.append(data.astype(np.float32))
                    elif np.issubdtype(data.dtype, np.floating):
                        max_val = np.max(data)
                        if max_val > 1.0:
                            images.append((data.astype(np.float32) / max_val) * 65535.0)
                        else:
                            images.append(data.astype(np.float32) * 65535.0)

                kwargs['images'] = images

                # Apply calibration if requested
                if self.parameters.get('calibrate_exposures', True) and len(images) > 1:
                    kwargs['calibrate_exposures'] = True

                self.progress.emit(60)

                if self.parameters['save_weights']:
                    result, variance, weights = hdr_merge_inverse_variance(**kwargs)
                    # Save weight maps
                    current_dir = os.getcwd()
                    self.status.emit("Saving weight maps...")
                    save_weight_maps(weights, current_dir)
                else:
                    result, variance = hdr_merge_inverse_variance(**kwargs)

            self.progress.emit(80)

            # Validate result
            if result is None or np.any(np.isnan(result)):
                self.finished_signal.emit(False, "HDR merge produced invalid results")
                return

            # Normalize output
            if self.parameters['normalize_output']:
                max_val = np.max(result)
                if max_val > 0:
                    result = result / max_val
                else:
                    result = result / 65535.0
            else:
                result = result / 65535.0

            result = result.astype(np.float32)

            # Send to Siril
            self.status.emit("Sending result to Siril...")
            self.progress.emit(90)
            self.siril.cmd("new", "1", "1", "1", "New HDR composition")
            with self.siril.image_lock():
                self.siril.set_image_pixeldata(result)

            self.progress.emit(100)
            self.status.emit("HDR composition complete!")

            memory_mode = "low memory" if use_low_memory else "standard"
            self.finished_signal.emit(True, f"HDR composition successful using {memory_mode} mode!")

        except Exception as e:
            self.finished_signal.emit(False, f"Error: {str(e)}")

class HDRMergeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.siril = None
        self.config_file = None
        self.init_siril()
        self.init_ui()
        self.load_config()
    def init_siril(self):
        try:
            self.siril = s.SirilInterface()
            self.siril.connect()
            config_dir = self.siril.get_siril_configdir()
            self.config_file = os.path.join(config_dir, "hdr_merge_config.json")
        except Exception as e:
            QMessageBox.critical(self, "Siril Connection Error",
                               f"Failed to connect to Siril: {str(e)}")
    def init_ui(self):
        self.setWindowTitle(f"Siril HDR Blender v{VERSION}")
        self.setGeometry(100, 100, 800, 600)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # File selection group
        file_group = QGroupBox("Image Files")
        file_layout = QVBoxLayout(file_group)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.file_list.setToolTip("List of images to merge. Images must have different exposure times.")
        file_layout.addWidget(self.file_list)

        file_buttons = QHBoxLayout()
        self.add_files_btn = QPushButton("Add Files")
        self.add_files_btn.setToolTip("Add image files for HDR merging")
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_selected_btn.setToolTip("Remove selected files from the list")
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.setToolTip("Remove all files from the list")

        self.add_files_btn.clicked.connect(self.add_files)
        self.remove_selected_btn.clicked.connect(self.remove_selected)
        self.clear_all_btn.clicked.connect(self.clear_all)

        file_buttons.addWidget(self.add_files_btn)
        file_buttons.addWidget(self.remove_selected_btn)
        file_buttons.addWidget(self.clear_all_btn)
        file_buttons.addStretch()
        file_layout.addLayout(file_buttons)
        layout.addWidget(file_group)

        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)

        # Basic parameters
        basic_layout = QHBoxLayout()

        basic_layout.addWidget(QLabel("Gain:"))
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.1, 100.0)
        self.gain_spin.setValue(1.0)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.setToolTip("Camera gain (e-/ADU). Higher values reduce noise weighting. This "
            "value is specific to your camera and gain setting.")
        basic_layout.addWidget(self.gain_spin)

        basic_layout.addWidget(QLabel("Saturation Level:"))
        self.sat_level_spin = QDoubleSpinBox()
        self.sat_level_spin.setRange(0, 65535)
        self.sat_level_spin.setValue(0)
        self.sat_level_spin.setSpecialValueText("Auto")
        self.sat_level_spin.setToolTip("Pixel saturation threshold (ADU). Set to 0 for auto-detection.")
        basic_layout.addWidget(self.sat_level_spin)

        basic_layout.addWidget(QLabel("Saturation Margin:"))
        self.sat_margin_spin = QDoubleSpinBox()
        self.sat_margin_spin.setRange(0.0, 65535.0)
        self.sat_margin_spin.setValue(0.0)
        self.sat_margin_spin.setToolTip("Soft transition margin below saturation level. Pixel values "
            "between (sat_level - sat_margin) and sat_level have smoothly decreasing weight (cosine "
            "taper). 0 = hard cutoff.")
        basic_layout.addWidget(self.sat_margin_spin)

        basic_layout.addStretch()
        params_layout.addLayout(basic_layout)

        # Advanced parameters
        advanced_layout = QHBoxLayout()

        self.noise_model_check = QCheckBox("Use noise model")
        self.noise_model_check.setChecked(True)
        self.noise_model_check.setToolTip("Use image statistics for proper noise weighting")
        advanced_layout.addWidget(self.noise_model_check)

        advanced_layout.addStretch()
        params_layout.addLayout(advanced_layout)

        # Output options
        output_layout = QHBoxLayout()

        self.normalize_check = QCheckBox("Normalize to [0,1]")
        self.normalize_check.setChecked(True)
        self.normalize_check.setToolTip("Normalize output to Siril float range [0,1]. Uncheck to keep output in radiance (ADU/s)")
        output_layout.addWidget(self.normalize_check)

        self.save_weights_check = QCheckBox("Save weight maps")
        self.save_weights_check.setToolTip("Save individual weight maps as weight_1.fits, weight_2.fits, etc. For debug or to "
            "understand how the parts of the different exposures are weighted.")
        output_layout.addWidget(self.save_weights_check)

        self.calibrate_exposures_check = QCheckBox("Calibrate exposures")
        self.calibrate_exposures_check.setChecked(True)
        self.calibrate_exposures_check.setToolTip("Match exposure levels in overlap regions to reduce discontinuities. "
            "Recommended for eliminating steps at saturation boundaries.")
        output_layout.addWidget(self.calibrate_exposures_check)

        output_layout.addStretch()
        params_layout.addLayout(output_layout)

        layout.addWidget(params_group)

        # Processing group
        process_group = QGroupBox("Processing")
        process_layout = QVBoxLayout(process_group)

        self.process_btn = QPushButton("Start HDR Merge")
        self.process_btn.setToolTip("Begin HDR processing with current settings")
        self.process_btn.clicked.connect(self.start_processing)
        process_layout.addWidget(self.process_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        process_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        process_layout.addWidget(self.status_label)

        layout.addWidget(process_group)

        # Log group
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setReadOnly(True)
        self.log_text.setToolTip("Processing log and status messages")
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Image Files", "",
            "Image Files (*.fits *.fit *.fts *.tif *.tiff *.xisf *.png *.bmp);;All Files (*)"
        )
        valid_files = []
        for file in files:
            if os.path.exists(file):
                valid_files.append(file)
                item = QListWidgetItem(file)
                self.file_list.addItem(item)

                # Load image to get exposure info
                try:
                    img = self.siril.load_image_from_file(file)
                    basename = os.path.basename(file)

                    # Get exposure time
                    exptime = getattr(img.keywords, 'livetime', None)
                    if exptime is None:
                        exptime = getattr(img.keywords, 'exposure', 1.0)
                    if exptime is None or exptime <= 0:
                        exptime = 1.0

                    # Get stack count
                    stackcnt = getattr(img.keywords, 'stackcnt', 1)
                    if stackcnt is None or stackcnt <= 0:
                        stackcnt = 1

                    # Calculate frame exposure time
                    frame_exp = exptime / stackcnt

                    self.log(f"Added {basename} (frame exp: {frame_exp:.2f}s, stack count: {stackcnt}, total exp: {exptime:.2f}s)")

                except Exception as e:
                    basename = os.path.basename(file)
                    self.log(f"Added {basename} (exposure info unavailable: {str(e)})")
            else:
                self.log(f"Skipped non-existent file: {file}")

    def remove_selected(self):
        selected_items = self.file_list.selectedItems()
        count = len(selected_items)
        for item in selected_items:
            self.file_list.takeItem(self.file_list.row(item))
        self.log(f"Removed {count} files")

    def clear_all(self):
        count = self.file_list.count()
        self.file_list.clear()
        self.log(f"Cleared {count} files")

    def get_parameters(self):
        return {
            'gain': self.gain_spin.value(),
            'sat_level': self.sat_level_spin.value(),
            'sat_margin': self.sat_margin_spin.value(),
            'use_noise_model': self.noise_model_check.isChecked(),
            'normalize_output': self.normalize_check.isChecked(),
            'save_weights': self.save_weights_check.isChecked(),
            'calibrate_exposures': self.calibrate_exposures_check.isChecked()
        }

    def validate_parameters(self):
        if self.file_list.count() < 2:
            return False, "Please select at least 2 images for HDR merge."
        if self.gain_spin.value() <= 0:
            return False, "Gain must be greater than 0."
        return True, ""

    def start_processing(self):
        valid, error_msg = self.validate_parameters()
        if not valid:
            QMessageBox.warning(self, "Validation Error", error_msg)
            return
        if not self.siril:
            QMessageBox.critical(self, "Error", "Siril connection not available.")
            return

        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        parameters = self.get_parameters()

        missing_files = [f for f in files if not os.path.exists(f)]
        if missing_files:
            QMessageBox.warning(self, "Missing Files",
                              f"The following files no longer exist:\n" + "\n".join(missing_files))
            return

        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.processing_thread = HDRProcessingThread(files, parameters, self.siril)
        self.processing_thread.progress.connect(self.progress_bar.setValue)
        self.processing_thread.status.connect(self.status_label.setText)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        self.processing_thread.start()

        self.log("Started HDR processing...")

    def processing_finished(self, success, message):
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready")

        if success:
            self.log(f"Success: {message}")
            QMessageBox.information(self, "Success", message)
        else:
            self.log(f"Error: {message}")
            QMessageBox.critical(self, "Error", message)

    def log(self, message):
        self.log_text.append(message)

    def save_config(self):
        if not self.config_file:
            return
        config = {
            'gain': self.gain_spin.value(),
            'sat_level': self.sat_level_spin.value(),
            'sat_margin': self.sat_margin_spin.value(),
            'use_noise_model': self.noise_model_check.isChecked(),
            'normalize_output': self.normalize_check.isChecked(),
            'save_weights': self.save_weights_check.isChecked(),
            'calibrate_exposures': self.calibrate_exposures_check.isChecked()
        }
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.log(f"Failed to save config: {str(e)}")

    def load_config(self):
        if not self.config_file or not os.path.exists(self.config_file):
            return
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            self.gain_spin.setValue(config.get('gain', 1.0))
            self.sat_level_spin.setValue(config.get('sat_level', 0))
            self.sat_margin_spin.setValue(config.get('sat_margin', 0.0))
            self.noise_model_check.setChecked(config.get('use_noise_model', True))
            self.normalize_check.setChecked(config.get('normalize_output', True))
            self.save_weights_check.setChecked(config.get('save_weights', False))
            self.calibrate_exposures_check.setChecked(config.get('calibrate_exposures', True))
            self.log("Config loaded successfully")
        except Exception as e:
            self.log(f"Failed to load config: {str(e)}")

    def closeEvent(self, event):
        self.save_config()
        if self.siril:
            try:
                self.siril.disconnect()
            except:
                pass
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = HDRMergeGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
