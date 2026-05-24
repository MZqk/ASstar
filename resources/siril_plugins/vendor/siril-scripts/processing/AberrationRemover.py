# Aberration Remover AI by Riccardo Alberghi
# Script version: 1.1.3
# SPDX-License-Identifier: GPL-3.0-or-later

# 1.0.0 Original release
# 1.0.1 bugfix: fixed incorrect handling of 16 bits images
# 1.0.2 Updates due to API changes
# 1.0.3 Implemented ONNXHelper
# 1.0.4 Fixed a copy/paste bug I overlooked when reviewing (AKB)
# 1.0.5 Refactor & Disabled CoreML acceleration for stability
# 1.0.6 Converted to PyQt6 & bugfix square pattern
# 1.1.0 Added strength slider with linear blending and dark ringing reduction with caching
#       Thanks lock042 for the beautiful UI!
#       Added `protect background` checkbox in case the model is messing up noise
#       Added tooltips. Thanks again lock042!
#       During execution added a Cancel Process button
#       Removed the bug that made a 1 pixel black frame around the image
#       Renewed the message when a new model is available to download
# 1.1.1 Fixed incorrect ensure_installed call (AKB)
# 1.1.2 Move onnxruntime import before PyQt6 (prevents DLL error in some cases) (AKB)
# 1.1.3 Added CLI args

import sirilpy as s

import os
import sys
import platform
import hashlib
import argparse

# Detect CLI mode early to avoid installing/importing GUI dependencies unnecessarily
_siril_early = s.SirilInterface()
try:
    _siril_early.connect()
    _IS_CLI = _siril_early.is_cli()
except s.SirilConnectionError:
    _IS_CLI = False
finally:
    _siril_early.disconnect()

if _IS_CLI:
    s.ensure_installed("numpy", "requests", "scipy")
else:
    s.ensure_installed("PyQt6", "numpy", "requests", "scipy")

onnx_helper = s.ONNXHelper()
onnx_helper.install_onnxruntime()

import requests
import numpy as np

import onnxruntime # for reliability, this must be imported before PyQt6
from scipy.ndimage import percentile_filter, zoom
if hasattr(onnxruntime, 'preload_dlls'):
    with s.SuppressedStderr(), s.SuppressedStdout():
        onnxruntime.preload_dlls()
onnxruntime.set_default_logger_severity(4)

# PyQt6 is only needed in GUI mode
if not _IS_CLI:
    import webbrowser
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                                QWidget, QLabel, QPushButton, QLineEdit, QMessageBox, QFileDialog, QSlider, QCheckBox,
                                QGroupBox, QFrame)
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QFont

VERSION = "1.1.3"
CONFIG_FILENAME = "aberration_remover_model.conf"

if not _IS_CLI:
    class ProcessingThread(QThread):
        """Thread for processing the image to avoid blocking the UI"""
        finished = pyqtSignal()
        progress = pyqtSignal(str, float)
        error = pyqtSignal(str)
        cancelled = pyqtSignal()

        def __init__(self, processor, use_cached_prediction=False):
            super().__init__()
            self.processor = processor
            self.use_cached_prediction = use_cached_prediction
            self._cancelled = False

        def cancel(self):
            """Request cancellation of the processing thread"""
            self._cancelled = True

        def is_cancelled(self):
            """Check if cancellation has been requested"""
            return self._cancelled

        def run(self):
            try:
                self.processor._calculate_deconvolution(use_cached_prediction=self.use_cached_prediction)
                if self._cancelled:
                    self.cancelled.emit()
                else:
                    self.finished.emit()
            except Exception as e:
                if not self._cancelled:
                    self.error.emit(str(e))


class DeconvolutionAIInterface:
    def __init__(self, github_repo="riccardoalberghi/abberation_models", cli_args=None):
        self.cli_mode = cli_args is not None

        # Fetch latest GitHub release version at init time
        self.latest_github_version = self._fetch_latest_github_release_version(github_repo)
        # Optionally, you can print or log this value for debugging
        print(f"Latest GitHub release version for {github_repo}: {self.latest_github_version}")

        system = platform.system().lower()
        self.use_hwd_acceleration = False if system == "darwin" else True

        # Initialize Siril connection
        self.siril = s.SirilInterface()
        try:
            self.siril.connect()
        except s.SirilConnectionError:
            if self.cli_mode:
                print("Error: Failed to connect to Siril")
            else:
                QMessageBox.critical(None, "Error", "Failed to connect to Siril")
            return

        if not self.siril.is_image_loaded():
            if self.cli_mode:
                self.siril.log("Error: No image loaded", color="red")
            else:
                QMessageBox.critical(None, "Error", "No image loaded")
            return

        try:
            self.siril.cmd("requires", "1.3.6")
        except s.CommandError:
            return

        # Load previously saved model path and max version used from configuration
        model_path, stored_max = self.check_config_file()

        # Initialize model path for UI
        self.model_path = model_path or ""

        # Keep stored max version for update comparisons
        self.stored_max = stored_max

        # Processing thread
        self.processing_thread = None

        # Caching for original and deconvolved images
        self.cached_original = None
        self.cached_deconvolved = None
        self.cached_original_format = None
        self.cached_original_dtype = None
        self.cached_image_hash = None
        self.cached_raw_prediction = None
        self.cached_dark_ringing_state = None
        self.cached_protect_background_state = None

        # Strength parameter (0 to 1)
        self.strength = 1.0

        # Processing option flags (used by both CLI and GUI modes)
        self.use_dark_ringing = False
        self.use_protect_background = False

        if self.cli_mode:
            # CLI mode: set parameters from args and run processing directly
            self.strength = cli_args.strength
            self.use_dark_ringing = cli_args.dark_ringing
            self.use_protect_background = cli_args.protect_background

            if cli_args.model_path:
                self.model_path = cli_args.model_path

            if not self.model_path or not os.path.isfile(self.model_path):
                self.siril.log("Error: No valid ONNX model file. Use -model_path or configure via GUI first.", color="red")
                self.siril.disconnect()
                return

            self.siril.log(f"Aberration Remover CLI: strength={self.strength}, "
                           f"dark_ringing={self.use_dark_ringing}, protect_background={self.use_protect_background}")
            self._calculate_deconvolution()
            self.siril.disconnect()
        else:
            # GUI mode: create widgets
            self.create_widgets()

            # Initial model update check based on saved config
            model_path, stored_max = self.check_config_file()
            if model_path:
                ai_version = None
                try:
                    session = onnxruntime.InferenceSession(model_path)
                    meta = session.get_modelmeta()
                    ai_version = meta.custom_metadata_map.get("ai_version", None)
                except Exception as e:
                    print(f"Could not read ai_version from saved model: {e}")
                if ai_version:
                    ai_cmp = ai_version.lstrip("vV")
                    latest_cmp = (self.latest_github_version or "").lstrip("vV")
                    def version_tuple(v):
                        try:
                            return tuple(map(int, v.split(".")))
                        except:
                            return ()
                    # Display update if saved model is older than latest GitHub release
                    if latest_cmp and version_tuple(ai_cmp) < version_tuple(latest_cmp):
                        self.model_update_label.setText("A new model version is available.\nClick 'Download model & info' to get it.")
                        self.model_update_label.show()
                        self.download_model_btn.setStyleSheet("QPushButton { color: #D32F2F; font-weight: bold; }")
                    else:
                        self.model_update_label.hide()
                        self.download_model_btn.setStyleSheet("")

    def _fetch_latest_github_release_version(self, repo):
        """
        Fetch the latest release version from the specified GitHub repository.
        :param repo: str, in the form "owner/repo"
        :return: str, version tag or None if not found/error
        """
        api_url = f"https://api.github.com/repos/{repo}/releases/latest"
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("tag_name")
            else:
                print(f"GitHub API error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error fetching GitHub release version: {e}")
        return None

    def create_widgets(self):
        # Create main window
        self.window = QMainWindow()
        self.window.setWindowTitle(f"Aberrations Remover - v{VERSION}")

        # Create central widget
        central_widget = QWidget()
        self.window.setCentralWidget(central_widget)

        # Main layout with increased margins and spacing
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # ===== Header Section =====
        title_label = QLabel("Aberrations Remover by Riccardo Alberghi")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        version_label = QLabel(f"Script version: {VERSION}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_font = QFont()
        version_font.setPointSize(9)
        version_label.setFont(version_font)
        main_layout.addWidget(version_label)

        # Add separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(separator1)

        # ===== Model Selection Group =====
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(15)
        model_layout.setContentsMargins(15, 20, 15, 15)

        # Model Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.load_model_btn = QPushButton("Load model")
        self.load_model_btn.setMinimumHeight(35)
        self.load_model_btn.clicked.connect(self._browse_model)
        button_layout.addWidget(self.load_model_btn)

        self.download_model_btn = QPushButton("Download model && info")
        self.download_model_btn.setMinimumHeight(35)
        self.download_model_btn.clicked.connect(self._download_model)
        button_layout.addWidget(self.download_model_btn)

        model_layout.addLayout(button_layout)

        # Display selected model path
        path_label = QLabel("Model path:")
        path_label_font = QFont()
        path_label_font.setPointSize(9)
        path_label.setFont(path_label_font)
        model_layout.addWidget(path_label)

        self.model_entry = QLineEdit()
        self.model_entry.setText(self.model_path)
        self.model_entry.setReadOnly(True)
        self.model_entry.setMinimumHeight(28)
        model_layout.addWidget(self.model_entry)

        # Model update message label
        self.model_update_label = QLabel("")
        self.model_update_label.setStyleSheet("QLabel { color : #D32F2F; font-weight: bold; }")
        self.model_update_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_update_label.setMinimumHeight(40)
        self.model_update_label.hide()  # Hidden by default, shown when update available
        model_layout.addWidget(self.model_update_label)

        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # ===== Processing Options Group =====
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout()
        options_layout.setSpacing(15)
        options_layout.setContentsMargins(15, 20, 15, 15)

        # Strength slider
        strength_label = QLabel("Strength:")
        strength_font = QFont()
        strength_font.setBold(True)
        strength_label.setFont(strength_font)
        options_layout.addWidget(strength_label)

        slider_layout = QHBoxLayout()
        slider_layout.setSpacing(10)

        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setMinimum(0)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setValue(100)
        self.strength_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.strength_slider.setTickInterval(10)
        self.strength_slider.setMinimumHeight(25)
        self.strength_slider.valueChanged.connect(self._on_strength_changed)
        self.strength_slider.setToolTip("Adjusts the strength of the aberration removal effect. 0% is the original image, 100% is the fully processed image.")
        slider_layout.addWidget(self.strength_slider)

        self.strength_value_label = QLabel("1.00")
        self.strength_value_label.setMinimumWidth(50)
        strength_value_font = QFont()
        strength_value_font.setPointSize(10)
        strength_value_font.setBold(True)
        self.strength_value_label.setFont(strength_value_font)
        self.strength_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_layout.addWidget(self.strength_value_label)

        options_layout.addLayout(slider_layout)

        # Dark Ringing Reduction
        self.dark_ringing_checkbox = QCheckBox("Dark ringing reduction")
        self.dark_ringing_checkbox.setMinimumHeight(25)
        self.dark_ringing_checkbox.setToolTip("If enabled, applies a filter to reduce dark ringing artifacts that may appear around bright stars.")
        self.dark_ringing_checkbox.toggled.connect(lambda checked: setattr(self, 'use_dark_ringing', checked))
        options_layout.addWidget(self.dark_ringing_checkbox)

        # Protect Background
        self.protect_background_checkbox = QCheckBox("Protect background")
        self.protect_background_checkbox.setToolTip("If enabled, builds a mask of the background pixels and does not apply the model on such pixels.")
        self.protect_background_checkbox.setMinimumHeight(25)
        self.protect_background_checkbox.toggled.connect(lambda checked: setattr(self, 'use_protect_background', checked))
        options_layout.addWidget(self.protect_background_checkbox)

        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)

        # ===== Process Section =====
        # Calculate button
        self.calc_btn = QPushButton("Calculate")
        self.calc_btn.setMinimumHeight(40)
        calc_btn_font = QFont()
        calc_btn_font.setBold(True)
        calc_btn_font.setPointSize(11)
        self.calc_btn.setFont(calc_btn_font)
        self.calc_btn.clicked.connect(self._on_calculate)
        main_layout.addWidget(self.calc_btn)

        # Progress message label
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setMinimumHeight(25)
        progress_font = QFont()
        progress_font.setItalic(True)
        self.progress_label.setFont(progress_font)
        self.progress_label.setStyleSheet("QLabel { color: #1976D2; }")
        main_layout.addWidget(self.progress_label)

        # Adjust window size to content
        self.window.adjustSize()
        self.window.setMinimumWidth(500)
        self.window.setFixedSize(self.window.size())

        # Show the window
        self.window.show()

    def _clear_cache(self):
        """Clear cached images"""
        self.cached_original = None
        self.cached_deconvolved = None
        self.cached_original_format = None
        self.cached_original_dtype = None
        self.cached_image_hash = None
        self.cached_raw_prediction = None
        self.cached_dark_ringing_state = None
        self.cached_protect_background_state = None

    def _set_ui_enabled(self, enabled):
        """Enable or disable all UI components except the Calculate/Cancel button"""
        self.load_model_btn.setEnabled(enabled)
        self.download_model_btn.setEnabled(enabled)
        self.model_entry.setEnabled(enabled)
        self.strength_slider.setEnabled(enabled)
        self.dark_ringing_checkbox.setEnabled(enabled)
        self.protect_background_checkbox.setEnabled(enabled)

    def _compute_image_hash(self, pixel_data):
        """
        Compute a hash of the image data for cache validation.

        Parameters
        ----------
        pixel_data : ndarray
            Image pixel data

        Returns
        -------
        str
            SHA256 hash of the image data
        """
        # Convert to bytes and compute hash
        return hashlib.sha256(pixel_data.tobytes()).hexdigest()

    def _browse_model(self):
        filename, _ = QFileDialog.getOpenFileName(
            self.window,
            "Select ONNX Model",
            os.path.expanduser("~"),
            "ONNX Model (*.onnx)"
        )
        if filename:
            # Clear cache when loading a new model
            self._clear_cache()
            self.model_path = filename
            self.model_entry.setText(filename)
            # Read stored max_version_used from config (if any)
            _, stored_max = self.check_config_file()
            # Try to extract ai_version from ONNX model metadata
            ai_version = None
            try:
                session = onnxruntime.InferenceSession(filename)
                meta = session.get_modelmeta()
                ai_version = meta.custom_metadata_map.get("ai_version", None)
            except Exception as e:
                print(f"Could not read ai_version from model: {e}")
            # Compare versions and compute new max_version_used
            show_update = False
            new_max = stored_max
            if ai_version is not None:
                # Normalize version strings
                ai_cmp = ai_version.lstrip("vV")
                latest_cmp = (self.latest_github_version or "").lstrip("vV")
                stored_cmp = (stored_max or "").lstrip("vV")
                # Version comparison helper
                def version_tuple(v):
                    try:
                        return tuple(map(int, v.split(".")))
                    except:
                        return ()
                # Update stored max if this ai_version is newer
                if not stored_cmp or version_tuple(ai_cmp) > version_tuple(stored_cmp):
                    new_max = ai_version
                # Show update if loaded model version is older than latest GitHub release
                if latest_cmp and version_tuple(ai_cmp) < version_tuple(latest_cmp):
                    show_update = True
            # Display update message
            if show_update:
                self.model_update_label.setText("A new model version is available.\nClick 'Download model & info' to get it.")
                self.model_update_label.show()
                self.download_model_btn.setStyleSheet("QPushButton { color: #D32F2F; font-weight: bold; }")
            else:
                self.model_update_label.hide()
                self.download_model_btn.setStyleSheet("")
            # Save config with updated max_version_used
            self.save_config_file(filename, new_max)

    def _download_model(self):
        webbrowser.open("https://github.com/riccardoalberghi/abberation_models/releases/latest")

    def _on_strength_changed(self, value):
        """Update strength value when slider changes"""
        self.strength = value / 100.0
        self.strength_value_label.setText(f"{self.strength:.2f}")

    def _on_calculate(self):
        # Check if we're currently processing - if so, cancel
        if self.processing_thread is not None and self.processing_thread.isRunning():
            self._cancel_processing()
            return

        # Check if we have cached data and if current image matches cached hash
        if self.cached_original is not None and self.cached_deconvolved is not None and self.cached_image_hash is not None:
            # Get current image and compute its hash
            try:
                with self.siril.image_lock():
                    current_pixel_data = self.siril.get_image_pixeldata()
                    current_hash = self._compute_image_hash(current_pixel_data)

                # If hash matches...
                if current_hash == self.cached_image_hash:
                    # Check if processing options changed
                    current_br_state = self.use_dark_ringing
                    current_pb_state = self.use_protect_background

                    if (current_br_state == self.cached_dark_ringing_state and
                        current_pb_state == self.cached_protect_background_state):
                        # Everything matches, just blend
                        self._apply_cached_blend()
                        return
                    else:
                        # Hash matches but processing options changed.
                        # We can reuse the raw prediction if available.
                        if self.cached_raw_prediction is not None:
                            self._start_processing_thread(use_cached_prediction=True)
                            return
                        # If raw prediction missing for some reason, fall through to full process
                else:
                    # Different image - clear cache and proceed with full processing
                    self._clear_cache()
            except Exception as e:
                print(f"Error checking image hash: {e}")
                # Clear cache on error and proceed with full processing
                self._clear_cache()

        # Do full processing
        model_path = self.model_path
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.critical(self.window, "Error", "Please select a valid ONNX model file.")
            return

        self._start_processing_thread(use_cached_prediction=False)

    def _cancel_processing(self):
        """Cancel the current processing operation"""
        if self.processing_thread is not None and self.processing_thread.isRunning():
            self.processing_thread.cancel()
            self.progress_label.setText("Cancelling...")
            self.calc_btn.setEnabled(False)  # Disable while waiting for cancellation

    def _start_processing_thread(self, use_cached_prediction=False):
        # Disable all UI components except the Calculate button
        self._set_ui_enabled(False)
        # Change button text to "Cancel process"
        self.calc_btn.setText("Cancel process")
        # Start the processing thread
        self.processing_thread = ProcessingThread(self, use_cached_prediction=use_cached_prediction)
        self.processing_thread.finished.connect(self._processing_finished)
        self.processing_thread.progress.connect(self._update_progress_ui)
        self.processing_thread.error.connect(self._processing_error)
        self.processing_thread.cancelled.connect(self._processing_cancelled)
        self.processing_thread.start()

    def _apply_cached_blend(self):
        """Apply strength blending using cached original and deconvolved images"""
        try:
            self._update_progress_ui("Applying strength blending...")

            with self.siril.image_lock():
                # Apply strength blending
                final_image = self.apply_strength_blend(
                    self.cached_original,
                    self.cached_deconvolved,
                    self.strength
                )

                # Restore original format and finalize
                final_image = self._finalize_image(
                    final_image,
                    self.cached_original_format,
                    self.cached_original_dtype
                )

                self.siril.set_image_pixeldata(final_image)

                # Update cached hash with the new result we just wrote
                result_pixel_data = self.siril.get_image_pixeldata()
                self.cached_image_hash = self._compute_image_hash(result_pixel_data)

                self._update_progress_ui("Process complete.")
                self.siril.log(f"Applied strength {self.strength:.2f} to cached deconvolution.")

        except Exception as e:
            self._update_progress_ui(f"Error: {str(e)}")
            print(f"Error applying cached blend: {str(e)}")
            QMessageBox.critical(self.window, "Error", f"Error applying cached blend:\n{str(e)}")

    def _processing_finished(self):
        """Called when processing is complete"""
        self._restore_ui_after_processing()
        self.progress_label.setText("Process complete.")

    def _processing_error(self, error_msg):
        """Called when processing encounters an error"""
        self._restore_ui_after_processing()
        self.progress_label.setText(f"Error: {error_msg}")
        QMessageBox.critical(self.window, "Processing Error", f"An error occurred during processing:\n{error_msg}")

    def _processing_cancelled(self):
        """Called when processing is cancelled"""
        self._restore_ui_after_processing()
        self.progress_label.setText("Process cancelled.")
        self.siril.log("Aberrations Remover processing cancelled.")

    def _restore_ui_after_processing(self):
        """Restore UI state after processing completes, errors, or is cancelled"""
        self._set_ui_enabled(True)
        self.calc_btn.setEnabled(True)
        self.calc_btn.setText("Calculate")

    def _update_progress_ui(self, message, progress=0):
        """Update the progress label from the UI thread"""
        self.progress_label.setText(message)
        QApplication.processEvents()  # Allow GUI to update

    def _calculate_deconvolution(self, use_cached_prediction=False):
        """Main deconvolution calculation method with improved error handling and structure."""
        try:
            with self.siril.image_lock():
                pixel_data = None
                deconvolved_image = None
                original_format = None
                original_dtype = None

                if use_cached_prediction and self.cached_raw_prediction is not None and self.cached_original is not None:
                    self._update_progress("Restoring from cache...")
                    pixel_data = self.cached_original
                    deconvolved_image = np.copy(self.cached_raw_prediction)
                    original_format = self.cached_original_format
                    original_dtype = self.cached_original_dtype
                else:
                    # Full inference flow
                    # Initialize ONNX session
                    session = self._initialize_onnx_session()
                    if session is None:
                        return

                    # Prepare image data
                    pixel_data, original_format, original_dtype = self._prepare_image_data()
                    if pixel_data is None:
                        return

                    # Process the image
                    self._update_progress("Saving undo state...")
                    self.siril.undo_save_state("Aberrations Remover")

                    # Cache the original image (in channels-first format, normalized)
                    self.cached_original = np.copy(pixel_data)
                    self.cached_original_format = original_format
                    self.cached_original_dtype = original_dtype

                    # Process and cache the raw deconvolved result
                    self._update_progress("Running inference...")
                    deconvolved_image = self._process_image_patches(session, pixel_data)

                    # Check for cancellation after inference
                    if self.processing_thread and self.processing_thread.is_cancelled():
                        return

                    self.cached_raw_prediction = np.copy(deconvolved_image)

                # Check for cancellation before post-processing
                if self.processing_thread and self.processing_thread.is_cancelled():
                    return

                # Post-processing (Dark Ringing Reduction)
                if self.use_dark_ringing:
                    self._update_progress("Applying dark ringing reduction...")

                    # Optimized threshold calculation (approx 4x faster)
                    threshold = self._calculate_dark_ringing_threshold(pixel_data)

                    # If prediction goes below threshold, keep original pixel
                    mask = deconvolved_image < threshold
                    deconvolved_image[mask] = pixel_data[mask]

                if self.use_protect_background:
                    self._update_progress("Applying background protection...")
                    bg_mask = self._calculate_background_mask(pixel_data)
                    deconvolved_image[bg_mask] = pixel_data[bg_mask]

                # Update cached processed result and state
                self.cached_deconvolved = np.copy(deconvolved_image)
                self.cached_dark_ringing_state = self.use_dark_ringing
                self.cached_protect_background_state = self.use_protect_background

                # Check for cancellation before final blending
                if self.processing_thread and self.processing_thread.is_cancelled():
                    return

                # Apply strength blending
                self._update_progress("Applying strength blending...")
                final_image = self.apply_strength_blend(self.cached_original, self.cached_deconvolved, self.strength)

                # Restore original format and finalize
                final_image = self._finalize_image(final_image, original_format, original_dtype)

                # Final cancellation check before writing
                if self.processing_thread and self.processing_thread.is_cancelled():
                    return

                self.siril.set_image_pixeldata(final_image)

                # Compute and cache hash of the result we just wrote
                # This allows us to detect if the image changes between Calculate clicks
                result_pixel_data = self.siril.get_image_pixeldata()
                self.cached_image_hash = self._compute_image_hash(result_pixel_data)

                self._update_progress("Process complete.")
                self.siril.log("Aberrations Remover processing complete.")
                self.siril.reset_progress()

        except Exception as e:
            self._update_progress(f"Error: {str(e)}")
            print(f"Error: {str(e)}")

    def _initialize_onnx_session(self):
        """Initialize ONNX runtime session with proper error handling."""
        self._update_progress("Loading ONNX model...")
        model_path = self.model_path

        # Initialize ONNX runtime session with ONNXHelper
        with s.SuppressedStderr():
            providers = onnx_helper.get_execution_providers_ordered(self.use_hwd_acceleration)

            try:
                session = onnxruntime.InferenceSession(model_path, providers=providers)
                print(f"Used providers: {providers}")
                return session
            except Exception as err:
                error_message = str(err)
                print("Warning: falling back to CPU.")
                if "cudaErrorNoKernelImageForDevice" in error_message \
                    or "Error compiling model" in error_message:
                    print("ONNX cannot build an inferencing kernel for this GPU.")

                # Retry with CPU only
                providers = ['CPUExecutionProvider']
                try:
                    session = onnxruntime.InferenceSession(model_path, providers=providers)
                    return session
                except onnxruntime.ONNXRuntimeError as err:
                    raise Exception("Cannot build an inference model on this device")

    def _prepare_image_data(self):
        """Prepare image data for processing, handling format conversions and normalization."""
        self._update_progress("Fetching image data...")
        pixel_data = self.siril.get_image_pixeldata()
        original_dtype = pixel_data.dtype

        # Convert to channels-first format
        original_format = None
        if pixel_data.ndim == 2:
            # Mono image: (H, W) -> (1, H, W)
            pixel_data = pixel_data[np.newaxis, ...]
            original_format = "mono"
        elif pixel_data.ndim == 3:
            if pixel_data.shape[0] in [1, 3]:
                # Already channels-first: (C, H, W)
                original_format = "channels_first"
            elif pixel_data.shape[2] in [1, 3]:
                # Channels-last: (H, W, C) -> (C, H, W)
                pixel_data = np.transpose(pixel_data, (2, 0, 1))
                original_format = "channels_last"
            else:
                raise ValueError(f"Unsupported image shape: {pixel_data.shape}")
        else:
            raise ValueError(f"Unsupported number of dimensions: {pixel_data.ndim}")

        # Normalize 16-bit images to [0,1] range
        if original_dtype == np.uint16:
            pixel_data = pixel_data.astype(np.float32) / 65535.0

        return pixel_data, original_format, original_dtype

    def _process_image_patches(self, session, pixel_data):
        """Process image using patch-based approach with proper blending."""
        patch_size = 512
        overlap = 64

        # Create Hann window for smooth blending
        hann1d = np.hanning(patch_size)
        window2d = np.outer(hann1d, hann1d).astype(np.float32)
        window2d = window2d / window2d.max()
        # Ensure non-zero weights at edges to prevent black border pixels
        window2d = np.maximum(window2d, 1e-3)

        _, H, W = pixel_data.shape

        if pixel_data.shape[0] == 1:
            # Process mono image
            return self._process_mono_image(session, pixel_data, patch_size, overlap, window2d, H, W)
        elif pixel_data.shape[0] == 3:
            # Process RGB image channel by channel
            return self._process_rgb_image(session, pixel_data, patch_size, overlap, window2d, H, W)
        else:
            raise ValueError("Only mono (1 channel) or RGB (3 channel) images are supported.")

    def _process_mono_image(self, session, pixel_data, patch_size, overlap, window2d, H, W):
        """Process a monochrome image using patch-based inference."""
        output_image = np.zeros_like(pixel_data, dtype=np.float32)
        weight_image = np.zeros_like(pixel_data, dtype=np.float32)

        h_starts = self._get_patch_indices(H, patch_size, overlap)
        w_starts = self._get_patch_indices(W, patch_size, overlap)
        total_patches = len(h_starts) * len(w_starts)

        self._update_progress("Process started.")
        patch_count = 0

        for i in h_starts:
            for j in w_starts:
                # Check for cancellation
                if self.processing_thread and self.processing_thread.is_cancelled():
                    return pixel_data  # Return original on cancellation

                patch_count += 1
                patch = np.copy(pixel_data[:, i:i + patch_size, j:j + patch_size])

                # Run inference
                processed_patch = self._run_inference(session, patch)
                processed_patch = np.nan_to_num(processed_patch, nan=0.0, posinf=0.0, neginf=0.0)

                # Apply blending and accumulate
                weighted_patch = processed_patch * window2d
                output_image[:, i:i + patch_size, j:j + patch_size] += weighted_patch
                weight_image[:, i:i + patch_size, j:j + patch_size] += window2d

                self._update_progress(
                    f"Patches done: {patch_count}/{total_patches}",
                    patch_count / total_patches
                )

        # Normalize by weights
        weight_image[weight_image == 0] = 1.0
        return output_image / weight_image

    def _process_rgb_image(self, session, pixel_data, patch_size, overlap, window2d, H, W):
        """Process an RGB image channel by channel using patch-based inference."""
        output_image = np.zeros_like(pixel_data, dtype=np.float32)
        weight_image = np.zeros_like(pixel_data, dtype=np.float32)

        h_starts = self._get_patch_indices(H, patch_size, overlap)
        w_starts = self._get_patch_indices(W, patch_size, overlap)
        total_patches = len(h_starts) * len(w_starts)

        for c in range(3):
            patch_count = 0
            self._update_progress(f"Process started on channel {c+1}/3.")

            for i in h_starts:
                for j in w_starts:
                    # Check for cancellation
                    if self.processing_thread and self.processing_thread.is_cancelled():
                        return pixel_data  # Return original on cancellation

                    patch_count += 1
                    # Extract single channel patch
                    patch = np.copy(pixel_data[c:c+1, i:i + patch_size, j:j + patch_size])

                    # Run inference
                    processed_patch = self._run_inference(session, patch)
                    processed_patch = np.nan_to_num(processed_patch, nan=0.0, posinf=0.0, neginf=0.0)

                    # Apply blending and accumulate
                    weighted_patch = processed_patch * window2d
                    output_image[c:c+1, i:i + patch_size, j:j + patch_size] += weighted_patch
                    weight_image[c:c+1, i:i + patch_size, j:j + patch_size] += window2d

                    self._update_progress(
                        f"Patches done (channel {c+1}/3): {patch_count}/{total_patches}",
                        patch_count / total_patches
                    )

        # Normalize by weights
        weight_image[weight_image == 0] = 1.0
        return output_image / weight_image

    def _run_inference(self, session, patch):
        """Run ONNX inference on a single patch with strong statistics preservation."""
        input_patch = patch.astype(np.float32)

        # Calculate strong statistics (median/MAD) of input patch BEFORE inference
        input_median = np.median(input_patch)
        input_mad = np.median(np.abs(input_patch - input_median))

        # Avoid division by zero
        if input_mad < 1e-8:
            input_mad = 1e-8

        input_batch = np.expand_dims(input_patch, axis=0)

        inputs = session.get_inputs()
        input_dict = {inputs[0].name: input_batch}
        outputs, session = onnx_helper.run(session, self.model_path, None, input_dict, return_first_output=True)

        output_patch = np.squeeze(outputs, axis=0)

        # Calculate strong statistics (median/MAD) of output patch AFTER inference
        output_median = np.median(output_patch)
        output_mad = np.median(np.abs(output_patch - output_median))

        # Avoid division by zero
        if output_mad < 1e-8:
            output_mad = 1e-8

        # Renormalize output to match input strong statistics
        # Remove output statistics and apply input statistics
        output_patch = (output_patch - output_median) / output_mad * input_mad + input_median

        return output_patch

    def _finalize_image(self, final_image, original_format, original_dtype):
        """Restore image to original format and apply final processing."""
        # Restore original format
        if original_format == "channels_last":
            final_image = np.transpose(final_image, (1, 2, 0))
        elif original_format == "mono":
            final_image = np.squeeze(final_image, axis=0)

        # Clip and handle NaN values
        final_image = np.clip(final_image, 0.0, 1.0)
        final_image = np.nan_to_num(final_image)

        # Scale back to original dtype
        if original_dtype == np.uint16:
            final_image = final_image * 65535.0
            final_image = final_image.astype(np.uint16)

        return final_image


    def _calculate_dark_ringing_threshold(self, image):
        """
        Calculate local threshold using a strided percentile filter for performance.
        Default: 2nd percentile, 15x15 window (scaled).
        """
        # Downsampling factor for performance (stride=2 -> 4x faster)
        stride = 2

        # Don't downsample very small images
        if image.shape[1] < 100 or image.shape[2] < 100:
            return percentile_filter(image, percentile=2, size=(1, 15, 15))

        # Strided slice
        small_image = image[:, ::stride, ::stride]

        # Adjust window size: 15 / stride
        # stride=2 -> 15//2 = 7.
        small_size = max(3, 15 // stride)

        # Calculate filter on small image
        small_threshold = percentile_filter(small_image, percentile=2, size=(1, small_size, small_size))

        # Upscale
        z_h = image.shape[1] / small_image.shape[1]
        z_w = image.shape[2] / small_image.shape[2]

        # Bilinear interpolation
        threshold = zoom(small_threshold, (1, z_h, z_w), order=1)

        return threshold

    def _calculate_background_mask(self, image):
        """
        Calculate background mask using robust statistics (Median + 3*Sigma).
        """
        mask = np.zeros_like(image, dtype=bool)
        # Downsample stride for statistics calculation
        stride = 4

        for c in range(image.shape[0]):
            channel_data = image[c]
            # Calculate stats on downsampled data for speed
            small_data = channel_data[::stride, ::stride]

            median = np.median(small_data)
            mad = np.median(np.abs(small_data - median))
            # Sigma approximation: 1.4826 * MAD
            # Threshold = Median + 1 * Sigma
            threshold = median + (1 * 1.4826 * mad)

            mask[c] = channel_data < threshold

        return mask

    def _get_patch_indices(self, dim_size, patch_size, overlap):
        """Compute start indices for patches with the given overlap.
        This function ensures full coverage even if the image dimensions aren't exact multiples of the stride.
        """
        stride = patch_size - overlap
        indices = []
        pos = 0
        while True:
            if pos + patch_size >= dim_size:
                indices.append(dim_size - patch_size)
                break
            else:
                indices.append(pos)
                pos += stride
        # Remove any duplicates and return in sorted order.
        return sorted(set(indices))

    def apply_strength_blend(self, original, deconvolved, strength):
        """
        Apply strength-based blending between original and deconvolved images in image space.
        Simple linear interpolation with no artifacts.

        Parameters
        ----------
        original : ndarray
            Original image (C, H, W) or (H, W) format
        deconvolved : ndarray
            Deconvolved image, same shape as original
        strength : float in [0, 1]
            Blending strength. 0 = all original, 1 = all deconvolved

        Returns
        -------
        blended : ndarray
            Blended result, same shape as input
        """
        # Simple linear blend in image space - no ringing artifacts
        return (1.0 - strength) * original + strength * deconvolved

    def check_config_file(self):
        """
        Check for a saved model path and max version used in the configuration file.
        Returns (model_path, max_version_used) or (None, None) if not found.
        """
        config_dir = self.siril.get_siril_configdir()
        config_file_path = os.path.join(config_dir, CONFIG_FILENAME)
        model_path = None
        max_version_used = None
        if os.path.isfile(config_file_path):
            with open(config_file_path, 'r') as file:
                lines = file.readlines()
                if len(lines) > 0:
                    model_path = lines[0].strip()
                    if not os.path.isfile(model_path):
                        model_path = None
                if len(lines) > 1:
                    max_version_used = lines[1].strip()
        return model_path, max_version_used

    def save_config_file(self, model_path, max_version_used=None):
        """
        Save the selected model path and max version used to the configuration file.
        """
        config_dir = self.siril.get_siril_configdir()
        config_file_path = os.path.join(config_dir, CONFIG_FILENAME)
        try:
            with open(config_file_path, 'w') as file:
                file.write(model_path + "\n")
                if max_version_used is not None:
                    file.write(str(max_version_used) + "\n")
        except Exception as e:
            print(f"Error saving config file: {str(e)}")

    def _update_progress(self, message, progress=0):
        # Emit progress signal if we're in a thread, otherwise update directly
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.progress.emit(message, progress)
        self.siril.update_progress(message, progress)

    def close_dialog(self):
        self.siril.disconnect()
        if hasattr(self, 'window'):
            self.window.close()


def main():
    try:
        if _IS_CLI:
            parser = argparse.ArgumentParser(description="Aberration Remover AI")
            parser.add_argument("-strength", type=float, default=1.0,
                                help="Strength of aberration removal (0.0 to 1.0, default: 1.0)")
            parser.add_argument("-dark_ringing", action="store_true",
                                help="Enable dark ringing reduction")
            parser.add_argument("-protect_background", action="store_true",
                                help="Enable background protection")
            parser.add_argument("-model_path", type=str, default=None,
                                help="Path to the ONNX model file (uses saved config if not provided)")
            args = parser.parse_args()
            DeconvolutionAIInterface(cli_args=args)
        else:
            app = QApplication(sys.argv)
            interface = DeconvolutionAIInterface()
            sys.exit(app.exec())
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
