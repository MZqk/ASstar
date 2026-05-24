##############################################
# Diffraction Spike Overlay Tool for Siril
#
# (c) G. Trainar 2026
# SPDX-License-Identifier: MIT License
##############################################

# Version 1.1.1

"""
Diffraction Spike Overlay Tool for Siril

Purpose:
  Add diffraction spikes to the current image loaded in Siril.
  Provides a live preview and allows saving the modified image under a new name and a mask with alpha channel (PNG).
"""

import sys
import time
import os
import logging
import base64
from typing import List, Tuple, Optional, Dict

# Siril imports
try:
    import sirilpy as s
    from sirilpy.exceptions import NoImageError, SirilConnectionError
    SIRIL = s.SirilInterface()
    if not SIRIL.connect():
        logging.error("Failed to connect to Siril. Is Siril running?")
        sys.exit(1)
except Exception as e:
    logging.exception(f"Error initializing Siril interface: {e}")
    sys.exit(1)

s.ensure_installed("numpy", "scipy", "scikit-image", "pillow", "PyQt6")

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import disk, polygon
from PIL import Image

# PyQt imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QComboBox, QPushButton,
    QFormLayout, QStatusBar, QMessageBox,
    QFileDialog, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QColorDialog, QDialog, QTextEdit, QProgressBar, 
    QGroupBox, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter, QIcon 

# Constants
VERSION = "1.1.1"

DEFAULT_CONFIG = {
    'FWHM_MIN': 20,
    'FWHM_MAX': 300,
    'FWHM_MAX_LIMIT': 999,
    'SPIKE_THICKNESS': 4,
    'SPIKE_COUNT': 4,          # 4, 6 or 8
    'BLUR_SIGMA': 0.5,         # 0.5 σ
    'ROTATION_ANGLE': 0,
    'SPIKE_LENGTH_PCT': 80,
    'SPIKE_BRIGHTNESS' : 80,
    'MASK_OPACITY': 0, 
}

CHANGELOG = """
Version 1.1.1
    - Added Spikes brightness, removed space bar toggling
    - GUI minor fixes

Version 1.1.0
    - GUI Update
    - Live Preview of the spiked-stars mask

Version 1.0.0 (Initial Release) - Feb 2026
*Features:
    -Diffraction Spike Generation: Add synthetic diffraction spikes to astronomical images
    -Real-time Preview: Live preview of spike effects with adjustable parameters
    -Star Detection: Automatic detection of stars in the image for spike placement
    -Customizable Parameters:
    -FWHM filtering (min/max)
    -Spike length and thickness
    -Number of spikes (4, 6, or 8)
    -Spike pattern shapes (Radial, Cross, X)
    -Rotation angle
    -Blur sigma for realistic effects
    -Color Customization: Choose spike color with a color picker
    -Image Formats: Support for FITS, TIFF formats

*Compatibility:
    -Siril Integration: Full compatibility with Siril's image processing pipeline
    -32-bit TIFF Support: Native handling of 32-bit floating point images

*Usage Notes:
    -The mask PNG can be opened in GIMP and used as a layer with transparency
    -The composed TIFF contains the final image with spikes applied
    -Both files are saved in the same directory as the original

*Known Limitations:
    -Requires Siril to be running with an image loaded
    -Works on RGB stretched images only
    -Large images can take a long time to process or cause the script to hang when it starts
"""

DARK_STYLESHEET = """
QWidget { background-color: #2b2b2b; color: #e0e0e0; font-size: 13pt; }
QToolTip { background-color: #333333; color: #ffffff; border: 1px solid #88aaff; }
QGroupBox {
    border: 1px solid #444444;
    margin-top: 10px;
    font-weight: bold;
    border-radius: 4px;
    padding-top: 14px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5x;
    color: #88aaff;
}
QLabel { color: #cccccc; }
QRadioButton, QCheckBox { color: #cccccc; spacing: 5px; }
QRadioButton::indicator, QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #666666;
    background: #3c3c3c;
    border-radius: 7px;
}
QCheckBox::indicator { border-radius: 3px; }
QRadioButton::indicator:checked {
    background: qradialgradient(cx:0.5, cy:0.5, radius: 0.4,
        fx:0.5, fy:0.5, stop:0 #ffffff, stop:1 #285299);
    border: 1px solid #88aaff;
}
QCheckBox::indicator:checked {
    background-color: #285299;
    border: 1px solid #88aaff;
}
QSlider::groove:horizontal {
    background: #444444;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background-color: #aaaaaa;
    width: 14px; height: 14px;
    margin: -4px 0;
    border-radius: 7px;
    border: 1px solid #555555;
}
QSlider::handle:horizontal:hover { background-color: #ffffff; }
QPushButton {
    background-color: #444444;
    color: #dddddd;
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover { background-color: #555555; border-color: #777777; }
QPushButton#ProcessButton {
    background-color: #285299;
    border: 1px solid #1e3f7a;
    font-size: 12pt;
    padding: 12px;
}
QPushButton#PrepButton:hover { background-color: #355ea1; }
"""

class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass

# Helper functions
def wait_for_lock(siril: s.SirilInterface, retries: int = 5, delay: float = 0.5) -> None:
    """Yield a lock on Siril's processing thread."""
    for _ in range(retries):
        try:
            return siril.image_lock()
        except s.ProcessingThreadBusyError:
            time.sleep(delay)
    raise s.ProcessingThreadBusyError(
        "Siril processing thread is still busy after retries"
    )

def array_to_qimage(arr: np.ndarray) -> QImage:
    """Return a QImage from a numpy array."""
    if arr.ndim == 2:  # grayscale
        return QImage(arr.tobytes(), arr.shape[1], arr.shape[0],
                      QImage.Format.Format_Grayscale8)
    else:  # RGB / RGBA
        fmt = QImage.Format.Format_RGBA8888 if arr.shape[2] == 4 else QImage.Format.Format_RGB888
        return QImage(arr.tobytes(), arr.shape[1], arr.shape[0],
                      arr.strides[0], fmt)

def calculate_spike_heights(fwhm: float, spike_length_pct: int, spike_count: int) -> List[float]:
    """
    Calculate heights for each spike based on configuration.
    For 8 spikes: alternate long/short (short = 0.7 × long).
    For other counts: all spikes have the same height.
    """
    base_height = fwhm * spike_length_pct / 100.0
    if spike_count == 8:
        return [base_height if i % 2 == 0 else base_height * 0.7
                for i in range(8)]
    return [base_height] * spike_count

def compute_spike_angles(spike_count: int, pattern: str, rotation_angle: float) -> List[float]:
    """Calculate spike angles based on pattern and rotation."""
    base = 2 * np.pi / spike_count
    angles = [i * base for i in range(spike_count)]

    if pattern == "Cross":
        angles = [a + np.pi/2 for a in angles]
    elif pattern == "X":
        angles = [a + np.pi/4 for a in angles] + \
                 [a + 3*np.pi/4 for a in angles]

    return [(a + np.radians(rotation_angle)) % (2 * np.pi) for a in angles]

def draw_spike_triangle(mask: np.ndarray, x: float, y: float,
                        angle: float, height: float, base: float) -> None:
    """Draw a single spike triangle on the mask."""
    tip_x = x + height * np.cos(angle)
    tip_y = y + height * np.sin(angle)

    perp1 = angle + np.pi/2
    perp2 = angle - np.pi/2

    left_x = x + (base/2) * np.cos(perp1)
    left_y = y + (base/2) * np.sin(perp1)
    right_x = x + (base/2) * np.cos(perp2)
    right_y = y + (base/2) * np.sin(perp2)

    rr, cc = polygon([tip_y, left_y, right_y],
                     [tip_x, left_x, right_x], mask.shape)
    valid = (rr >= 0) & (rr < mask.shape[0]) & \
            (cc >= 0) & (cc < mask.shape[1])
    rr, cc = rr[valid], cc[valid]
    mask[rr, cc] = 1.0

# Worker threads
class ImageLoader(QThread):
    """Background thread to load image from Siril."""
    finished = pyqtSignal(object)  # emits the fetched ffit

    def run(self):
        try:
            with wait_for_lock(SIRIL):
                ffit = SIRIL.get_image()
            self.finished.emit(ffit)
        except Exception as e:
            logging.exception(f"Error loading image from Siril: {e}")
            self.finished.emit(None)

class PreviewWorker(QThread):
    """Background thread to generate preview with spikes."""
    finished = pyqtSignal(QPixmap, int)

    def __init__(self, image: np.ndarray, params: Dict, parent=None):
        super().__init__(parent)
        self.image = image.copy()
        self.params = params

    def run(self):
        try:
            mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.float32)
            angles = compute_spike_angles(
                self.params["spike_count"],
                self.params["pattern"],
                self.params["rotation_angle"]
            )

            star_count = 0
            for x, y, fwhm in self.params["stars_list"]:
                if not (self.params["fwhm_min"] <= fwhm <= self.params["fwhm_max"]):
                    continue
                star_count += 1

                # height = fwhm * self.params["spike_length"] / 100.0
                base = max(self.params["spike_thickness"], fwhm / 10.0)

                base_factor = max(self.params["spike_length"] / 100.0, 0.2)
                base *= base_factor

                heights = calculate_spike_heights(fwhm, self.params["spike_length"], len(angles))

                for a, h in zip(angles, heights):
                    draw_spike_triangle(mask, x, y, a, h, base)

                # Apply blur around the star
                y0 = max(0, int(y) - 20)
                y1 = min(self.image.shape[0], int(y) + 21)
                x0 = max(0, int(x) - 20)
                x1 = min(self.image.shape[1], int(x) + 21)
                self.image[y0:y1, x0:x1] = gaussian_filter(
                    self.image[y0:y1, x0:x1], sigma=self.params["blur_sigma"]
                )

            # ----- Darken the base image according to opacity -----
            mask_opacity = self.params.get("opacity", 0.0)
            if mask_opacity > 0:
                # Multiply by (1 – opacity) to darken everything
                self.image *= (1.0 - mask_opacity)

            # Convert image to float if necessary
            if self.image.dtype.kind != 'f':
                self.image = self.image.astype(np.float32)

            # Add colored spikes
            brightness = self.params.get("spike_brightness", 1.0)
            color_norm = np.array(self.params["spike_color"], dtype=np.float32) / 255.0
            self.image[..., 0] += mask * color_norm[0] * brightness
            self.image[..., 1] += mask * color_norm[1] * brightness
            self.image[..., 2] += mask * color_norm[2] * brightness

            # Clip to valid range
            np.clip(self.image, 0.0, 1.0, out=self.image)

            # Create pixmap
            img = np.flipud(self.image)
            if img.dtype.kind != 'u':
                img = np.clip(img, 0.0, 1.0)
                img = (img * 255).astype(np.uint8)

            qimg = array_to_qimage(img)
            pixmap = QPixmap.fromImage(qimg)

            # Scale to view size
            target_size = self.parent().view.size()
            scaled_pixmap = pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.finished.emit(scaled_pixmap, star_count)

        except Exception as e:
            logging.error(f"Error generating preview: {e}")
            self.finished.emit(QPixmap(), 0)

# UI Components
class HelpDialog(QDialog):
    """Scrollable help dialog."""
    def __init__(self, title: str, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)

        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText(text)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        lay = QVBoxLayout()
        lay.addWidget(txt)
        lay.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
        self.setLayout(lay)

        # Start with a reasonable size; user can still resize
        self.resize(800, 600)

class SpikeWindow(QMainWindow):
    """Main application window for diffraction spike generation."""
    def __init__(self, preloaded_ffit=None):
        super().__init__()
        self.setWindowTitle(f"Diffraction Spikes Generator {VERSION}")
        self.setStyleSheet(DARK_STYLESHEET)
        self.resize(1200, 800)
        self.showing_original = False 
        self._prev_transform = None 

        # Image data
        self.image_orig = None
        self.header = None
        self.current_ffit = preloaded_ffit
        self.filename = ""
        self.stars_list = []

        # Spike appearance - RGB only
        self.spike_color = (255, 255, 255)  # Default white RGB

        # Preview timer
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.update_preview)

        # Resize debounce timer
        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.update_preview)

        # UI
        self.init_ui()

        # Progress bar overlay (hidden until we start loading)
        self.loading_widget = QWidget(self)
        self.loading_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.loading_widget.setStyleSheet("background-color: rgba(255, 255, 255, 200);")
        self.loading_layout = QVBoxLayout(self.loading_widget)
        self.loading_bar = QProgressBar()
        self.loading_bar.setRange(0, 0)  # indeterminate
        self.loading_label = QLabel("Loading image…")
        self.loading_layout.addWidget(self.loading_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.loading_layout.addWidget(self.loading_bar, alignment=Qt.AlignmentFlag.AlignCenter)
        self.loading_widget.hide()

        # Start loading the image in a background thread
        if preloaded_ffit is None:
            self._start_image_loader()
        else:
            self._set_image_from_ffit(preloaded_ffit)

    def init_ui(self):
        """Initialize the user interface."""
        # -----------------------------------------------------------------
        #  MAIN SPLITTER
        # -----------------------------------------------------------------
        from PyQt6.QtWidgets import QSplitter, QSpacerItem

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(splitter)
        splitter.setContentsMargins(10, 0, 0, 0)

        # -----------------------------------------------------------------
        #  LEFT PANE – ALL CONTROLS IN ONE FIXED WIDGET
        # -----------------------------------------------------------------
        left_widget = QWidget()
        left_widget.setFixedWidth(320)                       # fixed width
        left_widget.setSizePolicy(QSizePolicy.Policy.Fixed,
                                  QSizePolicy.Policy.Expanding)

        # The left pane will contain only one widget – the whole controls block
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # -----------------------------------------------------------------
        #  CONTROLS – SLIDERS & GROUP BOXES
        # -----------------------------------------------------------------

        # Helper to create sliders (unchanged)
        def make_slider(label: str,
                        min_: int, max_: int, init_: int,
                        unit: str,
                        display_scale: float = 1.0,
                        tooltip: Optional[str] = None) -> Tuple[QWidget, QSlider]:
            """Create a slider row with a right‑aligned value label and a single space
            between the number and its unit."""
            sld = QSlider(Qt.Orientation.Horizontal)
            sld.setRange(min_, max_)
            sld.setValue(init_)
            # Value label: fixed width, right‑aligned
            if display_scale == 10.0:
                val_lbl = QLabel(f"{init_/display_scale:.1f}")
            else:
                val_lbl = QLabel(str(init_))

            # Use a reference width based on the longest expected unit string
            ref_text = "99px"
            fixed_width = val_lbl.fontMetrics().horizontalAdvance(ref_text)
            val_lbl.setFixedWidth(fixed_width)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)

            # Unit label
            unit_lbl = QLabel(unit)
            def update_val(val):
                if display_scale == 10.0:   # blur sigma – one decimal
                    lbl_text = f"{val / display_scale:.1f}"
                else:                       # all other sliders – integer
                    lbl_text = str(int(val / display_scale))
                val_lbl.setText(lbl_text)
            sld.valueChanged.connect(update_val)
            sld.valueChanged.connect(self.schedule_preview_update)
            if tooltip:
                sld.setToolTip(tooltip)
                val_lbl.setToolTip(tooltip)
                unit_lbl.setToolTip(tooltip)
            w = QWidget()
            lay = QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            
            # Layout order: slider | value | space | unit
            lay.addWidget(sld)
            lay.addWidget(val_lbl)
            lay.addWidget(unit_lbl)
            return w, sld

        # Slider widgets
        
        self.fwhm_min_lbl, self.fwhm_min_slider = make_slider(
            "FWHM Min", 5, 40, DEFAULT_CONFIG['FWHM_MIN'], "px",
            tooltip="Select the minimum FWHM (in pixels) of stars to be spiked."
        )

        self.fwhm_max_lbl, self.fwhm_max_slider = make_slider(
            "FWHM Max", 5, DEFAULT_CONFIG['FWHM_MAX_LIMIT'],
            DEFAULT_CONFIG['FWHM_MAX'], "px",
            tooltip="Select the maximum FWHM (in pixels) of stars to be spiked."
        )

        self.length_lbl, self.spike_length_slider = make_slider(
            "Spike Length", 0, 100, DEFAULT_CONFIG['SPIKE_LENGTH_PCT'], "%",
            tooltip="Length of each spike as a percentage of the star’s FWHM."
        )

        self.thickness_lbl, self.spike_thickness_slider = make_slider(
            "Spike Thickness", 1, 10, DEFAULT_CONFIG['SPIKE_THICKNESS'], "px",
            tooltip="Base thickness (in pixels) of the spike triangles."
        )

        self.blur_lbl, self.blur_sigma_slider = make_slider(
            "Blur Sigma", 0, 30, int(DEFAULT_CONFIG['BLUR_SIGMA'] * 10), "σ",
            display_scale=10.0,
            tooltip="Gaussian blur sigma applied to each spike for realism."
        )

        self.rotation_lbl, self.rotation_slider = make_slider(
            "Rotation Angle", 0, 360, DEFAULT_CONFIG['ROTATION_ANGLE'], "°",
            tooltip="Rotate the entire spike pattern around each star."
        )

        self.brightness_lbl, self.brightness_slider = make_slider(
            "Brightness", 0, 100, DEFAULT_CONFIG['SPIKE_BRIGHTNESS'], "%",
            tooltip="Adjust spikes overlay brightness."
        )

        self.opacity_lbl, self.opacity_slider = make_slider(
            "Opacity", 0, 100, DEFAULT_CONFIG['MASK_OPACITY'], "%",
            tooltip="Set the opacity of the star mask overlay in the preview."
        )


        # Connect brightness slider to preview update
        self.brightness_slider.valueChanged.connect(self.schedule_preview_update)


        # Colour display box
        self.color_display = QLabel()
        self.color_display.setFixedSize(20, 20)
        self.color_display.setStyleSheet(
            "background-color: rgb(255,255,255); border: 1px solid black;"
        )

        # Color-picker button
        choose_btn = QPushButton("Choose…")
        choose_btn.clicked.connect(self.choose_color)

        color_row_widget = QWidget()
        row_lay = QHBoxLayout(color_row_widget)
        row_lay.setContentsMargins(0, 0, 0, 0)   # no extra spacing
        row_lay.addWidget(self.color_display)
        row_lay.addWidget(choose_btn)

        # Combo boxes
        self.spike_count_combo = QComboBox()
        self.spike_count_combo.addItems(["4", "6", "8"])
        self.spike_count_combo.setCurrentText(str(DEFAULT_CONFIG['SPIKE_COUNT']))
        self.spike_count_combo.currentIndexChanged.connect(self.schedule_preview_update)

        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["Radial", "Cross", "X"])
        self.pattern_combo.setCurrentText("Radial")
        self.pattern_combo.currentIndexChanged.connect(self.schedule_preview_update)

        # Star count label
        self.star_count_label = QLabel("0")

        # Buttons
        default_btn = QPushButton("Default")
        default_btn.clicked.connect(self.reset_to_defaults)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_spiked_image)

        help_btn = QPushButton("Help")
        help_btn.clicked.connect(self.show_help)

        btn_row_layout = QHBoxLayout()
        btn_row_layout.addWidget(default_btn)
        btn_row_layout.addWidget(save_btn)
        btn_row_layout.addWidget(help_btn)

        # -----------------------------------------------------------------
        #  ADD GROUP BOXES TO CONTROLS LAYOUT
        # -----------------------------------------------------------------

        # Star Selection group
        star_group = QGroupBox("Stars Selection")
        star_layout = QFormLayout()
        star_layout.addRow("FWHM Min:", self.fwhm_min_lbl)
        star_layout.addRow("FWHM Max:", self.fwhm_max_lbl)
        star_layout.addRow("Detected Stars:", self.star_count_label)
        star_group.setLayout(star_layout)
        star_group.setMinimumWidth(280)

        # Spike Setting group
        spike_group = QGroupBox("Spike Settings")
        spike_layout = QFormLayout()
        spike_layout.addRow("Spike Length:", self.length_lbl)
        spike_layout.addRow("Spike Thickness:", self.thickness_lbl)
        spike_layout.addRow("Number of Spikes:", self.spike_count_combo)
        spike_layout.addRow("Blur Sigma:", self.blur_lbl)
        spike_layout.addRow("Rotation Angle:", self.rotation_lbl)
        spike_layout.addRow("Brightness:", self.brightness_lbl)
        spike_layout.addRow("Pattern Shape:", self.pattern_combo)
        spike_layout.addRow("Spike Color:", color_row_widget)
        spike_group.setLayout(spike_layout)
        spike_group.setMinimumWidth(280)
        spike_group.setMinimumHeight(320)

        # Mask Setting group
        mask_group = QGroupBox("Mask")
        mask_layout = QFormLayout()
        mask_layout.addRow("Opacity:", self.opacity_lbl)
        mask_group.setLayout(mask_layout)
        mask_group.setMinimumWidth(280)

        # Add the groups and buttons to the controls layout
        controls_layout.addWidget(star_group)
        controls_layout.addWidget(spike_group)
        spacer_between = QSpacerItem(
            0, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        controls_layout.addSpacerItem(spacer_between)
        controls_layout.addWidget(mask_group)

        # Fixed spacer between the last group and the buttons
        controls_layout.addSpacerItem(QSpacerItem(0, 10,
                                                QSizePolicy.Policy.Minimum,
                                                QSizePolicy.Policy.Fixed))

        controls_layout.addLayout(btn_row_layout)

        # Lock the whole control block to its size‑hint
        controls_container.setFixedSize(320,
                                       controls_container.sizeHint().height())

        # Place the control block inside the left pane
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        left_layout.addSpacerItem(QSpacerItem(0, 50,
                                            QSizePolicy.Policy.Minimum,
                                            QSizePolicy.Policy.Fixed))
        left_layout.addWidget(controls_container)

        splitter.addWidget(left_widget)          # add the left pane first

        # -----------------------------------------------------------------
        #  RIGHT PANE – PREVIEW
        # -----------------------------------------------------------------
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)

        # --- Top button bar ------------------------------------------
        btn_bar = QHBoxLayout()
        zoom_in_btn = QPushButton("Zoom In")
        zoom_out_btn = QPushButton("Zoom Out")
        fit_view_btn = QPushButton("Fit")

        # Fixed size for the zoom buttons
        for btn in (zoom_in_btn, zoom_out_btn, fit_view_btn):
            btn.setFixedSize(100, 30)

        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_out_btn.clicked.connect(self.zoom_out)
        fit_view_btn.clicked.connect(self.fit_view)

        btn_bar.addWidget(zoom_in_btn)
        btn_bar.addWidget(zoom_out_btn)
        btn_bar.addWidget(fit_view_btn)
        btn_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addLayout(btn_bar)

        # --- Graphics view -------------------------------------------
        self.view = QGraphicsView()
        self.scene = QGraphicsScene(self.view)
        self.view.setScene(self.scene)

        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        preview_layout.addWidget(self.view)

        self._prev_transform = self.view.transform()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        splitter.addWidget(preview_widget)      # add the right pane
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)


    def zoom_in(self):
        """Zoom in by 20%."""
        self.view.scale(1.2, 1.2)

    def zoom_out(self):
        """Zoom out by 20%."""
        self.view.scale(1 / 1.2, 1 / 1.2)

    def fit_view(self):
        """Fit the image inside the view while keeping aspect ratio."""
        if not self.scene.items():
            return
        rect = self.scene.itemsBoundingRect()
        self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def choose_color(self):
        """Open a QColorDialog, store the chosen color and update preview."""
        color = QColorDialog.getColor(QColor(*self.spike_color), self, "Select Spike Colour")
        if color.isValid():
            rgb = color.getRgb()[:3]
            self.spike_color = tuple(rgb)
            # Update the display box
            self.color_display.setStyleSheet(
                f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border: 1px solid black;"
            )
            self.schedule_preview_update()

    def _set_image_from_ffit(self, ffit):
        """Set the image from FITS data with proper scaling for display."""
        img = ffit.data

        # Handle different image formats
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            # Move channel axis to last position
            img = np.moveaxis(img, 0, -1)

        # Convert to displayable format
        if img.dtype.kind == 'f':  # Floating point data
            # Normalize to 0-1 range if needed
            img_min = np.min(img)
            img_max = np.max(img)

            # If image is very dark, apply auto-scaling
            if img_max - img_min < 0.1:
                # Use percentiles to find better range
                lower = np.percentile(img, 1)
                upper = np.percentile(img, 99.5)
                img = (img - lower) / (upper - lower + 1e-6)
            else:
                img = (img - img_min) / (img_max - img_min + 1e-6)

            # Clip to valid range
            img = np.clip(img, 0, 1)

        elif img.dtype.kind == 'u':  # Unsigned integer
            if img.dtype == np.uint16:
                img = img / 65535.0
            elif img.dtype == np.uint8:
                img = img / 255.0
        elif img.dtype.kind == 'i':  # Signed integer
            if img.dtype in (np.int16, np.int32):
                img = (img - np.min(img)) / (np.max(img) - np.min(img))

        # Convert to RGB if grayscale
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)

        # Ensure we have 3 channels
        if img.shape[2] == 4:  # RGBA -> RGB
            img = img[..., :3]

        self.image_orig = img
        self.header = getattr(ffit, "header", None)
        self.current_ffit = ffit
        
        try:
            self.filename = SIRIL.get_image_filename()
        except NoImageError:
            SIRIL.log("No image loaded – cannot get filename",s.LogColor.RED)
            self.filename = ""

        self.stars_list = self.get_stars_from_siril()
        self.update_preview()

    def _start_image_loader(self):
        """Launch the worker thread that fetches the image from Siril."""
        self.loading_widget.show()
        self.loader_thread = ImageLoader(parent=self)
        self.loader_thread.finished.connect(self._on_image_loaded)
        self.loader_thread.start()

    def _on_image_loaded(self, ffit):
        """Called when the background loader finishes."""
        self.loading_widget.hide()
        if ffit is None:
            QMessageBox.critical(self, "Error",
                                 "Could not fetch image from Siril.")
            return
        self._set_image_from_ffit(ffit)

    def schedule_preview_update(self):
        self.preview_timer.start(200)

    def update_preview(self):
        """Update the preview with current settings."""
        if self.image_orig is None:
            return

        # Stop old worker
        if hasattr(self, "preview_thread") and self.preview_thread.isRunning():
            self.preview_thread.terminate()
            self.preview_thread.wait()

        params = self._get_preview_params()
        self.preview_thread = PreviewWorker(self.image_orig.copy(), params, parent=self)
        self.preview_thread.finished.connect(self._set_preview_pixmap)
        self.preview_thread.start()

    def _set_preview_pixmap(self, pixmap: QPixmap, star_count: int):
        """Display the preview inside the QGraphicsView."""
        # Preserve current zoom/rotation
        previous_transform = self.view.transform()

        # Clear the scene before adding the new image
        self.scene.clear()
        item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(item)
        self.scene.setSceneRect(item.boundingRect())

        # Restore the previous zoom/rotation instead of resetting
        self.view.setTransform(previous_transform)

        self.star_count_label.setText(f"{star_count}")

    def get_stars_from_siril(self) -> List[Tuple[float, float, float]]:
        """
        Retrieve the list of stars from Siril.

        If no stars are found or an error occurs, a message is logged
        to the Siril console and an empty list is returned.
        """
        try:
            # Prefer channel‑specific star detection; fall back otherwise
            stars = SIRIL.get_image_stars(channel=1)
        except Exception:
            try:
                stars = SIRIL.get_image_stars()
            except Exception as e:
                logging.error(f"Failed to retrieve stars: {e}")
                logging.error("No stars detected – spike generation skipped.")
                return []

        result = []
        for star in stars:
            x = getattr(star, "xpos", getattr(star, "x", None))
            y = getattr(star, "ypos", getattr(star, "y", None))
            if x is None or y is None:
                continue
            fwhm_x = getattr(star, "fwhmx", getattr(star, "fwhm_x", None))
            fwhm_y = getattr(star, "fwhmy", getattr(star, "fwhm_y", None))
            fwhm = getattr(star, "fwhm", None)
            if fwhm_x is not None and fwhm_y is not None:
                val = (float(fwhm_x) + float(fwhm_y)) / 2.0
            elif fwhm is not None:
                val = float(fwhm)
            else:
                val = 5.0
            result.append((float(x), float(y), val))

        if self.image_orig is not None:
            h = self.image_orig.shape[0]
            self.stars_list = [(x, h - 1 - y, fwhm) for (x, y, fwhm) in result]
        else:
            self.stars_list = result

        if not self.stars_list:
            logging.error("No stars detected – spike generation will be skipped.")
        
        return self.stars_list

    def reset_to_defaults(self):
        """Reset all controls to default values."""
        self.fwhm_min_slider.setValue(DEFAULT_CONFIG['FWHM_MIN'])
        # Reset max slider to default maximum
        self.fwhm_max_slider.setRange(1, DEFAULT_CONFIG['FWHM_MAX'])
        self.fwhm_max_slider.setValue(DEFAULT_CONFIG['FWHM_MAX'])

        self.spike_length_slider.setValue(DEFAULT_CONFIG['SPIKE_LENGTH_PCT'])
        self.spike_thickness_slider.setValue(DEFAULT_CONFIG['SPIKE_THICKNESS'])
        self.spike_count_combo.setCurrentText(str(DEFAULT_CONFIG['SPIKE_COUNT']))
        self.blur_sigma_slider.setValue(int(DEFAULT_CONFIG['BLUR_SIGMA'] * 10))
        self.rotation_slider.setValue(DEFAULT_CONFIG['ROTATION_ANGLE'])
        self.pattern_combo.setCurrentText("Radial")
        self.opacity_slider.setValue(DEFAULT_CONFIG['MASK_OPACITY'])

        # Reset color to default white
        self.spike_color = (255, 255, 255)
        self.color_display.setStyleSheet(
            "background-color: rgb(255,255,255); border: 1px solid black;"
        )

        if self.current_ffit is not None:
            fwhm_min = self.fwhm_min_slider.value()
            fwhm_max = self.fwhm_max_slider.value()
            self.fwhm_max_slider.blockSignals(True)
            self._set_image_from_ffit(self.current_ffit)
            self.fwhm_max_slider.blockSignals(False)
            self.fwhm_min_slider.setValue(fwhm_min)

        self.update_preview()

    def _get_preview_params(self) -> Dict:
        """Get current parameters for preview generation."""
        return {
            "fwhm_min": self.fwhm_min_slider.value(),
            "fwhm_max": self.fwhm_max_slider.value(),
            "spike_length": self.spike_length_slider.value(),
            "spike_thickness": self.spike_thickness_slider.value(),
            "spike_count": int(self.spike_count_combo.currentText()),
            "blur_sigma": self.blur_sigma_slider.value() / 10.0,
            "rotation_angle": self.rotation_slider.value(),
            "pattern": self.pattern_combo.currentText(),
            "stars_list": self.stars_list,
            "spike_color": self.spike_color,
            "opacity": self.opacity_slider.value() / 100.0,
            "spike_brightness": self.brightness_slider.value() / 100.0, 
        }

    def _apply_spikes(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Apply diffraction spikes to the provided image.

        Returns:
            Tuple of (processed_image, star_count)
        """
        params = self._get_preview_params()

        if not params["stars_list"]:
            # Nothing to do – log and return
            SIRIL.log("No stars detected; skipping spike generation.", s.LogColor.RED)
            return image, 0

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        angles = compute_spike_angles(
            params["spike_count"],
            params["pattern"],
            params["rotation_angle"]
        )

        star_count = 0
        for x, y, fwhm in params["stars_list"]:
            if not (params["fwhm_min"] <= fwhm <= params["fwhm_max"]):
                continue
            star_count += 1

            height = fwhm * params["spike_length"] / 100.0
            base = max(params["spike_thickness"], fwhm / 10.0)

            base_factor = max(params["spike_length"] / 100.0, 0.2)
            base *= base_factor

            heights = calculate_spike_heights(fwhm, params["spike_length"], len(angles))

            for a, h in zip(angles, heights):
                draw_spike_triangle(mask, x, y, a, h, base)

            # Apply blur to the mask
            y0 = max(0, int(y) - 20)
            y1 = min(image.shape[0], int(y) + 21)
            x0 = max(0, int(x) - 20)
            x1 = min(image.shape[1], int(x) + 21)
            image[y0:y1, x0:x1] = gaussian_filter(
                image[y0:y1, x0:x1], sigma=params["blur_sigma"]
            )

        if image.dtype.kind != 'f':
            image = image.astype(np.float32)

        color_norm = np.array(params["spike_color"], dtype=np.float32) / 255.0
        brightness = params.get("spike_brightness", 1.0)
        image[..., 0] += mask * color_norm[0] * brightness
        image[..., 1] += mask * color_norm[1] * brightness
        image[..., 2] += mask * color_norm[2] * brightness


        np.clip(image, 0.0,
                1.0 if image.dtype.kind == 'f' else 65535,
                out=image)

        return image, star_count

    def save_spiked_image(self):
        """Save the spiked image with separate mask layer for GIMP and export composed file."""
        if self.image_orig is None:
            return

        # Get the original image dimensions
        height, width = self.image_orig.shape[:2]

        # Create a blank mask image (same size as original)
        spikes_mask = np.zeros((height, width), dtype=np.float32)

        # Apply spikes to the mask (without color)
        params = self._get_preview_params()
        angles = compute_spike_angles(
            params["spike_count"],
            params["pattern"],
            params["rotation_angle"]
        )

        star_count = 0
        for x, y, fwhm in params["stars_list"]:
            if not (params["fwhm_min"] <= fwhm <= params["fwhm_max"]):
                continue
            star_count += 1

            height_val = fwhm * params["spike_length"] / 100.0
            base = max(params["spike_thickness"], fwhm / 10.0)
            base_factor = max(params["spike_length"] / 100.0, 0.2)
            base *= base_factor

            heights = calculate_spike_heights(fwhm, params["spike_length"], len(angles))

            for a, h in zip(angles, heights):
                draw_spike_triangle(spikes_mask, x, y, a, h, base)

            # Apply blur to the mask
            y0 = max(0, int(y) - 20)
            y1 = min(height, int(y) + 21)
            x0 = max(0, int(x) - 20)
            x1 = min(width, int(x) + 21)
            spikes_mask[y0:y1, x0:x1] = gaussian_filter(
                spikes_mask[y0:y1, x0:x1], sigma=params["blur_sigma"]
            )

        # Normalize the mask to 0-1 range
        spikes_mask = np.clip(spikes_mask, 0.0, 1.0)

        # Create the composed image (original + spikes)
        processed = self.image_orig.copy()
        if processed.dtype.kind != 'f':
            processed = processed.astype(np.float32)

        # Add colored spikes to the composed image
        color_norm = np.array(params["spike_color"], dtype=np.float32) / 255.0
        brightness = params.get("spike_brightness", 1.0)
        processed[..., 0] += spikes_mask * color_norm[0] * brightness
        processed[..., 1] += spikes_mask * color_norm[1] * brightness
        processed[..., 2] += spikes_mask * color_norm[2] * brightness

        np.clip(processed, 0.0, 1.0, out=processed)
        processed = np.flipud(processed)  # Vertical mirror for composed image

        # Prepare the pure spikes mask (vertically mirrored, white spikes)
        spikes_mask = np.flipud(spikes_mask)  # Flip the mask vertically
        spikes_mask_8bit = (spikes_mask * 255).astype(np.uint8)  # Convert to 0-255

        # Determine output filenames
        if self.filename:
            base, _ = os.path.splitext(self.filename)
            composed_file = f"{base}_spikes.tif"
            mask_file = f"{base}_spikes_mask.png"  # Pure alpha mask
        else:
            composed_file = "spiked.tif"
            mask_file = "spikes_mask.png"

        try:
            # Save the composed image (32‑bit TIFF)
            import imageio
            img_to_write = processed.astype(np.float32)
            imageio.imwrite(composed_file, img_to_write)

            # Save the pure spikes mask as PNG with transparency
            transparent_mask = np.zeros((height, width, 4), dtype=np.uint8)

            # Set the RGB channels to white (255,255,255) where spikes are
            transparent_mask[..., :3] = 255 * np.stack([spikes_mask_8bit > 0]*3, axis=-1)

            # Set the alpha channel to our spikes mask (white = 255)
            transparent_mask[..., 3] = spikes_mask_8bit

            # Save as PNG with transparency
            mask_img = Image.fromarray(transparent_mask, 'RGBA')
            mask_img.save(mask_file)

            # Load the composed file into Siril
            try:
                SIRIL.cmd("load" , f'"{composed_file}"')
                self.log_current_parameters()
            except Exception as e:
                logging.warning(f"Could not load {composed_file} into Siril: {e}")

            SIRIL.log(f"Composed image: {composed_file}")
            SIRIL.log(f"Spikes mask: {mask_file}")

        except Exception as e:
            logging.error(f"Failed to save spiked image: {e}")
            QMessageBox.critical(self, "Export Failed", f"Error saving files:\n{e}")

    def resizeEvent(self, event):
        """Debounce preview update when window size changes."""
        if self.size() != getattr(self, "_prev_size", None):
            self.resize_timer.start(100)
        self._prev_size = self.size()
        super().resizeEvent(event)

    def show_help(self):
        help_widget = self.help_tab()
        dlg = QDialog(self)
        dlg.setWindowTitle("Help")
        layout = QVBoxLayout(dlg)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True) 
        scroll.setWidget(help_widget)
        layout.addWidget(scroll)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        dlg.resize(750, 780)
        dlg.exec()

    def help_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        help_text = """
        <h1 style="color: #88aaff;">Diffraction‑Spike Overlay Tool</h1>

        <p>This tool adds synthetic diffraction spikes to the current image loaded in <strong>Siril</strong>.  
        It provides a live preview, star‑based placement, and the ability to save both the spiked image
        and an alpha‑masked PNG that can be opened in GIMP or other editors.</p>


        <h2 style="color: #88aaff;">Controls</h2>
        <table>
        <tr><th>Control</th><th>Description</th></tr>
        <tr><td><code>FWHM Min / Max</code></td><td>Filters stars by their FWHM (px). Only stars within this range are spiked.</td></tr>
        <tr><td><code>Detected Stars</code></td><td>Number of stars that will receive spikes.</td></tr>
        <tr><td><code>Spike Length (%)</code></td><td>Length of each spike relative to the star’s FWHM.</td></tr>
        <tr><td><code>Spike Thickness (px)</code></td><td>Width of the spike base.</td></tr>
        <tr><td><code>Number of Spikes</code></td><td>4, 6 or 8 radial spikes per star.</td></tr>
        <tr><td><code>Blur Sigma</code></td><td>Gaussian blur applied to each spike (σ).</td></tr>
        <tr><td><code>Rotation Angle (°)</code></td><td>Rotate the whole pattern.</td></tr>
        <tr><td><code>Brightness</code></td><td>Brightness of the spikes.</td></tr>
        <tr><td><code>Pattern Shape</code></td><td>Radial, Cross or X.</td></tr>        
        <tr><td><code>Spike Color</code></td><td>RGB picker; shown in the small box.</td></tr>
        <tr><td><code>Opacity (%)</code></td><td>Opacity of the star mask overlay in the preview.</td></tr>
    
        <ul>
        </table>

        <h2 style="color: #88aaff;">Workflow</h2>
        <ol>
        <li><strong>Load an image in Siril.</strong> The script automatically opens TIFF and FITS images.</li>
        <li><strong>Adjust controls.</strong> The preview updates in real time after each change.</li>
        <li><strong>Verify <code>Detected Stars</code>.</strong> Ensure the list matches the stars you want spiked.</li>
        <li><strong>Use zoom/fit buttons</strong> on the preview pane for a detailed view.</li>
        <li><strong>Press <code>Save</code></strong> to export the spiked image and mask.</li>
        <li><strong>Open the mask in GIMP</strong> (or similar) as a layer with transparency for further editing.</li>
        </ol>

        <h2 style="color: #88aaff;">Notes & Limitations</h2>
        <ul>
        <li>The tool works only on RGB stretched images; grayscale is unsupported.</li>
        <li>Large images may take several seconds to process in the preview; full‑resolution saving is faster because it skips live rendering.</li>
        <li>Star detection uses Siril’s <code>get_image_stars()</code>; if no stars are found the spike generation is skipped.</li>
        </ul>        

        <h4 style="color: #88aaff;">Credits</h4>
        <p>Developed for SIRIL.<br>
        (c) G. Trainar (2026)</p>
        """
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #cccccc; font-size: 12pt;")
        layout.addWidget(help_label)
        return widget

    def log_current_parameters(self):
        """Log current parameters to Siril console."""
        # Get parameter values
        fwhm_min = self.fwhm_min_slider.value()
        fwhm_max = self.fwhm_max_slider.value()
        length = self.spike_length_slider.value()
        thickness = self.spike_thickness_slider.value()
        spike_count = int(self.spike_count_combo.currentText())
        blur_sigma = self.blur_sigma_slider.value() / 10.0
        rotation = self.rotation_slider.value()
        pattern = self.pattern_combo.currentText()
        spike_color = self.spike_color
        star_count = self.star_count_label.text()

        # Log parameters to Siril console
        SIRIL.log(f"#################################################################")
        SIRIL.log(f"The composed file and the mask with alpha channel")
        SIRIL.log(f"have been successfully saved with the following settings:")
        SIRIL.log(f"FWHM Min: {fwhm_min}px")
        SIRIL.log(f"FWHM Max: {fwhm_max}px")
        SIRIL.log(f"Spike Length: {length}%")
        SIRIL.log(f"Spike Thickness: {thickness}px")
        SIRIL.log(f"Number of Spikes: {spike_count}")
        SIRIL.log(f"Blur Sigma: {blur_sigma:.1f}σ")
        SIRIL.log(f"Rotation Angle: {rotation}°")
        SIRIL.log(f"Pattern Shape: {pattern}")
        SIRIL.log(f"Spike Color: RGB{spike_color}")
        SIRIL.log(f"Detected Stars: {star_count}")
        SIRIL.log(f"###################################################################")

def is_tiff(filename: str) -> bool:
    """Check if a filename has a TIFF extension."""
    return os.path.splitext(filename)[1].lower() in {".tif", ".tiff"}

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    icon_data = base64.b64decode("""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""")
    pixmap = QPixmap()
    pixmap.loadFromData(icon_data)
    app_icon = QIcon(pixmap)
    app.setWindowIcon(app_icon) 
    

    SIRIL.log(f"##############################################")
    SIRIL.log(f"#  Diffraction Spike Overlay Tool for Siril  #")
    SIRIL.log(f"##############################################")

    try:
        # Get the filename of the image currently loaded in Siril
        current_fname = SIRIL.get_image_filename()
    except SirilConnectionError:
        # Prompt user to select an image file
        fname, _ = QFileDialog.getOpenFileName(
            None,
            "Select image to load",
            "",
            "Images (*.fits *.fit *.tif *.tiff)"
        )
        if not fname:
            QMessageBox.critical(
                None,
                "No Image Loaded",
                "Please open an image in Siril before running this tool."
            )
            return 1
        try:
            SIRIL.cmd("load", f'"{fname}"')
            current_fname = fname
        except Exception as e:
            QMessageBox.critical(
                None,
                "Load Failed",
                f"Could not load selected image: {e}"
            )
            return 1

    # Fetch the image
    try:
        with wait_for_lock(SIRIL):
            ffit = SIRIL.get_image()
            if ffit is None:
                raise NoImageError
    except NoImageError:
        QMessageBox.critical(
            None,
            "No Image Loaded",
            "Please open an image in Siril before running this tool."
        )
        return 1
    except s.ProcessingThreadBusyError:
        time.sleep(0.5)
        return 1

    # Check if image is RGB or grayscale
    try:
        img_data = ffit.data
        if len(img_data.shape) < 3 or (len(img_data.shape) == 3 and img_data.shape[2] in [1, 2]):
            QMessageBox.critical(
                None,
                "Unsupported Image Format",
                "This tool requires a color (RGB) image. Grayscale or single‑channel images are not supported.\n\n"
                "Please load a color image and try again."
            )
            return 1
    except Exception as e:
        QMessageBox.critical(
            None,
            "Image Processing Error",
            f"Could not verify image format: {e}"
        )
        return 1

    # Create and show the main window
    win = SpikeWindow(preloaded_ffit=ffit)
    win.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
