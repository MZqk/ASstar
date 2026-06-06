# (c) Cecile Melis 2025
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Displays the frames of the selected images of a sequence to check framing
Allows to plot various registration data fields and mask out bad frames by drawing a polygon on the data plot

Version history:
1.0.0: Initial version
1.0.1: If no CLI arguments, run in GUI mode by default
"""

import sirilpy as s
import numpy as np
import os, sys
import warnings

VERSION = "1.0.1"
DATA_FIELDS = [
            'FWHM',
            'wFWHM',
            'roundness',
            'quality',
            'background level',
            'number of stars',
            'X position',
            'Y position',
            'Relative angle'
]

# Check sirilpy version once at startup
if not s.check_module_version(f'>=1.0.16'):
    print(f"Please install sirilpy version 1.0.16 or higher")
    sys.exit(1)

s.ensure_installed('matplotlib')

from matplotlib.figure import Figure

s.ensure_installed('PyQt6')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QComboBox, QCheckBox, QLabel, QGroupBox, QPushButton, QSlider, QSplitter)
from PyQt6.QtCore import Qt

siril = s.SirilInterface()

try:
    siril.connect()
    print("Connected successfully!")
except Exception as e:
    print(f"Connection failed: {e}")
    quit()


class SequenceData:
    """Holds all sequence data and provides methods to compute/access frame information"""
    
    def __init__(self, siril_interface):
        self.siril = siril_interface
        self.seq = None
        self.reglayer = -1
        self.frame_segments = []  # List of segment arrays for each frame
        self.frame_indices = []   # Original frame indices from sequence
        self.filtered_frame_indices = None  # Indices to actually display (None = show all)
        self.ref = None
        self.Href = None
        self.cx = None
        self.cy = None
        self.image_data = None
    
    def load_sequence(self):
        """Load and validate sequence data"""
        try:
            self.seq = None
            self.reglayer = -1
            
            if not self.siril.is_sequence_loaded():
                raise Exception("No sequence loaded")
            
            seq = self.siril.get_seq()
            if seq is None:
                raise Exception("Failed to retrieve image sequence")
            
            if seq.regparam is None or len(seq.regparam) == 0:
                raise Exception("No registration data found in the sequence")
            
            # Find valid registration layer
            found = False
            for i, r in enumerate(seq.regparam):
                if any(rr is not None for rr in r):
                    self.seq = seq
                    self.reglayer = i
                    found = True
                    break
            
            if not found:
                raise Exception("No valid registration data found in the sequence")
            
            self.precompute_frame_data()
            self.filtered_frame_indices = None
            # Load and display reference image
            ref = self.seq.reference_image
            imgref = self.siril.get_seq_frame(ref, with_pixels=True, preview=True, linked=False)
            if imgref is None or imgref.data is None:
                raise Exception(f"Failed to load reference image {ref}")

            if imgref.naxis == 2 or imgref.naxes[2] == 1:
                image_data = imgref.data[::-1]
                # ax.imshow(image_data, cmap='gray')
            else:
                image_data = imgref.data.transpose(1, 2, 0)
                image_data = image_data[::-1]
                # ax.imshow(image_data)
            self.image_data = image_data
            return True
        
        except Exception as e:
            self.siril.log(f"Error loading sequence: {e}", s.LogColor.RED if hasattr(s, 'LogColor') else None)
            return False
    
    def is_valid(self):
        """Check if sequence data is valid"""
        return self.seq is not None and self.reglayer != -1
    
    def get_H_matrix(self, img_number):
        """Get the homography matrix for the given image number"""
        if self.seq is None or self.reglayer == -1:
            return None
        regparam = self.seq.regparam[self.reglayer]
        if img_number < 0 or img_number >= len(regparam):
            return None
        reg = regparam[img_number]
        if reg is None:
            return None
        H = np.zeros((3,3))
        H[0,0] = reg.H.h00
        H[0,1] = reg.H.h01
        H[0,2] = reg.H.h02
        H[1,0] = reg.H.h10
        H[1,1] = reg.H.h11
        H[1,2] = reg.H.h12
        H[2,0] = reg.H.h20
        H[2,1] = reg.H.h21
        H[2,2] = reg.H.h22
        if np.allclose(H.flatten(), np.zeros((3,3)).flatten()):
            return None
        return H
    
    def precompute_frame_data(self):
        """Pre-compute all corner coordinates"""
        self.frame_segments = []
        self.frame_indices = []

        ref = self.seq.reference_image
        Href = self.get_H_matrix(ref)
        self.ref = ref
        self.Href = Href
        self.cx = 0.5 * (self.seq.imgparam[ref].rx if self.seq.imgparam[ref].rx > 0 else self.seq.rx)
        self.cy = 0.5 * (self.seq.imgparam[ref].ry if self.seq.imgparam[ref].ry > 0 else self.seq.ry)
        
        for i in range(self.seq.number):
            if not self.seq.imgparam[i].incl:
                continue
            
            H_img = self.get_H_matrix(i)
            if H_img is None:
                continue
            
            self.frame_indices.append(i)
            
            H_rel = np.linalg.inv(self.Href) @ H_img
            rx = self.seq.imgparam[i].rx if self.seq.imgparam[i].rx > 0 else self.seq.rx
            ry = self.seq.imgparam[i].ry if self.seq.imgparam[i].ry > 0 else self.seq.ry
            r_top = int(min(rx,ry)/20)
            corners = np.array([[0, 0, 1],
                                [rx, 0, 1],
                                [rx, ry, 1],
                                [0, ry, 1],
                                [0, 0, 1],
                                [r_top, 0, 1],
                                [0, r_top, 1]]).T
            corners = H_rel @ corners
            x = corners[0, :] / corners[2, :]
            y = corners[1, :] / corners[2, :]
            
            segments = np.column_stack([x, y])
            self.frame_segments.append(segments)
    
    def get_num_display_frames(self):
        """Get the number of frames to display based on current filter"""
        if self.filtered_frame_indices is not None:
            return len(self.filtered_frame_indices)
        return len(self.frame_indices)

    def get_display_indices(self, up_to=None):
        """Get the indices to display, optionally limited to 'up_to' count"""
        if self.filtered_frame_indices is not None:
            indices = self.filtered_frame_indices
        else:
            indices = list(range(len(self.frame_indices)))
        
        if up_to is not None:
            indices = indices[:up_to]
        
        return indices
    
    def get_field_value(self, frame_idx, field):
        """Get the value for a specific field from a frame"""
        if self.seq is None or self.reglayer == -1:
            return None
            
        regparam = self.seq.regparam[self.reglayer][frame_idx]
        
        if field == 'background level':
            return regparam.background_lvl
        elif field == 'FWHM':
            return regparam.fwhm
        elif field == 'wFWHM':
            return regparam.weighted_fwhm
        elif field == 'roundness':
            return regparam.roundness
        elif field == 'quality':
            return regparam.quality
        elif field == 'number of stars':
            return regparam.number_of_stars
        elif field == 'X position' or field == 'Y position' or field == 'Relative angle':
            H_img = self.get_H_matrix(frame_idx)
            if H_img is not None:
                H_rel = np.linalg.inv(self.Href) @ H_img
                if field == 'Relative angle':
                    return np.arctan2(H_rel[1,0], H_rel[0,0]) * (180.0 / np.pi)
                else: 
                    dx = 0.5 * (self.seq.imgparam[frame_idx].rx if self.seq.imgparam[frame_idx].rx > 0 else self.seq.rx)
                    dy = 0.5 * (self.seq.imgparam[frame_idx].ry if self.seq.imgparam[frame_idx].ry > 0 else self.seq.ry)
                    center_img = np.array([dx, dy, 1.]).T
                    center_img = H_rel @ center_img
                    if field == 'X position':
                        return center_img[0] / center_img[2] - self.cx
                    else:
                        return center_img[1] / center_img[2] - self.cy
        
        return None
    
    def get_sequence_name(self):
        """Get the sequence name"""
        if self.seq is None:
            return "Unknown"
        return os.path.basename(self.seq.seqname) if self.seq.seqname else "Unknown"
    
    def get_info_text(self):
        """Get sequence info text"""
        if self.seq is None:
            return "No sequence loaded"
        
        seq_name = self.get_sequence_name()
        original_count = len(self.frame_indices)
        total_count = self.seq.number

        if self.filtered_frame_indices is not None:
            filtered_count = len(self.filtered_frame_indices)
            masked_count = original_count - filtered_count
            return f"{seq_name}: {original_count} registered frames [{filtered_count} after masking {masked_count} frames]"
        else:
            return f"{seq_name}: {original_count} registered frames / {total_count} total"
    
    def plot_frames_on_axis(self, ax, siril_interface, num_frames=None):
        """
        Plot frame segments on a given matplotlib axis.
        
        Args:
            ax: matplotlib axis to plot on
            siril_interface: SirilInterface instance for loading images
            num_frames: number of frames to display (None = all)
        
        Returns:
            The axis with plotted frames
        """
        if not self.is_valid():
            return ax
        
        # Load and display reference image

        ax.imshow(self.image_data, cmap='gray')
        ax.set_aspect('equal', adjustable='datalim')
        ax.yaxis.set_inverted(True)
        
        # Get number of frames to display
        if num_frames is None:
            num_frames = self.get_num_display_frames()
        
        # Get indices to display
        display_indices = self.get_display_indices(up_to=num_frames)
        
        # Draw frames with single color
        for idx in display_indices:
            segment = self.frame_segments[idx]
            ax.plot(segment[:, 0], segment[:, 1], 'b-', linewidth=1, alpha=0.7)
        
        return ax
    
    def draw_frames_on_axis(self, ax, num_frames=None):
        """
        Draw frame segments on an existing axis and return the line objects.
        Used for updating existing plot without reloading reference image.
        
        Args:
            ax: matplotlib axis to draw on
            num_frames: number of frames to display (None = all)
        
        Returns:
            List of line objects created
        """
        if not self.is_valid():
            return []
        
        # Get number of frames to display
        if num_frames is None:
            num_frames = self.get_num_display_frames()
        
        # Get indices to display
        display_indices = self.get_display_indices(up_to=num_frames)
        
        # Draw frames and collect line objects
        lines = []
        for idx in display_indices:
            segment = self.frame_segments[idx]
            line = ax.plot(segment[:, 0], segment[:, 1], 'b-', linewidth=1, alpha=0.7)[0]
            lines.append(line)
        
        return lines

class RegInspectorWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'Siril Registration Inspector v{VERSION}')
        self.setGeometry(100, 100, 1400, 800)

        # Set always on top by default
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        self.data = SequenceData(siril)
        self.ax1 = None
        
        self.anim = None
        self.frames_per_step = 1
        self.frame_lines = []  # Store line objects for animation
        self.updating_slider = False
        self.is_playing = False
        self.is_paused = False
        self.current_frame_line = None
        self.current_frame_marker = None
        self.splitter_move_timer = None
        self.resize_timer = None
        self.poly_selector = None

        self.init_ui()
        self.load_sequence_data()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Control panel
        control_layout = QVBoxLayout()
        
        # Sequence options
        sequence_group = QGroupBox("Sequence Options")
        sequence_layout = QVBoxLayout()
        
        # Progress slider at the top
        progress_layout = QVBoxLayout()
        self.progress_label = QLabel("Frames: 0 / 0")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.progress_label)
        
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(0)
        self.progress_slider.setValue(0)
        self.progress_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.progress_slider.valueChanged.connect(self.on_progress_slider_changed)
        progress_layout.addWidget(self.progress_slider)
        
        sequence_layout.addLayout(progress_layout)
        
        # Animation control buttons
        animation_buttons_layout = QHBoxLayout()
        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self.on_play_clicked)
        animation_buttons_layout.addWidget(self.play_button)
        
        self.pause_button = QPushButton("⏸ Pause")
        self.pause_button.clicked.connect(self.on_pause_clicked)
        self.pause_button.setEnabled(False)
        animation_buttons_layout.addWidget(self.pause_button)
        
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.clicked.connect(self.on_stop_clicked)
        self.stop_button.setEnabled(False)
        animation_buttons_layout.addWidget(self.stop_button)
        
        sequence_layout.addLayout(animation_buttons_layout)
        
        # Animation speed control with slider
        speed_layout = QVBoxLayout()
        speed_layout.addWidget(QLabel("Animation Speed:"))
        
        speed_slider_layout = QHBoxLayout()
        speed_slider_layout.addWidget(QLabel("Low"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(2)
        self.speed_slider.setValue(1)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        speed_slider_layout.addWidget(self.speed_slider)
        speed_slider_layout.addWidget(QLabel("High"))
        
        speed_layout.addLayout(speed_slider_layout)
        sequence_layout.addLayout(speed_layout)
        
        # Frame batch control
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("Draw frames per step:"))
        self.skip_combo = QComboBox()
        self.skip_combo.addItems(['1', '5', '10', '20', '50', '100'])
        self.skip_combo.setCurrentText('10')
        self.skip_combo.currentTextChanged.connect(self.on_skip_changed)
        skip_layout.addWidget(self.skip_combo)
        sequence_layout.addLayout(skip_layout)

        sequence_group.setLayout(sequence_layout)
        control_layout.addWidget(sequence_group)

        # Refresh button and sequence info
        refresh_group = QGroupBox("Sequence")
        refresh_layout = QVBoxLayout()
        
        self.sequence_info_label = QLabel("No sequence loaded")
        self.sequence_info_label.setWordWrap(True)
        self.sequence_info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sequence_info_label.setToolTip("Displays loaded sequence name and frame selection info")
        refresh_layout.addWidget(self.sequence_info_label)
        
        self.refresh_button = QPushButton("Refresh Sequence Data")
        self.refresh_button.setToolTip("Refresh the sequence data if you've made changes in Siril (e.g. re-registered, changed reference frame, loaded new sequence etc.)")
        self.refresh_button.clicked.connect(self.on_refresh_clicked)
        refresh_layout.addWidget(self.refresh_button)
        
        refresh_group.setLayout(refresh_layout)
        control_layout.addWidget(refresh_group)

        # Data plot group
        data_plot_group = QGroupBox("Sequence Data")
        data_plot_layout = QVBoxLayout()
        
        # Dropdown for selecting X field
        x_field_layout = QHBoxLayout()
        x_field_layout.addWidget(QLabel("X Axis:"))
        self.x_field_combo = QComboBox()
        self.x_field_combo.setToolTip("Select which data field to plot on the X axis")
        x_field_options = ['Frame Number'] + DATA_FIELDS
        self.x_field_combo.addItems(x_field_options)
        self.x_field_combo.setCurrentText('Frame Number')
        self.x_field_combo.currentTextChanged.connect(self.on_field_changed)
        x_field_layout.addWidget(self.x_field_combo)
        data_plot_layout.addLayout(x_field_layout)
        
        # Dropdown for selecting Y field
        y_field_layout = QHBoxLayout()
        y_field_layout.addWidget(QLabel("Y Axis:"))
        self.y_field_combo = QComboBox()
        self.y_field_combo.setToolTip("Select which data field to plot on the Y axis")
        self.y_field_combo.addItems(DATA_FIELDS)
        self.y_field_combo.currentTextChanged.connect(self.on_field_changed)
        y_field_layout.addWidget(self.y_field_combo)
        data_plot_layout.addLayout(y_field_layout)
        
        # Small plot canvas
        self.data_figure = Figure(figsize=(4, 3), dpi=80)
        self.data_canvas = FigureCanvas(self.data_figure)
        self.data_canvas.setToolTip("Draw a polygon on this plot to mask out frames.\n" \
        "Click on The Preview Button to show the removal effect on the main plot.\n" \
        "Click Apply to permanently remove masked frames from the sequence.\n" \
        "Double-click to clear the polygon mask.")
        self.data_ax = self.data_figure.add_subplot(111)
        data_plot_layout.addWidget(self.data_canvas, 1)
        
        # Mask control buttons
        mask_buttons_layout = QHBoxLayout()

        self.preview_toggle = QCheckBox("Preview removal")
        self.preview_toggle.setToolTip("Toggle preview of frame removal")
        self.preview_toggle.stateChanged.connect(self.on_preview_toggled)
        mask_buttons_layout.addWidget(self.preview_toggle)
        
        self.apply_mask_button = QPushButton("Apply")
        self.apply_mask_button.clicked.connect(self.on_apply_mask_clicked)
        self.apply_mask_button.setToolTip("Effectively remove masked frames from the sequence")
        mask_buttons_layout.addWidget(self.apply_mask_button)

        data_plot_layout.addLayout(mask_buttons_layout)
        data_plot_group.setLayout(data_plot_layout)
        control_layout.addWidget(data_plot_group, 1)
        control_layout.addStretch()

        # Always on Top at the bottom
        self.ontop_check = QCheckBox("Always on Top")
        self.ontop_check.setChecked(True)
        self.ontop_check.stateChanged.connect(self.on_ontop_toggled)
        control_layout.addWidget(self.ontop_check)
        
        # Plot canvas with toolbar
        self.figure = Figure(figsize=(12, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        canvas_layout = QVBoxLayout()
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)

        # Create a splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        
        canvas_widget = QWidget()
        canvas_widget.setLayout(canvas_layout)
        
        splitter.addWidget(control_widget)
        splitter.addWidget(canvas_widget)
        splitter.setSizes([420, 980])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        
        self.splitter = splitter
        self.splitter.splitterMoved.connect(self.on_splitter_moved)

        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #888888;
                width: 8px;
                margin: 2px 0px;
                border-radius: 4px;
            }
            QSplitter::handle:hover {
                background-color: #0078d4;
                width: 8px;
            }
            QSplitter::handle:pressed {
                background-color: #005a9e;
                width: 8px;
            }
        """)
        
        main_layout.addWidget(splitter)

    def update_sequence_info_label(self):
        """Update the sequence info label with current sequence data"""
        self.sequence_info_label.setText(self.data.get_info_text())
    
    def load_sequence_data(self):
        """Load sequence data"""
        if self.data.load_sequence():
            self.plot()
            self.plot_data_field()
            self.update_sequence_info_label()
        else:
            self.update_sequence_info_label()
    
    def get_H_matrix(self, img_number):
        """Proxy method to get H matrix from data"""
        return self.data.get_H_matrix(img_number)

    def plot(self):
        # Stop any existing animation
        if hasattr(self, 'anim') and self.anim is not None:
            try:
                self.anim.event_source.stop()
            except:
                pass
        self.anim = None
        self.is_playing = False
        self.is_paused = False
        self.update_button_states()
            
        self.figure.clear()
        if not self.data.is_valid():
            return
        
        title = f"{self.data.get_sequence_name()}"
        self.figure.suptitle(title, fontsize=12)
        self.ax1 = self.figure.add_subplot(111)
        self._suppress_tight_layout_warning(self.figure)
        
        # Get the number of frames to display
        num_frames = self.get_num_display_frames()
        
        # Update progress slider
        self.updating_slider = True
        self.progress_slider.setMaximum(num_frames)
        tick_interval = max(1, num_frames // 10)
        self.progress_slider.setTickInterval(tick_interval)
        self.progress_slider.setValue(num_frames)
        self.update_progress_label(num_frames)
        self.updating_slider = False
        
        # Plot all frames
        self.plot_static()
        self.canvas.draw()

    def get_num_display_frames(self):
        """Get the number of frames to display based on current filter"""
        return self.data.get_num_display_frames()

    def get_display_indices(self, up_to=None):
        """Get the indices to display, optionally limited to 'up_to' count"""
        return self.data.get_display_indices(up_to)

    def plot_static(self, num_frames=None):
        """Draw frames up to num_frames (or all if None)"""
        if num_frames is None:
            num_frames = self.get_num_display_frames()
        
        # Clear existing frame lines
        for line in self.frame_lines:
            try:
                line.remove()
            except:
                pass
        self.frame_lines = []

        # Check if axis needs setup (fresh axis has no images)
        if not self.ax1.images:
            # Set up the reference image and axis properties
            self.ax1.imshow(self.data.image_data, cmap='gray')
            self.ax1.set_aspect('equal', adjustable='datalim')
            self.ax1.yaxis.set_inverted(True)

        # Draw the frames and store the returned line objects
        try:
            self.frame_lines = self.data.draw_frames_on_axis(self.ax1, num_frames)
        except Exception as e:
            siril.log(f"Error redrawing frames: {e}", s.LogColor.RED)

    def animate_frames(self):
        """Animate drawing frames sequentially"""
        if not self.data.frame_segments:
            return
        
        # Clear existing frame lines
        for line in self.frame_lines:
            try:
                line.remove()
            except:
                pass
        self.frame_lines = []
        
        # Store axis limits
        xlim = self.ax1.get_xlim()
        ylim = self.ax1.get_ylim()
        
        # Get FPS
        speed_map = {0: 10, 1: 30, 2: 50}
        fps = speed_map[self.speed_slider.value()]
        interval = 1000 / fps
        
        # Get frames per step
        self.frames_per_step = int(self.skip_combo.currentText())
        
        # Calculate number of steps
        num_display_frames = self.get_num_display_frames()
        num_steps = (num_display_frames + self.frames_per_step - 1) // self.frames_per_step
        
        # Start position
        start_frame = 0 if not self.is_paused else self.progress_slider.value()
        start_step = start_frame // self.frames_per_step
        
        # Get display indices
        self.display_indices_for_anim = self.get_display_indices()
        
        # If resuming, redraw frames up to current position
        if self.is_paused and start_frame > 0:
            for i in range(start_frame):
                idx = self.display_indices_for_anim[i]
                segment = self.data.frame_segments[idx]
                line = self.ax1.plot(segment[:, 0], segment[:, 1], 
                             'b-', linewidth=1, alpha=0.7)[0]
                self.frame_lines.append(line)
        else:
            self.updating_slider = True
            self.progress_slider.setValue(0)
            self.updating_slider = False
        
        self.is_paused = False
        
        self.anim = FuncAnimation(
            self.figure,
            self.update_animation_batched,
            frames=range(start_step, num_steps),
            interval=interval,
            repeat=False,
            blit=True,
            init_func=lambda: self.init_animation(xlim, ylim)
        )
        
        # Connect to animation completion
        self.anim._stop = self.anim.event_source.stop
        original_stop = self.anim._stop
        def stop_wrapper():
            original_stop()
            self.on_animation_complete()
        self.anim.event_source.stop = stop_wrapper
        
        self.canvas.draw()
    
    def init_animation(self, xlim, ylim):
        """Initialize animation with preserved axis limits"""
        self.ax1.set_xlim(xlim)
        self.ax1.set_ylim(ylim)
        # Return the background artists (empty list since we're accumulating lines)
        return self.frame_lines

    def update_animation_batched(self, step_num):
        """Update function for batched animation"""
        start_idx = step_num * self.frames_per_step
        end_idx = min(start_idx + self.frames_per_step, len(self.display_indices_for_anim))
        
        # Draw new frames
        for i in range(start_idx, end_idx):
            idx = self.display_indices_for_anim[i]
            segment = self.data.frame_segments[idx]
            line = self.ax1.plot(segment[:, 0], segment[:, 1], 
                         'b-', linewidth=1, alpha=0.7, animated=True)[0]
            self.frame_lines.append(line)

        # Update progress (without triggering progress label update during animation)
        self.updating_slider = True
        self.progress_slider.setValue(end_idx)
        self.updating_slider = False
        
        # Only update label text, don't redraw data canvas during animation
        total = self.get_num_display_frames()
        self.progress_label.setText(f"Frames: {end_idx} / {total}")
        
        # Update frame indicator and redraw data canvas during animation
        self.update_current_frame_indicator()
        self.data_canvas.draw()

        # Return ALL accumulated lines so they persist with blitting
        return self.frame_lines

    def update_progress_label(self, current):
        """Update the progress label text"""
        total = self.get_num_display_frames()
        self.progress_label.setText(f"Frames: {current} / {total}")

    def on_progress_slider_changed(self, value):
        """Handle manual slider changes"""
        if self.updating_slider:
            return
        
        # Stop animation
        if hasattr(self, 'anim') and self.anim is not None:
            try:
                self.anim.event_source.stop()
            except:
                pass
            self.anim = None
        
        self.is_playing = False
        self.is_paused = False
        self.update_button_states()
        
        self.update_progress_label(value)
        self.plot_static(value)
        self.update_current_frame_indicator()
        self.data_canvas.draw()
        self.canvas.draw()

    def update_button_states(self):
        """Update button enabled/disabled states"""
        self.play_button.setEnabled(not self.is_playing)
        self.pause_button.setEnabled(self.is_playing)
        self.stop_button.setEnabled(self.is_playing or self.is_paused)

    def on_play_clicked(self):
        """Start or resume animation"""
        if not self.data.frame_segments:
            return
        
        if self.progress_slider.value() < self.get_num_display_frames():
            self.is_paused = True
        
        self.is_playing = True
        self.update_button_states()
        self.animate_frames()

    def on_pause_clicked(self):
        """Pause the animation"""
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except:
                pass
        
        self.is_playing = False
        self.is_paused = True
        self.update_button_states()

    def on_stop_clicked(self):
        """Stop the animation and reset"""
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except:
                pass
            self.anim = None
        
        self.is_playing = False
        self.is_paused = False
        self.update_button_states()
        
        num_frames = self.get_num_display_frames()
        self.updating_slider = True
        self.progress_slider.setValue(num_frames)
        self.update_progress_label(num_frames)
        self.updating_slider = False
        
        self.plot_static()
        self.canvas.draw()

    def on_animation_complete(self):
        """Called when animation reaches the end"""
        self.is_playing = False
        self.is_paused = False
        self.update_button_states()

    def on_refresh_clicked(self):
        """Reload sequence data"""
        self.data.filtered_frame_indices = None
        if self.poly_selector is not None:
            self.poly_selector.clear()
        self.load_sequence_data()

    def on_speed_changed(self, value):
        """Refresh animation when speed changes"""
        if self.is_playing:
            self.on_pause_clicked()
            self.is_paused = True
            self.on_play_clicked()

    def on_skip_changed(self, value):
        """Refresh plot when frame skip changes"""
        if self.is_playing:
            self.on_pause_clicked()
            self.is_paused = True
            self.on_play_clicked()

    def on_field_changed(self, value):
        """Handle field selection change"""
        self.plot_data_field()

    def plot_data_field(self):
        """Plot the selected data fields"""
        self.data_ax.clear()

        if not self.data.is_valid():
            self.data_ax.text(0.5, 0.5, 'No data available', 
                            ha='center', va='center', transform=self.data_ax.transAxes)
            self.data_canvas.draw()
            return
        
        x_field = self.x_field_combo.currentText()
        y_field = self.y_field_combo.currentText()
        
        # Collect data
        x_data = []
        y_data = []
        
        for i, frame_idx in enumerate(self.data.frame_indices):
            # Get X data
            if x_field == 'Frame Number':
                x_val = frame_idx
            else:
                x_val = self.data.get_field_value(frame_idx, x_field)
            
            # Get Y data
            y_val = self.data.get_field_value(frame_idx, y_field)
            
            if x_val is not None and y_val is not None:
                x_data.append(x_val)
                y_data.append(y_val)
            else:
                x_data.append(np.nan)
                y_data.append(np.nan)
        
        if not x_data:
            self.data_ax.text(0.5, 0.5, 'No data to plot', 
                            ha='center', va='center', transform=self.data_ax.transAxes)
            self.data_canvas.draw()
            return
        
        # Plot the data
        if x_field == 'Frame Number':
            self.data_ax.plot(x_data, y_data, 'o-', markersize=3, linewidth=1)
        else:
            self.data_ax.scatter(x_data, y_data, s=3)
        
        if y_field == 'Y position' and not self.data_ax.yaxis_inverted():
            self.data_ax.invert_yaxis()
        if not y_field == 'Y position' and self.data_ax.yaxis_inverted():
            self.data_ax.invert_yaxis()
        
        self.data_ax.set_xlabel(x_field, fontsize=9)
        self.data_ax.set_ylabel(y_field, fontsize=9)
        self.data_ax.tick_params(labelsize=8)
        self.data_ax.grid(True, alpha=0.3)
        
        self._suppress_tight_layout_warning(self.data_figure)
        
        # Setup polygon selector
        self.setup_polygon_selector()
        
        # Add current frame indicator
        if self.data.frame_segments and hasattr(self, 'progress_slider'):
            self.update_current_frame_indicator()
        
        self.data_canvas.draw()
    
    def _get_field_value(self, frame_idx, field):
        """Get the value for a specific field from a frame - delegates to data"""
        return self.data.get_field_value(frame_idx, field)

    def update_current_frame_indicator(self):
        """Update the red line/marker showing current frame"""
        if not self.data.is_valid() or not self.data.frame_segments:
            return

        x_field = self.x_field_combo.currentText()
        y_field = self.y_field_combo.currentText()

        # Get current frame
        current_frame_count = self.progress_slider.value()
        display_indices = self.get_display_indices()

        # Remove old indicators
        if self.current_frame_line is not None:
            try:
                self.current_frame_line.remove()
            except:
                pass
            self.current_frame_line = None

        if self.current_frame_marker is not None:
            try:
                self.current_frame_marker.remove()
            except:
                pass
            self.current_frame_marker = None

        if current_frame_count > 0 and current_frame_count <= len(display_indices):
            # Get actual frame index
            idx = display_indices[current_frame_count - 1]
            actual_frame_idx = self.data.frame_indices[idx]

            if x_field == 'Frame Number':
                self.current_frame_line = self.data_ax.axvline(x=actual_frame_idx, color='red', 
                                                            linewidth=2, linestyle='--', 
                                                            alpha=0.7, label='Current Frame')
            else:
                val_x = self._get_field_value(actual_frame_idx, x_field)
                val_y = self._get_field_value(actual_frame_idx, y_field)
                if val_x is not None and val_y is not None:
                    self.current_frame_marker = self.data_ax.plot(val_x, val_y, 'rX', markersize=10)[0]
            
            if not self.is_playing:
                self.data_canvas.draw()

    def on_splitter_moved(self, pos, index):
        """Handle splitter movement"""
        total_width = self.splitter.width()
        max_control_width = total_width // 2

        sizes = self.splitter.sizes()

        if sizes[0] > max_control_width:
            sizes[0] = max_control_width
            sizes[1] = total_width - max_control_width
            self.splitter.setSizes(sizes)

        if self.splitter_move_timer is not None:
            self.killTimer(self.splitter_move_timer)

        self.splitter_move_timer = self.startTimer(300)

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        
        if self.resize_timer is not None:
            self.killTimer(self.resize_timer)
        
        self.resize_timer = self.startTimer(300)

    def timerEvent(self, event):
        """Handle timer events for debounced redraws"""
        if event.timerId() == self.splitter_move_timer:
            self.killTimer(self.splitter_move_timer)
            self.splitter_move_timer = None
            self._redraw_plots()
        
        elif event.timerId() == self.resize_timer:
            self.killTimer(self.resize_timer)
            self.resize_timer = None
            self._redraw_plots()
    
    def _suppress_tight_layout_warning(self, figure):
        """Suppress matplotlib tight_layout warning"""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Tight layout not applied')
            figure.tight_layout()
    
    def _redraw_plots(self):
        """Redraw both plots"""
        if self.data.is_valid():
            if not self.is_playing:
                self.plot_static()
            
            self._suppress_tight_layout_warning(self.figure)
            self.canvas.draw()
            
            self._suppress_tight_layout_warning(self.data_figure)
            self.data_canvas.draw()

    def on_ontop_toggled(self, state):
        """Toggle always on top window flag"""
        if state == Qt.CheckState.Checked.value:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.show()

    def setup_polygon_selector(self):
        """Setup the polygon selector with callback"""
        if self.poly_selector is not None:
            self.poly_selector.disconnect_events()
        
        self.poly_selector = PolygonSelector(
            self.data_ax,
            onselect=self.on_polygon_changed,  # Callback when polygon changes
            useblit=True,
            props=dict(color='red', linewidth=2, alpha=0.8),
            handle_props=dict(markersize=8, markeredgecolor='red'),
        )
        
        # Connect double-click to clear
        self.data_canvas.mpl_connect('button_press_event', self.on_data_plot_click)

    def on_polygon_changed(self, verts):
        """Callback when polygon is drawn or modified"""
        if len(verts) < 3:
            return
        
        # If preview is enabled, update the filter
        if self.preview_toggle.isChecked():
            self.apply_polygon_filter()

    def apply_polygon_filter(self):
        """Apply the polygon filter to frame data"""
        if self.poly_selector is None or len(self.poly_selector.verts) < 3:
            return
        
        x_field = self.x_field_combo.currentText()
        y_field = self.y_field_combo.currentText()
        
        path = Path(self.poly_selector.verts)
        
        # Find frames inside polygon
        frames_inside = []
        for i, frame_idx in enumerate(self.data.frame_indices):
            if x_field == 'Frame Number':
                x_val = frame_idx
            else:
                x_val = self.data.get_field_value(frame_idx, x_field)
            
            y_val = self.data.get_field_value(frame_idx, y_field)
            
            if x_val is None or y_val is None:
                continue
            
            if path.contains_point((x_val, y_val)):
                frames_inside.append(i)
        
        if not frames_inside:
            siril.log("No frames fall within the polygon", s.LogColor.BLUE)
            return
        
        # Create filtered indices (all except those inside polygon)
        self.data.filtered_frame_indices = [i for i in range(len(self.data.frame_indices)) if i not in frames_inside]
        
        # Update UI
        self.update_sequence_info_label()
        self.update_slider_for_filtered_data()
        self.plot_static()
        self.canvas.draw()

    def update_slider_for_filtered_data(self):
        """Update slider when filtered data changes"""
        num_frames = self.get_num_display_frames()
        self.updating_slider = True
        self.progress_slider.setMaximum(num_frames)
        tick_interval = max(1, num_frames // 10)
        self.progress_slider.setTickInterval(tick_interval)
        self.progress_slider.setValue(num_frames)
        self.update_progress_label(num_frames)
        self.updating_slider = False
    
    def on_preview_toggled(self, state):
        """Handle preview toggle"""
        if state == Qt.CheckState.Checked.value:
            self.apply_polygon_filter()
        else:
            self.data.filtered_frame_indices = None
            self.update_sequence_info_label()
            self.update_slider_for_filtered_data()
            self.plot_static()
            self.canvas.draw()
    
    def on_apply_mask_clicked(self):
        """Apply mask permanently to sequence"""
        if self.poly_selector is None or len(self.poly_selector.verts) < 3:
            siril.log("No polygon drawn (need at least 3 points)", s.LogColor.BLUE)
            return

        if self.data.filtered_frame_indices is not None:
            indices_to_remove = list(set(self.data.frame_indices[i] for i in range(len(self.data.frame_indices)) if i not in self.data.filtered_frame_indices))
            siril.set_seq_frame_incl(indices_to_remove, False)
            self.on_refresh_clicked()
        else:
            siril.log("No frames are currently masked to apply", s.LogColor.BLUE)

    def on_data_plot_click(self, event):
        """Handle clicks on data plot - double-click to clear polygon"""
        if event.dblclick and event.inaxes == self.data_ax:
            if self.poly_selector is not None and len(self.poly_selector.verts) > 0:
                # Disconnect the polygon selector before clearing to prevent starting a new selection
                self.poly_selector.disconnect_events()
                self.poly_selector.clear()
                
                # Reconnect the polygon selector with fresh state
                self.setup_polygon_selector()
                
                # If preview was on, reset filter
                if self.preview_toggle.isChecked():
                    self.data.filtered_frame_indices = None
                    self.update_sequence_info_label()
                    self.update_slider_for_filtered_data()
                    self.plot_static()
                    self.canvas.draw()

def create_registration_plot_cli(siril_interface, output_path=None):
    """
    Create and save a registration inspection plot for CLI mode.

    Args:
        siril_interface: SirilInterface instance
        output_path: Path to save the plot. If None, uses sequence name.

    Returns:
        Path to the saved plot
    """
    # Load sequence data
    data = SequenceData(siril_interface)
    if not data.load_sequence():
        raise Exception("Failed to load sequence")

    fig = Figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)
    seq_name = data.get_sequence_name()
    fig.suptitle(seq_name, fontsize=12)
    data.plot_frames_on_axis(ax, siril_interface)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Tight layout not applied')
        fig.tight_layout()

    if output_path is None:
        output_path = f"{seq_name}_registration.png" if seq_name and seq_name[-1] != '_' else f"{seq_name}registration.png"

    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Registration plot saved to: {output_path}")
    
    return output_path

def main():

    if siril.is_cli() and len(sys.argv) > 1: #CLI mode
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Create a registration inspection plot for a sequence")
            parser.add_argument("-o", "--output", type=str, default=None,
                                help="Output file path for the plot (default: <sequence_name>_registration.png)")
            args = parser.parse_args()

            if not siril.is_sequence_loaded():
                siril.log("No sequence is loaded", s.LogColor.RED)
                sys.exit(1)

            _ = create_registration_plot_cli(siril, args.output)
            sys.exit(0)

        except Exception as e:
            siril.log(f"Error: {str(e)}", s.LogColor.RED)
            sys.exit(1)
    else:  # GUI mode
        app = QApplication.instance() or QApplication(sys.argv)
        window = RegInspectorWidget()
        window.show()
        sys.exit(app.exec())

if __name__ == "__main__":
    main()