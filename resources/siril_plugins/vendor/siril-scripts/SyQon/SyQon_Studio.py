# (c) SyQon 2026 
# SPDX-License-Identifier: see https://siril.syqon.it/LICENSE.pdf
"""
    ███████╗██╗   ██╗ ██████╗  ██████╗ ███╗   ██╗
    ██╔════╝╚██╗ ██╔╝██╔═══██╗██╔═══██╗████╗  ██║
    ███████╗ ╚████╔╝ ██║   ██║██║   ██║██╔██╗ ██║
    ╚════██║  ╚██╔╝  ██║▄▄ ██║██║   ██║██║╚██╗██║
    ███████║   ██║   ╚██████╔╝╚██████╔╝██║ ╚████║
    ╚══════╝   ╚═╝    ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═══╝
          ★  S T A R L E S S   A C C E L E R A T O R  ★
     ┌───────────────────────────────────────────┐
     │            Siril Edition v3.0             │
     │        C++ Accelerator Script             │
     │         https://syqon.it/starless         │
     └───────────────────────────────────────────┘

Usage:
    Add this script to the Scripts menu and run it.
    It will save the current Siril image to a temporary TIFF,
    open the high-performance C++ Qt6 GUI, and reload the
    starless result back into Siril once closed.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import sirilpy as s

# Dynamically import Qt for GUI
QApplication = None
QDialog = None
QVBoxLayout = None
QHBoxLayout = None
QLabel = None
QLineEdit = None
QPushButton = None
QFileDialog = None
QCheckBox = None

for qt_lib in ["PySide6", "PyQt6", "PyQt5", "PySide2"]:
    try:
        from importlib import import_module
        QtWidgets = import_module(f"{qt_lib}.QtWidgets")
        QApplication = QtWidgets.QApplication
        QDialog = QtWidgets.QDialog
        QVBoxLayout = QtWidgets.QVBoxLayout
        QHBoxLayout = QtWidgets.QHBoxLayout
        QLabel = QtWidgets.QLabel
        QLineEdit = QtWidgets.QLineEdit
        QPushButton = QtWidgets.QPushButton
        QFileDialog = QtWidgets.QFileDialog
        QCheckBox = QtWidgets.QCheckBox
        print(f"Successfully loaded Qt from {qt_lib}")
        break
    except ImportError:
        continue

if QDialog is not None:
    class SyQonStartDialog(QDialog):
        def __init__(self, initial_path=None, is_windows=False, home_dir="", parent=None):
            super().__init__(parent)
            self.selected_path = initial_path
            self.is_windows = is_windows
            self.home_dir = home_dir
            self.init_ui()

        def init_ui(self):
            self.setWindowTitle("SyQon Studio")
            self.resize(550, 350)
            self.setStyleSheet("background-color: #0F0F12;")
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(15)
            
            # Header Container
            header_layout = QVBoxLayout()
            header_layout.setSpacing(2)
            
            welcome_lbl = QLabel("SYQON STUDIO", self)
            welcome_lbl.setStyleSheet("font-size: 22px; font-weight: bold; color: #10B981; letter-spacing: 1px;")
            header_layout.addWidget(welcome_lbl)
            
            sub_lbl = QLabel("STARLESS ACCELERATOR", self)
            sub_lbl.setStyleSheet("font-size: 9px; font-weight: bold; color: #6B7280; letter-spacing: 2px;")
            header_layout.addWidget(sub_lbl)
            
            layout.addLayout(header_layout)
            
            # Status label
            self.status_lbl = QLabel(self)
            self.status_lbl.setWordWrap(True)
            layout.addWidget(self.status_lbl)
            
            # Path selection row
            path_layout = QHBoxLayout()
            path_layout.setSpacing(10)
            
            self.path_edit = QLineEdit(self)
            self.path_edit.setReadOnly(True)
            self.path_edit.setPlaceholderText("No SyQonStarless executable selected")
            self.path_edit.setStyleSheet(
                "padding: 8px; font-size: 12px; border: 1px solid #2D2D35; "
                "border-radius: 6px; background-color: #1A1A22; color: #E5E7EB;"
            )
            path_layout.addWidget(self.path_edit)
            
            self.browse_btn = QPushButton("Browse...", self)
            self.browse_btn.setStyleSheet(
                "QPushButton {"
                "  padding: 8px 16px; font-size: 12px; font-weight: bold; "
                "  background-color: #272730; color: #F3F4F6; border: 1px solid #3F3F46; border-radius: 6px;"
                "}"
                "QPushButton:hover {"
                "  background-color: #3F3F4E; border-color: #52525B;"
                "}"
            )
            self.browse_btn.clicked.connect(self.browse_binary)
            path_layout.addWidget(self.browse_btn)
            
            layout.addLayout(path_layout)

            # Don't show this again CheckBox
            self.dont_show_cb = QCheckBox("Don't show this again", self)
            self.dont_show_cb.setStyleSheet(
                "QCheckBox {"
                "  color: #9CA3AF;"
                "  font-size: 12px;"
                "}"
                "QCheckBox::indicator {"
                "  width: 14px;"
                "  height: 14px;"
                "  border: 1px solid #3F3F46;"
                "  border-radius: 3px;"
                "  background-color: #1A1A22;"
                "}"
                "QCheckBox::indicator:checked {"
                "  background-color: #10B981;"
                "  border-color: #10B981;"
                "}"
                "QCheckBox:disabled {"
                "  color: #4B5563;"
                "}"
            )
            layout.addWidget(self.dont_show_cb)
            
            # Start button
            self.start_btn = QPushButton("Start SyQon Studio", self)
            self.start_btn.clicked.connect(self.accept)
            layout.addWidget(self.start_btn)
            
            self.update_ui_state()

        def update_ui_state(self):
            if self.selected_path and os.path.exists(self.selected_path):
                self.path_edit.setText(self.selected_path)
                self.status_lbl.setText("SyQonStarless binary found successfully!\nReady to start accelerating.")
                self.status_lbl.setStyleSheet(
                    "font-size: 13px; padding: 12px; border-radius: 6px; "
                    "background-color: rgba(16, 185, 129, 0.08); color: #10B981; "
                    "border: 1px solid rgba(16, 185, 129, 0.4);"
                )
                self.start_btn.setEnabled(True)
                self.start_btn.setStyleSheet(
                    "QPushButton {"
                    "  padding: 12px; font-size: 14px; font-weight: bold; "
                    "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #10B981, stop:1 #059669); "
                    "  color: white; border: none; border-radius: 6px; margin-top: 10px;"
                    "}"
                    "QPushButton:hover {"
                    "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #059669, stop:1 #047857);"
                    "}"
                )
                self.dont_show_cb.setEnabled(True)
            else:
                self.path_edit.clear()
                self.status_lbl.setText("If the script has not found the binary, please select it from the path where it was installed.")
                self.status_lbl.setStyleSheet(
                    "font-size: 13px; padding: 12px; border-radius: 6px; "
                    "background-color: rgba(245, 158, 11, 0.08); color: #F59E0B; "
                    "border: 1px solid rgba(245, 158, 11, 0.4);"
                )
                self.start_btn.setEnabled(False)
                self.start_btn.setStyleSheet(
                    "QPushButton {"
                    "  padding: 12px; font-size: 14px; font-weight: bold; "
                    "  background-color: #1E1E24; color: #4B5563; border: 1px solid #2D2D35; "
                    "  border-radius: 6px; margin-top: 10px;"
                    "}"
                )
                self.dont_show_cb.setChecked(False)
                self.dont_show_cb.setEnabled(False)

        def accept(self):
            dont_show_val = "1" if self.dont_show_cb.isChecked() else "0"
            dont_show_file = os.path.join(self.home_dir, ".siril", "syqon_starless", "dont_show_gui.txt")
            try:
                os.makedirs(os.path.dirname(dont_show_file), exist_ok=True)
                with open(dont_show_file, "w") as f:
                    f.write(dont_show_val)
            except Exception:
                pass
            super().accept()

        def browse_binary(self):
            filter_str = "Executables (*.exe *.dmg *.app);;All Files (*)" if not self.is_windows else "Executables (*.exe);;All Files (*)"
            file_path, _ = QFileDialog.getOpenFileName(self, "Select SyQonStarless Executable", self.home_dir, filter_str)
            
            if file_path:
                if not self.is_windows and file_path.endswith(".app"):
                    app_name = os.path.splitext(os.path.basename(file_path))[0]
                    internal_bin = os.path.join(file_path, "Contents", "MacOS", app_name)
                    if os.path.exists(internal_bin):
                        file_path = internal_bin
                    else:
                        macos_dir = os.path.join(file_path, "Contents", "MacOS")
                        if os.path.exists(macos_dir):
                            files = os.listdir(macos_dir)
                            if files:
                                file_path = os.path.join(macos_dir, files[0])
                
                if os.path.exists(file_path):
                    self.selected_path = file_path
                    self.update_ui_state()
else:
    SyQonStartDialog = None

def _siril_quoted_path(path: str) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'

def main():
    # Ensure tifffile is installed
    try:
        s.ensure_installed("tifffile")
    except Exception as e:
        print(f"Warning: Could not check or install 'tifffile' dependency: {e}")

    # Connect to Siril
    siril = None
    try:
        siril = s.SirilInterface()
        siril.connect()
        print("Connected to Siril")
    except Exception as e:
        print(f"Could not connect to Siril: {e}")
        sys.exit(1)

    is_single = siril.is_image_loaded()
    is_seq    = siril.is_sequence_loaded()

    if not is_single and not is_seq:
        print("Error: No image or sequence loaded in Siril")
        sys.exit(1)

    cwd = siril.get_siril_wd()
    
    # Paths for compiled binary and model
    # Detect dynamically based on script location, common home dir paths, PATH environment, or fallback
    script_dir = os.path.dirname(os.path.abspath(__file__))
    home_dir = str(Path.home())
    
    is_windows = (sys.platform == "win32")

    # 1. Search for binary
    saved_path_file = os.path.join(home_dir, ".siril", "syqon_starless", "saved_binary_path.txt")
    binary_path = None

    # Try reading from cache first
    if os.path.exists(saved_path_file):
        try:
            with open(saved_path_file, "r") as f:
                cached_path = f.read().strip()
                if cached_path and os.path.exists(cached_path):
                    binary_path = cached_path
        except Exception:
            pass

    if not binary_path:
        if is_windows:
            binary_search_paths = [
                "C:\\Program Files\\SyQon\\Starless\\SyQonStarless.exe",
                "C:\\Program Files\\SyQon\\Starless\\syqonstarless.exe",
                os.path.join(script_dir, "SyQonStarless.exe"),
                os.path.join(script_dir, "syqonstarless.exe"),
            ]
        else:
            binary_search_paths = [
                "/Applications/SyQonStarless.app/Contents/MacOS/SyQonStarless",
                "/usr/local/bin/SyQonStarless"
            ]
        
        # Check if the binary is in the system PATH
        system_path_binary = shutil.which("SyQonStarless")
        if system_path_binary:
            binary_search_paths.append(system_path_binary)
            
        if not is_windows:
            binary_search_paths.extend([
                os.path.join(script_dir, "SyQonStarless"),
                os.path.join(script_dir, "build/SyQonStarless.app/Contents/MacOS/SyQonStarless"),
                os.path.join(script_dir, "../build/SyQonStarless.app/Contents/MacOS/SyQonStarless"),
                os.path.join(home_dir, ".siril/syqon_starless/SyQonStarless"),
                "/Users/michaelruggeri/Desktop/Neural Model SyQon/Axiom V3/build/SyQonStarless.app/Contents/MacOS/SyQonStarless"
            ])
        else:
            binary_search_paths.extend([
                os.path.join(home_dir, ".siril/syqon_starless/SyQonStarless.exe"),
            ])

        for p in binary_search_paths:
            if os.path.exists(p):
                binary_path = p
                break

    # Open GUI Dialog unless skipped by user preference (and binary is valid)
    dont_show_file = os.path.join(home_dir, ".siril", "syqon_starless", "dont_show_gui.txt")
    skip_gui = False
    if binary_path and os.path.exists(binary_path):
        if os.path.exists(dont_show_file):
            try:
                with open(dont_show_file, "r") as f:
                    if f.read().strip() == "1":
                        skip_gui = True
            except Exception:
                pass

    if skip_gui:
        print("SyQon Studio: GUI skipped (user preference 'Don't show this again' is active).")
    else:
        if SyQonStartDialog is not None:
            app = QApplication.instance()
            if not app:
                app = QApplication(sys.argv)
            
            dialog = SyQonStartDialog(initial_path=binary_path, is_windows=is_windows, home_dir=home_dir)
            result_code = dialog.exec()
            
            if result_code != 1:
                print("SyQon Studio startup cancelled by user.")
                sys.exit(0)
                
            binary_path = dialog.selected_path
            
            # Save it to cache
            if binary_path and os.path.exists(binary_path):
                try:
                    os.makedirs(os.path.dirname(saved_path_file), exist_ok=True)
                    with open(saved_path_file, "w") as f:
                        f.write(binary_path)
                    print(f"Cached binary path: {binary_path}")
                except Exception as e:
                    print(f"Warning: Could not cache binary path: {e}")
        else:
            print("Warning: Could not open SyQonStartDialog because Qt is not available.")

    if not binary_path or not os.path.exists(binary_path):
        print("Error: SyQonStarless binary not found. Please select it in the GUI or check its location.")
        sys.exit(1)

    print(f"[SyQon Starless] Using binary at: {binary_path}")

    if is_single:
        # 1. Save current image in Siril to a temporary TIFF
        temp_in_base = os.path.join(cwd, "syqon_temp_in")
        temp_in = temp_in_base + ".tif"
        temp_out = os.path.join(cwd, "starless_syqon_temp_in.tif")

        # Cleanup old temps
        if os.path.exists(temp_in):
            try: os.remove(temp_in)
            except Exception: pass
        if os.path.exists(temp_out):
            try: os.remove(temp_out)
            except Exception: pass
            
        path_file = os.path.join(cwd, "syqon_output_path.txt")
        if os.path.exists(path_file):
            try: os.remove(path_file)
            except Exception: pass

        print(f"Exporting current image to temporary TIFF: {temp_in}")
        siril.cmd("savetif32", _siril_quoted_path(temp_in_base))

        # 2. Launch Standalone CLI in Seti mode (with GUI)
        print("Launching SyQonStarless (Seti CLI mode with GUI)...")
        cmd = [binary_path, "-i", temp_in, "-o", temp_out, "-d", "Auto", "-t", "512", "-v", "256", "-c", "seti", "--gui"]
        result = subprocess.run(cmd)

        # 3. Import result back to Siril
        actual_temp_out = temp_out
        if result.returncode == 0:
            if os.path.exists(path_file):
                try:
                    with open(path_file, "r") as f:
                        actual_temp_out = f.read().strip()
                    os.remove(path_file)
                except Exception:
                    pass
        
        # Only import if GUI exited cleanly (code 0) and the output exists
        if result.returncode == 0 and os.path.exists(actual_temp_out):
            # Get original image name
            orig_filename = siril.get_image_filename()
            base = os.path.splitext(os.path.basename(orig_filename))[0]
            
            # Form permanent filenames
            starless_perm = os.path.join(cwd, f"starless_{base}.tif")
            starmask_perm = os.path.join(cwd, f"starmask_{base}.tif")
            
            # Rename temp_out to permanent starless
            if os.path.exists(starless_perm):
                try: os.remove(starless_perm)
                except Exception: pass
            os.rename(actual_temp_out, starless_perm)
            
            # Rename temp_mask to permanent starmask if it exists
            temp_mask = os.path.join(os.path.dirname(actual_temp_out), "starmask_syqon_temp_in.tif")
            if os.path.exists(temp_mask):
                if os.path.exists(starmask_perm):
                    try: os.remove(starmask_perm)
                    except Exception: pass
                os.rename(temp_mask, starmask_perm)
                print(f"Saved starmask to: {starmask_perm}")
                # We deliberately do not load the starmask into Siril here,
                # as loading any file resets Siril's undo history.
            
            print(f"Importing starless image back to Siril: {starless_perm}")
            try:
                import tifffile
                import numpy as np

                # Read pixels from the saved TIFF file
                starless_pixels = tifffile.imread(starless_perm)
                
                # Get the original image from Siril to check shape and dtype
                orig_image = siril.get_image()
                orig_shape = orig_image.data.shape
                orig_dtype = orig_image.data.dtype
                
                # Check if original was mono
                orig_was_mono = (len(orig_shape) == 2 or orig_shape[0] == 1)
                
                # Prepare the array shape to match Siril's planar layout (C, H, W)
                if orig_was_mono:
                    if starless_pixels.ndim == 3:
                        # If the output has multiple channels but original was mono, take the mean
                        starless_pixels = starless_pixels.mean(axis=-1)
                    if starless_pixels.ndim == 2:
                        starless_pixels = starless_pixels[np.newaxis, ...]
                else:
                    if starless_pixels.ndim == 2:
                        # If output was mono but original was color, stack channels
                        starless_pixels = np.stack([starless_pixels] * 3, axis=-1)
                    if starless_pixels.ndim == 3:
                        # Check channel position. tifffile reads as (H, W, C).
                        # We transpose it to (C, H, W).
                        if starless_pixels.shape[-1] in (3, 4):
                            # Transpose from HWC to CHW
                            starless_pixels = starless_pixels.transpose(2, 0, 1)
                            # Discard alpha channel if present
                            starless_pixels = starless_pixels[:3, ...]
                
                # Flip vertically to align standard top-down TIFF coordinate system
                # with Siril's bottom-up memory layout (H is axis 1 in planar layout C, H, W)
                starless_pixels = np.flip(starless_pixels, axis=1)

                # Cast to original dtype to match memory expectations
                starless_pixels = starless_pixels.astype(orig_dtype)
                
                # Apply in-place inside Siril to keep undo history
                siril.undo_save_state("SyQon Starless - star removal")
                with siril.image_lock():
                    siril.set_image_pixeldata(starless_pixels)
                siril.set_image_filename(starless_perm)
                print("Starless image successfully updated in-place.")
            except Exception as e:
                print(f"Error importing starless image in-place: {e}")
                print("Falling back to standard load (which resets undo history)...")
                siril.undo_save_state("SyQon Starless - star removal")
                siril.cmd("load", _siril_quoted_path(starless_perm))
            
            # Try to cleanup temporary files
            try:
                os.remove(temp_in)
            except Exception:
                pass
        else:
            print("Starless extraction cancelled or output not generated.")
            
    elif is_seq:
        # Processing a sequence using CLI mode headlessly (extremely fast with ANE)
        try:
            sequence = siril.get_seq()
            seq_length = sequence.number
            included_frames = [i for i in range(seq_length) if sequence.imgparam[i].incl]
            print(f"Sequence mode: Processing {len(included_frames)} included frames using Apple Neural Engine CLI")
            
            if len(included_frames) == 0:
                print("Error: No frames are included in the sequence")
                sys.exit(1)
        except Exception as e:
            print(f"Could not load sequence: {e}")
            sys.exit(1)

        # Create output directory for sequence
        output_dir = os.path.join(cwd, "starless_sequence")
        os.makedirs(output_dir, exist_ok=True)

        for idx, actual in enumerate(included_frames):
            frame_filename = siril.get_seq_frame_filename(actual)
            base = os.path.splitext(os.path.basename(frame_filename))[0]
            
            # Temporary TIFF path
            temp_frame_in_base = os.path.join(output_dir, f"temp_{base}")
            temp_frame_in = temp_frame_in_base + ".tif"
            temp_frame_out = os.path.join(output_dir, f"starless_{base}.tif")
            
            # Load frame, export to TIFF
            siril.cmd("load", _siril_quoted_path(frame_filename))
            siril.cmd("savetif32", _siril_quoted_path(temp_frame_in_base))
            
            # Run headless CLI inference
            print(f"[{idx+1}/{len(included_frames)}] Processing frame {base}...")
            cmd = [binary_path, "-i", temp_frame_in, "-o", temp_frame_out, "-d", "Auto", "-t", "512", "-v", "256", "-c", "seti"]
            subprocess.run(cmd, stdout=subprocess.DEVNULL)
            
            # Cleanup temp frame in
            try: os.remove(temp_frame_in)
            except Exception: pass
            
        # Re-import sequence
        print("Sequence processing complete. Loading sequence back in Siril...")
        try:
            siril.cmd("cd", _siril_quoted_path(output_dir))
            
            # Create sequence for starless
            siril.create_new_seq(f"starless_{sequence.seqname}")
            
            # Create sequence for starmask if any starmask files exist
            starmask_files = [f for f in os.listdir(output_dir) if f.startswith("starmask_")]
            if len(starmask_files) > 0:
                siril.create_new_seq(f"starmask_{sequence.seqname}")
                
            siril.cmd("load_seq", f"starless_{sequence.seqname}")
        except Exception as e:
            print(f"Error rebuilding sequence: {e}")

if __name__ == "__main__":
    main()
