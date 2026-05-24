# Jun. 16, 2025
# (c) Eduardo Ramírez - @edramigon
# SPDX-License-Identifier: GPL-3.0-or-later
# Script for reducing stars using pixel math.
#
# Instagram https://www.instagram.com/edramigon/
# TikTok https://www.tiktok.com/@edramigon

import sirilpy as s
s.ensure_installed("PyQt6")

import os
import sys
from sirilpy import LogColor

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSlider, QCheckBox, QPushButton,
                             QGroupBox, QRadioButton, QSpinBox, QButtonGroup)
from PyQt6.QtCore import Qt

VERSION = "1.0.2"

# 1.0.0 Original release by Eduardo Ramírez
# 1.0.1 Add siril.update_progress() calls
# 1.0.2 Convert to PyQt6 in preparation for deprecating tksiril submodule

class StarReductionInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"ER-Bill_Star Reduction - v{VERSION}")
        self.setFixedSize(350, 410)

        # Connect to Siril
        self.siril = s.SirilInterface()
        try:
            self.siril.connect()
        except s.SirilConnectionError as e:
            self.siril.log("Connection failed: {e}", color=LogColor.RED)
            sys.exit()

        self.siril.log("Connected successfully!", color=LogColor.GREEN)

        # Initial checks
        if not self.initial_checks():
            self.close()
            sys.exit()

        # Get image info
        self.setup_image_info()

        # Create GUI
        self._build_ui()

    def initial_checks(self):

        # Check Siril version
        require_version = "1.3.6"
        self.default_ext = self.siril.get_siril_config("core", "extension")  # Default FITS extension
        self.final_name = "0"
        try:
            self.siril.cmd("requires", require_version)
        except:
            self.siril.error_messagebox(f"This script requires Siril version {require_version} or later!")
            return False

        # Check Starnet configuration
        starnet_path = self.siril.get_siril_config("core", "starnet_exe")
        if not starnet_path or not os.path.isfile(starnet_path) or not os.access(starnet_path, os.X_OK):
            self.siril.error_messagebox("Starnet Command Line Tool was not found or is not configured!")
            return False

        # Check an image is loaded
        if not self.siril.is_image_loaded():
            self.siril.error_messagebox("Open a FITS image before running Star Reduction!")
            return False

        # Check current image file extension
        path = self.siril.get_image_filename()
        basename = os.path.basename(path)
        get_extension = os.path.splitext(basename)[1]
        if get_extension.lower() not in (".fit", ".fits", ".fts", ".fit.fz", ".fits.fz", ".fts.fz"):
            self.siril.error_messagebox(f"The image that is open is a {get_extension} and is not supported.\nPlease open a FITS file and run the script again.")
            return False

        # Check if already reduced
        if "_ReducedStars" in basename:
            self.siril.error_messagebox("This image has already had star reduction applied."
            "\n\nOpen an image that has not yet undergone star reduction.")
            return False

        return True

    def setup_image_info(self):
        # Get current image filename & set working directory to opened image directory.
        path = self.siril.get_image_filename()
        self.img_name = os.path.basename(path)
        self.img_dir = os.path.dirname(os.path.abspath(path))
        self.siril.cmd("cd", f'"{self.img_dir}"')
        os.chdir(self.img_dir)
        self.img_name_pm = f"${self.img_name}$" # Wrap in $ for PixelMath use
        self.get_extension = os.path.splitext(self.img_name)[1] # Get file extension
        self.file_name_without_ext = os.path.splitext(os.path.basename(path))[0] # Get filename without extension

    def _build_ui(self):
        self.resolution = 0.01

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Title
        title_label = QLabel("Bill Blanshan's Star Reduction by @edramigon")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Reminder Label
        reminder_label = QLabel("Image must already be stretched!")
        reminder_label.setStyleSheet("font-weight: bold;")
        reminder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(reminder_label)

        # Reduction method group
        method_group = QGroupBox("Reduction method")
        method_layout = QHBoxLayout(method_group)
        main_layout.addWidget(method_group)

        self.method_button_group = QButtonGroup(self)
        for method in ["Transfer", "Halo", "Star"]:
            rb = QRadioButton(method)
            method_layout.addWidget(rb)
            self.method_button_group.addButton(rb)
            if method == "Transfer":
                rb.setChecked(True)
        self.method_button_group.buttonClicked.connect(self._update_widgets)

        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)
        main_layout.addWidget(params_group)

        # Stretch factor row
        stretch_layout = QHBoxLayout()
        params_layout.addLayout(stretch_layout)
        stretch_label = QLabel("Stretch factor: ")
        stretch_layout.addWidget(stretch_label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(int(0.99 / self.resolution))
        self.slider.setValue(int(0.2 / self.resolution))
        stretch_layout.addWidget(self.slider)
        self.slider_value_label = QLabel("0.20")
        stretch_layout.addWidget(self.slider_value_label)
        self.slider.valueChanged.connect(self._update_slider_value)
        self._stretch_widgets = [stretch_label, self.slider, self.slider_value_label]

        # Mode row
        mode_layout = QHBoxLayout()
        params_layout.addLayout(mode_layout)
        mode_label = QLabel("Mode: ")
        mode_layout.addWidget(mode_label)
        self.mode_button_group = QButtonGroup(self)
        self._mode_widgets = [mode_label]
        for mode in ["Strong", "Moderate", "Soft"]:
            rb = QRadioButton(mode)
            mode_layout.addWidget(rb)
            self.mode_button_group.addButton(rb)
            if mode == "Moderate":
                rb.setChecked(True)
            self._mode_widgets.append(rb)

        # Interactions row
        interactions_layout = QHBoxLayout()
        params_layout.addLayout(interactions_layout)
        interactions_label = QLabel("Interactions: ")
        interactions_layout.addWidget(interactions_label)
        self.interactions_spin = QSpinBox()
        self.interactions_spin.setRange(1, 3)
        self.interactions_spin.setValue(2)
        interactions_layout.addWidget(self.interactions_spin)
        self._interactions_widgets = [interactions_label, self.interactions_spin]

        # Disable mode and interactions initially (Transfer is default)
        self._set_widgets_enabled(self._mode_widgets, False)
        self._set_widgets_enabled(self._interactions_widgets, False)

        # Options group
        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout(options_group)
        main_layout.addWidget(options_group)

        self.overwrite_checkbox = QCheckBox("Overwrite Output File")
        self.overwrite_checkbox.setChecked(True)
        options_layout.addWidget(self.overwrite_checkbox)

        # Buttons
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)

        help_btn = QPushButton("Help")
        help_btn.clicked.connect(self._show_help)
        button_layout.addWidget(help_btn)

        button_layout.addStretch()

        save_close_btn = QPushButton("Close")
        save_close_btn.clicked.connect(self.close)
        button_layout.addWidget(save_close_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._run_reduction)
        button_layout.addWidget(apply_btn)

    def _set_widgets_enabled(self, widgets, enabled):
        for w in widgets:
            w.setEnabled(enabled)

    def _update_widgets(self):
        method = self.method_button_group.checkedButton().text()
        if method == "Halo":
            self._set_widgets_enabled(self._stretch_widgets, True)
            self._set_widgets_enabled(self._mode_widgets, False)
            self._set_widgets_enabled(self._interactions_widgets, False)
        elif method == "Star":
            self._set_widgets_enabled(self._stretch_widgets, False)
            self._set_widgets_enabled(self._mode_widgets, True)
            self._set_widgets_enabled(self._interactions_widgets, True)
        elif method == "Transfer":
            self._set_widgets_enabled(self._stretch_widgets, True)
            self._set_widgets_enabled(self._mode_widgets, False)
            self._set_widgets_enabled(self._interactions_widgets, False)

    def _update_slider_value(self):
        rounded_value = round(self.slider.value() * self.resolution, 2)
        self.slider_value_label.setText(f"{rounded_value:.2f}")

    def _show_help(self):
        self.siril.info_messagebox("Reduces the size of the stars"
        " in an image based on the selected method and parameters. There are 3 methods:"
        "\n\nTransfer: The lower the number, the stronger the reduction. You can also increase"
        " the size of the stars by raising the value. A value of 0.5 will have no affect; values below"
        " 0.5 reduce the stars, and values above increase their size."
        "\n\nHalo: Works the same as the Transfer method but preserves the original halo around the stars"
        "\n\nStar:\nStrong Mode: Produces smaller, sharper stars while removing the tiny ones"
        "\nModerate Mode: Keeps stars sharp and includes some small stars"
        "\nSoft Mode: Applies a simple reduction to the stars in the original image"
        "\n\n-StarNet Command Line Tool must be installed and configured in Siril."
        "\n\n-Image must be in FITS format and stretched (non-linear)."
        "\n\n-Set the value using the slider, and click Apply."
        "\n\n-Uncheck 'Overwrite Output File' to save a new image each time."
        " Each image will include the selected value in its filename. Leaving this option checked will overwrite the file on each run"
        " with a file named {image_name}_ReducedStars.fit"
        "\n\n-Once you're satisfied with the result, click 'Save & Close' to save your work and exit the script", True)
        self.raise_()
        self.activateWindow()

    def _run_reduction(self):
        try:
            star_reduction_value = round(self.slider.value() * self.resolution, 2)

            img_name_default_ext = f"{self.file_name_without_ext}{self.default_ext}"

            # Check if Apply was previously clicked by checking for starless file. If so, no need to run Starnet again
            if os.path.exists(f"starless_{img_name_default_ext}"):
                self.siril.log("Previous star reduction was detected.", color=LogColor.GREEN)
                starless = f"$starless_{img_name_default_ext}$"  # Wrap in $ for PixelMath
            else:
                self.siril.cmd("starnet", "-nostarmask")
                path = self.siril.get_image_filename()
                starless = f"${os.path.basename(path)}$"  # Wrap in $ for PixelMath

            # Pixel math for star reduction using selected value
            method = self.method_button_group.checkedButton().text()

            if method == "Halo":
                self.siril.update_progress("Generating PixelMath expressions",0.0)
                h1=f"((~(~{self.img_name_pm}/~{starless})-~(~mtf(~{star_reduction_value},{self.img_name_pm})/~mtf(~{star_reduction_value},{starless})))*~{starless})"
                h2=f"(~(~{self.img_name_pm}/~{starless})-~(~mtf(~{star_reduction_value},{self.img_name_pm})/~mtf(~{star_reduction_value},{starless})))"
                self.siril.update_progress("Processing image data with PixelMath",0.1)
                self.siril.cmd("pm", f"\"{self.img_name_pm}*~(({h1}+{h2})/2)\"")
                self.final_name = "Halo_{star_reduction_value}"
                self.siril.update_progress("Star Reduction is complete",1.0)

            elif method == "Star":
                self.siril.update_progress("Generating PixelMath expressions",0.0)
                s1=f"({self.img_name_pm}*~(~(max(0,min(1,{starless}/{self.img_name_pm})))*~{self.img_name_pm}))"
                s2=f"(max({s1},({self.img_name_pm}*{s1})+({s1}*~{s1})))"
                s3=f"({s1}*~(~(max(0,min(1,{starless}/{s1})))*~{s1}))"
                s4=f"(max({s3},({self.img_name_pm}*{s3})+({s3}*~{s3})))"
                s5=f"({s3}*~(~(max(0,min(1,{starless}/{s3})))*~{s3}))"
                s6=f"(max({s5},({self.img_name_pm}*{s5})+({s5}*~{s5})))"
                self.siril.update_progress("Processing image data with PixelMath",0.1)
                interactions = self.interactions_spin.value()
                mode = self.mode_button_group.checkedButton().text()
                if mode == "Strong":
                    self.siril.cmd("pm", f"\"(iif({interactions}==1,{s1},iif({interactions}==2,{s3},{s5})))\"")
                    self.final_name = "Star_Strong_{interactions}interactions"
                elif mode == "Moderate":
                    self.siril.cmd("pm", f"\"(iif({interactions}==1,{s2},iif({interactions}==2,{s4},{s6})))\"")
                    self.final_name = "Star_Moderate_{interactions}interactions"
                elif mode == "Soft":
                    self.siril.cmd("pm", f"\"((({self.img_name_pm}-({self.img_name_pm}-iif({interactions}==1,{s2},iif({interactions}==2,{s4},{s6}))))+({self.img_name_pm}*~({self.img_name_pm}-iif({interactions}==1,{s2},iif({interactions}==2,{s4},{s6})))))/2)\"")
                    self.final_name = "Star_Soft_{interactions}interactions"
                self.siril.update_progress("Star Reduction is complete",1.0)

            elif method == "Transfer":
                self.siril.update_progress("Generating PixelMath expressions",0.0)
                self.siril.update_progress("Processing image data with PixelMath",0.1)
                self.siril.cmd("pm", f"\"~((~mtf(~{star_reduction_value},{self.img_name_pm})/~mtf(~{star_reduction_value},{starless}))*~{starless})\"")
                self.final_name = "Transfer_{star_reduction_value}"
                self.siril.update_progress("Star Reduction is complete",1.0)

            self.siril.log("Star Reduction is complete!", color=LogColor.GREEN)
            return True
        except Exception as e:
            self.siril.log(f"Error in run_reduction: {str(e)}", color=LogColor.RED)
            return False

    def closeEvent(self, event):
        filename_overwrite = self.overwrite_checkbox.isChecked()
        current_img = self.img_name.replace(self.get_extension, "")
        if self.final_name != "0":
            if filename_overwrite:
                self.siril.cmd("save", f"\"{current_img}_ReducedStars{self.default_ext}\"")
                self.siril.cmd("load", f"\"{current_img}_ReducedStars{self.default_ext}\"")
            else:
                print("overwrite no")
                self.siril.cmd("save", f"\"{current_img}_ReducedStars_{self.final_name}{self.default_ext}\"")
                self.siril.cmd("load", f"\"{current_img}_ReducedStars_{self.final_name}{self.default_ext}\"")
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = StarReductionInterface()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
