# (c) Quark-Coder 2025
# Storage Friendly Stacking Script
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 1.2.0

import sys
import time
import threading
import sirilpy as s
s.ensure_installed("PyQt6", "watchdog")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTreeWidget, QTreeWidgetItem, QFileDialog,
    QMessageBox, QStyleFactory, QCheckBox, QSpacerItem, QSizePolicy,
    QDialog, QLabel, QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox,
    QLineEdit, QMenu, QTextEdit, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QPoint, QTimer
from PyQt6.QtGui import QPalette, QColor, QFont, QIcon

import json
import re
import glob
import os
import platform
import subprocess
import shutil
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

VERSION = "1.2.0"

DEFAULT_SETTINGS = {
    "general": {
        "clear_intermediate_files": True,
        "show_help_on_startup": True,
    },
    "bias_stack": {
        "method": "rej",          # "rej", "med", "sum"
        "sigma_low": 3.0,
        "sigma_high": 3.0,
        "norm": "none",           # "none", "add", "mul", "addscale", "mulscale"
        "output_norm": False,
        "rgb_equal": False,
        "force_32b": False,
        "use_external_master": False,        
        "external_master_path": "",          
    },
    "dark_stack": {
        "method": "rej",
        "sigma_low": 3.0,
        "sigma_high": 3.0,
        "norm": "none",
        "output_norm": False,
        "rgb_equal": False,
        "force_32b": False,
        "use_external_master": False,        
        "external_master_path": "",          
    },
    "flat_stack": {
        "method": "rej",
        "sigma_low": 3.0,
        "sigma_high": 3.0,
        "norm": "mul",            
        "output_norm": False,     
        "rgb_equal": False,
        "force_32b": False,
    },
    "light_stack": {
        "method": "rej",
        "sigma_low": 3.0,
        "sigma_high": 3.0,
        "norm": "addscale",
        "output_norm": True,
        "rgb_equal": True,
        "force_32b": True,
    },
    "calibrate_lights": {
        "use_cfa": True,          
        "use_equalize_cfa": True, 
        "use_debayer": True,      
    },
}

class CleaningHandler(FileSystemEventHandler):
    def __init__(self, si, monitor_dir, delete_dir, pattern_map):
        self.si = si
        self.monitor_dir = monitor_dir
        self.delete_dir = delete_dir
        self.pattern_map = pattern_map

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        filename = file_path.name

        for created_pat, (delete_pre, action_type) in self.pattern_map.items():
            if re.match(created_pat, filename):
                if action_type == 'group':
                    for f in self.delete_dir.glob(f'{delete_pre}_*.fit*'):
                        print(f"Deleting {f.name} because {filename} was created")
                        f.unlink()

                    seq = self.delete_dir / f'{delete_pre}_.seq'
                    if seq.exists():
                        print(f"Deleting {seq.name} because {filename} was created")
                        seq.unlink()

                    conv = self.delete_dir / f'{delete_pre}_conversion.txt'
                    if conv.exists():
                        print(f"Deleting {conv.name} because {filename} was created")
                        conv.unlink()

                elif action_type == 'individual':
                    match = re.match(r'.*_(\d{5})\.fit(s)?$', filename)
                    if match:
                        num = match.group(1)
                        to_del_fit = self.delete_dir / f'{delete_pre}_{num}.fit'
                        to_del_fits = self.delete_dir / f'{delete_pre}_{num}.fits'
                        if to_del_fit.exists():
                            print(f"Deleting {to_del_fit.name} because {filename} was created")
                            to_del_fit.unlink()
                        elif to_del_fits.exists():
                            print(f"Deleting {to_del_fits.name} because {filename} was created")
                            to_del_fits.unlink()

                elif action_type == 'seq':
                    seq = self.delete_dir / f'{delete_pre}_.seq'
                    if seq.exists():
                        print(f"Deleting {seq.name} because {filename} was created")
                        seq.unlink()

                    conv = self.delete_dir / f'{delete_pre}_conversion.txt'
                    if conv.exists():
                        print(f"Deleting {conv.name} because {filename} was created")
                        conv.unlink()

class SettingsDialog(QDialog):
    def __init__(self, parent, settings: dict, is_mono_mode: bool = False):
        super().__init__(parent)

        self.is_mono_mode = is_mono_mode
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(420, 380)

        self._settings = json.loads(json.dumps(settings))

        main_layout = QVBoxLayout(self)

        tabs = QTabWidget(self)
        main_layout.addWidget(tabs)
        
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)

        self._chk_clear_intermediate = QCheckBox("Clear intermediate files", general_tab)
        gset = self._settings.get("general", {})
        self._chk_clear_intermediate.setChecked(
            bool(gset.get("clear_intermediate_files", True))
        )

        general_layout.addWidget(self._chk_clear_intermediate)
        general_layout.addStretch(1)

        tabs.addTab(general_tab, "General")

        self._bias_widgets = self._create_stack_tab(
            title="Bias",
            key="bias_stack",
            parent=tabs,
            allow_output_norm=False,
            allow_rgb_equal=False,
            allow_32b=False,
            allow_external_master=True,
        )
        tabs.addTab(self._bias_widgets["widget"], "Bias")

        self._dark_widgets = self._create_stack_tab(
            title="Dark",
            key="dark_stack",
            parent=tabs,
            allow_output_norm=False,
            allow_rgb_equal=False,
            allow_32b=False,
            allow_external_master=True,
        )
        tabs.addTab(self._dark_widgets["widget"], "Dark")

        self._flat_widgets = self._create_stack_tab(
            title="Flat",
            key="flat_stack",
            parent=tabs,
            allow_output_norm=False,
            allow_rgb_equal=False,
            allow_32b=False,
            allow_external_master=False,
        )
        tabs.addTab(self._flat_widgets["widget"], "Flat")

        lights_tab = QWidget()
        lights_layout = QVBoxLayout(lights_tab)

        self._light_widgets = self._create_stack_tab(
            title="Light",
            key="light_stack",
            parent=lights_tab,          
            allow_output_norm=True,
            allow_rgb_equal=True,
            allow_32b=True,
        )
        lights_layout.addWidget(self._light_widgets["widget"])

        calib_group = QGroupBox("Calibrate lights", lights_tab)
        form = QFormLayout(calib_group)

        self._chk_cfa = QCheckBox("Cosmetic correction in CFA mode")
        self._chk_equalize = QCheckBox("Equalize CFA master flat")
        self._chk_debayer = QCheckBox("Demosaic before saving")

        cset = self._settings.get("calibrate_lights", {})
        self._chk_cfa.setChecked(bool(cset.get("use_cfa", True)))
        self._chk_equalize.setChecked(bool(cset.get("use_equalize_cfa", True)))
        self._chk_debayer.setChecked(bool(cset.get("use_debayer", True)))

        form.addRow(self._chk_cfa)
        form.addRow(self._chk_equalize)
        form.addRow(self._chk_debayer)

        if self.is_mono_mode:
            self._chk_cfa.setEnabled(False)
            self._chk_cfa.setChecked(False)
            self._chk_equalize.setEnabled(False)
            self._chk_equalize.setChecked(False)
            self._chk_debayer.setEnabled(False)
            self._chk_debayer.setChecked(False)

            # Add explanatory label
            mono_label = QLabel(
                "CFA and debayering options are not applicable for monochrome sensors.\n"
                "Each filter will be calibrated and stacked separately without demosaicing."
            )
            mono_label.setStyleSheet("color: #FFA500; font-style: italic; padding: 5px;")
            lights_layout.addWidget(mono_label)

        lights_layout.addWidget(calib_group)
        lights_layout.addStretch(1)

        tabs.addTab(lights_tab, "Lights")

        btns_layout = QHBoxLayout()
        btns_layout.addStretch(1)
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btns_layout.addWidget(ok_btn)
        btns_layout.addWidget(cancel_btn)
        main_layout.addLayout(btns_layout)

    def _create_stack_tab(
        self,
        title: str,
        key: str,
        parent: QWidget,
        allow_output_norm: bool,
        allow_rgb_equal: bool,
        allow_32b: bool,
        allow_external_master: bool = False,  
    ) -> dict:
        cfg = self._settings.get(key, {})

        tab = QWidget(parent)
        vbox = QVBoxLayout(tab)

        group = QGroupBox(title, tab)
        form = QFormLayout(group)

        method_combo = QComboBox(group)
        method_combo.addItem("Average with rejection", "rej")
        method_combo.addItem("Median", "med")
        method_combo.addItem("Sum", "sum")
        current_method = cfg.get("method", "rej")
        idx = max(0, method_combo.findData(current_method))
        method_combo.setCurrentIndex(idx)
        form.addRow("Stacking method:", method_combo)

        sigma_low = QDoubleSpinBox(group)
        sigma_low.setRange(0.0, 20.0)
        sigma_low.setDecimals(2)
        sigma_low.setSingleStep(0.1)
        sigma_low.setValue(float(cfg.get("sigma_low", 3.0)))

        sigma_high = QDoubleSpinBox(group)
        sigma_high.setRange(0.0, 20.0)
        sigma_high.setDecimals(2)
        sigma_high.setSingleStep(0.1)
        sigma_high.setValue(float(cfg.get("sigma_high", 3.0)))

        form.addRow("Sigma low:", sigma_low)
        form.addRow("Sigma high:", sigma_high)

        norm_combo = QComboBox(group)
        norm_combo.addItem("None", "none")
        norm_combo.addItem("Additive", "add")
        norm_combo.addItem("Multiplicative", "mul")
        norm_combo.addItem("Additive with scale", "addscale")
        norm_combo.addItem("Multiplicative with scale", "mulscale")
        current_norm = cfg.get("norm", "none")
        idx_n = max(0, norm_combo.findData(current_norm))
        norm_combo.setCurrentIndex(idx_n)
        form.addRow("Normalisation:", norm_combo)

        chk_output_norm = None
        chk_rgb_equal = None
        chk_32b = None

        if allow_output_norm or allow_rgb_equal or allow_32b:
            chk_output_norm = QCheckBox("Output normalization", group)
            chk_output_norm.setChecked(bool(cfg.get("output_norm", False)))
            chk_output_norm.setEnabled(allow_output_norm)

            chk_rgb_equal = QCheckBox("Equalize RGB channels", group)
            chk_rgb_equal.setChecked(bool(cfg.get("rgb_equal", False)))
            chk_rgb_equal.setEnabled(allow_rgb_equal)

            chk_32b = QCheckBox("Force 32-bit result", group)
            chk_32b.setChecked(bool(cfg.get("force_32b", False)))
            chk_32b.setEnabled(allow_32b)

            if allow_output_norm:
                form.addRow("Result options:", chk_output_norm)
            if allow_rgb_equal:
                form.addRow("", chk_rgb_equal)
            if allow_32b:
                form.addRow("", chk_32b)

        use_master_cb = None
        master_path_edit = None
        
        def _update_enabled_by_master(state: bool):
            enable_stack = not state
            method_combo.setEnabled(enable_stack)
            sigma_low.setEnabled(enable_stack)
            sigma_high.setEnabled(enable_stack)
            norm_combo.setEnabled(enable_stack)
            if chk_output_norm is not None:
                chk_output_norm.setEnabled(enable_stack)
            if chk_rgb_equal is not None:
                chk_rgb_equal.setEnabled(enable_stack)
            if chk_32b is not None:
                chk_32b.setEnabled(enable_stack)

        if allow_external_master:
            label_text = "Use masterBias instead" if key == "bias_stack" else "Use masterDark instead"
            use_master_cb = QCheckBox(label_text, group)
            use_master_cb.setChecked(bool(cfg.get("use_external_master", False)))
            form.addRow(use_master_cb)

            master_path_edit = QLineEdit(group)
            master_path_edit.setText(cfg.get("external_master_path", ""))

            browse_btn = QPushButton("...", group)
            browse_btn.setFixedWidth(30)

            path_layout = QHBoxLayout()
            path_layout.addWidget(master_path_edit)
            path_layout.addWidget(browse_btn)

            def _browse_master():
                start_dir = master_path_edit.text() or str(Path.home())
                fname, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select master file",
                    start_dir,
                    "FITS files (*.fit *.fits *.fts);;All files (*.*)",
                )
                if fname:
                    master_path_edit.setText(fname)

            browse_btn.clicked.connect(_browse_master)
            form.addRow("Master file:", path_layout)
            _update_enabled_by_master(use_master_cb.isChecked())
            use_master_cb.toggled.connect(_update_enabled_by_master) 
        vbox.addWidget(group)
        vbox.addStretch(1)

        return {
            "widget": tab,
            "key": key,
            "method": method_combo,
            "sigma_low": sigma_low,
            "sigma_high": sigma_high,
            "norm": norm_combo,
            "output_norm": chk_output_norm,
            "rgb_equal": chk_rgb_equal,
            "force_32b": chk_32b,
            "use_master": use_master_cb,          
            "master_path": master_path_edit,       
        }

    def apply_to_settings(self, settings: dict) -> None:
        for widgets in (
            self._bias_widgets,
            self._dark_widgets,
            self._flat_widgets,
            self._light_widgets,
        ):
            key = widgets["key"]
            sub = settings.setdefault(key, {})
            sub["method"] = widgets["method"].currentData()
            sub["sigma_low"] = round(float(widgets["sigma_low"].value()), 2)
            sub["sigma_high"] = round(float(widgets["sigma_high"].value()), 2)
            sub["norm"] = widgets["norm"].currentData()

            out_cb = widgets.get("output_norm")
            if out_cb is not None and out_cb.isEnabled():
                sub["output_norm"] = bool(out_cb.isChecked())

            rgb_cb = widgets.get("rgb_equal")
            if rgb_cb is not None and rgb_cb.isEnabled():
                sub["rgb_equal"] = bool(rgb_cb.isChecked())

            bit_cb = widgets.get("force_32b")
            if bit_cb is not None and bit_cb.isEnabled():
                sub["force_32b"] = bool(bit_cb.isChecked())

            use_master_cb = widgets.get("use_master")
            master_edit = widgets.get("master_path")
            if use_master_cb is not None and master_edit is not None:
                sub["use_external_master"] = bool(use_master_cb.isChecked())
                sub["external_master_path"] = master_edit.text().strip()

        cset = settings.setdefault("calibrate_lights", {})
        cset["use_cfa"] = bool(self._chk_cfa.isChecked())
        cset["use_equalize_cfa"] = bool(self._chk_equalize.isChecked())
        cset["use_debayer"] = bool(self._chk_debayer.isChecked())
        
        gen = settings.setdefault("general", {})
        gen["clear_intermediate_files"] = bool(
            self._chk_clear_intermediate.isChecked()
        )


class HelpDialog(QDialog):
    """Dialog showing usage instructions with a 'Never show again' checkbox."""

    OSC_TEXT = """This script allows stacking data from multiple imaging nights that use different calibration frames.
Master bias and Master dark can be placed in the main object folder (where the session_x folders are located).

In <b>OSC mode</b>, the folder hierarchy should look like this:

<pre>F:\\ASTROPHOTOGRAPHY\\SCRIPT_TEST\\DEV\\BODE_AND_CIGAR
├───[OPTIONAL]bias_stacked.fit
├───[OPTIONAL]dark_stacked.fit
├───session_1
│   ├───biases
│   ├───darks
│   ├───flats
│   └───lights
└───session_2
    ├───biases
    ├───darks
    ├───flats
    └───lights</pre>

Fill the folders with the corresponding files. After pressing the Session+ button, select the main folder containing the object name (in this example BODE_AND_CIGAR).

If you do not have a specific calibration frame, simply do not create that folder."""

    MONO_TEXT = """

In <b>Mono mode</b>, the folder hierarchy should look like this:

<pre>F:\\ASTROPHOTOGRAPHY\\SCRIPT_TEST\\DEV\\MONOCHROME_EXAMPLE_FOLDER_HIERARCHY
├───bias_stacked.fit
├───dark_stacked.fit
├───session_1
│   ├───R
│   │   ├───flats
│   │   └───lights
│   ├───G
│   │   ├───flats
│   │   └───lights
│   └───B
│       ├───flats
│       └───lights
└───session_2
    ├───R
    │   ├───flats
    │   └───lights
    ├───G
    │   ├───flats
    │   └───lights
    └───B
        ├───flats
        └───lights</pre>"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Storage Friendly Stacking - Help")
        self.setMinimumSize(700, 550)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        # Content widget with text
        content = QWidget()
        content_layout = QVBoxLayout(content)

        # Create label with rich text
        help_label = QLabel(self.OSC_TEXT + self.MONO_TEXT)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        help_label.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                color: #e0e0e0;
                padding: 16px;
                border-radius: 8px;
            }
            pre {
                background-color: #1a1a1a;
                padding: 12px;
                border-radius: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)

        content_layout.addWidget(help_label)
        scroll.setWidget(content)

        layout.addWidget(scroll)

        self.never_show_cb = QCheckBox("Never show this again")
        self.never_show_cb.setStyleSheet("""
            QCheckBox {
                color: white;
                spacing: 5px;
                padding: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #2e2e2e;
                background-color: #2e2e2e;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #228B22;
                border-color: #228B22;
            }
        """)
        layout.addWidget(self.never_show_cb)

        # OK button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setMinimumWidth(80)
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #5a5a5a;
            }
        """)
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

    def never_show_again(self) -> bool:
        """Return True if user checked 'Never show this again'."""
        return self.never_show_cb.isChecked()



class LauncherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.si = s.SirilInterface()
        self.si.connect()
        self.base_dir = Path(self.si.get_siril_wd().strip() or Path.cwd())
        self.process_dir = self.base_dir / 'process'
        self.masters_dir = self.base_dir / 'masters'
        
        user_data_dir = Path(self.si.get_siril_userdatadir().strip())
        self.settings_path = user_data_dir / "StorageFriendlyStacking" / "settings.json"
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings = self._load_settings()
        
        self.sessions = {}
        self.current_session = None
        self.sessions_parent = None
        self.first_session_added = False
        self.is_mono_mode = False
        self.frame_order = ['light', 'dark', 'flat', 'bias']
        self.image_extensions = {
            '.fit', '.fits', '.fts', '.tif', '.tiff', '.xisf',
            '.cr2', '.cr3', '.nef', '.arw', '.orf', '.raf', '.rw2',
            '.dng', '.pef', '.sr2', '.raw', '.jpg', '.jpeg', '.png', '.bmp'
        }
        self.frame_names = {
            'light': ['light', 'lights'],
            'dark': ['dark', 'darks'],
            'flat': ['flat', 'flats'],
            'bias': ['bias', 'biases']
        }
        self.all_frame_names = set(name for names in self.frame_names.values() for name in names)

        # Monochrome filter names and priorities
        self.mono_filters = ['L', 'R', 'G', 'B', 'Ha', 'Oiii', 'Sii']
        self.mono_filter_priority = {
            'L': 0, 'R': 1, 'G': 2, 'B': 3, 'Ha': 4, 'Oiii': 5, 'Sii': 6
        }

        self._build_ui()

        if self.settings.get('general', {}).get('show_help_on_startup', True):
            QTimer.singleShot(100, self._show_startup_help)

    def _show_startup_help(self):
        """Show the help dialog and save the 'never show again' preference."""
        dialog = HelpDialog(self)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2a2a2a;
            }
        """)
        result = dialog.exec()
        if dialog.never_show_again():
            self.settings.setdefault('general', {})['show_help_on_startup'] = False
            self._save_settings()

    def _build_ui(self):
        self.setWindowTitle(f"Storage Friendly Stacking v{VERSION}")
        win_width, win_height = 530, 350
        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()
        x = (screen_width // 2) - (win_width // 2)
        y = (screen_height // 2) - (win_height // 2)
        self.setGeometry(x, y, win_width, win_height)

        self._set_dark_theme()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedSize(30, 30)
        self.settings_btn.clicked.connect(self._open_settings)
        self.settings_btn.setStyleSheet("""
        QPushButton {
            background-color: #2e2e2e;
            color: white;
            border: none;
            border-radius: 15px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #444444;
        }
        QPushButton:pressed {
            background-color: #5a5a5a;
        }
        """)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(4, 4, 4, 0)
        top_bar.addStretch(1)
        top_bar.addWidget(self.settings_btn)
        main_layout.addLayout(top_bar)

        tab_widget = QTabWidget()
        tab_widget.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(tab_widget)

        multisession_widget = QWidget()
        multisession_layout = QVBoxLayout(multisession_widget)
        multisession_layout.setContentsMargins(0, 0, 0, 0)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setColumnWidth(0, 470)
        self.tree.setColumnWidth(1, 30)
        self.tree.setStyleSheet("""
            QTreeWidget {
                background-color: #2a2a2a;
                color: white;

            }
            QTreeWidget::item:selected {
                background-color: #505050;

            }
            """)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)

        multisession_layout.addWidget(self.tree)

        checkbox_layout = QHBoxLayout()
        self.osc_checkbox = QCheckBox("Color")
        self.mono_checkbox = QCheckBox("Monochrome")
        self.osc_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                border: 2px solid #2e2e2e;
                background-color: #2e2e2e;
            }
            QCheckBox::indicator:checked {
                background-color: #228B22;
                border-color: #228B22;
            }
        """)
        self.mono_checkbox.setStyleSheet(self.osc_checkbox.styleSheet())

        self.osc_checkbox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.mono_checkbox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        checkbox_layout.addWidget(self.osc_checkbox)
        checkbox_layout.addSpacing(10)  
        checkbox_layout.addWidget(self.mono_checkbox)
        checkbox_layout.addStretch(1)  

        def on_osc_toggled(checked):
            if checked:
                self.is_mono_mode = False
                self.mono_checkbox.blockSignals(True)
                self.mono_checkbox.setChecked(False)
                self.mono_checkbox.blockSignals(False)

        def on_mono_toggled(checked):
            if checked:
                self.is_mono_mode = True
                self.osc_checkbox.blockSignals(True)
                self.osc_checkbox.setChecked(False)
                self.osc_checkbox.blockSignals(False)

        self.osc_checkbox.toggled.connect(on_osc_toggled)
        self.mono_checkbox.toggled.connect(on_mono_toggled)

        self.osc_checkbox.setChecked(True)

        multisession_layout.addLayout(checkbox_layout)

        btn_layout = QHBoxLayout()
        self.session_btn = QPushButton("Session+")
        self.session_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.session_btn.clicked.connect(self._add_session)
        self.session_btn.setStyleSheet("""
            QPushButton {
                background-color: #2e2e2e;
                color: white;
                padding: 5px 10px;
                border: none;

            }
            QPushButton:hover {
                background-color: #444444;

            }
            QPushButton:pressed {
                background-color: #5a5a5a;

            }
            """)
        btn_layout.addWidget(self.session_btn)

        self.light_btn = QPushButton("Light+")
        self.light_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.light_btn.clicked.connect(self._on_light_btn_clicked)
        self.light_btn.setEnabled(False)
        self.light_btn.setStyleSheet("""
            QPushButton {
                background-color: #2e2e2e;
                color: white;
                padding: 5px 10px;
                border: none;

            }
            QPushButton:hover {
                background-color: #444444;

            }
            QPushButton:pressed {
                background-color: #5a5a5a;

            }
            QPushButton:disabled {
                background-color: #2e2e2e;
                color: gray;

            }
            """)
        btn_layout.addWidget(self.light_btn)

        self.dark_btn = QPushButton("Dark+")
        self.dark_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.dark_btn.clicked.connect(self._on_dark_btn_clicked)
        self.dark_btn.setEnabled(False)
        self.dark_btn.setStyleSheet(self.light_btn.styleSheet())
        btn_layout.addWidget(self.dark_btn)

        self.flat_btn = QPushButton("Flat+")
        self.flat_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.flat_btn.clicked.connect(self._on_flat_btn_clicked)
        self.flat_btn.setEnabled(False)
        self.flat_btn.setStyleSheet(self.light_btn.styleSheet())
        btn_layout.addWidget(self.flat_btn)

        self.bias_btn = QPushButton("Bias+")
        self.bias_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.bias_btn.clicked.connect(self._on_bias_btn_clicked)
        self.bias_btn.setEnabled(False)
        self.bias_btn.setStyleSheet(self.light_btn.styleSheet())
        btn_layout.addWidget(self.bias_btn)

        self.run_multi_btn = QPushButton("Run")
        self.run_multi_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.run_multi_btn.clicked.connect(self._run_multisession)
        self.run_multi_btn.setStyleSheet("""
            QPushButton {
                background-color: #228B22;
                color: white;
                padding: 7px 40px;
                border: none;

            }
            QPushButton:hover {
                background-color: #32CD32;

            }
            QPushButton:pressed {
                background-color: #006400;

            }
            """)
        btn_layout.addWidget(self.run_multi_btn)

        multisession_layout.addLayout(btn_layout)

        tab_widget.addTab(multisession_widget, "Multisession")
        
    def _open_settings(self):
        dialog = SettingsDialog(self, self.settings, self.is_mono_mode)
        if dialog.exec():
            dialog.apply_to_settings(self.settings)
            self._save_settings()
            
    def _load_settings(self) -> dict:
        cfg = json.loads(json.dumps(DEFAULT_SETTINGS))  
        if self.settings_path.exists():
            try:
                with self.settings_path.open("r", encoding="utf-8") as f:
                    stored = json.load(f)
                for key, sub in stored.items():
                    if key in cfg and isinstance(sub, dict):
                        cfg[key].update(sub)
            except Exception as e:
                print(f"Failed to load settings {self.settings_path}: {e}")
        return cfg

    def _save_settings(self) -> None:
        try:
            with self.settings_path.open("w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save settings {self.settings_path}: {e}")
        
    def _build_stack_cmd(self, sequence: str, cfg_key: str, out_arg: str) -> str:
        cfg = self.settings.get(cfg_key, {})
        parts: list[str] = ["stack", sequence]

        method = cfg.get("method", "rej") or "rej"
        parts.append(method)

        if method == "rej":
            sigma_low = float(cfg.get("sigma_low", 3.0))
            sigma_high = float(cfg.get("sigma_high", 3.0))
            parts.append(f"{sigma_low:g}")
            parts.append(f"{sigma_high:g}")

        norm = cfg.get("norm", "none")
        if norm == "none" or not norm:
            parts.append("-nonorm")
        else:
            parts.append(f"-norm={norm}")

        if cfg.get("output_norm"):
            parts.append("-output_norm")
        if cfg.get("rgb_equal"):
            parts.append("-rgb_equal")
        if cfg.get("force_32b"):
            parts.append("-32b")

        parts.append(f"-out={out_arg}")
        return " ".join(parts)
        
    def _build_light_calibrate_args(self) -> list[str]:
        cfg = self.settings.get("calibrate_lights", {})
        args: list[str] = []
        if cfg.get("use_cfa", True):
            args.append("-cfa")
        if cfg.get("use_equalize_cfa", True):
            args.append("-equalize_cfa")
        if cfg.get("use_debayer", True):
            args.append("-debayer")
        return args

    def _detect_mono_sessions(self, folder_path: Path) -> dict:
        """
        Detect monochrome session structure.
        Returns: {filter_name: {'lights': [...], 'flats': [...]}}
        """
        filter_data = {}
        for filter_name in self.mono_filters:
            filter_dir = folder_path / filter_name
            if not filter_dir.is_dir():
                continue

            lights_dir = filter_dir / 'lights'
            flats_dir = filter_dir / 'flats'

            lights = []
            flats = []

            if lights_dir.exists():
                lights = [f for f in lights_dir.rglob('*')
                         if f.suffix.lower() in self.image_extensions]

            if flats_dir.exists():
                flats = [f for f in flats_dir.rglob('*')
                        if f.suffix.lower() in self.image_extensions]

            # Accept filter if directory structure exists, even if empty
            # This allows users to set up structure and add files later
            if lights_dir.exists() or flats_dir.exists():
                filter_data[filter_name] = {
                    'lights': lights if lights else [],
                    'flats': flats if flats else [],
                    'lights_dir': lights_dir if lights_dir.exists() else None,
                    'flats_dir': flats_dir if flats_dir.exists() else None,
                }

        return filter_data

    def _add_mono_session(self, folder_path: Path) -> bool:
        """
        Add monochrome session(s) to tree.
        Handles both single session and multi-session structures.
        Returns: True if successful, False otherwise
        """
        # Check for space in path
        if ' ' in str(folder_path):
            QMessageBox.critical(self, "Error", "The path must not contain spaces.")
            return False

        potential_sessions = []

        potential_sessions.append(folder_path)

        sub_sessions = []
        for sub in sorted(folder_path.iterdir()):
            if not sub.is_dir():
                continue

            if sub.name.lower() in ['biases', 'darks', 'flats', 'lights', 'process', 'masters', 'calibrated']:
                continue

            filter_data = self._detect_mono_sessions(sub)
            if filter_data:
                sub_sessions.append((sub, filter_data))

        has_filter_data = self._detect_mono_sessions(folder_path) is not None
        if not has_filter_data and not sub_sessions:
            return False

        if self.sessions_parent is None:
            self.sessions_parent = folder_path
            # storage when _detect_and_auto_select_masters is called after item creation

        existing_names = [
            self.tree.topLevelItem(i).text(0)
            for i in range(self.tree.topLevelItemCount())
        ]

        parent_bias_dir = self.sessions_parent / 'biases'
        parent_dark_dir = self.sessions_parent / 'darks'

        shared_bias_data = None
        shared_dark_data = None

        if parent_bias_dir.exists() and parent_bias_dir.is_dir():
            bias_files = [f for f in parent_bias_dir.rglob('*')
                         if f.suffix.lower() in self.image_extensions]
            if bias_files:
                shared_bias_data = {
                    'files': bias_files,
                    'count': len(bias_files),
                    'folder': parent_bias_dir
                }

        if parent_dark_dir.exists() and parent_dark_dir.is_dir():
            dark_files = [f for f in parent_dark_dir.rglob('*')
                         if f.suffix.lower() in self.image_extensions]
            if dark_files:
                shared_dark_data = {
                    'files': dark_files,
                    'count': len(dark_files),
                    'folder': parent_dark_dir
                }

        has_sub_sessions = len(sub_sessions) > 0

        # Add each session
        for sess_path in potential_sessions:
            if sess_path.name in existing_names:
                continue

            filter_data = self._detect_mono_sessions(sess_path) if not has_sub_sessions or sess_path != folder_path else None

            # Create session item
            item = QTreeWidgetItem([sess_path.name, '❌'])

            if has_sub_sessions and sess_path == folder_path:
                self.tree.addTopLevelItem(item)
                iid = id(item)

                # Store minimal data for parent folder (mainly for folder reference)
                self.sessions[iid] = {
                    'folder': sess_path,
                    'is_mono': True,
                    'filters': {},
                    'bias': shared_bias_data,
                    'dark': shared_dark_data,
                    'is_parent': True,  # Mark as parent folder
                    'external_masters': {},  # Store session-local master paths
                }

                if shared_bias_data:
                    bias_text = f'bias ({shared_bias_data["count"]} images)'
                    bias_child = QTreeWidgetItem([bias_text, '❌'])
                    item.addChild(bias_child)

                if shared_dark_data:
                    dark_text = f'dark ({shared_dark_data["count"]} images)'
                    dark_child = QTreeWidgetItem([dark_text, '❌'])
                    item.addChild(dark_child)

                parent_filter_data = self._detect_mono_sessions(folder_path)
                if parent_filter_data:
                    for filter_name in sorted(parent_filter_data.keys(),
                                             key=lambda x: self.mono_filter_priority.get(x, 99)):
                        data = parent_filter_data[filter_name]

                        filter_item = QTreeWidgetItem([filter_name, '❌'])
                        item.addChild(filter_item)

                        # Add lights sub-item
                        light_count = len(data['lights'])
                        lights_text = f'lights ({light_count} images)'
                        lights_child = QTreeWidgetItem([lights_text, '❌'])
                        filter_item.addChild(lights_child)

                        # Add flats sub-item
                        flat_count = len(data['flats'])
                        flats_text = f'flats ({flat_count} images)'
                        flats_child = QTreeWidgetItem([flats_text, '❌'])
                        filter_item.addChild(flats_child)

                        # Store data for easy access
                        self.sessions[iid][f'{filter_name}_lights'] = {
                            'files': data['lights'],
                            'count': light_count,
                            'folder': data['lights_dir'],
                            'filter_name': filter_name,
                            'frame_type': 'lights'
                        }

                        self.sessions[iid][f'{filter_name}_flats'] = {
                            'files': data['flats'],
                            'count': flat_count,
                            'folder': data['flats_dir'],
                            'filter_name': filter_name,
                            'frame_type': 'flats'
                        }

                # Add sub-sessions as children
                for sub_path, sub_filter_data in sub_sessions:
                    sub_item = QTreeWidgetItem([sub_path.name, '❌'])
                    item.addChild(sub_item)
                    sub_iid = id(sub_item)

                    session_bias_dir = sub_path / 'biases'
                    session_dark_dir = sub_path / 'darks'

                    session_bias_data = None
                    session_dark_data = None

                    if session_bias_dir.exists() and session_bias_dir.is_dir():
                        bias_files = [f for f in session_bias_dir.rglob('*')
                                     if f.suffix.lower() in self.image_extensions]
                        if bias_files:
                            session_bias_data = {
                                'files': bias_files,
                                'count': len(bias_files),
                                'folder': session_bias_dir
                            }

                    if session_dark_dir.exists() and session_dark_dir.is_dir():
                        dark_files = [f for f in session_dark_dir.rglob('*')
                                     if f.suffix.lower() in self.image_extensions]
                        if dark_files:
                            session_dark_data = {
                                'files': dark_files,
                                'count': len(dark_files),
                                'folder': session_dark_dir
                            }

                    # Priority: session-specific > parent-level shared
                    bias_to_use = session_bias_data if session_bias_data else shared_bias_data
                    dark_to_use = session_dark_data if session_dark_data else shared_dark_data

                    # Store session data for sub-session
                    self.sessions[sub_iid] = {
                        'folder': sub_path,
                        'is_mono': True,
                        'filters': sub_filter_data,
                        'bias': bias_to_use,
                        'dark': dark_to_use,
                        'parent_iid': iid,  # Reference to parent
                    }

                    for filter_name in sorted(sub_filter_data.keys(),
                                             key=lambda x: self.mono_filter_priority.get(x, 99)):
                        data = sub_filter_data[filter_name]

                        filter_item = QTreeWidgetItem([filter_name, '❌'])
                        sub_item.addChild(filter_item)

                        # Add lights sub-item
                        light_count = len(data['lights'])
                        lights_text = f'lights ({light_count} images)'
                        lights_child = QTreeWidgetItem([lights_text, '❌'])
                        filter_item.addChild(lights_child)

                        # Add flats sub-item
                        flat_count = len(data['flats'])
                        flats_text = f'flats ({flat_count} images)'
                        flats_child = QTreeWidgetItem([flats_text, '❌'])
                        filter_item.addChild(flats_child)

                        self.sessions[sub_iid][f'{filter_name}_lights'] = {
                            'files': data['lights'],
                            'count': light_count,
                            'folder': data['lights_dir'],
                            'filter_name': filter_name,
                            'frame_type': 'lights'
                        }

                        self.sessions[sub_iid][f'{filter_name}_flats'] = {
                            'files': data['flats'],
                            'count': flat_count,
                            'folder': data['flats_dir'],
                            'filter_name': filter_name,
                            'frame_type': 'flats'
                        }

                    if session_bias_data:
                        bias_text = f'bias ({session_bias_data["count"]} images)'
                        bias_child = QTreeWidgetItem([bias_text, '❌'])
                        sub_item.addChild(bias_child)

                    if session_dark_data:
                        dark_text = f'dark ({session_dark_data["count"]} images)'
                        dark_child = QTreeWidgetItem([dark_text, '❌'])
                        sub_item.addChild(dark_child)

                    sub_item.setExpanded(True)

                self._detect_and_auto_select_masters(folder_path, parent_item=item, parent_iid=iid)
            else:
                # Regular session (standalone or when parent has its own filter data)
                self.tree.addTopLevelItem(item)
                iid = id(item)

                # Store session data
                self.sessions[iid] = {
                    'folder': sess_path,
                    'is_mono': True,
                    'filters': filter_data if filter_data else {},
                    'bias': shared_bias_data,
                    'dark': shared_dark_data,
                    'external_masters': {},  # Store session-local master paths
                }

                if shared_bias_data:
                    bias_text = f'bias ({shared_bias_data["count"]} images)'
                    bias_child = QTreeWidgetItem([bias_text, '❌'])
                    item.addChild(bias_child)

                if shared_dark_data:
                    dark_text = f'dark ({shared_dark_data["count"]} images)'
                    dark_child = QTreeWidgetItem([dark_text, '❌'])
                    item.addChild(dark_child)

                if filter_data:
                    for filter_name in sorted(filter_data.keys(),
                                             key=lambda x: self.mono_filter_priority.get(x, 99)):
                        data = filter_data[filter_name]

                        filter_item = QTreeWidgetItem([filter_name, '❌'])
                        item.addChild(filter_item)

                        # Add lights sub-item
                        light_count = len(data['lights'])
                        lights_text = f'lights ({light_count} images)'
                        lights_child = QTreeWidgetItem([lights_text, '❌'])
                        filter_item.addChild(lights_child)

                        # Add flats sub-item
                        flat_count = len(data['flats'])
                        flats_text = f'flats ({flat_count} images)'
                        flats_child = QTreeWidgetItem([flats_text, '❌'])
                        filter_item.addChild(flats_child)

                        self.sessions[iid][f'{filter_name}_lights'] = {
                            'files': data['lights'],
                            'count': light_count,
                            'folder': data['lights_dir'],
                            'filter_name': filter_name,
                            'frame_type': 'lights'
                        }

                        self.sessions[iid][f'{filter_name}_flats'] = {
                            'files': data['flats'],
                            'count': flat_count,
                            'folder': data['flats_dir'],
                            'filter_name': filter_name,
                            'frame_type': 'flats'
                        }

            item.setExpanded(True)

        if potential_sessions:
            last_item = self.tree.topLevelItem(
                self.tree.topLevelItemCount() - 1
            )
            self.tree.setCurrentItem(last_item)
            self.current_session = id(last_item)

        return True

    def _run_monochrome_session(self, sess: dict, sess_id: int,
                               base_dir: Path, calibrated_base: Path,
                               clear_intermediate: bool) -> list:
        """
        Process a single monochrome session.
        Returns: List of (filter_name, list_of_calibrated_files) tuples
        """
        session_dir = sess["folder"]
        process_dir = session_dir / "process"
        session_masters_dir = session_dir / "masters"
        global_masters_dir = base_dir / "masters"  # Shared across all sessions

        process_dir.mkdir(exist_ok=True)
        session_masters_dir.mkdir(exist_ok=True)
        global_masters_dir.mkdir(exist_ok=True)

        if clear_intermediate:
            global_masters_map = {
                r'^bias_stacked\.fit(s)?$': ('bias', 'group'),
                r'^dark_stacked\.fit(s)?$': ('dark', 'group'),
            }

            if str(global_masters_dir) not in self._observer_dirs:
                global_masters_handler = CleaningHandler(
                    self.si, global_masters_dir, process_dir, global_masters_map
                )
                global_masters_observer = Observer()
                global_masters_observer.schedule(
                    global_masters_handler, str(global_masters_dir), recursive=False
                )
                self.observers.append(global_masters_observer)
                global_masters_observer.start()
                self._observer_dirs.add(str(global_masters_dir))

            session_masters_map = {
                r'^pp_flat_stacked\.fit(s)?$': ('pp_flat', 'group'),
                r'^flat_stacked\.fit(s)?$': ('flat', 'group'),
            }

            session_masters_handler = CleaningHandler(
                self.si, session_masters_dir, process_dir, session_masters_map
            )
            session_masters_observer = Observer()
            session_masters_observer.schedule(
                session_masters_handler, str(session_masters_dir), recursive=False
            )
            self.observers.append(session_masters_observer)
            session_masters_observer.start()

            process_map = {
                r'^pp_light_\d{5}\.fit(s)?$': ('light', 'individual'),
                r'^pp_flat_\d{5}\.fit(s)?$': ('flat', 'individual'),
                r'^pp_flat_stacked\.fit(s)?$': ('pp_flat', 'group'),
            }

            process_handler = CleaningHandler(
                self.si, process_dir, process_dir, process_map
            )
            process_observer = Observer()
            process_observer.schedule(
                process_handler, str(process_dir), recursive=False
            )
            self.observers.append(process_observer)
            process_observer.start()

            for filter_name in sess['filters'].keys():
                filter_calibrated_dir = calibrated_base / filter_name
                filter_calibrated_dir.mkdir(exist_ok=True)

                if str(filter_calibrated_dir) not in self._observer_dirs:
                    # Watch in the filter's calibrated directory
                    filter_calib_map = {
                        r'^r_pp_light_\d{5}\.fit(s)?$': ('pp_light', 'individual'),
                        rf'^{re.escape(filter_name)}_stacked\.fit(s)$': ('r_pp_light', 'group'),
                    }

                    filter_calib_handler = CleaningHandler(
                        self.si, filter_calibrated_dir, filter_calibrated_dir, filter_calib_map
                    )
                    filter_calib_observer = Observer()
                    filter_calib_observer.schedule(
                        filter_calib_handler, str(filter_calibrated_dir), recursive=False
                    )
                    self.observers.append(filter_calib_observer)
                    filter_calib_observer.start()
                    self._observer_dirs.add(str(filter_calibrated_dir))

        self.si.cmd(f"cd {str(session_dir)}")

        bias_master_available = False
        dark_master_available = False

        session_ext_masters = sess.get('external_masters', {})

        bias_external_path = None
        if 'bias' in session_ext_masters:
            bias_external_path = session_ext_masters['bias']
        else:
            bias_cfg = self.settings.get("bias_stack", {})
            if bias_cfg.get("use_external_master") and bias_cfg.get("external_master_path"):
                bias_external_path = bias_cfg["external_master_path"]

        use_bias_ext = bias_external_path is not None

        if sess["bias"]:
            if use_bias_ext:
                src = Path(bias_external_path)
                if src.is_file():
                    dest = global_masters_dir / f"bias_stacked{src.suffix}"
                    shutil.copy2(src, dest)
                    print(f"Using external master bias: {src} -> {dest}")
                    bias_master_available = True
                else:
                    print(f"External master bias not found: {src}, falling back to stacking")
                    use_bias_ext = False

            if not use_bias_ext:
                bias_dir = sess["bias"]["folder"]
                self.si.cmd(f"cd {str(bias_dir)}")
                self.si.cmd("convert bias -out=../process")
                self.si.cmd("cd ../process")

                bias_output = str(global_masters_dir / "bias_stacked")
                bias_stack_cmd = self._build_stack_cmd(
                    sequence="bias",
                    cfg_key="bias_stack",
                    out_arg=bias_output,
                )
                self.si.cmd(bias_stack_cmd)

                stacked_file = global_masters_dir / "bias_stacked.fit"
                if not stacked_file.exists():
                    stacked_file = global_masters_dir / "bias_stacked.fits"

                if not stacked_file.exists():
                    raise RuntimeError(
                        f"Bias stacking failed: No output file found at {global_masters_dir / 'bias_stacked.fit[rip]'}"
                    )

                print(f"Successfully created global bias master: {stacked_file}")
                self.si.cmd("cd ..")
                bias_master_available = True
        elif use_bias_ext:
            src = Path(bias_external_path)
            if src.is_file():
                dest = global_masters_dir / f"bias_stacked{src.suffix}"
                shutil.copy2(src, dest)
                print(f"Using external master bias (no session bias): {src} -> {dest}")
                bias_master_available = True
            else:
                print(f"External master bias not found: {src}")

        dark_external_path = None
        if 'dark' in session_ext_masters:
            dark_external_path = session_ext_masters['dark']
        else:
            dark_cfg = self.settings.get("dark_stack", {})
            if dark_cfg.get("use_external_master") and dark_cfg.get("external_master_path"):
                dark_external_path = dark_cfg["external_master_path"]

        use_dark_ext = dark_external_path is not None

        if sess["dark"]:
            if use_dark_ext:
                src = Path(dark_external_path)
                if src.is_file():
                    dest = global_masters_dir / f"dark_stacked{src.suffix}"
                    shutil.copy2(src, dest)
                    print(f"Using external master dark: {src} -> {dest}")
                    dark_master_available = True
                else:
                    print(f"External master dark not found: {src}, falling back to stacking")
                    use_dark_ext = False

            if not use_dark_ext:
                dark_dir = sess["dark"]["folder"]
                self.si.cmd(f"cd {str(dark_dir)}")
                self.si.cmd("convert dark -out=../process")
                self.si.cmd("cd ../process")

                dark_output = str(global_masters_dir / "dark_stacked")
                dark_cmd = self._build_stack_cmd(
                    sequence="dark",
                    cfg_key="dark_stack",
                    out_arg=dark_output,
                )
                self.si.cmd(dark_cmd)

                print(f"Successfully created global dark master: {global_masters_dir / 'dark_stacked'}")
                self.si.cmd("cd ..")
                dark_master_available = True
        elif use_dark_ext:
            src = Path(dark_external_path)
            if src.is_file():
                dest = global_masters_dir / f"dark_stacked{src.suffix}"
                shutil.copy2(src, dest)
                print(f"Using external master dark (no session dark): {src} -> {dest}")
                dark_master_available = True
            else:
                print(f"External master dark not found: {src}")

        calibrated_results = []

        for filter_name in sorted(sess['filters'].keys(),
                                 key=lambda x: self.mono_filter_priority.get(x, 99)):
            filter_data = sess['filters'][filter_name]

            if filter_data.get('invalid') or not filter_data['lights']:
                if filter_data.get('invalid'):
                    print(f"\n=== Skipping filter {filter_name} (marked as INVALID - no lights) ===")
                continue

            print(f"\n=== Processing filter {filter_name} ===")

            filter_calibrated_dir = calibrated_base / filter_name
            filter_calibrated_dir.mkdir(exist_ok=True)

            flats_bias_calibrated = False

            if filter_data['flats_dir'] and filter_data['flats']:
                flats_dir = filter_data['flats_dir']
                self.si.cmd(f"cd {str(flats_dir)}")
                self.si.cmd(f"convert flat -out=../../process")
                self.si.cmd("cd ../../process")

                # Verify flat files were created
                flat_files = list(process_dir.glob("flat_*.fit*"))
                if not flat_files:
                    print(f"Warning: No flat files found in {process_dir} after conversion")
                    self.si.cmd(f"cd {str(session_dir)}")
                    continue

                if bias_master_available:
                    bias_file = global_masters_dir / "bias_stacked.fit"
                    if not bias_file.exists():
                        bias_file = global_masters_dir / "bias_stacked.fits"

                    if bias_file.exists():
                        try:

                            bias_path_with_ext = str(bias_file)
                            self.si.cmd(f"calibrate flat -bias={bias_path_with_ext}")

                            pp_flat_files = list(process_dir.glob("pp_flat_*.fit*"))

                            # Save flats to SESSION masters (filter-specific)
                            pp_flat_cmd = self._build_stack_cmd(
                                sequence="pp_flat",
                                cfg_key="flat_stack",
                                out_arg=str(session_masters_dir / "pp_flat_stacked"),
                            )
                            self.si.cmd(pp_flat_cmd)
                            flats_bias_calibrated = True  # Mark as bias-calibrated
                        except Exception as e:
                            print(f"Warning: Flat calibration/stacking failed: {e}")
                            print(f"Warning: Skipping flat calibration for filter {filter_name}")
                            filter_data['flats'] = None
                    else:
                        print(f"Warning: Bias master file not found at {global_masters_dir}, skipping flat calibration")
                        flat_output = str(session_masters_dir / "flat_stacked")
                        flat_cmd = self._build_stack_cmd(
                            sequence="flat",
                            cfg_key="flat_stack",
                            out_arg=flat_output,
                        )
                        self.si.cmd(flat_cmd)
                else:
                    flat_output = str(session_masters_dir / "flat_stacked")
                    flat_cmd = self._build_stack_cmd(
                        sequence="flat",
                        cfg_key="flat_stack",
                        out_arg=flat_output,
                    )
                    self.si.cmd(flat_cmd)

                self.si.cmd("cd ..")

            self.si.cmd(f"cd {str(session_dir)}")

            lights_dir = filter_data['lights_dir']
            self.si.cmd(f"cd {str(lights_dir)}")
            self.si.cmd("convert light -out=../../process")
            self.si.cmd("cd ../../process")

            args: list[str] = []

            if dark_master_available:
                args.append(f"-dark={global_masters_dir / 'dark_stacked'}")

            # Flat calibration is filter-specific in monochrome
            if filter_data['flats']:
                if flats_bias_calibrated:
                    # Flats were calibrated with bias (SESSION-specific)
                    args.append(f"-flat={session_masters_dir / 'pp_flat_stacked'}")
                else:
                    # Flats were NOT calibrated with bias (use raw stacked flats, SESSION-specific)
                    args.append(f"-flat={session_masters_dir / 'flat_stacked'}")

            # NO debayer for monochrome!

            # Calibrate lights
            if args:
                calibrate_cmd = "calibrate light " + " ".join(args)
                self.si.cmd(calibrate_cmd)

            # Move pp_light files to filter-specific calibrated directory
            existing = sorted(filter_calibrated_dir.glob("pp_light_*.fit*"))
            offset = len(existing)

            for idx, f in enumerate(sorted(process_dir.glob("pp_light_*.fit*")),
                                   start=1):
                new_idx = offset + idx
                ext = f.suffix
                new_name = f"pp_light_{new_idx:05d}{ext}"
                shutil.move(str(f), str(filter_calibrated_dir / new_name))

            # Delete pp_light_.seq and pp_light_conversion.txt if enabled
            if clear_intermediate:
                pp_seq = process_dir / 'pp_light_.seq'
                if pp_seq.exists():
                    print(f"Deleting {pp_seq.name} after moving pp_light files")
                    pp_seq.unlink()

                pp_conv = process_dir / 'pp_light_conversion.txt'
                if pp_conv.exists():
                    print(f"Deleting {pp_conv.name} after moving pp_light files")
                    pp_conv.unlink()

            calibrated_results.append((filter_name, filter_calibrated_dir))

            self.si.cmd(f"cd {str(session_dir)}")

        return calibrated_results

    def _stack_monochrome_filters(self, all_calibrated: dict,
                                 base_dir: Path, clear_intermediate: bool):
        """
        Stack each filter's calibrated lights separately.
        all_calibrated: {filter_name: [list of calibrated files]}
        """
        for filter_name in sorted(all_calibrated.keys(),
                                 key=lambda x: self.mono_filter_priority.get(x, 99)):
            print(f"\n=== Stacking filter {filter_name} ===")

            filter_calibrated_dir = base_dir / "calibrated" / filter_name
            self.si.cmd(f"cd {str(filter_calibrated_dir)}")

            pp_light_files = list(filter_calibrated_dir.glob("pp_light_*.fit*"))
            print(f"DEBUG: Found {len(pp_light_files)} calibrated file(s) for {filter_name}")
            if len(pp_light_files) > 0:
                print(f"DEBUG: Files: {[f.name for f in pp_light_files]}")

            if len(pp_light_files) == 0:
                print(f"Warning: No calibrated files found for {filter_name}, skipping")
                continue

            if len(pp_light_files) == 1:
                print(f"Only 1 file for {filter_name}, skipping registration")
                seq_to_stack = "pp_light"
            else:
                print(f"Found {len(pp_light_files)} files for {filter_name}, attempting registration")

                try:
                    self.si.cmd("close")
                except:
                    pass  # Ignore if nothing is loaded

                old_reg_files = list(filter_calibrated_dir.glob("r_pp_light_*.fit*"))
                if old_reg_files:
                    print(f"Cleaning up {len(old_reg_files)} old registration files")
                    for f in old_reg_files:
                        try:
                            f.unlink()
                        except Exception as e:
                            print(f"Warning: Could not delete {f.name}: {e}")

                old_seq = filter_calibrated_dir / "r_pp_light_.seq"
                if old_seq.exists():
                    try:
                        old_seq.unlink()
                        print(f"Deleted old sequence file: {old_seq.name}")
                    except Exception as e:
                        print(f"Warning: Could not delete {old_seq.name}: {e}")

                try:
                    print(f"DEBUG: Current directory: {filter_calibrated_dir}")
                    print(f"DEBUG: Attempting: register pp_light")
                    self.si.cmd("register pp_light")
                    seq_to_stack = "r_pp_light"
                    print(f"DEBUG: Registration successful for {filter_name}")
                except Exception as e:
                    print(f"ERROR: Registration failed for {filter_name}: {e}")
                    print(f"Skipping {filter_name} filter - cannot stack unregistered images")
                    print(f"Please check the calibrated images in {filter_calibrated_dir}")
                    print(f"Possibilities: images have different framing, insufficient stars, or field rotation")
                    continue

            # Stack
            try:
                stack_cmd = self._build_stack_cmd(
                    sequence=seq_to_stack,
                    cfg_key="light_stack",
                    out_arg=f"{filter_name}_stacked",
                )
                self.si.cmd(stack_cmd)
            except Exception as e:
                print(f"ERROR: Stacking failed for {filter_name}: {e}")
                print(f"Skipping {filter_name} filter")
                continue

            if clear_intermediate:
                if len(pp_light_files) > 1 and seq_to_stack == "r_pp_light":
                    for f in sorted(filter_calibrated_dir.glob("r_pp_light_*.fit*")):
                        print(f"Deleting {f.name} after stacking {filter_name}")
                        f.unlink()

                    r_seq = filter_calibrated_dir / "r_pp_light_.seq"
                    if r_seq.exists():
                        r_seq.unlink()

                    r_conv = filter_calibrated_dir / "r_pp_light_conversion.txt"
                    if r_conv.exists():
                        r_conv.unlink()

                # Also clean up pp_light intermediates if not needed anymore
                pp_seq = filter_calibrated_dir / "pp_light_.seq"
                if pp_seq.exists():
                    pp_seq.unlink()

                pp_conv = filter_calibrated_dir / "pp_light_conversion.txt"
                if pp_conv.exists():
                    pp_conv.unlink()

            stacked_file = filter_calibrated_dir / f"{filter_name}_stacked.fit"
            if not stacked_file.exists():
                stacked_file = filter_calibrated_dir / f"{filter_name}_stacked.fits"

            if stacked_file.exists():
                self.si.cmd(f"load {stacked_file}")
                self.si.cmd("mirrorx -bottomup")

                livetime = 0
                try:
                    from astropy.io import fits
                    with fits.open(stacked_file) as hdul:
                        header = hdul[0].header
                        livetime = int(header.get('LIVETIME', 0))
                except ImportError:
                    print("astropy not available, will save without livetime in filename")
                except Exception as e:
                    print(f"Could not read LIVETIME from header: {e}")

                # Determine output filename
                if livetime and livetime > 0:
                    output_filename = f"{filter_name}_stacked_{livetime}s"
                else:
                    output_filename = f"{filter_name}_stacked"

                self.si.cmd(f"cd {str(base_dir)}")

                # Save directly to base directory from Siril (avoids file locking issues)
                self.si.cmd(f"save {output_filename} -ext=fit")

                print(f"Saved final stack as {output_filename}.fit in base directory")

                if clear_intermediate:
                    if stacked_file.exists():
                        print(f"Deleting {stacked_file.name} after creating final result")
                        stacked_file.unlink()
            else:
                print(f"Warning: Stacked file not found for {filter_name}")

    def _set_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#3c3c3c"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("white"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#2a2a2a"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#3c3c3c"))
        palette.setColor(QPalette.ColorRole.Text, QColor("white"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#2e2e2e"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("white"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#505050"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("white"))
        
        palette.setColor(QPalette.ColorRole.Light, QColor("#3c3c3c"))
        palette.setColor(QPalette.ColorRole.Midlight, QColor("#3c3c3c"))
        palette.setColor(QPalette.ColorRole.Dark, QColor("#1f1f1f"))
        palette.setColor(QPalette.ColorRole.Mid, QColor("#2a2a2a"))
        palette.setColor(QPalette.ColorRole.Shadow, QColor("#000000"))

        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#2a2a2a"))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor("white"))
        QApplication.setPalette(palette)

    def _on_item_clicked(self, item, column):
        if column == 1:  
            self._delete_item(item)

    def _delete_item(self, item):
        parent = item.parent()
        iid = id(item)

        if iid in self.sessions and self.sessions[iid].get('is_master'):
            QMessageBox.information(
                self,
                "Master File",
                "Master calibration files are managed automatically. \n"
                "To remove a master file, delete or rename the file from the main directory, "
                "then re-add your sessions."
            )
            return

        # Traverse up to find the session item
        session_item = item
        while session_item.parent():
            potential_session = session_item
            potential_iid = id(potential_session)
            if potential_iid in self.sessions and not self.sessions[potential_iid].get('is_parent'):
                break
            session_item = session_item.parent()

        session_iid = id(session_item)

        if not parent:
            # Deleting a top-level session
            reply = QMessageBox.question(self, "Delete Session", "This will remove the session and all associated file mentions from the list. Continue?")
            if reply == QMessageBox.StandardButton.Yes:
                del self.sessions[session_iid]
                self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(item))
                self.current_session = None
        else:
            # Deleting a sub-item (filter or frame type)
            sess = self.sessions[session_iid]
            is_mono = sess.get('is_mono', False)

            if is_mono:
                item_text = item.text(0)
                frame_type = item_text.split()[0].lower()  # "bias", "dark", "lights", "flats", or filter name

                if frame_type in ['bias', 'dark']:
                    # Remove from session data
                    sess[frame_type] = None
                    parent.takeChild(parent.indexOfChild(item))
                elif frame_type in self.mono_filters:
                    # Deleting an entire filter
                    if frame_type in sess['filters']:
                        del sess['filters'][frame_type]
                    # Also remove the frame type entries
                    sess.pop(f'{frame_type}_lights', None)
                    sess.pop(f'{frame_type}_flats', None)
                    parent.takeChild(parent.indexOfChild(item))
                else:
                    # Deleting a frame type under a filter (e.g., "lights" under "L")
                    filter_name = parent.text(0)  # e.g., "L"
                    key = f'{filter_name}_{frame_type}'
                    sess.pop(key, None)
                    # Also update the filters dict
                    if filter_name in sess['filters']:
                        if frame_type == 'lights':
                            sess['filters'][filter_name]['lights'] = []
                            sess['filters'][filter_name]['invalid'] = True
                            parent.setForeground(0, QColor("red"))
                        elif frame_type == 'flats':
                            sess['filters'][filter_name]['flats'] = []
                    parent.takeChild(parent.indexOfChild(item))
            else:
                frame_type = item.text(0).split()[0].lower()
                self.sessions[session_iid][frame_type] = None
                parent.takeChild(parent.indexOfChild(item))

                if frame_type == 'light':
                    self.sessions[session_iid]['invalid'] = True
                    session_item.setForeground(0, QColor("red"))

            self._update_frame_buttons()

    def _on_item_double_clicked(self, item, column):
        if column != 0:
            return

        parent = item.parent()
        folder = None

        # Traverse up to find the session item
        session_item = item
        while session_item.parent():
            session_item = session_item.parent()

        session_iid = id(session_item)
        sess = self.sessions[session_iid]
        is_mono = sess.get('is_mono', False)

        item_iid = id(item)
        if item_iid in self.sessions and self.sessions[item_iid].get('is_master'):
            master_path = Path(self.sessions[item_iid].get('path', ''))
            if master_path.exists():
                system = platform.system()
                if system == 'Windows':
                    os.startfile(str(master_path.parent))
                elif system == 'Darwin':
                    subprocess.call(['open', str(master_path.parent)])
                elif system == 'Linux':
                    subprocess.call(['xdg-open', str(master_path.parent)])
            return

        if not parent:
            # Session level item - open session folder
            folder = sess.get('folder')
        elif is_mono and not parent.parent():
            item_text = item.text(0).split()[0].lower()  # "bias", "dark", or filter name

            if item_text in ['bias', 'dark']:
                # Bias or dark folder
                folder = sess.get(item_text, {}).get('folder')
            else:
                filter_name = item.text(0)
                filter_dir = sess['folder'] / filter_name
                if filter_dir.exists():
                    folder = filter_dir
        elif parent.parent():
            if is_mono:
                filter_name = parent.text(0)  # e.g., "L"
                frame_type = item.text(0).split()[0].lower()  # "lights" or "flats"
                key = f'{filter_name}_{frame_type}'
                folder = sess.get(key, {}).get('folder')
            else:
                frame_type = item.text(0).split()[0].lower()
                folder = sess.get(frame_type, {}).get('folder')
        else:
            frame_type = item.text(0).split()[0].lower()
            folder = sess.get(frame_type, {}).get('folder')

        if folder and folder.exists():
            system = platform.system()
            if system == 'Windows':
                os.startfile(str(folder))
            elif system == 'Darwin':
                subprocess.call(['open', str(folder)])
            elif system == 'Linux':
                subprocess.call(['xdg-open', str(folder)])

    def _on_selection_changed(self):
        selected = self.tree.selectedItems()
        if not selected:
            self.current_session = None
            self._update_frame_buttons()
            return

        item = selected[0]

        # Traverse up the tree to find the top-level session item
        # This handles both OSC (session -> frame) and monochrome (session -> filter -> frame)
        session_item = item
        while session_item.parent():
            session_item = session_item.parent()

        self.current_session = id(session_item)
        self._update_frame_buttons()

    def _update_frame_buttons(self):
        btns = {
            'light': self.light_btn,
            'dark': self.dark_btn,
            'flat': self.flat_btn,
            'bias': self.bias_btn
        }

        if not self.current_session:
            for btn in btns.values():
                btn.setEnabled(False)
            return

        sess = self.sessions[self.current_session]
        if sess.get('is_master'):
            for btn in btns.values():
                btn.setEnabled(False)
            return

        sess = self.sessions[self.current_session]
        is_mono = sess.get('is_mono', False)

        if is_mono:
            selected = self.tree.selectedItems()
            if not selected:
                for btn in btns.values():
                    btn.setEnabled(False)
                return

            item = selected[0]
            item_text = item.text(0).split()[0]  # First word only (bias, dark, L, R, etc.)

            is_filter_selected = item_text in self.mono_filters

            # Light+ and Flat+ only enabled when a filter is selected
            self.light_btn.setEnabled(is_filter_selected)
            self.flat_btn.setEnabled(is_filter_selected)

            # Dark+ and Bias+ enabled if they don't exist yet (shared across filters)
            self.dark_btn.setEnabled(sess.get('dark') is None)
            self.bias_btn.setEnabled(sess.get('bias') is None)
        else:
            for ft, btn in btns.items():
                btn.setEnabled(self.sessions[self.current_session][ft] is None)

    def _add_session(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Session Folder")
        if not folder:
            return

        folder_path = Path(folder)

        if ' ' in str(folder_path):
            QMessageBox.critical(self, "Error", "The path must not contain spaces.")
            return

        # Branch based on mode
        if self.is_mono_mode:
            success = self._add_mono_session(folder_path)
            if not success:
                QMessageBox.critical(
                    self,
                    "Error",
                    "No valid monochrome session structure found. "
                    "Expected filter directories (L, R, G, B, Ha, Oiii, Sii) "
                    "with lights/ and optionally flats/ subdirectories."
                )
        else:
            self._add_osc_session(folder_path)

    def _add_osc_session(self, folder_path: Path):
        """Original OSC session detection logic"""
        potential_sessions = []

        potential_sessions.append(folder_path)

        sub_sessions = []
        for sub in sorted(folder_path.iterdir()):
            if not sub.is_dir():
                continue

            is_session = any(
                d.is_dir() and d.name.lower() in self.all_frame_names
                for d in sub.iterdir()
            )
            if is_session:
                sub_sessions.append(sub)

        # At minimum, the selected folder itself is in potential_sessions
        has_frame_data = any(
            d.is_dir() and d.name.lower() in self.all_frame_names
            for d in folder_path.iterdir()
        )
        if not has_frame_data and not sub_sessions:
            QMessageBox.critical(
                self,
                "Error",
                "No valid session folders found in the selected directory."
            )
            return

        if self.sessions_parent is None:
            self.sessions_parent = folder_path
            # storage when _detect_and_auto_select_masters is called after item creation

        existing_names = [
            self.tree.topLevelItem(i).text(0)
            for i in range(self.tree.topLevelItemCount())
        ]

        has_sub_sessions = len(sub_sessions) > 0

        for sess_path in potential_sessions:
            if sess_path.name in existing_names:
                continue

            item = QTreeWidgetItem([sess_path.name, '❌'])

            if has_sub_sessions and sess_path == folder_path:
                self.tree.addTopLevelItem(item)
                iid = id(item)

                # Store minimal data for parent folder
                self.sessions[iid] = {
                    'folder': sess_path,
                    'is_mono': False,
                    'bias': None,
                    'dark': None,
                    'flat': None,
                    'light': None,
                    'is_parent': True,  # Mark as parent folder
                    'external_masters': {},  # Store session-local master paths
                }

                for ft in self.frame_order:
                    subdir = next(
                        (
                            d for d in folder_path.iterdir()
                            if d.is_dir() and d.name.lower() in self.frame_names[ft]
                        ),
                        None,
                    )

                    if subdir:
                        files = [
                            f for f in subdir.rglob('*')
                            if f.suffix.lower() in self.image_extensions
                        ]
                        if files:
                            count = len(files)
                            child = QTreeWidgetItem(
                                [f'{ft} ({count} images)', '❌']
                            )
                            item.addChild(child)
                            self.sessions[iid][ft] = {
                                'files': files,
                                'count': count,
                                'folder': subdir,
                            }

                # Add sub-sessions as children
                for sub_path in sub_sessions:
                    sub_item = QTreeWidgetItem([sub_path.name, '❌'])
                    item.addChild(sub_item)
                    sub_iid = id(sub_item)

                    # Store session data for sub-session
                    self.sessions[sub_iid] = {
                        'folder': sub_path,
                        'is_mono': False,
                        'bias': None,
                        'dark': None,
                        'flat': None,
                        'light': None,
                        'parent_iid': iid,  # Reference to parent
                    }

                    for ft in self.frame_order:
                        subdir = next(
                            (
                                d for d in sub_path.iterdir()
                                if d.is_dir() and d.name.lower() in self.frame_names[ft]
                            ),
                            None,
                        )

                        if subdir:
                            files = [
                                f for f in subdir.rglob('*')
                                if f.suffix.lower() in self.image_extensions
                            ]
                            if files:
                                count = len(files)
                                child = QTreeWidgetItem(
                                    [f'{ft} ({count} images)', '❌']
                                )
                                sub_item.addChild(child)
                                self.sessions[sub_iid][ft] = {
                                    'files': files,
                                    'count': count,
                                    'folder': subdir,
                                }

                    sub_item.setExpanded(True)

                self._detect_and_auto_select_masters(folder_path, parent_item=item, parent_iid=iid)
            else:
                # Regular session (standalone or when parent has its own frame data)
                self.tree.addTopLevelItem(item)
                iid = id(item)

                self.sessions[iid] = {
                    'folder': sess_path,
                    'is_mono': False,
                    'bias': None,
                    'dark': None,
                    'flat': None,
                    'light': None,
                    'external_masters': {},  # Store session-local master paths
                }

                for ft in self.frame_order:
                    subdir = next(
                        (
                            d for d in sess_path.iterdir()
                            if d.is_dir() and d.name.lower() in self.frame_names[ft]
                        ),
                        None,
                    )

                    if subdir:
                        files = [
                            f for f in subdir.rglob('*')
                            if f.suffix.lower() in self.image_extensions
                        ]
                        if files:
                            count = len(files)
                            child = QTreeWidgetItem(
                                [f'{ft} ({count} images)', '❌']
                            )
                            item.addChild(child)
                            self.sessions[iid][ft] = {
                                'files': files,
                                'count': count,
                                'folder': subdir,
                            }

            item.setExpanded(True)

        if potential_sessions:
            last_item = self.tree.topLevelItem(
                self.tree.topLevelItemCount() - 1
            )
            self.tree.setCurrentItem(last_item)

        self._update_frame_buttons()

    def _on_light_btn_clicked(self):
        """Handle Light+ button click - show filter menu for mono, open dialog for OSC"""
        if not self.current_session:
            return

        sess = self.sessions[self.current_session]
        is_mono = sess.get('is_mono', False)

        if is_mono:
            self._show_filter_menu('lights')
        else:
            # Direct file dialog for OSC
            self._add_frames('light')

    def _on_flat_btn_clicked(self):
        """Handle Flat+ button click - show filter menu for mono, open dialog for OSC"""
        if not self.current_session:
            return

        sess = self.sessions[self.current_session]
        is_mono = sess.get('is_mono', False)

        if is_mono:
            self._show_filter_menu('flats')
        else:
            # Direct file dialog for OSC
            self._add_frames('flat')

    def _on_dark_btn_clicked(self):
        """Handle Dark+ button click - open dialog for both modes"""
        if not self.current_session:
            return
        self._add_frames('dark')

    def _on_bias_btn_clicked(self):
        """Handle Bias+ button click - open dialog for both modes"""
        if not self.current_session:
            return
        self._add_frames('bias')

    def _show_filter_menu(self, frame_type):
        """Show dropdown menu with available filters for monochrome mode"""
        if not self.current_session:
            return

        sess = self.sessions[self.current_session]
        filters = sess.get('filters', {})

        if not filters:
            QMessageBox.warning(self, 'No Filters', 'No filters found in this session.')
            return

        # Create menu
        menu = QMenu(self)

        sender = self.sender()
        if sender == self.light_btn:
            btn = self.light_btn
        elif sender == self.flat_btn:
            btn = self.flat_btn
        else:
            btn = sender

        for filter_name in sorted(filters.keys(), key=lambda x: self.mono_filter_priority.get(x, 99)):
            action = menu.addAction(f"{filter_name} {frame_type}")

            action.triggered.connect(lambda checked, f=filter_name: self._add_mono_frames(f, frame_type))

        # Position at bottom-left corner of the button in global coordinates
        global_pos = btn.mapToGlobal(QPoint(0, btn.height()))
        menu.exec(global_pos)

    def _add_mono_frames(self, filter_name: str, frame_type: str):
        """Add frames for a specific filter in monochrome mode"""
        if not self.current_session:
            return

        current_item = self.tree.currentItem()
        session_item = current_item
        while session_item.parent():
            potential_session = session_item
            potential_iid = id(potential_session)
            if potential_iid in self.sessions and not self.sessions[potential_iid].get('is_parent'):
                break
            session_item = session_item.parent()

        session_iid = id(session_item)
        sess = self.sessions[session_iid]
        is_mono = sess.get('is_mono', False)

        if not is_mono:
            # Shouldn't happen, but fallback to regular add
            self._add_frames(frame_type)
            return

        filter_key = f'{filter_name}_{frame_type}'
        current_data = sess.get(filter_key, {})

        # Open file dialog
        frame_display = frame_type.capitalize()
        folder = QFileDialog.getExistingDirectory(self, f'Select {filter_name} {frame_display} Folder')
        if not folder:
            return

        folder_path = Path(folder)
        if ' ' in str(folder_path):
            QMessageBox.critical(self, "Error", "The path must not contain spaces.")
            return

        # Find files
        files = [f for f in folder_path.rglob('*') if f.suffix.lower() in self.image_extensions]
        if not files:
            QMessageBox.critical(self, 'Error', f'No image files found in the selected folder.')
            return

        count = len(files)

        # Update session data
        sess[filter_key] = {
            'files': files,
            'count': count,
            'folder': folder_path,
            'filter_name': filter_name,
            'frame_type': frame_type
        }

        if frame_type == 'lights' and 'filters' in sess and filter_name in sess['filters']:
            if sess['filters'][filter_name].get('invalid'):
                sess['filters'][filter_name]['invalid'] = False
                for i in range(session_item.childCount()):
                    child = session_item.child(i)
                    if child.text(0) == filter_name:
                        child.setForeground(0, QColor("white"))
                        break

        if 'filters' in sess and filter_name in sess['filters']:
            sess['filters'][filter_name][f'{frame_type}_dir'] = folder_path
            if frame_type == 'lights':
                sess['filters'][filter_name]['lights'] = files
            elif frame_type == 'flats':
                sess['filters'][filter_name]['flats'] = files

        filter_item = None
        for i in range(session_item.childCount()):
            child = session_item.child(i)
            if child.text(0) == filter_name:
                filter_item = child
                break

        if filter_item:
            for i in range(filter_item.childCount()):
                child = filter_item.child(i)
                if child.text(0).startswith(frame_type):
                    child.setText(0, f'{frame_type} ({count} images)')
                    break

            filter_item.setExpanded(True)
        session_item.setExpanded(True)


    def _add_frames(self, frame_type):
        if not self.current_session:
            return

        sess = self.sessions[self.current_session]
        is_mono = sess.get('is_mono', False)

        folder = QFileDialog.getExistingDirectory(self, f'Select {frame_type.capitalize()} Folder')
        if not folder:
            return

        folder_path = Path(folder)
        if ' ' in str(folder_path):
            QMessageBox.critical(self, "Error", "The path must not contain spaces.")
            return

        files = [f for f in folder_path.rglob('*') if f.suffix.lower() in self.image_extensions]
        if not files:
            QMessageBox.critical(self, 'Error', f'No image files found in the selected folder for {frame_type}.')
            return

        count = len(files)

        if is_mono and frame_type in ['bias', 'dark']:
            # Traverse up to find the session item (stop at first non-parent session)
            current_item = self.tree.currentItem()
            session_item = current_item
            while session_item.parent():
                potential_session = session_item
                potential_iid = id(potential_session)
                if potential_iid in self.sessions and not self.sessions[potential_iid].get('is_parent'):
                    break
                session_item = session_item.parent()

            session_iid = id(session_item)

            existing_child = None
            for i in range(session_item.childCount()):
                child = session_item.child(i)
                child_text = child.text(0).split()[0].lower()
                if child_text == frame_type:
                    existing_child = child
                    break

            if existing_child:
                existing_child.setText(0, f'{frame_type} ({count} images)')
            else:
                new_child = QTreeWidgetItem([f'{frame_type} ({count} images)', '❌'])
                # Insert after any existing bias/dark, before filters
                insert_pos = 0
                for i in range(session_item.childCount()):
                    child = session_item.child(i)
                    child_text = child.text(0).split()[0].lower()
                    if child_text in ['bias', 'dark']:
                        insert_pos = i + 1
                    else:
                        break
                session_item.insertChild(insert_pos, new_child)

            self.sessions[session_iid][frame_type] = {
                'files': files,
                'count': count,
                'folder': folder_path
            }
            session_item.setExpanded(True)
        else:
            current_item = self.tree.currentItem()

            if is_mono and current_item.parent():
                parent = current_item.parent()
            else:
                parent = current_item
                while parent.parent():
                    potential_session = parent
                    potential_iid = id(potential_session)
                    if potential_iid in self.sessions and not self.sessions[potential_iid].get('is_parent'):
                        break
                    parent = parent.parent()

            parent_iid = id(parent)

            existing_child = None
            for i in range(parent.childCount()):
                child = parent.child(i)
                if child.text(0).startswith(frame_type):
                    existing_child = child
                    break

            if existing_child:
                existing_child.setText(0, f'{frame_type} ({count} images)')
            else:
                existing_child = QTreeWidgetItem([f'{frame_type} ({count} images)', '❌'])
                parent.addChild(existing_child)

            self.sessions[parent_iid][frame_type] = {'files': files, 'count': count, 'folder': folder_path}

            if frame_type == 'light' and self.sessions[parent_iid].get('invalid'):
                self.sessions[parent_iid]['invalid'] = False
                parent.setForeground(0, QColor("white"))

            children = [parent.child(i) for i in range(parent.childCount())]
            sorted_children = sorted(children, key=lambda c: self.frame_order.index(c.text(0).split()[0].lower()))
            for i, child in enumerate(sorted_children):
                parent.takeChild(parent.indexOfChild(child))
                parent.insertChild(i, child)

            parent.setExpanded(True)

        self._update_frame_buttons()

    def _detect_and_auto_select_masters(self, base_dir: Path, parent_item: QTreeWidgetItem = None, parent_iid: int = None):
        """
        Detect master calibration files in the base directory and auto-select them in settings.
        Looks for: bias_stacked.fit, dark_stacked.fit, masterBias.fit, masterDark.fit
        Also renders them in the tree list as children of the parent folder item.

        Args:
            base_dir: The base directory to scan for master files
            parent_item: The parent tree item to add master children to (optional)
            parent_iid: The id of the parent item for session storage (optional)
        """
        if not base_dir or not base_dir.exists():
            return {}

        master_patterns = {
            'bias_stacked.fit': ('bias_stack', 'bias', 'masterBias'),
            'bias_stacked.fits': ('bias_stack', 'bias', 'masterBias'),
            'masterBias.fit': ('bias_stack', 'bias', 'masterBias'),
            'masterBias.fits': ('bias_stack', 'bias', 'masterBias'),
            'dark_stacked.fit': ('dark_stack', 'dark', 'masterDark'),
            'dark_stacked.fits': ('dark_stack', 'dark', 'masterDark'),
            'masterDark.fit': ('dark_stack', 'dark', 'masterDark'),
            'masterDark.fits': ('dark_stack', 'dark', 'masterDark'),
        }

        found_masters = {}
        for filename, (settings_key, frame_type, display_name) in master_patterns.items():
            master_file = base_dir / filename
            if master_file.exists():
                if settings_key not in found_masters:
                    found_masters[settings_key] = {
                        'path': str(master_file),
                        'display_name': display_name,
                        'frame_type': frame_type
                    }

        for settings_key, master_info in found_masters.items():
            master_path = master_info['path']
            display_name = master_info['display_name']
            frame_type = master_info['frame_type']

            if parent_iid is not None and parent_iid in self.sessions:
                self.sessions[parent_iid]['external_masters'][frame_type] = master_path

            if parent_item and parent_iid is not None:
                existing_master = None
                for i in range(parent_item.childCount()):
                    child = parent_item.child(i)
                    text = child.text(0)
                    if text.startswith(display_name) or (text.startswith(frame_type) and 'master' in text.lower()):
                        existing_master = child
                        break

                if existing_master:
                    existing_master.setText(0, f'{display_name} (master)')
                    existing_master.setText(1, '✓')
                else:
                    master_item = QTreeWidgetItem([f'{display_name} (master)', '✓'])
                    parent_item.insertChild(0, master_item)
                    master_iid = id(master_item)
                    self.sessions[master_iid] = {
                        'is_master': True,
                        'type': frame_type,
                        'path': master_path,
                        'parent_iid': parent_iid
                    }


        if found_masters and parent_iid is not None:
            frame_names = []
            if 'bias_stack' in found_masters:
                frame_names.append('master bias')
            if 'dark_stack' in found_masters:
                frame_names.append('master dark')

        return found_masters

    def _check_calibration_consistency(self):
        cal_types = ['bias', 'dark', 'flat']
        session_cal_sets = [
            {ct for ct in cal_types if sess.get(ct)}
            for sess in self.sessions.values()
            if not sess.get('is_parent') and not sess.get('is_master')
        ]
        if not session_cal_sets:
            return True
        return len(set(frozenset(s) for s in session_cal_sets)) <= 1

    def _run_multisession(self):
        if not self.sessions:
            QMessageBox.warning(self, 'No sessions', 'Please add sessions first.')
            return

        if self.sessions_parent is None:
            QMessageBox.critical(self, 'Error', 'No sessions parent directory set.')
            return

        first_session = next(iter(self.sessions.values()))
        first_is_mono = first_session.get('is_mono', False)

        if first_is_mono != self.is_mono_mode:
            mode_name = "monochrome" if first_is_mono else "OSC (Color)"
            current_mode = "monochrome" if self.is_mono_mode else "OSC (Color)"
            QMessageBox.critical(
                self,
                'Error',
                f'Mode mismatch: Sessions were added in {mode_name} mode '
                f'but {current_mode} mode is currently selected.\n\n'
                f'Please switch to the correct mode or clear sessions and re-add them.'
            )
            return

        if self.is_mono_mode:
            self._run_monochrome_workflow()
        else:
            self._run_osc_workflow()

    def _run_osc_workflow(self):
        """Original OSC processing workflow"""
        if not self._check_calibration_consistency():
            reply = QMessageBox.question(
                self,
                "Warning",
                "Different sessions have different types of calibration frames. "
                "This may degrade the quality of the result. Continue anyway?"
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        gen_cfg = self.settings.get("general", {})
        clear_intermediate = gen_cfg.get("clear_intermediate_files", True)

        self.base_dir = self.sessions_parent
        self.process_dir = self.base_dir / 'process'
        self.masters_dir = self.base_dir / 'masters'

        self.si.cmd(f"cd {str(self.base_dir)}")

        calibrated_dir = self.base_dir / "calibrated"
        calibrated_dir.mkdir(exist_ok=True)

        self.observers = []

        if clear_intermediate:
            cal_map = {
                r'^r_pp_light_\d{5}\.fit(s)?$': ('pp_light', 'individual'),
                r'^r_pp_light_\.seq$': ('pp_light', 'seq'),
            }

            cal_handler = CleaningHandler(self.si, calibrated_dir, calibrated_dir, cal_map)
            cal_observer = Observer()
            cal_observer.schedule(cal_handler, str(calibrated_dir), recursive=False)
            self.observers.append(cal_observer)
            cal_observer.start()

        sessions_to_process = []
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            iid = id(item)
            sess = self.sessions[iid]

            if sess.get('is_parent'):
                parent_ext_masters = sess.get('external_masters', {})
                for j in range(item.childCount()):
                    child_item = item.child(j)
                    child_iid = id(child_item)
                    if child_iid in self.sessions:
                        child_sess = self.sessions[child_iid]
                        if child_sess.get('is_master'):
                            continue
                        if 'external_masters' not in child_sess or not child_sess['external_masters']:
                            child_sess['external_masters'] = parent_ext_masters.copy()
                        sessions_to_process.append((child_item, child_iid, child_sess))
            else:
                if sess.get('is_master'):
                    continue
                sessions_to_process.append((item, iid, sess))

        for item, iid, sess in sessions_to_process:

            session_dir = sess["folder"]
            process_dir_path = session_dir / "process"
            masters_dir_path = session_dir / "masters"

            process_dir_path.mkdir(exist_ok=True)
            masters_dir_path.mkdir(exist_ok=True)

            if clear_intermediate:
                masters_map = {
                    r'^bias_stacked\.fit(s)?$': ('bias', 'group'),
                    r'^dark_stacked\.fit(s)?$': ('dark', 'group'),
                    r'^flat_stacked\.fit(s)?$': ('flat', 'group'),
                    r'^pp_flat_stacked\.fit(s)?$': ('pp_flat', 'group'),
                }

                masters_handler = CleaningHandler(
                    self.si, masters_dir_path, process_dir_path, masters_map
                )
                masters_observer = Observer()
                masters_observer.schedule(
                    masters_handler, str(masters_dir_path), recursive=False
                )
                self.observers.append(masters_observer)
                masters_observer.start()

                process_map = {
                    r'^pp_flat_\d{5}\.fit(s)?$': ('flat', 'individual'),
                    r'^pp_light_\d{5}\.fit(s)?$': ('light', 'individual'),
                }

                process_handler = CleaningHandler(
                    self.si, process_dir_path, process_dir_path, process_map
                )
                process_observer = Observer()
                process_observer.schedule(
                    process_handler, str(process_dir_path), recursive=False
                )
                self.observers.append(process_observer)
                process_observer.start()

            self.si.cmd(f"cd {str(session_dir)}")

            bias_master_available = False
            dark_master_available = False

            session_ext_masters = sess.get('external_masters', {})

            bias_external_path = None
            if 'bias' in session_ext_masters:
                bias_external_path = session_ext_masters['bias']
            else:
                bias_cfg = self.settings.get("bias_stack", {})
                if bias_cfg.get("use_external_master") and bias_cfg.get("external_master_path"):
                    bias_external_path = bias_cfg["external_master_path"]

            use_bias_ext = bias_external_path is not None

            if sess["bias"]:
                # Session has bias frames - stack them
                if use_bias_ext:
                    src = Path(bias_external_path)
                    if src.is_file():
                        dest = masters_dir_path / f"bias_stacked{src.suffix}"
                        shutil.copy2(src, dest)
                        print(f"Using external master bias: {src} -> {dest}")
                        bias_master_available = True
                    else:
                        print(f"External master bias not found: {src}, falling back to stacking")
                        use_bias_ext = False

                if not use_bias_ext:
                    bias_dir = sess["bias"]["folder"]
                    self.si.cmd(f"cd {str(bias_dir)}")
                    self.si.cmd("convert bias -out=../process")
                    self.si.cmd("cd ../process")

                    bias_stack_cmd = self._build_stack_cmd(
                        sequence="bias",
                        cfg_key="bias_stack",
                        out_arg="../masters/bias_stacked",
                    )
                    self.si.cmd(bias_stack_cmd)
                    self.si.cmd("cd ..")
                    bias_master_available = True
            elif use_bias_ext:
                # Session has NO bias frames, but external master is configured
                src = Path(bias_external_path)
                if src.is_file():
                    dest = masters_dir_path / f"bias_stacked{src.suffix}"
                    shutil.copy2(src, dest)
                    print(f"Using external master bias (no session bias): {src} -> {dest}")
                    bias_master_available = True
                else:
                    print(f"External master bias not found: {src}")

            dark_external_path = None
            if 'dark' in session_ext_masters:
                dark_external_path = session_ext_masters['dark']
            else:
                dark_cfg = self.settings.get("dark_stack", {})
                if dark_cfg.get("use_external_master") and dark_cfg.get("external_master_path"):
                    dark_external_path = dark_cfg["external_master_path"]

            use_dark_ext = dark_external_path is not None

            if sess["dark"]:
                # Session has dark frames - stack them
                if use_dark_ext:
                    src = Path(dark_external_path)
                    if src.is_file():
                        dest = masters_dir_path / f"dark_stacked{src.suffix}"
                        shutil.copy2(src, dest)
                        print(f"Using external master dark: {src} -> {dest}")
                        dark_master_available = True
                    else:
                        print(f"External master dark not found: {src}, falling back to stacking")
                        use_dark_ext = False

                if not use_dark_ext:
                    dark_dir = sess["dark"]["folder"]
                    self.si.cmd(f"cd {str(dark_dir)}")
                    self.si.cmd("convert dark -out=../process")
                    self.si.cmd("cd ../process")

                    dark_cmd = self._build_stack_cmd(
                        sequence="dark",
                        cfg_key="dark_stack",
                        out_arg="../masters/dark_stacked",
                    )
                    self.si.cmd(dark_cmd)
                    self.si.cmd("cd ..")
                    dark_master_available = True
            elif use_dark_ext:
                # Session has NO dark frames, but external master is configured
                src = Path(dark_external_path)
                if src.is_file():
                    dest = masters_dir_path / f"dark_stacked{src.suffix}"
                    shutil.copy2(src, dest)
                    print(f"Using external master dark (no session dark): {src} -> {dest}")
                    dark_master_available = True
                else:
                    print(f"External master dark not found: {src}")

            if sess["flat"]:
                flat_dir = sess["flat"]["folder"]
                self.si.cmd(f"cd {str(flat_dir)}")
                self.si.cmd("convert flat -out=../process")
                self.si.cmd("cd ../process")

                if bias_master_available:
                    self.si.cmd("calibrate flat -bias=../masters/bias_stacked")
                    pp_flat_cmd = self._build_stack_cmd(
                        sequence="pp_flat",
                        cfg_key="flat_stack",
                        out_arg="../masters/pp_flat_stacked",
                    )
                    self.si.cmd(pp_flat_cmd)
                else:
                    flat_cmd = self._build_stack_cmd(
                        sequence="flat",
                        cfg_key="flat_stack",
                        out_arg="../masters/flat_stacked",
                    )
                    self.si.cmd(flat_cmd)

                self.si.cmd("cd ..")

            if sess["light"]:
                light_dir = sess["light"]["folder"]
                self.si.cmd(f"cd {str(light_dir)}")
                self.si.cmd("convert light -out=../process")
                self.si.cmd("cd ../process")

                args: list[str] = []

                if dark_master_available:
                    args.append("-dark=../masters/dark_stacked")
                if sess["flat"]:
                    if bias_master_available:
                        args.append("-flat=../masters/pp_flat_stacked")
                    else:
                        args.append("-flat=../masters/flat_stacked")

                args.extend(self._build_light_calibrate_args())
                self.si.cmd("calibrate light " + " ".join(args))

                # Move pp_light* to calibrated with unique names
                existing = sorted(calibrated_dir.glob("pp_light_*.fit*"))
                offset = len(existing)

                for idx, f in enumerate(sorted(process_dir_path.glob("pp_light_*.fit*")), start=1):
                    new_idx = offset + idx
                    ext = f.suffix
                    new_name = f"pp_light_{new_idx:05d}{ext}"
                    shutil.move(str(f), str(calibrated_dir / new_name))

                # Delete pp_light_.seq and pp_light_conversion.txt if enabled
                if clear_intermediate:
                    pp_seq = process_dir_path / 'pp_light_.seq'
                    if pp_seq.exists():
                        print(f"Deleting {pp_seq.name} after moving pp_light files")
                        pp_seq.unlink()

                    pp_conv = process_dir_path / 'pp_light_conversion.txt'
                    if pp_conv.exists():
                        print(f"Deleting {pp_conv.name} after moving pp_light files")
                        pp_conv.unlink()

                self.si.cmd("cd ..")

        # Final stacking
        self.si.cmd(f"cd {str(calibrated_dir)}")
        self.si.cmd("register pp_light")

        final_stack_cmd = self._build_stack_cmd(
            sequence="r_pp_light",
            cfg_key="light_stack",
            out_arg="result",
        )
        self.si.cmd(final_stack_cmd)

        self.si.cmd("load result")
        self.si.cmd("mirrorx -bottomup")
        self.si.cmd("save ../result_$LIVETIME:%d$s")
        self.si.cmd("cd ..")

        # Delete r_pp_light intermediates only if enabled
        if clear_intermediate:
            for f in sorted(calibrated_dir.glob("r_pp_light_*.fit*")):
                print(f"Deleting {f.name} after final stacking")
                f.unlink()

            r_seq = calibrated_dir / "r_pp_light_.seq"
            if r_seq.exists():
                print(f"Deleting {r_seq.name} after final stacking")
                r_seq.unlink()

            r_conv = calibrated_dir / "r_pp_light_conversion.txt"
            if r_conv.exists():
                print(f"Deleting {r_conv.name} after final stacking")
                r_conv.unlink()

        for obs in self.observers:
            obs.stop()
            obs.join()
        self.observers = []

    def _run_monochrome_workflow(self):
        """Monochrome processing workflow - each filter processed separately"""
        gen_cfg = self.settings.get("general", {})
        clear_intermediate = gen_cfg.get("clear_intermediate_files", True)

        base_dir = self.sessions_parent
        process_dir = base_dir / 'process'
        masters_dir = base_dir / 'masters'

        process_dir.mkdir(exist_ok=True)
        masters_dir.mkdir(exist_ok=True)

        calibrated_base = base_dir / "calibrated"
        calibrated_base.mkdir(exist_ok=True)

        self.si.cmd(f"cd {str(base_dir)}")

        self.observers = []
        self._observer_dirs = set()  # Reset observer tracking

        all_calibrated = {}

        sessions_to_process = []
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            iid = id(item)
            sess = self.sessions[iid]

            if sess.get('is_parent'):
                parent_ext_masters = sess.get('external_masters', {})
                for j in range(item.childCount()):
                    child_item = item.child(j)
                    child_iid = id(child_item)
                    if child_iid in self.sessions:
                        child_sess = self.sessions[child_iid]
                        # Skip master items
                        if child_sess.get('is_master'):
                            continue
                        # Inherit parent's external_masters if child doesn't have its own
                        if 'external_masters' not in child_sess or not child_sess['external_masters']:
                            child_sess['external_masters'] = parent_ext_masters.copy()
                        sessions_to_process.append((child_item, child_iid, child_sess))
            else:
                if sess.get('is_master'):
                    continue
                sessions_to_process.append((item, iid, sess))

        # Process each session
        for item, iid, sess in sessions_to_process:
            results = self._run_monochrome_session(
                sess, iid, base_dir, calibrated_base, clear_intermediate
            )

            for filter_name, calib_dir in results:
                if filter_name not in all_calibrated:
                    all_calibrated[filter_name] = []

                all_calibrated[filter_name].extend(list(calib_dir.glob("pp_light_*.fit*")))

        # Stack each filter
        if all_calibrated:
            self._stack_monochrome_filters(all_calibrated, base_dir, clear_intermediate)

        # Stop all observers
        for obs in self.observers:
            obs.stop()
            obs.join()
        self.observers = []


    def closeEvent(self, event):
        self.si.disconnect()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    launcher = LauncherApp()
    launcher.show()
    sys.exit(app.exec())