"""
GPU Framework Manager - A GUI tool for managing ONNX, PyTorch, and JAX installations
Uses sirilpy.gpuhelper to detect hardware and install appropriate backends

(c) 2026 Adrian Knagg-Baugh
# SPDX-License-Identifier: GPL-3.0-or-later

Version History
===============
1.0.0 Initial release
1.0.1 Improve a string
1.0.2 Hide the Jax tab: Jax s not yet mature enough across all architectures to be
      used in scripts. It looks like a good package though so hopefully it will improve
      and become usable in the future.
1.1.0 Move all install, uninstall and test methods to subprocesses. This avoids
      a problem where importing onnxruntime can fail if it occurs after impoorting
      PyQt6 and also means that the script process sys.modules is kept free of
      the GPU frameworks, so there is no longer any need to lock tabs and require
      script restarts.
1.1.1 Improve support (or clearly explain lack of it) for torch+rocm on Windows
      machines. Note that only a more limited range of AMD GPUs are supported on
      Windows at present.
"""

import os
import re
import sys
import subprocess
from typing import Optional, Dict, Any

import sirilpy as s
if not s.utility.check_module_version(">=1.0.13"):
    print("Error: sirilpy module is too old and does not support this script.")
    sys.exit(1)

s.ensure_installed("PyQt6")
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QTextEdit, QGroupBox, QComboBox,
    QMessageBox, QProgressBar, QScrollArea, QGridLayout, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor

# Import sirilpy modules - these should be available if sirilpy is installed
try:
    from sirilpy.gpuhelper import (
        get_gpu_detection_report, ONNXHelper, TorchHelper, JaxHelper
    )
except ImportError:
    print("Error: sirilpy.gpuhelper not found. Please install sirilpy first.")
    sys.exit(1)




def get_installed_version_safe(module_name: str) -> Optional[str]:
    """
    Get installed version of a module WITHOUT importing it.
    Uses pip to check installation status to avoid import conflicts.
    """
    try:
        # Map module names to package names
        package_map = {
            'onnxruntime': ['onnxruntime-gpu', 'onnxruntime-directml',
                           'onnxruntime-openvino', 'onnxruntime-rocm', 'onnxruntime'],
            'torch': ['torch'],
            'jax': ['jax', 'jax-metal']
        }

        packages = package_map.get(module_name, [module_name])

        for package in packages:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', package],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        version = line.split(':', 1)[1].strip()
                        return f"{package} {version}"

        return None
    except Exception:
        return None


class FrameworkSubprocessWorker(QThread):
    """
    Worker thread that runs a framework operation (install / uninstall / test)
    in a fresh subprocess so that libraries such as onnxruntime, torch, and jax
    are never imported inside the PyQt6 process.  On some platforms importing
    those libraries *after* PyQt6 causes crashes or silent failures.

    The subprocess communicates back over stdout using a simple line protocol:
        LOG:<message>          - a progress line to display in the UI
        RESULT:<1|0>:<message> - final result (1 = success, 0 = failure)
    Any other non-empty output line is treated as a LOG line (e.g. raw pip output).
    """
    finished = pyqtSignal(bool, str)   # success, final message
    progress = pyqtSignal(str)         # intermediate log lines

    def __init__(self, script: str, parent=None):
        super().__init__(parent)
        self.script = script

    def run(self):
        try:
            proc = subprocess.Popen(
                [sys.executable, '-c', self.script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1,
                cwd=os.path.dirname(sys.executable),
            )
            final_success = False
            final_message = "Operation finished"
            for line in proc.stdout:
                line = line.rstrip('\n')
                if line.startswith('LOG:'):
                    self.progress.emit(line[4:])
                elif line.startswith('RESULT:'):
                    parts = line[7:].split(':', 1)
                    final_success = parts[0] == '1'
                    final_message = parts[1] if len(parts) > 1 else ''
                elif line:
                    self.progress.emit(line)
            proc.wait()
            self.finished.emit(final_success, final_message)
        except Exception as e:
            self.finished.emit(False, f"Subprocess error: {str(e)}")


class HardwareInfoTab(QWidget):
    """Tab showing detected hardware information"""

    def __init__(self, report: Dict[str, Any]):
        super().__init__()
        self.report = report
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Detected Hardware")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # System info
        system_group = QGroupBox("System Information")
        system_layout = QGridLayout()
        system_layout.addWidget(QLabel("Operating System:"), 0, 0)
        system_layout.addWidget(QLabel(self.report['system'].capitalize()), 0, 1)
        system_group.setLayout(system_layout)
        scroll_layout.addWidget(system_group)

        # GPU Priority
        priority_group = QGroupBox("GPU Priority")
        priority_layout = QVBoxLayout()

        primary_gpu = self.report['gpu_priority']['primary_gpu']
        reason = self.report['gpu_priority']['reason']

        priority_label = QLabel(f"<b>Primary GPU:</b> {primary_gpu}")
        reason_label = QLabel(f"<b>Reason:</b> {reason}")
        reason_label.setWordWrap(True)

        priority_layout.addWidget(priority_label)
        priority_layout.addWidget(reason_label)
        priority_group.setLayout(priority_layout)
        scroll_layout.addWidget(priority_group)

        # NVIDIA GPU info
        if self.report['hardware']['nvidia']:
            nvidia_group = self._create_gpu_group("NVIDIA GPU", self.report['hardware']['nvidia'])
            scroll_layout.addWidget(nvidia_group)

        # AMD GPU info
        if self.report['hardware']['amd']:
            amd_group = self._create_gpu_group("AMD GPU", self.report['hardware']['amd'])
            scroll_layout.addWidget(amd_group)

        # Intel GPU info
        if self.report['hardware']['intel']:
            intel_group = self._create_gpu_group("Intel GPU", self.report['hardware']['intel'])
            scroll_layout.addWidget(intel_group)

        # Apple Silicon info
        if self.report['hardware']['apple_silicon']:
            apple_group = self._create_gpu_group("Apple Silicon", self.report['hardware']['apple_silicon'])
            scroll_layout.addWidget(apple_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        self.setLayout(layout)

    def _create_gpu_group(self, title: str, gpu_info: Dict[str, Any]) -> QGroupBox:
        """Create a group box for GPU information"""
        group = QGroupBox(title)
        layout = QGridLayout()

        row = 0
        for key, value in gpu_info.items():
            if value is not None:
                # Format the key nicely
                display_key = key.replace('_', ' ').title() + ":"
                layout.addWidget(QLabel(display_key), row, 0)
                layout.addWidget(QLabel(str(value)), row, 1)
                row += 1

        group.setLayout(layout)
        return group


class FrameworkTab(QWidget):
    """Base class for ONNX/Torch/JAX tabs"""

    def __init__(self, framework_name: str, module_name: str, helper_class,
                 recommendation: Dict[str, Any], is_apple_silicon: bool = False):
        super().__init__()
        self.framework_name = framework_name
        self.module_name = module_name
        self.helper = helper_class()
        self.recommendation = recommendation
        self.is_apple_silicon = is_apple_silicon
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel(f"{self.framework_name} Framework Manager")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Current installation status
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        status_font = QFont()
        status_font.setPointSize(10)
        self.status_label.setFont(status_font)
        layout.addWidget(self.status_label)

        # Update initial status
        self.update_status()

        # Separator
        layout.addWidget(QLabel(""))

        # Recommended configuration
        rec_group = QGroupBox("Recommended Configuration")
        rec_layout = QGridLayout()
        row = 0
        for key, value in self.recommendation.items():
            if value is not None:
                display_key = key.replace('_', ' ').title() + ":"
                rec_layout.addWidget(QLabel(display_key), row, 0)

                if isinstance(value, list):
                    value_str = ", ".join(value)
                else:
                    value_str = str(value)

                value_label = QLabel(value_str)
                value_label.setWordWrap(True)
                rec_layout.addWidget(value_label, row, 1)
                row += 1
        rec_group.setLayout(rec_layout)
        layout.addWidget(rec_group)

        # Action buttons
        button_layout = QHBoxLayout()

        self.install_recommended_btn = QPushButton("Install Recommended")
        self.install_recommended_btn.clicked.connect(self.install_recommended)
        button_layout.addWidget(self.install_recommended_btn)

        self.uninstall_btn = QPushButton("Uninstall")
        self.uninstall_btn.clicked.connect(self.uninstall)
        button_layout.addWidget(self.uninstall_btn)

        self.test_btn = QPushButton("Run Tests")
        self.test_btn.clicked.connect(self.run_tests)
        button_layout.addWidget(self.test_btn)

        layout.addLayout(button_layout)

        # Manual installation section
        manual_group = QGroupBox("Manual Installation (Advanced)")
        manual_layout = QVBoxLayout()

        self.manual_combo = QComboBox()
        self.populate_manual_options()
        manual_layout.addWidget(QLabel("Select variant:"))
        manual_layout.addWidget(self.manual_combo)

        self.install_manual_btn = QPushButton("Install Selected Variant")
        self.install_manual_btn.clicked.connect(self.install_manual)
        manual_layout.addWidget(self.install_manual_btn)

        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Log output
        log_label = QLabel("Output Log:")
        layout.addWidget(log_label)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(200)
        layout.addWidget(self.log_output)

        layout.addStretch()
        self.setLayout(layout)

    def update_status(self):
        """Update the installation status display"""
        version = get_installed_version_safe(self.module_name)

        if version:
            self.status_label.setText(f"<b>Current Installation:</b> {version}")
            self.status_label.setStyleSheet("color: green;")
        else:
            self.status_label.setText(f"<b>Current Installation:</b> Not installed")
            self.status_label.setStyleSheet("color: red;")

    def populate_manual_options(self):
        """Override in subclasses to provide framework-specific options"""
        pass

    def log(self, message: str):
        """Add message to log output"""
        self.log_output.append(message)
        self.log_output.moveCursor(QTextCursor.MoveOperation.End)
        # Force UI update
        QApplication.processEvents()

    def set_buttons_enabled(self, enabled: bool):
        """Enable/disable all action buttons during a running operation."""
        self.install_recommended_btn.setEnabled(enabled)
        self.uninstall_btn.setEnabled(enabled)
        self.install_manual_btn.setEnabled(enabled)
        self.test_btn.setEnabled(enabled)

        self.progress_bar.setVisible(not enabled)
        if not enabled:
            self.progress_bar.setRange(0, 0)  # Indeterminate progress

    def _start_framework_worker(self, script: str, on_finished_callback):
        """Launch a FrameworkSubprocessWorker and wire up its signals."""
        worker = FrameworkSubprocessWorker(script, parent=self)
        worker.progress.connect(self.log)
        worker.finished.connect(on_finished_callback)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        return worker

    def install_recommended(self):
        """Override in subclasses"""
        pass

    def install_manual(self):
        """Override in subclasses"""
        pass

    def uninstall(self):
        """Override in subclasses"""
        pass

    def run_tests(self):
        """Override in subclasses"""
        pass


class ONNXTab(FrameworkTab):
    """Tab for ONNX Runtime management"""

    def __init__(self, recommendation: Dict[str, Any], is_apple_silicon: bool = False):
        super().__init__("ONNX Runtime", "onnxruntime", ONNXHelper,
                        recommendation, is_apple_silicon)

    def populate_manual_options(self):
        if self.is_apple_silicon:
            options = [
                "onnxruntime (CoreML/Apple acceleration)",
                "onnxruntime-openvino (Intel OpenVINO)",
            ]
        else:
            options = [
                "onnxruntime (CPU)",
                "onnxruntime-gpu (NVIDIA CUDA)",
                "onnxruntime-directml (DirectML - Windows)",
                "onnxruntime-openvino (Intel OpenVINO)",
            ]
        self.manual_combo.addItems(options)

    def install_recommended(self):
        current_version = get_installed_version_safe(self.module_name)
        if current_version:
            self.log(f"Current installation detected: {current_version}")
            self.log("Uninstalling existing version before installing recommended...")

        self.log(f"\n{'='*60}")
        self.log("Installing recommended ONNX Runtime configuration...")
        self.log(f"Package: {self.recommendation['package']}")
        self.log(f"Backend: {self.recommendation['backend']}")
        self.set_buttons_enabled(False)

        do_uninstall = bool(current_version)
        script = f"""\
try:
    from sirilpy.gpuhelper import ONNXHelper
    helper = ONNXHelper()
    if {do_uninstall!r}:
        print('LOG:Uninstalling existing ONNX Runtime...')
        helper.uninstall_onnxruntime()
    print('LOG:Running installer...')
    success = helper.install_onnxruntime()
    if success:
        print('RESULT:1:Installation completed')
    else:
        print('RESULT:0:Installation failed')
except Exception as e:
    print(f'RESULT:0:{{e}}')
"""
        self._start_framework_worker(script, self.on_install_finished)

    def install_manual(self):
        selected = self.manual_combo.currentText()
        package = selected.split(" ")[0]

        reply = QMessageBox.question(
            self, 'Confirm Manual Installation',
            f"Install {package}?\n\nThis may not be optimal for your hardware.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            current_version = get_installed_version_safe(self.module_name)

            self.log(f"\n{'='*60}")
            self.log(f"Manually installing {package}...")
            if current_version:
                self.log(f"Current installation: {current_version}")
                self.log("Uninstalling existing version first...")

            self.set_buttons_enabled(False)

            script = f"""\
import sys, subprocess
try:
    from sirilpy.gpuhelper import ONNXHelper
    helper = ONNXHelper()
    print('LOG:Uninstalling existing ONNX Runtime packages...')
    helper.uninstall_onnxruntime()
    print('LOG:Installing {package}...')
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', {package!r}],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print('RESULT:1:Successfully installed {package}')
    else:
        err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'pip failed'
        print(f'RESULT:0:{{err}}')
except Exception as e:
    print(f'RESULT:0:{{e}}')
"""
            self._start_framework_worker(script, self.on_install_finished)

    def uninstall(self):
        reply = QMessageBox.question(
            self, 'Confirm Uninstallation',
            "Uninstall all ONNX Runtime packages?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.log(f"\n{'='*60}")
            self.log("Uninstalling ONNX Runtime...")
            self.set_buttons_enabled(False)

            script = """\
try:
    from sirilpy.gpuhelper import ONNXHelper
    helper = ONNXHelper()
    result = helper.uninstall_onnxruntime()
    pkg_list = ', '.join(result) if result else 'none'
    print(f'RESULT:1:Uninstalled packages: {pkg_list}')
except Exception as e:
    print(f'RESULT:0:{e}')
"""
            self._start_framework_worker(script, self.on_uninstall_finished)

    def run_tests(self):
        if not get_installed_version_safe(self.module_name):
            self.log(f"\n{'='*60}")
            self.log("✗ Error: ONNX Runtime is not installed. Please install it first.")
            return

        self.log(f"\n{'='*60}")
        self.log("Running ONNX Runtime tests...")
        self.set_buttons_enabled(False)

        expected_backend = self.recommendation.get('backend', '')

        script = f"""\
try:
    import onnxruntime as ort
    from sirilpy.gpuhelper import ONNXHelper
    helper = ONNXHelper()

    available_providers = ort.get_available_providers()
    print(f'LOG:Available providers: {{", ".join(available_providers)}}')

    working_providers = helper.test_onnxruntime(ort)

    gpu_providers = [
        'CUDAExecutionProvider', 'TensorrtExecutionProvider',
        'DmlExecutionProvider', 'OpenVINOExecutionProvider',
        'ROCMExecutionProvider', 'CoreMLExecutionProvider',
    ]

    if working_providers:
        working_names = []
        for p in working_providers:
            working_names.append(p[0] if isinstance(p, tuple) else str(p))

        has_gpu   = any(g in working_names       for g in gpu_providers)
        avail_gpu = any(g in available_providers for g in gpu_providers)

        if has_gpu:
            print(f'RESULT:1:\u2713 Tests passed! Working providers: {{", ".join(working_names)}}')
        elif avail_gpu:
            failed = [p for p in available_providers
                      if p in gpu_providers and p not in working_names]
            msg = (
                '\u26a0 GPU acceleration NOT working!\\n'
                f'Working providers: {{", ".join(working_names)}}\\n'
                f'Failed providers: {{", ".join(failed)}}\\n'
                f'Expected backend: {expected_backend}\\n\\n'
                'Possible causes:\\n'
                '- Missing CUDA libraries (check nvidia-smi)\\n'
                '- Wrong CUDA version for this onnxruntime build\\n'
                '- Missing TensorRT libraries\\n'
                '- DirectML not properly configured\\n\\n'
                'Recommendation: Check system libraries or try onnxruntime-directml on Windows'
            )
            print(f'RESULT:0:{{msg}}')
        else:
            expected_backend = {expected_backend!r}
            if expected_backend not in ('cpu', 'coreml'):
                msg = (
                    '\u26a0 CPU-only runtime installed but GPU expected!\\n'
                    f'Working providers: {{", ".join(working_names)}}\\n'
                    f'Expected backend: {expected_backend}\\n'
                    'Recommendation: Reinstall with the recommended GPU package'
                )
                print(f'RESULT:0:{{msg}}')
            else:
                print(f'RESULT:1:\u2713 Tests passed! Working providers: {{", ".join(working_names)}}')
    else:
        print('RESULT:0:\u2717 Tests failed - no working execution providers found')
except ImportError:
    print('RESULT:0:\u2717 ONNX Runtime not installed')
except Exception as e:
    print(f'RESULT:0:\u2717 Test error: {{e}}')
"""
        self._start_framework_worker(script, self.on_test_finished)

    def on_install_finished(self, success: bool, message: str):
        self.set_buttons_enabled(True)
        self.log(f"{'✓' if success else '✗'} {message}")
        self.update_status()
        if success:
            self.log("Running post-installation tests...")
            self.run_tests()

    def on_uninstall_finished(self, success: bool, message: str):
        self.set_buttons_enabled(True)
        self.log(f"{'✓' if success else '✗'} {message}")
        self.update_status()

    def on_test_finished(self, success: bool, message: str):
        self.set_buttons_enabled(True)
        self.log(message)
        if not success:
            self.log("\nRecommendation: Restart the script and reinstall with the recommended configuration.")


# AMD-hosted ROCm wheels for Windows (Python 3.12 only, ROCm 7.2)
_WINDOWS_ROCM_WHEELS = [
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1+rocmsdk20260116-cp312-cp312-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchvision-0.24.1+rocmsdk20260116-cp312-cp312-win_amd64.whl",
]

# RX 7000/9000 series and Ryzen AI 300/Max are the only supported Windows ROCm GPUs
_WINDOWS_ROCM_GPU_PATTERN = re.compile(
    r'RX\s*[79]\d{3}|Ryzen\s+AI\s+(3\d\d|Max)',
    re.IGNORECASE
)

class TorchTab(FrameworkTab):
    """Tab for PyTorch management"""

    def __init__(self, recommendation: Dict[str, Any], is_apple_silicon: bool = False,
                 report: Optional[Dict[str, Any]] = None):
        self.report = report or {}
        super().__init__("PyTorch", "torch", TorchHelper, recommendation, is_apple_silicon)

    def populate_manual_options(self):
        options = [
            "torch (CPU only)",
            "torch+cu118 (CUDA 11.8 - GTX 9xx-RTX 40xx)",
            "torch+cu121 (CUDA 12.1 - GTX 10xx-RTX 40xx)",
            "torch+cu126 (CUDA 12.6 - RTX 10xx-RTX 40xx)",
            "torch+cu128 (CUDA 12.8 - GTX 16xx-RTX 50xx)",
            "torch+cu130 (CUDA 13.0 - GTX 16xx-RTX 50xx)",
            "torch+rocm6.2 (AMD ROCm - Vega, RDNA1 cards)",
            "torch+rocm7.1 (AMD ROCm - modern AMD cards)",
            "torch+rocm7.2-win (AMD ROCm 7.2 - Windows, RX 7000/9000 only)",
            "torch+xpu (Intel Arc GPUs / Xe-core iGPUs)"
        ]
        self.manual_combo.addItems(options)

    # ------------------------------------------------------------------
    # Windows ROCm gating helpers
    # ------------------------------------------------------------------

    def _is_windows_rocm_case(self) -> bool:
        """Return True when the recommendation is ROCm and we are on Windows."""
        backend = self.recommendation.get('backend', '').lower()
        return sys.platform == 'win32' and 'rocm' in backend

    def _check_windows_rocm_gates(self) -> tuple[bool, str]:
        """
        Verify all prerequisites for AMD's Windows ROCm wheels.
        Returns (ok, human-readable error string).
        """
        errors = []

        # 1. Python 3.12 — the only version AMD ships wheels for
        if sys.version_info[:2] != (3, 12):
            errors.append(
                f"Python 3.12 is required for Windows ROCm wheels.\n"
                f"You are running Python {sys.version_info.major}.{sys.version_info.minor}.\n"
                "AMD does not currently publish wheels for any other Python version."
            )

        # 2. Windows 11 (build ≥ 22000) — Windows 10 is not supported
        try:
            build = sys.getwindowsversion().build
            if build < 22000:
                errors.append(
                    f"Windows 11 is required for AMD ROCm support.\n"
                    f"Detected Windows build: {build} (Windows 10 or earlier)."
                )
        except AttributeError:
            errors.append("Could not determine Windows build number.")

        # 3. Compatible GPU — RX 7000/9000 series or Ryzen AI 300/Max only
        amd_info = self.report.get('hardware', {}).get('amd') or {}
        gpu_name = str(amd_info.get('gpu_name', ''))
        if not _WINDOWS_ROCM_GPU_PATTERN.search(gpu_name):
            errors.append(
                f"GPU not supported by AMD's Windows ROCm distribution.\n"
                f"Detected: '{gpu_name or 'unknown'}'\n"
                "Supported hardware: Radeon RX 7000/9000 series, "
                "Ryzen AI 300 series, Ryzen AI Max."
            )

        if errors:
            return False, "\n\n".join(errors)
        return True, ""

    def _show_windows_rocm_gate_error(self, message: str):
        QMessageBox.critical(
            self,
            "Windows ROCm — Prerequisites Not Met",
            f"The AMD Windows ROCm wheels cannot be installed:\n\n{message}\n\n"
            "On Windows, ROCm PyTorch wheels are distributed separately by AMD "
            "from repo.radeon.com and carry stricter requirements than the Linux "
            "build. Consider installing CPU-only PyTorch instead, or switch to "
            "Linux for full ROCm support."
        )

    # ------------------------------------------------------------------
    # Install helpers
    # ------------------------------------------------------------------

    def _build_windows_rocm_script(self, do_uninstall: bool) -> str:
        wheels_repr = repr(_WINDOWS_ROCM_WHEELS)
        return f"""\
import sys
import subprocess
try:
    from sirilpy.gpuhelper import TorchHelper
    helper = TorchHelper()
    if {do_uninstall!r}:
        print('LOG:Uninstalling existing PyTorch...')
        helper.uninstall_torch()
    print('LOG:Installing AMD Windows ROCm 7.2 wheels...')
    print('LOG:(These are hosted on repo.radeon.com, not PyPI — download may be slow)')
    wheels = {wheels_repr}
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--no-cache-dir'] + wheels,
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        print(f'LOG:{{line}}')
    for line in result.stderr.splitlines():
        print(f'LOG:{{line}}')
    if result.returncode == 0:
        print('RESULT:1:PyTorch ROCm (Windows) installation completed')
    else:
        last_err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'pip failed'
        print(f'RESULT:0:{{last_err}}')
except Exception as e:
    print(f'RESULT:0:{{e}}')
"""

    # ------------------------------------------------------------------
    # Public actions
    # ------------------------------------------------------------------

    def install_recommended(self):
        current_version = get_installed_version_safe(self.module_name)
        if current_version:
            self.log(f"Current installation detected: {current_version}")
            self.log("Uninstalling existing version before installing recommended...")

        self.log(f"\n{'='*60}")
        self.log("Installing recommended PyTorch configuration...")
        self.log(f"Backend: {self.recommendation['backend']}")
        if self.recommendation.get('cuda_version'):
            self.log(f"CUDA version: {self.recommendation['cuda_version']}")
        self.set_buttons_enabled(False)

        do_uninstall = bool(current_version)

        # ---- Windows ROCm: special path via AMD's wheel repo ----
        if self._is_windows_rocm_case():
            ok, error_msg = self._check_windows_rocm_gates()
            if not ok:
                self.set_buttons_enabled(True)
                self._show_windows_rocm_gate_error(error_msg)
                self.log("✗ Windows ROCm prerequisites not met — installation aborted.")
                self.log("  See the error dialogue for details.")
                return

            self.log("NOTE: AMD Windows ROCm wheels are hosted on repo.radeon.com,")
            self.log("      not on PyPI. A separate AMD driver is also required.")
            self.log(f"      Wheels: ROCm 7.2 / Python 3.12 / win_amd64")
            script = self._build_windows_rocm_script(do_uninstall)
            self._start_framework_worker(script, self.on_install_finished)
            return

        # ---- All other platforms / backends: existing path ----
        cuda_version = self.recommendation.get('cuda_version')
        extra_index_url = self.recommendation.get('extra_index_url')
        packages = self.recommendation.get('packages', ['torch', 'torchvision', 'torchaudio'])

        script = f"""\
try:
    from sirilpy.gpuhelper import TorchHelper
    helper = TorchHelper()
    if {do_uninstall!r}:
        print('LOG:Uninstalling existing PyTorch...')
        helper.uninstall_torch()
    print('LOG:Running installer...')
    helper.install_torch(
        cuda_version={cuda_version!r},
        extra_index_url={extra_index_url!r},
        packages={packages!r},
    )
    print('RESULT:1:PyTorch installation completed')
except Exception as e:
    print(f'RESULT:0:{{e}}')
"""
        self._start_framework_worker(script, self.on_install_finished)

    def install_manual(self):
        selected = self.manual_combo.currentText()

        reply = QMessageBox.question(
            self, 'Confirm Manual Installation',
            f"Install {selected}?\n\nThis may not be optimal for your hardware.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            current_version = get_installed_version_safe(self.module_name)

            self.log(f"\n{'='*60}")
            self.log(f"Manually installing {selected}...")
            if current_version:
                self.log(f"Current installation: {current_version}")
                self.log("Uninstalling existing version first...")

            self.set_buttons_enabled(False)

            # ---- Windows ROCm 7.2 manual option ----
            if "rocm7.2-win" in selected:
                ok, error_msg = self._check_windows_rocm_gates()
                if not ok:
                    self.set_buttons_enabled(True)
                    self._show_windows_rocm_gate_error(error_msg)
                    self.log("✗ Windows ROCm prerequisites not met — installation aborted.")
                    return
                self.log("NOTE: Installing AMD Windows ROCm 7.2 wheels from repo.radeon.com")
                script = self._build_windows_rocm_script(bool(current_version))
                self._start_framework_worker(script, self.on_install_finished)
                return

            # ---- Standard manual options ----
            cuda_version = None
            extra_index_url = None

            if "cu118" in selected:
                cuda_version = "cu118"
                extra_index_url = "https://download.pytorch.org/whl/cu118"
            elif "cu121" in selected:
                cuda_version = "cu121"
                extra_index_url = "https://download.pytorch.org/whl/cu121"
            elif "cu126" in selected:
                cuda_version = "cu126"
                extra_index_url = "https://download.pytorch.org/whl/cu126"
            elif "cu128" in selected:
                cuda_version = "cu128"
                extra_index_url = "https://download.pytorch.org/whl/cu128"
            elif "cu130" in selected:
                cuda_version = "cu130"
                extra_index_url = "https://download.pytorch.org/whl/cu130"
            elif "rocm6.2" in selected:
                extra_index_url = "https://download.pytorch.org/whl/rocm6.2"
            elif "rocm7.1" in selected:
                extra_index_url = "https://download.pytorch.org/whl/rocm7.1"
            elif "xpu" in selected:
                extra_index_url = "https://download.pytorch.org/whl/xpu"

            script = f"""\
try:
    from sirilpy.gpuhelper import TorchHelper
    helper = TorchHelper()
    print('LOG:Uninstalling existing PyTorch packages...')
    helper.uninstall_torch()
    print('LOG:Installing {selected}...')
    helper.install_torch(
        cuda_version={cuda_version!r},
        extra_index_url={extra_index_url!r},
        packages=['torch', 'torchvision'],
    )
    print('RESULT:1:Successfully installed {selected}')
except Exception as e:
    print(f'RESULT:0:{{e}}')
"""
            self._start_framework_worker(script, self.on_install_finished)

    def uninstall(self):
        reply = QMessageBox.question(
            self, 'Confirm Uninstallation',
            "Uninstall all PyTorch packages?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.log(f"\n{'='*60}")
            self.log("Uninstalling PyTorch...")
            self.set_buttons_enabled(False)

            script = """\
try:
    from sirilpy.gpuhelper import TorchHelper
    helper = TorchHelper()
    result = helper.uninstall_torch()
    pkg_list = ', '.join(result) if result else 'none'
    print(f'RESULT:1:Uninstalled packages: {pkg_list}')
except Exception as e:
    print(f'RESULT:0:{e}')
"""
            self._start_framework_worker(script, self.on_uninstall_finished)

    def run_tests(self):
        if not get_installed_version_safe(self.module_name):
            self.log(f"\n{'='*60}")
            self.log("✗ Error: PyTorch is not installed. Please install it first.")
            return

        self.log(f"\n{'='*60}")
        self.log("Running PyTorch tests...")
        self.set_buttons_enabled(False)

        script = """\
try:
    from sirilpy.gpuhelper import TorchHelper
    helper = TorchHelper()
    success = helper.test_torch()
    if success:
        print('RESULT:1:\u2713 All PyTorch tests passed!')
    else:
        print('RESULT:0:\u2717 Some PyTorch tests failed')
except Exception as e:
    print(f'RESULT:0:\u2717 Test error: {e}')
"""
        self._start_framework_worker(script, self.on_test_finished)

    def on_install_finished(self, success: bool, message: str):
        self.set_buttons_enabled(True)
        self.log(f"{'✓' if success else '✗'} {message}")
        self.update_status()
        if success:
            self.log("Running post-installation tests...")
            self.run_tests()

    def on_uninstall_finished(self, success: bool, message: str):
        self.set_buttons_enabled(True)
        self.log(f"{'✓' if success else '✗'} {message}")
        self.update_status()

    def on_test_finished(self, success: bool, message: str):
        self.set_buttons_enabled(True)
        self.log(message)
        if not success:
            self.log("\nRecommendation: Restart the script and reinstall with the recommended configuration.")


class JAXTab(FrameworkTab):
    """Tab for JAX management"""

    def __init__(self, recommendation: Dict[str, Any], is_apple_silicon: bool = False):
        super().__init__("JAX", "jax", JaxHelper, recommendation, is_apple_silicon)

    def populate_manual_options(self):
        options = [
            "jax (CPU only)",
            "jax[cuda12] (NVIDIA CUDA 12 - Linux)",
            "jax[rocm] (AMD ROCm - Linux)",
            "jax-metal (Apple Silicon - macOS)",
        ]
        self.manual_combo.addItems(options)

    def install_recommended(self):
        current_version = get_installed_version_safe(self.module_name)
        if current_version:
            self.log(f"Current installation detected: {current_version}")
            self.log("Uninstalling existing version before installing recommended...")

        self.log(f"\n{'='*60}")
        self.log("Installing recommended JAX configuration...")
        self.log(f"Backend: {self.recommendation['backend']}")
        self.log(f"Packages: {', '.join(self.recommendation['packages'])}")
        self.set_buttons_enabled(False)

        do_uninstall = bool(current_version)
        packages = self.recommendation['packages']

        if 'jax[cuda12]' in packages:
            variant = 'jax[cuda12]'
        elif 'jax[rocm]' in packages:
            variant = 'jax[rocm]'
        elif 'jax-metal' in packages:
            variant = 'jax-metal'
        else:
            variant = 'jax[cpu]'

        script = f"""\
try:
    from sirilpy.gpuhelper import JaxHelper
    helper = JaxHelper()
    if {do_uninstall!r}:
        print('LOG:Uninstalling existing JAX...')
        helper.uninstall_jax()
    print('LOG:Running installer...')
    success = helper.install_jax(force_variant={variant!r})
    if success:
        print('RESULT:1:JAX installation completed')
    else:
        print('RESULT:0:JAX installation failed')
except Exception as e:
    print(f'RESULT:0:{{e}}')
"""
        self._start_framework_worker(script, self.on_install_finished)

    def install_manual(self):
        selected = self.manual_combo.currentText()
        variant = selected.split(" ")[0]

        reply = QMessageBox.question(
            self, 'Confirm Manual Installation',
            f"Install {variant}?\n\nThis may not be optimal for your hardware.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            current_version = get_installed_version_safe(self.module_name)

            self.log(f"\n{'='*60}")
            self.log(f"Manually installing {variant}...")
            if current_version:
                self.log(f"Current installation: {current_version}")
                self.log("Uninstalling existing version first...")

            self.set_buttons_enabled(False)

            script = f"""\
try:
    from sirilpy.gpuhelper import JaxHelper
    helper = JaxHelper()
    print('LOG:Uninstalling existing JAX packages...')
    helper.uninstall_jax()
    print('LOG:Installing {variant}...')
    success = helper.install_jax(force_variant={variant!r})
    if success:
        print('RESULT:1:Successfully installed {variant}')
    else:
        print('RESULT:0:Installation of {variant} failed')
except Exception as e:
    print(f'RESULT:0:{{e}}')
"""
            self._start_framework_worker(script, self.on_install_finished)

    def uninstall(self):
        reply = QMessageBox.question(
            self, 'Confirm Uninstallation',
            "Uninstall all JAX packages?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.log(f"\n{'='*60}")
            self.log("Uninstalling JAX...")
            self.set_buttons_enabled(False)

            script = """\
try:
    from sirilpy.gpuhelper import JaxHelper
    helper = JaxHelper()
    result = helper.uninstall_jax()
    uninstalled = result.get('uninstalled_packages', [])
    pkg_names = [p['name'] for p in uninstalled]
    pkg_list = ', '.join(pkg_names) if pkg_names else 'none'
    print(f'RESULT:1:Uninstalled packages: {pkg_list}')
except Exception as e:
    print(f'RESULT:0:{e}')
"""
            self._start_framework_worker(script, self.on_uninstall_finished)

    def run_tests(self):
        if not get_installed_version_safe(self.module_name):
            self.log(f"\n{'='*60}")
            self.log("✗ Error: JAX is not installed. Please install it first.")
            return

        self.log(f"\n{'='*60}")
        self.log("Running JAX tests...")
        self.set_buttons_enabled(False)

        script = """\
try:
    from sirilpy.gpuhelper import JaxHelper
    helper = JaxHelper()
    success, device = helper.test_jax()
    if success:
        print(f'RESULT:1:\u2713 JAX tests passed! Using device: {device}')
    else:
        print('RESULT:0:\u2717 JAX tests failed')
except Exception as e:
    print(f'RESULT:0:\u2717 Test error: {e}')
"""
        self._start_framework_worker(script, self.on_test_finished)

    def on_install_finished(self, success: bool, message: str):
        self.set_buttons_enabled(True)
        self.log(f"{'✓' if success else '✗'} {message}")
        self.update_status()
        if success:
            self.log("Running post-installation tests...")
            self.run_tests()

    def on_uninstall_finished(self, success: bool, message: str):
        self.set_buttons_enabled(True)
        self.log(f"{'✓' if success else '✗'} {message}")
        self.update_status()

    def on_test_finished(self, success: bool, message: str):
        self.set_buttons_enabled(True)
        self.log(message)
        if not success:
            self.log("\nRecommendation: Restart the script and reinstall with the recommended configuration.")

class AboutTab(QWidget):
    """Tab showing information about the application"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("About GPU Framework Manager")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Separator
        layout.addWidget(QLabel(""))

        # Read-only text box with content
        self.about_text = QTextEdit()
        self.about_text.setReadOnly(True)

        # Lorem ipsum content (you can modify this)
        about_content = """
GPU Framework Manager

Version 1.0.0

This script separates management of installation of GPU frameworks from the image
processing scripts that use them. This allows for a much more comprehensive
dedicated interface showing detected hardware info, recommended versions of each
framework and options for installation of either the recommended backend or a
manually selected backend, as well as testing.

All installation, uninstallation, and testing operations are run in isolated
subprocesses. This means framework libraries are never imported into the GUI
process itself, avoiding import-order conflicts with PyQt6 that occur on some
platforms. It also means you can freely install, test, and reinstall without
needing to restart the script between operations.

Features:
• Automatic GPU hardware detection
• Smart framework recommendations
• Support for ONNX Runtime, PyTorch, and JAX
• Easy installation and testing

Note that while the script can supervise installation of Torch linked against
different CUDA versions, this is only half of the need: you must ensure that your
system GPU drivers are suitable to support the version of CUDA you want to select.
For example, CUDA 13.0 requires at least NVIDIA driver version 580: GPU drivers
must be managed through the usual Operating System-level mechanisms.

Note also that some frameworks rely on packages provided by the Operating System.
So while Torch-CUDA will pull in all the NVIDIA libraries it needs as wheels,
onnxruntime-gpu (used for the NVIDIA ONNX backend on Linux) will not, and relies
on properly installed and reachable system CUDA libraries.

As of version 1.0.2 of this script the Jax tab has been hidden. Although Jax looks like
an interesting framework and one that has the potential to become really useful during
the 1.4.x development cycle, its current status is very variable between OSes:
it works well on Linux configurations, but is much more experimental and unstable
on other architectures. It is NOT recommended to write scripts using Jax at the
present time: they will not yet be acccepted into the scripts repository as there
is insufficient confidence that they will work on all Siril platforms.

        """

        self.about_text.setPlainText(about_content.strip())
        layout.addWidget(self.about_text)

        self.setLayout(layout)

class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPU Framework Manager")
        self.setMinimumSize(900, 700)

        # Get hardware detection report
        try:
            self.report = get_gpu_detection_report()
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to detect GPU hardware:\n{str(e)}\n\nPlease ensure sirilpy is properly installed."
            )
            sys.exit(1)

        # Check if Apple Silicon
        self.is_apple_silicon = bool(self.report['hardware'].get('apple_silicon'))

        self.init_ui()

    def init_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()

        # Title
        title = QLabel("GPU Framework Manager")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Manage ONNX Runtime, PyTorch, and JAX installations")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(subtitle)

        # Tab widget
        tabs = QTabWidget()

        # Hardware info tab
        hardware_tab = HardwareInfoTab(self.report)
        tabs.addTab(hardware_tab, "Hardware Info")

        # ONNX tab
        onnx_tab = ONNXTab(self.report['recommendations']['onnx'], self.is_apple_silicon)
        tabs.addTab(onnx_tab, "ONNX Runtime")

        # PyTorch tab

        torch_tab = TorchTab(self.report['recommendations']['torch'], self.is_apple_silicon, self.report)
        tabs.addTab(torch_tab, "PyTorch")

        # JAX tab
        # jax_tab = JAXTab(self.report['recommendations']['jax'], self.is_apple_silicon)
        # tabs.addTab(jax_tab, "JAX")


        # About tab
        about_tab = AboutTab()
        tabs.addTab(about_tab, "About")
        main_layout.addWidget(tabs)

        # Footer
        footer = QLabel("Powered by sirilpy.gpuhelper")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer_font = QFont()
        footer_font.setPointSize(8)
        footer.setFont(footer_font)
        main_layout.addWidget(footer)

        central_widget.setLayout(main_layout)

def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
