"""
(c) 2026, Stefan Schmidt-Bilkenroth
SPDX-License-Identifier: GPL-3.0-or-later

Dwarfum Session Selector
Version 0.9.0
"""

"""
ChangeLog:
    0.9.0   initial release, beta state for gathering feedback
"""

from genericpath import exists

import sirilpy as s
from sirilpy import LogColor

s.ensure_installed(["PyQt6"])

import json
import os
import shutil
import sys
from datetime import datetime

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

APP_NAME = "Dwarfium Archive Selector"
VERSION = "0.9.0"
BUILD = "20260314"
AUTHOR = "Stefan Schmidt-Bilkenroth"

PRESET_FILE = "dwarfium-archive-select.json"
PRESET_JSON = """
{
    "archive_dir": null
}
"""

HELP_MD = """
## Help for Dwarfium Archive Selector

>This script is for convenience when preparing processing sessions shot
with smart telescopes made by Dwarf Lab. Its intention is, to ease preparation
of light images for stacking inside Siril.
The script assumes, that the Archive is organized the same way as on the
Dwwarf Lab devices. This can be achieved by either using tool like `rsync`
to archive the sessions or by using **Dwarfium Scope Archive**, which is my
recommendation. Thanks to JC. Lesaint for his tool.

### Step 1: Select the Folder containing your archived Dwarf session

>Click the button `Change` and use the File Selector Dialog to chose
the **Astronomy** folder containing all the Dwarf Sessions you have collected.
After choosing a folder, the path to the folder is shown in green color.
The folder is saved in the preferences of the script (in Siril preferences folder)
When the path to the Archive is shown in red color, the path is not available.
This can happen, when the Archive is located on an external drive and this one
is not connected, when the script is started.

>With the `Refresh` button you can trigger to scan the archive once again
in case you have added or removed sessions in the archive, after the script
was already started

>You can't change the Siril Home directory within this script, the path is
only shown for your information.

### Step 2: Select the Target in the left panel

>After choosing an Archive folder, the script will parse the file tree inside
the archive and gather information for all sessions found in the Archive.
The folder names inside the archive are parsed based on the naming pattern used
by Dwarf Lab when storing sessions onto their devices.
The script also inspects the `shotsinfo.json` file inside each session for
additional information (number of stacked images, filter selection, ...)

>All the targets, based on the Dwarf Lab naming schema, found in these sessions
are then displayed in the left panel. Ths number of session found for each target
is shown in parenthesis.

>It can happen, that some folder names with sessions do not match the naming
schema. These are ignored by the script but you can check the Siril Log for
more information.

>MegaStack sessions are ignored, as they do not contain indiviudal image, only
the result of the MegaStack processing. Check the Siril Log for more information.

### Step 3: Select the Sessions in the right panel

>After choosing a target in the left panel the "live-stack" image ("stacked.jpg")
of the first session for this target is shown together with some basic information
on the target. Below the image and these target information the list of all sessions
on this target is shown. The list includes session start date, number of stacked
images, exposure time, gain, filter and temperature range. These informations are
taken from directory name and the file "shotsinfo.json", which is automatically
created by the Dwarf telescope while capturing.

>You can choose one or more sessions. In the bottom the number of selected images
updates accordingly and is showing the total count of images across the selected sessions.

### Step 4: Copy the lights from the selected sessions

>When you have at least selected one session with at least 1 image, the `Copy`
Button gets enabled. When you are happy with your selection, click on the
`Copy` button.

>The script will check if there are any images in the lights folder inside
the current Siril Home and if necessary ask for confirmation to delete them.
Also the `process` folder in Siril Home is deleted.

>Then it will start to copy all the images of the selected session from the
Archive into the new "lights" folder inside Siril home.
>This can take a while, but the progress of these file copy operation is
shown next to the `Copy` button.

### Step 5: Close the script window and continue your workflow

>After copying the files is completed you can close the script and
continue your workflow.

### ToDos:
- add safeguards around file operations
- proper handling of Mosaic sessions

### Caveats
>This script is provided as is and the Author takes now liability on its
functionality and any damage or loss of data when using the script.

### Legal Information:

**Author**: Stefan Schmidt-Bilkenroth

For suggestions, support and issues feel free to contact me:

>facebook: `https://www.facebook.com/stefan.ssb`

>mastodon: `https://gruene.social/@ssb`

>e-mail: `ssb@mac.com`

(c) 2026, Stefan Schmidt-Bilkenroth

SPDX-License-Identifier: GPL-3.0-or-later
"""


class Target:
    """Target Class contains the targets from the Dwarfium archives with their sessions"""

    def __init__(self, target):
        self.target = target
        self.sessions = []

    def add_session(self, session):
        self.sessions.append(session)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.target}: {len(self.sessions)} sessions"


class Session:
    """Session Class contains the information of the single session"""

    def __init__(self, path, cam, target, exp, gain, date):
        self.cam = cam
        self.target = target
        self.exposure = exp
        self.gain = gain
        self.date = date
        self.thumbnail = os.path.join(path, "stacked.jpg")
        self.ra = None
        self.dec = None
        self.min_temp = None
        self.max_temp = None
        try:
            self.prettydate = datetime.strptime(date, "%Y-%m-%d-%H-%M-%S-%f")
        except:
            self.siril.log(f"Invalid date format: {date}", LogColor.RED)
            return None

        self.calendardate = datetime(
            self.prettydate.year,
            self.prettydate.month,
            self.prettydate.day,
            self.prettydate.hour,
            self.prettydate.minute,
        )
        self.path = path
        self.ir = "unknown"
        self.taken = 0
        self.stacked = 0

    def __str__(self):
        return "'{}' with {}, {}s/{}g on {}".format(
            self.target, self.cam.lower(), self.exposure, self.gain, self.date
        )


class TreeWalkThread(QThread):
    """Thread used for walking the file structure of the archive, in case of slower hardware"""

    parse_tick = pyqtSignal(int, str)
    parse_done = pyqtSignal(int)

    def __init__(self, path):
        QThread.__init__(self, None)
        self.path = path

    def run(self):
        found = 0
        skip_folders = ["Thumbnail", "", "Solving_Failed", "CALI_FRAME", "DWARF_DARK"]

        for root, folders, _ in os.walk(self.path):
            for folder in list(folders):
                if folder in skip_folders:
                    folders.remove(folder)
                    continue
                if len(folder) == 0:
                    folders.remove(folder)
                    continue
                if "DWARF_RAW_" in folder and "_MOSAIC_" not in folder:
                    found += 1
                    self.parse_tick.emit(found, os.path.join(root, folder))

        self.parse_done.emit(found)


class CopyThread(QThread):
    """Thread to copy the files in the background, otherise the progress bar would not update"""

    progress = pyqtSignal(int)
    done = pyqtSignal(int)

    def __init__(self, files, dest):
        QThread.__init__(self, None)
        self.files = files
        self.dest = dest

    def run(self):
        done = 0
        for f in self.files:
            shutil.copy2(f, self.dest)
            done += 1
            self.progress.emit(done)
        self.done.emit(done)


def close_dialog(self):
    try:
        self.siril.disconnect()
    except Exception:
        pass  # Ignore disconnect errors
    self.close()


class PreprocessingInterface(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - v{VERSION}")
        self.initialization_successful = False

        self.siril = s.SirilInterface()

        # declare internal members
        self.archive_dir = None
        self.targets = []
        self.target_list = []
        self.selected_target = None
        self.selected_sessions = []

        try:
            self.siril.connect()
            self.siril.log("Connected to Siril", LogColor.GREEN)
        except s.SirilConnectionError:
            self.siril.log("Failed to connect to Siril", LogColor.RED)
            self.close_dialog()
            return

        try:
            self.siril.cmd("requires", "1.3.6")
        except s.CommandError:
            self.close_dialog()
            return

        self.fits_extension = self.siril.get_siril_config("core", "extension")
        # home directory is unchanged
        self.home_directory = self.siril.get_siril_wd()
        self.current_working_directory = self.siril.get_siril_wd()
        self.cwd_label = self.current_working_directory

        # Assigns collected_lights directory to store all pp_lights files
        self.collected_lights_dir = os.path.join(
            self.current_working_directory, "collected_lights"
        )
        # defaults for presets
        self.load_presets()
        self.create_widgets()
        self.initialization_successful = True  # Flag to track successful initialization

    def create_path_widgets(self, main_layout):
        """create Path box widgets"""
        paths_group = QGroupBox("Current paths:")
        cwd_layout = QVBoxLayout()
        cwd_label = QLabel(f"Current working directory: {self.cwd_label}")
        cwd_layout.addWidget(cwd_label)

        # Label with Archive path
        archive_layout = QHBoxLayout()
        self.archive_label = QLabel(f"Selected archive directory: {self.archive_dir}")
        if self.archive_dir is None or not os.path.exists(self.archive_dir):
            self.archive_label.setStyleSheet("QLabel {color: #FF0000};")
        else:
            self.archive_label.setStyleSheet("QLabel {color: #00FF00};")
        archive_layout.addWidget(self.archive_label)

        # chose archive path button
        archive_choose_btn = QPushButton("Change")
        archive_choose_btn.setMinimumWidth(80)
        archive_choose_btn.setMaximumWidth(80)
        archive_choose_btn.setMinimumHeight(35)
        archive_choose_btn.clicked.connect(self.chose_archive)
        archive_layout.addWidget(archive_choose_btn)
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setMinimumWidth(80)
        refresh_btn.setMaximumWidth(80)
        refresh_btn.setMinimumHeight(35)
        refresh_btn.clicked.connect(lambda: self.scan_archive(self.archive_dir))
        archive_layout.addWidget(refresh_btn)

        cwd_layout.addLayout(archive_layout)
        paths_group.setLayout(cwd_layout)
        main_layout.addWidget(paths_group)

    def create_list_widgets(self, main_layout):
        """create widgets with the listboxes for targets and sessions"""
        files_layout = QVBoxLayout()
        list_group = QGroupBox("Targets and Session in Dwarf Archive:")
        list_layout = QHBoxLayout()
        # Listbox for targets (left)
        self.target_listbox = QListWidget()
        self.target_listbox.setMaximumWidth(300)
        self.target_listbox.itemSelectionChanged.connect(self.target_selected)
        list_layout.addWidget(self.target_listbox)

        right_layout = QVBoxLayout()
        right_info = QHBoxLayout()
        self.thumb = QLabel()
        self.thumb.setMinimumHeight(160)
        self.thumb.setMaximumHeight(160)
        self.thumb.setMinimumWidth(300)
        self.thumb.setMaximumWidth(300)
        right_info.addWidget(self.thumb)

        self.info_box = QTextEdit(readOnly=True)
        self.info_box.setMinimumHeight(self.thumb.height())
        self.info_box.setMaximumHeight(self.thumb.height())
        # self.info_box.setMaximumWidth(300)
        right_info.addWidget(self.info_box)

        right_layout.addLayout(right_info)
        # Listbox for sessions
        self.session_listbox = QListWidget()
        self.session_listbox.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.session_listbox.itemSelectionChanged.connect(self.session_selected)
        right_layout.addWidget(self.session_listbox)
        list_layout.addLayout(right_layout)

        list_group.setLayout(list_layout)
        files_layout.addWidget(list_group)
        main_layout.addWidget(list_group)

    def create_bottom_widgets(self, main_layout):
        """create the widgets for the bottom part - progress bar and action buttons"""
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(
            0, 15, 0, 0
        )  # Add top margin to separate from content

        help_btn = QPushButton("?")
        help_btn.setMinimumWidth(35)
        help_btn.setMaximumWidth(35)
        help_btn.setMinimumHeight(35)
        help_btn.clicked.connect(self.show_help)
        button_layout.addWidget(help_btn)

        # progressbar for copy operation
        self.progressbar = QProgressBar()
        self.progressbar.setTextVisible(False)
        self.progressbar.setMinimumHeight(35)
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(1)
        self.progressbar.setValue(0)
        button_layout.addWidget(self.progressbar)

        # label showing number of files to copy
        self.progress_label = QLabel("0 / 0")
        self.progress_label.setMinimumWidth(100)
        self.progress_label.setMaximumWidth(100)
        self.progress_label.setMinimumHeight(35)
        self.progress_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        button_layout.addWidget(self.progress_label)

        # copy button
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setMinimumHeight(35)
        self.copy_btn.setMinimumWidth(80)
        self.copy_btn.setMaximumWidth(80)
        self.copy_btn.setDisabled(True)
        self.copy_btn.clicked.connect(self.start_copy)
        button_layout.addWidget(self.copy_btn)

        # close button
        close_button = QPushButton("Close")
        close_button.setMinimumWidth(100)
        close_button.setMinimumHeight(35)
        close_button.clicked.connect(self.close_dialog)
        button_layout.addWidget(close_button)

        main_layout.addLayout(button_layout)

    def create_widgets(self):
        """Creates the UI widgets using PyQt6."""

        # Main layout
        main_widget = QWidget()
        self.setMinimumSize(750, 600)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 6, 12, 12)
        main_layout.setSpacing(6)

        copyright = QLabel("(c) Stefan Schmidt-Bilkenroth, 2026")
        copyright.setStyleSheet("QLabel {font-size: 10px; color: grey}")
        copyright.setAlignment(Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(copyright)

        self.create_path_widgets(main_layout)

        self.create_list_widgets(main_layout)
        self.create_bottom_widgets(main_layout)
        QTimer.singleShot(0, self.inited)

    def select_folder(self, start_dir):
        """folder selection for the Dwarfium archive"""
        if start_dir is None or not os.path.exists(start_dir):
            start_dir = self.current_working_directory
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Archive directory",
            start_dir,
        )
        return selected_dir

    def chose_archive(self):
        """function to chose Dwarfium archive"""
        archive_dir = self.select_folder(self.archive_dir)
        if archive_dir is None:
            # do not update if file selector has been cancelled
            return
        self.archive_dir = archive_dir
        self.archive_label.setText(f"Selected archive directory: {self.archive_dir}")
        self.archive_label.setStyleSheet("QLabel {color: #00FF00};")
        self.save_presets()
        self.scan_archive(self.archive_dir)
        return

    def scan_archive(self, archive_dir=None):
        """scan the archive path for subdirectories matching Dwarf naming scheme"""
        self.siril.log(
            f"Running script version {VERSION} with arguments:\n"
            f"archive_dir: {archive_dir}",
            LogColor.GREEN,
        )
        start_dir = archive_dir
        if archive_dir is None or not os.path.exists(archive_dir):
            archive_dir = None
            start_dir = self.current_working_directory

        if archive_dir is None:
            archive_dir = self.select_folder(start_dir)

        self.archive_dir = archive_dir
        self.parse_filetree(self.archive_dir)

        self.siril.cmd("close")

    def inited(self):
        self.scan_archive(self.archive_dir)

    def ask(self, title, msg):
        """helper displaying a Question Message Box"""
        reply = QMessageBox.question(
            self,
            title,
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return reply

    def progress_update(self, value):
        """callback for progress update emitted from Copy QThread"""
        self.progressbar.setValue(value)
        self.progress_label.setText(
            f"{self.progressbar.value()} / {self.progressbar.maximum()}"
        )

    def progress_done(self, value):
        """callback for progress completed emitted from Copy QThread"""
        self.copy_btn.setDisabled(False)
        self.progressbar.setValue(0)
        self.progressbar.setMaximum(1)
        self.progress_label.setText("completed")

        reply = QMessageBox.question(
            self,
            "File copy complete",
            f"{value} files have been copied to {os.path.join(self.current_working_directory, 'lights')}",
            buttons=QMessageBox.StandardButton(
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Close
            ),
            defaultButton=QMessageBox.StandardButton.Ok,
        )
        if reply == QMessageBox.StandardButton.Close:
            self.close_dialog()

    def start_copy(self):
        """action for copy button - clean current lights and start copy operation"""
        lights_dir = os.sep.join([self.current_working_directory, "lights"])
        process_dir = os.sep.join([self.current_working_directory, "process"])

        # check if lights dir contains files and ask for confirmation to delete them
        if os.path.exists(lights_dir):
            file_cnt = len(os.listdir(lights_dir))

            if file_cnt > 0:
                self.progressbar.setMaximum(0)
                self.progress_label.setText("cleanup")

                reply = self.ask(
                    "Delete existing lights?",
                    f'This operation will delete {file_cnt} files in\n"{lights_dir}".\nProceed?',
                )
                if reply != QMessageBox.StandardButton.Yes:
                    # abort procesing, when not confirmed
                    return
                shutil.rmtree(lights_dir)

        # create new lights dir and delete process dir, when there is one
        os.mkdir(lights_dir)
        if os.path.exists(process_dir):
            shutil.rmtree(process_dir)

        # collect all the files that will get copied
        all_files = []
        for session in self.selected_sessions:
            src = session.path
            files = os.listdir(src)
            for f in files:
                if f.startswith(session.target) and f.endswith(".fits"):
                    all_files.append(os.sep.join([src, f]))
        # prepare progress bar
        self.progressbar.setMaximum(len(all_files))
        self.progressbar.setValue(0)
        self.progress_label.setText(f"0 / {len(all_files)}")
        # prepare and start background thread to actually copy the files
        # This need to be done in a backgrond thread, otherwise the progressbar does not update
        self.copyThread = CopyThread(files=all_files, dest=lights_dir)
        self.copyThread.progress.connect(self.progress_update)
        self.copyThread.done.connect(self.progress_done)
        self.copy_btn.setDisabled(True)
        self.copyThread.start()
        return

    def close_help(self):
        """action for close help button"""
        self.help_window.close()

    def show_help(self):
        """show help, which is provided in Markdown syntax in the beginning of the script"""
        self.help_window = QDialog(self)
        self.help_window.setModal(True)
        self.help_window.setWindowTitle("Dwarfium Archive Selector Help")
        self.help_window.setMinimumSize(750, 600)

        help_layout = QVBoxLayout()
        help_text = QTextEdit(readOnly=True)
        help_text.setMarkdown(HELP_MD)
        help_layout.addWidget(help_text)

        help_close = QPushButton("Close")
        help_close.setMinimumHeight(35)
        help_close.setMinimumWidth(80)
        help_close.setMaximumWidth(80)
        help_close.clicked.connect(self.close_help)
        help_layout.addWidget(help_close)
        self.help_window.setLayout(help_layout)
        self.help_window.exec()

    def close_dialog(self):
        try:
            self.siril.disconnect()
        except Exception:
            pass  # Ignore disconnect errors
        self.close()

    def print_footer(self):
        self.siril.log(
            f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            LogColor.GREEN,
        )

    def save_presets(self):
        """Save current settings and session data to preset file"""
        # Collect settings
        presets = {
            "archive_dir": self.archive_dir,
        }

        # Create presets directory if it doesn't exist
        presets_file = os.path.join(self.siril.get_siril_configdir(), PRESET_FILE)

        try:
            with open(presets_file, "w") as f:
                json.dump(presets, f, indent=4)
            self.siril.log(f"Saved preset to {presets_file}", LogColor.GREEN)
        except Exception as e:
            self.siril.log(
                f"Failed to save presets to {presets_file}: {e}", LogColor.RED
            )

    def load_presets(self):
        """Load settings and session data from preset file"""
        presets_file = os.path.join(self.siril.get_siril_configdir(), PRESET_FILE)
        try:
            presets = None
            if os.path.exists(presets_file):
                self.siril.log(
                    f"Loading preset from {presets_file}",
                    LogColor.GREEN,
                )
                with open(presets_file) as f:
                    presets = json.load(f)
            else:
                # If default presets don't exist, use defaults
                presets = json.loads(PRESET_JSON)
                self.siril.log(
                    "No presets file found.",
                    LogColor.GREEN,
                )

            self.archive_dir = presets.get("archive_dir", None)

        except Exception as e:
            self.siril.log(
                f"Error loading presets from {presets_file}: {str(e)}", LogColor.RED
            )

    def session_selected(self):
        """collect the selected sessions"""
        selection = self.session_listbox.selectedItems()
        lights_count = 0
        self.selected_sessions = []
        if self.selected_target is None:
            return
        for item in selection:
            i = self.session_list.index(item.text())
            if self.selected_target.sessions is None:
                continue
            self.selected_sessions.append(self.selected_target.sessions[i])
            lights_count += self.selected_target.sessions[i].stacked
        self.progress_label.setText(f"0 / {lights_count}")
        if lights_count > 0:
            self.copy_btn.setDisabled(False)
        else:
            self.copy_btn.setDisabled(True)

    def target_selected(self):
        """chose the selected target and update the session list"""
        selection = self.target_listbox.selectedItems()
        self.session_listbox.clear()
        self.session_list = []
        self.selected_target = None
        self.info_box.clear()
        self.thumb.clear()
        if len(selection) == 1:
            i = self.target_list.index(selection[0].text())
            self.selected_target = self.targets[i]
            for s in self.selected_target.sessions:
                self.session_list.append(
                    f"{s.calendardate}: {s.stacked} x {s.exposure}s / {s.gain}g, {s.ir}, {s.min_temp}°C - {s.max_temp}°C"
                )
            self.session_listbox.addItems(self.session_list)
            first = self.selected_target.sessions[0]
            self.info_box.setText(
                f"{first.target}\n\nRA:\t{first.ra:.2f}\nDEC:\t{first.dec:.2f}\n"
            )
            thumb = QPixmap(first.thumbnail).scaledToHeight(self.thumb.height())
            self.thumb.setPixmap(thumb)

    def parse_foldername(self, path):
        """Helper function parsing folder names according to usualy Dwarf Lab naming scheme"""
        foldername = os.path.basename(path)
        parts = foldername.split("_")
        if "RESTACKED" in parts:
            # RESTACK sessions have a different format
            self.siril.log(
                f"skip MegaStack of {parts[3]} - {parts[4]} - {parts[5]}",
                LogColor.BLUE,
            )
        else:
            if len(parts) != 9:
                self.siril.log(
                    f"Path name '{os.path.basename(path)}' does not match the expected pattern",
                    LogColor.BLUE,
                )
            else:
                session = Session(
                    path, parts[2], parts[3], parts[5], parts[7], parts[8]
                )
                # add more info from shotsinfo.json if avaliable
                if os.path.isfile(os.path.join(path, "shotsinfo.json")):
                    with open(path + os.sep + "shotsinfo.json") as fp:
                        info = json.load(fp)
                        session.ir = info["ir"]
                        session.taken = info["shotsTaken"]
                        session.stacked = info["shotsStacked"]
                        session.ra = info["RA"]
                        session.dec = info["DEC"]
                        session.min_temp = info["minTemp"]
                        session.max_temp = info["maxTemp"]
                if not os.path.isfile(session.thumbnail):
                    session.thumbnail = ""
                return session
        return None

    def parse_tick(self, _, path):
        session = self.parse_foldername(path)
        if session is not None:
            self.sessions.append(session)

    def parse_done(self, cnt):
        self.targets = self.get_targets(self.sessions)
        self.refresh_target_list()

    def parse_filetree(self, path):
        """walk the archive path and collect dir names matching dwarf naming schema"""
        if path is None:
            return []
        path = os.path.abspath(path)
        self.siril.log(
            f"Parsing '{path}'",
            LogColor.BLUE,
        )

        self.sessions = []
        self.parserThread = TreeWalkThread(path=path)
        self.parserThread.parse_tick.connect(self.parse_tick)
        self.parserThread.parse_done.connect(self.parse_done)
        self.parserThread.start()

    def get_targets(self, sessions):
        """inspect the collected sessions and retrieve a list of targets"""
        targets = []

        sessions_by_target = sorted(sessions, key=lambda s: (s.target, s.cam, s.date))
        last = ""
        target = None
        for session in sessions_by_target:
            if session is None:
                continue
            if session.target != last:
                last = session.target
                target = Target(session.target)
                targets.append(target)
            if target is not None:
                target.add_session(session)

        return targets

    def refresh_target_list(self):
        """action for refresh button, rescan the archive"""
        self.target_listbox.clear()  # clear QListWidget instead of delete()
        self.target_list = []
        self.siril.log(f"Showing {self.archive_dir}", LogColor.BLUE)

        for t in self.targets:
            self.target_list.append(
                f"{t.target} ({len(t.sessions)}x {t.sessions[0].cam.capitalize()})"
            )
        self.target_listbox.addItems(self.target_list)


def main():
    try:
        app = QApplication(sys.argv)
        window = PreprocessingInterface()
        # Only show window if initialization was successful
        if window.initialization_successful:
            window.show()
            sys.exit(app.exec())
        else:
            # User canceled during initialization - exit gracefully
            sys.exit(0)
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
