"""
Workflow Summarizer Script
(c) Adrian Knagg-Baugh 2025
based on an idea by Adrian Nowik
SPDX-License-Identifier: GPL-3.0-or-later
Version: 1.3.0

This script uses Google Gemini to process the Siril log file and generate a
summary of the workflow. The use case is to provide a complete and easily-
readable documentation of a workflow you have just completed so that it can
be replicated on other images. Although individual operations are saved in
HISTORY FITS header cards, this script can provide a higher-level overview of
workflows involving combination of multiple images, e.g. star separation or
separate processing and combination of multiple narrowband filters. Note that
you will need a Google Gemini API key and usage will be subject to your free or
paid tier token limits. For very long logs, the script will upload the log as
a file attachment for more efficient processing, however extremely long logs
may still exceed Google Gemini's input token limit.
If you have set Siril's language preference, the output will be requested in
that language; if the preference is not set, output will default to English.

*** NOTE: ***
The minimum version required for this script has been updated to sirilpy 1.1.0
(i.e. the 1.5 development branch). Unfortunately it was not possible to update 
the 1.4 branch to support accurate image processing logging in time, so this
is left for the next major version.

"""

import sys
import os
import tempfile
import time
import sirilpy as s
s.ensure_installed("PyQt6", "google-genai")
version_ok = s.check_module_version(">=1.1.0")

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QTextEdit, QFileDialog, QMessageBox, QProgressBar,
                             QComboBox, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices, QFont
import configparser
from google import genai
from google.genai import types

# Token estimation: roughly 4 characters per token
CHARS_PER_TOKEN = 4
# Use file upload for logs longer than 50k tokens (~200k characters)
FILE_UPLOAD_THRESHOLD = 200000

def build_prompt(level, language=None, text=None, use_file_upload=False, for_preview=False, custom_prompt=None):
    """
    Build the prompt for Gemini API.

    Args:
        level: Either "highlevel", "detailed", or "custom"
        language: Target language for the summary (e.g., "English")
        text: The actual log text to include (ignored if use_file_upload=True)
        use_file_upload: Whether the log will be uploaded as a file
        for_preview: If True, show placeholder variables; if False, populate with actual values
        custom_prompt: Custom prompt text (only used when level="custom")

    Returns:
        The complete prompt string
    """
    # Handle custom prompt
    if level == "custom":
        if custom_prompt:
            return custom_prompt
        else:
            # Return a default custom prompt template for preview
            return "Enter your custom prompt here. The log will be automatically appended or uploaded as a file."

    # Build components with conditional variable substitution
    if for_preview:
        lang_placeholder = "{language}"
        text_placeholder = "{text}"
    else:
        lang_placeholder = language if language else "English"
        text_placeholder = text if text else ""

    upload_text = "I have uploaded a Siril log from my astrophotography processing workflow.\n\n" if use_file_upload else ""

    undo_instructions = (
        f"IMPORTANT: Before summarizing, you must first identify and remove any operations that were undone. "
        f"When you see an 'Undo' command, it cancels the operation that came immediately before it. "
        f"Both the undone operation AND the Undo command itself should be excluded from your summary.\n\n"
        f"After filtering out undone operations, please provide in {lang_placeholder} language:\n"
    )

    if "highlevel" in level:
        summary_type = (
            f"- A chronological high-level summary of each remaining processing step\n"
            f"- Do not include the parameters used or their values\n"
        )
    else:
        summary_type = (
            f"- A chronological detailed summary of each remaining processing step\n"
            f"- Include the parameters used and their values where applicable\n"
        )

    format_instructions = (
        f"- Format as a structured markdown document\n"
        f"- Do not include citation references\n\n"
    )

    # Build final prompt
    prompt = upload_text + undo_instructions + summary_type + format_instructions

    # Add text content if not using file upload and not for preview
    if not use_file_upload and not for_preview:
        prompt += text_placeholder
    elif for_preview and not use_file_upload:
        prompt += text_placeholder

    return prompt

class WorkflowWorker(QThread):
    """Worker thread to process the Siril workflow without freezing the GUI"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, api_key, siril, level, custom_prompt=None):
        super().__init__()
        self.api_key = api_key
        self.siril = siril
        self.level = level
        self.custom_prompt = custom_prompt
        self.uploaded_file = None

    def run(self):
        try:
            # Configure API with the new client
            client = genai.Client(api_key=self.api_key)

            # Get log and language from the passed Siril interface
            self.progress.emit("Retrieving Siril log...")
            text = self.siril.get_siril_log()
            language = self.siril.get_siril_config("core", "lang")
            if "not set" in language:
                language = "English"

            # Check log length
            log_length = len(text)
            estimated_tokens = log_length // CHARS_PER_TOKEN
            self.progress.emit(f"Log size: {log_length:,} characters (~{estimated_tokens:,} tokens)")

            # Determine whether to use file upload
            use_file_upload = log_length > FILE_UPLOAD_THRESHOLD

            # Build prompt using the unified function
            prompt = build_prompt(
                level=self.level,
                language=language,
                text=text,
                use_file_upload=use_file_upload,
                for_preview=False,
                custom_prompt=self.custom_prompt
            )

            # For custom prompts, we need to append the text if not using file upload
            if self.level == "custom" and not use_file_upload:
                prompt = prompt + "\n\n" + text

            # Submit the prompt
            if use_file_upload:
                self.progress.emit("Log is large, uploading as file...")

                # Create a temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                                 delete=False, encoding='utf-8') as tmp_file:
                    tmp_file.write(text)
                    tmp_file_path = tmp_file.name

                try:
                    # Upload the file using the new API
                    self.progress.emit("Uploading to Gemini...")
                    with open(tmp_file_path, 'rb') as f:
                        self.uploaded_file = client.files.upload(
                            file=f,
                            config=types.UploadFileConfig(
                                display_name="siril_log.txt",
                                mime_type="text/plain"
                            )
                        )

                    # Wait for file to be processed
                    self.progress.emit("Waiting for file processing...")
                    while self.uploaded_file.state == types.FileState.PROCESSING:
                        time.sleep(1)
                        self.uploaded_file = client.files.get(name=self.uploaded_file.name)

                    if self.uploaded_file.state == types.FileState.FAILED:
                        raise Exception("File processing failed")

                    self.progress.emit("Generating summary...")
                    # Generate response with file using the new API
                    response = client.models.generate_content(
                        model='gemini-flash-latest',
                        contents=[
                            types.Part.from_uri(
                                file_uri=self.uploaded_file.uri,
                                mime_type=self.uploaded_file.mime_type
                            ),
                            prompt
                        ]
                    )

                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass

                    # Delete uploaded file from Gemini
                    if self.uploaded_file:
                        try:
                            client.files.delete(name=self.uploaded_file.name)
                        except:
                            pass
            else:
                self.progress.emit("Generating summary...")
                # Direct generation without file upload
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )

            # Extract the text from response
            result = response.text

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

class SirilSummaryGUI(QMainWindow):
    def __init__(self, siril):
        super().__init__()
        self.siril = siril
        self.markdown_content = ""
        self.config_file = os.path.join(os.path.expanduser("~"), ".siril_workflow_summarizer.ini")

        self.initUI()

        # Load saved API key if it exists
        saved_key = self.load_api_key()
        if saved_key:
            self.api_key_input.setText(saved_key)

        # Load saved custom prompt if it exists
        saved_custom_prompt = self.load_custom_prompt()
        if saved_custom_prompt:
            self.saved_custom_prompt = saved_custom_prompt

    def load_api_key(self):
        """Load the API key from the config file"""
        try:
            if os.path.exists(self.config_file):
                config = configparser.ConfigParser()
                config.read(self.config_file)
                if 'API' in config and 'key' in config['API']:
                    return config['API']['key']
        except Exception:
            pass
        return None

    def save_api_key(self, api_key):
        """Save the API key to the config file"""
        try:
            config = configparser.ConfigParser()
            if os.path.exists(self.config_file):
                config.read(self.config_file)

            if 'API' not in config:
                config['API'] = {}

            config['API']['key'] = api_key

            with open(self.config_file, 'w') as f:
                config.write(f)

            return True
        except Exception:
            return False

    def load_custom_prompt(self):
        """Load the saved custom prompt from the config file"""
        try:
            if os.path.exists(self.config_file):
                config = configparser.ConfigParser()
                config.read(self.config_file)
                if 'Prompts' in config and 'custom' in config['Prompts']:
                    return config['Prompts']['custom']
        except Exception:
            pass
        return None

    def save_custom_prompt(self, custom_prompt):
        """Save the custom prompt to the config file"""
        try:
            config = configparser.ConfigParser()
            if os.path.exists(self.config_file):
                config.read(self.config_file)

            if 'Prompts' not in config:
                config['Prompts'] = {}

            config['Prompts']['custom'] = custom_prompt

            with open(self.config_file, 'w') as f:
                config.write(f)

            return True
        except Exception:
            return False

    def update_prompt_preview(self):
        """Update the prompt preview based on selected level"""
        level = self.level_combo.currentData()

        if level == "custom":
            # Enable editing for custom prompts
            self.prompt_preview.setReadOnly(False)

            # Check if we have a saved custom prompt
            if hasattr(self, 'saved_custom_prompt') and self.saved_custom_prompt:
                self.prompt_preview.setPlainText(self.saved_custom_prompt)
            else:
                # Show the default custom prompt template
                preview = build_prompt(level, for_preview=True)
                self.prompt_preview.setPlainText(preview)
        else:
            # Disable editing for pre-defined prompts
            self.prompt_preview.setReadOnly(True)

            # Show the preview with placeholders
            preview = build_prompt(level, for_preview=True)
            self.prompt_preview.setPlainText(preview)

    def initUI(self):
        self.setWindowTitle("Siril Workflow Summarizer")
        self.setGeometry(100, 100, 900, 800)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)

        # Title
        title = QLabel("Siril Workflow Summarizer")
        title.setStyleSheet("font-size: 18pt; font-weight: bold; margin: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # API Key section
        api_group = QGroupBox("Google Gemini API Configuration")
        api_layout = QVBoxLayout()

        # API key input
        key_layout = QHBoxLayout()
        key_label = QLabel("API Key:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter your Google Gemini API key")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)

        get_key_btn = QPushButton("Get API Key")
        get_key_btn.clicked.connect(self.open_api_key_url)
        get_key_btn.setMaximumWidth(120)

        key_layout.addWidget(key_label)
        key_layout.addWidget(self.api_key_input)
        key_layout.addWidget(get_key_btn)
        api_layout.addLayout(key_layout)

        # Info label
        info_label = QLabel("Your API key will be saved locally and used for future sessions.")
        info_label.setStyleSheet("color: gray; font-size: 9pt;")
        api_layout.addWidget(info_label)

        api_group.setLayout(api_layout)
        layout.addWidget(api_group)

        # Summary level section
        level_group = QGroupBox("Summary Detail Level")
        level_layout = QVBoxLayout()

        self.level_combo = QComboBox()
        self.level_combo.addItem("High-Level Overview", "highlevel")
        self.level_combo.addItem("Detailed (with parameters)", "detailed")
        self.level_combo.addItem("Custom Prompt", "custom")
        self.level_combo.currentIndexChanged.connect(self.update_prompt_preview)

        level_layout.addWidget(self.level_combo)
        level_group.setLayout(level_layout)
        layout.addWidget(level_group)

        # Prompt preview section
        prompt_group = QGroupBox("Prompt Preview")
        prompt_layout = QVBoxLayout()

        preview_info = QLabel("This shows the prompt that will be sent to Gemini. "
                             "For custom prompts, you can edit the text below:")
        preview_info.setWordWrap(True)
        preview_info.setStyleSheet("color: gray; font-size: 9pt;")
        prompt_layout.addWidget(preview_info)

        self.prompt_preview = QTextEdit()
        self.prompt_preview.setReadOnly(True)
        self.prompt_preview.setMaximumHeight(150)
        self.prompt_preview.setStyleSheet("background-color: #202020; font-family: monospace; font-size: 9pt;")

        prompt_layout.addWidget(self.prompt_preview)
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)

        # Initialize prompt preview
        self.update_prompt_preview()

        # Generate button
        self.generate_btn = QPushButton("Generate Summary")
        self.generate_btn.clicked.connect(self.generate_summary)
        self.generate_btn.setMinimumHeight(40)
        layout.addWidget(self.generate_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        layout.addWidget(self.progress_bar)

        # Result section
        result_label = QLabel("Workflow Summary:")
        result_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(result_label)

        # Rich text display
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("Generated summary will appear here...")
        layout.addWidget(self.result_text)

        # Button layout
        button_layout = QHBoxLayout()

        self.save_btn = QPushButton("Save Markdown")
        self.save_btn.clicked.connect(self.save_markdown)
        self.save_btn.setEnabled(False)
        self.save_btn.setMinimumHeight(35)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_results)
        self.clear_btn.setMinimumHeight(35)

        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.clear_btn)
        layout.addLayout(button_layout)

        # Status bar
        self.statusBar().showMessage("Ready")

    def open_api_key_url(self):
        """Open the Google AI Studio page to get an API key"""
        url = QUrl("https://aistudio.google.com/app/apikey")
        QDesktopServices.openUrl(url)

    def generate_summary(self):
        """Start the workflow summary generation"""
        api_key = self.api_key_input.text().strip()

        if not api_key:
            QMessageBox.warning(self, "API Key Required",
                              "Please enter your Google Gemini API key.")
            return

        # Save the API key to config file
        if self.save_api_key(api_key):
            self.statusBar().showMessage("API key saved", 2000)

        # Get selected level
        level = self.level_combo.currentData()

        # Handle custom prompt
        custom_prompt = None
        if level == "custom":
            custom_prompt = self.prompt_preview.toPlainText().strip()
            if not custom_prompt:
                QMessageBox.warning(self, "Empty Custom Prompt",
                                  "Please enter a custom prompt before generating.")
                return
            # Save the custom prompt
            if self.save_custom_prompt(custom_prompt):
                self.statusBar().showMessage("Custom prompt saved", 2000)

        # Disable controls during processing
        self.generate_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.level_combo.setEnabled(False)
        self.prompt_preview.setReadOnly(True)
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage("Starting...")

        # Start worker thread
        self.worker = WorkflowWorker(api_key, self.siril, level, custom_prompt)
        self.worker.finished.connect(self.on_generation_complete)
        self.worker.error.connect(self.on_generation_error)
        self.worker.progress.connect(self.on_progress_update)
        self.worker.start()

    def on_progress_update(self, message):
        """Update status bar with progress messages"""
        self.statusBar().showMessage(message)

    def on_generation_complete(self, result):
        """Handle successful generation"""
        self.markdown_content = result
        self.result_text.setMarkdown(result)

        # Re-enable controls
        self.generate_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.level_combo.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Update prompt preview readonly state based on current level
        level = self.level_combo.currentData()
        if level == "custom":
            self.prompt_preview.setReadOnly(False)

        self.statusBar().showMessage("Workflow summary generated successfully!")

    def on_generation_error(self, error_msg):
        """Handle generation errors"""
        self.generate_btn.setEnabled(True)
        self.level_combo.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Update prompt preview readonly state based on current level
        level = self.level_combo.currentData()
        if level == "custom":
            self.prompt_preview.setReadOnly(False)

        self.statusBar().showMessage("Error occurred")

        QMessageBox.critical(self, "Error",
                           f"An error occurred while generating the summary:\n\n{error_msg}")

    def save_markdown(self):
        """Save the markdown content to a file"""
        if not self.markdown_content:
            QMessageBox.warning(self, "No Content",
                              "There is no content to save. Please generate a summary first.")
            return

        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Markdown File",
            "siril_workflow_summary.md",
            "Markdown Files (*.md);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.markdown_content)

                self.statusBar().showMessage(f"Saved to {file_path}")
                QMessageBox.information(self, "Success",
                                      f"Markdown file saved successfully to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error",
                                   f"Failed to save file:\n\n{str(e)}")

    def clear_results(self):
        """Clear the results display"""
        self.result_text.clear()
        self.markdown_content = ""
        self.save_btn.setEnabled(False)
        self.statusBar().showMessage("Cleared")

def main():
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    if not version_ok:
        print("Error: sirilpy version requirement not met, aborting...")
        exit()
    # Initialize Siril interface once
    try:
        siril = s.SirilInterface()
        siril.connect()
    except Exception as e:
        QMessageBox.critical(None, "Siril Connection Error",
                           f"Failed to connect to Siril:\n\n{str(e)}\n\n"
                           "Please ensure Siril is running and try again.")
        sys.exit(1)

    window = SirilSummaryGUI(siril)
    window.show()

    sys.exit(app.exec())

if __name__ == '__main__':
    main()
