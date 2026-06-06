# (c) Rich Stevenson - Deep Space Astro
# Replacement notice script

import sirilpy as s
s.ensure_installed("PyQt6")

import sys
from PyQt6.QtWidgets import QApplication, QMessageBox, QStyle

def main():
    # Connect to Siril
    siril = s.SirilInterface()
    try:
        siril.connect()
    except s.SirilConnectionError:
        sys.exit()

    # Create Qt app
    app = QApplication(sys.argv)

    # Show message box
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Information)
    msg.setWindowTitle("Notice")
    msg.setText(
        "This script has been deprecated & replaced by the \"Workflow Companion\" script.\n\n"
        "Please go to the Scripts menu, then Get Scripts, & select \"Workflow Companion\".\n\n"
        "\"Workflow Companion\" will then be available in the Utility menu."
    )
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    info_icon = app.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
    msg.setWindowIcon(info_icon)

    msg.exec()

    sys.exit()

if __name__ == "__main__":
    main()
