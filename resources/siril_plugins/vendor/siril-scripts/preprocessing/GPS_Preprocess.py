"""
**Preprocessing script using sirilpy**

This script executes siril commands to calibrate, background extract(optional), platesolve(optional), register and stack subs. It accepts arguments to define working directory, filters, feathering and drizzling setting. It can run headlessly i.e. without the siril UI open, so it can run on a server or cloud instance.

With the platesolving option the script can register mosaics and with multi session it can calibrate subs from mulitple session and combine to a single stack.

The script can be started directly from the siril GUI, siril command line and can also be started from a ssf script using the 'pyscript' command i.e. pyscript GPS_Preprocess.py. Such a ssf script can run the this script several times i.e. to try different settings, with unique result files based on setting values. 

The script also skips restacking master biases, flats, darks, background exaction and platesolving if they already exist. 

Example ssf script to run this script
---
requires 1.3.6
pyscript GPS_Preprocess.py -d <your-workspace-directory-path> -b 95% -r 85% -w 85% -z 1 -ps -bg
---

Preprocessing for Siril
from Graham Smith (2025, 2026)

SPDX-License-Identifier: GPL-3.0-or-later
-----
0.1.3	Initial submittal for merge request
0.1.4   Added nbstars as an optional filter. 
0.1.5   Adds multi-session support to calibrate each session separately
0.1.6   Handle working directory with spaces
0.1.7   Added tooltips
0.1.8   Adds satellite removal using setiastropro AI4
0.1.9	Add GUI to create excutable config file 
 
"""

import sys
import os
import subprocess
import sirilpy as s
import argparse
import re
import shlex

VERSION = "0.1.9"
	
# PyQt6 for GUI
try:
	from PyQt6.QtWidgets import (
		QApplication, QDialog, QLabel, QLineEdit, QPushButton, QCheckBox, QToolTip, QGroupBox,
		QVBoxLayout, QHBoxLayout, QFormLayout, QDialogButtonBox, QMessageBox, QTextEdit, QFileDialog
	)
except ImportError:
	# Silently fail if PyQt6 is not installed, as it's optional for CLI mode.
	pass

def create_execute_config(config_file_path, ):
	utility = config_file_path.split('_')[1].split('.')[0]
	start_dir = ""  # e.g., "/home/user" or "" for default
	file_filter = "All Files (*)"
	file_path, _ = QFileDialog.getOpenFileName(parent=None, caption=f"Select the executable for {utility}", directory=start_dir, filter=file_filter)
	
	if file_path:
		print("Selected file:", file_path)
		with open(config_file_path, 'w') as file:
			file.write(file_path)
		sys.exit(0)
	else:
		print("No file selected")
		sys.exit(0)	

def master_bias(bias_dir, process_dir):
	if os.path.exists(os.path.join(workdir, 'process/bias_stacked.fit')) or os.path.exists(os.path.join(workdir, 'process/bias_stacked.fit.fz')):
		print('master bias exists, skipping')
		return
	else:
		siril.cmd(f"cd {bias_dir}")
		siril.cmd(f"convert bias -out={process_dir}")
		siril.cmd(f"cd  {process_dir}")
		siril.cmd(f"stack bias rej 3 3 -nonorm")

def master_flat(flat_dir, process_dir):
	if os.path.exists(os.path.join(workdir, 'process/pp_flat_stacked.fit')) or os.path.exists(os.path.join(workdir, 'process/pp_flat_stacked.fit.fz')):
		print('master flat exists, skipping')
		return
	else:
		siril.cmd(f"cd {flat_dir}")
		siril.cmd(f"convert flat -out={process_dir}")
		siril.cmd(f"cd {process_dir}")
		siril.cmd("calibrate flat -bias=bias_stacked")
		siril.cmd("stack pp_flat rej 3 3 -norm=mul")

def master_dark(dark_dir, process_dir):
	if os.path.exists(os.path.join(workdir, 'process/dark_stacked.fit')) or os.path.exists(os.path.join(workdir, 'process/dark_stacked.fit.fz')):
		print('master dark exists, skipping')
		return
	else:
		siril.cmd(f"cd {dark_dir}")
		siril.cmd(f"convert dark -out={process_dir}")
		siril.cmd(f"cd {process_dir}")
		siril.cmd(f"stack dark rej 3 3 -nonorm")

def light(light_dir, process_dir):
	if os.path.exists(os.path.join(workdir, 'process/pp_light_.seq')):
		print('pp_light exists, skipping')
		return
	else:
		siril.cmd(f"cd {light_dir}")
		siril.cmd(f"convert light -out={process_dir}")
		siril.cmd(f"cd {process_dir}")
		siril.cmd(
			f"calibrate light -dark=dark_stacked -flat=pp_flat_stacked -cc=dark -cfa -equalize_cfa")

def light_nc(light_dir, process_dir):
	if os.path.exists(os.path.join(workdir, 'process/pp_light_.seq')):
		print('pp_light exists, skipping')
		return
	else:
		siril.cmd(f"cd {light_dir}")
		siril.cmd(f"convert pp_light -out={process_dir}")

def bkg_extract(process_dir):
	if os.path.exists(os.path.join(workdir, 'process/bkg_pp_light_.seq')):
		print('background extracted, skipping')
		return
	else:
		siril.cmd(f"cd {process_dir}")
		siril.cmd(f"seqsubsky pp_light 1 -samples=10")

def platesolve(process_dir):
	if args.bkg:
		if os.path.exists(os.path.join(workdir, 'process/r_bkg_pp_light_.seq')):
			print('sequence platesolved, skipping')
			return
	else:
		if os.path.exists(os.path.join(workdir, 'process/r_pp_light_.seq')):
			print('sequence platesolved, skipping')
			return
	siril.cmd("cd " + process_dir)
	if args.platesolve == True :
		focal = " "
	else:
		focal = (f"-focal={args.platesolve}")		
	siril.cmd(f"seqplatesolve {light_seq} -nocache -catalog=nomad -force -disto=ps_distortion {focal}")

def register(process_dir):
	siril.cmd(f"cd {process_dir}")
	flat = " " if args.no_calibration else " -flat=pp_flat_stacked"
	if args.platesolve:
		siril.cmd(
			f"seqapplyreg {light_seq} -framing=max -filter-bkg={bkg} -filter-nbstars={stars} -filter-round={roundf} -filter-wfwhm={wfwhm} {drizzle} {flat}")
	else:
		siril.cmd(f"register {light_seq} -2pass")
		siril.cmd(
			f"seqapplyreg {light_seq} -filter-bkg={bkg} -filter-nbstars={stars} -filter-round={roundf} -filter-wfwhm={wfwhm} {drizzle} {flat}")

def satellite_removal(workdir):
	os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
	config_dir = siril.get_siril_configdir()
	if os.path.isfile (f"{config_dir}/sirilcc_saspro.conf"):
		config_file_path = (f"{config_dir}/sirilcc_saspro.conf")
		with open(config_file_path, 'r') as file:
			executable_path = file.readline().strip()
			python_path = executable_path.replace("setiastrosuitepro", "python")
	else:
		if len(sys.argv) == 1:
			create_execute_config(f"{config_dir}/sirilcc_saspro.conf")
		else:	
			print(f"Executable not configured. Please create file 'sirilcc_saspro.conf' in your siril config directory {config_dir} with a line containing the path to setiastrosuitepro.")
			sys.exit(1)

	os.chdir(workdir)		
	os.rename('lights', 'lights_preremoval')
	os.mkdir('lights')
	os.chdir('lights_preremoval')
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')):
			siril.log(f"Running satellite trail removal on {image}")
			newimage = (f"{os.path.splitext(image)[0]}.fit")
			cmd = [python_path, executable_path, "cc", "satellite", "--gpu", "--mode", "full", "--clip-trail", "-i", f"{image}", "-o", f"{workdir}/lights/{newimage}"]
			
			my_env = os.environ.copy()
			my_env.pop("PYTHONPATH", None)
			process = subprocess.Popen(cmd, shell=False, env=my_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
			percent_re = re.compile(r"(\d+\.?\d*)" + "%")
			if process.stdout:
				for line in iter(process.stdout.readline, ''):
					line = line.strip()
					if not line:
						continue
					m = percent_re.search(line)
					if m:
						try:
							pct = float(m.group(1))
							siril.update_progress(f"Satellite trail removal", pct / 100.0)
						except ValueError:
							siril.log(line)
					else:
							siril.log(line)
			process.wait()

def stack(process_dir):
	siril.cmd(f"cd {process_dir}")
	siril.cmd("load pp_light_00001")
	try:
		obj = (siril.get_image_fits_header(return_as='dict')['OBJECT']).replace(" ", "")
	except KeyError:
		obj = ("")
	siril.cmd(f"stack r_{light_seq} rej 3 3 -norm=addscale -output_norm -rgb_equal -maximize -filter-included -weight=wfwhm  -feather={feather} -out=../{obj}_b{bkg}-s{stars}-r{roundf}-w{wfwhm}-z{drizzle_scale}-f{feather}-$LIVETIME:%d$s")
	siril.cmd("close")

# ==============================================================================
# GUI Mode
# ==============================================================================

def run_gui():
	if 'QApplication' not in globals():
		print("PyQt6 is not installed. Please install it to use the GUI.")
		print("pip install PyQt6")
		return

	class PreprocessingDialog(QDialog):
		def __init__(self, parent=None):
			super().__init__(parent)
			self.setWindowTitle("Siril Preprocessing")
			self.setFixedWidth(int(self.width() * 1.10))

			layout = QVBoxLayout(self)
			form_layout = QFormLayout()

			try:
				self.workdir_input = QLineEdit(os.getcwd())
				form_layout.addRow("Working Directory:", self.workdir_input)
			except Exception as e:
				print("Working directory does not exist:", e)
				sys.exit(1)

			self.drizzle = QCheckBox("Drizzle (scale), enable for OSC")
			self.drizzle_input = QLineEdit("1")
			self.drizzle_input.setToolTip("File size inceases by square of value")
			self.drizzle_input.setEnabled(False)
			self.drizzle.toggled.connect(self.drizzle_input.setEnabled)
			drizzle_layout = QHBoxLayout()
			drizzle_layout.addWidget(self.drizzle_input)
			form_layout.addRow(self.drizzle, drizzle_layout)
			
			self.satellite_cb = QCheckBox("Satellite trail removal")
			form_layout.addRow(self.satellite_cb)

			self.no_calibration_cb = QCheckBox("No Calibration")
			form_layout.addRow(self.no_calibration_cb)

			self.bkg_extract_cb = QCheckBox("Background extraction")
			form_layout.addRow(self.bkg_extract_cb)
			
			filter_group = QGroupBox("Quality filters")
			filter_form = QFormLayout(filter_group)
						
			self.background_input = QLineEdit("100%")
			filter_form.addRow("Background filter (XX% or X):", self.background_input)
			
			self.round_input = QLineEdit("100%")
			filter_form.addRow("Round filter (XX% or X):", self.round_input)

			self.stars_input = QLineEdit("100%")
			filter_form.addRow("Number of stars filter (XX% or X):", self.stars_input)
				
			self.wfwhm_input = QLineEdit("100%")
			filter_form.addRow("wFWHM filter (XX% or X):", self.wfwhm_input)

			form_layout.addRow(filter_group)

			self.platesolve_cb = QCheckBox("Platesolve, optionally provide focal lenght")
			self.platesolve_input = QLineEdit()
			self.platesolve_input.setToolTip("Override value in fits header")
			self.platesolve_input.setEnabled(False)
			self.platesolve_cb.toggled.connect(self.platesolve_input.setEnabled)
			platesolve_layout = QHBoxLayout()
			platesolve_layout.addWidget(self.platesolve_input)
			form_layout.addRow(self.platesolve_cb, platesolve_layout)

			self.feather_input = QLineEdit("0")
			self.feather_input.setToolTip("Apply feather mask on image borders in 'n' pixels")
			form_layout.addRow("Feathering (px):", self.feather_input)

			self.multi_group = QGroupBox("Multi session calibration")
			self.multi_group.setCheckable(True)
			self.multi_group.setChecked(False)
			multi_layout = QVBoxLayout(self.multi_group)
			self.multi_input = QTextEdit()
			self.multi_input.setFixedHeight(100)
			self.multi_input.setToolTip("Add working directories here")
			multi_layout.addWidget(self.multi_input)
			form_layout.addRow(self.multi_group)

			layout.addLayout(form_layout)

			self.button_box = QDialogButtonBox(
				QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
			self.button_box.accepted.connect(self.accept)
			self.button_box.rejected.connect(self.reject)
			layout.addWidget(self.button_box)

		def get_values(self):
			return {
				"workdir": self.workdir_input.text(),
				"satellite": self.satellite_cb.isChecked(),
				"background": self.background_input.text(),
				"round": self.round_input.text(),
				"stars": self.stars_input.text(),
				"wfwhm": self.wfwhm_input.text(),
				"feather": self.feather_input.text(),
				"drizzle": self.drizzle_input.text() if self.drizzle.isChecked() else None,
				"bkg_extract": self.bkg_extract_cb.isChecked(),
				"platesolve": self.platesolve_input.text() if self.platesolve_cb.isChecked() and self.platesolve_input.text() else self.platesolve_cb.isChecked(),
				"no_calibration": self.no_calibration_cb.isChecked(),
				"multi": shlex.split(self.multi_input.toPlainText()) if self.multi_group.isChecked() else False,
			}

	app = QApplication.instance() or QApplication(sys.argv)
	
	dark_stylesheet = """
		QWidget {
			background-color: #2b2b2b;
			color: #efefef;
		}
		QLineEdit, QTextEdit {
			background-color: #3b3b3b;
			color: #efefef;
			border: 1px solid #555;
			padding: 2px;
		}
		QPushButton {
			background-color: #4b4b4b;
			color: #efefef;
			border: 1px solid #555;
			padding: 5px;
			border-radius: 3px;
		}
		QPushButton:hover {
			background-color: #5b5b5b;
		}
		QPushButton:pressed {
			background-color: #3b3b3b;
		}
		QCheckBox {
			spacing: 5px;
		}
		QCheckBox::indicator {
			width: 15px;
			height: 15px;
			background-color: #3b3b3b;
			border: 1px solid #555;
		}
		QCheckBox::indicator:checked {
			background-color: #4b9ee3;
		}
	"""
	app.setStyleSheet(dark_stylesheet)

	dialog = PreprocessingDialog()
	if dialog.exec():
		values = dialog.get_values()
		cli_args = []

		if values["workdir"]:
			cli_args.extend(["-d", values["workdir"]])
		if values["satellite"]:
			cli_args.append("-sr")
		if values["background"]:
			cli_args.extend(["-b", values["background"]])
		if values["round"]:
			cli_args.extend(["-r", values["round"]])
		if values["stars"]:
			cli_args.extend(["-s", values["stars"]])
		if values["wfwhm"]:
			cli_args.extend(["-w", values["wfwhm"]])
		if values["feather"]:
			cli_args.extend(["-f", values["feather"]])
		if values["drizzle"]:
			cli_args.extend(["-z", values["drizzle"]])
		if values["bkg_extract"]:
			cli_args.append("-bg")
		if values["platesolve"]:
			cli_args.append("-ps")
			if isinstance(values["platesolve"], str):
				cli_args.append(values["platesolve"])
		if values["no_calibration"]:
			cli_args.append("-nc")
		if values["multi"]:
			cli_args.append("-m")
			if isinstance(values["multi"], list):
				cli_args.extend(values["multi"])
			elif isinstance(values["multi"], str):
				cli_args.append(values["multi"])

		print(*cli_args)
		main_logic(cli_args)

		msg_box = QMessageBox()
		msg_box.setIcon(QMessageBox.Icon.Information)
		msg_box.setText("Processing finished.")
		msg_box.setWindowTitle("Success")
		msg_box.exec()

def main_logic(argv):
	global args, workdir, bkg, stars, roundf, wfwhm, drizzle, drizzle_scale, feather, light_seq

	parser = argparse.ArgumentParser()
	parser.add_argument("-b", "--background", nargs='+', help="background filter settings, XX%% or X")
	parser.add_argument("-bg", "--bkg", help="extract background", action="store_true")
	parser.add_argument("-d", "--workdir", nargs='+', help="set working directory")
	parser.add_argument("-f", "--feather", nargs='+', help="set feathering amount in px")
	parser.add_argument("-m","--multi_session", nargs='+', help="Calibrates multiple sessions, provide working directory for each session")	
	parser.add_argument("-nc", "--no_calibration", help="do not calibrate", action="store_true")
	parser.add_argument("-ps", "--platesolve", nargs='?', const=True, help="platesolve, optionally provide focal lenght")
	parser.add_argument("-r", "--round", nargs='+',	help="round filter settings, XX%% or X")
	parser.add_argument("-s", "--stars", nargs='+', help="# of stars filter settings, XX%% or X")
	parser.add_argument("-sr", "--satellite", help="satellite trail removal", action="store_true")	
	parser.add_argument("-v","--version", help="print the version and exit",action="store_true")
	parser.add_argument("-w", "--wfwhm", nargs='+',	help="wfwhm filter settings, XX%% or X")
	parser.add_argument("-z", "--drizzle", nargs='+', help="set drizzle scaling, required for OSC images")
	args = parser.parse_args(argv)

	if args.version:
		print('version ' + VERSION)
		sys.exit(1)

	bkg = (args.background[0]) if args.background else '100%'
	stars = (args.stars[0]) if args.stars else '100%'
	roundf = (args.round[0]) if args.round else '100%'
	wfwhm = (args.wfwhm[0]) if args.wfwhm else '100%'
	if args.drizzle:
		drizzle_scale = args.drizzle[0]
		pix_frac = str(1 / float(drizzle_scale))
		drizzle = f"-kernel=square -drizzle -scale={drizzle_scale} -pixfrac={pix_frac}"
	else:
		drizzle, drizzle_scale = " ", 0
	feather = args.feather[0] if args.feather else '0'

	try:
		siril.connect()
		siril.cmd("requires", "1.3.6")
		siril.log("Running preprocessing")
		workdir = args.workdir[0] if args.workdir else os.getcwd()
		# handle working directory with spaces
		olddir = None
		if ' ' in os.path.basename(workdir):
			olddir = workdir
			workdir = workdir.replace(" ", "_")
			os.rename(olddir, workdir)

		siril.cmd("cd", workdir)
		process_dir = os.path.join(workdir, 'process')
		siril.cmd("set32bits")
		siril.cmd("setext", "fit")

		if args.no_calibration:
			light_nc(os.path.join(workdir, 'lights'), process_dir)

		elif args.multi_session:
			w = 0
			n = len (args.multi_session)
			while n > w:
				workdir = args.multi_session[w]
				if not os.path.exists(workdir):
					print (f"Directory {workdir} not found")
					sys.exit(1)
				w +=1
			w = 0
			while n > w:
				workdir = args.multi_session[w]
				if args.satellite:
					satellite_removal(workdir)
				process_dir = os.path.join(workdir, 'process')
				master_bias(os.path.join(workdir, 'biases'), process_dir)
				master_flat(os.path.join(workdir, 'flats'), process_dir)
				master_dark(os.path.join(workdir, 'darks'), process_dir)
				light(os.path.join(workdir, 'lights'), process_dir)
				if w > 0:
					# find highest pp_light in 1st working directory/process
					for pp in os.listdir(os.path.join(args.multi_session[0], 'process')):
						if pp.startswith('pp_light_') and pp.endswith('.fit'):
							next = int(pp.split('_')[-1].split('.')[0])
					next += 1
					print(f"pp_light_{next}.fit")
					for pp in os.listdir(process_dir):
						if pp.startswith('pp_light_') and pp.endswith('.fit'):
							new = f"pp_light_{next:05}.fit"
							oldpath = os.path.join(args.multi_session[w], 'process', pp)
							newpath = os.path.join(args.multi_session[0], 'process', new)
							os.rename(oldpath, newpath)
							next += 1
				w +=1
			workdir = args.multi_session[0]
			process_dir = os.path.join(workdir, 'process')
		else:
			if args.satellite:
				satellite_removal(workdir)
			master_bias(os.path.join(workdir, 'biases'), process_dir)
			master_flat(os.path.join(workdir, 'flats'), process_dir)
			master_dark(os.path.join(workdir, 'darks'), process_dir)
			light(os.path.join(workdir, 'lights'), process_dir)

		if args.bkg:
			bkg_extract(process_dir)
			light_seq = 'bkg_pp_light'
		else:
			light_seq = 'pp_light'

		if args.platesolve:
			platesolve(process_dir)
		register(process_dir)
		stack(process_dir)

		if olddir:
			os.rename(workdir, olddir)
			siril.cmd("cd",f'"{olddir}"')
		else:
			siril.cmd("cd",f'"{workdir}"')

	except Exception as e:
		print("\n**** ERROR *** " + str(e) + "\n")
		if 'QApplication' in globals() and QApplication.instance():
			msg_box = QMessageBox()
			msg_box.setIcon(QMessageBox.Icon.Critical)
			msg_box.setText("An error occurred during processing.")
			msg_box.setInformativeText(str(e))
			msg_box.setWindowTitle("Error")
			msg_box.exec()

if __name__ == '__main__':
	siril = s.SirilInterface()

	if len(sys.argv) == 1:
		run_gui()
	else:
		main_logic(sys.argv[1:])
