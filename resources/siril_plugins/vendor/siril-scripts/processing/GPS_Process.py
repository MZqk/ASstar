"""
**Processing script using sirilpy**

This script executes siril commands for background extraction, denoise and/or sharpening. It accepts arguments to define working directory, strengths, amounts etc. It can run headlessly i.e. without the siril UI open, so it can run on a server or cloud instance.

With the CosmicClarity and GraXpert the executables must be installed and configured using their sirilpy scripts 1st.

The script can be started directly from the siril GUI, siril command line and can also be started from a ssf script using the 'pyscript' command i.e. pyscript GPS_Process.py. Such a ssf script can run the script several times i.e. to try different settings. The commands will process all fit(s) files found in the supplied working directory.

Example ssf script to run this script
---
requires 1.3.6
pyscript GPS_Process.py -d <workspace> -dc 0.5
pyscript GPS_Process.py -d <workspace> -sc 0.5 0.5 5
---

Processing for Siril
from Graham Smith (2025, 2026)

SPDX-License-Identifier: GPL-3.0-or-later
-----
0.1.4	Initial submittal for merge request
0.1.5   Adds AutoBGE, Autostretch, Statistical Stretch and Multiple process file handling
0.1.6   Adds SetiAstroPro CC denoise and sharpen. Script now sharpens before denoise. 
0.1.7   Handle working directory with spaces. Show siril config path when config file is missing.
0.1.8   Set Cuda 'expandable_segments:True' for better GPU memory allocation
0.1.9	SPCC improvements
0.2.0	Adds cropping and better handling of siril and setiastro python version mismatch
0.2.1   Add option to separate starmask and starless processing, with starmask combine factor 
0.2.2   Add option to run synthstar on starmask and GUI improvements
"""

import sys
import os
import subprocess
import shutil
import sirilpy as s
import argparse
import re

VERSION = "0.2.2"

# PyQt6 for GUI
try:
	from PyQt6.QtWidgets import (
		QApplication, QComboBox, QDialog, QLabel, QLineEdit, QPushButton, QCheckBox, QToolTip, QScrollArea,
		QVBoxLayout, QHBoxLayout, QFormLayout, QDialogButtonBox, QMessageBox, QFrame, QGroupBox, QWidget
	)
except ImportError:
	# Silently fail if PyQt6 is not installed, as it's optional for CLI mode.
	pass

original_images = []
processed_images = []

# ==============================================================================
# Prototype sirilpy processing script
# ==============================================================================

def abe(workdir):
	os.chdir(workdir)
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:
			siril.log("Starting background extraction on " + image)
			siril.cmd("load", image)
			siril.cmd(f"pyscript AutoBGE.py -npoints {npoints} -polydegree {polydegree} -rbfsmooth {rbfsmooth}")
			newimage = (f"{(image).rsplit('.', 1)[0]}_ab{npoints}-{polydegree}-{rbfsmooth}")
			siril.cmd("save", newimage)
			processed_images.append(f"{image}")

def autostretch(workdir):
	os.chdir(workdir)
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:
			siril.log("Auto stretching " + image)
			siril.cmd("load", image)
			siril.cmd("autostretch -linked")
			newimage = (f"{os.path.splitext(image)[0]}_as")
			siril.cmd("save", newimage)
			processed_images.append(f"{image}")

def bkg(workdir):
	os.chdir(workdir)
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:
			siril.log("Starting background extraction on " + image)
			siril.cmd("load", image)
			siril.cmd("subsky -rbf -samples=20 -tolerance=1.0 -smooth=" + smooth)
			newimage = (f"{(image).rsplit('.', 1)[0]}_b{smooth}")
			siril.cmd("save", newimage)
			processed_images.append(f"{image}")
			
def bkg_GraX(workdir):
	os.chdir(workdir)
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:
			siril.log("Starting GraXpert background extraction on " + image)			
			siril.cmd("load", image)
			siril.cmd("pyscript GraXpert-AI.py -bge -smoothing " + bkgGraX)
			newimage = (f"{(image).rsplit('.', 1)[0]}_bg{bkgGraX}")
			siril.cmd("save", newimage)
			processed_images.append(f"{image}")

def crop(workdir):
	os.chdir(workdir)
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:
			for c in args.crop:
				siril.log(f"Cropping {image} by {c}%")			
				siril.cmd("load", image)
				x_crop = (float(siril.get_image_fits_header(return_as='dict')['NAXIS1']) * (float(c) / 100.0))
				y_crop = (float(siril.get_image_fits_header(return_as='dict')['NAXIS2']) * (float(c) / 100.0))
				x = (float(siril.get_image_fits_header(return_as='dict')['NAXIS1']) - (float(x_crop) * 2.0))			
				y = (float(siril.get_image_fits_header(return_as='dict')['NAXIS2']) - (float(y_crop) * 2.0))
				siril.cmd(f"crop {x_crop} {y_crop} {x} {y}")
				newimage = (f"{(image).rsplit('.', 1)[0]}_c{c}")
				siril.cmd("save", newimage)
				processed_images.append(f"{image}")				

def denoise(workdir):
	os.chdir(workdir)
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:
			siril.log("Starting denoise on " + image)
			siril.cmd("load", image)
			siril.cmd("denoise -indep -vst")
			newimage = (f"{os.path.splitext(image)[0]}_d")
			siril.cmd("save", newimage)
			processed_images.append(f"{image}")

def denoise_CC(workdir):
	compress = (siril.get_siril_config('compression','enabled'))
	config_dir = siril.get_siril_configdir()
	if os.path.isfile (f"{config_dir}/sirilcc_denoise.conf"):
		config_file_path = (f"{config_dir}/sirilcc_denoise.conf")
		with open(config_file_path, 'r') as file:
			executable_path = file.readline().strip()
			cc_input_dir = (executable_path.rsplit('/', 1)[0])+"/input"
			cc_output_dir = (executable_path.rsplit('/', 1)[0])+"/output"
	else:
		print("Executable not yet configured. It is recommended to use Seti Astro Cosmic Clarity v5.4 or higher.")
		sys.exit(1)

	os.chdir(cc_input_dir)
	for oldimage in os.listdir():
		os.remove(oldimage)
	os.chdir(workdir)

	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:
			siril.log(image)
			# CosmicClaritySuite does not support compressed fit files
			if image.endswith(('.fz')):
				if compress:
					siril.cmd("setcompress 0")
				siril.cmd("load", image)
				siril.cmd("save", image)
				image = os.path.splitext(image)[0]
				os.remove(f"{image}.fz")
				if compress:
					siril.cmd("setcompress 1")
			shutil.copy(image, cc_input_dir)			

			cmd = f"{executable_path} --denoise_mode {denoiseCC_mode} --denoise_strength {denoiseCC_strength} --separate_channels"
			process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')

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
							siril.update_progress(f"Denoise:", pct / 100.0)
						except ValueError:
							siril.log(line)
					else:
							siril.log(line)
			process.wait()
		
			os.chdir(cc_input_dir)
			for oldimage in os.listdir():
				os.remove(oldimage)
			os.chdir(cc_output_dir)
			for ccimage in os.listdir():
				newimage = (f"{(image).rsplit('.', 1)[0]}_dc{denoiseCC_mode}{denoiseCC_strength}.fit")		
				shutil.move(ccimage, (f"{workdir}/{newimage}"))
				processed_images.append(f"{image}")		
			os.chdir(workdir)
	
def denoise_GraX(workdir):
	os.chdir(workdir)
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:
			siril.log("Starting GraXpert denoise on " + image)
			siril.cmd("load", image)
			siril.cmd("pyscript GraXpert-AI.py -gpu -denoise -strength " + denoiseGraX)
			newimage = (f"{(image).rsplit('.', 1)[0]}_dg{denoiseGraX}")
			siril.cmd("save", newimage)
			processed_images.append(f"{image}")

def denoise_SA(workdir):
	os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
	config_dir = siril.get_siril_configdir()
	if os.path.isfile (f"{config_dir}/sirilcc_saspro.conf"):
		config_file_path = (f"{config_dir}/sirilcc_saspro.conf")
		with open(config_file_path, 'r') as file:
			executable_path = file.readline().strip()
			python_path = executable_path.replace("setiastrosuitepro", "python")
	else:
		print(f"Executable not configured. Please create file 'sirilcc_saspro.conf' in your siril config directory {config_dir} with a line containing the path to setiastrosuitepro.")
		sys.exit(1)
		
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:
			siril.log(image)
			newimage = (f"{(image).rsplit('.', 1)[0]}_dsa-{denoiseSA_mode}-{denoiseSA_luma_amount}-{denoiseSA_color_amount}.fit")
			cmd = [python_path, executable_path, "cc", "denoise", "--gpu", "--denoise-mode", f"{denoiseSA_mode}", "--denoise-luma", f"{denoiseSA_luma_amount}", "--denoise-color", f"{denoiseSA_color_amount}", "--separate-channels", "-i", f"{image}", "-o", f"{newimage}"]
			print(" ".join(cmd))
			
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
							siril.update_progress(f"Denoise", pct / 100.0)
						except ValueError:
							siril.log(line)
					else:
							siril.log(line)
			process.wait()
			processed_images.append(f"{image}")		
			
def multiprocess(workdir):
	os.chdir(workdir)
	base_directory = 'Processed_' 
	index = 1
	while True:
	    path = f"{base_directory}{index}"
	    if os.path.isdir(path):
	        index += 1  # Increment the index for the next iteration
	    else:
	        os.makedirs(f"{base_directory}{index}")
	        break
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in original_images:
			shutil.move(image, (f"{workdir}/{base_directory}{index}"))

def pixelmath(workdir):
	os.chdir(workdir)
	combine_factor = (args.starnet[1])
	for starmask in os.listdir():
		if starmask.startswith("starmask"):
			stars = starmask
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:	
			less = image
	siril.cmd(f"PM '${less}$ + (${stars}$ * {combine_factor}) / 1 + ${less}$ * ${stars}$'")
	newimage = f"{os.path.splitext(less.removeprefix('starless_'))[0]}_combined"
	siril.cmd("save", newimage)
	processed_images.append(f"{newimage}")

def sharpen(workdir):
	os.chdir(workdir)
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:
			siril.log("Starting sharpen on " + image)
			siril.cmd("load", image)
			siril.cmd("rl -gdstep=0.0003 -iters=40 -alpha=3000 -tv")
			siril.cmd("rl -gdstep=0.0002 -iters=40 -alpha=3000 -tv")
			siril.cmd("rl -gdstep=0.0001 -iters=40 -alpha=3000 -tv")
			newimage = (f"{os.path.splitext(image)[0]}_ss")
			siril.cmd("save", newimage)
			processed_images.append(f"{image}")

def sharpen_CC(workdir):
	compress = (siril.get_siril_config('compression','enabled'))
	config_dir = siril.get_siril_configdir()
	if os.path.isfile (f"{config_dir}/sirilcc_sharpen.conf"):
		config_file_path = (f"{config_dir}/sirilcc_sharpen.conf")
		with open(config_file_path, 'r') as file:
			executable_path = file.readline().strip()
			cc_input_dir = (executable_path.rsplit('/', 1)[0])+"/input"
			cc_output_dir = (executable_path.rsplit('/', 1)[0])+"/output"
	else:
		print("Executable not yet configured. It is recommended to use Seti Astro Cosmic Clarity v5.4 or higher.")
		sys.exit(1)

	os.chdir(cc_input_dir)
	for image in os.listdir():
		os.remove(image)
	os.chdir(workdir)		

	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:
			siril.log(image)
			# CosmicClaritySuite does not support compressed fit files
			if image.endswith(('.fz')):
				if compress:
					siril.cmd("setcompress 0")
				siril.cmd("load", image)
				siril.cmd("save", image)
				image = os.path.splitext(image)[0]
				os.remove(f"{image}.fz")
				if compress:
					siril.cmd("setcompress 1")
			shutil.copy(image, cc_input_dir)
			cmd = f"{executable_path} --sharpening_mode '{sharpenCC_mode}' --nonstellar_strength {sharpenCC_non_stellar_strength} --stellar_amount {sharpenCC_stellar_amount} --nonstellar_amount  {sharpenCC_non_stellar_amount} --auto_detect_psf"
			print(cmd)
			process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')

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
							siril.update_progress(f"Sharpen", pct / 100.0)
						except ValueError:
							siril.log(line)
					else:
							siril.log(line)
			process.wait()

			os.chdir(cc_input_dir)
			for oldimage in os.listdir():
				os.remove(oldimage)
			os.chdir(cc_output_dir)
			for ccimage in os.listdir():
				newimage = (f"{(image).rsplit('.', 1)[0]}_sc{sharpenCC_mode}-{sharpenCC_non_stellar_strength}-{sharpenCC_stellar_amount}-{sharpenCC_non_stellar_amount}.fit")		
				shutil.move(ccimage, (f"{workdir}/{newimage}"))
				processed_images.append(f"{image}")		
			os.chdir(workdir)

def sharpen_SA(workdir):
	os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
	config_dir = siril.get_siril_configdir()
	if os.path.isfile (f"{config_dir}/sirilcc_saspro.conf"):
		config_file_path = (f"{config_dir}/sirilcc_saspro.conf")
		with open(config_file_path, 'r') as file:
			executable_path = file.readline().strip()
			python_path = executable_path.replace("setiastrosuitepro", "python")
	else:
		print(f"Executable not configured. Please create file 'sirilcc_saspro.conf' in your siril config directory {config_dir} with a line containing the path to setiastrosuitepro.")
		sys.exit(1)
		
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:
			siril.log(image)
			newimage = (f"{(image).rsplit('.', 1)[0]}_ssa-{sharpenSA_mode.rsplit( )[0]}-{sharpenSA_non_stellar_amount}-{sharpenSA_stellar_amount}.fit")
			cmd = [python_path, executable_path, "cc", "sharpen", "--gpu", "--sharpening-mode", f"{sharpenSA_mode}", "--nonstellar-amount", f"{sharpenSA_non_stellar_amount}", "--stellar-amount", f"{sharpenSA_stellar_amount}", "--auto-psf", "-i", f"{image}", "-o", f"{newimage}"]
			print(" ".join(cmd))
			
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
							siril.update_progress(f"Sharpen", pct / 100.0)
						except ValueError:
							siril.log(line)
					else:
							siril.log(line)
			process.wait()
			processed_images.append(f"{image}")		
			
def sharpen_GraX(workdir):
	os.chdir(workdir)
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:
			siril.log("Starting GraXpert sharpen on " + image)
			siril.cmd("load", image)
			if sharpenGraX_mode == "both":
				siril.cmd("pyscript GraXpert-AI.py -gpu -deconv_obj -strength " + sharpenGraX_strength)
				siril.cmd("pyscript GraXpert-AI.py -gpu -deconv_stellar -strength " + sharpenGraX_strength)
			if sharpenGraX_mode == "object":
				siril.cmd("pyscript GraXpert-AI.py -gpu -deconv_obj -strength " + sharpenGraX_strength)			
			if sharpenGraX_mode == "stellar":
				siril.cmd("pyscript GraXpert-AI.py -gpu -deconv_stellar -strength " + sharpenGraX_strength)			
			newimage = (f"{(image).rsplit('.', 1)[0]}_sg{sharpenGraX_mode}{sharpenGraX_strength}")
			siril.cmd("save", newimage)
			processed_images.append(f"{image}")

def spcc(workdir):
	os.chdir(workdir)
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:
			siril.log("Starting SPCC on " + image)
			siril.cmd("load", image)			
			siril.cmd("platesolve")
			if Type == 'OSC':
				print(f'spcc \"-oscsensor={spcc_sensor}\" \"-oscfilter={spcc_oscfilter}\"')
				siril.cmd(f'spcc \"-oscsensor={spcc_sensor}\" \"-oscfilter={spcc_oscfilter}\"')
			if Type == 'mono':
				siril.cmd(f'spcc \"-monosensor={spcc_sensor}\" \"-rfilter={spcc_rfilter}\" \"-gfilter={spcc_gfilter}\" \"-bfilter={spcc_bfilter}\"')
			newimage = (f"{os.path.splitext(image)[0]}_spcc")
			siril.cmd("save", newimage)
#			os.remove(image)
			processed_images.append(f"{image}")

def starnet(workdir):
	os.chdir(workdir)
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:
			siril.log("Running starnet on " + image)
			siril.cmd("load", image)
			if args.starnet[0] == 2:
				upscale = '-upscale'
			elif args.starnet[0] == 1:
				upscale = ''
			else:
				print("Upscale factor needs to be a 1 or 2")
				sys.exit(1)
			siril.cmd(f"starnet -stretch {upscale}")
			for starmask in os.listdir():
				if starmask.startswith("starmask"):
					siril.cmd("load", starmask)
					if args.synthstar:
						siril.cmd("synthstar")
					siril.cmd("gauss 1.2")
					siril.cmd("save", starmask)
					processed_images.append(f"{starmask}")
			processed_images.append(f"{image}")

def statstretch(workdir):
	os.chdir(workdir)
	for image in os.listdir():
		if image.endswith(('.fits', '.fit', '.fts', '.fz')) and image not in processed_images:
			siril.log("Stretching " + image)
			siril.cmd("load", image)
			siril.cmd(f"pyscript Statistical_Stretch.py -linked -normalize -hdr -hdramount {stretch_hdr_amount} -hdrknee {stretch_hdr_knee} -boost {stretch_boost_amount}")
			newimage = (f"{os.path.splitext(image)[0]}_ss{stretch_hdr_amount}-{stretch_hdr_knee}-{stretch_boost_amount}")
			siril.cmd("save", newimage)
			processed_images.append(f"{image}")			

def run_gui():
	if 'QApplication' not in globals():
		print("PyQt6 is not installed. Please install it to use the GUI.")
		print("pip install PyQt6")
		return

	class ProcessingDialog(QDialog):
		def __init__(self, parent=None):
			super().__init__(parent)
			self.setWindowTitle("Siril GPS_Processing")
			self.setMinimumWidth(920)
			self.setMinimumHeight(800)
			
			main_layout = QVBoxLayout(self)

			# --- Working Directory (Top) ---
			dir_group = QGroupBox()
			dir_form = QFormLayout(dir_group)
			self.workdir_input = QLineEdit(os.getcwd())
			self.workdir_input.setToolTip("All fit(s) files in working directory will be processed")
			dir_form.addRow("Working Directory:", self.workdir_input)
			main_layout.addWidget(dir_group)

			# --- Scroll Area for Parameters ---
			scroll = QScrollArea()
			scroll.setWidgetResizable(True)
			scroll_content = QWidget()
			scroll_layout = QVBoxLayout(scroll_content)
			scroll.setWidget(scroll_content)
			main_layout.addWidget(scroll)

			def create_aligned_label(text, width=85):
				lbl = QLabel(text)
				lbl.setFixedWidth(width)
				return lbl

			def add_aligned_row(form, checkbox, layout):
				checkbox.setFixedWidth(220)
				form.addRow(checkbox, layout)

			# --- 1. Basic Setup Group ---
			basic_group = QGroupBox("1. Crop, Star processing and/or Background Extraction")
			basic_form = QFormLayout(basic_group)

			self.crop_cb = QCheckBox("Crop image")
			self.crop_value = QLineEdit("1")
			self.crop_value.setFixedWidth(80)
			self.crop_value.setToolTip("Provide one or more crop percentage(s)")
			self.crop_value.setEnabled(False)
			self.crop_cb.toggled.connect(self.crop_value.setEnabled)
			crop_layout = QHBoxLayout()
			crop_layout.setSpacing(10)
			crop_layout.addWidget(create_aligned_label("%(s)"))
			crop_layout.addWidget(self.crop_value)
			crop_layout.addStretch()
			add_aligned_row(basic_form, self.crop_cb, crop_layout)

			self.starnet_cb = QCheckBox("Starnet")
			self.scale_factor = QLineEdit("1")
			self.scale_factor.setFixedWidth(80)
			self.scale_factor.setEnabled(False)
			self.combine_factor = QLineEdit("0.5")
			self.combine_factor.setFixedWidth(80)
			self.combine_factor.setEnabled(False)
			self.synthstar_cb = QCheckBox("Synthstar on starmask")
			self.synthstar_cb.setEnabled(False)
			self.starnet_cb.toggled.connect(self.scale_factor.setEnabled)
			self.starnet_cb.toggled.connect(self.combine_factor.setEnabled)
			self.starnet_cb.toggled.connect(self.synthstar_cb.setEnabled)
			starnet_layout = QHBoxLayout()
			starnet_layout.setSpacing(10)
			starnet_layout.addWidget(create_aligned_label("Scale"))
			starnet_layout.addWidget(self.scale_factor)
			starnet_layout.addWidget(create_aligned_label("Star factor:"))
			starnet_layout.addWidget(self.combine_factor)
			starnet_layout.addWidget(self.synthstar_cb)
			starnet_layout.addStretch()
			add_aligned_row(basic_form, self.starnet_cb, starnet_layout)

			self.abe_cb = QCheckBox("Auto Bkg Extraction")
			self.abe_npoints = QLineEdit("100")
			self.abe_polydegree = QLineEdit("2")
			self.abe_rbfsmooth = QLineEdit("0.1")
			self.abe_npoints.setFixedWidth(80)
			self.abe_npoints.setEnabled(False)
			for w in [self.abe_polydegree, self.abe_rbfsmooth]:
				w.setFixedWidth(40)
				w.setEnabled(False)
			self.abe_cb.toggled.connect(self.abe_npoints.setEnabled)
			self.abe_cb.toggled.connect(self.abe_polydegree.setEnabled)
			self.abe_cb.toggled.connect(self.abe_rbfsmooth.setEnabled)
			abe_layout = QHBoxLayout()
			abe_layout.setSpacing(10)
			abe_layout.addWidget(create_aligned_label("Npoints"))
			abe_layout.addWidget(self.abe_npoints)
			abe_layout.addWidget(create_aligned_label("Polydegree"))
			abe_layout.addWidget(self.abe_polydegree)
			abe_layout.addWidget(create_aligned_label("Rbfsmooth"))
			abe_layout.addWidget(self.abe_rbfsmooth)
			abe_layout.addStretch()
			add_aligned_row(basic_form, self.abe_cb, abe_layout)
			
			self.bkg_cb = QCheckBox("Siril Bkg Extraction")
			self.bkg_smooth = QLineEdit("0.5")
			self.bkg_smooth.setFixedWidth(40)
			self.bkg_smooth.setEnabled(False)
			self.bkg_cb.toggled.connect(self.bkg_smooth.setEnabled)
			bkg_layout = QHBoxLayout()
			bkg_layout.setSpacing(10)
			bkg_layout.addWidget(create_aligned_label("Smoothing"))
			bkg_layout.addWidget(self.bkg_smooth)
			bkg_layout.addStretch()
			add_aligned_row(basic_form, self.bkg_cb, bkg_layout)

			self.bkg_grax_cb = QCheckBox("GraXpert Bkg Extraction")
			self.bkg_grax_smooth = QLineEdit("0.5")
			self.bkg_grax_smooth.setFixedWidth(40)
			self.bkg_grax_smooth.setEnabled(False)
			self.bkg_grax_cb.toggled.connect(self.bkg_grax_smooth.setEnabled)
			bkg_grax_layout = QHBoxLayout()
			bkg_grax_layout.setSpacing(10)
			bkg_grax_layout.addWidget(create_aligned_label("Smoothing"))
			bkg_grax_layout.addWidget(self.bkg_grax_smooth)
			bkg_grax_layout.addStretch()
			add_aligned_row(basic_form, self.bkg_grax_cb, bkg_grax_layout)

#			self.spcc_cb = QCheckBox("Siril SPCC")
#			self.spcc_sensor = QLineEdit("")
#			self.spcc_sensor.setPlaceholderText("sensor name")
#			self.spcc_filters = QLineEdit("")
#			self.spcc_filters.setPlaceholderText("filters OSC or R G B")
#			self.spcc_sensor.setEnabled(False)
#			self.spcc_filters.setEnabled(False)
#			self.spcc_cb.toggled.connect(self.spcc_sensor.setEnabled)
#			self.spcc_cb.toggled.connect(self.spcc_filters.setEnabled)
#			spcc_layout = QHBoxLayout()
#			spcc_layout.setSpacing(10)
#			spcc_layout.addWidget(create_aligned_label("Sensor:"))
#			spcc_layout.addWidget(self.spcc_sensor)
#			spcc_layout.addWidget(create_aligned_label("Filter(s):"))
#			spcc_layout.addWidget(self.spcc_filters)
#			spcc_layout.addStretch()
#			add_aligned_row(basic_form, self.spcc_cb, spcc_layout)

			scroll_layout.addWidget(basic_group)

			# --- 2. Sharpening Group ---
			sharpen_group = QGroupBox("2. Sharpening")
			sharpen_form = QFormLayout(sharpen_group)

			self.sharpen_cb = QCheckBox("Siril Sharpen")
			sharpen_form.addRow(self.sharpen_cb)

			self.sharpen_cc_cb = QCheckBox("CC Sharpen")
			self.sharpen_cc_mode = QComboBox()
			self.sharpen_cc_mode.addItems(['Both', 'Stellar Only' ,'Non-Stellar Only'])
			self.sharpen_cc_stellar_amount = QLineEdit("0.5")
			self.sharpen_cc_non_stellar_amount = QLineEdit("0.5")
			self.sharpen_cc_non_stellar_strength = QLineEdit("5")
			self.sharpen_cc_mode.setFixedWidth(80)
			self.sharpen_cc_mode.setEnabled(False)
			for w in [self.sharpen_cc_stellar_amount, self.sharpen_cc_non_stellar_amount, self.sharpen_cc_non_stellar_strength]:
				w.setFixedWidth(40)
				w.setEnabled(False)
			self.sharpen_cc_cb.toggled.connect(self.sharpen_cc_mode.setEnabled)
			self.sharpen_cc_cb.toggled.connect(self.sharpen_cc_stellar_amount.setEnabled)
			self.sharpen_cc_cb.toggled.connect(self.sharpen_cc_non_stellar_amount.setEnabled)
			self.sharpen_cc_cb.toggled.connect(self.sharpen_cc_non_stellar_strength.setEnabled)
			self.sharpen_cc_mode.currentTextChanged.connect(self.update_sharpen_cc_options)
			sharpen_cc_layout = QHBoxLayout()
			sharpen_cc_layout.setSpacing(10)
			sharpen_cc_layout.addWidget(create_aligned_label("Mode"))
			sharpen_cc_layout.addWidget(self.sharpen_cc_mode)
			sharpen_cc_layout.addWidget(create_aligned_label("Stellar"))
			sharpen_cc_layout.addWidget(self.sharpen_cc_stellar_amount)
			sharpen_cc_layout.addWidget(create_aligned_label("Non-Stellar"))
			sharpen_cc_layout.addWidget(self.sharpen_cc_non_stellar_amount)
			sharpen_cc_layout.addWidget(create_aligned_label("Strength"))
			sharpen_cc_layout.addWidget(self.sharpen_cc_non_stellar_strength)
			sharpen_cc_layout.addStretch()
			add_aligned_row(sharpen_form, self.sharpen_cc_cb, sharpen_cc_layout)

			self.sharpen_ssa_cb = QCheckBox("Setiastro CC Sharpen")
			self.sharpen_ssa_mode = QComboBox()
			self.sharpen_ssa_mode.addItems(['Both', 'Stellar Only' ,'Non-Stellar Only'])
			self.sharpen_ssa_stellar_amount = QLineEdit("0.5")
			self.sharpen_ssa_non_stellar_amount = QLineEdit("0.5")
			self.sharpen_ssa_mode.setFixedWidth(80)
			self.sharpen_ssa_mode.setEnabled(False)
			for w in [self.sharpen_ssa_stellar_amount, self.sharpen_ssa_non_stellar_amount]:
				w.setFixedWidth(40)
				w.setEnabled(False)
			self.sharpen_ssa_cb.toggled.connect(self.sharpen_ssa_mode.setEnabled)
			self.sharpen_ssa_cb.toggled.connect(self.sharpen_ssa_stellar_amount.setEnabled)
			self.sharpen_ssa_cb.toggled.connect(self.sharpen_ssa_non_stellar_amount.setEnabled)
			self.sharpen_ssa_mode.currentTextChanged.connect(self.update_sharpen_ssa_options)
			sharpen_ssa_layout = QHBoxLayout()
			sharpen_ssa_layout.setSpacing(10)
			sharpen_ssa_layout.addWidget(create_aligned_label("Mode"))
			sharpen_ssa_layout.addWidget(self.sharpen_ssa_mode)
			sharpen_ssa_layout.addWidget(create_aligned_label("Stellar"))
			sharpen_ssa_layout.addWidget(self.sharpen_ssa_stellar_amount)
			sharpen_ssa_layout.addWidget(create_aligned_label("Non-Stellar"))
			sharpen_ssa_layout.addWidget(self.sharpen_ssa_non_stellar_amount)
			sharpen_ssa_layout.addStretch()
			add_aligned_row(sharpen_form, self.sharpen_ssa_cb, sharpen_ssa_layout)

			self.sharpen_grax_cb = QCheckBox("GraXpert Sharpen")
			self.sharpen_grax_mode = QComboBox()
			self.sharpen_grax_mode.addItems(['both', 'object' ,'stellar'])
			self.sharpen_grax_strength = QLineEdit("0.5")
			self.sharpen_grax_mode.setFixedWidth(80)
			self.sharpen_grax_mode.setEnabled(False)
			for w in [self.sharpen_grax_strength]:
				w.setFixedWidth(40)
				w.setEnabled(False)
			self.sharpen_grax_cb.toggled.connect(self.sharpen_grax_mode.setEnabled)
			self.sharpen_grax_cb.toggled.connect(self.sharpen_grax_strength.setEnabled)
			sharpen_grax_layout = QHBoxLayout()
			sharpen_grax_layout.setSpacing(10)
			sharpen_grax_layout.addWidget(create_aligned_label("Mode"))
			sharpen_grax_layout.addWidget(self.sharpen_grax_mode)	
			sharpen_grax_layout.addWidget(create_aligned_label("Strength"))
			sharpen_grax_layout.addWidget(self.sharpen_grax_strength)
			sharpen_grax_layout.addStretch()
			add_aligned_row(sharpen_form, self.sharpen_grax_cb, sharpen_grax_layout)
			scroll_layout.addWidget(sharpen_group)

			# --- 3. Denoising Group ---
			denoise_group = QGroupBox("3. Denoising")
			denoise_form = QFormLayout(denoise_group)
			self.denoise_cb = QCheckBox("Siril Denoise")
			self.denoise_cb.setFixedWidth(250)
			denoise_form.addRow(self.denoise_cb)
			
			self.denoise_cc_cb = QCheckBox("CC Denoise")
			self.denoise_cc_mode = QComboBox()
			self.denoise_cc_mode.addItems(['full', 'luminance', 'separate'])
			self.denoise_cc_strength = QLineEdit("0.5")
			self.denoise_cc_mode.setFixedWidth(80)
			self.denoise_cc_mode.setEnabled(False)
			self.denoise_cc_strength.setFixedWidth(40)
			self.denoise_cc_strength.setEnabled(False)
			self.denoise_cc_cb.toggled.connect(self.denoise_cc_mode.setEnabled)
			self.denoise_cc_cb.toggled.connect(self.denoise_cc_strength.setEnabled)
			denoise_cc_layout = QHBoxLayout()
			denoise_cc_layout.setSpacing(10)
			denoise_cc_layout.addWidget(create_aligned_label("Mode"))
			denoise_cc_layout.addWidget(self.denoise_cc_mode)
			denoise_cc_layout.addWidget(create_aligned_label("Strength"))
			denoise_cc_layout.addWidget(self.denoise_cc_strength)
			denoise_cc_layout.addStretch()
			add_aligned_row(denoise_form, self.denoise_cc_cb, denoise_cc_layout)

			self.denoise_dsa_cb = QCheckBox("Setiastro CC Denoise")
			self.denoise_dsa_mode = QComboBox()
			self.denoise_dsa_mode.addItems(['full', 'luminance'])
			self.denoise_dsa_mode.setFixedWidth(80)
			self.denoise_dsa_luma_amount = QLineEdit("0.5")
			self.denoise_dsa_luma_amount.setFixedWidth(40)
			self.denoise_dsa_color_amount = QLineEdit("0.5")
			self.denoise_dsa_color_amount.setFixedWidth(40)
			self.denoise_dsa_mode.setEnabled(False)
			self.denoise_dsa_luma_amount.setEnabled(False)
			self.denoise_dsa_color_amount.setEnabled(False)			
			self.denoise_dsa_cb.toggled.connect(self.denoise_dsa_mode.setEnabled)
			self.denoise_dsa_cb.toggled.connect(self.denoise_dsa_luma_amount.setEnabled)
			self.denoise_dsa_cb.toggled.connect(self.denoise_dsa_color_amount.setEnabled)			
			denoise_dsa_layout = QHBoxLayout()
			denoise_dsa_layout.setSpacing(10)
			denoise_dsa_layout.addWidget(create_aligned_label("Mode"))
			denoise_dsa_layout.addWidget(self.denoise_dsa_mode)
			denoise_dsa_layout.addWidget(create_aligned_label("Luma"))
			denoise_dsa_layout.addWidget(self.denoise_dsa_luma_amount)
			denoise_dsa_layout.addWidget(create_aligned_label("Color"))
			denoise_dsa_layout.addWidget(self.denoise_dsa_color_amount)
			denoise_dsa_layout.addStretch()
			add_aligned_row(denoise_form, self.denoise_dsa_cb, denoise_dsa_layout)

			self.denoise_grax_cb = QCheckBox("GraXpert Denoise")
			self.denoise_grax_strength = QLineEdit("0.5")
			self.denoise_grax_strength.setFixedWidth(40)
			self.denoise_grax_strength.setEnabled(False)
			self.denoise_grax_cb.toggled.connect(self.denoise_grax_strength.setEnabled)
			denoise_grax_layout = QHBoxLayout()
			denoise_grax_layout.setSpacing(10)
			denoise_grax_layout.addWidget(create_aligned_label("Strength"))
			denoise_grax_layout.addWidget(self.denoise_grax_strength)
			denoise_grax_layout.addStretch()
			add_aligned_row(denoise_form, self.denoise_grax_cb, denoise_grax_layout)
			scroll_layout.addWidget(denoise_group)

			# --- 4. Finalization Group ---
			final_group = QGroupBox("4. Stretch and/or Multiprocess")
			final_form = QFormLayout(final_group)
			self.autostretch_cb = QCheckBox("Autostretch (linked)")
			self.autostretch_cb.setFixedWidth(250)
			final_form.addRow(self.autostretch_cb)
			
			self.stretch_cb = QCheckBox("Statistical Stretch")
			self.stretch_hdr_amount = QLineEdit("0.15")
			self.stretch_hdr_knee = QLineEdit("0.75")
			self.stretch_boost_amount = QLineEdit("0.2")
			self.stretch_hdr_amount.setFixedWidth(80)
			self.stretch_hdr_amount.setEnabled(False)
			for w in [self.stretch_hdr_knee, self.stretch_boost_amount]:
				w.setFixedWidth(40)
				w.setEnabled(False)
			self.stretch_cb.toggled.connect(self.stretch_hdr_amount.setEnabled)
			self.stretch_cb.toggled.connect(self.stretch_hdr_knee.setEnabled)
			self.stretch_cb.toggled.connect(self.stretch_boost_amount.setEnabled)
			stretch_layout = QHBoxLayout()
			stretch_layout.setSpacing(10)
			stretch_layout.addWidget(create_aligned_label("HDR"))
			stretch_layout.addWidget(self.stretch_hdr_amount)
			stretch_layout.addWidget(create_aligned_label("Knee"))
			stretch_layout.addWidget(self.stretch_hdr_knee)
			stretch_layout.addWidget(create_aligned_label("Boost"))
			stretch_layout.addWidget(self.stretch_boost_amount)
			stretch_layout.addStretch()
			add_aligned_row(final_form, self.stretch_cb, stretch_layout)

			self.multiprocess_cb = QCheckBox("Multiprocess directories")
			self.multiprocess_cb.setToolTip("Saves processed images in unique Processed_N directory")
			self.multiprocess_cb.setFixedWidth(250)
			final_form.addRow(self.multiprocess_cb)
			scroll_layout.addWidget(final_group)

			# --- Buttons ---
			self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
			self.button_box.accepted.connect(self.accept)
			self.button_box.rejected.connect(self.reject)
			main_layout.addWidget(self.button_box)

			# Initial UI updates
			self.update_sharpen_cc_options(self.sharpen_cc_mode.currentText())
			self.update_sharpen_ssa_options(self.sharpen_ssa_mode.currentText())
		
		def update_sharpen_cc_options(self, mode):
			if mode == 'Both':
				self.sharpen_cc_stellar_amount.setEnabled(True)
				self.sharpen_cc_non_stellar_amount.setEnabled(True)
				self.sharpen_cc_non_stellar_strength.setEnabled(True)
			elif mode == 'Stellar Only':
				self.sharpen_cc_stellar_amount.setEnabled(True)
				self.sharpen_cc_non_stellar_amount.setEnabled(False)
				self.sharpen_cc_non_stellar_strength.setEnabled(False)
			elif mode == 'Non-Stellar Only':
				self.sharpen_cc_stellar_amount.setEnabled(False)
				self.sharpen_cc_non_stellar_amount.setEnabled(True)
				self.sharpen_cc_non_stellar_strength.setEnabled(True)

		def update_sharpen_ssa_options(self, mode):
			if mode == 'Both':
				self.sharpen_ssa_stellar_amount.setEnabled(True)
				self.sharpen_ssa_non_stellar_amount.setEnabled(True)
			elif mode == 'Stellar Only':
				self.sharpen_ssa_stellar_amount.setEnabled(True)
				self.sharpen_ssa_non_stellar_amount.setEnabled(False)
			elif mode == 'Non-Stellar Only':
				self.sharpen_ssa_stellar_amount.setEnabled(False)
				self.sharpen_ssa_non_stellar_amount.setEnabled(True)

		def get_values(self):
			return {
				"workdir": self.workdir_input.text(),
				"crop": self.crop_value.text().split() if self.crop_cb.isChecked() else None,
				"abe": [self.abe_npoints.text(), self.abe_polydegree.text(), self.abe_rbfsmooth.text()] if self.abe_cb.isChecked() else None,				
				"bkg": self.bkg_smooth.text() if self.bkg_cb.isChecked() else None,
				"bkgGraX": self.bkg_grax_smooth.text() if self.bkg_grax_cb.isChecked() else None,
				"denoise": self.denoise_cb.isChecked(),
				"denoiseCC": [self.denoise_cc_mode.currentText(), self.denoise_cc_strength.text()] if self.denoise_cc_cb.isChecked() else None,
				"denoiseSA": [self.denoise_dsa_mode.currentText(), self.denoise_dsa_luma_amount.text(), self.denoise_dsa_color_amount.text()] if self.denoise_dsa_cb.isChecked() else None,
				"denoiseGraX": self.denoise_grax_strength.text() if self.denoise_grax_cb.isChecked() else None,
				"multiprocess": self.multiprocess_cb.isChecked(),
				"sharpen": self.sharpen_cb.isChecked(),
				"sharpenCC": [self.sharpen_cc_mode.currentText(), self.sharpen_cc_stellar_amount.text(), self.sharpen_cc_non_stellar_amount.text(), self.sharpen_cc_non_stellar_strength.text()] if self.sharpen_cc_cb.isChecked() else None,
				"sharpenGraX": [self.sharpen_grax_mode.currentText(), self.sharpen_grax_strength.text()] if self.sharpen_grax_cb.isChecked() else None,
				"sharpenSA": [self.sharpen_ssa_mode.currentText(), self.sharpen_ssa_stellar_amount.text(), self.sharpen_ssa_non_stellar_amount.text()] if self.sharpen_ssa_cb.isChecked() else None,
				"starnet": [self.scale_factor.text(), self.combine_factor.text()] if self.starnet_cb.isChecked() else None,
				"synthstar": self.synthstar_cb.isChecked(),
				"autostretch": self.autostretch_cb.isChecked(),				
				"statstretch": [self.stretch_hdr_amount.text(), self.stretch_hdr_knee.text(), self.stretch_boost_amount.text()] if self.stretch_cb.isChecked() else None,
			}

	app = QApplication.instance() or QApplication(sys.argv)

	dark_stylesheet = """
		QWidget {
			background-color: #2b2b2b;
			color: #efefef;
		}
		QLineEdit, QTextEdit, QComboBox {
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

	dialog = ProcessingDialog()
	if dialog.exec():
		values = dialog.get_values()
		cli_args = []		
		
		if values["workdir"]:
			cli_args.extend(["-d", values["workdir"]])
		if values["crop"]:
			cli_args.extend(["-c"] + values["crop"])
		if values["abe"]:
			cli_args.extend(["-ab", values["abe"][0], values["abe"][1], values["abe"][2]])		
		if values["bkg"]:
			cli_args.extend(["-b", values["bkg"]])
		if values["bkgGraX"]:
			cli_args.extend(["-bg", values["bkgGraX"]])
		if values["denoise"]:
			cli_args.append("-ds")
		if values["denoiseCC"]:
			cli_args.extend(["-dc", values["denoiseCC"][0], values["denoiseCC"][1]])	
		if values["denoiseSA"]:
			cli_args.extend(["-dsa", values["denoiseSA"][0], values["denoiseSA"][1], values["denoiseSA"][2]])
		if values["denoiseGraX"]:
			cli_args.extend(["-dg", values["denoiseGraX"]])
		if values["sharpen"]:
			cli_args.append("-s")
		if values["sharpenCC"]:
			cli_args.extend(["-sc", values["sharpenCC"][0], values["sharpenCC"][1], values["sharpenCC"][2], values["sharpenCC"][3]])
		if values["sharpenSA"]:
			cli_args.extend(["-ssa", values["sharpenSA"][0], values["sharpenSA"][1], values["sharpenSA"][2]])		
		if values["sharpenGraX"]:
			cli_args.extend(["-sg", values["sharpenGraX"][0], values["sharpenGraX"][1]])
		if values["starnet"]:
			cli_args.extend(["-sn", values["starnet"][0], values["starnet"][1]])
		if values["synthstar"]:
			cli_args.append("-sy")		
		if values["autostretch"]:
			cli_args.append("-as")
		if values["statstretch"]:
			cli_args.extend(["-ss", values["statstretch"][0], values["statstretch"][1], values["statstretch"][2]])
		if values["multiprocess"]:
			cli_args.append("-m")

		print(*cli_args)
		main_logic(cli_args)

		msg_box = QMessageBox()
		msg_box.setIcon(QMessageBox.Icon.Information)
		msg_box.setText("Processing finished.")
		msg_box.setWindowTitle("Success")
		msg_box.exec()

# ==============================================================================	
# Main execution
# ==============================================================================	

def main_logic(argv):
	global args, npoints, crop, crop_value, polydegree, rbfsmooth, smooth, bkgGraX, denoiseCC_mode, denoiseCC_strength, denoiseGraX, denoiseSA_mode, denoiseSA_luma_amount, denoiseSA_color_amount, sharpenGraX_mode, sharpenGraX_strength, sharpenCC_mode, sharpenCC_stellar_amount, sharpenCC_non_stellar_amount, sharpenCC_non_stellar_strength, sharpenSA_mode, sharpenSA_stellar_amount, sharpenSA_non_stellar_amount, autostretch, starnet, stretch_hdr_amount, stretch_hdr_knee, stretch_boost_amount, spcc_sensor, spcc_oscfilter, spcc_rfilter, spcc_gfilter, spcc_bfilter, Type
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-ab","--abe", nargs='+', action='append', help="AutoBGE, provide npoints, polydegree and rbfsmooth")
	parser.add_argument("-as","--autostretch", help="Siril autostretch (linked)" ,action="store_true")
	parser.add_argument("-b","--bkg", nargs='+', action='append', help="siril background extraction, provide smoothing 0.0-1.0")
	parser.add_argument("-bg","--bkgGraX", nargs='+', action='append', help="GraXpert background extraction, provide smoothing 0.0-1.0")
	parser.add_argument("-c","--crop", nargs='+', help="crop image by percentage, accepts multiple values for different crops")
	parser.add_argument("-cc","--spcc", nargs='+', help="spcc color calibration, provide sensor and filter(s) OSC or R, G & B, using quotes")
	parser.add_argument("-d","--workdir", nargs='+', help="set working directory")

	parser.add_argument("-dc","--denoiseCC", nargs='+', action='append', help="run CC denoise, provide mode (luminance, full, separate) and denoise strength 0.0-1.0")
	parser.add_argument("-dg","--denoiseGraX", nargs='+', action='append', help="denoise using GraXpert-AI, provide strength 0.0-1.0")
	parser.add_argument("-dsa","--denoiseSA", nargs='+', action='append' ,help="run SASpro CC denoise, provide mode (full or luminance), luminance denoise strength (0.0-1.0) and color denoise strength (0.0-1.0)")
	parser.add_argument("-ds","--denoise", help="run denoise" ,action="store_true")
	parser.add_argument("-m","--multiprocess", help="saves processed images in unique Processed_N directory" ,action="store_true")	
	parser.add_argument("-s","--sharpen", help="sharpen (deconvolution)" ,action="store_true")
	parser.add_argument("-sc","--sharpenCC", nargs='+' ,help="run CC sharpen, provide mode (Stellar Only,Non-Stellar Only,Both), Stellar_amount and/or Non_stellar_amount and Non_stellar_strength")
	parser.add_argument("-sg","--sharpenGraX", nargs='+', action='append', help="sharpen (deconvolution) using GraXpert-AI, provide mode (both, object, stellar) and strength 0.0-1.0")
	parser.add_argument("-sn","--starnet", nargs=2, help="create starless & starmask images, sharpen and/or denoise on starless, then recombines. Provide scale factor (1 or 2) and star combine factor (0-1)", type = float)
	parser.add_argument("-sy","--synthstar", help="Runs synthstar on starmask to correct misshapen stars" ,action="store_true")	
	parser.add_argument("-ssa","--sharpenSA", nargs='+', action='append' ,help="run SASpro CC sharpen, provide mode (Stellar Only,Non-Stellar Only,Both), Stellar_amount and/or Non_stellar_amount")
	parser.add_argument("-ss","--statstretch", nargs='+', action='append', help="statistical stretch, provide HDR amount, HDR knee and boost amount")
	parser.add_argument("-v","--version", help="print the version and exit",action="store_true")
	args = parser.parse_args(argv)

	if args.version:
		print('version ' + VERSION)
		sys.exit(1)
	
	try:
		siril.connect()
		siril.cmd("requires", "1.3.6")	
		siril.log("Running processing")
		workdir = args.workdir[0] if args.workdir else os.getcwd()	
		siril.cmd("cd",f'"{workdir}"')
		siril.cmd("set32bits")
		siril.cmd("setext", "fit")

		os.chdir(workdir)
		for image in os.listdir():
			if image.endswith(('.fits', '.fit', '.fts', '.fz')):
				original_images.append(f"{image}")

		if args.crop:
			crop(workdir)
		
		if args.starnet:
			starnet(workdir)

		if args.abe:
			for n in args.abe:
				npoints = (n[0])
				polydegree = (n[1])
				rbfsmooth = (n[2])
				abe(workdir)
				
		if args.bkg:
			for n in args.bkg:
				smooth = (n[0])			
				bkg(workdir)

		if args.bkgGraX:
			for n in args.bkgGraX:
				bkgGraX = (n[0])
				bkg_GraX(workdir)

		if args.spcc:
			spcc_sensor = (args.spcc[0])
			if (len (args.spcc)) == 2:
				Type = 'OSC'
				spcc_oscfilter = (args.spcc[1])
			elif (len (args.spcc)) == 4:
				Type = 'mono'
				spcc_rfilter = (args.spcc[1])
				spcc_gfilter = (args.spcc[2])				
				spcc_bfilter = (args.spcc[3])				
			else:
				print("spcc needs 2 args for OSC or 4 args for mono")
				sys.exit(1)
			spcc(workdir)			

		if args.sharpen:
			sharpen(workdir)

		if args.sharpenCC:
			for n in args.sharpenCC:			
				sharpenCC_mode = (n[0])
				if (n[0]) == 'Both':
					sharpenCC_stellar_amount = (n[1])
					sharpenCC_non_stellar_amount = (n[2])	
					sharpenCC_non_stellar_strength = (n[3])
				elif (n[0]) == 'Non-Stellar Only':
					sharpenCC_non_stellar_amount = (n[1])	
					sharpenCC_non_stellar_strength = (n[2])						
					sharpenCC_stellar_amount = '0'
				elif (n[0]) == 'Stellar Only':
					sharpenCC_stellar_amount = (n[1])			
					sharpenCC_non_stellar_amount = '0'	
					sharpenCC_non_stellar_strength = '0'		
				else:
					print('Mode needs to be either Both, Stellar Only or Non-Stellar Only')
					sys.exit(1)
				sharpen_CC(workdir)
			
		if args.sharpenGraX:
			for n in args.sharpenGraX:
				sharpenGraX_mode = (n[0])
				sharpenGraX_strength = (n[1])
				sharpen_GraX(workdir)

		if args.sharpenSA:
			for n in args.sharpenSA:			
				sharpenSA_mode = (n[0])
				if (n[0]) == 'Both':
					sharpenSA_stellar_amount = (n[1])
					sharpenSA_non_stellar_amount = (n[2])	
				elif (n[0]) == 'Non-Stellar Only':
					sharpenSA_non_stellar_amount = (n[1])	
					sharpenSA_stellar_amount = '0'
				elif (n[0]) == 'Stellar Only':
					sharpenSA_stellar_amount = (n[1])			
					sharpenSA_non_stellar_amount = '0'	
				else:
					print('Mode needs to be either Both, Stellar Only or Non-Stellar Only')
					sys.exit(1)
				sharpen_SA(workdir)
				
		if args.denoise:
			denoise(workdir)

		if args.denoiseCC:
			for n in args.denoiseCC:
				denoiseCC_mode = (n[0])
				denoiseCC_strength = (n[1])
				denoise_CC(workdir)
			
		if args.denoiseSA:
			for n in args.denoiseSA:
				denoiseSA_mode = (n[0])
				denoiseSA_luma_amount = (n[1])
				denoiseSA_color_amount = (n[2])
				denoise_SA(workdir)

		if args.denoiseGraX:
			for n in args.denoiseGraX:
				denoiseGraX = (n[0])
				denoise_GraX(workdir)

		if args.starnet:
			pixelmath(workdir)

		if args.autostretch:
			autostretch(workdir)
		
		if args.statstretch:
			for n in args.statstretch:
				stretch_hdr_amount = (n[0])
				stretch_hdr_knee = (n[1])
				stretch_boost_amount = (n[2])
				statstretch(workdir)
				
		if args.multiprocess:
			multiprocess(workdir)

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
