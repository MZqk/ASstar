# (c) Steffen Schreiber, Patrick Wagner 2025
# SPDX-License-Identifier: GPL-3.0-or-later
#
# For bug reports or feature requests, please open an issue at
# https://gitlab.com/schreiberste/siril-scripts/-/issues

"""
Creates galaxy annotations from a Simbad query with several 
catalogs. Combines the original image with annotation overlays
and a thumbnail table of the found galaxies.
"""

# Version History
# 1.0.0 Initial release
# 1.0.1 Fix saving result images when output name contains dots
# 1.0.2 Added option to select circle or box annotation overlays
# 1.0.3 Update the GUI to use a ScrollableFrame for the catalogs list
# 1.0.4 Add Arecibo General Catalog, add redshift value 
# 1.0.5 More dynamic font sizes, always use scollable catalog selection
# 1.0.6 Add elliptic annotation overlays
# 1.0.7 Support for "Other" catalogs, option to include galaxies without size data
# 1.2.0 New GUI based on PyQT6
# 1.2.1 Remove "subprocess" from ensure_installed as it is a core python module (AKB)
# 1.2.2 Fix selection of loading result image in Siril
# 1.2.3 Add minimal support for object types other than galaxies
# 1.2.4 If no CLI arguments, run in GUI mode by default
# 1.2.5 Add LDN/LBN catalog entries, tweak object type filters (still not working completely)
# 1.2.6 Add optical magnitude limit filter (B/V/R/G/g/r bands)
# 1.2.7 Pixel-perfect overlay image: fixed border budget, exact axes placement

# Core module imports
import os
import sys
import math
import argparse
import numpy as np

import sirilpy as s
from sirilpy import SirilError

# Check the module version is enough to provide get_image_fits_header(return_as = 'dict')
if not s.check_module_version('>=0.6.37'):
    print("Error: requires sirilpy module >= 0.6.37 (Siril 1.4.0 Beta 2)")
    sys.exit(1)

s.ensure_installed("PyQt6", "astropy", "astroquery", "matplotlib", "numpy", 
                   "pandas", "Pillow", "scikit-image")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QLabel, QSlider, QPushButton, QRadioButton, QGroupBox, QMessageBox, QColorDialog,
    QCheckBox, QComboBox, QScrollArea, QFileDialog, QLineEdit, QFrame)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor, QGuiApplication, QIcon

# Add any additional imports here
import configparser
import ast
from dataclasses import dataclass
from enum import Enum
import subprocess
import matplotlib
matplotlib.use('agg')  # headless matplotlib backend, only for writing files
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse
from skimage.transform import resize
from PIL import Image
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import coordinates as coord
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.table import Table
import astropy.units as u
from astroquery.simbad import Simbad
import pandas as pd

VERSION = "1.2.7"
CONFIG_FILENAME = "Galaxy_Annotations.conf"

# ============================================================================
# Dataclasses for options
# ============================================================================

class OverlayType(Enum):
    ELLIPSES = "ellipses"
    CIRCLES = "circles"
    BOXES = "boxes"
    
    @staticmethod
    def from_str(label):
        if label in ('boxes', 'BOXES'):
            return OverlayType.BOXES
        elif label in ('circles', 'CIRCLES'):
            return OverlayType.CIRCLES
        else:
            return OverlayType.ELLIPSES

@dataclass
class QueryOptions:
    catalogs: dict
    include_unknown_size: bool = True
    include_other_types: bool = False
    mag_limit: float = 99.0  # 99 = no limit

    def get_config(self):
        return {
           'include_unknown_size': self.include_unknown_size,
           'include_other_types': self.include_other_types,
           'mag_limit': self.mag_limit
        }
    
@dataclass
class OutputOptions:
    output: str = ""
    title: str = ""
    logo_path: str = ""
    label_amount: int = 50
    overlay_alpha: float = 0.75
    overlay_type: OverlayType = OverlayType.ELLIPSES
    
    def get_config(self):
        return {
            'logo_path': self.logo_path,
            'label_amount': self.label_amount,
            'overlay_alpha' : self.overlay_alpha,
            'overlay_type' : self.overlay_type.value
        }
    

class CatalogEntry:
    """ This class provides properties of a catalog entry """
    def __init__(self, description, color='#ffffff', selection_default=True):
        self.description = description
        self.color_default = color
        self.color = color
        self.selection_default = selection_default
        self.selection = None
        
    def get_selected(self):
        if self.selection is None:
            return self.selection_default
        else:
            return self.selection
            
    def set_selection(self, selection):
        if selection is None:
            self.selection = self.selection_default
        else:
            newSelection = (selection == True) or (selection == Qt.CheckState.Checked.value)
            self.selection = newSelection
            
    def set_color(self, color):
        if color is None:
            self.color = self.color_default
        else:
            self.color = color

# ============================================================================
# Processing Worker Thread
# ============================================================================

class AnnotationWorker(QObject):
    """
    Worker thread for Simbad query and creating the annotation images.
    """
    progress_update = pyqtSignal(str, float)
    finished = pyqtSignal(object, int)
   
    def __init__(self, siril, fit, query_opts, output_opts):
        super().__init__()
        self.siril = siril
        self.fit = fit
        self.query_opts = query_opts
        self.output_opts = output_opts
    
    def run(self):
        try:
            self.annotate_fit()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(f"Error: {str(e)}", None)
    
    def annotate_fit(self):
        """
        The main processing function for creating the annotation images.
        """
        self.progress_update.emit("Preparing Simbad query...", 0.01)

        print(f"Title: {self.output_opts.title}")
        print(f"Logo:  {self.output_opts.logo_path}")
    
        main_object = self.output_opts.title
        output_fname = get_combined_filename(self.output_opts.output)
        output_overlay_fname = get_overlay_filename(self.output_opts.output)
        output_table_fname = get_table_filename(self.output_opts.output)
        overlay_type = self.output_opts.overlay_type

        # Validate / convert input image format
        if self.fit.data.ndim == 2:
            # Convert mono image to color image
            img = np.expand_dims(self.fit.data, -1)
            img = np.tile(img, (1,1,3))
        else:
            # fit.data is channels-first: (C, H, W)
            # get the input image in channels-last format
            img = np.transpose(self.fit.data, (1, 2, 0))
        # normalize to float, range [0,1]
        if img.dtype == np.uint16:
            print("Input image is 16 bit")
            img = img.astype(np.float32) / 65535.0
        elif img.dtype == np.uint8:
            print("Input image is 8 bit")
            img = img.astype(np.float32) / 255.0
        else:
            # should be float now, but make sure...
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            # at least some 8 bit images were represented 
            # as float in range [0.. 255/65535] -> normalize them
            maxValue = np.max(img)
            if maxValue <= (255.0 / 65535.0) or maxValue > 1.0:
                print(f"Normalizing value range to [0..1]")
                img = img / maxValue

        # calculating some scaling and sizes, 
        # depending on source image dimensions
        H, W, C = img.shape
        base_scale_pixels = max(W,H) / 100.0   # 1% of image size
        print(f"Input dimensions: {W} x {H}, Base scale: {base_scale_pixels:.1f} pixels")

        # minimum size of galaxies to annotate, in pixels
        minsize_pixels = 2
        # minimum size of annotation patches in pixels
        min_patch_size = max(20, int(round(base_scale_pixels / 2)))

        # get center coordinates from fit
        (center_ra, center_dec) = self.siril.pix2radec(W / 2, H / 2)
        print(f"Center: {center_ra, center_dec}")
    
        # get world coordinates system
        header = self.siril.get_image_fits_header(return_as = 'dict')
        wcs = WCS(header,naxis=[1,2])

        # Query Simbad
        simbad = Simbad()
        simbad.TIMEOUT = 120
        simbad.add_votable_fields("otype",
          "galdim_majaxis", "galdim_minaxis", "galdim_angle", "galdim_qual",
          "rvz_redshift", "rvz_qual", "V", "B", "R", "G", "g", "r")
    
        target_coord = SkyCoord(ra=center_ra, dec=center_dec, unit=(u.deg, u.deg), frame='icrs')
        (TL_ra, TL_dec) = self.siril.pix2radec(0, 0)
        corner_coord = SkyCoord(ra=TL_ra, dec=TL_dec, unit=(u.deg, u.deg), frame='icrs')
        radius_deg = target_coord.separation(corner_coord).deg
        # minimum size of galaxies we want to annotate
        (pixsize_ra, pixsize_dec) = self.siril.pix2radec(1, 1)
        pixel_coord = SkyCoord(ra=pixsize_ra, dec=pixsize_dec, unit=(u.deg, u.deg), frame='icrs')
        origin_coord = SkyCoord(ra=TL_ra, dec=TL_dec, unit=(u.deg, u.deg), frame='icrs')
        pixsize_arcmin = origin_coord.separation(pixel_coord).arcminute
        minsize_arcmin = minsize_pixels * pixsize_arcmin 
        radius = f"{radius_deg}d"
        # query for galaxies only or also other objects
        type_criteria = "otype='Galaxy..'"
        if self.query_opts.include_other_types:
           ism_types = " OR ".join([f"otype='{t}'" for t in (
               'SNR', 'PN', 'CL*', 'CL*..',
               'ISM', 'HII', 'SFR', 'Cld', 'MoC', 'DNe', 'GNe', 'RNe',
               'sh', 'glb', 'CGb', 'flt', 'cor', 'bub')])
           type_criteria = f"(otype='Galaxy..' OR {ism_types})"
        # limit the query to galaxies with at least minsize (or unknown size)
        size_criteria = f"galdim_majaxis>{minsize_arcmin}"
        if self.query_opts.include_unknown_size: 
           size_criteria = f"({size_criteria} OR (galdim_majaxis IS NULL))"
        # resulting query filter criteria
        criteria_opt = f"{type_criteria} AND {size_criteria}"
        
        print(f"Query radius: {radius}")
        print(f"      minimum size: {minsize_pixels} pixels ~ {minsize_arcmin:.5f}′")
        print(f"      criteria: {criteria_opt}")
        result_table = simbad.query_region(target_coord, radius, criteria=criteria_opt)

        self.progress_update.emit("Filtering query results...", 0.05)
    
        result_table.sort("galdim_majaxis", reverse=True)
        df = result_table.to_pandas()
        print(f"Simbad query results: {df.shape[0]} entries")

        # filter by position, remove anything outside the image or too close to border
        df['Pixel_Position'] = df.apply(lambda row: self.siril.radec2pix(row['ra'], row['dec']), axis=1)
        df['px'] = df['Pixel_Position'].apply(lambda x: int(round(x[0])))
        df['py'] = df['Pixel_Position'].apply(lambda x: int(round(x[1])))
        df = df[(df.px > min_patch_size) & (df.py > min_patch_size) 
              & (df.px < W-min_patch_size) & (df.py < H-min_patch_size)]
        print(f"Filtered query result by image coordinates: {df.shape[0]} entries")

        # filter by magnitude (optical bands only; objects with no optical mag are kept)
        if self.query_opts.mag_limit < 99.0:
            optical_mag = df[['V', 'B', 'R', 'G', 'g', 'r']].min(axis=1)
            df = df[optical_mag.isna() | (optical_mag <= self.query_opts.mag_limit)]
            print(f"Filtered by magnitude limit {self.query_opts.mag_limit}: {df.shape[0]} entries")

        # filter by catalog
        # get catalog/TYPE of object by simple string manipulation
        df['TYPE'] = df['main_id'].apply(lambda x: x.split(' ')[0].split('+')[0])
  
        catalogs = self.query_opts.catalogs
        if catalogs["Other"].get_selected():
            # include selected and others, exclude unselected
            exclude_types = []
            for key, value in catalogs.items():
              if not value.get_selected():
                  exclude_types.append(key)
            print(f"Excluding catalogs: {exclude_types}")
            filtered_result = Table.from_pandas(df[~df['TYPE'].isin(exclude_types)])
        else: 
            # include only selected
            include_types = []
            for key, value in catalogs.items():
              if value.get_selected():
                  include_types.append(key)
            print(f"Filtering by catalogs: {include_types}")
            filtered_result = Table.from_pandas(df[df['TYPE'].isin(include_types)])

        dfi = filtered_result.to_pandas()
        print(f"Filtered by catalog: {dfi.shape[0]} entries")

        # remove duplicates
        dfi = dfi.drop_duplicates(subset=['px', 'py'])
        print(f"Filtered after removing duplicates: {dfi.shape[0]} entries")

        if dfi.shape[0] == 0:
            self.finished.emit("No objects found in image boundary.", None)
            return
    
        # sorting order from catalogs list position
        rep_dic = {}
        for i, key in enumerate(catalogs.keys()):
            rep_dic[key] = f"{i:02d}"
        
        dfi['sorting'] = dfi.TYPE.replace(rep_dic)
        dfi.main_id = dfi.main_id.apply(lambda x: x.replace(' ', ''))
        dfi = dfi.sort_values(['sorting', 'main_id'], ascending=True).reset_index()

        # set up the plot
        sub = 1
        dpi = 200
        label_distance = base_scale_pixels / 5
        plt.style.use('dark_background')

        # Border sizes in pixels to accommodate axis labels and title.
        # These are intentionally generous so labels are never clipped.
        border_left   = max(120, int(base_scale_pixels * 1.2))
        border_bottom = max(80,  int(base_scale_pixels * 0.8))
        border_right  = 20
        border_top    = max(120,  int(base_scale_pixels))

        fig_w_px = W + border_left + border_right
        fig_h_px = H + border_bottom + border_top

        fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)

        # Position axes so the image region is exactly W×H pixels
        ax1 = fig.add_axes([
            border_left   / fig_w_px,
            border_bottom / fig_h_px,
            W             / fig_w_px,
            H             / fig_h_px,
        ], projection=wcs, label='overlays')

        ax1.imshow(img[::sub, ::sub])
        ax1.coords.grid(True, color='white', ls=':', alpha=self.output_opts.overlay_alpha)
        ax1.coords[0].set_axislabel('Right Ascension (J2000)')
        ax1.coords[1].set_axislabel('Declination (J2000)', minpad=-1)
        ax1.set_title(self.output_opts.title, fontsize=base_scale_pixels / 2.2)

        all_patches = []
        filter_idxs = []
    
        for i, row in dfi.iterrows():
            self.progress_update.emit("Creating patches...", 0.1 + (i / (10 * dfi.shape[0])))
            
            # patch border color based on catalog color
            color = self.get_color_for_type(row.TYPE)
        
            # try to derive the patch size from galaxy angular size
            # row.galdim_majaxis is in arcmin
            angular_size = row.galdim_majaxis
            size_factor = 2
            if math.isnan(angular_size):
                print(f"No angular size information for {row.main_id}")
                angular_size = 0
                if row.TYPE == 'M':
                    # Simbad is missing angular size for: Messier 8, 40, 43, 78, 82
                    # Only m82 is a galaxy, but handle the others just in case...
                    if row.main_id == "M8":
                        angular_size = 90
                    elif row.main_id == "M40":
                        angular_size = 0.86
                    elif row.main_id == "M43":
                        angular_size = 20
                    elif row.main_id == "M78":
                        angular_size = 8
                    elif row.main_id == "M82":
                        angular_size = 11.2

            if angular_size == 0:
                patch_size = min_patch_size
                diameter_pix = min_patch_size
            else:
                # angular size is the major axis diameter in arcmin
                patch_diameter_deg = angular_size / 60.0
                # convert patch size to pixels, depends on position in the grid
                tmp = self.siril.radec2pix(row.ra, row.dec + patch_diameter_deg)
                dx = tmp[0] - row.Pixel_Position[0]
                dy = tmp[1] - row.Pixel_Position[1]
                diameter_pix = math.sqrt(dx * dx + dy * dy)
                patch_size = int(round(diameter_pix * size_factor))
                patch_size = max(min_patch_size, patch_size)
            
            # dynamic font size, depending on object size
            fontsize = (base_scale_pixels / 4) + math.sqrt(patch_size - min_patch_size) / 4
                    
            # type dependent style
            if row.main_id == main_object:
                fontsize += 2
                color = 'white'
            elif row.TYPE == 'LEDA':
                # LEDA apparently sometimes has large errors on galdim_majaxis
                # resulting in much too large patches. Usually, LEDA galaxies
                # are small...
                if not math.isnan(row.galdim_majaxis) and row.galdim_majaxis > 1.8:
                    patch_size = min_patch_size
                    diameter_pix = min_patch_size
                    print(f"Unlikely large LEDA galaxy {row.main_id}: {row.galdim_majaxis} -> patch size {patch_size}")

            # show full annotation with object name for larger objects
            annotation_text = str(i + 1)
            min_size_for_label = 0.04 * (100 - self.output_opts.label_amount) * base_scale_pixels 
            if patch_size > min_size_for_label:
                annotation_text = f"{annotation_text}: {row.main_id}"
        
            # clip the patch size on image borders
            clipped = min(patch_size, (W - row.px) * 2)
            clipped = min(clipped, (H - row.py) * 2)
            clipped = min(clipped, row.px * 2)
            clipped = min(clipped, row.py * 2)
        
            x1 = row.px - clipped // 2
            x2 = row.px + clipped // 2
            y1 = H-row.py - clipped // 2
            y2 = H-row.py + clipped // 2
        
            if overlay_type == OverlayType.BOXES:
                # annotation rectangle
                rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                    alpha=self.output_opts.overlay_alpha, linewidth=1, edgecolor=color, facecolor='none')
                ax1.add_patch(rect)
                text_y = y1 - label_distance
                v_align = 'top'
                if text_y < (2*fontsize):
                    text_y = min(y2 + label_distance, H - (3*fontsize))
                    v_align = 'bottom'

            elif overlay_type == OverlayType.ELLIPSES:
                # annotation ellipse
                el_width = max(min_patch_size, 1.2 * diameter_pix)
                el_height = el_width
                angle = 0
                if not math.isnan(row.galdim_minaxis):
                   el_height = el_width * (row.galdim_minaxis / row.galdim_majaxis)
                   try:
                      angle = float(row.galdim_angle)
                      pa = math.radians(angle)
                      x1a, y1a = self.siril.radec2pix(row.ra, row.dec)
                      x2a, y2a = self.siril.radec2pix(row.ra, row.dec + pixsize_arcmin)
                      base_angle = math.degrees(math.atan2(y1a - y2a, x2a - x1a))
                      angle += base_angle
                   except TypeError:
                      # can't get angle, fall back to circle
                      el_height = el_width
                ellipse = Ellipse((row.px, H-row.py), width=el_width, height=el_height, angle=angle,
                          alpha=self.output_opts.overlay_alpha, linewidth=1, edgecolor=color, facecolor='none')
                ax1.add_patch(ellipse)
                widthComp = abs(math.sin(math.radians(angle)))
                text_distance = (widthComp * el_width + (1-widthComp) * el_height) / 2.0 + label_distance
                text_y = H-row.py - text_distance
                v_align = 'top'
                if text_y < (2*fontsize):
                    text_y = min(H-row.py + text_distance, H - (3*fontsize))
                    v_align = 'bottom'
        
            else:
                # annotation circle
                annot_radius = max(min_patch_size, 1.2 * diameter_pix / 2.0) 
                circ = Circle((row.px, H-row.py), radius=annot_radius,
                    alpha=self.output_opts.overlay_alpha, linewidth=1, edgecolor=color, facecolor='none')
                ax1.add_patch(circ)
                text_y = H-row.py - label_distance - annot_radius
                v_align = 'top'
                if text_y < (2*fontsize):
                    text_y = min(H-row.py + label_distance  + annot_radius, H - (3*fontsize))
                    v_align = 'bottom'
                
            ax1.text(row.px, text_y, annotation_text, 
                ha='center', va=v_align, color=color, alpha=self.output_opts.overlay_alpha, fontsize=fontsize)
        
            patch = img[y1:y2, x1:x2]
            all_patches.append(patch)
            filter_idxs.append(i)

        # Measure the actual space consumed by axis labels and title via a
        # dry-run render, then resize the figure so the image area stays
        # exactly W×H pixels with no wasted border space.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight_bb = fig.get_tightbbox(renderer)          # full content bbox in inch
        ax_bb    = ax1.get_window_extent(renderer)      # axes (image) region in display pixels
        
        # Overrun on each side: how much content sticks out beyond the axes edge
        pad = 4  # small constant padding in pixels to avoid clipping
        over_left   = max(0, ax_bb.x0 - (dpi * tight_bb.x0))   + pad
        over_bottom = max(0, ax_bb.y0 - (dpi * tight_bb.y0))   + pad
        over_right  = max(over_left, (dpi * tight_bb.x1) - ax_bb.x1 + pad)
        over_top    = max(0, (dpi * tight_bb.y1) - ax_bb.y1)   + pad

        new_fig_w_px = W + over_left + over_right
        new_fig_h_px = H + over_bottom + over_top
        fig.set_size_inches(new_fig_w_px / dpi, new_fig_h_px / dpi)
        ax1.set_position([
            over_left   / new_fig_w_px,
            over_bottom / new_fig_h_px,
            W           / new_fig_w_px,
            H           / new_fig_h_px,
        ])
        
        self.progress_update.emit("Saving overlay image...", 0.2)
        plt.savefig(output_overlay_fname, dpi=dpi)
        self.progress_update.emit("Finished overlay image", 0.3)
        overlay_image = plt.imread(output_overlay_fname)

        # Calculate numbers of rows and columns for the patch table
        n = len(all_patches)
        mincols = 6 if (self.output_opts.logo_path != "") else 5 
        ncols = max(mincols, int(np.floor(np.sqrt(n))))
        nrows = int(np.ceil(n / ncols))
        print(f"Grid size: nrows={nrows}, ncols={ncols}")

        # Resize and process patches
        # Image size for each patch in the table,
        # based on the size of the overlay image and the number of columns
        # (avoid having to upscale afterwards!)
        tab_patch_size = int(round(1.2 * overlay_image.shape[1] / ncols))
        # 3 inch is a good size for the patches, so that the label font size
        # matches nicely
        tab_patch_size_inch = 3
        # adapt the output dpi of the table so that the resolution matches
        # without scaling 
        dpi = tab_patch_size / tab_patch_size_inch
        print(f"Patch size: {tab_patch_size}px / {tab_patch_size_inch}inch, {dpi:.2f}dpi")

        self.progress_update.emit("Resizing patch images...", 0.4)
        all_patches_resized = [resize(patch, (tab_patch_size, tab_patch_size)) for patch in all_patches]
        all_patches = np.array(all_patches_resized)
        self.progress_update.emit("Patch images resized", 0.5)

        # Create the subplots with the adjusted dimensions
        fig, axarr = plt.subplots(nrows, ncols, figsize=(ncols * tab_patch_size_inch, nrows * tab_patch_size_inch))

        # Create a filtered DataFrame with the desired indices
        dft = dfi.iloc[filter_idxs].reset_index()

        for i, row in dft.iterrows():
            if nrows > 1:
                ax = axarr[i // ncols, i % ncols]
            else:
                ax = axarr[i]

            color = self.get_color_for_type(row.TYPE)
            ax.imshow(all_patches[i][::-1])
            ax.set_title(row.main_id, fontsize=12, color=color)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
            label_spacing = tab_patch_size / 100    
            ax.text(label_spacing, label_spacing, str(i + 1), ha='left', va='top', color='white', fontsize=18)
            if not math.isnan(row.rvz_redshift):
              redshift_text = f"z={row.rvz_redshift: .5f}"
              ax.text(tab_patch_size / 2, tab_patch_size - label_spacing, 
                      redshift_text, ha='center', va='bottom', color='white', fontsize=12)


        for i in range(n, nrows * ncols):
            if nrows > 1:
                ax = axarr[i // ncols, i % ncols]
            else:
                ax = axarr[i]
            ax.axis('off')

        # If logo is requested, put it in the last grid cell
        # (as long as the cell is free)
        if (self.output_opts.logo_path != "") and (nrows * ncols > n):
            logo_img = plt.imread(self.output_opts.logo_path)
            if nrows > 1:
                axarr[nrows - 1, ncols - 1].imshow(logo_img)
            else:
                axarr[ncols - 1].imshow(logo_img)

        self.progress_update.emit("Creating thumbnail table...", 0.6)
        plt.tight_layout()
        plt.savefig(output_table_fname, bbox_inches='tight', pad_inches=.1, dpi=dpi)
        self.progress_update.emit("Saved thumbnail table image", 0.7)
        table_image = plt.imread(output_table_fname)

        # Resize table_image to match overlay_image dimensions
        self.progress_update.emit("Creating combined output image...", 0.8)
        output_shape = (int(table_image.shape[0] * (overlay_image.shape[1] / table_image.shape[1])), overlay_image.shape[1])
        table_image_scaled = (resize(table_image, output_shape) * 255).astype(np.uint8)
        im = Image.fromarray(np.vstack([(overlay_image * 255).astype(np.uint8), table_image_scaled])[:, :, :3])
        self.progress_update.emit("Saving combined output image...", 0.9)
        im.save(output_fname)
        
        print("")
        print("===================================================")
        print("Created output image files:")
        print("  overlay:  ", output_overlay_fname)
        print("  table:    ", output_table_fname)
        print("  combined: ", output_fname)
        print("===================================================")
    
        self.progress_update.emit("Finished.", 1.0)
        self.finished.emit(None, dfi.shape[0])


    def get_color_for_type(self, row_type):
        try:
            return self.query_opts.catalogs[row_type].color
        except KeyError:
            if self.query_opts.catalogs["Other"].get_selected():
              return self.query_opts.catalogs["Other"].color
            else:
              # unexpected type, mark in red
              return '#ff0000'



# ============================================================================
# GUI Window
# ============================================================================

class ColorButton(QPushButton):
    """
    Custom Qt Widget to show and edit a chosen color.
    """
    colorChanged = pyqtSignal(object)

    def __init__(self, color=None, default_color=None):
        super().__init__()

        self._color = None
        self._default = default_color if default_color is not None else color
        self.pressed.connect(self.onColorPicker)

        # Set the initial/default state.
        self.setColor(color)

    def setColor(self, color):
        if color != self._color:
            self._color = color
            self.colorChanged.emit(color)

        if self._color:
            self.setStyleSheet("background-color: %s;" % self._color)
        else:
            self.setStyleSheet("")

    def color(self):
        return self._color

    def onColorPicker(self):
        dlg = QColorDialog(self)
        if self._color:
            dlg.setCurrentColor(QColor(self._color))

        if dlg.exec():
            self.setColor(dlg.currentColor().name())

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            self.setColor(self._default)

        return super().mousePressEvent(e)


class AnnotationsScriptGUI(QMainWindow):
    """ 
    This class provides the GUI and related callbacks 
    """
    
    def __init__(self, siril):
        super().__init__()
        self.setWindowTitle(f"Galaxy Annotations - v{VERSION}")
        
        self.siril = siril

        # load options from config file
        self.query_opts, self.output_opts = self.load_config_file()
                       
        # get the default output file name and title 
        # from input image file name
        basename = os.path.basename(self.siril.get_image_filename())
        filename, extension = os.path.splitext(basename)
        self.output_opts.output = "annotated_" + filename 
        self.output_opts.title = filename

        # Create the UI 
        self.create_widgets()

    def _browse_logo_file(self, file_attr: str, lineedit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Select a Logo Image File", "", 
            "Image files (*.png *.jpg *.jpeg *.ico *.bmp *.gif);;All files (*)")
        if path:
            lineedit.setText(path)
            setattr(self, file_attr, path)    

    def create_widgets(self):
        """
        Create the GUI's widgets, connect signals etc. 
        """
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        # layout.setSpacing(10)
        
        title_label = QLabel("Galaxy Annotations")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        subtitle_label = QLabel("Find and annotate galaxies in your image with Simbad")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_font = QFont()
        subtitle_font.setPointSize(9)
        subtitle_label.setFont(subtitle_font)
        layout.addWidget(subtitle_label)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        # Output options frame
        output_group = QGroupBox("Output Options")
        output_layout = QGridLayout(output_group)
        
        # image title
        row = 0
        output_layout.addWidget(QLabel("Title: "), row, 0)
        self.title_line = QLineEdit(self.output_opts.title)
        self.title_line.setToolTip( 
            "The title to show on top of the overlay image")
        output_layout.addWidget(self.title_line, row, 1)
        
        # Logo file selection
        row = row + 1
        output_layout.addWidget(QLabel("Logo: "), row, 0)
        self.logo_line = QLineEdit(self.output_opts.logo_path)
        self.logo_line.setToolTip(
            "A logo image that will be shown at the end of the result table, if there is a free cell")
        output_layout.addWidget(self.logo_line, row, 1)
        btn = QPushButton("", icon=QIcon.fromTheme("document-open"))
        btn.setToolTip("Select a logo image file")
        btn.clicked.connect(lambda: self._browse_logo_file("self.output_opts.logo file", self.logo_line))
        output_layout.addWidget(btn, row, 2)
        
        # Output file name
        """
        row = row + 1
        output_layout.addWidget(QLabel("Output file: "), row, 0)
        self.output_line = QLineEdit(self.output_opts.output)
        output_layout.addWidget(self.output_line, row, 1)
        """

        # Overlay settings: alpha and overlay type...
        row = row + 1
        output_layout.addWidget(QLabel("Overlay: "), row, 0)
        self.overlay_type_combo = QComboBox()
        self.overlay_type_combo.setToolTip( 
            "The type of annotations outlines to draw around galaxies.")
        for typeEntry in OverlayType:
            self.overlay_type_combo.addItem(typeEntry.value)
        self.overlay_type_combo.setCurrentText(self.output_opts.overlay_type.value)
        output_layout.addWidget(self.overlay_type_combo, row, 1)
        row = row + 1
        alpha_opts_box = QHBoxLayout()
        alpha_opts_box.addWidget(QLabel("alpha: "))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(int(round(self.output_opts.overlay_alpha * 100)))
        self.alpha_slider.setToolTip(
            "Adjust the visibility of the annotation overlays.\n"
            "Smaller values result in more transparent overlays.")
        alpha_opts_box.addWidget(self.alpha_slider, 1)
        self.alpha_value_label = QLabel(f"{self.alpha_slider.value() / 100:.2f}")
        alpha_opts_box.addWidget(self.alpha_value_label)
        self.alpha_slider.valueChanged.connect(lambda v: self.alpha_value_label.setText(f"{v / 100:.2f}"))
        output_layout.addLayout(alpha_opts_box, row, 1)
        row = row + 1
        label_opts_box = QHBoxLayout()
        label_opts_box.addWidget(QLabel("labels: "))
        minLabel = QLabel("only largest")
        minLabel.setFont(subtitle_font)
        label_opts_box.addWidget(minLabel)
        self.labels_slider = QSlider(Qt.Orientation.Horizontal)
        self.labels_slider.setRange(0, 100)
        self.labels_slider.setValue(self.output_opts.label_amount)
        self.labels_slider.setToolTip(
            "Adjust the size of galaxies that get full labels.")
        label_opts_box.addWidget(self.labels_slider, 1)
        maxLabel = QLabel("all")
        maxLabel.setFont(subtitle_font)
        label_opts_box.addWidget(maxLabel)
        # self.labels_slider.valueChanged.connect(lambda v: self.labelopt_label.setText(f"{v}"))
        output_layout.addLayout(label_opts_box, row, 1)

        # Load in Siril options
        row = row + 1
        output_layout.addWidget(QLabel("Load in Siril: "), row, 0)
        load_opts_box = QHBoxLayout()
        self.load_in_siril = 'C'
        rbtn_c = QRadioButton("Combined")
        rbtn_c.setToolTip("Load the combined result image in Siril")
        rbtn_c.setChecked(True)
        rbtn_c.toggled.connect(lambda: self.on_toggle_load(rbtn_c.isChecked(), 'C'))
        load_opts_box.addWidget(rbtn_c)
        rbtn_o = QRadioButton("Overlay")
        rbtn_o.setToolTip("Load the result image with annotation overlays in Siril")
        rbtn_o.toggled.connect(lambda: self.on_toggle_load(rbtn_o.isChecked(), 'O'))
        load_opts_box.addWidget(rbtn_o)
        rbtn_t = QRadioButton("Table")
        rbtn_t.setToolTip("Load the thumbnail table image in Siril")
        rbtn_t.toggled.connect(lambda: self.on_toggle_load(rbtn_t.isChecked(), 'T'))
        load_opts_box.addWidget(rbtn_t)
        rbtn_n = QRadioButton("None")
        rbtn_n.setToolTip("Just create output files without loading anything in Siril")
        rbtn_n.toggled.connect(lambda: self.on_toggle_load(rbtn_n.isChecked(), ''))
        load_opts_box.addWidget(rbtn_n)
        output_layout.addLayout(load_opts_box, row, 1, 1, 2)
        
        layout.addWidget(output_group)


        # Query options frame
        query_group = QGroupBox("Query Options")
        query_layout = QVBoxLayout(query_group)
        
        self.include_unknown_size_check = QCheckBox("Include galaxies without size data")
        self.include_unknown_size_check.setChecked(self.query_opts.include_unknown_size)
        self.include_unknown_size_check.setToolTip(
            "Galaxies without size information in Simbad will be annotated as small circles.\n"
            "You may want to exclude these if you select many catalogs.")
        # self.include_unknown_size_check.stateChanged.connect(self.update_query_opts)
        query_layout.addWidget(self.include_unknown_size_check)
        print(f"other types: {self.query_opts.include_other_types} unknown size: {self.query_opts.include_unknown_size}")
        self.include_other_types_check = QCheckBox("Include some other object types")
        self.include_other_types_check.setChecked(self.query_opts.include_other_types)
        self.include_other_types_check.setToolTip(
            "Include not only galaxies, but also some other object types, like\n"
            "planetary nebula (PN) or globular clusters. May or may not work\n"
            "depending on the field of view of your image.")
        query_layout.addWidget(self.include_other_types_check)

        mag_box = QHBoxLayout()
        mag_box.addWidget(QLabel("Magnitude limit (optical): "))
        from PyQt6.QtWidgets import QDoubleSpinBox
        self.mag_limit_spin = QDoubleSpinBox()
        self.mag_limit_spin.setRange(1.0, 99.0)
        self.mag_limit_spin.setSingleStep(0.5)
        self.mag_limit_spin.setDecimals(1)
        self.mag_limit_spin.setValue(self.query_opts.mag_limit)
        self.mag_limit_spin.setSpecialValueText("no limit")
        self.mag_limit_spin.setToolTip(
            "Exclude objects dimmer than this magnitude (B, V, R, G, g, or r band — whichever is available).\n"
            "Objects with no optical magnitude data are always included.\n"
            "Set to 99 to disable the filter.")
        mag_box.addWidget(self.mag_limit_spin)
        mag_box.addStretch()
        query_layout.addLayout(mag_box)
        
        layout.addWidget(query_group)        


        # Catalogs 
        catalogs_group = QGroupBox("Catalogs")
        catalogs_layout = QVBoxLayout(catalogs_group)

        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll.setWidget(scroll_content)
        scroll_content_grid = QGridLayout(scroll_content)
        scroll_content_grid.setVerticalSpacing(1)
        
        self.catalog_checkboxes = []
        for i, (key, value) in enumerate(self.query_opts.catalogs.items()):
            checkbox = QCheckBox(key)
            checkbox.setChecked(value.get_selected())
            checkbox.stateChanged.connect(value.set_selection)
            scroll_content_grid.addWidget(checkbox, i, 0)
            self.catalog_checkboxes.append(checkbox)
            desc = value.description if value.description else key
            scroll_content_grid.addWidget(QLabel(desc), i, 1)
            colBtn = ColorButton(color=value.color, default_color=value.color_default)
            colBtn.colorChanged.connect(value.set_color)
            colBtn.setMaximumSize(22,22)
            colBtn.setToolTip(
                "Select the annotation color for this catalog.\n"
                "Right mouse button to reset to default color.")
            scroll_content_grid.addWidget(colBtn, i, 2)

        catalogs_layout.addWidget(scroll)
        
        select_actions_box = QHBoxLayout()
        select_actions_box.addWidget(QLabel("Select: "))
        btn = QPushButton("All")
        btn.clicked.connect(self.select_all)
        select_actions_box.addWidget(btn)
        btn = QPushButton("None")
        btn.clicked.connect(self.select_none)
        select_actions_box.addWidget(btn)
        btn = QPushButton("Defaults")
        btn.clicked.connect(self.select_default)
        select_actions_box.addWidget(btn)
        catalogs_layout.addLayout(select_actions_box)

        layout.addWidget(catalogs_group)

        # Apply / Close action buttons
        apply_actions_box = QHBoxLayout()
        self.btn_close = QPushButton("Close")
        self.btn_close.setToolTip("Close, no changes will be made to the current image.")
        self.btn_close.clicked.connect(self.close)
        apply_actions_box.addWidget(self.btn_close)
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setToolTip(
            "Create the annotated output images.\n"
            "The image file names are based on the input image file name.")
        self.btn_apply.setDefault(True)
        self.btn_apply.clicked.connect(self.start_processing)
        apply_actions_box.addWidget(self.btn_apply)
        apply_actions_box.setContentsMargins(6,16,6,6)
        layout.addLayout(apply_actions_box)
        
    def on_toggle_load(self, checked, code):
        if checked:
            self.load_in_siril = code

    def start_processing(self):
        try:
            # Get the thread
            with self.siril.image_lock():

                # Get current image
                fit = self.siril.get_image()
                fit.ensure_data_type(np.float32)

                # ensure is plate solved
                try:
                    self.siril.pix2radec(0, 0)
                except ValueError:
                    self.siril.log("The image is not plate solved", color=s.LogColor.RED)
                    self.siril.error_messagebox("The image is not plate solved")
                    return

                # Disable controls during processing
                self.btn_apply.setEnabled(False)
                self.btn_close.setEnabled(False)

                # get option values from widgets
                self.output_opts.title = self.title_line.text()
                # self.output_opts.output = self.output_line.text()
                self.output_opts.logo_path = self.logo_line.text()
                self.output_opts.overlay_type = OverlayType.from_str(self.overlay_type_combo.currentText())
                self.output_opts.overlay_alpha = 0.01 * self.alpha_slider.value()
                self.output_opts.label_amount = self.labels_slider.value()
                self.query_opts.include_unknown_size = self.include_unknown_size_check.isChecked()
                self.query_opts.include_other_types = self.include_other_types_check.isChecked()
                self.query_opts.mag_limit = self.mag_limit_spin.value()

                # update config file
                self.save_config_file(self.query_opts, self.output_opts)

                self.thread = QThread(self)
                self.worker = AnnotationWorker(self.siril, fit, self.query_opts, self.output_opts)
                self.worker.moveToThread(self.thread)
                self.thread.started.connect(self.worker.run)
                self.worker.progress_update.connect(self.on_progress)
                self.worker.finished.connect(self.on_finished)
                self.worker.finished.connect(self.thread.quit)
                self.worker.finished.connect(self.worker.deleteLater)
                self.thread.finished.connect(self.thread.deleteLater)
                self.thread.start()

        except SirilError as e:
            QMessageBox.critical(self, "Error", str(e))

    
    def on_progress(self, message, percent):
        self.siril.update_progress(message, percent)
    
    def on_finished(self, error, found):
        # Re-enable controls
        self.btn_apply.setEnabled(True)
        self.btn_close.setEnabled(True)
        
        if error:
            self.on_progress("Error", 0)
            QMessageBox.warning(self, "Processing Error", error)
            return
    
        if found > 0:
            self.siril.log("Annotation images created successfully.", color=s.LogColor.GREEN)
            # Optionally load the annotated image in Siril
            if self.load_in_siril == 'C':
                self.siril.cmd("load", "\"" + get_combined_filename(self.output_opts.output) + "\"")
            elif self.load_in_siril == 'O':
                self.siril.cmd("load", "\"" + get_overlay_filename(self.output_opts.output) + "\"")
            elif self.load_in_siril == 'T':
                self.siril.cmd("load", "\"" + get_table_filename(self.output_opts.output) + "\"")
        else:
            self.on_progress("No data", 0)

            
    def select_all(self):
        """ Select all catalogs """
        for checkbox in self.catalog_checkboxes:
            checkbox.setChecked(True)
    
    def select_none(self):
        """ Select no catalogs """
        for checkbox in self.catalog_checkboxes:
            checkbox.setChecked(False)

    def select_default(self):
        """ Select default set of catalogs """
        i = 0
        for key, value in self.query_opts.catalogs.items():
            self.catalog_checkboxes[i].setChecked(value.selection_default)
            i = i + 1

    def load_config_file(self):
        """
        Check for a saved options in the configuration file.
        Returns (query_opts, output_opts) with the config values or defaults if not found.
        """
        config_dir = self.siril.get_siril_configdir()
        config_file_path = os.path.join(config_dir, CONFIG_FILENAME)
        
        # create options structures with default values...
        # Catalog definitions
        catalogs = {
            'M': CatalogEntry('Messier Catalog', '#3e6ebe', True), 
            'IC': CatalogEntry('Index Catalogue', '#b2c5eb', True),
            'NGC': CatalogEntry('New General Catalogue', '#9ae483', True),
            'MGC': CatalogEntry('Millennium Galaxy Catalogue', '#30a500', True),
            'UGC': CatalogEntry('Uppsala General Catalogue', '#3abed1', True),
            'MCG': CatalogEntry('Morphological Catalogue of Galaxies', '#955ec2', True),
            'Mrk': CatalogEntry('Markarian galaxies', '#fbbd70', True),
            'LEDA': CatalogEntry('Lyon-Meudon Extragalactic Database', '#c29d94', True),
            'Z': CatalogEntry('Zwicky Catalogue of galaxies and clusters', '#fb9795', True),
            'Gaia': CatalogEntry('Gaia catalogues', '#c6aed8', True),
            '2MASX': CatalogEntry('Two Micron All Sky Survey, Extended', '#895447', True),
            'SDSS': CatalogEntry('Sloan Digital Sky Survey', '#b2c5eb', False),
            'SDSSCGB': CatalogEntry('SDSS DR6 Compact Group Catalogue B', '#b2c5eb', False),
            'UGCA': CatalogEntry('Uppsala Selected non-UGC Galaxies', '#f5b3d3', True),
            'MASS': CatalogEntry(None, '#c8c8c8', True),
            'MFGC': CatalogEntry(None, '#b9c200', True),
            '2MFGC': CatalogEntry('2MASS Flat Galaxy Catalog', '#d9df85', True),
            'AGC': CatalogEntry('Arecibo General Catalog', '#895d8a', True),
            'FIRST': CatalogEntry('FIRST Survey Catalogs', '#a3dae7', False),
            '2MASS': CatalogEntry('Two Micron All Sky Survey', '#895447', False),
            'LDN': CatalogEntry('Lynds Dark Nebula Catalog', '#a0522d', False),
            'LBN': CatalogEntry('Lynds Bright Nebula Catalog', '#ffd700', False),
            'Other': CatalogEntry('Other catalogs', '#ad7fa8', False)
        }
        
        query_opts = QueryOptions(catalogs)
        output_opts = OutputOptions()

        if os.path.isfile(config_file_path):
            config = configparser.ConfigParser()
            config.optionxform = str
            try:
                config.read(config_file_path)
                output_opts.logo_path = config['Output']['logo_path']
                if not os.path.isfile(output_opts.logo_path):
                    output_opts.logo_path = ""
                output_opts.label_amount = config['Output'].getint('label_amount')
                output_opts.overlay_alpha = config['Output'].getfloat('overlay_alpha')
                output_opts.overlay_type = OverlayType.from_str(config['Output']['overlay_type'])
                query_opts.include_unknown_size = config['Query'].getboolean('include_unknown_size', fallback=True)
                query_opts.include_other_types = config['Query'].getboolean('include_other_types', fallback=False)
                query_opts.mag_limit = config['Query'].getfloat('mag_limit', fallback=99.0)
                
                for key,value in config['Catalogs'].items():
                    value_dict = ast.literal_eval(value)
                    entry = catalogs[key]
                    entry.set_color(value_dict['color'])
                    entry.set_selection(value_dict['selection'])
                    
            except configparser.ParsingError:
                self.siril.log("Failed to load config file, reset to defaults.", color=s.LogColor.RED)

        return query_opts, output_opts

    def save_config_file(self, query_opts, output_opts):
        """
        Save the options to the configuration file.
        """
        config_dir = self.siril.get_siril_configdir()
        config_file_path = os.path.join(config_dir, CONFIG_FILENAME)
        
        config = configparser.ConfigParser()
        config.optionxform = str
        config['Output'] = output_opts.get_config()
        config['Query'] = query_opts.get_config()
        # remember all catalog selections/colors that are not defaults...
        cat_config = {}
        for key,value in query_opts.catalogs.items():
            if (value.color_default != value.color) or (value.selection is not None and value.selection != value.selection_default):
                entry_conf = {"color": value.color, "selection": value.selection}
                cat_config[key] = entry_conf
        config['Catalogs'] = cat_config

        with open(config_file_path, 'w') as configfile:
            config.write(configfile)


# ============================================================================
# Main Entry Point
# ============================================================================
    
def get_output_filename(output_basename, suffix=''):
    filename, extension = os.path.splitext(output_basename)
    if extension == '':
        extension = '.png'
    else:
        # is extension a supported output formats for matplotlib savefig?
        if not extension.lower() in (".eps", ".jpeg", ".jpg", ".pdf", ".pgf", 
                                     ".png", ".ps", ".raw", ".rgba", ".svg", 
                                     ".svgz", ".tif", ".tiff", ".webp"):
            extension = '.png'
            filename = output_basename
    return f"{filename}{suffix}{extension}"
    
def get_overlay_filename(output_basename):
    return get_output_filename(output_basename, '_overlay')
    
def get_table_filename(output_basename):
    return get_output_filename(output_basename, '_table')
    
def get_combined_filename(output_basename):
    return get_output_filename(output_basename, '')


def main():
    try:
        siril = s.SirilInterface()
        try:
            siril.connect()
        except s.SirilConnectionError:
            app = QApplication(sys.argv)
            QMessageBox.critical(None, "Error", "Failed to connect to Siril")
            sys.exit(1)
        
        try:
            siril.cmd("requires", "1.4.0-beta2")
        except s.CommandError:
            return
        
        if siril.is_cli() and len(sys.argv) > 1:
            print("This script requires GUI mode")
            sys.exit(1)

        # Initial checks: example - check if an image is loaded
        if not siril.is_image_loaded():
            app = QApplication(sys.argv)
            QMessageBox.critical(None, "Error", 
                "No image is loaded.\n"
                "To use this script, please make sure that a stretched and plate-solved image is loaded.")
            sys.exit(1)
        
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        app.setApplicationName("Galaxy Annotations")
        
        # Get dark/light theme configuration from Siril
        try:
            theme_value = siril.get_siril_config("gui", "theme")
            if (theme_value == 0):
                QGuiApplication.styleHints().setColorScheme(Qt.ColorScheme.Dark)
        except Exception as e:
            raise AttributeError(f"Unable to retrieve theme configuration: {e}") from e
                
        window = AnnotationsScriptGUI(siril)
        window.show()
        sys.exit(app.exec())
    
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
