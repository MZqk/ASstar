"""
Siril Plugin: Asteroid & Comet Finder

Searches for asteroids and comets in the current FITS image using SkyBot service.
Allows user to select an object and display its predicted path on the image.

Features:
- Search known asteroids/comets in image field of view
- Interactive PyQt6 GUI with sortable results
- Display object trajectory on FITS image with markers
- Calculate and display object magnitude at different times
- Support for custom object searches via JPL Horizons database

Dependencies:
- sirilpy: Siril interface
- astropy: Coordinate transformations and time handling
- astroquery: Access to SkyBot and JPL Horizons services
- PyQt6: GUI framework

Author: Alexand Salamatov (goopilot@gmail.com)
Version: 1.0.1
SPDX-License-Identifier: GPL-3.0-or-later
"""


import sys
import argparse
import sirilpy as s
s.ensure_installed("astropy", "astroquery", "tzlocal", "PyQt6")
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton,
                             QTableWidget, QTableWidgetItem, QFrame, QMessageBox, QTextEdit, QHeaderView, QScrollArea)
from datetime import datetime, timedelta, UTC
from typing import List, Dict
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astroquery.imcce import Skybot
from astroquery.jplhorizons import Horizons
import tzlocal

VERSION = "1.0.1"

# ============================================================================
# CONFIGURATION & DATA
# ============================================================================


class AppConfig:
    """Application configuration and constants.

    Stores default values and configuration parameters
    """
    DEFAULT_SEARCH_RADIUS = 3.0
    DEFAULT_MAX_MAGNITUDE = 20.0
    DEFAULT_TRAJECTORY_MINS = 10000
    DEFAULT_TRAJECTORY_POINTS = 30
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    DEFAULT_LOCATION = '729'
    COLUMN_HEADERS = ["Name", "Type",
                      "Separation (°)", "RA (HMS)", "Dec (DMS)", "Mag"]


# ============================================================================
# SKYBOT DATABASE & ORBITAL MECHANICS
# ============================================================================


class SkyBotDatabase:
    """Interface to SkyBot and JPL Horizons services for asteroid/comet searches.

    Provides methods to query the SkyBot service for known asteroids and comets
    in a specified region, and uses JPL Horizons to predict object trajectories
    over time.
    """

    def __init__(self):
        """Initialize the SkyBot service interface."""
        self.skybot = Skybot()

    def search_region(self, ra_center: float, dec_center: float,
                      search_radius: float, obs_date: str,
                      max_magnitude: float = 15.0, obs_id: int = AppConfig.DEFAULT_LOCATION) -> List[Dict]:
        """Search SkyBot service for asteroids/comets in a region.

        Queries the SkyBot web service to find all known asteroids and comets
        within a specified cone search area at a given observation date.

        Args:
            ra_center (float): Right ascension of search center in degrees.
            dec_center (float): Declination of search center in degrees.
            search_radius (float): Search radius in degrees.
            obs_date (str): Observation date in ISO format (YYYY-MM-DD HH:MM:SS).
            max_magnitude (float): Maximum magnitude to include in results. Defaults to 15.0.
            obs_id (int): Observatory identifier for location. Defaults to AppConfig.DEFAULT_LOCATION.

        Returns:
            List[Dict]: List of found objects with keys: name, type, ra, dec, magnitude,
                       separation, and skybot_data (raw SkyBot row data).
        """
        results = []
        try:
            print(
                f"Querying SkyBot for RA={ra_center:.2f}°, Dec={dec_center:.2f}°, Radius={search_radius}°")

            # Convert parameters for SkyBot
            coo = SkyCoord(ra=ra_center*u.deg, dec=dec_center*u.deg)
            rad = search_radius*u.deg
            epoch = Time(obs_date)
            # Query SkyBot service using cone_search
            skybot_results = self.skybot.cone_search(
                coo=coo,
                rad=rad,
                epoch=epoch,
                location=obs_id,
                position_error=120,
                find_planets=False,
                find_asteroids=True,
                find_comets=True
            )

            if skybot_results is None or len(skybot_results) == 0:
                print("No objects found from SkyBot service")
                return []

            print(
                f"SkyBot found {len(skybot_results)} objects, showing those with mag <= {max_magnitude}")

            # Process SkyBot results
            for row in skybot_results:
                try:
                    # Extract key information from SkyBot result
                    obj_name = row['Name']
                    obj_type = str(row['Type']).lower()

                    # Handle both Quantity and regular float values
                    ra = float(row['RA'].value) if hasattr(
                        row['RA'], 'value') else float(row['RA'])
                    dec = float(row['DEC'].value) if hasattr(
                        row['DEC'], 'value') else float(row['DEC'])
                    magnitude = float(row['V'].value) if hasattr(
                        row['V'], 'value') else float(row['V'])

                    # Filter by magnitude
                    if magnitude > max_magnitude:
                        continue

                    # Calculate separation from center
                    center_coord = SkyCoord(
                        ra=ra_center*u.deg, dec=dec_center*u.deg)
                    obj_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
                    separation = center_coord.separation(obj_coord).deg

                    results.append({
                        'name': obj_name,
                        'type': obj_type,
                        'ra': ra,
                        'dec': dec,
                        'magnitude': magnitude,
                        'separation': separation,
                        'skybot_data': dict(row)
                    })
                except Exception as e:
                    print(f"Error processing SkyBot result: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Sort by separation
            results.sort(key=lambda x: x['magnitude'])

        except Exception as e:
            print(f"Error searching SkyBot: {e}")
            import traceback
            traceback.print_exc()

        return results

    def predict_trajectory(self, obj_name: str, orbital_elements: Dict,
                           start_date: str, num_points: int = 20,
                           minutes_ahead: int = 60, obs_lon: float = 0.0,
                           obs_lat: float = 0.0, obs_elev: float = 0.0) -> List[Dict]:
        """Query JPL Horizons for object ephemerides (position, magnitude over time).

        Retrieves predicted positions and magnitudes for an object at multiple time points
        from the JPL Horizons service. Useful for drawing trajectory paths on images.

        Args:
            obj_name (str): Name or ID of the object (e.g., 'Ceres', '1P/Halley').
            orbital_elements (Dict): Orbital elements dict (currently unused, for future expansion).
            start_date (str): Central observation date in ISO format (YYYY-MM-DD HH:MM:SS).
            num_points (int): Number of trajectory points to compute. Defaults to 20.
            minutes_ahead (int): Minutes before and after start_date to query. Defaults to 60.
            obs_lon (float): Observatory longitude in degrees (positive east). Defaults to 0.0.
            obs_lat (float): Observatory latitude in degrees (positive north). Defaults to 0.0.
            obs_elev (float): Observatory elevation above sea level in meters. Defaults to 0.0.

        Returns:
            List[Dict] | None: List of trajectory points with keys: date, date_str, ra, dec, magnitude.
                               Returns None if object is ambiguous or cannot be resolved.
        """
        try:
            print(f"Querying Horizons for {obj_name}...")

            midpoint_time = Time(start_date).to_datetime()
            start_time = midpoint_time - timedelta(minutes=minutes_ahead)
            end_time = midpoint_time + timedelta(minutes=minutes_ahead)

            # Calculate step size in minutes
            step_size = max(1, int(minutes_ahead / (num_points - 1)))

            # Use time range format for Horizons
            epochs = {
                'start': start_time.strftime('%Y-%m-%d %H:%M'),
                'stop': end_time.strftime('%Y-%m-%d %H:%M'),
                'step': f'{step_size}m'
            }

            try:
                location = AppConfig.DEFAULT_LOCATION
                if obs_lon != 0 or obs_lat != 0:
                    location = {
                        'lon': obs_lon,
                        'lat': obs_lat,
                        'elevation': obs_elev
                    }
                    print(
                        f"Using observatory location: Lon={location['lon']:.2f}°, Lat={location['lat']:.2f}°, Elev={location['elevation']}m")

                obj = Horizons(id=obj_name, location=location, epochs=epochs)
                ephemeris = obj.ephemerides()
            except ValueError as e:
                error_msg = str(e).lower()
                if 'ambiguous' in error_msg or 'target' in error_msg:
                    print(f"Name '{obj_name}' is ambiguous in Horizons.")
                    return None
                else:
                    raise

            if ephemeris is None or len(ephemeris) == 0:
                print(f"No Horizons data for {obj_name}")
                return None

            trajectory = []
            for i, row in enumerate(ephemeris):
                try:
                    ra = float(row['RA'])
                    dec = float(row['DEC'])
                    jd = None
                    for col_name in ['jd', 'datetime_jd', 'datetime_str', 'epoch']:
                        if col_name in row.colnames if hasattr(row, 'colnames') else col_name in ephemeris.colnames:
                            try:
                                if col_name == 'datetime_str':
                                    jd = Time(row[col_name], format='iso').jd
                                else:
                                    jd = float(row[col_name])
                                break
                            except (ValueError, TypeError, AttributeError):
                                continue

                    # If still no JD, calculate it from the epoch range
                    if jd is None:
                        epoch_start = Time(epochs['start']).jd
                        step_minutes = int(epochs['step'].replace('m', ''))
                        # Convert minutes to days
                        jd = epoch_start + (i * step_minutes / (24 * 60))

                    # Convert JD to datetime
                    date_obj = Time(jd, format='jd').to_datetime()

                    # Get magnitude if available
                    mag = None
                    if 'mag' in row.colnames if hasattr(row, 'colnames') else 'mag' in ephemeris.colnames:
                        try:
                            mag = float(row['mag'])
                        except (ValueError, TypeError):
                            pass

                    trajectory.append({
                        'date': date_obj,
                        'date_str': date_obj.strftime("%Y-%m-%d %H:%M:%S"),
                        'ra': ra,
                        'dec': dec,
                        'magnitude': mag
                    })
                except Exception as e:
                    print(f"Error processing Horizons row: {e}")
                    continue

            return trajectory if trajectory else None
        except ValueError as e:
            error_msg = str(e).lower()
            if 'ambiguous' in error_msg or 'unknown' in error_msg or 'target' in error_msg:
                print(f"Horizons cannot resolve '{obj_name}'")
                return None
            else:
                print(f"Error querying Horizons: {e}")
                import traceback
                traceback.print_exc()
                return None
        except Exception as e:

            print(f"Error querying Horizons: {e}")

            import traceback
            traceback.print_exc()
            return None

# ============================================================================
# BACKGROUND WORKER THREADS
# ============================================================================

class SearchWorker(QThread):
    """Background thread for searching objects."""
    finished = pyqtSignal(list)
    custom_finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, app):
        super().__init__()
        self.app = app

    def run(self):
        """Run search in background thread."""
        try:
            results = self.app.search_skybot()
            results = self.app.filter_objects_in_image(results)
            self.finished.emit(results)

            if self.app.custom_object_var.strip():
                custom_results = self.app.search_for_custom_object()
                custom_results = self.app.filter_objects_in_image(
                    custom_results)
                if custom_results:
                    res = custom_results[2] if len(
                        custom_results) > 2 else custom_results[-1]
                    res['name'] = self.app.custom_object_var.strip().upper()
                    res['type'] = 'custom'
                    res['separation'] = float(0.0)
                    if 'magnitude' not in res or res['magnitude'] is None:
                        res['magnitude'] = float(0.0)
                    self.custom_finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class TrajectoryWorker(QThread):
    """Background thread for calculating and returning trajectory data."""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, app, obj, date_var, trajectory_period_var):
        super().__init__()
        self.app = app
        self.obj = obj
        self.date_var = date_var
        self.trajectory_period_var = trajectory_period_var
        self.mode = None  # 'display_path' or 'show_trajectory'

    def run(self):
        try:
            trajectory = self.app.skybot.predict_trajectory(
                self.obj['name'],
                self.obj.get('orbital_elements', {}),
                self.date_var,
                num_points=AppConfig.DEFAULT_TRAJECTORY_POINTS,
                minutes_ahead=self.trajectory_period_var,
                obs_lon=self.app.obs_longitude,
                obs_lat=self.app.obs_latitude,
                obs_elev=self.app.obs_elevation
            )
            trajectory = self.app.filter_objects_in_image(trajectory)
            self.finished.emit(trajectory)
        except Exception as e:
            self.error.emit(str(e))

# ============================================================================
# MAIN APPLICATION
# ============================================================================


class AsteroidCometFinder(QMainWindow):
    """Main application class for asteroid/comet detection and visualization.

    Integrates SkyBot and Horizons databases with a PyQt6 GUI to search for
    asteroids and comets in FITS images, display their positions, and show
    predicted trajectories over time.
    """

    def __init__(self, siril_interface):
        super().__init__()
        self.siril = siril_interface
        self.search_results = []
        self.selected_object = None

        # Image information
        self.image_ra = None
        self.image_dec = None
        self.image_width = None
        self.image_height = None

        # SkyBot database
        self.skybot = SkyBotDatabase()

        # UI variables
        self.obs_longitude = 0.0
        self.obs_latitude = 0.0
        self.obs_elevation = 0.0
        self.image_fov = AppConfig.DEFAULT_SEARCH_RADIUS
        self.date_var = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        self.magnitude_var = AppConfig.DEFAULT_MAX_MAGNITUDE
        self.trajectory_period_var = AppConfig.DEFAULT_TRAJECTORY_MINS
        self.custom_object_var = ""

        # Initialize UI first
        self.initialize_ui()
        
        # Connect input fields AFTER UI is initialized
        self.radius_spinbox.valueChanged.connect(self._on_radius_changed)
        self.date_entry.textChanged.connect(self._on_date_changed)
        self.magnitude_spinbox.valueChanged.connect(self._on_magnitude_changed)
        self.trajectory_spinbox.valueChanged.connect(self._on_trajectory_period_changed)
        self.custom_object_entry.textChanged.connect(self._on_custom_object_changed)
        self.lon_spinbox.valueChanged.connect(self._on_longitude_changed)
        self.lat_spinbox.valueChanged.connect(self._on_latitude_changed)
        self.elev_spinbox.valueChanged.connect(self._on_elevation_changed)
        
        # Initialize from image
        try:
            if self.initialize_from_image():
                self.statusBar().showMessage("Ready")
            else:
                self.statusBar().showMessage("Warning: Could not read all image data")
        except Exception as e:
            self.statusBar().showMessage(f"Initialization error: {str(e)}")

    def _on_radius_changed(self, value):
        self.image_fov = value

    def _on_date_changed(self, value):
        self.date_var = value

    def _on_magnitude_changed(self, value):
        self.magnitude_var = value

    def _on_trajectory_period_changed(self, value):
        self.trajectory_period_var = value

    def _on_custom_object_changed(self, value):
        self.custom_object_var = value

    def _on_longitude_changed(self, value):
        self.obs_longitude = value

    def _on_latitude_changed(self, value):
        self.obs_latitude = value

    def _on_elevation_changed(self, value):
        self.obs_elevation = value

    def filter_objects_in_image(self, objects: list) -> list:
        if not self.image_width or not self.image_height:
            return objects
        filtered = []
        for obj in objects:
            try:
                x, y = self.siril.radec2pix(obj['ra'], obj['dec'])
                if 0 <= x < self.image_width and 0 <= y < self.image_height:
                    filtered.append(obj)
            except Exception:
                filtered.append(obj)
        return filtered

    @staticmethod
    def convert_ra_degrees_to_hms(ra_degrees: float) -> str:
        # Convert degrees to hours (360° = 24h)
        ra_hours = ra_degrees / 15.0

        # Extract hours, minutes, seconds
        hours = int(ra_hours)
        remaining = (ra_hours - hours) * 60
        minutes = int(remaining)
        seconds = (remaining - minutes) * 60

        return f"{hours}h{minutes}m{seconds:.2f}s"

    @staticmethod
    def convert_dec_degrees_to_dms(dec_degrees: float) -> str:
        sign = '-' if dec_degrees < 0 else '+'
        dec_degrees = abs(dec_degrees)
        degrees = int(dec_degrees)
        remaining = (dec_degrees - degrees) * 60
        arcminutes = int(remaining)
        arcseconds = (remaining - arcminutes) * 60
        return f"{sign}{degrees}°{arcminutes}'{arcseconds:.2f}''"

    def initialize_from_image(self) -> bool:
        # Get image dimensions
        try:
            channels, height, width = self.siril.get_image_shape()
            self.image_width = width
            self.image_height = height
            print(f"Image size: {width} x {height}")
        except Exception as e:
            print(f"Error getting image size: {e}")
            return False

        # Get image center coordinates
        try:
            ra_center, dec_center = self.siril.pix2radec(width / 2, height / 2)
            self.image_ra = ra_center
            self.image_dec = dec_center
            print(
                f"Image center: RA={self.convert_ra_degrees_to_hms(ra_center)}°, Dec={self.convert_dec_degrees_to_dms(dec_center)}°")
        except Exception as e:
            print(f"Error getting image coordinates: {e}")
            print(
                "LIKELY CAUSE: FITS image lacks proper WCS (World Coordinate System) calibration")
            return False

        # Check if coordinates are valid (not NaN or invalid)
        if not (isinstance(ra_center, (int, float)) and isinstance(dec_center, (int, float))):
            print("Error: Invalid coordinate values returned")
            return False

        if np.isnan(ra_center) or np.isnan(dec_center):
            print("Error: Coordinate values are NaN")
            return False

        # Calculate field of view
        try:
            ra_edge, dec_edge = self.siril.pix2radec(width, height)
            fov_value = SkyCoord(ra=ra_center*u.deg, dec=dec_center*u.deg).separation(
                SkyCoord(ra=ra_edge*u.deg, dec=dec_edge*u.deg)
            ).deg
            print(f"Estimated FOV: {fov_value:.2f}°")
            self.image_fov = round(fov_value, 2)
            self.radius_spinbox.setValue(self.image_fov)
        except Exception as e:
            print(
                f"Warning: Could not calculate FOV, using default value: {e}")
        # Extract date from FITS header
        try:
            kwds = self.siril.get_image_keywords()
            if kwds.date_obs:
                local_tz = tzlocal.get_localzone()
                dt_local = kwds.date_obs.replace(tzinfo=local_tz)
                obs_time = Time(dt_local, scale='utc')
                self.date_var = obs_time.utc.iso
                self.date_entry.setText(self.date_var)
                print(
                    f"Using observation date from FITS header: {self.date_var}")

        # Extract observatory location if available and round to 2 decimal places
            if kwds.sitelong:
                self.obs_longitude = round(kwds.sitelong, 2)
                self.lon_spinbox.setValue(self.obs_longitude)
                print(
                    f"Using observatory Longitude from FITS header: {kwds.sitelong}")
            if kwds.sitelat:
                self.obs_latitude = round(kwds.sitelat, 2)
                self.lat_spinbox.setValue(self.obs_latitude)
                print(
                    f"Using observatory Latitude from FITS header: {kwds.sitelat}")
            if kwds.siteelev:
                self.obs_elevation = round(kwds.siteelev, 2)
                self.elev_spinbox.setValue(self.obs_elevation)
                print(
                    f"Using observatory Elevation from FITS header: {kwds.siteelev}")
        except Exception as e:
            print(f"Could not read FITS header: {e}")

        return True

    def search_for_custom_object(self) -> List[Dict]:
        if not self.image_ra or not self.image_dec:
            return []

        obj = {}
        obj['name'] = self.custom_object_var.strip().upper()
        obj['orbital_elements'] = {}

        try:
            trajectory = self.skybot.predict_trajectory(
                obj['name'],
                obj.get('orbital_elements', {}),
                self.date_var,
                num_points=3,
                minutes_ahead=1,
                obs_lon=self.obs_longitude,
                obs_lat=self.obs_latitude,
                obs_elev=self.obs_elevation
            )
            return trajectory if trajectory else []

        except Exception as e:
            print(f"Error predicting trajectory for custom object: {e}")
            return []

    def search_skybot(self) -> List[Dict]:
        if not self.image_ra or not self.image_dec:
            return []

        try:
            results = self.skybot.search_region(
                self.image_ra,
                self.image_dec,
                self.radius_spinbox.value(),
                self.date_entry.text(),
                max_magnitude=self.magnitude_spinbox.value(),
                obs_id=AppConfig.DEFAULT_LOCATION
            )
            return results
        except Exception as e:
            QMessageBox.critical(self, "Search Error", str(e))
            return []

    def initialize_ui(self):
        self.setWindowTitle(f"Asteroid & Comet Finder v{VERSION}")
        self.setGeometry(100, 100, AppConfig.WINDOW_WIDTH,
                         AppConfig.WINDOW_HEIGHT)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ===== HEADER =====
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #2c3e50;")
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 10, 0, 10)

        title = QLabel("SkyBot Service - Asteroid & Comet Finder")
        title.setFont(QFont("Helvetica", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)

        main_layout.addWidget(header_frame)

        # ===== SEARCH PARAMETERS =====
        search_frame = QFrame()
        search_frame.setStyleSheet(
            "border: 1px solid #ccc; border-radius: 4px; padding: 10px;")
        search_layout = QVBoxLayout(search_frame)

        # Row 1: Search radius, date, magnitude
        row1_layout = QHBoxLayout()

        row1_layout.addWidget(QLabel("Search Radius (°):"))
        self.radius_spinbox = QDoubleSpinBox()
        self.radius_spinbox.setRange(0.1, 10.0)
        self.radius_spinbox.setValue(AppConfig.DEFAULT_SEARCH_RADIUS)
        self.radius_spinbox.setMaximumWidth(80)
        row1_layout.addWidget(self.radius_spinbox)

        row1_layout.addSpacing(20)
        row1_layout.addWidget(QLabel("Datetime UTC:"))
        self.date_entry = QLineEdit()
        self.date_entry.setText(self.date_var)
        self.date_entry.setMaximumWidth(200)
        row1_layout.addWidget(self.date_entry)

        row1_layout.addSpacing(20)
        row1_layout.addWidget(QLabel("Max Magnitude:"))
        self.magnitude_spinbox = QDoubleSpinBox()
        self.magnitude_spinbox.setRange(-100.0, 100.0)
        self.magnitude_spinbox.setValue(AppConfig.DEFAULT_MAX_MAGNITUDE)
        self.magnitude_spinbox.setMaximumWidth(80)
        row1_layout.addWidget(self.magnitude_spinbox)

        row1_layout.addStretch()
        search_layout.addLayout(row1_layout)

        # Row 2: Observatory location
        row2_layout = QHBoxLayout()

        row2_layout.addWidget(QLabel("Lon (°):"))
        self.lon_spinbox = QDoubleSpinBox()
        self.lon_spinbox.setRange(-180, 180)
        self.lon_spinbox.setValue(0.0)
        self.lon_spinbox.setMaximumWidth(100)
        row2_layout.addWidget(self.lon_spinbox)

        row2_layout.addSpacing(10)
        row2_layout.addWidget(QLabel("Lat (°):"))
        self.lat_spinbox = QDoubleSpinBox()
        self.lat_spinbox.setRange(-90, 90)
        self.lat_spinbox.setValue(0.0)
        self.lat_spinbox.setMaximumWidth(100)
        row2_layout.addWidget(self.lat_spinbox)

        row2_layout.addSpacing(10)
        row2_layout.addWidget(QLabel("Elev (m):"))
        self.elev_spinbox = QDoubleSpinBox()
        self.elev_spinbox.setRange(-500, 10000)
        self.elev_spinbox.setValue(0.0)
        self.elev_spinbox.setMaximumWidth(100)
        row2_layout.addWidget(self.elev_spinbox)

        row2_layout.addSpacing(10)
        row2_layout.addWidget(QLabel("Trajectory Period (min):"))
        self.trajectory_spinbox = QSpinBox()
        self.trajectory_spinbox.setRange(1, 1000000)
        self.trajectory_spinbox.setValue(AppConfig.DEFAULT_TRAJECTORY_MINS)
        self.trajectory_spinbox.setMaximumWidth(100)
        row2_layout.addWidget(self.trajectory_spinbox)

        row2_layout.addStretch()
        search_layout.addLayout(row2_layout)

        # Row 3: Custom object search
        row3_layout = QHBoxLayout()
        row3_layout.addWidget(QLabel("Custom Object Name:"))
        self.custom_object_entry = QLineEdit()
        self.custom_object_entry.setMaximumWidth(200)
        row3_layout.addWidget(self.custom_object_entry)

        self.search_btn = QPushButton("Search Objects")
        # light grey button
        self.search_btn.setStyleSheet(
            "background-color: #d3d3d3; color: black; font-weight: bold; padding: 8px 15px;")
        self.search_btn.clicked.connect(self._execute_search)
        row3_layout.addWidget(self.search_btn)

        row3_layout.addStretch()
        search_layout.addLayout(row3_layout)

        main_layout.addWidget(search_frame, stretch=0)

        # ===== RESULTS TABLE =====
        results_frame = QFrame()
        results_frame.setStyleSheet(
            "border: 1px solid #ccc; border-radius: 4px; padding: 10px;")
        results_layout = QVBoxLayout(results_frame)

        results_label = QLabel("Search Results")
        results_label.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
        results_layout.addWidget(results_label)

        self.results_table = QTableWidget()
        self.results_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.results_table.setColumnCount(len(AppConfig.COLUMN_HEADERS))
        self.results_table.setHorizontalHeaderLabels(AppConfig.COLUMN_HEADERS)
        self.results_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(
            QTableWidget.SelectionMode.ExtendedSelection)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.verticalHeader().setVisible(False)
        header = self.results_table.horizontalHeader()
        header.setVisible(True)
        header.setStretchLastSection(False)
        for i in range(6):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        header.setMinimumHeight(50)
        header.sectionClicked.connect(self._on_table_header_clicked)
        self.results_table.itemSelectionChanged.connect(
            self._on_table_object_selected)
        self.results_table.doubleClicked.connect(
            self._on_show_object_requested)

        # Sort tracking
        self.sort_column = None
        self.sort_reverse = False

        results_layout.addWidget(self.results_table)
        main_layout.addWidget(results_frame, stretch=1)

        # ===== OBJECT DETAILS =====
        details_frame = QFrame()
        details_frame.setStyleSheet(
            "border: 1px solid #ccc; border-radius: 4px; padding: 10px;")
        details_layout = QVBoxLayout(details_frame)

        details_label = QLabel("Object Details & Actions")
        details_label.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
        details_layout.addWidget(details_label)

        self.details_text = QLabel("Select an object to view details...")
        self.details_text.setStyleSheet(
            "background-color: #ecf0f1; padding: 10px; border-radius: 4px;")
        self.details_text.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.details_text.setWordWrap(True)
        self.details_text.setMinimumHeight(80)
        details_layout.addWidget(self.details_text)

        button_layout = QHBoxLayout()

        self.show_object_btn = QPushButton("Show Selected Objects")
        self.show_object_btn.setStyleSheet(
            "background-color: #d3d3d3; color: black; font-weight: bold; padding: 8px 15px;")
        self.show_object_btn.clicked.connect(self._on_show_object_requested)
        button_layout.addWidget(self.show_object_btn)

        self.display_path_btn = QPushButton("Display Path on Image")
        self.display_path_btn.setStyleSheet(
            "background-color: #d3d3d3; color: black; font-weight: bold; padding: 8px 15px;")
        self.display_path_btn.clicked.connect(self._on_display_path_requested)
        button_layout.addWidget(self.display_path_btn)

        self.trajectory_btn = QPushButton("Show Trajectory")
        self.trajectory_btn.setStyleSheet(
            "background-color: #d3d3d3; color: black; font-weight: bold; padding: 8px 15px;")
        self.trajectory_btn.clicked.connect(self._on_show_trajectory_requested)
        button_layout.addWidget(self.trajectory_btn)

        button_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.setStyleSheet(
            "background-color: #d3d3d3; color: black; font-weight: bold; padding: 8px 15px;")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)

        details_layout.addLayout(button_layout)
        main_layout.addWidget(details_frame, stretch=0)

        # ===== STATUS BAR =====
        self.statusBar().setStyleSheet(
            "background-color: #34495e; color: white; padding: 5px;")
        self.statusBar().showMessage("Ready. Initializing...")

    def _on_initialization_finished(self):
        """Handle initialization completion."""
        self.statusBar().showMessage("Ready")

    def _on_initialization_error(self, error):
        """Handle initialization error."""
        self.statusBar().showMessage(f"Initialization error: {error}")

    def _execute_search(self):
        """Execute SkyBot and custom object searches in a background thread.

        Validates image calibration, searches the database, and updates the results
        table. Handles both SkyBot searches and custom object Horizons queries.
        """
        if not self.image_ra or not self.image_dec:
            error_msg = (
                "Could not determine image center coordinates.\n\n"
                "POSSIBLE CAUSES:\n"
                "1. FITS image has no WCS (World Coordinate System) calibration\n"
                "2. Image was not properly plate-solved\n"
                "3. Siril interface issue\n\n"
                "SOLUTION:\n"
                "• Use Siril's Astrometry tool to calibrate your image\n"
                "• Or use online plate-solving (e.g., nova.astrometry.net)\n"
                "• Then reload the image and try again"
            )
            QMessageBox.critical(self, "Image Coordinates Error", error_msg)
            return

        self.statusBar().showMessage("Searching objects using SkyBot service...")
        self.search_btn.setEnabled(False)

        self.search_worker = SearchWorker(self)
        self.search_worker.finished.connect(self._on_search_complete)
        self.search_worker.custom_finished.connect(
            self._on_custom_search_complete)
        self.search_worker.error.connect(self._on_search_failed)
        self.search_worker.start()

    def _on_search_complete(self, results):
        """Handle search completion."""
        self.search_results = results
        self.populate_results_table(results, 'skybot')
        self.statusBar().showMessage(
            f"Found {len(results)} objects in search radius")
        self.search_btn.setEnabled(True)

    def _on_custom_search_complete(self, result):
        """Handle custom search completion."""
        self.search_results.insert(0, result)
        self.populate_results_table(result, 'horizon', append=True)

    def _on_search_failed(self, error):
        """Handle search error."""
        self.statusBar().showMessage(f"Search error: {error}")
        QMessageBox.critical(self, "Search Error", str(error))
        self.search_btn.setEnabled(True)

    def populate_results_table(self, search_results, result_type, append=False):
        """Update the results table with search results.

        Args:
            search_results (List[Dict] | Dict): Search results from SkyBot or Horizons.
            result_type (str): 'skybot' for multiple results or 'horizon' for single object.
            append (bool): If True, append results; if False, replace existing. Defaults to False.
        """
        if not append:
            self.results_table.setRowCount(0)

        if result_type == 'skybot':
            for result in search_results:
                row_position = self.results_table.rowCount()
                self.results_table.insertRow(row_position)

                self.results_table.setItem(
                    row_position, 0, QTableWidgetItem(result['name']))
                self.results_table.setItem(
                    row_position, 1, QTableWidgetItem(result['type'].capitalize()))
                self.results_table.setItem(
                    row_position, 2, QTableWidgetItem(f"{result['separation']:.4f}"))
                self.results_table.setItem(row_position, 3, QTableWidgetItem(
                    self.convert_ra_degrees_to_hms(result['ra'])))
                self.results_table.setItem(row_position, 4, QTableWidgetItem(
                    self.convert_dec_degrees_to_dms(result['dec'])))
                self.results_table.setItem(
                    row_position, 5, QTableWidgetItem(f"{result['magnitude']:.2f}"))
        elif result_type == 'horizon':
            row_position = 0
            self.results_table.insertRow(row_position)

            self.results_table.setItem(
                row_position, 0, QTableWidgetItem(search_results['name']))
            self.results_table.setItem(row_position, 1, QTableWidgetItem(
                search_results['type'].capitalize()))
            self.results_table.setItem(row_position, 2, QTableWidgetItem(
                f"{search_results['separation']:.4f}"))
            self.results_table.setItem(row_position, 3, QTableWidgetItem(
                self.convert_ra_degrees_to_hms(search_results['ra'])))
            self.results_table.setItem(row_position, 4, QTableWidgetItem(
                self.convert_dec_degrees_to_dms(search_results['dec'])))
            mag_str = f"{search_results['magnitude']:.2f}" if search_results['magnitude'] is not None else "N/A"
            self.results_table.setItem(
                row_position, 5, QTableWidgetItem(mag_str))

    def _on_table_header_clicked(self, column):
        col_name = AppConfig.COLUMN_HEADERS[column] if column < len(
            AppConfig.COLUMN_HEADERS) else "Name"

        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = False

        try:
            if column == 0:  # Name
                self.search_results.sort(
                    key=lambda x: x['name'], reverse=self.sort_reverse)
            elif column == 1:  # Type
                self.search_results.sort(
                    key=lambda x: x['type'], reverse=self.sort_reverse)
            elif column == 2:  # Separation
                self.search_results.sort(
                    key=lambda x: x['separation'], reverse=self.sort_reverse)
            elif column == 3:  # RA
                self.search_results.sort(
                    key=lambda x: x['ra'], reverse=self.sort_reverse)
            elif column == 4:  # Dec
                self.search_results.sort(
                    key=lambda x: x['dec'], reverse=self.sort_reverse)
            elif column == 5:  # Magnitude
                self.search_results.sort(
                    key=lambda x: x['magnitude'], reverse=self.sort_reverse)
        except Exception as e:
            print(f"Error sorting results: {e}")
            return

        self.populate_results_table(self.search_results, 'skybot')

    def _on_table_object_selected(self):
        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows:
            self.selected_object = None
            return

        self.selected_object = []
        for index in selected_rows:
            row = index.row()
            if row < len(self.search_results):
                self.selected_object.append(self.search_results[row])

        self.update_object_details_panel()

    def update_object_details_panel(self):
        if not self.selected_object:
            return

        obj = self.selected_object[0]
        details = (f"Object: {obj['name']} ({obj['type'].upper()})\n"
                   f"Position: RA = {self.convert_ra_degrees_to_hms(obj['ra'])}, Dec = {self.convert_dec_degrees_to_dms(obj['dec'])}\n"
                   f"Separation from Image Center: {obj['separation']:.4f}°\n"
                   f"Current Magnitude: {obj['magnitude']:.2f}")

        self.details_text.setText(details)

    def _on_display_path_requested(self):
        if not self.selected_object:
            QMessageBox.warning(
                self, "Warning", "Please select an object first")
            return
        obj = self.selected_object[0]
        self.statusBar().showMessage("Calculating trajectory...")
        self.display_path_btn.setEnabled(False)
        self.traj_worker = TrajectoryWorker(
            self, obj, self.date_var, self.trajectory_period_var)
        self.traj_worker.finished.connect(self._on_trajectory_calculated)
        self.traj_worker.error.connect(self._on_trajectory_calculation_failed)
        self.traj_worker.mode = 'display_path'
        self.traj_worker.start()

    def _on_trajectory_calculated(self, trajectory):
        if self.traj_worker.mode == 'display_path':
            if not trajectory:
                QMessageBox.critical(
                    self, "Error", "Could not predict trajectory")
            else:
                self.draw_trajectory_on_image(trajectory)
            self.statusBar().showMessage("Trajectory displayed.")
            self.display_path_btn.setEnabled(True)
        elif self.traj_worker.mode == 'show_trajectory':
            self.display_trajectory_window(self.traj_worker.obj, trajectory)
            self.statusBar().showMessage("Trajectory table displayed.")
            self.trajectory_btn.setEnabled(True)

    def _on_trajectory_calculation_failed(self, error):
        QMessageBox.critical(
            self, "Error", f"Failed to calculate trajectory: {error}")
        self.statusBar().showMessage(f"Trajectory error: {error}")
        self.display_path_btn.setEnabled(True)
        self.trajectory_btn.setEnabled(True)

    def draw_trajectory_on_image(self, trajectory: List[Dict]):
        try:
            for i, point in enumerate(trajectory):
                try:
                    self.siril.cmd(
                        "show", f"{point['ra']}", f"{point['dec']}", f"\"{point['date_str']}\"")
                except Exception as e:
                    print(f"Error drawing marker {i}: {e}")
        except Exception as e:
            raise Exception(f"Error drawing trajectory: {e}")

    def _on_show_trajectory_requested(self):
        """Display a detailed trajectory table in a new window using TrajectoryWorker."""
        if not self.selected_object:
            QMessageBox.warning(
                self, "Warning", "Please select an object first")
            return
        obj = self.selected_object[0]
        self.statusBar().showMessage("Calculating trajectory table...")
        self.trajectory_btn.setEnabled(False)
        self.traj_worker = TrajectoryWorker(
            self, obj, self.date_var, self.trajectory_period_var)
        self.traj_worker.finished.connect(self._on_trajectory_calculated)
        self.traj_worker.error.connect(self._on_trajectory_calculation_failed)
        self.traj_worker.mode = 'show_trajectory'
        self.traj_worker.start()

    def display_trajectory_window(self, obj, trajectory):
        self.traj_win = QMainWindow()
        self.traj_win.setWindowTitle(f"Trajectory: {obj['name']}")
        self.traj_win.resize(700, 600)
        central_widget = QWidget()
        self.traj_win.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        title_label = QLabel(f"Predicted Trajectory for {obj['name']}")
        title_label.setFont(QFont("Helvetica", 12, QFont.Weight.Bold))
        title_label.setStyleSheet(
            "background-color: #2c3e50; color: white; padding: 10px;")
        layout.addWidget(title_label)
        text_widget = QTextEdit()
        text_widget.setReadOnly(True)
        text_widget.setFont(QFont("Menlo", 9))
        header = f"{'Date':<20} {'RA (HMS)':<18} {'Dec (DMS)':<20} {'Mag':<8}\n"
        text_content = header
        text_content += "-" * 70 + "\n"
        for point in trajectory:
            mag = point['magnitude']
            mag_str = f"{mag:<8.2f}" if mag is not None else f"{'—':<8}"
            ra_hms = self.convert_ra_degrees_to_hms(point['ra'])
            dec_dms = self.convert_dec_degrees_to_dms(point['dec'])
            line = f"{point['date_str']:<20} {ra_hms:<18} {dec_dms:<20} {mag_str}\n"
            text_content += line
        text_widget.setText(text_content)
        layout.addWidget(text_widget)
        self.traj_win.show()

    def _on_show_object_requested(self):
        if self.selected_object is None:
            print("No object selected to show.")
            return
        for obj in self.selected_object:
            try:
                # print message round ccordinates to 2 decimal places
                print(
                    f"Showing object: {obj['name']} at RA={self.convert_ra_degrees_to_hms(obj['ra'])}, Dec={self.convert_dec_degrees_to_dms(obj['dec'])}")
                self.siril.cmd(
                    "show", f"{obj['ra']}", f"{obj['dec']}", f"\"{obj['name']}\"")
            except Exception as e:
                print(f"Error drawing marker: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Siril Plugin: Search asteroids and comets using SkyBot service"
    )
    # TODO : add any command-line implementation if needed in future
    args = parser.parse_args()

    print(f"Running Asteroid & Comet Finder v{VERSION}...")

    siril = s.SirilInterface()
    if not siril.connect():
        print("Failed to connect to Siril", file=sys.stderr)
        return 1
    try:
        if not siril.is_image_loaded():
            siril.error_messagebox(
                "No FITS image loaded. Please load an image first.")
            return 1

        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        window = AsteroidCometFinder(siril)
        window.show()
        sys.exit(app.exec())
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        siril.error_messagebox(f"Error: {str(e)}")
        return 1
    finally:
        siril.disconnect()


if __name__ == "__main__":
    sys.exit(main())
