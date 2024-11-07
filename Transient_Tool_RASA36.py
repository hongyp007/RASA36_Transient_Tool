"""
Transient Detection Tool
Version: 1.0.0
Author: YoungPyo Hong
Date: 2024-11-06

A GUI application for viewing and classifying astronomical transient candidates.
This tool provides:
- FITS and PNG image support
- Configurable display settings
- Classification categories
- Image caching and preloading
- Progress tracking
- Keyboard shortcuts
"""
# Standard library imports
import glob
import logging
import os
import re
import threading
import configparser
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from matplotlib import colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from tkinter import (Button, Checkbutton, Entry, Frame, IntVar, Label, 
                    Tk, ttk, Text, messagebox)
from tkinter.ttk import Progressbar, Style
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


def handle_exceptions(func):
    """
    Decorator to handle exceptions in user-facing methods.
    
    Catches any exceptions, logs them, and displays error messages to the user.
    
    Args:
        func: The function to wrap
        
    Returns:
        Wrapped function that handles exceptions gracefully
    """
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logging.exception(f"Error in {func.__name__}: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")
    return wrapper

@dataclass
class Config:
    """
    Configuration settings for the TransientTool application.
    
    This class manages all configuration parameters loaded from config.ini,
    providing type hints and validation for each setting.
    
    Key Features:
    - Type validation for all settings
    - Default values for optional parameters
    - Configuration file loading and parsing
    - Logging setup
    """
    data_directory: str
    file_pattern: str
    output_csv_file: str
    zoom_min: float
    zoom_max: float
    zoom_step: float
    initial_zoom: float
    default_sci_ref_visible: bool
    scale: str
    vmin_subtracted: str
    vmax_subtracted: str
    vmin_science: str
    vmax_science: str
    vmin_reference: str
    vmax_reference: str
    log_file: str
    log_level: str
    shortcuts: Dict[str, str]
    file_type: str
    tile_ids: List[str]
    cache_size: int
    classification_labels: List[str]
    cache_window: int
    preload_batch_size: int
    columns_order: List[str] = field(default_factory=lambda: [
        'file_index', 'tile_id', 'unique_number', 'Memo', 'Scale'
    ])
    
    # Optional parameters (with defaults)
    view_mode: bool = False
    specific_view_mode: Optional[str] = None
    quick_start: bool = False

    @staticmethod
    def load_config(config_path: str = 'config.ini') -> 'Config':
        """
        Load and validate configuration settings from INI file.
        
        Performs the following:
        1. Reads the INI file
        2. Validates required settings
        3. Sets default values for optional settings
        4. Configures logging
        5. Validates data types and value ranges
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Config object with validated settings
            
        Raises:
            ValueError: If required settings are missing or invalid
        """
        try:
            config = configparser.ConfigParser()
            config.read(config_path)

            def get_config_option(section: str, option: str, type_func: Any, default: Any) -> Any:
                """
                Get config option value, ignoring comments after #.
                
                Args:
                    section: Config section name
                    option: Option name within section
                    type_func: Type conversion function
                    default: Default value if option not found
                    
                Returns:
                    Converted option value or default
                """
                try:
                    # Get raw value and strip comments after #
                    value = config.get(section, option)
                    if '#' in value:
                        value = value.split('#')[0].strip()
                        
                    # Handle special case for None values
                    if value.lower() == 'none':
                        return None
                    
                    # Handle boolean values specially
                    if type_func == bool:
                        return value.lower() in ['true', '1', 'yes']
                    
                    # Convert value to specified type
                    return type_func(value)
                    
                except (configparser.NoSectionError, configparser.NoOptionError):
                    return default
                
                except ValueError as e:
                    logging.warning(f"Error parsing {option} from config: {e}")
                    return default

            # Load shortcuts
            shortcuts = {}
            if config.has_section('Shortcuts'):
                for key in config.options('Shortcuts'):
                    shortcuts[key] = config.get('Shortcuts', key).strip()

            # Load mode settings
            view_mode = get_config_option('Mode', 'view_mode', bool, False)
            specific_view_mode = get_config_option('Mode', 'specific_view_mode', str, None)

            # Load tile IDs
            raw_tile_ids = config.get('TileSettings', 'tile_ids', fallback='').split(',')
            tile_ids = []
            if any(tid.strip() for tid in raw_tile_ids):  # If tile_ids is not empty
                for tid in raw_tile_ids:
                    if tid.strip():
                        tile_id = DataManager.get_tile_id(tid.strip())
                        if tile_id:
                            tile_ids.append(tile_id)
                if not tile_ids:
                    logging.warning("No valid tile IDs found in config, will auto-detect")
            else:
                logging.info("No tile IDs specified in config, will auto-detect")

            # Load classification labels
            classification_labels = [label.strip() for label in 
                                  config.get('Settings', 'classification_labels', fallback='').split(',')]

            # Set up logging configuration
            log_file = get_config_option('Logging', 'log_file', str, 'transient_tool.log')
            log_level = get_config_option('Logging', 'log_level', str, 'INFO').upper()
            
            # Configure logging
            logging.basicConfig(
                filename=log_file,
                level=getattr(logging, log_level),
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            logging.info("Configuration loaded successfully")
            
            # Create config instance with all parameters
            config_obj = Config(
                data_directory=get_config_option('Paths', 'data_directory', str, ''),
                file_pattern=get_config_option('Paths', 'file_pattern', str, ''),
                output_csv_file=get_config_option('Paths', 'output_csv_file', str, ''),
                zoom_min=get_config_option('Settings', 'zoom_min', float, 1.0),
                zoom_max=get_config_option('Settings', 'zoom_max', float, 10.0),
                zoom_step=get_config_option('Settings', 'zoom_step', float, 0.1),
                initial_zoom=get_config_option('Settings', 'initial_zoom', float, 1.0),
                default_sci_ref_visible=get_config_option('Settings', 'default_sci_ref_visible',bool, True),
                scale=get_config_option('Settings', 'scale', str, 'zscale').lower(),
                vmin_subtracted=get_config_option('Settings', 'vmin_subtracted', str, 'median').lower(),
                vmax_subtracted=get_config_option('Settings', 'vmax_subtracted', str, 'max').lower(),
                vmin_science=get_config_option('Settings', 'vmin_science', str, 'median').lower(),
                vmax_science=get_config_option('Settings', 'vmax_science', str, 'max').lower(),
                vmin_reference=get_config_option('Settings', 'vmin_reference', str, 'median').lower(),
                vmax_reference=get_config_option('Settings', 'vmax_reference', str, 'max').lower(),
                log_file=log_file,
                log_level=log_level,
                shortcuts=shortcuts,
                file_type=get_config_option('Settings', 'file_type', str, 'fits').lower(),
                tile_ids=tile_ids,  # 빈 리스트여도 괜찮음
                cache_size=get_config_option('TileSettings', 'cache_size', int, 100),
                classification_labels=classification_labels,
                cache_window=get_config_option('TileSettings', 'cache_window', int, 10),
                preload_batch_size=get_config_option('TileSettings', 'preload_batch_size', int, 5),
                view_mode=view_mode,
                specific_view_mode=specific_view_mode,
                quick_start=get_config_option('Mode', 'quick_start', bool, False)
            )

            # Validation of required fields
            required_options = ['data_directory', 'file_pattern', 'output_csv_file']
            for option in required_options:
                if not getattr(config_obj, option):
                    raise ValueError(f"Missing required configuration option: {option} in section 'Paths'.")

            # Validate scale option
            if config_obj.scale not in ['zscale', 'linear', 'log']:
                logging.warning(f"Invalid scale '{config_obj.scale}' in configuration. Using 'linear' as default.")
                config_obj.scale = 'linear'

            # Validate file_type option
            if config_obj.file_type not in ['fits', 'png']:
                raise ValueError("Invalid file_type option in configuration. Choose 'fits' or 'png'.")

            return config_obj

        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")

class DataManager:
    """
    Handles data loading, processing and persistence for astronomical images.
    
    Key responsibilities:
    1. Image data loading and caching
    2. Classification data management
    3. Progress tracking
    4. File operations (CSV read/write)
    5. Cache management
    
    The DataManager maintains a thread-safe environment for concurrent operations
    and implements efficient caching strategies for optimal performance.
    """

    def __init__(self, config: Config):
        """
        Initialize DataManager with configuration.
        
        Sets up:
        - Data structures for image and metadata management
        - Thread locks for concurrent operations
        - Cache initialization
        - Helper components (ImageProcessor, DataValidator)
        
        Args:
            config: Configuration settings
        """
        # Configuration and basic attributes
        self.config = config
        self.region_df = None
        self.index = 0
        self.total_images = 0
        
        # Cache related attributes
        self.image_cache = {}
        self.cache_size = config.cache_size
        self.cache_window = config.cache_window
        self.preload_batch_size = config.preload_batch_size
        
        # Thread locks
        self.cache_lock = threading.Lock()
        self.preload_lock = threading.Lock()
        self.preload_thread = threading.Thread()
        self.file_lock = threading.Lock()
        
        # Helper components
        self.image_processor = ImageProcessor(config)
        self.data_validator = DataValidator(config)

        # Load data based on quick_start mode
        if hasattr(self.config, 'quick_start') and self.config.quick_start:
            self._quick_start_load()
        else:
            self._full_load()

        if hasattr(self.config, 'specific_view_mode') and self.config.specific_view_mode:
            self.valid_indices = self.region_df[
                self.region_df[self.config.specific_view_mode] == 1
            ].index.tolist()
            
        logging.info("DataManager initialized.")

    def _quick_start_load(self):
        """Load data directly from CSV without scanning directories."""
        try:
            if os.path.exists(self.config.output_csv_file):
                self.region_df = pd.read_csv(self.config.output_csv_file)
                logging.info(f"Found existing CSV with {len(self.region_df)} entries")
                # Initialize DataFrame structure
                self.init_dataframe()
            else:
                raise FileNotFoundError("Required CSV file not found for quick start mode")
        except Exception as e:
            logging.error(f"Error in quick start load: {e}")
            raise

    def _full_load(self):
        """Perform full load with directory scanning."""
        try:
            self.load_files()  # This includes directory scanning
        except Exception as e:
            logging.error(f"Error in full load: {e}")
            raise

    def load_files(self):
        """Load files and initialize DataFrame."""
        try:
            # Load existing data if available
            existing_data = None
            if os.path.exists(self.config.output_csv_file):
                existing_data = pd.read_csv(self.config.output_csv_file)
                logging.info(f"Found existing CSV with {len(existing_data)} entries")

            # Get current files, using existing data to avoid reprocessing
            if existing_data is not None:
                self.region_df = existing_data.copy()
                logging.info("Using existing data without rescanning")
            else:
                # No existing data, scan all files
                self.region_df = self.scan_directory_for_files().copy()
                logging.info(f"Created new index with {len(self.region_df)} entries")

            # Initialize DataFrame with proper structure
            self.init_dataframe()

        except Exception as e:
            logging.exception(f"Error loading files: {e}")
            raise

    def scan_directory_for_files(self, existing_keys=None):
        """Scan directory and create new index for files."""
        try:
            file_data = []
            base_dir = self.config.data_directory
            
            tile_ids = self.config.tile_ids if self.config.tile_ids else self.get_all_tile_ids()
            
            for tile_id in tile_ids:
                pattern = f"**/*{tile_id}*.com.*.sub.{self.config.file_type}"
                full_pattern = os.path.join(base_dir, pattern)
                files = glob.glob(full_pattern, recursive=True)
                
                if files:
                    logging.info(f"Found {len(files)} files for tile {tile_id}")
                    
                    # Create temporary list for this tile's files
                    tile_data = []
                    for filename in files:
                        unique_number = self.get_unique_number(filename)
                        if unique_number is not None:
                            # Skip if file already exists in CSV
                            if existing_keys and (tile_id, unique_number) in existing_keys:
                                continue
                                
                            file_data_dict = {
                                'tile_id': tile_id,
                                'unique_number': unique_number,
                                'Memo': '',
                                'Scale': ''
                            }
                            for label in self.config.classification_labels:
                                file_data_dict[label] = 0
                            tile_data.append(file_data_dict)
                    
                    # Sort tile_data by unique_number before adding to main list
                    tile_data.sort(key=lambda x: x['unique_number'])
                    file_data.extend(tile_data)
                else:
                    logging.info(f"No files found for tile {tile_id}")

            # Add file_index after sorting
            for i, data in enumerate(file_data):
                data['file_index'] = i

            df = pd.DataFrame(file_data)
            if len(file_data) > 0:
                logging.info(f"Found {len(df)} new files across all tiles")
            return df

        except Exception as e:
            logging.exception(f"Error scanning directory: {e}")
            raise

    @staticmethod
    def get_unique_number(filename: str) -> Optional[int]:
        """
        Extract unique identifier from filename.
        
        Args:
            filename: Full path to image file
            
        Returns:
            Unique number from filename or None if not found
            
        Example:
            >>> get_unique_number("path/to/com.123.sub.fits")
            123
        """
        basename = os.path.basename(filename)
        match = re.search(r'com\.(\d+)\.', basename)
        if match:
            return int(match.group(1))
        return None

    def get_tile_id(filename: str) -> Optional[str]:
        """
        Extract tile ID from filename.
        Expected format: 'T<number>' in the filename
        Returns the full ID including 'T' prefix
        """
        match = re.search(r'(T\d+)', filename)
        if match:
            return match.group(1)
        return None

    def init_dataframe(self):
        """Initialize DataFrame with proper structure and defaults."""
        try:
            # Create a copy to avoid chained assignment
            df = self.region_df.copy()
            
            # Remove total row (file_index == -1) if exists
            df = df[df['file_index'] != -1]
            

            # Validate DataFrame structure
            is_valid, errors = self.data_validator.validate_dataframe(df)
            if not is_valid:
                logging.warning(f"DataFrame validation failed: {errors}")
                
                # Add missing columns if needed
                for col in self.config.classification_labels:
                    if col not in df.columns:
                        df.loc[:, col] = 0
                    df.loc[:, col] = df[col].fillna(0).astype(int)

                if 'Memo' not in df.columns:
                    df.loc[:, 'Memo'] = ''
                df['Memo'] = df['Memo'].fillna('').astype(str)

                if 'Scale' not in df.columns:
                    df.loc[:, 'Scale'] = self.config.scale
                df['Scale'] = df['Scale'].fillna(self.config.scale).astype(str)

            # Ensure proper column order including classification labels
            all_columns = [*self.config.columns_order, *self.config.classification_labels]
            existing_columns = [col for col in all_columns if col in df.columns]
            
            # Verify all required columns exist
            missing_columns = set(self.config.columns_order) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            df = df[existing_columns]

            # Sort by file_index and reset index to ensure alignment
            df = df.sort_values('file_index').reset_index(drop=True)

            # Fix file_index alignment
            if not df['file_index'].equals(pd.Series(range(len(df)))):
                logging.warning("Fixing non-sequential file_index values")
                df['file_index'] = range(len(df))
            
            # Assign back to self.region_df
            self.region_df = df
            
            # Save DataFrame to CSV
            self.save_dataframe()
            
            logging.info(f"DataFrame initialized with {len(self.region_df)} rows")

        except Exception as e:
            logging.error(f"Error initializing DataFrame: {e}")
            raise

    def get_starting_index(self) -> int:
        """
        Determine starting index based on first unprocessed image.
        Returns index of first image where all classification values are 0.
        """
        try:
            if self.region_df.empty:
                return 0
                
            # Check if any classification label is 1 for each row
            classified = self.region_df[self.config.classification_labels].any(axis=1)
            
            # Find first unclassified image (where all labels are 0)
            unclassified = self.region_df[~classified]
            
            if not unclassified.empty:
                # Get the file_index of first unclassified image
                first_unclassified_index = unclassified['file_index'].iloc[0]
                logging.info(f"Starting from first unclassified image at index {first_unclassified_index}")
                return first_unclassified_index
            
            # If all images are classified, start from beginning
            logging.info("All images are classified, starting from index 0")
            return 0
            
        except Exception as e:
            logging.error(f"Critical error in get_starting_index: {e}")
            raise

    def save_dataframe(self, mode='w', callback=None):
        """Save the DataFrame to CSV file."""
        try:
            # Validate DataFrame before saving
            is_valid, errors = self.data_validator.validate_dataframe(self.region_df)
            if not is_valid:
                error_msg = "\n".join(errors)
                logging.error(f"DataFrame validation failed:\n{error_msg}")
                raise ValueError(f"DataFrame validation failed:\n{error_msg}")

            # Create a copy to avoid modifying the original
            df_to_save = self.region_df.copy()
            
            # Ensure Memo column is preserved as string
            df_to_save['Memo'] = df_to_save['Memo'].astype(str)
            
            # Calculate totals for classifications
            totals = {}
            for col in self.config.classification_labels:
                totals[col] = int(df_to_save[col].sum())
            
            # Count total images and processed images
            total_images = len(df_to_save)
            total_processed = len(df_to_save[df_to_save[self.config.classification_labels].any(axis=1)])
            percent_processed = (total_processed/total_images*100) if total_images > 0 else 0
            
            # Create total row
            total_dict = {
                'file_index': -1,
                'tile_id': 'Total',
                'unique_number': len(df_to_save['tile_id'].unique()),
                'Memo': f"{total_processed}/{total_images}",
                'Scale': f"{percent_processed:.2f}%"
            }
            total_dict.update(totals)

            # Remove any existing total rows and duplicates
            df_to_save = df_to_save[df_to_save['file_index'] != -1].drop_duplicates(subset=['tile_id', 'unique_number'])
            
            # Append total row
            df_to_save = pd.concat([df_to_save, pd.DataFrame([total_dict])], ignore_index=True)

            # Save with file lock
            with self.file_lock:
                df_to_save.to_csv(self.config.output_csv_file, index=False, mode='w', na_rep='')
                logging.info(f"DataFrame saved successfully to {self.config.output_csv_file}")

            if callback:
                callback()

        except Exception as e:
            logging.error(f"Error saving DataFrame: {e}")
            raise

    def load_image_data(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load image data with improved caching.
        """
        try:
            index = int(index)
            
            if index < 0 or index >= len(self.region_df):
                raise ValueError(f"Index {index} out of bounds")
                
            current_row = self.region_df.iloc[index]
            tile_id = current_row['tile_id']
            unique_number = current_row['unique_number']
            cache_key = f"{tile_id}_{unique_number}"
            
            # Check cache with minimal locking
            with self.cache_lock:
                if cache_key in self.image_cache:
                    return self.image_cache[cache_key]
            
            # Load images using ImageProcessor
            images = self.image_processor.load_and_process_images(tile_id, unique_number)
            
            # Update cache
            with self.cache_lock:
                self.image_cache[cache_key] = images
                
                # Cleanup cache if needed
                if len(self.image_cache) > self.cache_size:
                    self.cleanup_cache(index)
            
            return images
                
        except Exception as e:
            logging.error(f"Error in load_image_data: {e}")
            raise

    def start_preloading(self, current_index: int):
        """
        Start preloading images in a background thread, prioritizing current tile ID.
        """
        try:
            if self.preload_thread.is_alive():
                return  # Skip if previous preload is still running
                
            current_row = self.region_df.iloc[current_index]
            current_tile = current_row['tile_id']
            
            # Get indices to preload (prioritize same tile, then sequential)
            next_indices = self._get_preload_indices(current_index, current_tile)
            
            if next_indices:
                self.preload_thread = threading.Thread(
                    target=self.preload_images,
                    args=(next_indices,),
                    daemon=True
                )
                self.preload_thread.start()
                logging.debug(f"Started preloading {len(next_indices)} images")
                
        except Exception as e:
            logging.error(f"Error starting preload thread: {e}")

    def _get_preload_indices(self, current_index: int, current_tile: str) -> List[int]:
        """
        Get optimized list of indices to preload, prioritizing same tile ID.
        """
        try:
            indices_to_preload = []
            
            # First, get next few images from same tile
            same_tile_indices = self.region_df[
                (self.region_df['tile_id'] == current_tile) & 
                (self.region_df.index > current_index)
            ].index[:self.preload_batch_size].tolist()
            
            # Then, get next sequential images
            sequential_indices = range(
                current_index + 1,
                min(current_index + self.preload_batch_size, len(self.region_df))
            )
            
            # Combine and remove duplicates while maintaining order
            all_indices = []
            for idx in same_tile_indices + list(sequential_indices):
                if idx not in all_indices:
                    all_indices.append(idx)
            
            # Filter out already cached images
            for idx in all_indices:
                row = self.region_df.iloc[idx]
                cache_key = f"{row['tile_id']}_{row['unique_number']}"
                if cache_key not in self.image_processor.image_cache:
                    indices_to_preload.append(idx)
                    
                if len(indices_to_preload) >= self.preload_batch_size:
                    break
                    
            return indices_to_preload
            
        except Exception as e:
            logging.error(f"Error getting preload indices: {e}")
            return []

    def preload_images(self, indices: Union[int, List[int]]):
        """
        Preload images with improved performance.
        """
        try:
            if isinstance(indices, int):
                indices = [indices]
                
            # Group indices by tile_id for efficient loading
            tile_groups = {}
            for idx in indices:
                row = self.region_df.iloc[idx]
                tile_id = row['tile_id']
                if tile_id not in tile_groups:
                    tile_groups[tile_id] = []
                tile_groups[tile_id].append((idx, row['unique_number']))
            
            # Process each tile group
            for tile_id, index_pairs in tile_groups.items():
                try:
                    # Sort by unique_number for sequential file access
                    index_pairs.sort(key=lambda x: x[1])
                    
                    for idx, unique_number in index_pairs:
                        cache_key = f"{tile_id}_{unique_number}"
                        if cache_key not in self.image_processor.image_cache:
                            self.load_image_data(idx)
                            
                except Exception as e:
                    logging.error(f"Error preloading tile {tile_id}: {e}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error in preload_images: {e}")

    def calculate_progress(self) -> dict:
        """Calculate progress statistics for total and per-tile."""
        try:
            progress_stats = {}
            
            # Calculate total progress
            total_images = len(self.region_df)
            total_classified = len(self.region_df[
                self.region_df[self.config.classification_labels].any(axis=1)
            ])
            total_percent = (total_classified / total_images * 100) if total_images > 0 else 0
            
            progress_stats['total'] = {
                'classified': total_classified,
                'total': total_images,
                'percent': total_percent
            }
            
            # Calculate progress by tile
            progress_stats['tiles'] = {}
            for tile_id in sorted(self.region_df['tile_id'].unique()):
                tile_df = self.region_df[self.region_df['tile_id'] == tile_id]
                tile_total = len(tile_df)
                tile_classified = len(tile_df[tile_df[self.config.classification_labels].any(axis=1)])
                tile_percent = (tile_classified / tile_total * 100) if tile_total > 0 else 0
                
                progress_stats['tiles'][tile_id] = {
                    'classified': tile_classified,
                    'total': tile_total,
                    'percent': tile_percent
                }
            
            return progress_stats
            
        except Exception as e:
            logging.error(f"Error calculating progress: {e}")

    def cleanup_cache(self, current_index: int):
        """Remove images outside cache window with tile ID priority."""
        try:
            with self.cache_lock:
                current_tile = self.region_df.iloc[current_index]['tile_id']
                
                # Keep images from current tile and within window
                window_start = max(0, current_index - self.cache_window)
                window_end = min(len(self.region_df), current_index + self.cache_window)
                
                valid_indices = set(range(window_start, window_end + 1))
                valid_keys = set()
                
                # Add keys for current tile
                for idx in self.region_df[self.region_df['tile_id'] == current_tile].index:
                    row = self.region_df.iloc[idx]
                    valid_keys.add(f"{row['tile_id']}_{row['unique_number']}")
                
                # Add keys for window
                for idx in valid_indices:
                    row = self.region_df.iloc[idx]
                    valid_keys.add(f"{row['tile_id']}_{row['unique_number']}")
                
                # Remove invalid keys
                for key in list(self.image_cache.keys()):
                    if key not in valid_keys:
                        del self.image_cache[key]
                        
        except Exception as e:
            logging.error(f"Error cleaning cache: {e}")

    def get_all_tile_ids(self):
        """Scan directory to find all available tile IDs."""
        try:
            base_dir = self.config.data_directory
            pattern = "**/*RASA36-T*-*.com.*.sub." + self.config.file_type
            all_files = glob.glob(os.path.join(base_dir, pattern), recursive=True)
            
            # Extract tile IDs using regex
            tile_ids = set()
            for filepath in all_files:
                match = re.search(r'RASA36-(T\d{5})-', filepath)
                if match:
                    tile_ids.add(match.group(1))
            
            if not tile_ids:
                logging.warning("No tile IDs found in directory")
                return []
                
            sorted_tile_ids = sorted(list(tile_ids))
            logging.info(f"Found {len(sorted_tile_ids)} tile IDs: {', '.join(sorted_tile_ids)}")
            return sorted_tile_ids
            
        except Exception as e:
            logging.error(f"Error finding tile IDs: {e}")
            return []

class DataValidator:
    """Class to handle data validation."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame structure and content.
        Single source of truth for DataFrame validation.
        """
        errors = []
        
        try:
            # Validate required columns
            errors.extend(self._validate_columns(df))
            
            # Validate data types
            errors.extend(self._validate_data_types(df))
            
            # Validate classification values
            errors.extend(self._validate_classifications(df))
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, errors
            
    def _validate_columns(self, df: pd.DataFrame) -> List[str]:
        """Validate required columns exist."""
        errors = []
        required_cols = self.config.columns_order + self.config.classification_labels
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        return errors
        
    def _validate_data_types(self, df: pd.DataFrame) -> List[str]:
        """Validate data types of key columns."""
        errors = []
        if 'unique_number' in df.columns:
            if not pd.to_numeric(df['unique_number'], errors='coerce').notna().all():
                errors.append("Invalid unique_number values found")
        return errors
        
    def _validate_classifications(self, df: pd.DataFrame) -> List[str]:
        """Validate classification column values."""
        errors = []
        for col in self.config.classification_labels:
            if col in df.columns:
                invalid = ~df[col].isin([0, 1, np.nan])
                if invalid.any():
                    invalid_rows = df.loc[invalid, 'unique_number'].tolist()
                    errors.append(f"Invalid {col} values in rows: {invalid_rows}")
        return errors

class ImageProcessor:
    """
    Handles image processing operations and caching.
    
    Key features:
    1. FITS and PNG image support
    2. Image scaling (zscale, linear, log)
    3. Normalization and value range management
    4. Thread-safe caching
    5. Memory optimization
    
    The ImageProcessor ensures efficient image loading and processing
    while maintaining memory usage within configured limits.
    """
    def __init__(self, config: Config):
        self.config = config
        self.image_cache = {}
        self.cache_size = config.cache_size
        self.cache_lock = threading.Lock()
        self.zscale = ZScaleInterval()
        # Add image type flag
        self.is_fits = config.file_type.lower() == 'fits'

    def _update_cache(self, key: str, value: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Thread-safe cache update."""
        try:
            with self.cache_lock:
                if len(self.image_cache) >= self.cache_size:
                    oldest_key = next(iter(self.image_cache))
                    del self.image_cache[oldest_key]
                self.image_cache[key] = value
        except Exception as e:
            logging.error(f"Error updating cache: {e}")

    def load_and_process_images(self, tile_id: str, unique_number: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and process images with parallel loading for better performance.
        """
        try:
            # Get paths for all image types
            paths = self._get_image_paths(tile_id, unique_number)
            
            # Parallel loading using ThreadPoolExecutor
            images = {'sub': None, 'new': None, 'ref': None}
            
            # Create a thread pool with 3 workers (one for each image type)
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    'sub': executor.submit(self._load_single_image, paths['sub']),
                    'new': executor.submit(self._load_single_image, paths['new']),
                    'ref': executor.submit(self._load_single_image, paths['ref'])
                }
                
                # Wait for all tasks to complete and get results
                for img_type, future in futures.items():
                    try:
                        images[img_type] = future.result(timeout=30)
                    except Exception as e:
                        logging.error(f"Error loading {img_type} image: {e}")

            # Validate required images
            if images['sub'] is None:
                raise FileNotFoundError(f"Failed to load subtracted image for {tile_id}-{unique_number}")
            
            # Process images based on configuration - now using is_fits flag
            processed_images = {
                img_type: img_data 
                for img_type, img_data in images.items() 
                if img_data is not None
            }
            
            return (processed_images.get('sub'), 
                   processed_images.get('new'), 
                   processed_images.get('ref'))
            
        except Exception as e:
            logging.error(f"Error in load_and_process_images for {tile_id}-{unique_number}: {e}")
            raise

    def _load_single_image(self, filepath: str) -> Optional[np.ndarray]:
        """Load a single image file with proper error handling."""
        try:
            if not os.path.exists(filepath):
                logging.warning(f"File not found: {filepath}")
                return None

            if self.is_fits:
                with fits.open(filepath, memmap=True) as hdul:
                    data = hdul[0].data
                    if data is None:
                        logging.error(f"No data in FITS file: {filepath}")
                        return None
                    return data.astype(np.float32)
            else:  # PNG case
                return plt.imread(filepath)

        except Exception as e:
            logging.error(f"Error loading image {filepath}: {e}")
            return None

    def _get_image_paths(self, tile_id: str, unique_number: int) -> dict:
        """Get paths for all image types."""
        try:
            # First find the .sub file
            base_pattern = f"**/*{tile_id}*.com.{unique_number}.sub.{self.config.file_type}"
            sub_files = glob.glob(os.path.join(self.config.data_directory, base_pattern), recursive=True)
            
            if not sub_files:
                raise FileNotFoundError(f"No subtracted image found for {tile_id} number {unique_number}")
                
            # Use the found sub file to construct paths for new and ref
            sub_path = sub_files[0]
            base_path = sub_path.replace(f".sub.{self.config.file_type}", "")
            
            return {
                'sub': sub_path,
                'new': f"{base_path}.new.{self.config.file_type}",
                'ref': f"{base_path}.ref.{self.config.file_type}"
            }
            
        except Exception as e:
            logging.error(f"Error getting image paths: {e}")
            raise
            
    def prepare_normalization(self, image: np.ndarray, vmin: Optional[float], vmax: Optional[float]) -> Any:
        """Create image normalization based on configured scale type."""
        try:
            # Only process normalization for FITS files
            if not self.is_fits:
                return None

            # Only convert to float if needed
            if image.dtype != np.float32 and image.dtype != np.float64:
                image = image.astype(np.float32)
            
            # Get min/max values using validate_value
            v_min = self.validate_value(vmin, image) if isinstance(vmin, str) else vmin
            v_max = self.validate_value(vmax, image) if isinstance(vmax, str) else vmax
            
            if self.config.scale == 'zscale':
                interval = self.zscale
                v_min, v_max = interval.get_limits(image)
                return colors.Normalize(vmin=v_min, vmax=v_max)
                
            elif self.config.scale == 'log':
                return colors.LogNorm(vmin=v_min, vmax=v_max)
                
            else:  # 'linear'
                return colors.Normalize(vmin=v_min, vmax=v_max)
                
        except Exception as e:
            logging.error(f"Error preparing normalization: {e}")
            return colors.Normalize()
            
    def validate_value(self, value: str, image: np.ndarray) -> float:
        """
        Validate and calculate value based on string descriptor or numeric value.
        
        Args:
            value: String descriptor ('max', 'min', etc.) or numeric value
            image: Image data to calculate statistics from
            
        Returns:
            float: Calculated value
        """
        try:
            if isinstance(value, (int, float)):
                return float(value)
                
            if value == 'max':
                return float(np.max(image))
            elif value == 'min':
                return float(np.min(image))
            elif value == 'median':
                return float(np.median(image))
            elif value == 'mean':
                return float(np.mean(image))
            elif value == 'std':
                return float(np.std(image))
            else:
                raise ValueError(f"Invalid value descriptor: {value}")
                
        except Exception as e:
            logging.error(f"Error validating value: {e}")
            raise

    def load_image(self, tile_id: str, unique_number: int, image_type: str) -> np.ndarray:
        """
        Load and process a single image with caching.
        
        Args:
            tile_id: Identifier for the image tile
            unique_number: Unique number within the tile
            image_type: Type of image ('sub', 'new', or 'ref')
            
        Returns:
            np.ndarray: Processed image data
        """
        cache_key = f"{tile_id}_{unique_number}_{image_type}"
        
        # Check cache first
        if cache_key in self.image_cache:
            return self.image_cache[cache_key]
        
        # Load and process image
        filepath = self._get_image_path(tile_id, unique_number, image_type)
        image_data = self._load_single_image(filepath)
        
        # Cache result
        if image_data is not None:
            self.image_cache[cache_key] = image_data
            
        return image_data

class TransientTool:
    """
    GUI application for classifying transient astronomical objects.
    
    Key features:
    1. Interactive image display with zoom
    2. Classification management
    3. Progress tracking
    4. Keyboard shortcuts
    5. Memo management
    
    The TransientTool provides a user-friendly interface for viewing
    and classifying astronomical transient candidates with support
    for various image types and display configurations.
    """
    def __init__(self, master: Tk, config: Config):
        """Initialize the TransientTool application."""
        self.master = master
        self.config = config
        
        # Initialize image processor
        self.image_processor = ImageProcessor(config)
        
        # Initialize zoom and display attributes
        self.zoom_level = self.config.initial_zoom
        self.original_size = [1, 1]  # Will be updated when first image is loaded
        self.zoom_center = [0.5, 0.5]  # Center point for zoom
        self.view_size = [1/self.zoom_level, 1/self.zoom_level]
        
        # Initialize classification buttons dictionary
        self.classification_buttons = {}
        
        # Initialize data attributes
        self.data_manager = DataManager(config)
        self.index = self.data_manager.index
        self.region_df = self.data_manager.region_df
        self.num_images = len(self.region_df)
        self.science_data = None
        self.reference_data = None
        self.sci_ref_visible = self.config.default_sci_ref_visible
        
        # Preload initial batch of images before UI setup
        initial_index = self.data_manager.get_starting_index()
        self.data_manager.start_preloading(initial_index)
        
        # Set window title based on mode
        if self.config.specific_view_mode:
            self.master.title(f"Transient Tool - {self.config.specific_view_mode} View Mode")
        elif self.config.view_mode:
            self.master.title("Transient Tool - View Mode")
        else:
            self.master.title("Transient Detection Tool")
        
        # Set up the main window
        self.setup_ui()

        # Bind keyboard shortcuts after UI setup
        self.bind_shortcuts()
        
        # Start from first unclassified image 
        self.goto_unclassified()
            
        logging.info(f"Initializing in {'view' if self.config.view_mode else 'normal'} mode")

    def setup_logging(self):
        """
        Set up logging configuration.
        """
        logging.basicConfig(
            filename=self.config.log_file,
            level=getattr(logging, self.config.log_level, logging.INFO),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("Application started.")

    def setup_ui(self):
        """Initialize and layout all UI components."""
        # Configure main window grid
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        # Create main frame with grid configuration
        frame = Frame(self.master)
        frame.grid(row=0, column=0, sticky='nsew')
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        # Initialize matplotlib figure and canvas for image display
        self.fig = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().configure(takefocus=1)  # Make canvas focusable
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew')

        # Create subplots with adjusted spacing
        self.axes = [self.fig.add_subplot(1, 3, i + 1) for i in range(3)]
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.85, wspace=0.05)

        # Zoom Slider
        self.create_zoom_slider()

        # Memo Section
        self.create_memo_section(frame)

        # Control Buttons
        self.create_control_buttons(frame)

        # Progress Bar
        self.create_progress_bar(frame)

        # Scale, Vmin, Vmax Labels
        self.create_scale_labels(frame)

        # Directories Text
        self.create_directories_text(frame)

    def create_zoom_slider(self):
        """Create a zoom slider below the images."""
        self.zoom_ax = self.fig.add_axes([0.25, 0.1, 0.5, 0.03])
        self.zoom_slider = Slider(
            self.zoom_ax, 'Zoom', 
            self.config.zoom_min, 
            self.config.zoom_max,
            valinit=self.config.initial_zoom, 
            valstep=self.config.zoom_step
        )
        self.zoom_slider.on_changed(self.on_zoom_change)

    def on_zoom_change(self, val):
        """Handle zoom slider value changes."""
        try:
            if val != self.zoom_level:
                self.zoom_level = val
                self.view_size = [size/self.zoom_level for size in self.original_size]
                self.update_zoom()
                logging.debug(f"Zoom changed to {val}x via slider")
        except Exception as e:
            logging.error(f"Error in zoom slider change: {e}")

    def create_memo_section(self, parent_frame: Frame):
        """
        Create the memo text box that activates on click and saves on any outside click.
        """
        memo_frame = Frame(parent_frame)
        memo_frame.grid(row=2, column=0, sticky='ew', padx=5, pady=5)

        # Memo Text Box
        Label(memo_frame, text="Memo:").pack(anchor='w')
        self.memo_text = Text(
            memo_frame,
            height=4,
            wrap='word',
            state='disabled',  # Start disabled
            bd=1,             # Add a border
            bg='#F0F0F0',     # Light gray background when disabled
            fg='black'        # Text color
        )
        self.memo_text.pack(fill='x', expand=True)

        # Bind click events
        self.memo_text.bind('<Button-1>', self.activate_memo)
        
        # Bind click event to the main window for saving memo
        self.master.bind('<Button-1>', self.check_memo_click, add='+')

    def activate_memo(self, event=None):
        """Activate memo text box for editing."""
        if self.memo_text['state'] == 'disabled':
            self.memo_text.config(state='normal', bg='white')  # Enable and change background to white
            self.memo_text.focus_set()  # Set focus to the text box
            return 'break'  # Prevent event propagation

    def check_memo_click(self, event):
        """
        Check if click was outside memo box and save if necessary.
        """
        if not hasattr(self, 'memo_text'):
            return
        
        # Get memo widget coordinates
        memo_x = self.memo_text.winfo_rootx()
        memo_y = self.memo_text.winfo_rooty()
        memo_width = self.memo_text.winfo_width()
        memo_height = self.memo_text.winfo_height()
        
        # Check if click was outside memo box
        if (event.x_root < memo_x or event.x_root > memo_x + memo_width or
            event.y_root < memo_y or event.y_root > memo_y + memo_height):
            if self.memo_text['state'] == 'normal':
                self.save_and_disable_memo()

    # Get shortcuts from config
    def get_shortcut_key(self, shortcut_name: str) -> str:
        return self.config.shortcuts.get(shortcut_name, '')
    
    def create_control_buttons(self, parent_frame: Frame):
        """
        Create classification buttons and navigation controls.
        """
        button_frame = Frame(parent_frame)
        button_frame.grid(row=3, column=0, padx=5, pady=5, sticky='ew')
        button_frame.columnconfigure((0, 1, 2, 3), weight=1)

        # Only create classification buttons if not in view mode
        if not self.config.view_mode and self.config.specific_view_mode is None:
            # Create classification buttons
            self.create_classification_buttons(button_frame)

        # Create navigation buttons (these are always visible)
        self.create_navigation_buttons(button_frame)
        

    @handle_exceptions
    def jump_to_image(self):
        """
        Jump to a specific image by index (1-based).
        """
        try:
            index = int(self.jump_entry.get())
            # Convert 1-based index to 0-based index
            zero_based_index = index - 1
            
            if 0 <= zero_based_index < self.num_images:
                self.index = zero_based_index
                self.science_data = None
                self.reference_data = None
                self.display_images()
                logging.info(f"Jumped to image {index} of {self.num_images}.")
                # Clear the entry box after successful jump
                self.jump_entry.delete(0, 'end')
                # Remove focus from the entry box
                self.master.focus_set()
            else:
                messagebox.showerror(
                    "Error", 
                    f"Please enter a number between 1 and {self.num_images}."
                )
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number.")

    def create_navigation_buttons(self, button_frame: Frame):
        """Create navigation buttons"""
        # Main navigation frame
        nav_frame = Frame(button_frame)
        nav_frame.grid(row=1, column=0, columnspan=4, pady=2)
        
        # Center the navigation buttons
        nav_buttons_frame = Frame(nav_frame)
        nav_buttons_frame.pack(expand=True)
        
        self.prev_button = Button(
            nav_buttons_frame,
            text=f"Previous ({self.config.shortcuts.get('prev_key', '')})",
            command=self.prev_image
        )
        self.prev_button.pack(side='left', padx=5)
        
        # Only show goto_unclassified button if not in specific view mode
        if not self.config.specific_view_mode:
            self.goto_unclassified_button = Button(
                nav_buttons_frame,
                text=f"Go to Unclassified ({self.config.shortcuts.get('goto_unclassified_key', '')})",
                command=self.goto_unclassified
            )
            self.goto_unclassified_button.pack(side='left', padx=5)
        
        self.next_button = Button(
            nav_buttons_frame,
            text=f"Next ({self.config.shortcuts.get('next_key', '')})",
            command=self.next_image
        )
        self.next_button.pack(side='left', padx=5)

        # Create a frame for all "Go to" controls
        goto_controls_frame = Frame(button_frame)
        goto_controls_frame.grid(row=2, column=0, columnspan=4, pady=2)
        
        # Center the "Go to" controls
        goto_inner_frame = Frame(goto_controls_frame)
        goto_inner_frame.pack(expand=True)
        
        # Tile ID selection controls
        tile_frame = Frame(goto_inner_frame)
        tile_frame.pack(side='left', padx=20)
        Label(tile_frame, text="Go to Tile ID:").pack(side='left', padx=2)
        
        # Get unique tile IDs from DataFrame and sort them
        available_tiles = sorted(self.data_manager.region_df['tile_id'].unique())
        self.tile_combobox = ttk.Combobox(tile_frame, values=available_tiles, width=10)
        if available_tiles:  # Set default value if available
            self.tile_combobox.set(available_tiles[0])
        self.tile_combobox.pack(side='left', padx=2)
        
        # Bind combobox selection to goto_tile_id
        self.tile_combobox.bind('<<ComboboxSelected>>', self.goto_tile_id)
        
        # Jump to Unique Number controls
        unique_frame = Frame(goto_inner_frame)
        unique_frame.pack(side='left', padx=20)
        Label(unique_frame, text="Go to Unique Number:").pack(side='left', padx=2)
        self.unique_entry = Entry(unique_frame, width=10)
        self.unique_entry.pack(side='left', padx=2)
        Button(unique_frame, text="Go", command=self.goto_unique_number).pack(side='left', padx=2)        
        
        # Jump to Image controls
        jump_frame = Frame(goto_inner_frame)
        jump_frame.pack(side='left', padx=20)
        max_index = len(self.data_manager.region_df) if not self.config.specific_view_mode else \
                    len(self.data_manager.region_df[self.data_manager.region_df[self.config.specific_view_mode] == 1])
        Label(jump_frame, text=f"Go to Image (1-{max_index}):").pack(side='left', padx=2)
        self.jump_entry = Entry(jump_frame, width=10)
        self.jump_entry.pack(side='left', padx=2)
        Button(jump_frame, text="Go", command=self.jump_to_image).pack(side='left', padx=2)

        # Zoom controls
        zoom_frame = Frame(button_frame)
        zoom_frame.grid(row=5, column=0, columnspan=4, pady=2)
        
        # Center the zoom controls
        zoom_inner_frame = Frame(zoom_frame)
        zoom_inner_frame.pack(expand=True)

        self.zoom_in_button = Button(
            zoom_inner_frame,
            text=f"Zoom In ({self.config.shortcuts.get('zoom_in_key', '')})",
            command=self.zoom_in
        )
        self.zoom_in_button.pack(side='left', padx=5)

        self.reset_zoom_button = Button(
            zoom_inner_frame,
            text=f"Reset Zoom ({self.config.shortcuts.get('reset_zoom_key', '')})",
            command=self.reset_zoom
        )
        self.reset_zoom_button.pack(side='left', padx=5)

        self.zoom_out_button = Button(
            zoom_inner_frame,
            text=f"Zoom Out ({self.config.shortcuts.get('zoom_out_key', '')})",
            command=self.zoom_out
        )
        self.zoom_out_button.pack(side='left', padx=5)

        # Show Sci & Ref Image controls
        checkbox_frame = Frame(button_frame)
        checkbox_frame.grid(row=6, column=0, columnspan=4, pady=2)
        
        self.sci_ref_var = IntVar(value=int(self.sci_ref_visible))
        self.show_sci_ref_checkbox = Checkbutton(
            checkbox_frame,
            text=f"Show Sci & Ref Images ({self.config.shortcuts.get('toggle_sci_ref_key', '')})",
            variable=self.sci_ref_var,
            command=self.toggle_sci_ref_image
        )
        self.show_sci_ref_checkbox.pack(expand=True)

    def create_classification_buttons(self, button_frame: Frame):
        """Create classification buttons dynamically from config."""
        for i, label in enumerate(self.config.classification_labels):
            # Fix: Change shortcut key lookup to match config file format
            shortcut_key = self.config.shortcuts.get(f'{label.lower()}_key', '')
            button = Button(
                button_frame,
                text=f"{label} ({shortcut_key})",
                command=lambda l=label: self.save_classification(l, 1)
            )
            button.grid(row=0, column=i, padx=5, pady=2, sticky='ew')
            self.classification_buttons[label] = button

    @handle_exceptions
    def goto_tile_id(self, event=None):
        """Jump to the first image of the selected tile ID."""
        try:
            selected_tile = self.tile_combobox.get()
            if not selected_tile:
                messagebox.showwarning("Warning", "Please select a tile ID")
                return
                
            # Find first image with selected tile ID using data_manager's region_df
            tile_images = self.data_manager.region_df[
                self.data_manager.region_df['tile_id'] == selected_tile
            ]
            
            if tile_images.empty:
                messagebox.showwarning("Warning", f"No images found for tile {selected_tile}")
                return
                
            # Get the index of the first image for this tile
            first_tile_index = tile_images.index[0]
            
            # Update display
            self.index = first_tile_index
            self.science_data = None
            self.reference_data = None
            self.display_images()
            
            # Remove focus from combobox without text selection
            self.master.focus_set()
            self.tile_combobox.selection_clear()
            
            logging.info(f"Jumped to first image of tile {selected_tile} at index {first_tile_index}")
            
        except Exception as e:
            logging.error(f"Error jumping to tile ID: {e}")
            messagebox.showerror("Error", f"Failed to jump to tile: {e}")

    def goto_unclassified(self):
        """Find and navigate to the first unclassified image."""
        try:
            # Filter out the total row (file_index == -1) and find first unclassified image
            unclassified = self.data_manager.region_df[
                (self.data_manager.region_df['file_index'] != -1) &  # Exclude total row
                ~self.data_manager.region_df[self.config.classification_labels].any(axis=1)
            ]
            
            if not unclassified.empty:
                first_unclassified = unclassified.index[0]
                # Use jump to image functionality
                self.index = first_unclassified
                self.science_data = None
                self.reference_data = None
                self.display_images()
                logging.info(f"Moved to unclassified image: Index {first_unclassified}")
            else:
                messagebox.showinfo("Info", "No unclassified images found!")
                logging.info("No unclassified images found")
                # If no unclassified images found, go to first valid image
                valid_images = self.data_manager.region_df[
                    self.data_manager.region_df['file_index'] != -1
                ]
                if not valid_images.empty:
                    self.index = valid_images.index[0]
                    self.display_images()
                
        except Exception as e:
            logging.error(f"Error finding unclassified image: {e}")
            raise

    def create_progress_bar(self, parent_frame: Frame):
        """
        Create a progress bar to display processing progress.
        """
        # Progress bar frame
        progress_frame = Frame(parent_frame)
        progress_frame.grid(row=4, column=0, padx=5, pady=5, sticky='ew')
        
        # Configure progress bar style
        style = Style()
        style.layout('text.Horizontal.TProgressbar',
                     [('Horizontal.Progressbar.trough',
                       {'children': [('Horizontal.Progressbar.pbar',
                                      {'side': 'left', 'sticky': 'ns'})],
                        'sticky': 'nswe'}),
                      ('Horizontal.Progressbar.label', {'sticky': ''})])
        style.configure('text.Horizontal.TProgressbar', text='0%')

        self.progress_var = IntVar()
        self.progress = Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            style='text.Horizontal.TProgressbar'
        )
        self.progress.grid(row=0, column=0, sticky='ew', padx=5, pady=2)

        # Create status text widget
        self.status_text = Text(
            progress_frame,
            height=4,
            width=40,
            font=("Helvetica", 10),
            state='disabled'
        )
        self.status_text.grid(row=1, column=0, sticky='ew', padx=5, pady=2)

        # Configure grid weights for progress_frame
        progress_frame.columnconfigure(0, weight=1)

    def create_scale_labels(self, parent_frame: Frame):
        """
        Create labels to display scale, vmin, and vmax settings.
        """
        scale_frame = Frame(parent_frame)
        scale_frame.grid(row=5, column=0, sticky='ew', padx=5, pady=5)
        scale_frame.columnconfigure((0, 1, 2), weight=1)

        # Labels for Subtracted Image
        self.scale_sub_label = Label(scale_frame, text="Subtracted Image - Scale: ")
        self.vmin_sub_label = Label(scale_frame, text="Vmin: ")
        self.vmax_sub_label = Label(scale_frame, text="Vmax: ")

        self.scale_sub_label.grid(row=0, column=0, sticky='w')
        self.vmin_sub_label.grid(row=1, column=0, sticky='w')
        self.vmax_sub_label.grid(row=2, column=0, sticky='w')

        # Labels for Science Image
        self.scale_sci_label = Label(scale_frame, text="Science Image - Scale: ")
        self.vmin_sci_label = Label(scale_frame, text="Vmin: ")
        self.vmax_sci_label = Label(scale_frame, text="Vmax: ")

        self.scale_sci_label.grid(row=0, column=1, sticky='w', padx=(20, 0))
        self.vmin_sci_label.grid(row=1, column=1, sticky='w', padx=(20, 0))
        self.vmax_sci_label.grid(row=2, column=1, sticky='w', padx=(20, 0))

        # Labels for Reference Image
        self.scale_ref_label = Label(scale_frame, text="Reference Image - Scale: ")
        self.vmin_ref_label = Label(scale_frame, text="Vmin: ")
        self.vmax_ref_label = Label(scale_frame, text="Vmax: ")

        self.scale_ref_label.grid(row=0, column=2, sticky='w', padx=(20, 0))
        self.vmin_ref_label.grid(row=1, column=2, sticky='w', padx=(20, 0))
        self.vmax_ref_label.grid(row=2, column=2, sticky='w', padx=(20, 0))

    def create_directories_text(self, parent_frame: Frame):
        """
        Create directories text below the progress bar to prevent overlapping.
        """
        directories_frame = Frame(parent_frame)
        directories_frame.grid(row=6, column=0, sticky='ew', padx=5, pady=(0, 5))
        directories_frame.columnconfigure(0, weight=1)
        directories_text = f"Image Pattern: {self.config.file_pattern}\nCSV File: {self.config.output_csv_file}"
        Label(
            directories_frame,
            text=directories_text,
            justify='left',
            font=("Helvetica", 10)
        ).pack(anchor='w')

    def bind_shortcuts(self):
        """
        Bind keyboard shortcuts to their respective functions.
        All shortcuts are loaded from config file.
        """
        def handle_shortcut(event):
            if hasattr(self, 'memo_editing') and self.memo_editing:  # Skip if editing memo
                return
                
            key = event.keysym.lower()  # Convert to lowercase for consistent matching
            
            # Handle classification shortcuts - disabled in view modes
            if not self.config.view_mode and self.config.specific_view_mode is None:
                for label in self.config.classification_labels:
                    shortcut = self.config.shortcuts.get(f'{label.lower()}_key', '').lower()
                    if key == shortcut:
                        self.save_classification(label, 1)
                        return
                    
            # Handle navigation shortcuts
            if key == self.config.shortcuts.get('next_key', '').lower():
                self.next_image()
            elif key == self.config.shortcuts.get('prev_key', '').lower():
                self.prev_image()
            elif key == self.config.shortcuts.get('goto_unclassified_key', '').lower():
                self.goto_unclassified()
            elif key == self.config.shortcuts.get('zoom_in_key', '').lower():
                self.zoom_in()
            elif key == self.config.shortcuts.get('zoom_out_key', '').lower():
                self.zoom_out()
            elif key == self.config.shortcuts.get('reset_zoom_key', '').lower():
                self.reset_zoom()
            
            # Handle Control key combinations
            if event.state & 4:  # Control key is pressed
                if key == self.config.shortcuts.get('toggle_sci_ref_key', '').split('-')[-1].lower():
                    self.toggle_sci_ref_var()

        # Bind keyboard events to the shortcut handler
        self.master.bind('<Key>', handle_shortcut)


    def update_progress_display(self):
        """Update progress information display in GUI."""
        try:
            # Get progress statistics from DataManager
            progress_stats = self.data_manager.calculate_progress()
            
            if not progress_stats:
                return
            
            # Format total progress
            total = progress_stats['total']
            progress_text = (f"Total Progress: {total['classified']}/{total['total']} "
                           f"({total['percent']:.2f}%)\n\n")
            
            # Format progress by tile
            progress_text += "Progress by Tile:\n"
            for tile_id, stats in progress_stats['tiles'].items():
                progress_text += (f"{tile_id}: {stats['classified']}/{stats['total']} "
                                f"({stats['percent']:.2f}%)\n")
            
            # Update status text widget
            self.status_text.config(state='normal')
            self.status_text.delete('1.0', 'end')
            self.status_text.insert('1.0', progress_text)
            self.status_text.config(state='disabled')
            
            # Update progress bar if needed
            total_percent = total['percent']
            self.progress_var.set(int(total_percent))
            style = Style()
            style.configure('text.Horizontal.TProgressbar', 
                          text=f'{total_percent:.2f}%')
            
        except Exception as e:
            logging.error(f"Error updating progress display: {e}")

    @handle_exceptions
    def display_images(self):
        """Display current set of images with proper scaling and normalization."""
        try:
            # Get current row outside the if statement
            current_row = self.data_manager.region_df.iloc[self.index]
            
            # Initialize image data variables
            sub_data, new_data, ref_data = None, None, None
            
            # Check if we need to load new images or use cached ones
            if not hasattr(self, '_current_index') or self._current_index != self.index:
                self._current_index = self.index
                cache_key = f"{current_row['tile_id']}_{current_row['unique_number']}"
                
                # Load new images if needed
                if not hasattr(self, '_current_cache_key') or self._current_cache_key != cache_key:
                    self._current_cache_key = cache_key
                    sub_data, new_data, ref_data = self.data_manager.load_image_data(self.index)
                    self._current_image_data = (sub_data, new_data, ref_data)
                else:
                    sub_data, new_data, ref_data = self._current_image_data
                    
                # Clear all axes when changing images
                for ax in self.axes:
                    ax.clear()
            else:
                # Use cached image data
                sub_data, new_data, ref_data = self._current_image_data
                # Clear all axes when toggling visibility
                for ax in self.axes:
                    ax.clear()

            # Set up the figure title
            title = f"Tile ID: {current_row['tile_id']} - Unique Number: {current_row['unique_number']}"
            self.fig.suptitle(title, fontsize=14, fontweight='bold')

            # Prepare common image display arguments
            img_args = {'origin': 'lower'}
            if self.image_processor.is_fits:
                img_args['cmap'] = 'gray'

            # Display images based on visibility settings
            if self.sci_ref_visible:
                # Science Image (left)
                if new_data is not None:
                    if self.image_processor.is_fits:
                        img_args['norm'] = self.image_processor.prepare_normalization(
                            new_data, self.config.vmin_science, self.config.vmax_science
                        )
                    self.axes[0].imshow(new_data, **img_args)
                    self.axes[0].set_title("Science Image")
                
                # Reference Image (right)
                if ref_data is not None:
                    if self.image_processor.is_fits:
                        img_args['norm'] = self.image_processor.prepare_normalization(
                            ref_data, self.config.vmin_reference, self.config.vmax_reference
                        )
                    self.axes[2].imshow(ref_data, **img_args)
                    self.axes[2].set_title("Reference Image")

            # Always display subtracted image (center)
            if sub_data is not None:
                if self.image_processor.is_fits:
                    img_args['norm'] = self.image_processor.prepare_normalization(
                        sub_data, self.config.vmin_subtracted, self.config.vmax_subtracted
                    )
                self.axes[1].imshow(sub_data, **img_args)
                self.axes[1].set_title("Subtracted Image")

            # Remove ticks from all axes
            for ax in self.axes:
                ax.set_xticks([])
                ax.set_yticks([])

            # Update zoom parameters when new image is loaded
            if sub_data is not None:
                self.original_size = [sub_data.shape[1], sub_data.shape[0]]
                self.zoom_center = [self.original_size[0]/2, self.original_size[1]/2]
                self.view_size = [size/self.zoom_level for size in self.original_size]

            # Update zoom and draw
            self.update_zoom()
            self.canvas.draw()
            
            # Update progress display
            self.update_progress_display()
            
            # Load memo for current image
            self.load_memo(current_row['unique_number'])
            
            # Update scale labels
            self.update_scale_labels(sub_data, new_data, ref_data)
            
            # Explicitly manage cache
            self.data_manager.cleanup_cache(self.index)
            
            # Start preloading next batch
            self.data_manager.start_preloading(self.index)

        except Exception as e:
            logging.error(f"Error in display_images: {e}")
            raise

    def update_zoom(self):
        """Update the display with current zoom level."""
        if not hasattr(self, 'zoom_level'):
            self.zoom_level = self.config.initial_zoom
            
        for ax in self.axes:
            ax.set_xlim(self.zoom_center[0] - self.view_size[0]/2, 
                        self.zoom_center[0] + self.view_size[0]/2)
            ax.set_ylim(self.zoom_center[1] - self.view_size[1]/2, 
                        self.zoom_center[1] + self.view_size[1]/2)
        
        self.canvas.draw_idle()
            
    def zoom_in(self):
        """Zoom in on the images."""
        try:
            new_zoom = min(self.zoom_level * (1 + self.config.zoom_step), 
                          self.config.zoom_max)
            if new_zoom != self.zoom_level:
                self.zoom_level = new_zoom
                self.view_size = [size/self.zoom_level for size in self.original_size]
                self.update_zoom()
                # Update slider value without triggering callback
                self.zoom_slider.set_val(new_zoom)
                logging.debug(f"Zoomed in to {new_zoom}x")
        except Exception as e:
            logging.error(f"Error in zoom_in: {e}")

    def zoom_out(self):
        """Zoom out from the images."""
        try:
            new_zoom = max(self.zoom_level / (1 + self.config.zoom_step), 
                          self.config.zoom_min)
            if new_zoom != self.zoom_level:
                self.zoom_level = new_zoom
                self.view_size = [size/self.zoom_level for size in self.original_size]
                self.update_zoom()
                # Update slider value without triggering callback
                self.zoom_slider.set_val(new_zoom)
                logging.debug(f"Zoomed out to {new_zoom}x")
        except Exception as e:
            logging.error(f"Error in zoom_out: {e}")

    def reset_zoom(self):
        """Reset zoom to initial value."""
        try:
            self.zoom_level = self.config.initial_zoom
            self.view_size = [size/self.zoom_level for size in self.original_size]
            self.update_zoom()
            # Update slider value without triggering callback
            self.zoom_slider.set_val(self.config.initial_zoom)
            logging.debug("Zoom reset to initial value")
        except Exception as e:
            logging.error(f"Error resetting zoom: {e}")

    @handle_exceptions
    def update_scale_labels(self, sub_data, new_data, ref_data):
        """Update the scale, vmin, vmax labels with current settings."""
        try:
            if self.config.file_type == 'fits':
                # Update labels for FITS files
                if sub_data is not None:
                    vmin = np.min(sub_data) if self.config.vmin_subtracted == 'min' else np.median(sub_data)
                    vmax = np.max(sub_data) if self.config.vmax_subtracted == 'max' else np.median(sub_data)
                    self.scale_sub_label.config(text=f"Subtracted Image - Scale: {self.config.scale}")
                    self.vmin_sub_label.config(text=f"Vmin: {vmin:.2f}")
                    self.vmax_sub_label.config(text=f"Vmax: {vmax:.2f}")

                if new_data is not None:
                    vmin = np.min(new_data) if self.config.vmin_science == 'min' else np.median(new_data)
                    vmax = np.max(new_data) if self.config.vmax_science == 'max' else np.median(new_data)
                    self.scale_sci_label.config(text=f"Science Image - Scale: {self.config.scale}")
                    self.vmin_sci_label.config(text=f"Vmin: {vmin:.2f}")
                    self.vmax_sci_label.config(text=f"Vmax: {vmax:.2f}")

                if ref_data is not None:
                    vmin = np.min(ref_data) if self.config.vmin_reference == 'min' else np.median(ref_data)
                    vmax = np.max(ref_data) if self.config.vmax_reference == 'max' else np.median(ref_data)
                    self.scale_ref_label.config(text=f"Reference Image - Scale: {self.config.scale}")
                    self.vmin_ref_label.config(text=f"Vmin: {vmin:.2f}")
                    self.vmax_ref_label.config(text=f"Vmax: {vmax:.2f}")
            else:
                # Simple labels for PNG files
                self.scale_sub_label.config(text="Subtracted Image - Scale: PNG")
                self.vmin_sub_label.config(text="")
                self.vmax_sub_label.config(text="")
                self.scale_sci_label.config(text="Science Image - Scale: PNG")
                self.vmin_sci_label.config(text="")
                self.vmax_sci_label.config(text="")
                self.scale_ref_label.config(text="Reference Image - Scale: PNG")
                self.vmin_ref_label.config(text="")
                self.vmax_ref_label.config(text="")
                
        except Exception as e:
            logging.error(f"Error updating scale labels: {e}")
    
    def save_and_disable_memo(self):
        """Save memo content and disable editing."""
        try:
            if self.memo_text['state'] == 'normal':
                # Get memo text (strip to remove extra whitespace)
                memo_text = self.memo_text.get('1.0', 'end-1c').strip()
                
                # Update DataFrame with memo text
                self.data_manager.region_df.at[self.index, 'Memo'] = memo_text
                self.data_manager.save_dataframe()
                
                # Disable text box and change appearance
                self.memo_text.config(state='disabled', bg='#F0F0F0')
                
                current_row = self.data_manager.region_df.iloc[self.index]
                logging.info(f"Memo saved for Tile: {current_row['tile_id']}, "
                            f"Number: {current_row['unique_number']}, "
                            f"Memo: '{memo_text}'")
                
        except Exception as e:
            logging.error(f"Error saving memo: {e}")
            messagebox.showerror("Error", f"Failed to save memo: {e}")

    def load_memo(self, unique_number: int):
        """Load and display memo for the current image."""
        try:
            current_row = self.data_manager.region_df.iloc[self.index]
            
            # Convert memo to string and handle NaN/float cases
            memo = str(current_row.get('Memo', ''))
            if memo.lower() == 'nan':
                memo = ''
                
            # Enable temporarily to update text
            self.memo_text.config(state='normal')
            self.memo_text.delete('1.0', 'end')
            self.memo_text.insert('1.0', memo)
            self.memo_text.config(state='disabled', bg='#F0F0F0')
            
            logging.debug(f"Loaded memo for Tile: {current_row['tile_id']}, "
                         f"Number: {current_row['unique_number']}, "
                         f"Memo: '{memo}'")
            
        except Exception as e:
            logging.error(f"Error loading memo: {e}")

    def toggle_sci_ref_var(self):
        """
        Toggle the Sci & Ref IntVar value.
        """
        current = self.sci_ref_var.get()
        self.sci_ref_var.set(0 if current else 1)
        logging.debug(f"toggle_sci_ref_var called. Sci & Ref visibility set to {bool(self.sci_ref_var.get())}")
        self.toggle_sci_ref_image()

    @handle_exceptions
    def toggle_sci_ref_image(self):
        """
        Toggle the visibility of science and reference images.
        """
        self.sci_ref_visible = bool(self.sci_ref_var.get())
        self.science_data = None
        self.reference_data = None
        self.display_images()
        logging.info(f"Sci & Ref images visibility set to {self.sci_ref_visible}.")

    @handle_exceptions
    def next_image(self):
        """
        Navigate to the next image, ensuring current image is classified.
        """
        try:
            # Get current unique number from DataFrame
            current_row = self.data_manager.region_df.iloc[self.index]
            unique_number = current_row['unique_number']
            
            # Check if current image is classified
            if not self.is_classified(unique_number) and not self.config.view_mode:
                response = messagebox.askyesno(
                    "Unclassified Image",
                    "The current image has not been classified. Do you want to proceed to the next image?"
                )
                if not response:
                    logging.info("User chose not to navigate to the next image without classification.")
                    return

            # Actually increment the index if we're not at the end
            if self.index < self.num_images - 1:
                self.index += 1  # Add this line to increment the index
                self.science_data = None
                self.reference_data = None
                self.display_images()
                
                # Start preloading for next batch
                self.data_manager.start_preloading(self.index)
            else:
                messagebox.showinfo("Info", "You have reached the last image.")
                logging.info("Reached the last image.")
                
        except Exception as e:
            logging.error(f"Error in next_image: {e}")
            raise

    @handle_exceptions
    def prev_image(self):
        """
        Navigate to the previous image.
        """
        if self.index > 0:
            self.index -= 1
            self.science_data = None
            self.reference_data = None
            self.display_images()
        else:
            messagebox.showinfo("Info", "You are at the first image.")
            logging.info("At the first image.")

    @handle_exceptions
    def save_classification(self, classification: str, value: int = 1):
        """Save classification and move to next image."""
        if self.config.view_mode:
            messagebox.showinfo("View Mode", "Classification is disabled in View Mode")
            return

        try:
            current_row = self.data_manager.region_df.iloc[self.index]
            
            # Reset all classification columns to 0 first
            for label in self.config.classification_labels:
                self.data_manager.region_df.at[self.index, label] = 0
                
            # Set the classification value to 1 for the clicked label
            self.data_manager.region_df.at[self.index, classification] = value
            
            # Update metadata
            memo_text = self.memo_text.get('1.0', 'end').strip()
            # Convert empty string to '' instead of NaN and ensure string type
            memo_value = str('' if not memo_text else memo_text)
            self.data_manager.region_df.at[self.index, 'Memo'] = memo_value
            self.data_manager.region_df.at[self.index, 'Scale'] = str(self.config.scale)
            
            # Save DataFrame
            self.data_manager.save_dataframe(callback=self.after_classification_save)
            
            # Log the save
            tile_id = current_row['tile_id']
            unique_number = current_row['unique_number']
            logging.info(f"Saved classification '{classification}' for Tile: {tile_id}, "
                        f"Number: {unique_number}, Index: {self.index}")
            
        except Exception as e:
            logging.error(f"Error in save_classification: {e}")
            messagebox.showerror("Error", f"Failed to save classification: {e}")

    def after_classification_save(self):
        """Called after classification is saved successfully."""
        self.hide_saving_indicator()
        self.update_progress_display()
        self.next_image()

    def hide_saving_indicator(self):
        """Hide the saving in progress indicator."""
        # Re-enable classification buttons
        for button in self.classification_buttons.values():
            button.config(state='normal')
        
        # Clear saving status
        if hasattr(self, 'status_text'):
            self.status_text.config(state='normal')
            self.status_text.delete('1.0', 'end')
            self.status_text.config(state='disabled')

    def is_classified(self, unique_number: int) -> bool:
        """
        Check if an image has been classified.

        Parameters:
            unique_number (int): The unique number of the image.

        Returns:
            bool: True if the image is classified, False otherwise.
        """
        try:
            # Get current row from DataFrame
            current_row = self.data_manager.region_df.iloc[self.index]
            
            # Check if any classification column has value 1
            return any(current_row[col] == 1 for col in self.config.classification_labels)
            
        except Exception as e:
            logging.error(f"Error checking classification status: {e}")
            return False

    def init_mode_settings(self):
        """Initialize mode-specific settings based on config"""
        if self.config.specific_view_mode:
            # Specific view mode
            self.title = f"Transient Tool - {self.config.specific_view_mode} View Mode"
            logging.info(f"Initializing in specific view mode: {self.config.specific_view_mode}")
            self.filter_specific_images()
            
        elif self.config.view_mode:
            # View Mode
            self.title = "Transient Tool - View Mode"
            logging.info("Initializing in view mode")

        else:
            # Normal Mode
            self.title = "Transient Tool"
            logging.info("Initializing in normal mode")
            
        self.master.title(self.title)
        
    def filter_specific_images(self):
        """Filter images based on specific view mode"""
        if self.config.specific_view_mode and hasattr(self.data_manager, 'region_df'):
            # Get images where the specified column has value 1
            filtered_df = self.data_manager.region_df[
                self.data_manager.region_df[self.config.specific_view_mode] == 1
            ]
            
            if filtered_df.empty:
                messagebox.showwarning("Warning", 
                    f"No images found with classification '{self.config.specific_view_mode}'")
                # Reset to normal mode if no matching images found
                self.config.specific_view_mode = None
                self.title = "Transient Tool"
                self.master.title(self.title)
                return
                
            # Update data manager with filtered dataframe
            self.data_manager.region_df = filtered_df
            self.num_images = len(filtered_df)
            self.index = 0  # Reset index to start
            
            logging.info(f"Filtered to {self.num_images} images with classification "
                        f"'{self.config.specific_view_mode}'")
            
    def goto_unique_number(self, event=None):
        """Jump to image with specific unique number in current tile."""
        try:
            unique_num = int(self.unique_entry.get())
            current_row = self.region_df.iloc[self.index]
            current_tile = current_row['tile_id']
            
            # Find the unique number within the current tile
            query = (self.region_df['tile_id'] == current_tile) & \
                    (self.region_df['unique_number'] == unique_num)
            matching_rows = self.region_df[query]
            
            if matching_rows.empty:
                messagebox.showwarning(
                    "Warning", 
                    f"No image found with unique number {unique_num} in tile {current_tile}"
                )
                return
                
            # Jump to the found image
            self.index = matching_rows.index[0]
            self.science_data = None
            self.reference_data = None
            self.display_images()
            
            # Clear entry and remove focus
            self.unique_entry.delete(0, 'end')
            self.master.focus_set()
            
            logging.info(f"Jumped to unique number {unique_num} in tile {current_tile}")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
        except Exception as e:
            logging.error(f"Error jumping to unique number: {e}")
            messagebox.showerror("Error", f"Failed to jump to unique number: {e}")



def main():
    """
    Main function to run the TransientTool application.
    """
    try:
        # Set up basic logging before loading config
        logging.basicConfig(
            filename='transient_tool.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True  # Force reconfiguration of the logging
        )
        
        logging.info("=" * 50)
        logging.info("Starting Transient Tool Application")
        logging.info("=" * 50)
        
        root = Tk()
        config = Config.load_config()
        
        # Log important configuration settings
        logging.info("-" * 50)
        logging.info("Configuration Summary:")
        logging.info(f"- Data Directory: {config.data_directory}")
        logging.info(f"- File Type: {config.file_type}")
        logging.info(f"- Output CSV: {config.output_csv_file}")
        logging.info(f"- View Mode: {config.view_mode}")
        logging.info(f"- Quick Start: {config.quick_start}")
        logging.info(f"- Scale: {config.scale}")
        if config.tile_ids:
            logging.info(f"- Tile IDs: {', '.join(config.tile_ids)}")
        else:
            logging.info("- Tile IDs: Auto-detected")
        logging.info("-" * 50)
        
        # Update logging configuration with settings from config
        logging.getLogger().setLevel(getattr(logging, config.log_level))
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                handler.baseFilename = os.path.abspath(config.log_file)
        
        app = TransientTool(root, config)
        root.mainloop()
        
        # After the application is closed, log the end of the session
        logging.info("=" * 50)
        logging.info("Transient Tool Application Closed")
        logging.info("=" * 50)

    except Exception as e:
        logging.exception("Failed to start application")
        if 'root' in locals():
            messagebox.showerror("Error", f"Failed to start application: {e}")
            root.quit()

if __name__ == "__main__":
    main()
