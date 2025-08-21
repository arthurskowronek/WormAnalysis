"""
Global configuration file for the project.
"""
import os
import csv
import yaml
import datetime
import pymmcore 
import traceback
from pathlib import Path

# Get the project root (assuming we run from the project root)
PROJECT_ROOT = Path.cwd()

# Define all project directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESSOURCES_DIR = PROJECT_ROOT / "ressources"
SRC_DIR = PROJECT_ROOT / "src"
USER_DIR = PROJECT_ROOT / "user"
LOG_DIR = PROJECT_ROOT / "logs"

# Path to the parameters file
PARAMETERS_FILE = Path(RESSOURCES_DIR) / "parameters.yaml"

# Date format for file naming
DATE_FORMAT = "%Y%m%d_%H%M%S"
CURRENT_DATE = datetime.datetime.now().strftime(DATE_FORMAT) 

# File naming
DEFAULT_PKL_NAME = f'dataset_{CURRENT_DATE}'
DEFAULT_MODEL_NAME = f'model_{CURRENT_DATE}.pkl'

# Image processing parameters
IMAGE_SIZE = (1024, 1024)

# Model parameters
DEFAULT_RANDOM_STATE = 42  # For reproducibility
DEFAULT_CV_FOLDS = 5  # Number of cross-validation folds

# Microscope parameters
EXPOSURE_TIME_LIVE = 50  # Default exposure time for live mode in milliseconds
NAME_CAMERA = "Camera-1" # TODO : name of the camera


def set_up_environment():
    """
    Set up the project's directory structure and clear specific subdirectories.

    This function ensures that all required directories (e.g., for data, models, 
    resources, source code, and user-specific files) exist by creating them if needed. 
    It also clears the contents of predefined subdirectories inside the data directory 
    (e.g., 'Unclassified', 'Mutant_prediction', etc.) by deleting all files within them.
    
    Directories are created with `parents=True` and `exist_ok=True` to handle nested 
    paths and avoid errors if they already exist.
    """
    # Create root-level directories
    for directory in [DATA_DIR, MODELS_DIR, RESSOURCES_DIR, SRC_DIR, USER_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # List of subdirectories to create in DATA_DIR
    data_subdirs = [
        "Dataset_pkl",
        "Mutant",
        "Mutant_prediction",
        "Scan",
        "Scan_modified",
        "Unclassified",
        "WT",
        "WT_prediction"
    ]

    # Create all subdirectories in DATA_DIR
    for subdir in data_subdirs:
        subdir_path = Path(DATA_DIR) / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)

    # Clear specific subdirectories
    dirs_to_clear = ["Unclassified", "Mutant_prediction", "WT_prediction"]
    for subdir in dirs_to_clear:
        directory = Path(DATA_DIR) / subdir
        for file in directory.iterdir():
            if file.is_file():
                file.unlink()

def clear_scan_directory():
    dirs_to_clear = ["Scan", "Scan_modified"]
    for subdir in dirs_to_clear:
        directory = Path(DATA_DIR) / subdir
        for file in directory.iterdir():
            if file.is_file():
                file.unlink()
    
def loadCore(verbose = False):
    """
    Initialize and configure the Micro-Manager core interface.

    This function sets the working directory to the script's location, initializes a 
    `CMMCore` object from the Micro-Manager API, sets the device adapter search path, 
    loads a specified configuration file, and sets the default exposure time.

    Returns:
        pymmcore.CMMCore: The initialized Micro-Manager core object ready for device control.

    Raises:
        Any exception raised by pymmcore methods (e.g., loading configuration or setting devices).
    """
    DIRECTORY = "C:/Program Files/Micro-Manager-2.0gamma" # Select the folder which contains Micro-Manager. # TODO
    CONFIG = "BESSEREAU_Lab.cfg" # Name of the config file (has to be in the Micro-Manager root folder) # TODO
    os.chdir(os.path.dirname(os.path.abspath(__file__))) # Set the current working directory
    config = load_config_file()
    
    mmc = pymmcore.CMMCore()
    mmc.setDeviceAdapterSearchPaths([DIRECTORY])
    mmc.loadSystemConfiguration(os.path.join(DIRECTORY, CONFIG))
    mmc.setExposure(EXPOSURE_TIME_LIVE)
    mmc.setAutoShutter(False)
    mmc.setProperty(NAME_CAMERA, "Binning",str(config.get("binning")))
    mmc.setProperty("Camera-1","PixelType","32bit")

    if verbose: 
        print(mmc.getDevicePropertyNames("Camera-1"))

    return mmc

def load_config_file():
    """
    Loads application parameters from a YAML file.

    Returns:
        dict: A dictionary containing parameter keys and their saved values.
            Returns an empty dictionary if the file does not exist.

    Notes:
        The parameters are loaded from the file defined by `PARAMETERS_FILE`.
    """
    if os.path.exists(PARAMETERS_FILE):
        with open(PARAMETERS_FILE, "r") as f:
            return yaml.safe_load(f)
    else:
        return {}

def save_corner_positions_into_yaml_config_file(start_x, start_y, end_x, end_y):
    """
    Updates the corner position parameters (start_x, start_y, end_x, end_y)
    in the YAML file by parsing and updating the existing configuration.
    
    Parameters:
    start_x (float/int): Starting X coordinate
    start_y (float/int): Starting Y coordinate  
    end_x (float/int): Ending X coordinate
    end_y (float/int): Ending Y coordinate
    """
    
    # New corner parameters to update
    corner_params = {
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y
    }
    
    # Read and parse the existing YAML file
    try:
        with open(PARAMETERS_FILE, "r") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config = {}
    
    # Update only the corner position parameters
    config.update(corner_params)
    
    # Write the updated configuration back to file
    with open(PARAMETERS_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def log_error(error, context=""):
    """
    Log an error to log.txt with timestamp and full traceback
    Also update CSV file with error count by context
    
    Args:
        error: The exception object
        context: Optional context description
    """
    timestamp = datetime.datetime.now().strftime(DATE_FORMAT)
    
    # Log to text file
    with open("logs/log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"\n{'='*50}\n")
        log_file.write(f"ERROR LOGGED: {timestamp}\n")
        if context:
            log_file.write(f"CONTEXT: {context}\n")
        log_file.write(f"ERROR TYPE: {type(error).__name__}\n")
        log_file.write(f"ERROR MESSAGE: {str(error)}\n")
        log_file.write(f"FULL TRACEBACK:\n{traceback.format_exc()}\n")
    
    # Update CSV file with error counts
    csv_file = "logs/error_counts.csv"
    error_counts = {}
    
    # Read existing CSV if it exists
    if os.path.exists(csv_file):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                error_counts[row["context"]] = int(row["count"])
    
    # Update count for this context
    if context in error_counts:
        error_counts[context] += 1
    else:
        error_counts[context] = 1
    
    # Write updated counts back to CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["context", "count"])  # Header
        for ctx, count in error_counts.items():
            writer.writerow([ctx, count])
            
    return context

def start_new_session_get_statistics():
    timestamp = datetime.datetime.now().strftime(DATE_FORMAT)
    
    # Update CSV file 
    csv_file = "logs/user_statistics.csv"
    
    # Créer le dossier logs s'il n'existe pas
    os.makedirs("logs", exist_ok=True)
    
    # Lire les données existantes
    existing_data = []
    fieldnames = ['id_session', 'date_session', 'nb_scans', 'nb_vers_detected', 
                 'nb_false_positives', 'nb_vers_missed', 'nb_vers_final']
    
    if os.path.exists(csv_file):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)
    
    # Calculer le numéro de session
    id_session = len(existing_data) + 1
    
    # Préparer les nouvelles données
    new_row = {
        'id_session': id_session,
        'date_session': timestamp,
        'nb_scans': 0,
        'nb_vers_detected': 0,
        'nb_false_positives': 0,
        'nb_vers_missed': 0,
        'nb_vers_final': 0
    }
    
    # Ajouter la nouvelle ligne
    existing_data.append(new_row)
    
    # Écrire toutes les données dans le CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_data)

def update_user_statistics(field_name, value):
    """
    Helper function to update a specific field in the last line of the CSV
    
    Args:
        field_name: Name of the field to update
        value: Value to set
    """
    csv_file = "logs/user_statistics.csv"
    fieldnames = ['id_session', 'date_session', 'nb_scans', 'nb_vers_detected',
                 'nb_false_positives', 'nb_vers_missed', 'nb_vers_final']
    
    if not os.path.exists(csv_file):
        print("CSV file does not exist. Please start a new session first.")
        return
    
    # Read existing data
    existing_data = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_data = list(reader)
    
    if not existing_data:
        print("No data in CSV file. Please start a new session first.")
        return
    
    # Update the last row
    last_row = existing_data[-1]
    last_row[field_name] = int(value)
    
    # Write back to CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_data)
        
def increment_user_statistics(field_name):
    """
    Add 1 to 'field_name' in the last line of the CSV file
    """
    csv_file = "logs/user_statistics.csv"
    fieldnames = ['id_session', 'date_session', 'nb_scans', 'nb_vers_detected',
                 'nb_false_positives', 'nb_vers_missed', 'nb_vers_final']
    
    if not os.path.exists(csv_file):
        print("CSV file does not exist. Please start a new session first.")
        return
    
    # Read existing data
    existing_data = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_data = list(reader)
    
    if not existing_data:
        print("No data in CSV file. Please start a new session first.")
        return
    
    # Increment nb_scans in the last row
    last_row = existing_data[-1]
    current_scans = int(last_row[field_name]) if last_row[field_name].isdigit() else 0
    last_row[field_name] = int(current_scans + 1) 
    
    # Write back to CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_data)
        