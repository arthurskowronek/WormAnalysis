"""
Global configuration file for the project.
"""
import os
import yaml
import datetime
import pymmcore #Library to connect the the Micro-Manager core
from pathlib import Path

# Get the project root (assuming we run from the project root)
PROJECT_ROOT = Path.cwd()

# Define all project directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESSOURCES_DIR = PROJECT_ROOT / "ressources"
SRC_DIR = PROJECT_ROOT / "src"
USER_DIR = PROJECT_ROOT / "user"

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
DEFAULT_TEST_SIZE = 0.2  # 20% for testing
DEFAULT_CV_FOLDS = 5  # Number of cross-validation folds


def set_up_environment():
    # Create directories if they don't exist
    for directory in [DATA_DIR, MODELS_DIR, RESSOURCES_DIR, SRC_DIR, USER_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        
    # Initialize directory
    dirs_to_clear = ["Unclassified","Mutant_prediction","WT_prediction","Scan","Scan_modified"]
    for subdir in dirs_to_clear:
        directory = Path(DATA_DIR) / subdir
        for file in directory.iterdir():
            if file.is_file():
                file.unlink()
    
def loadCore():
    DIRECTORY = "C:/Program Files/Micro-Manager-2.0gamma" # Select the folder which contains Micro-Manager.
    CONFIG = "BESSEREAU_Lab.cfg" # Name of the config file (has to be in the Micro-Manager root folder)
    os.chdir(os.path.dirname(os.path.abspath(__file__))) # Set the current working directory
    mmc = pymmcore.CMMCore()
    mmc.setDeviceAdapterSearchPaths([DIRECTORY])
    mmc.loadSystemConfiguration(os.path.join(DIRECTORY, CONFIG))
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
    in the YAML file, preserving the first 9 lines.
    
    Parameters:
        start_x (float/int)
        start_y (float/int)
        end_x (float/int)
        end_y (float/int)
    """
    # New values to insert
    corner_params = {
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y
    }

    # Read the existing file
    try:
        with open(PARAMETERS_FILE, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    # Dump the corner parameters to YAML-formatted lines
    corner_yaml_lines = yaml.dump(corner_params, default_flow_style=False).splitlines(keepends=True)

    # Replace or append corner positions starting at line 10
    # Pad the list if it's shorter than 9 lines
    while len(lines) < 9:
        lines.append("\n")
    
    # Replace lines 9–12 or add them if not present
    lines = lines[:9] + corner_yaml_lines + lines[9 + len(corner_yaml_lines):]

    # Write everything back
    with open(PARAMETERS_FILE, "w") as f:
        f.writelines(lines)


    

