"""
Global configuration file for the project.
"""
import os
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




    

