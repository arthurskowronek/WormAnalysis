"""
Global configuration file for the project.
"""
from pathlib import Path
import datetime

# Get the project root (assuming we run from the project root)
PROJECT_ROOT = Path.cwd()

# Define all project directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESSOURCES_DIR = PROJECT_ROOT / "ressources"
SRC_DIR = PROJECT_ROOT / "src"
USER_DIR = PROJECT_ROOT / "user"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Image processing parameters
IMAGE_SIZE = (1024, 1024)

# Model parameters
DEFAULT_RANDOM_STATE = 42  # For reproducibility
DEFAULT_TEST_SIZE = 0.2  # 20% for testing
DEFAULT_CV_FOLDS = 5  # Number of cross-validation folds

# Date format for file naming
DATE_FORMAT = "%Y%m%d_%H%M%S"
CURRENT_DATE = datetime.datetime.now().strftime(DATE_FORMAT) 

# File naming
DEFAULT_PKL_NAME = f'dataset_{CURRENT_DATE}'
DEFAULT_MODEL_NAME = f'model_{CURRENT_DATE}.pkl'

# Live exposure time
LIVE_EXPOSURE_TIME = 10


# Print configuration for debugging
if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Current Date: {CURRENT_DATE}") 
    print(f"Default PKL Name: {DEFAULT_PKL_NAME}")
