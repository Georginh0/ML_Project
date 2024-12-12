from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Project root directory
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Directory structure for datasets and models
DATA_DIR = PROJ_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Dataset file path
DATASET_FILENAME = "stud copy.csv"
DATASET_PATH = PROCESSED_DATA_DIR / DATASET_FILENAME

# Artifacts directory
ARTIFACTS_DIR = PROJ_ROOT / "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Define exports
__all__ = ["DATASET_PATH", "ARTIFACTS_DIR", "PROJ_ROOT", "DATA_DIR", "PROCESSED_DATA_DIR"]
