# config.py
import os
from pathlib import Path

# Root of the project
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data" / "raw"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")