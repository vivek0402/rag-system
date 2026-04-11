# config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
INDEX_DIR = BASE_DIR / "data" / "index"
LC_INDEX_DIR = BASE_DIR / "data" / "lc_index"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")