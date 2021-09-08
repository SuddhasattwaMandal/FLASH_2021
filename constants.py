from pathlib import Path
import os

CWD = Path(os.getcwd())
DATA_PATH = CWD / "data"
SAVE_PATH = CWD / "saved_files"
SAVE_PATH_IMAGES = SAVE_PATH / "images"
SAVE_PATH_TABLE = SAVE_PATH / "tables"
PNCCD_DATA_PATH = DATA_PATH / "pnCCDsamplefiles"

def create_save_folders():
    [path.mkdir() for path in [DATA_PATH, SAVE_PATH, SAVE_PATH_IMAGES, SAVE_PATH_TABLE] if not path.is_dir()]
