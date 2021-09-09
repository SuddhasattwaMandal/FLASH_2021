from pathlib import Path
import os

CWD = Path(os.getcwd())
DATA_PATH = CWD / "data" #/ "pnCCDsamplefiles"
PNCCD_DATA_PATH = DATA_PATH

SAVE_PATH = CWD / "saved_files"

SAVE_PATH_IMAGES = SAVE_PATH / "images"
SAVE_PATH_TABLE = SAVE_PATH / "tables"


class Constants():

    def __init__(self):
        global DATA_PATH, SAVE_PATH_IMAGES, SAVE_PATH_TABLE, SAVE_PATH

        self.DATA_PATH = DATA_PATH
        self.SAVE_PATH = SAVE_PATH
        self.SAVE_PATH_IMAGES = SAVE_PATH_IMAGES
        self.SAVE_PATH_TABLE = SAVE_PATH_TABLE

    def change_data_path(self, new_path):
        self.DATA_PATH = Path(new_path)
        self.PNCCD_DATA_PATH = Path(new_path)



    def create_save_folders(self):
        [path.mkdir() for path in [self.DATA_PATH, self.SAVE_PATH, self.SAVE_PATH_IMAGES, self.SAVE_PATH_TABLE] if not path.is_dir()]
