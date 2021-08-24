# modules for data and path/file handling
import numpy as np
from pathlib import Path
import h5py

# variables which will be used often
PNCCD_DATA_PATH = Path("./data/pnCCD sample files/")

PNCCD_KEY = "PNCCD"
IMAGE_KEY = "image"

DATASETID_KEY = "dataSetID"
TEMPERATURE_KEY = "temperature"
TRAINID_KEY = "train ID"
TIMESTAMP_KEY = "timestamp sec"

KEY_LIST = [DATASETID_KEY, TEMPERATURE_KEY, TRAINID_KEY, TIMESTAMP_KEY]

H5_TYPE = ".h5"
# use this set to check for new files in directory
SAVED_FILES = set()

# threshold for hitfinding
THRESHOLD = 12000


def select_h5_files(PNCCD_DATA_PATH):
    nameSet = []
    if isinstance(PNCCD_DATA_PATH, str):
        PNCCD_DATA_PATH = Path(PNCCD_DATA_PATH)
    for file in PNCCD_DATA_PATH.rglob("*"):
        # read new files containing the KEYWORDS and add to set
        if H5_TYPE and PNCCD_KEY.lower() in str(file.name):
            if file.is_file():
                nameSet.append(str(file.name))
                parent = str(file.parent)
    return nameSet, parent


def data_from_key(h5_fp: Path, key: str) -> np.array:
    """ read pnccd .h5 file and return data to given key as numpy array"""
    with h5py.File(h5_fp, "r") as data:
        if not key in data[PNCCD_KEY].keys():
            raise ValueError(f"could not find key '{key}' in file {h5_fp}")
        else:
            return np.asarray(data[PNCCD_KEY][key])


def load_pnccd_image_data(h5_fp: Path):
    """
    loads newest .h5 file in directory containing PNCCD_KEY and IMAGE_KEY-
    return 2 dimensional numpy array representing the image
    if more than one image was taken, mean of image is returned.
    """
    pnccd_image = data_from_key(h5_fp, IMAGE_KEY)
    return pnccd_image


def load_temperature(h5_fp):
    return data_from_key(h5_fp, TEMPERATURE_KEY)


def load_trainID(h5_fp):
    return data_from_key(h5_fp, TRAINID_KEY)


def load_timestamp(h5_fp):
    return data_from_key(h5_fp, TIMESTAMP_KEY)


def load_datasetID(h5_fp):
    return data_from_key(h5_fp, DATASETID_KEY)


def basic_hitfinding(image: np.ndarray, threshold: int = 15000):
    hit = np.array(image > threshold, dtype='int')
    return int(np.count_nonzero(hit))
