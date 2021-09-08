# modules for data and path/file handling
import numpy as np
from pathlib import Path
import h5py
import json
import pandas as pd
from logging import warning, info
from datetime import datetime

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

######### H5 to Dataframe #########


def df_from_h5(h5_fp, threshold):
    columns = ["Train ID", "Cuml. Intensity", f"Hits", "Mean Value", "Dataset ID", "Timestamp",  "Temperature"]
    data = list([])

    images = load_pnccd_image_data(h5_fp, (threshold[0], threshold[1]))
    data.append(load_trainID(h5_fp))
    _, intensities, _ = integrated_intensity(h5_fp, 0)
    data.append(intensities)
    data.append([basic_hitfinding(img) for img in images])
    data.append([np.mean(img.flatten()) for img in images])
    id = load_datasetID(h5_fp)
    data.append([str(id) for _ in range(0, len(images))])
    data.append([datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S') for time in load_timestamp(h5_fp)])
    data.append(load_temperature(h5_fp))
    df = pd.DataFrame(data=data).transpose()
    df.columns = columns
    return df


def save_df_as_json(df: pd.DataFrame, save_path):
    if save_path:
        f = save_path / f"table_{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}.json"
        result = df.to_json(f, orient="table")
        return True
    else:
        warning("no save path has been specified yet")
        return False


def df_from_path(file_path):
    if file_path:
        data = json.load(open(file_path))
        df = pd.DataFrame(data["data"])
        return df
    return None


######### H5 FILEHANDLING ###########

def select_h5_files(PNCCD_DATA_PATH):
    nameSet = []
    times = []
    if isinstance(PNCCD_DATA_PATH, str):
        PNCCD_DATA_PATH = Path(PNCCD_DATA_PATH)
    for file in PNCCD_DATA_PATH.rglob("*"):
        # read new files containing the KEYWORDS and add to set
        if H5_TYPE and PNCCD_KEY.lower() in str(file.name):
            if file.is_file():
                stat = file.stat()
                time = stat.st_ctime
                nameSet.append(str(file.name))
                times.append(datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S'))
                parent = str(file.parent)
    return nameSet, parent, times


def data_from_key(h5_fp: Path, key: str) -> np.array:
    """ read pnccd .h5 file and return data to given key as numpy array"""
    with h5py.File(h5_fp, "r") as data:
        if not key in data[PNCCD_KEY].keys():
            raise ValueError(f"could not find key '{key}' in file {h5_fp}")
        else:
            return np.asarray(data[PNCCD_KEY][key])


def load_pnccd_image_data(h5_fp: Path, threshold):
    """
    loads newest .h5 file in directory containing PNCCD_KEY and IMAGE_KEY-
    return 2 dimensional numpy array representing the image
    if more than one image was taken, mean of image is returned.
    """
    gain = int(str(load_datasetID(h5_fp)).rsplit("_G")[-1][0:3])
    if not threshold:
        return data_from_key(h5_fp, IMAGE_KEY) * gain
    else:
        images = data_from_key(h5_fp, IMAGE_KEY) * gain
        return [np.clip(array, threshold[0], threshold[1]) for array in images]


def load_temperature(h5_fp):
    return data_from_key(h5_fp, TEMPERATURE_KEY)


def load_trainID(h5_fp):
    return data_from_key(h5_fp, TRAINID_KEY)


def load_timestamp(h5_fp):
    return data_from_key(h5_fp, TIMESTAMP_KEY)


def load_datasetID(h5_fp):
    return data_from_key(h5_fp, DATASETID_KEY)


def select_recent_h5_file(PNCCD_DATA_PATH: Path):
    """
    check for most recent .h5 file in the PNCCD_DATA_PATH
    containing PNCCD_KEY and IMAGE_KEY in the file name. return filepath of most recent file.
    Breaks down if files are removed from the directory while running.
    """
    global SAVED_FILES

    nameSet = set()
    for file in PNCCD_DATA_PATH.iterdir():
        # read new files containing the KEYWORDS and add to set
        if H5_TYPE and PNCCD_KEY.lower() in str(file):
            if file.is_file():
                nameSet.add(file)

    RETRIEVED_FILES = set()

    for name in nameSet:
        stat = name.stat()
        time = stat.st_ctime
        size = stat.st_size
        # Also consider using ST_MTIME to detect last time modified
        # Note that the time is saved at second positions
        # for comparision in next step
        RETRIEVED_FILES.add((name, time, size))

    NEW_FILES = RETRIEVED_FILES - SAVED_FILES

    # sort for most recent file and return file_path
    if NEW_FILES:
        if len(NEW_FILES) != 1:
            warning("more than one new file in directory")

        s = sorted(NEW_FILES, key=lambda file: file[1])
        file_path, time, size = s[-1]
        # reset variables
        NEW_FILES = set()
        SAVED_FILES = SAVED_FILES.union(RETRIEVED_FILES)
        return file_path, datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S'), size
    else:
        SAVED_FILES = RETRIEVED_FILES
        s = sorted(SAVED_FILES, key=lambda file: file[1])
        file_path, time, size = s[-1]
        return file_path, datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S'), size


def integrated_intensity(h5_fp, threshold=0):
    # calculate summed up image intensity
    img_ids = [ID for ID in load_trainID(h5_fp)]
    index = np.arange(0, len(img_ids), 1)
    intgr_intensities = [sum_array(image, threshold) for image in load_pnccd_image_data(h5_fp, threshold)]
    return img_ids, intgr_intensities, index



def N_brightest_images(h5_fp, N: int, threshold: int):
    if threshold is None:
        threshold = 0
    img_ids, intensities, index = integrated_intensity(h5_fp, threshold=threshold)
    unsorted = [(intensity, i, id) for intensity, i, id in zip(intensities, index, img_ids)]
    s = sorted(unsorted, key=lambda item: item[0])
    s = s[len(s)-N:len(s)]
    intensity, index, train_id = [entry[0] for entry in s], [entry[1] for entry in s], [entry[2] for entry in s]
    images = load_pnccd_image_data(h5_fp, threshold)
    brightest_images = [images[i] for i in index]
    return brightest_images, intensity, index, train_id


######## ARRAY CALCULATIONS ###########
def sum_array(image: np.ndarray, threshold):
    if isinstance(threshold, int):
        return np.sum(np.array(image) >= threshold)
    elif len(threshold) == 2:
        return np.sum(np.clip(image, threshold[0], threshold[1]))


def basic_hitfinding(image: np.ndarray):
    hit_array = np.array(image, dtype='int')
    return int(np.count_nonzero(hit_array))
