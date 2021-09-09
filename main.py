import PySimpleGUI as sg
from pathlib import Path
import numpy as np

from utils import select_h5_files
from utils import data_from_key, TRAINID_KEY, IMAGE_KEY
from constants import Constants
from live_window import live_window
from table_window import table
from single_window import single_window
import cv2
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

C = Constants()

SAVE_PATH = None
SELECTED_FILE = None
SHOWING_TABLE = False
THRESHOLD = 0

sg.theme('black')


def save(train_id: int):
    # 21526319
    all_train_ids = data_from_key(SELECTED_FILE, TRAINID_KEY)
    if not np.isin(train_id, all_train_ids):
        sg.popup_error(f"Train ID {train_id} not in dataset."
                       f"\nValid choices: {min(all_train_ids)} - {max(all_train_ids)}")
        return False
    image = data_from_key(SELECTED_FILE, IMAGE_KEY)[np.where(all_train_ids == train_id)]
    image = image[0, :, :] / np.max(image[0, :, :])
    uint_img = np.array(image * 255).astype('uint8')
    grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(grayImage, (450, 450))

    name = f"pnncd_trainID{train_id}.png"
    success = cv2.imwrite(str(C.SAVE_PATH / name), image)
    cv2.destroyWindows()
    return success


def main_window():
    global SELECTED_FILE, SHOWING_TABLE, THRESHOLD, C
    # File list
    file_list_column = [[
        sg.Text("Data Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(), sg.Button("Refresh", key="-REFRESH-"),
    ],
        [sg.Text(f"Current data directory : {C.DATA_PATH}", key="-DATA_DIR-")],
        [
            sg.Listbox(
                values=[], enable_events=True, size=(40, 20), key="-FILE_LIST-"
            )
        ],

    ]
    button_column = [
        [sg.Button("LIVE VIEW", key="-LIVE_VIEW-")],
        [sg.HorizontalSeparator(color="white")],
        [sg.Button("SINGLE VIEW", key="-SINGLE_VIEW-")],
        [sg.Text("No File selected!", key="-FILE_SELECTED-")],
        [sg.HorizontalSeparator(color="white")],
        [sg.Button("STATISTICS", key="-TABLE-")]
        ]

    # ----- Full layout -----
    layout = [
        [sg.Column(file_list_column), sg.Column(button_column)],
    ]
    return sg.Window('pnCCD CAMP', layout, finalize=True, size=(800, 300), modal=True, resizable=True)


main_win, live_win, table_win, single_win = main_window(), None, None, None
C.create_save_folders()

while True:
    # POSSIBLE EVENT KEYS: -LIVE_VIEW-, -SINGLE_VIEW-, -TABLE- , -SET_SAVEPATH-, -REFRESH-, -FOLDER-, -FILE_LIST-
    window, event, values = sg.read_all_windows()

    #print(window, event, values)
    if window == main_win and event in (sg.WIN_CLOSED, 'Exit'):
        break

    ########## FILE SELECTION ##########
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        fnames, parent, times = select_h5_files(C.DATA_PATH)
        window["-FILE_LIST-"].update([f"{n} | {t}" for n, t in zip(fnames, times)])
        window["-DATA_DIR-"].update(parent)
        C.change_data_path(new_path=parent)
    if event == "-REFRESH-":
        fnames, parent, times = select_h5_files(C.DATA_PATH)
        window["-FILE_LIST-"].update([f"{n} | {t}" for n, t in zip(fnames, times)])
        window["-DATA_DIR-"].update(parent)
    if event == "-FILE_LIST-":
        NEW_FILE = Path(parent) / values["-FILE_LIST-"][0]
        if NEW_FILE != SELECTED_FILE:
            SELECTED_FILE = NEW_FILE
            (f"Selected new file " + str(SELECTED_FILE).rsplit("|")[0])
            window["-FILE_SELECTED-"].update(value=SELECTED_FILE.name)

    ########## LIVE VIEW ##########
    if event == "-LIVE_VIEW-":
        logger.info("opening live view window displaying most recent file.")
        live_win = live_window(const=C)

    ########## SINGLE VIEW ##########
    if event == "-SINGLE_VIEW-" and SELECTED_FILE:
        logger.info(f"opening single view window for {SELECTED_FILE}")
        single_win = single_window(str(SELECTED_FILE).rsplit(" |")[0], const=C)

    ########## STATISTICS ##########
    if event == "-TABLE-":
        logger.info("opening table window.")
        table_win = table(const = C)

window.close()
