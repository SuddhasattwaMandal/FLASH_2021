import PySimpleGUI as sg
import os.path
from pathlib import Path
import numpy as np

from utils import select_h5_files
import pandas as pd
from utils import data_from_key, TRAINID_KEY, basic_hitfinding, IMAGE_KEY
from live_window import live_window
import cv2

CWD = Path(os.getcwd())
DATA_PATH = CWD / "data"
SAVE_PATH = None
SELECTED_FILE = None
SHOWING_TABLE = False
THRESHOLD = 0

sg.theme('BluePurple')


def save(train_id: int):
    # 21526319
    all_train_ids = data_from_key(SELECTED_FILE, TRAINID_KEY)
    if not np.isin(train_id, all_train_ids):
        sg.popup_error(f"Train ID {train_id} not in dataset."
                       f"\nValid choices: {min(all_train_ids)} - {max(all_train_ids)}")
        return False
    image = data_from_key(SELECTED_FILE, IMAGE_KEY)[np.where(all_train_ids == train_id)]
    image = image[0, :, :] / np.max(image[0, : , :])
    uint_img = np.array(image*255).astype('uint8')
    grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(grayImage, (450, 450))

    name = f"pnncd_trainID{train_id}.png"
    success = cv2.imwrite(str(SAVE_PATH / name), image)
    cv2.destroyWindows()
    return success


def animation_window():
    layout_anim = [[sg.Text('Animated Matplotlib', size=(40, 1),
                            justification='center')],
                   [sg.pin(sg.Image(key='-IMAGE-'))],
                   [sg.T(size=(50, 1), k='-STATS-')],
                   [sg.B('Animate', focus=True, k='-ANIMATE-')],
                   [sg.Button('Exit', key="-EXIT_ANIM-", size=(10, 1), pad=((280, 0), 3))]]
    return sg.Window("pnCCD images", layout_anim, finalize=True)


def table(filename):
    # in this window general information about the selected .h5 file is shown
    global THRESHOLD
    sg.set_options(auto_size_buttons=True)
    header_list = ["------ Label  ------", "------ Value  ------"]

    if filename is not None:
        train_ids = sorted(data_from_key(filename, TRAINID_KEY))
        first_train, last_train = train_ids[0], train_ids[-1]
        num_images = len(train_ids)
        hits = [basic_hitfinding(image, THRESHOLD) for image in data_from_key(filename, IMAGE_KEY)]
        max_hits = max(hits)
        index = hits.index(max_hits)
    print(f"Showing statistics for: {SELECTED_FILE.name}")
    values = pd.DataFrame([["File name", f"{SELECTED_FILE.name}"],
                           ["Train ID", f"{first_train} -  {last_train}"],
                           ["Number of Images", num_images],
                           ["Max. Hits (Image Num.)", str(max_hits) + f" ({index})"]
                           ]
                          ).values.tolist()

    layout = [[sg.Table(values=values, headings=header_list,
                        auto_size_columns=False, font=("Arial", 16),
                        justification='left', key="-TABLE-",
                        row_height=40, col_widths=[30, 30])],
              [sg.Button('Exit', key="-EXIT_TABLE-", size=(10, 1), pad=((280, 0), 3))]]

    return sg.Window('General Data Statistics', layout, finalize=True, size=(800, 500))


def main_window():
    global DATA_PATH
    global SELECTED_FILE, SHOWING_TABLE, THRESHOLD
    # File list
    file_list_column = [[
        sg.Text("Data Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),        sg.Button("Refresh", key="-REFRESH-"),
    ],
        [sg.Text(f"Current data directory : {DATA_PATH}", key="-DATA_DIR-")],
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
                     [sg.Button("STATISTICS TABLE", key="-TABLE-")],
                     [sg.HorizontalSeparator(color="white")],
                     [sg.Text("Save path : "), sg.Text(SAVE_PATH, key="-SAVE_PATH-"), sg.Button("Set", key="-SET_SAVEPATH-")],
                     ]

    # ----- Full layout -----
    layout = [
        [sg.Column(file_list_column), sg.Column(button_column)],
    ]
    return sg.Window('pnCCD CAMP', layout, finalize=True, size=(900, 400))


main_win, animation_win, table_win = main_window(), None, None

while True:
    # POSSIBLE EVENT KEYS: -LIVE_VIEW-, -SINGLE_VIEW-, -TABLE- , -SET_SAVEPATH-, -REFRESH-, -FOLDER-, -FILE_LIST-
    window, event, values = sg.read_all_windows()

    print(window, event, values)
    if window == main_win and event in (sg.WIN_CLOSED, 'Exit'):
        break

    ########## FILE SELECTION ##########
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        fnames, parent = select_h5_files(DATA_PATH)
        window["-FILE_LIST-"].update(fnames)
        window["-DATA_DIR-"].update(parent)
    elif event == "-REFRESH-":
        fnames, parent = select_h5_files(DATA_PATH)
        window["-FILE_LIST-"].update(fnames)
        window["-DATA_DIR-"].update(parent)
    elif event == "-FILE_LIST-":
        NEW_FILE = Path(parent) / values["-FILE_LIST-"][0]
        if NEW_FILE != SELECTED_FILE:
            SELECTED_FILE = NEW_FILE
            print("Selected new file : ", SELECTED_FILE)
            window["-FILE_SELECTED-"].update(value=SELECTED_FILE.name)
    ########## SET SAVEPATH ##########
    elif event == "-SET_SAVEPATH-":
        save_folder = sg.popup_get_folder("Select folder for saving",
                                          title="Select save folder", default_path=CWD)
        SAVE_PATH = Path(save_folder)
        window["-SAVE_PATH-"].update(".../" + SAVE_PATH.name)
    ########## LIVE VIEW ##########
    elif event == "-LIVE_VIEW-":
        live_window()


    ########## SINGLE VIEW ##########


    ########## STATISTICS ##########

window.close()
