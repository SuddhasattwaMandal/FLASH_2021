#!/usr/bin/env python
import PySimpleGUI as sg
from utils import save_df_as_json, df_from_path, select_h5_files, df_from_h5
from functools import reduce
from pathlib import Path
import pandas as pd
from constants import Constants

_VARS = {"threshold": (0, 100000)}


def get_table_data_from_dir(const: Constants):
    fnames, parent, times = select_h5_files(const.DATA_PATH)
    dframes = [df_from_h5(Path(parent) / path, _VARS["threshold"]) for path in fnames]
    df = reduce(lambda left, right: pd.concat([left, right]), dframes)
    return df, fnames


def table(const: Constants):
    df, fnames = get_table_data_from_dir(const)
    data = df.values.tolist()
    header_list = df.columns.values.tolist()
    layout = [
        [sg.Button("Save", key="-SAVE_TAB-"), sg.Button("Open", key="-OPEN-")],
        [[sg.Text(text="Thresholds for hitfinding (lower / upper)", pad=((0, 0), (10, 0)), text_color='white'),
          sg.Slider(range=(0, 42000), orientation='h', size=(25, 10),
                    default_value=0,
                    text_color='white',
                    key='-SLIDER_0-',
                    enable_events=True),
          sg.Slider(range=(0, 42000), orientation='h', size=(25, 10),
                    default_value=42000,
                    text_color='white',
                    key='-SLIDER_1-',
                    enable_events=True)],
         sg.Button("Apply", pad=((4, 0), (10, 0)), key="-SET_THRESHOLD-")],
        [sg.Text(f"#images: {df.shape[0]}"), sg.Text(f"#files: {len(fnames)}")],
        [sg.Table(values=data,
                  headings=header_list,
                  display_row_numbers=True,
                  auto_size_columns=True,
                  num_rows=min(25, len(data)),
                  key="-TABLE-")]
    ]

    window = sg.Window('TABLE', layout, size=(1200, 400), grab_anywhere=False, resizable=True, modal=True)
    while True:
        event, values = window.read()
        print(event, values)
        if event == "-SAVE_TAB-":
            save_df_as_json(df, const.SAVE_PATH_TABLE)

        if event == "-OPEN-":
            fp = SAVE_PATH_TABLE
            path = sg.popup_get_file("Select file ", title="Select file to open in table", default_path=fp)
            df = df_from_path(path)
            if df is not None:
                df.drop(columns="index", inplace=True)
                window["-TABLE-"].update(df.values.tolist())
        if event == "-SET_THRESHOLD-":
            _VARS["threshold"] = (values["-SLIDER_0-"], values["-SLIDER_1-"])
            df = get_table_data_from_dir()
            data = df.values.tolist()
            window["-TABLE-"].update(data)
        if event == sg.WIN_CLOSED:
            window.close()
            break
