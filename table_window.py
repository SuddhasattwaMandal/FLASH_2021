#!/usr/bin/env python
import PySimpleGUI as sg
from utils import save_df_as_json, df_from_path
from constants import SAVE_PATH_TABLE


def table(df):
    data = df.values.tolist()
    header_list = df.columns.values.tolist()
    layout = [
        [sg.Button("Save", key="-SAVE_TAB-"), sg.Button("Open", key="-OPEN-")],
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
            save_df_as_json(df, SAVE_PATH_TABLE)

        if event == "-OPEN-":
            fp = SAVE_PATH_TABLE
            path = sg.popup_get_file("Select file ", title="Select file to open in table", default_path=fp)
            df = df_from_path(path)
            if df is not None:
                df.drop(columns="index", inplace=True)
                window["-TABLE-"].update(df.values.tolist())

        if event == sg.WIN_CLOSED:
            window.close()
            break
