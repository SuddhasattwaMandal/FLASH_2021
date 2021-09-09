import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, CenteredNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from logging import info, warning

from utils import select_recent_h5_file, integrated_intensity, N_brightest_images, load_datasetID
from constants import Constants

plt.style.use('Solarize_Light2')
sg.theme('black')


def get_plots(scale: str, bar_scale: str, thresholds, const: Constants):
    ###### INITAL DATA LOADING ########
    file_path, time, size = select_recent_h5_file(const.DATA_PATH)
    img_ids, intensities, index = integrated_intensity(file_path, thresholds)
    max_images, max_intensity, max_index, max_train_id = N_brightest_images(file_path, 4, threshold=thresholds)

    # GLobal plot parameters
    plt.rcParams['axes.grid'] = False
    plt.rcParams["image.cmap"] = "magma"

    cm_scale = LogNorm() if scale == "-LOG-" else CenteredNorm() \
        if scale == "-AUTO-" else Normalize() if scale == "-LIN-" else None

    ####### DRAW FIGURES ##########
    main_fig = plt.figure(0, figsize=(6, 6))
    main_ax = main_fig.add_subplot(111)
    main_img = main_ax.imshow(max_images[-1], aspect="auto", norm=cm_scale)
    main_ax.set_title(f"Train ID: {max_train_id[-1]} Index: {max_index[-1]}\nIntgr. Intensity: {max_intensity[-1]}")
    divider = make_axes_locatable(main_ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    main_fig.colorbar(main_img, cax=cax, orientation="vertical")
    #main_ax.set_axis_off()
    main_ax.set_facecolor("white")
    main_ax.set(xticklabels=[]); main_ax.set(yticklabels=[])


    sub_one_fig = plt.figure(1, figsize=(3, 3))
    sub_one_ax = sub_one_fig.add_subplot(111)
    sub_one_img = sub_one_ax.imshow(max_images[2], norm=cm_scale)
    sub_one_ax.set_title(f"Train ID: {max_train_id[2]} Index: {max_index[2]}\nIntgr. Intensity: {max_intensity[2]}",
                         fontsize=8)
    sub_one_ax.set_facecolor("white")
    divider = make_axes_locatable(sub_one_ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    sub_one_fig.colorbar(sub_one_img, cax=cax, orientation="horizontal")
    #sub_one_ax.set_axis_off()
    sub_one_ax.set(xticklabels=[]); sub_one_ax.set(yticklabels=[])


    sub_two_fig = plt.figure(2, figsize=(3, 3))
    sub_two_ax = sub_two_fig.add_subplot(111)
    sub_two_img = sub_two_ax.imshow(max_images[1], norm=cm_scale)
    sub_two_ax.set_title(f"Train ID: {max_train_id[1]} Index: {max_index[1]}\nIntgr. Intensity: {max_intensity[1]}",
                         fontsize=8)
    sub_two_ax.set_facecolor("white")
    divider = make_axes_locatable(sub_two_ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    sub_two_fig.colorbar(sub_two_img, cax=cax, orientation="horizontal")
    #sub_two_ax.set_axis_off()
    sub_two_ax.set(xticklabels=[]); sub_two_ax.set(yticklabels=[])


    sub_three_fig = plt.figure(3, figsize=(3, 3))
    sub_three_ax = sub_three_fig.add_subplot(111)
    sub_three_img = sub_three_ax.imshow(max_images[0], norm=cm_scale)
    sub_three_ax.set_title(f"Train ID: {max_train_id[0]} Index: {max_index[0]}\nIntgr. Intensity: {max_intensity[0]}",
                           fontsize=8)
    sub_three_ax.set_facecolor("white")
    divider = make_axes_locatable(sub_three_ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    sub_three_fig.colorbar(sub_three_img, cax=cax, orientation="horizontal")
    #sub_three_ax.set_axis_off()
    sub_three_ax.set(xticklabels=[]); sub_three_ax.set(yticklabels=[])


    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    intgr_fig = plt.figure(4, figsize=(6, 6))
    intgr_ax = intgr_fig.add_subplot(211)
    intgr_ax.set_title("Integrated Intensities", fontsize=9)
    intgr_ax.set_facecolor("white")
    bar_log = True if bar_scale == "-BAR_LOG-" else False
    intgr_ax.bar(index, intensities, width=0.3, align="center", alpha=0.9, log=bar_log)
    #intgr_ax.set_xlabel("Index")
    intgr_ax.set_ylabel("Integrated Intensity")
    intgr_ax.set_xlabel("Image index")

    # load without threshold for histogram
    max_images_wh_thresh, _, _, _ = N_brightest_images(file_path, 4, threshold=None)
    main_image_wh_thresh = max_images_wh_thresh[-1]

    hist_ax = intgr_fig.add_subplot(212)
    hist_ax.set_title("Intensity Histogram Of Main Image", fontsize=9)
    hist_ax.hist(main_image_wh_thresh.ravel(), bins="auto", range=(1, np.max(main_image_wh_thresh)), fc='k', ec='k')
    hist_ax.axvline(x=thresholds[0], linewidth=1.3, color="black", linestyle="-.", alpha=0.8, ymin=0)
    hist_ax.axvline(x=thresholds[1], linewidth=1.3, color="black", linestyle="-.", alpha=0.8, ymin=0)
    hist_ax.set_facecolor("white")
    hist_ax.grid(True)
    hist_ax.set_ylabel("counts")
    hist_ax.set_xlabel("px. value")


    return main_fig, sub_one_fig, sub_two_fig, sub_three_fig, intgr_fig


def draw_figure(canvas, figure):
    info("drawing plots")
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def destroy(canvas_list):
    info("destroying canvas plots.")
    [c.get_tk_widget().destroy() for c in canvas_list]
    plt.close("all")
    return


PLOTS_DRAWN = False


def live_window(const: Constants):
    global PLOTS_DRAWN
    scale_events = ["-LOG-", "-LIN-", "-AUTO-"]
    bar_events = ["-BAR_LOG-", "-BAR_LIN-"]
    scale = "-LOG-"
    bar_scale = "-BAR_LIN-"
    thresholds = (0, 30000)

    ####### LAYOUT ##########
    left_col = [[sg.Canvas(key="-MAIN_PLOT-")]]
    middle_col = [[sg.Canvas(key="-INTENSITY_PLOT-")]]
    right_col = [
        [sg.Text("Bar plot scale")],
        [sg.Checkbox("log", key="-BAR_LOG-", enable_events=True),
         sg.Checkbox("linear", key="-BAR_LIN-", enable_events=True)
         ],
        [sg.HorizontalSeparator(color="white")],
        [sg.Text("Images colorbar")],
        [sg.Checkbox("log", key="-LOG-", enable_events=True),
         sg.Checkbox("linear", key="-LIN-", enable_events=True),
         sg.Checkbox("auto", key="-AUTO-", enable_events=True)],
        [sg.HorizontalSeparator(color="white")],
        [sg.Text(f"Threshold (lower / upper): {round(thresholds[0], 0)}, {round(thresholds[1], 0)}", key="-THRESHOLD_TXT-")],
        [sg.Button("Set", key="-THRESHOLD-")],
        [sg.Slider(range=(0, 30000), orientation='h', size=(34, 20),
                  default_value=66,
                  background_color='#FDF6E3',
                  text_color='Black',
                  key='-SLIDER-',
                  enable_events=False)],
        [sg.Slider(range=(0, 30000), orientation='h', size=(34, 20),
                   default_value=6666,
                   background_color='#FDF6E3',
                   text_color='Black',
                   key='-SLIDER_THRESH_UPPER-',
                   enable_events=False)],
        [sg.HorizontalSeparator(color="white")],
        [sg.Button("Reset", key="-REFRESH-"), sg.Button("Save", key="-SAVE-")]

    ]
    bottom_row = [[sg.Canvas(key="-SUB_ONE-"), sg.Canvas(key="-SUB_TWO-"), sg.Canvas(key="-SUB_THREE-")]]

    layout = [[sg.T("Recent File: "), sg.Text('NONE', key="-SELECTED_FILE-")],
              [sg.Column(left_col), sg.Column(middle_col)],
              [sg.Column(bottom_row), sg.Column(right_col)]]

    window = sg.Window('LIVE VIEW', layout, finalize=True, size=(1300, 1000), resizable=True, modal=True)

    # Global plot parameters
    plt.rcParams['axes.grid'] = False
    plt.rcParams["image.cmap"] = "magma"

    file_path, time, size = select_recent_h5_file(const.DATA_PATH)
    window["-SELECTED_FILE-"].update(f"{file_path}  {time}")
    window["-LOG-"].update(True)
    window["-BAR_LIN-"].update(True)

    ####### DRAW FIGURES ##########

    window_keys = [window["-MAIN_PLOT-"], window["-SUB_ONE-"], window["-SUB_TWO-"], window["-SUB_THREE-"],
                   window["-INTENSITY_PLOT-"]]
    figures = get_plots(scale, bar_scale, thresholds, const)
    canvas = [draw_figure(window_keys[i].TKCanvas, figures[i]) for i in range(0, len(figures))]

    while True:
        event, values = window.read()
        # print(event, values)
        file_path, time, size = select_recent_h5_file(const.DATA_PATH)
        window["-SELECTED_FILE-"].update(f"{file_path}  {time}")
        if event in scale_events:
            [window[scale].update(False) if event != scale else window[scale].update(True) for scale in scale_events]
            scale = event
            PLOTS_DRAWN = False
        if event in bar_events:
            [window[scale].update(False) if event != scale else window[scale].update(True) for scale in bar_events]
            bar_scale = event
            PLOTS_DRAWN = False
        if event == 'Exit' or event == sg.WIN_CLOSED:
            destroy(canvas)
            plt.close("all")
            break
        if event == "-REFRESH-":
            destroy(canvas)
            thresholds = [0, 30000]
            window["-THRESHOLD_TXT-"].update(f"Value: {round(thresholds[0], 0)}, {round(thresholds[1], 0)}")
            PLOTS_DRAWN = False
        if event == "-SLIDER-" or event == "-SLIDER_THRESH_UPPER-":
            thresholds = (values["-SLIDER-"], values['-SLIDER_THRESH_UPPER-'])
            window["-THRESHOLD_TXT-"].update(f"Value: {round(thresholds[0], 0)}, {round(thresholds[1], 0)}")
        if event == "-THRESHOLD-":
            thresholds = (values["-SLIDER-"], values['-SLIDER_THRESH_UPPER-'])
            PLOTS_DRAWN = False
        if event == "-SAVE-":
            dataset_ID = load_datasetID(file_path)
            save_dir = const.SAVE_PATH_IMAGES / f"live_view_{dataset_ID}"
            if not save_dir.is_dir():
                save_dir.mkdir()
            num = 0
            for _ in save_dir.iterdir():
                num += 1
            [c.print_png(save_dir / str(f"plot_{i}.png")) for c, i in zip(canvas,range(num, num+len(canvas)))]

        if not PLOTS_DRAWN:
            destroy(canvas)
            figures = get_plots(scale, bar_scale, thresholds, const)
            canvas = [draw_figure(window_keys[i].TKCanvas, figures[i]) for i in range(0, len(figures))]
            PLOTS_DRAWN = True

    window.close()


if __name__ == "__main__":
    live_window(Constants)
