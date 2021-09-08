import PySimpleGUI as sg
import numpy as np
from utils import load_pnccd_image_data, load_trainID, load_datasetID, integrated_intensity, basic_hitfinding
from constants import SAVE_PATH_IMAGES
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, CenteredNorm, Normalize
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import cv2

# VARS CONSTS:
_VARS = {'window': False,
         'fig_agg': False,
         'pltFig': False,
         'threshold': (0, 30000),
         'contrast': 1,
         'brightness': 0,
         'index': 0,
         'scale': "-LOG-",
         'num': 0,
         'roi': None,
         'roi_enabled': False,
         'square': False}

plt.style.use('Solarize_Light2')
plt.rcParams["image.cmap"] = "magma"
plt.rcParams['axes.grid'] = False

sg.theme('black')


# Helper Functions


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def save_figure(figure, h5_fp):
    train_ID = load_trainID(h5_fp)[_VARS['index']]
    dataset_ID = load_datasetID(h5_fp)
    name = f"single_ROI_{dataset_ID}_{train_ID}_{_VARS['index']}.png" if _VARS[
        "roi_enabled"] else f"single_{dataset_ID}_{train_ID}_{_VARS['index']}.png"
    figure.savefig(SAVE_PATH_IMAGES / name)


def get_image_data(h5_fp, threshold=0):
    images = load_pnccd_image_data(h5_fp, threshold)
    _VARS["num"] = len(images)
    image = images[_VARS["index"]]
    return image


def drawPlot(h5_fp):
    img_ids, intensities, index = integrated_intensity(h5_fp, _VARS["threshold"])
    train_ID = load_trainID(h5_fp)[_VARS['index']]
    dataset_ID = load_datasetID(h5_fp)
    intensity = intensities[_VARS['index']]

    cm_scale = LogNorm() if _VARS["scale"] == "-LOG-" else CenteredNorm() \
        if _VARS["scale"] == "-AUTO-" else Normalize() if _VARS["scale"] == "-LIN-" else None

    _VARS['pltFig'] = plt.figure(figsize=(8, 14))
    img_ax = _VARS['pltFig'].add_subplot(211)
    hist_ax = _VARS['pltFig'].add_subplot(212)

    img = get_image_data(h5_fp, _VARS["threshold"]) if not _VARS['roi_enabled'] else _VARS["roi"]
    img = np.square(img) if _VARS["square"] else img
    if _VARS['roi_enabled'] and _VARS["threshold"] != None:
        img = np.asarray(np.clip(img, _VARS["threshold"][0], _VARS["threshold"][1]))
        img = cv2.addWeighted(img, _VARS["contrast"], img, 0, _VARS["brightness"])

    hits = basic_hitfinding(img)
    img_fig = img_ax.imshow(img, norm=cm_scale, aspect='auto')
    img_ax.set_title(
        f"DatasetID: {dataset_ID}\nTrain ID: {train_ID} Index: {_VARS['index']}\nCuml. Intensity: {intensity} Hits: {hits}")
    img_ax.set_facecolor("white")
    divider = make_axes_locatable(img_ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    _VARS['pltFig'].colorbar(img_fig, cax=cax, orientation="vertical")

    hist_ax.hist(img.ravel(), bins="auto", range=(1, np.max(img)), fc='k', ec='k')
    hist_ax.set_facecolor("white")
    hist_ax.grid(True)
    if _VARS["threshold"] != 0:
        img_inset = get_image_data(h5_fp, None) if not _VARS['roi_enabled'] else _VARS["roi"]
        img_inset = np.square(img_inset) if _VARS["square"] else img_inset
        #img_inset = cv2.addWeighted(img_inset, _VARS["contrast"], img, 0, _VARS["brightness"])
        ax_ins = inset_axes(hist_ax, width="40%", height="40%", loc='upper right')
        ax_ins.hist(img_inset.ravel(), bins="auto", range=(1, np.max(img_inset)), fc='k', ec='k')
        ax_ins.axvline(x=_VARS["threshold"][0], linewidth=2, color="black", linestyle="-.", alpha=0.8, ymin=0)
        ax_ins.axvline(x=_VARS["threshold"][1], linewidth=2, color="black", linestyle="-.", alpha=0.8, ymin=0)
        ax_ins.grid(True)
        ax_ins.set_facecolor('white')

    _VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])


def nextImage(h5_fp):
    _VARS['fig_agg'].get_tk_widget().forget()
    _VARS["index"] += 1
    if _VARS["index"] > _VARS["num"]:
        _VARS["index"] = 0
    plt.clf()
    drawPlot(h5_fp)


def prevImage(h5_fp):
    _VARS['fig_agg'].get_tk_widget().forget()
    _VARS["index"] -= 1
    if _VARS["index"] < 0:
        _VARS["index"] = _VARS["num"] - 1
    plt.clf()
    drawPlot(h5_fp)


def update(h5_fp):
    _VARS['fig_agg'].get_tk_widget().forget()
    plt.clf()
    drawPlot(h5_fp)


def single_window(h5_fp):
    # New layout with slider and padding
    layout = [
        [
            sg.InputText(default_text=" ", background_color="white", enable_events=True, s=(4, 4),
                         text_color="black", pad=((4, 0), (10, 0)), key="-ID_IN-"),
            sg.Button("Jump to Image ID ", key="-SET_ID-", pad=((4, 0), (10, 0))),
            sg.Button('Prev', pad=((4, 0), (10, 0))), sg.Button('Next', pad=((4, 0), (10, 0))),
            sg.Button("Save", pad=((4, 0), (10, 0)))],
        [sg.HorizontalSeparator(color="white")],
        [sg.Button("Draw ROI", key="-ROI-"), sg.Checkbox("ROI visible", key="-ROI_VIS-", enable_events=True)],
        [sg.Checkbox("Square pixels", key="-SQUARE-", enable_events=True)],
        [sg.HorizontalSeparator(color="white")],
        [sg.Text(text="Thresholds (lower / upper)", pad=((0, 0), (10, 0)), text_color='white'),
         sg.Slider(range=(0, 42000), orientation='h', size=(25, 10),
                   default_value=0,
                   text_color='white',
                   key='-SLIDER_0-',
                   enable_events=True), sg.Slider(range=(0, 42000), orientation='h', size=(25, 10),
                                                  default_value=42000,
                                                  text_color='white',
                                                  key='-SLIDER_1-',
                                                  enable_events=True)],
        [sg.Text(text="Contrast", pad=((0, 0), (10, 0)), text_color='white'),
         sg.Slider(range=(1, 42), orientation='h', size=(50, 10),
                   default_value=1,
                   text_color='white',
                   key='-SLIDER_CONTRAST-',
                   enable_events=True)],
        [sg.Text(text="Brightness", pad=((0, 0), (10, 0)), text_color='white'),
         sg.Slider(range=(-2000, 12000), orientation='h', size=(50, 10),
                   default_value=0,
                   text_color='white',
                   key='-SLIDER_BRIGHT-',
                   enable_events=True)],
        [sg.Button("Apply", pad=((4, 0), (10, 0)), key="-SET_THRESHOLD-")],
        [sg.HorizontalSeparator(color="white")],
        [sg.Text("Images colorbar")],
        [sg.Checkbox("log", key="-LOG-", enable_events=True),
         sg.Checkbox("linear", key="-LIN-", enable_events=True),
         sg.Checkbox("centered", key="-AUTO-", enable_events=True)],
        [sg.Canvas(key='figCanvas')]
    ]
    _VARS['window'] = sg.Window('SINGLE VIEW', layout, finalize=True, resizable=True, location=(100, 100),
                                element_justification="center", modal=True)

    # MAIN LOOP
    drawPlot(h5_fp)
    _VARS['window'][_VARS["scale"]].update(True)
    scale_events = ["-LOG-", "-LIN-", "-AUTO-"]
    while True:
        event, values = _VARS['window'].read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event in scale_events:
            [_VARS['window'][scale].update(False) if event != scale else _VARS['window'][scale].update(True) for scale
             in scale_events]
            _VARS["scale"] = event
            update(h5_fp)
        elif event == 'Next':
            nextImage(h5_fp)
        elif event == 'Prev':
            prevImage(h5_fp)
        elif event == "-SET_ID-":
            id_in = int(values["-ID_IN-"])
            if id_in > _VARS["num"] or id_in < 0:
                warnings.warn(f"invalid ID: valid IDs range: 0 - {_VARS['num']}")
                _VARS["index"] = _VARS["num"]
            else:
                _VARS["index"] = id_in
            update(h5_fp)
        elif event == '-SET_THRESHOLD-':
            _VARS["threshold"] = (int(values["-SLIDER_0-"]), int(values["-SLIDER_1-"]))
            update(h5_fp)
        elif event == '-SLIDER_CONTRAST-':
            _VARS['contrast'] = values['-SLIDER_CONTRAST-']
        elif event == "-SLIDER_BRIGHT-":
            _VARS['brightness'] = values['-SLIDER_BRIGHT-']
        elif event == "-ROI-":
            img = get_image_data(h5_fp, _VARS["threshold"])
            cv2.namedWindow("Draw ROI", 4)
            roi = cv2.selectROI("Draw ROI", img, False, False)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(30)
            roi_cropped = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
            _VARS["roi"] = roi_cropped
        elif event == "-ROI_VIS-":
            _VARS["roi_enabled"] = True if not _VARS["roi_enabled"] else False
            update(h5_fp)
        elif event == "-SQUARE-":
            _VARS["square"] = True if not _VARS["square"] else False
            update(h5_fp)
        elif event == "Save":
            save_figure(_VARS['pltFig'], h5_fp)

    _VARS['window'].close()
