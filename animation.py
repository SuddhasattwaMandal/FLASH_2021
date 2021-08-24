import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets
import numpy as np
# own modules
from utils import load_timestamp, load_temperature, \
    load_trainID, load_datasetID, load_pnccd_image_data, basic_hitfinding
from matplotlib.widgets import Slider


class Player(FuncAnimation):
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, mini=0, maxi=100, pos=(0.125, 0.92), **kwargs):
        self.i = 0
        self.min = mini
        self.max = maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self, self.fig, self.func, frames=self.play(),
                               init_func=init_func, fargs=fargs,
                               save_count=save_count, **kwargs)

    def play(self):
        while self.runs:
            self.i = self.i + self.forwards - (not self.forwards)
            if self.min < self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        self.runs = True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def close(self, event=None):
        plt.close(self.fig)

    def forward(self, event=None):
        self.forwards = True
        self.start()

    def backward(self, event=None):
        self.forwards = False
        self.start()

    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()

    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i + self.forwards - (not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i += 1
        elif self.i == self.max and not self.forwards:
            self.i -= 1
        self.func(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0], pos[1], 0.22, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        cax = divider.append_axes("right", size="100%", pad=0.05)
        self.button_oneback = matplotlib.widgets.Button(playerax, label=u'$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, label=u'$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, label=u'$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, label=u'$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label=u'$\u29D0$')
        self.button_closefig = matplotlib.widgets.Button(cax, label='X')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.button_closefig.on_clicked(self.close)


def run_animation(PNCCD_DATA_PATH, thresh):
    # add histogram plot
    # add contrast / brightness sliders
    images = load_pnccd_image_data(PNCCD_DATA_PATH)
    trainIDs = load_trainID(PNCCD_DATA_PATH)
    timestamps = load_timestamp(PNCCD_DATA_PATH)
    temperatures = load_temperature(PNCCD_DATA_PATH)
    datasetID = load_datasetID(PNCCD_DATA_PATH)

    # MATPLOTLIB ANIMATION OF PNCCD IMAGES
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 14))

    # create values for table
    table_data = [
        ["DATASET INFORMATION", ""],
        ["Dataset ID", datasetID],
        ["Train ID", trainIDs[0]],
        ["Timestamp [sec]", timestamps[0]],
        ["Temperature [K]", temperatures[0]],
        ["IMAGE STATISTICS", ""],
        [f"Hitscore (threshold: {thresh})", basic_hitfinding(images[0], thresh)],
        ["Mean ", round(np.mean(images[0]), 2)],
        ["Max ", np.max(images[0])],
        ["Min ", np.min(images[0])]
    ]

    # create table
    table = ax2.table(cellText=table_data, loc='center')

    # modify table
    dataset_info_column = (table.get_celld()[0, 0], table.get_celld()[0, 1])
    image_stats_column = (table.get_celld()[5, 0], table.get_celld()[5, 1])
    [entry.set_color("tab:blue") for entry in (dataset_info_column + image_stats_column)]

    table.set_fontsize(16)
    table.scale(1, 3)
    ax2.axis('off')

    # initialize pnccd images
    initial_img = images[0]

    im = ax1.imshow(initial_img, interpolation='none', aspect='auto', cmap="magma")
    cb = fig.colorbar(im, orientation='horizontal', ax=ax1)
    cb.ax.set_ylabel('Intensity [ADU]')

    # intialize histogram
    ax3.hist(initial_img.ravel(), bins="auto", range=(0, 6000), fc='k', ec='k')
    ax4.clear()
    ax4.axis("off")

    def update(i):
        # update pnCCD image
        im = ax1.imshow(images[i], interpolation='none', aspect='auto', cmap="magma")
        ax1.set_title(f" pnCCD Image No. {i} / {len(images)}\n Train ID : {trainIDs[i]}")
        ax1.set_xlabel("px")
        ax1.set_ylabel("px")

        # update table
        table_data = [
            ["DATASET INFORMATION", ""],
            ["Dataset ID", datasetID],
            ["Train ID", trainIDs[i]],
            ["Timestamp [sec]", timestamps[i]],
            ["Temperature [K]", temperatures[i]],
            ["IMAGE STATISTICS", ""],
            [f"Hitscore (threshold: {thresh})", basic_hitfinding(images[i], thresh)],
            ["Mean ", round(np.mean(images[i]), 2)],
            ["Max ", np.max(images[i])],
            ["Min ", np.min(images[i])]
        ]

        # modify table
        ax2.clear()
        ax2.axis('off')
        table = ax2.table(cellText=table_data, loc="center")
        dataset_info_column = (table.get_celld()[0, 0], table.get_celld()[0, 1])
        image_stats_column = (table.get_celld()[5, 0], table.get_celld()[5, 1])
        [entry.set_color("tab:blue") for entry in (dataset_info_column + image_stats_column)]
        table.set_fontsize(16)
        table.scale(1, 3)

        # update histogram
        ax3.hist(images[i].ravel(), bins="auto",
                 range=(0, 6000),
                 fc='k', ec='k')


        ax4.clear()
        ax4.axis("off")

        return [im]

    anim = Player(fig, update, maxi=len(images), pos=(0.19, 0.47))
    plt.show()
