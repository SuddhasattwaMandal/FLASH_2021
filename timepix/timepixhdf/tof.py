import numpy as np
import matplotlib.pyplot as plt
from timepixhdf.utils import hist_to_xy


class TimeAxis:

    def __init__(self, time_axis_in_seconds, new_time_unit):
        units = [None, 'milli', 'micro', 'nano']
        factors = [1, 10 ** 3, 10 ** 6, 10 ** 9]
        plot_units = ['s', 'ms', '\u03BCs', 'ns']
        index = int(np.array([i for i in range(len(units)) if units[i] == new_time_unit]))
        self.array = time_axis_in_seconds * factors[index]
        self.unit = new_time_unit
        self.plot_unit = plot_units[index]


class Tof():

    def __init__(self, tof, time_unit='micro', bins=100):
        time_axis = TimeAxis(tof, time_unit)
        xlabel = 'ToF [{}]'.format(time_axis.plot_unit)
        x, y = hist_to_xy(time_axis.array, bins)
        plt.plot(x, y)
        plt.title('time-of-flight')
        plt.xlabel(xlabel)
        plt.ylabel('number of events')
        plt.show()


class TofvsPos1D():

    def __init__(self, tof, dim, time_unit='micro'):
        time_axis = TimeAxis(tof, time_unit)
        plt.scatter(time_axis.array, dim, s=10)
        plt.title('position vs time-of-flight')
        plt.xlabel('ToF [{}]'.format(time_axis.plot_unit))
        plt.ylabel('position [px]')
        plt.show()


class TofvsPos2D():

    def __init__(self, tof, dim, time_unit='micro', bin_tof=6000, bin_space=256):
        time_axis = TimeAxis(tof, time_unit)
        self.bins = (bin_tof, np.linspace(0, bin_space, bin_space + 1))
        plt.hist2d(time_axis.array, dim, bins=self.bins)
        plt.title('position vs time-of-flight')
        plt.xlabel('ToF [{}]'.format(time_axis.plot_unit))
        plt.ylabel('position [px]')
        plt.show()

