import numpy as np
import matplotlib.pyplot as plt
import peakutils
from peakutils.plot import plot as pplot
from scipy.optimize import curve_fit
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
        self.xlabel = 'ToF [{}]'.format(time_axis.plot_unit)
        self.x, self.y = hist_to_xy(time_axis.array, bins)
        self.ylabel = "number of event"

    def __add_labels(self) :
        plt.title('time-of-flight')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

    def show(self) :
        plt.plot(self.x, self.y)
        self.__add_labels()
        plt.show()



#    def calibrate_tof(self, thres, dist):
#
#        xdata = self.x
#        ydata = self.y
#
#        indexes = peakutils.indexes(ydata, thres=thres, min_dist=dist,
#                thres_abs=True)
#        peaks_x = peakutils.interpolate(xdata, ydata, ind=indexes)
#        pplot(xdata, ydata, indexes)
#        plt.title("Peaks")
#
#        return indexes, xdata[indexes], peaks_x
#
#
##        def tellme(s):
#            plt.title(s, fontsize=16)
#            plt.draw()
#
#        def Gaussian(x, A, sigma, mu):
#            y = A*(1/(np.sqrt(2*np.pi)*sigma))*np.exp(-0.5*((x-mu)/sigma)**2)
#            return y
#
#        def line(x, a, b):
#            y = a*x + b
#            return y
#
#
#        fig1, ax1 = plt.subplots()
#        ax1.plot(xdata, ydata)
#        npeak = 0
#        ToF_peak = []
#        while True:
#            tellme(f'Peak no.{str(npeak+1)}: Press any key to continue.')
#            keypressed = plt.waitforbuttonpress()
#            if keypressed:
#                tellme('Select both side of the peak')
#                pts = plt.ginput(2)
#                (x0, y0), (x1, y1) = pts
#                xmin, xmax = sorted([x0, x1])
#
#                # gaussian fitting
#                xdata_fit = xdata[(xdata >= xmin) & (xdata <= xmax)]
#                ydata_fit = ydata[(xdata >= xmin) & (xdata <= xmax)]
#                mm = np.amax(ydata_fit)
#                ii = np.argmax(ydata_fit)
#                ff = abs(ydata_fit - mm/2)
#                m1 = np.amin(ff)
#                lim1 = np.argmin(ff)
#                ff[lim1] = ff[lim1]+np.amax(ff)
#                m2 = np.amin(ff)
#                lim2 = np.argmin(ff)
#                sigma0 = abs(xdata_fit[lim1]-xdata_fit[lim2])/(2*np.sqrt(2*np.log(2)))
#                guess = [0.7*mm, sigma0, xdata_fit[ii]]
#                parameters, covariance = curve_fit(Gaussian, xdata_fit, ydata_fit)
#                A = parameters[0]
#                sigma = parameters[1]
#                mu = parameters[2]
#                fit_y = Gaussian(xdata_fit, A, sigma, mu)
#                ax1.plot(xdata, ydata, label='data')
#                ax1.plot(xdata_fit, fit_y, label='fit')
#                npeak = npeak + 1
#                ToF_peak.append(mu)
#                answer = messagebox.askyesno("Question", "Do you want to add mass peak?")
#                if not answer:
#                    tellme('All Done!')
#                    break
#
#        mass_sqrt_peak = []
#        for i in range(npeak):
#            m = simpledialog.askfloat("Question", f"Eneter the correseponding masss for peak no.{str(i+1)}")
#            mass_sqrt_peak.append(np.sqrt(m))
#
#
#        # fitting
#        mass_sqrt = np.array(mass_sqrt_peak)
#        ToF = np.array(ToF_peak)
#        a0 = (ToF[-1]-ToF[0])/(mass_sqrt[-1]-mass_sqrt[0])
#        b0 = ToF[0]-a0*mass_sqrt[0]
#        guess = [a0, b0]
#        print(guess)
#        parameters, covariance = curve_fit(line, mass_sqrt, ToF, p0=guess)
#        a = parameters[0]
#        b = parameters[1]
#        print(parameters)
#        fit_y = line(mass_sqrt, a, b)
#        fig2, ax2 = plt.subplots()
#        ax2.plot(mass_sqrt, ToF, label='data')
#        ax2.plot(mass_sqrt, fit_y, label='fit')
#
#        mass_data = ((xdata-b)/a)**2
#        fig3, ax3 = plt.subplots()
#        ax3.plot(mass_data, ydata)
#        data_to_save = np.array([mass_data, ydata])
#        data_to_save = data_to_save.transpose()
#        np.savetxt(f'mass_cali_coeff.dat', parameters, delimiter=' ')
#
#
#        plt.show()

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

