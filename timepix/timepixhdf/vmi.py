import numpy as np
from timepixhdf.utils import reproject_image_into_polar
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class VmiImage():
    bin_space = 256  # number of pixel

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.bins = np.linspace(0, self.bin_space, self.bin_space + 1)
        self.title = 'VMI image'
        self.xlabel = 'x [px]'
        self.ylabel = 'y [px]'
        self.image = self.__create_image()

    def __create_image(self):
        ''' Transpose array results in x = 1st y = 2nd dimension '''
        counts, xbins, ybins = np.histogram2d(self.x, self.y, bins=(self.bins, self.bins))
        return np.transpose(counts)

    def __add_labels(self):
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

    def show(self):
        plt.imshow(self.image, origin='lower')
        self.__add_labels()
        plt.show()

    def zoom_in(self, x_start, x_end, y_start, y_end):
        zoom_image = np.transpose(np.transpose(self.image)[x_start:x_end, y_start: y_end])
        plt.imshow(zoom_image, extent=[x_start, x_end, y_start, y_end], origin='lower')
        self.__add_labels()
        plt.show()

    def __create_cart_image(self, image, center, angles=None, radii=None):
        plt.figure()
        plt.imshow(image, origin='lower')
        plt.plot(center[0], center[1], 'ro')
        self.__add_labels()
        if angles or radii:
            if not angles:
                angles = (0, 360)
            if not radii:
                radii = (0, 1000)
                # patches have diff coordinates and direction
            plt.gcf().gca().add_patch(
                patches.Wedge((center[0], center[1]), radii[1], (270 - angles[1]), (270 - angles[0]),
                              color="r", alpha=0.2, width=radii[1] - radii[0]))

    def __create_polar_image(self, image_polar, angles=None, radii=None):
        plt.figure()
        plt.imshow(image_polar, origin='lower')
        plt.xlabel('angle [°]')
        plt.ylabel('radius [px]')
        plt.title('polar coordinates')
        if angles or radii:
            if not angles:
                size = (0, image_polar.shape[1])
            if not radii:
                radii = (0, image_polar.shape[0])
            origin = (angles[0], radii[0])
            height = radii[1] - radii[0]
            if angles[0] < 0:
                origin2 = (360 + angles[0], radii[0])
                plt.gcf().gca().add_patch(patches.Rectangle(origin2, -angles[0], height, color='r', alpha=0.2))
                angles = (0, angles[1])
            size = angles[1] - angles[0]
            origin = (angles[0], radii[0])
            plt.gcf().gca().add_patch(patches.Rectangle(origin, size, height, color='r', alpha=0.2))

    def __extract_profileline(self, image_polar, angles=None, radii=None):
        if not angles:
            angles = (0, image_polar.shape[1])
        if not radii:
            radii = (0, image_polar.shape[0])
        if angles[0] >= 0:
            radial_average = np.average(image_polar[radii[0]:radii[1], angles[0]:angles[1]], axis=1)
        else:
            radial_average_1 = np.average(image_polar[radii[0]:radii[1], 0:angles[1]], axis=1)
            radial_average_2 = np.average(image_polar[radii[0]:radii[1], angles[0]:-1], axis=1)
            radial_average = np.average(np.array([radial_average_1, radial_average_2]),
                                        axis=0, weights=np.array([angles[1], -angles[0]]))
        pixel_from_center = np.arange(radii[0], radii[1])
        assert len(pixel_from_center) == len(radial_average)
        return (pixel_from_center, radial_average)

    def plot_profileline(self, radial_average):
        plt.figure()
        plt.plot(radial_average[0], radial_average[1])
        plt.xlabel('radius [px]')
        plt.ylabel('profile line [a.u.]')
        plt.title('radial average')

    def create_radial_average(self, center, angles=None, radii=None):
        self.__create_cart_image(self.image, center, angles, radii)
        image_polar, _, _ = reproject_image_into_polar(self.image, origin=center, dt=np.pi / 180)
        self.__create_polar_image(image_polar, angles, radii)
        radial_average = self.__extract_profileline(image_polar, angles, radii)
        self.plot_profileline(radial_average)
        plt.show()
        return radial_average
