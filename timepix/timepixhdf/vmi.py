import numpy as np
from timepixhdf.utils import reproject_image_into_polar
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage


class VmiImage():
    bin_space = 256  # number of pixel

    def __init__(self, x, y, center=[128, 128], angle=0, crop=False,
            radius=bin_space, rect=False, table_path=None):

    # User Input
        self.x = x
        self.y = y
        self.center = center
        self.angle = angle
        self.crop = False
        self.radius = radius
        self.rect = rect
        self.table_path = table_path
        
        self.bins = np.linspace(0, self.bin_space, self.bin_space + 1)
        self.title = 'VMI image'
        self.xlabel = 'x [px]'
        self.ylabel = 'y [px]'
        self.image = self.__create_image()
        self.process = self.__process_image()
        self.invert, self.mexdis = self.__do_meveler()

    def __create_image(self):
        ''' Transpose array results in x = 1st y = 2nd dimension '''
        counts, xbins, ybins = np.histogram2d(self.x, self.y, bins=(self.bins, self.bins))
        vmi_image = np.transpose(counts)
        np.savetxt(f'./Images/vmi_image.dat', vmi_image, delimiter="\t")
        return vmi_image

    def __process_image(self) :
        import subprocess

        image_in = self.image
        cx, cy = self.center

        image_translated = np.zeros(image_in.shape)
        nrow, ncol = image_in.shape

        for irow, row in enumerate(image_in):
            for icol, num in enumerate(row):
                ir = irow + round(nrow / 2) - cy
                ic = icol + round(ncol / 2) - cx
                if (ir >= 0 and ir < nrow) and (ic >= 0 and ic < ncol):
                    image_translated[ir, ic] = num

        image_rotated = ndimage.rotate(image_translated, self.angle)

        if self.crop:
            mask = np.zeros(image_rotated.shape)
            nrow, ncol = image_rotated.shape
            crow = round(nrow / 2)
            ccol = round(ncol / 2)
            for irow, row in enumerate(mask):
                for icol, num in enumerate(row):
                    iradius = ((irow - crow) ** 2 + (icol - ccol) ** 2) ** 0.5
                    if iradius <= self.radius:
                        mask[irow, icol] = 1
            image_out = mask * image_rotated
        else:
            image_out = image_rotated

        np.savetxt(f'./Images/Proc_vmi_image.dat', image_out, delimiter='\t')

        if self.rect:
            max_val = str(math.floor(np.amax(image_out)))
            command = f'wine ./programs/polar Proc_vmi_image -t {self.table_path} -r 170 0.5 0.5 -I 0 {max_val} !'
            subprocess.run(command, shell=True)
            image_out = np.loadtxt(f'./Images/Proc_vmi_image_rect.dat', delimiter='\t')
            subprocess.run('rm ./Images/Proc_vmi_image_rect.dat', shell=True)

        subprocess.run('rm ./Images/Proc_vmi_image.dat', shell=True)

        return image_out

    def __do_meveler(self):
        import subprocess

        np.savetxt('./Images/vmi_image_proc', self.process, delimiter='\t')

        nrow, ncol = self.image.shape
        ix = round(ncol/2)
        iz = round(nrow/2)
        command = f'./programs/F2QC ./Images/vmi_image_proc -IX{ix} -IZ{iz} -M0'
        subprocess.run(command, shell=True)

        command = './programs/Meveler2 DefaultQ.dat'
        subprocess.run(command, shell=True)

        im1 = np.loadtxt('MEXmap.dat', delimiter=',')
        im2 = np.fliplr(im1)
        im3 = np.flipud(im2)
        im4 = np.fliplr(im3)
        mev1 = np.append(im4, im1, axis=0)
        mev2 = np.append(im3, im2, axis=0)
        mev_image = np.append(mev2, mev1, axis=1)

        with open('MEXdis.dat', 'r') as f:
            data = []
            for line in f:
                line = line.replace('D', 'E')
                data.append(line.split())
        mexdis = np.array(data, dtype=float)

        subprocess.run('rm DefaultQ.dat', shell=True)
        subprocess.run('rm MEXini.dat', shell=True)
        subprocess.run('rm MEXdis.dat', shell=True)
        subprocess.run('rm MEXmap.dat', shell=True)
        subprocess.run('rm MEXres.dat', shell=True)
        subprocess.run('rm MEXsim.dat', shell=True)

        np.savetxt('./Images/mev_image.dat', mev_image, delimiter='\t')
        np.savetxt('./Images/mexdis.dat', mexdis, delimiter='\t')

        return mev_image, mexdis

    def __add_labels(self):
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

    def show(self):
        plt.imshow(self.image, origin='lower')
        self.__add_labels()
        plt.colorbar()
        plt.show()

    def show_mev(self):
        plt.imshow(self.invert)
        plt.title("Invert Image")
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.colorbar()
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

