from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from scipy.ndimage import map_coordinates
import numpy as np


def hist_to_xy(array, bins):
    hist = np.histogram(array, bins=bins)
    y, x = [hist[0], 0.5 * (hist[1][1:] + hist[1][:-1])]
    return x, y


def find_nearest(array, values):
    indices = np.abs(np.subtract.outer(array, values)).argmin(0)
    return indices


def check_for_completeness(list):
    start, end = int(list[0]), int(list[-1])
    return sorted(set(range(start, end + 1)).difference(list))


def print_table(table):
    longest_cols = [(max([len(str(row[i])) for row in table]) + 3) for i in range(len(table[0]))]
    row_format = "".join(["{:>" + str(longest_col) + "}" for longest_col in longest_cols])
    for row in table:
        print(row_format.format(*row))


def reproject_image_into_polar(data, origin, Jacobian=False, dr=1, dt=None):
    """
    Reprojects a 2D numpy array (``data``) into a polar coordinate system.
    "origin" is a tuple of (x0, y0) relative to the top-left image corner
    Parameters
    ----------
    data : 2D np.array
    origin : tuple
        The coordinate of the image center, relative to top-left
    Jacobian : boolean
        Include ``r`` intensity scaling in the coordinate transform.
        This should be included to account for the changing pixel size that
        occurs during the transform.
    dr : float
        Radial coordinate spacing for the grid interpolation
        tests show that there is not much point in going below 0.5
    dt : float
        Angular coordinate spacing (in radians)
        if ``dt=None``, dt will be set such that the number of theta values
        is equal to the maximum value between the height or the width of
        the image.
    Returns
    -------
    output : 2D np.array
        The polar image (r, theta)
    r_grid : 2D np.array
        meshgrid of radial coordinates
    theta_grid : 2D np.array
        meshgrid of theta coordinates
    """
    #     data = np.flipud(data) # bottom-left coordinate system requires numpy image to be np.flipud

    ny, nx = data.shape[:2]

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin)  # (x,y) coordinates of each pixel
    r, theta = cart2polar(x, y)  # convert (x,y) -> (r,θ), note θ=0 is vertical

    nr = np.int(np.ceil((r.max() - r.min()) / dr))

    if dt is None:
        nt = max(nx, ny)
    else:
        # dt in radians
        nt = np.int(np.ceil((theta.max() - theta.min()) / dt))

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nr, endpoint=False)
    theta_i = np.linspace(theta.min(), theta.max(), nt, endpoint=False)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    X, Y = polar2cart(r_grid, theta_grid)

    X += origin[0]  # We need to shift the origin
    Y += origin[1]  # back to the bottom-left corner...
    xi, yi = X.flatten(), Y.flatten()
    coords = np.vstack((yi, xi))  # (map_coordinates requires a 2xn array)

    zi = map_coordinates(data, coords)
    output = zi.reshape((nr, nt))

    if Jacobian:
        output = output * r_i[:, np.newaxis]

    return output, r_grid, theta_grid


def index_coords(data, origin):
    """ Creates x & y coords for the indicies in a numpy array """
    ny, nx = data.shape[:2]
    origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(float(nx)), np.arange(float(ny)))
    x -= origin_x
    y -= origin_y
    return x, y


def cart2polar(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(x, y)  # θ referenced to vertical
    return r, theta


def polar2cart(r, theta):
    y = r * np.cos(theta)  # θ referenced to vertical
    x = r * np.sin(theta)
    return x, y
