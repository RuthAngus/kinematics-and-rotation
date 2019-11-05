"Running mean sigma clipping."

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import scipy.interpolate as spi


def spline_sigma_clip(x, y, s, sigma=3):
    """
    Sigma clipping using a spline.

    Args:
        x (array): The x array.
        y (array): The y array.
        nsigma (float): The number of sigma to clip on.

    Returns:
        mask (array): The mask for clipping data.

    """

    spl = spi.UnivariateSpline(x, y, s=s)

    y_clipped, mask = sigma_clip(y - spl(x), sigma)

    m = np.ones(len(x)) == 1  # Create Bool array
    newx = x - spi.UnivariateSpline(x, y, s=s)
    oldm = np.array([False])  # Old mask
    i = 0
    while sum(oldm) != sum(m):  # Until no new outliers
        oldm = m*1
        sigma = np.std(newx)    # Calculate sigma
        m &= np.abs(np.median(newx) - x)/sigma < nsigma  # find outliers
        # m &= m
        newx = x[m]
        i += 1

    return mask, spl


def running_sigma_clip(N, x, y, nsigma=3):
    """
    Sigma clipping using a running mean.

    Args:
        N (float): number of points to bin on.
        x (array): The x array.
        y (array): The y array.
        nsigma (float): The number of sigma to clip on.

    Returns:
        x_clipped (array): The clipped data.
        mask (array): The mask for clipping data.

    """

    inds = np.arange(len(x))
    clip_inds = []
    for i in trange(len(x)):

        # Simple sigma clip
        newy, clip = sigma_clip(y[i:N+i], nsigma=nsigma)

        # Add the clipped indices to an array
        clip_inds.append(inds[i:N+i][clip])

    # Convert to flat list.
    full_inds = np.array([i for j in clip_inds for i in j])

    # Remove duplicates
    full_inds = np.unique(full_inds)

    # Convert short index list to full mask.
    mask = np.zeros(len(x), dtype=bool)
    for ind in full_inds:
        mask[ind] = True
    return mask


def interval_sigma_clip(interval, x, y, nsigma=3):
    """
    Sigma clipping using an interval mean.

    Args:
        interval (float): Interval to bin on.
        x (array): The x array.
        y (array): The y array.
        nsigma (float): The number of sigma to clip on.

    Returns:
        mask (array): The mask for clipping data.
        intervals (array): The edges of the intervals used

    """

    nints = int((max(x) - min(x)) / interval)
    # intervals =

    full_mask = []
    n = 0
    for i in trange(nints):
        m = (min(x) + i*interval <= x) * (x < min(x) + (i + 1)*interval)
        newy, clip = sigma_clip(y[m], nsigma=nsigma)
        full_mask.append(clip)
        n += sum(m)

    if len(full_mask) < len(x):
        m = (min(x) + (i + 1)*interval <= x)
        newy, clip = sigma_clip(y[m], nsigma=nsigma)
        full_mask.append(clip)
        n += sum(m)

    full_mask = np.array([i for j in full_mask for i in j])
    return full_mask, intervals


def sigma_clip(x, nsigma=3):
    """
    Sigma clipping for 1D data.

    Args:
        x (array): The data array. Assumed to be Gaussian in 1D.
        nsigma (float): The number of sigma to clip on.

    Returns:
        newx (array): The clipped x array.
        mask (array): The mask used for clipping.
    """

    m = np.ones(len(x)) == 1
    newx = x*1
    oldm = np.array([False])
    i = 0
    while sum(oldm) != sum(m):
        oldm = m*1
        sigma = np.std(newx)
        m &= np.abs(np.median(newx) - x)/sigma < nsigma
        # m &= m
        newx = x[m]
        i += 1
    return x[m], m
