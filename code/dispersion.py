import numpy as np
from tqdm import trange
import scipy.stats as sps
import astropy.stats as aps
import matplotlib.pyplot as plt


def MC_dispersion(x, y, xerr, yerr, bins, nsamps, method="std"):
    """
    Calculate the dispersion in a set of bins, with Monte Carlo uncertainties.

    Args:
        x (array): The x-values.
        y (array): The y-values.
        xerr (array): The x-value uncertainties.
        yerr (array): The y-value uncertainties.
        bins (array): The bin edges. length = number of bins + 1
        nsamps (int): Number of Monte Carlo samples.
        method (str): The method to use. Either "std" for standard deviation
            or "mad" for median absolute deviation.

    Returns:
        dispersion (array): The dispersion array. length = number of bins.
        dispersion_err (array): The uncertainty on the dispersion.
            length = number of bins.
        kurtosis (array): The kurtosis array. length = number of bins.
        kurtosis_err (array): The uncertainty on the kurtosis.
            length = number of bins.

    """
    nbins = len(bins) - 1

    # Sample from Gaussians
    xsamps = np.zeros((len(x), nsamps))
    ysamps = np.zeros((len(y), nsamps))
    dsamps, ksamps = [np.zeros((nbins, nsamps)) for i in range(2)]

    for i in trange(nsamps):
        xsamps[:, i] = np.random.randn(len(x))*xerr + x
        ysamps[:, i] = np.random.randn(len(y))*yerr + y
        d, k = dispersion(xsamps[:, i], ysamps[:, i], bins, method=method)
        dsamps[:, i] = d
        ksamps[:, i] = k

    dispersion_err = np.std(dsamps, axis=1)
    kurtosis_err = np.std(ksamps, axis=1)
    return np.mean(dsamps, axis=1), dispersion_err, np.mean(ksamps, axis=1), \
        kurtosis_err


def dispersion(x, y, bins, method):
    """
    Calculate the dispersion in a set of bins.

    Args:
        x (array): The x-values.
        y (array): The y-values.
        bins (array): The bin edges. length = number of bins + 1
        method (str): The method to use. Either "std" for standard deviation
            or "mad" for median absolute deviation.

    Returns:
        dispersion (array): The dispersion array. length = number of bins.

    """

    d, k = [np.zeros(len(bins)-1) for i in range(2)]
    for i in range(len(bins)-1):
        m = (bins[i] < x) * (x < bins[i+1])
        if method == "std":
            d[i] = np.std(y[m])
        if method == "mad":
            d[i] = 1.5*np.median(np.abs(y[m] - np.median(y[m])))

        # Calculate kurtosis
        k[i] = sps.kurtosis(y[m])

    return d, k


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
        m &= m
        newx = x[m]
        i += 1
    print("niter = ", i, len(x) - sum(m), "stars removed", "kurtosis = ",
          sps.kurtosis(x[m]))
    return x[m], m

    # _m = np.ones(len(x))
    # m = _m == 1
    # newx = x*1
    # for i in trange(10):
    #     sigma = np.std(newx)
    #     oldm = m*1
    #     m &= np.abs(np.median(newx) - x)/sigma < nsigma
    #     if sum(oldm) != sum(m):
    #         m &= m
    #         newx = x[m]
    # return x[m], m


def running_dispersion(x, y, bsize, mad=False):

    assert x[0] == np.sort(x)[0], "arrays must be sorted on x."

    # Calculate running std of points in bin of bsize
    d, newx = [], []
    for i in range(len(x)):
        if i+bsize < len(x):
            if not mad:
                d.append(np.std(y[i:i + bsize]))
            else:
                d.append(np.median(np.abs(y[i:i + bsize])))
            newx.append(x[i])

    return np.array(newx), np.array(d)


def binned_dispersion(x, y, nbins, method="rms"):

    d, N, mean = [np.zeros(nbins) for i in range(3)]
    bin_width = (max(x) - min(x))/nbins
    left_bin_edges = np.array([min(x) + i*bin_width for i in range(nbins)])
    right_bin_edges = left_bin_edges + bin_width
    mid_bin = left_bin_edges + .5*bin_width

    for i in range(nbins):
        m = (left_bin_edges[i] < x) * (x < right_bin_edges[i])
        mean[i] = np.mean(y[m])
        N[i] = sum(m)
        if method == "std":
            d[i] = np.std(y[m])
        elif method == "mad":
            d[i] = np.median(np.abs(y[m]))
            # d[i] = np.sqrt(np.median(y[m]))
        elif method == "rms":
            d[i] = np.sqrt(np.mean((y[m])**2))
    return mid_bin, d, d/np.sqrt(N), mean


def select_stars(df, bins, column_name):
    """
    Select groups of stars, based on bins.

    Args:
        df (pandas.DataFrame): a pandas dataframe.
        bins (array): The list or array of bin edges.
        column_name (str): The name of the column to cut on.

    Returns:
        ms (list): a list of masks to select stars with.

    """

    ms = []
    for i in range(len(bins)-1):
        m = (df["{}".format(column_name)] > bins[i]) * \
            (df["{}".format(column_name)] < bins[i+1])
        ms.append(m)

    return ms


def calc_dispersion_and_dispersion_err(v, verr, nsamples):
    """
    Calculate velocity dispersion and uncertainty on the dispersion,
    given velocities and uncertainties.
    This version uses broadcasting to be much faster than MC_dispersion,
    defined above.

    Args:
        v (array): The velocity array.
        verr (array): The array of velocity uncertainties.
        nsamples (int): The number of Monte Carlo samples to draw.

    Returns:
        dispersion (float): The standard deviation of the velocities.
        dispersion_err (float): The Monte Carlo uncertainty on the velocity
            dispersion.

    """

    # Calculate velocity samples
    v_samples = np.random.randn((len(v)), nsamples)*verr[:, np.newaxis] \
        + v[:, np.newaxis]

    # Calculate velocity dispersion samples
    dispersion_samples = np.std(v_samples, axis=0)
    dispersion = np.mean(dispersion_samples)

    # Calculate uncertainty on velocity dispersion
    dispersion_err = np.std(dispersion_samples)

    return dispersion, dispersion_err


def fit_line(x, y, yerr):
    """
    w = (AT C^-1 A)^-1 AT C^-1 y
    Cw = (AT C^-1 A)^-1

    Returns weights, w and covariance Cw.
    """

    AT = np.vstack((np.ones(len(x)), x))
    C = np.eye(len(x))*yerr**2
    Cinv = np.linalg.inv(C)

    CinvA = np.dot(Cinv, AT.T)
    ATCinvA = np.dot(AT, CinvA)

    Cinvy = np.dot(Cinv, y)
    ATCinvy = np.dot(AT, Cinvy)

    w = np.linalg.solve(ATCinvA, ATCinvy)
    Cw = np.linalg.inv(np.dot(AT, CinvA))

    return w, Cw


def fit_cubic(x, y, yerr):
    """
    w = (AT C^-1 A)^-1 AT C^-1 y
    Cw = (AT C^-1 A)^-1

    Returns weights, w and covariance Cw.
    """

    AT = np.vstack((np.ones(len(x)), x, x**2))
    C = np.eye(len(x))*yerr**2
    Cinv = np.linalg.inv(C)

    CinvA = np.dot(Cinv, AT.T)
    ATCinvA = np.dot(AT, CinvA)

    Cinvy = np.dot(Cinv, y)
    ATCinvy = np.dot(AT, Cinvy)

    w = np.linalg.solve(ATCinvA, ATCinvy)
    Cw = np.linalg.inv(np.dot(AT, CinvA))

    return w, Cw


def err_to_log10_err(value, err):
    return err/value/np.log(10)


def err_on_sample_std_dev(std_dev_of_distribution, n):
    """
    from https://stats.stackexchange.com/questions/156518/what-is-the-
    standard-error-of-the-sample-standard-deviation

    Which takes the derivation from
    Rao (1973) Linear Statistical Inference and its Applications 2nd Ed, John
    Wiley & Sons, NY

    Derivation for standard error on the variance is here:
    https://math.stackexchange.com/questions/72975/variance-of-sample-variance

    Args:
        std_dev_of_distribution (float): The standard deviation of the
            Gaussian distribution
        n (int): The number of data points.

    Returns:
        The standard error of the sample standard deviation (not variance).
    """

    sig = std_dev_of_distribution
    return 1./(2*sig) * np.sqrt(2*sig**4/(n-1))

# Calculate tangential distance.
def tan_dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# Find N nearest points
def n_nearest_points(x1, y1, x2, y2, z2, N):
    td = tan_dist(x1, y1, x2, y2)
    inds = np.argsort(td)
    return x2[inds[:N]], y2[inds[:N]], z2[inds[:N]]


def make_bin(x1, y1, x2, y2, z2, xrange, yrange):
    dx, dy = xrange/2., yrange/2.
    xlower, xupper = x1 - dx, x1 + dx
    ylower, yupper = y1 - dy, y1 + dy

    m = (xlower < x2) * (x2 < xupper) * (ylower < y2) * (y2 < yupper)
    return x2[m], y2[m], z2[m]


# Run on each star
def calc_dispersion_nearest(x, y, z, N):

    dispersions = np.zeros(len(x))
    for i in trange(len(x)):
        nx, ny, nz = n_nearest_points(x[i], y[i], x, y, z, N)
        dispersions[i] = 1.5*aps.median_absolute_deviation(nz, ignore_nan=True)

    return dispersions


def calc_dispersion_bins(x, y, z, xrange, yrange):

    dispersions = np.zeros(len(x))
    for i in trange(len(x)):
        nx, ny, nz = make_bin(x[i], y[i], x, y, z, xrange, yrange)
        dispersions[i] = 1.5*aps.median_absolute_deviation(nz, ignore_nan=True)

    return dispersions
