import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def MC_dispersion(x, y, xerr, yerr, bins, nsamps):
    """
    Calculate the dispersion in a set of bins, with Monte Carlo uncertainties.

    Args:
        x (array): The x-values.
        y (array): The y-values.
        xerr (array): The x-value uncertainties.
        yerr (array): The y-value uncertainties.
        bins (array): The bin edges. length = number of bins + 1
        nsamps (int): Number of Monte Carlo samples.

    Returns:
        dispersion (array): The dispersion array. length = number of bins.
        dispersion_err (array): The uncertainty on the dispersion.
            length = number of bins.

    """
    nbins = len(bins) - 1

    # Sample from Gaussians
    xsamps = np.zeros((len(x), nsamps))
    ysamps = np.zeros((len(y), nsamps))
    dsamps = np.zeros((nbins, nsamps))

    for i in trange(nsamps):
        xsamps[:, i] = np.random.randn(len(x))*xerr + x
        ysamps[:, i] = np.random.randn(len(y))*yerr + y
        dsamps[:, i] = dispersion(xsamps[:, i], ysamps[:, i], bins)

    return np.mean(dsamps, axis=1), np.std(dsamps, axis=1)


def dispersion(x, y, bins):
    """
    Calculate the dispersion in a set of bins.

    Args:
        x (array): The x-values.
        y (array): The y-values.
        bins (array): The bin edges. length = number of bins + 1

    Returns:
        dispersion (array): The dispersion array. length = number of bins.

    """

    d = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        m = (bins[i] < x) * (x < bins[i+1])
        d[i] = np.std(y[m])
    return d


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


def test_dispersion():
    np.random.seed(42)
    N = 10000
    x = np.random.uniform(0, 100, N)
    y = np.random.randn(N)*x
    inds = np.argsort(x)
    x, y = x[inds], y[inds]

    # Running dispersion
    newx, d = running_dispersion(x, y, 100, mad=False)

    AT = np.vstack((newx, np.ones_like(newx)))
    ATA = np.dot(AT, AT.T)
    w = np.linalg.solve(ATA, np.dot(AT, d))

    # Binned dispersion
    bins, dbins, err, mean = binned_dispersion(x, y, 10, method="rms")

    # Dispersion where you define the bins.
    db = dispersion(x, y, bins)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, ".")
    plt.plot(x, x, zorder=3, label="True dispersion")
    plt.plot(newx, d, ".", label="running")
    plt.plot(x, w[0]*x + w[1], label="fit to running")
    plt.plot(np.diff(bins)*.5+bins[:-1], db, "k*", label="pre-set bins",
             zorder=10)
    plt.title("TESTING DISPERSION")
    plt.errorbar(bins, dbins, yerr=err, fmt="wo", markeredgecolor="k",
                 label="RMS")
    plt.legend()
    plt.savefig("dispersion_test")

    assert np.isclose(w[0], 1, atol=.1)


def test_MC():
    np.random.seed(42)
    N = 10000
    x = np.random.uniform(0, 100, N)
    y = np.random.randn(N)*x
    xerr, yerr = 5, 10
    xerrs = np.random.randn(N)*xerr
    yerrs = np.random.randn(N)*yerr
    bins = np.linspace(0, 100, 10)
    MC_dispersion(x+xerrs, y+yerrs, xerrs, yerrs, bins, 100)


if __name__ == "__main__":
    # test_dispersion()
    test_MC()
