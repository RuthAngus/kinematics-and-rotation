import numpy as np
import matplotlib.pyplot as plt
from dispersion import *
import scipy.stats as sps
import pandas as pd


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
    db, k = dispersion(x, y, bins, method="std")

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
    d, d_err, k, k_err = MC_dispersion(x+xerrs, y+yerrs, xerrs, yerrs, bins,
                                       100, method="std")
    d2, d2_err, k2, k2_err = MC_dispersion(x+xerrs, y+yerrs, xerrs, yerrs,
                                           bins, 100, method="mad")
    plt.plot(x, y, ".")
    print(np.shape(d))
    plt.plot(bins[:-1], d)
    plt.plot(bins[:-1], d2)
    plt.savefig("test")


def test_sigma_clip():
    np.random.seed(42)

    x1 = np.random.randn(1000)
    x2 = np.random.randn(100)*5
    x = np.concatenate((x1, x2))
    np.random.shuffle(x)

    print(sps.kurtosis(x))

    plt.plot(np.arange(1100), x, ".")
    xnew, m = sigma_clip(x, 3)

    print(sps.kurtosis(xnew))

    # plt.plot(np.arange(1100)[m], xnew, ".", alpha=.5)
    # plt.savefig("test")


def test_select_stars():
    df = pd.DataFrame(dict({"A": np.arange(10), "B": np.arange(10, 20)}))
    ms = select_stars(df, [1, 5, 8], "A")
    assert np.all(df.A.values[ms[0]] > 1)
    assert np.all(df.A.values[ms[0]] < 5)
    assert np.all(df.A.values[ms[1]] < 8)
    assert np.all(df.A.values[ms[1]] > 5)


def test_fit_line():
    w_true = [15, .2]

    n = 100
    x = np.linspace(0, 100, n)

    err = .5
    yerr = np.ones_like(x)*err
    np.random.seed(42)
    y = w_true[0] + w_true[1]*x + np.random.randn(n)*err

    w, wvar = fit_line(x, y, yerr)

    assert np.isclose(w[0], 15, atol=1*np.sqrt(wvar[0, 0]))
    assert np.isclose(w[1], .2, atol=1*np.sqrt(wvar[1, 1]))


def test_err_to_log10_err():
    value = 20
    err = .1
    assert np.isclose(10**(np.log10(value) + err_to_log10_err(value, err)),
                      value + err, atol=.01*value)


def test_tan_dist():
    x1, y1 = 1, 1
    x2, y2 = 2, 2
    assert tan_dist(x1, y1, x2, y2) == np.sqrt(2)


def test_n_nearest_points():
    x1, y1 = 10, 12
    np.random.seed(42)
    x2, y2 = [np.random.randn(1000) + 10 for i in range(2)]
    z2 = np.random.randn(1000)*y2
    nx, ny, nz = n_nearest_points(x1, y1, x2, y2, z2, 50)


def test_make_bin():
    np.random.seed(42)
    x, y, z = [np.random.randn(1000) + 10 for i in range(3)]
    bx, by, bz = make_bin(10, 10, x, y, z, 1, 1)
    plt.plot(x, y, ".")
    plt.plot(bx, by, ".")


def test_calc_dispersion():
    x2, y2 = [np.random.randn(1000) + 10 for i in range(2)]
    z2 = np.random.randn(1000)*y2

    dispersions_nearest = calc_dispersion_nearest(x2, y2, z2, 100);
    dispersions_bins = calc_dispersion_bins(x2, y2, z2, .5, .5);

    return dispersions_nearest, dispersions_bins


if __name__ == "__main__":
    # test_dispersion()
    # test_MC()
    # test_sigma_clip()
    # test_select_stars()
    # test_fit_line()
    test_err_to_log10_err()
    test_tan_dist()
    test_n_nearest_points()
    test_make_bin()
    test_calc_dispersion()


