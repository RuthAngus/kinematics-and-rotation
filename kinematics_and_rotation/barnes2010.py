import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate


def teff_to_mass(teff):
    df = pd.read_csv("mamajek_table.csv", usecols=["Teff", "Msun"])
    f = interpolate.interp1d(df.Teff, df.Msun)
    return f(teff)


def convective_overturn_time(M):
    """Estimate the convective overturn time.

    Estimate the convective overturn time using equation 11 in Wright et al.
    (2011): https://arxiv.org/abs/1109.4634
    log tau = 1.16 - 1.49log(M/M⊙) - 0.54log^2(M/M⊙)

    Args:
        M (float): Mass in Solar units

    Returns:
        The convective overturn time in days.

    """
    log_tau = 1.16 - 1.49*np.log10(M) - .54*(np.log10(M))**2
    return 10**log_tau


def dpdt(prot, tau):
    ki = 740
    return tau / (ki * prot)


def barnes_2010_model(p_init, age, dt, tau):

    p_current = p_init
    for i in np.arange(0, age, dt):
        dp_dt = dpdt(p_current, tau)
        p_current += (dp_dt * dt)

    return p_current


def test_teff_to_mass():
    assert np.isclose(teff_to_mass(5777), 1., atol=.1)


def test_barnes_2010_model():
    assert np.isclose(barnes_2010_model(1., 4560, 1000, mass=1.), 26, atol=3)
    assert np.isclose(barnes_2010_model(1., 4560, 1000, teff=5777), 26, atol=3)


def mass_to_tau_barnes(mass):
    df = pd.read_csv("../data/barnes_and_kim_table.csv")
    f = interpolate.interp1d(df.Mass, df.Global_tauc)
    return f(mass)


if __name__ == "__main__":
    print(mass_to_tau_barnes(1.))
    test_mass = np.linspace(1.2, .4, 100)
    test_tau = mass_to_tau_barnes(test_mass)

    ages = np.linspace(.1, 12, 6)*1e3
    ages[2] = 4560
    for a in ages:
        print(a)
        p = barnes_2010_model(15., a, 100, test_tau)
        plt.plot(test_mass, p)

    plt.plot(1, 26., "o")
    plt.gca().invert_xaxis()
    plt.savefig("test")
    assert 0

    # test_teff_to_mass()
    # test_barnes_2010_model()

    teffs = np.linspace(6000, 3500, 100)
    masses = np.linspace(.5, 1.2, 100)
    ages = np.linspace(1000, 10000, 10)
    for a in ages:
        # prots = barnes_2010_model(1., a, 1000, teff=teffs)
        prots = barnes_2010_model(1., a, 1000, mass=masses)
        plt.plot(masses, prots)
    plt.plot(1., 26, "o")
    plt.gca().invert_xaxis()
    plt.savefig("test")

    # # test teff to mass.
    # df = pd.read_csv("mamajek_table.csv", usecols=["Teff", "Msun"])
    # plt.plot(df.Teff, df.Msun, ".")
    # mass = []
    # # teffs = np.linspace(2000, 30000, 100)
    # teffs = np.linspace(2000, 8000, 100)
    # for t in teffs:
    #     mass.append(teff_to_mass(t))
    # plt.plot(teffs, mass)
    # plt.xlim(1500, 9000)
    # plt.savefig("mamajek")
