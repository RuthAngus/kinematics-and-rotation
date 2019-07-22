import numpy as np


def bprp_to_teff(bprp):
    """
    Calculate an effective temperature from a bp-rp color.

    """

    p = [0.01868672, -0.14779631, 0.17718147, 1.44461987, -5.50943128,
         9.26444289]
    return np.polyval(p, bprp)*1000


def teff_to_bprp(teff):
    """
    Calculate a bp-rp color from an effective temperature.

    """

    p = [-2.6948047689554756e-18, 8.945100145214491e-14,
         -1.1752321538397444e-09, 7.673913205006288e-06,
         -0.025302316464295933, 35.17655191550194]
    return np.polyval(p, teff)


def bprp_to_bv(bprp):
    """
    Calculate a B-V color from a G_BP - G_RP color.

    Poor fit at large and small bp-rp. (Especially bad for low-mass stars).

    """

    p = [-0.0334115504231673, 0.3754200727545447, -1.4691791496336342,
         2.209433533403669, -0.39641699367657274, 0.14752490961352926]
    return np.polyval(p, bprp)


def bprp_to_bv(bv):
    """
    Calculate a G_BP - G_RP color from a B-V color.

    Poor fit at large and small B-V. (Especially bad for low-mass stars).

    """

    p = [-0.9933881241365243, 2.20310834260743, 1.839433873944796,
         -6.714250044137177, 4.501655472990212, 0.2647400296095147,
         0.0845117268687072]

    return np.polyval(p, bv)


def teff_to_bv(teff):

    p = [-1.6959955278110665e-19, 9.283100332998309e-16,
         4.761475302858526e-11, -6.354265999828208e-07,
         0.0024740588508326837, -1.5621606432703632]

    return np.polyval(p, teff)


def bv_to_teff(bv):

    p = [3578.180565278925, -13195.543683684042, 15201.484715176379,
         -3669.547034686415, -6212.974928561518, 9105.456630795197]

    return np.polyval(p, bv)
