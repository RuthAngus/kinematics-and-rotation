# Calculate photometric Teff from Gaia color
# This should be dereddened colour!

import numpy as np


def bprp_to_teff(bprp):
    """
    Calculate photometric Teff from Gaia color (use dereddened color!)

    Args:
        bprp (array): Gaia G_BP colour minus Gaia G_RP colour.

    Returns:
        teffs (array): Photometric effective temperatures.

    """

    # coeffs = [8959.8112335205078, -4801.5566310882568, 1931.4756631851196,
    #         -2445.9980716705322, 2669.0248055458069, -1324.0671020746231,
    #         301.13205924630165, -25.923997443169355]

    # Jason's updated parameters:
    coeffs = [-416.585, 39780.0, -84190.5, 85203.9, -48225.9, 15598.5,
              -2694.76, 192.865]

    return np.polyval(coeffs[::-1], bprp)
