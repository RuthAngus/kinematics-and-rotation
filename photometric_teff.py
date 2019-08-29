# Calculate photometric Teff from Gaia color
# This should be dereddened colour!


def bprp_to_teff(bp, rp):
    """
    Calculate photometric Teff from Gaia color (use dereddened color!)

    Args:
        bp (array): Gaia G_BP colour.
        rp (array): Gaia G_RP colour.

    Returns:
        teffs (array): Photometric effective temperatures.

    """

    coeffs = [8959.8112335205078, -4801.5566310882568, 1931.4756631851196,
            -2445.9980716705322, 2669.0248055458069, -1324.0671020746231,
            301.13205924630165, -25.923997443169355]
    return np.polyval(coeffs[::-1], bp - rp)
