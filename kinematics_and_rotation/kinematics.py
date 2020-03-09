"""
Calculate velocities and vertical actions
"""


import astropy.units as u
from astropy.coordinates import ICRS
from astropy.coordinates import Galactic
import astropy.coordinates as coord
import numpy as np


def calc_vb(mean_list):
    """
    Calculate latitudinal velocity.

    Args:
        mean_list (list): list of arrays of,
            ra: Right Ascension in degrees.
            dec: Declination in degrees.
            parallax: parallax in milliarcseconds.
            pmra: RA proper motion in milliarcseconds per year.
            pmdec: Dec proper motion in milliarcseconds per year.

    Returns:
        The array of latitudinal velocities.

    """
    ra, dec, parallax, pmra, pmdec = mean_list

#     icrs = ICRS(ra=ra*u.degree,
#                 dec=dec*u.degree,
#                 distance=distance*u.pc,
#                 pm_ra_cosdec=pmra*u.mas/u.yr,
#                 pm_dec=pmdec*u.mas/u.yr)
#     vels = icrs.transform_to(Galactic)

    d = coord.Distance(parallax=parallax*u.mas)
    vra = (pmra*u.mas/u.yr * d).to(u.km/u.s, u.dimensionless_angles())
    vdec = (pmdec*u.mas/u.yr * d).to(u.km/u.s, u.dimensionless_angles())

    c = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=d,
                       pm_ra_cosdec=pmra*u.mas/u.yr,
                       pm_dec=pmdec*u.mas/u.yr)
    gal = c.galactic
    v_b = (gal.pm_b * gal.distance).to(u.km/u.s, u.dimensionless_angles())

    return v_b


def vb_with_err(mean_list, cov_list, Nsamples):
    """
    Calculate latitudinal velocities with uncertainties.

    Args:
        mean_list (list): A list of arrays of astrometric data.
            [ra, dec, plx, pmra, pmdec]
        cov_list (array): A list of all the uncertainties and covariances:
            [ra_err, dec_err, plx_err, pmra_err, pmdec_err, ra_dec_corr,
            ra_plx_corr, ra_pmra_corr, ra_pmdec_corr, dec_plx_corr,
            dec_pmra_corr, dec_pmdec_corr, plx_pmra_corr, plx_pmdec_corr,
            pmra_pmdec_corr]
        Nsamples: (int): The number of samples.

    """


def sample_from_cov(mean_list, cov_list, Nsamples):
    """
    Sample from the multivariate Gaussian of Gaia astrometric data.

    Args:
        mean_list (list): A list of arrays of astrometric data.
            [ra, dec, plx, pmra, pmdec]
        cov_list (array): A list of all the uncertainties and covariances:
            [ra_err, dec_err, plx_err, pmra_err, pmdec_err, ra_dec_corr,
            ra_plx_corr, ra_pmra_corr, ra_pmdec_corr, dec_plx_corr,
            dec_pmra_corr, dec_pmdec_corr, plx_pmra_corr, plx_pmdec_corr,
            pmra_pmdec_corr]
        Nsamples: (int): The number of samples.

    """

    Ndim = len(mean_list)  # 5 dimensions: ra, dec, plx, pmra, pmdec
    Nstars = len(mean_list[0])

    # Construct the mean and covariance matrices.
    mean = np.vstack(([i for i in mean_list]))
    cov = construct_cov(cov_list, Ndim)

    # Sample from the multivariate Gaussian
    samples = np.zeros((Nsamples, Ndim, Nstars))
    for i in range(Nstars):
        samples[:, :, i] = np.random.multivariate_normal(
            mean[:, i], cov[:, :, i], Nsamples)
    return samples


def construct_cov(element_list, Ndim):
    """
    Construct the covariance matrix between ra, dec, pmra, pmdec and parallax.

    Args:
        element_list (list): All the uncertainties and covariances:
            [ra_err, dec_err, plx_err, pmra_err, pmdec_err, ra_dec_corr,
            ra_plx_corr, ra_pmra_corr, ra_pmdec_corr, dec_plx_corr,
            dec_pmra_corr, dec_pmdec_corr, plx_pmra_corr, plx_pmdec_corr,
            pmra_pmdec_corr]
        Ndim (int): The number of dimensions. 5 for Gaia (ra, dec, plx, pmra,
            pmdec)

    Returns:
        cov (array): The 5 x 5 x Nstar covariance matrix.
        cov = [[ra**2    rad_c    raplx_c    rapmra_c    rapmd_c]
               [rad_c    d**2     dplx_c     dpmra_c     dpmdec_c]
               [raplx_c  dplx_c   plx**2     plxpmra_c   plxpmdec_c]
               [rapmra_c dpmra_c  plxpmra_c  pmra**2     pmrapmdec_c]
               [rapmd_c  dpmdec_c plxpmdec_c pmrapmdec_c pmdec**2]]

    """

    ra_err, dec_err, plx_err, pmra_err, pmdec_err, ra_dec_corr, ra_plx_corr, \
        ra_pmra_corr, ra_pmdec_corr, dec_plx_corr, dec_pmra_corr, \
        dec_pmdec_corr, plx_pmra_corr, plx_pmdec_corr, pmra_pmdec_corr \
        = element_list

    cov = np.zeros((Ndim, Ndim, len(ra_err)))
    cov[0, 0, :] = ra_err**2
    cov[0, 1, :] = ra_dec_corr
    cov[0, 2, :] = ra_plx_corr
    cov[0, 3, :] = ra_pmra_corr
    cov[0, 4, :] = ra_pmdec_corr

    cov[1, 0, :] = ra_dec_corr
    cov[1, 1, :] = dec_err**2
    cov[1, 2, :] = dec_plx_corr
    cov[1, 3, :] = dec_pmra_corr
    cov[1, 4, :] = dec_pmdec_corr

    cov[2, 0, :] = ra_plx_corr
    cov[2, 1, :] = dec_plx_corr
    cov[2, 2, :] = plx_err**2
    cov[2, 3, :] = plx_pmra_corr
    cov[2, 4, :] = plx_pmdec_corr

    cov[3, 0, :] = ra_pmra_corr
    cov[3, 1, :] = dec_pmra_corr
    cov[3, 2, :] = plx_pmra_corr
    cov[3, 3, :] = pmra_err**2
    cov[3, 4, :] = pmra_pmdec_corr

    cov[4, 0, :] = ra_pmdec_corr
    cov[4, 1, :] = dec_pmdec_corr
    cov[4, 2, :] = plx_pmdec_corr
    cov[4, 3, :] = pmra_pmdec_corr
    cov[4, 4, :] = pmdec_err**2

    return cov
