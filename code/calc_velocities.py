"""
Calculate proper motion in the b direction.
"""

import numpy as np
from astropy.io import fits
from pyia import GaiaData
import pandas as pd
from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import ICRS
from astropy.coordinates import Galactic


def calc_vb(pandas_df, nsamples=1000):
    """
    Calculate velocity samples in the b direction from a pandas DataFrame.

    Args:
        df (DataFrame): pandas dataframe containing Gaia columns

    Returns:
        pm_b (array): samples of velocity in the b direction.
            Shape = nstars x nsamples

    """

    df = Table.from_pandas(pandas_df)
    g = GaiaData(df)
    g_samples = g.get_error_samples(size=nsamples,
                                    rnd=np.random.RandomState(seed=42))
    c_samples = g_samples.get_skycoord()
    vels = c_samples.transform_to(coord.Galactic)
    pm_b = np.array(vels.pm_b.value)

    gal = c_samples.galactic
    v_b = (gal.pm_b * gal.distance).to(u.km/u.s, u.dimensionless_angles())
    return pm_b, np.array(v_b.value)


def calc_vl(pandas_df, nsamples=1000):
    """
    Calculate velocity samples in the l direction from a pandas DataFrame.

    Args:
        df (DataFrame): pandas dataframe containing Gaia columns

    Returns:
        pm_l (array): samples of velocity in the l direction.
            Shape = nstars x nsamples

    """

    df = Table.from_pandas(pandas_df)
    g = GaiaData(df)
    g_samples = g.get_error_samples(size=nsamples,
                                    rnd=np.random.RandomState(seed=42))
    c_samples = g_samples.get_skycoord()
    vels = c_samples.transform_to(coord.Galactic)
    # pm_l = np.array(vels.pm_l.value)

    gal = c_samples.galactic
    v_l = (gal.pm_l * gal.distance).to(u.km/u.s, u.dimensionless_angles())
    return np.array(v_l.value)


def calc_vz(pandas_df, nsamples=1000):
    """
    Calculate W samples from a pandas DataFrame.

    Args:
        df (DataFrame): pandas dataframe containing Gaia columns

    Returns:
        W (array): samples of W velocity.
            Shape = nstars x nsamples

    """

    df = Table.from_pandas(pandas_df)
    g = GaiaData(df)
    g_samples = g.get_error_samples(size=nsamples,
                                    rnd=np.random.RandomState(seed=42))
    c_samples = g_samples.get_skycoord()
    vels = c_samples.transform_to(coord.Galactocentric)
    W = vels.v_z.value
    return np.mean(W, axis=1), np.std(W, axis=1)
