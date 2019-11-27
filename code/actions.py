# Calculate posterior samples of actions.
# Code from Wilma Trick's github repo:
# https://github.com/wilmatrick/GaiaSprint/blob/master/Action_Galpy_Tutorial
# .ipynb

import numpy as np

from galpy.util import bovy_coords
from galpy.potential import MWPotential2014 as pot
from galpy.actionAngle import actionAngleStaeckel


def action(ra_deg, dec_deg, d_kpc, pm_ra_masyr, pm_dec_masyr, v_los_kms,
           verbose=False):
    """
    parameters:
    ----------
    ra_deg: (float)
        RA in degrees.
    dec_deg: (float)
        Dec in degress.
    d_kpc: (float)
        Distance in kpc.
    pm_ra_masyr: (float)
        RA proper motion in mas/yr.
    pm_decmasyr: (float)
        Dec proper motion in mas/yr.
    v_los_kms: (float)
        RV in kms.
    returns:
    ------
    R_kpc, phi_rad, z_kpc, vR_kms, vT_kms, vz_kms
    jR: (float)
        Radial action.
    lz: (float)
        Vertical ang mom.
    jz: (float)
        Vertical action.
    """
    ra_rad = ra_deg * (np.pi / 180.)  # RA [rad]
    dec_rad = dec_deg * (np.pi / 180.)  # dec [rad]

    # Galactocentric position of the Sun:
    X_gc_sun_kpc = 8.  # [kpc]
    Z_gc_sun_kpc = 0.025  # [kpc]

    # Galactocentric velocity of the Sun:
    vX_gc_sun_kms = -9.58  # = -U              [kms]
    vY_gc_sun_kms = 10.52 + 220.  # = V+v_circ(R_Sun) [kms]
    vZ_gc_sun_kms = 7.01  # = W               [kms]

    # a. convert spatial coordinates (ra,dec,d) to (R,z,phi)

    # (ra,dec) --> Galactic coordinates (l,b):
    lb = bovy_coords.radec_to_lb(ra_rad, dec_rad, degree=False, epoch=2000.0)
    # l_rad = lb[:, 0]
    # b_rad = lb[:, 1]
    l_rad = lb[0]
    b_rad = lb[1]

    # (l,b,d) --> Galactocentric cartesian coordinates (x,y,z):
    xyz = bovy_coords.lbd_to_XYZ(l_rad, b_rad, d_kpc, degree=False)
    # x_kpc = xyz[:, 0]
    # y_kpc = xyz[:, 1]
    # z_kpc = xyz[:, 2]
    x_kpc = xyz[0]
    y_kpc = xyz[1]
    z_kpc = xyz[2]

    # (x,y,z) --> Galactocentric cylindrical coordinates (R,z,phi):
    Rzphi = bovy_coords.XYZ_to_galcencyl(x_kpc, y_kpc, z_kpc,
                                         Xsun=X_gc_sun_kpc, Zsun=Z_gc_sun_kpc)
    # R_kpc = Rzphi[:, 0]
    # phi_rad = Rzphi[:, 1]
    # z_kpc = Rzphi[:, 2]
    R_kpc = Rzphi[0]
    phi_rad = Rzphi[1]
    z_kpc = Rzphi[2]

    # b. convert velocities (pm_ra,pm_dec,vlos) to (vR,vz,vT)

    # (pm_ra,pm_dec) --> (pm_l,pm_b):
    pmlpmb = bovy_coords.pmrapmdec_to_pmllpmbb(pm_ra_masyr, pm_dec_masyr,
                                               ra_rad, dec_rad, degree=False,
                                               epoch=2000.0)
    # pml_masyr = pmlpmb[:, 0]
    # pmb_masyr = pmlpmb[:, 1]
    pml_masyr = pmlpmb[0]
    pmb_masyr = pmlpmb[1]

    # (v_los,pm_l,pm_b) & (l,b,d) --> (vx,vy,vz):
    vxvyvz = bovy_coords.vrpmllpmbb_to_vxvyvz(v_los_kms, pml_masyr, pmb_masyr,
                                              l_rad, b_rad, d_kpc, XYZ=False,
                                              degree=False)
    # vx_kms = vxvyvz[:, 0]
    # vy_kms = vxvyvz[:, 1]
    # vz_kms = vxvyvz[:, 2]
    vx_kms = vxvyvz[0]
    vy_kms = vxvyvz[1]
    vz_kms = vxvyvz[2]

    # (vx,vy,vz) & (x,y,z) --> (vR,vT,vz):
    vRvTvZ = bovy_coords.vxvyvz_to_galcencyl(vx_kms, vy_kms, vz_kms, R_kpc,
                                             phi_rad, z_kpc,
                                             vsun=[vX_gc_sun_kms,
                                                   vY_gc_sun_kms,
                                                   vZ_gc_sun_kms],
                                             galcen=True)
    # vR_kms = vRvTvZ[:, 0]
    # vT_kms = vRvTvZ[:, 1]
    # vz_kms = vRvTvZ[:, 2]
    vR_kms = vRvTvZ[0]
    vT_kms = vRvTvZ[1]
    vz_kms = vRvTvZ[2]

    if verbose:
        print("R = ", R_kpc, "\t kpc")
        print("phi = ", phi_rad, "\t rad")
        print("z = ", z_kpc, "\t kpc")
        print("v_R = ", vR_kms, "\t km/s")
        print("v_T = ", vT_kms, "\t km/s")
        print("v_z = ", vz_kms, "\t km/s")

    jR, lz, jz = calc_actions(R_kpc, phi_rad, z_kpc, vR_kms, vT_kms, vz_kms)

    return R_kpc, phi_rad, z_kpc, vR_kms, vT_kms, vz_kms, jR, lz, jz


def calc_actions(R_kpc, phi_rad, z_kpc, vR_kms, vT_kms, vz_kms, verbose=False):
    _REFR0 = 8.  # [kpc]  --> galpy length unit
    _REFV0 = 220.  # [km/s] --> galpy velocity unit

    R = np.array([R_kpc, 8.1]) / _REFR0  # Galactocentric radius
    vR = np.array([vR_kms, 0.]) / _REFV0  # radial velocity
    phi = np.array([phi_rad, 0.])  # Galactocentric azimuth angle
    # (not needed for actions in axisymmetric potential)
    vT = np.array([vT_kms, 230.]) / _REFV0  # tangential velocity
    z = np.array([z_kpc, 0.1]) / _REFR0  # height above plane
    vz = np.array([vz_kms, 0.]) / _REFV0  # vertical velocity

    # delta = focal length of confocal coordinate system
    # Use C code (for speed)
    aAS = actionAngleStaeckel(pot=pot, delta=0.45, c=True)

    jR, lz, jz = aAS(R, vR, vT, z, vz)
    if verbose:
        print("Radial action J_R = ", jR[0]*_REFR0*_REFV0, "\t kpc km/s")
        print("Angular momentum L_z = ", lz[0]*_REFR0*_REFV0, "\t kpc km/s")
        print("Vertical action J_z = ", jz[0]*_REFR0*_REFV0, "\t kpc km/s")
    return jR[0]*_REFR0*_REFV0, lz[0]*_REFR0*_REFV0, jz[0]*_REFR0*_REFV0,
