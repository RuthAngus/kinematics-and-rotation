# Converting Exploring data into a script.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.utils as au
from astropy.coordinates import SkyCoord
from dustmaps.bayestar import BayestarQuery
import astropy.units as units
from tools import getDust
from stardate.lhf import age_model
from calc_velocities import calc_vb, calc_vz, calc_vl
import astropy.units as u
from astropy.coordinates import ICRS
from astropy.coordinates import Galactic
from astropy.table import Table
from pyia import GaiaData
import astropy.coordinates as coord
from photometric_teff import bprp_to_teff

plotpar = {'axes.labelsize': 30,
           'font.size': 30,
           'legend.fontsize': 15,
           'xtick.labelsize': 30,
           'ytick.labelsize': 30,
           'text.usetex': True}
plt.rcParams.update(plotpar)


print("Load McQuillan data")
mc = pd.read_csv("../Table_1_Periodic.txt")

print("Loading Gaia catalog")
with fits.open("../kepler_dr2_1arcsec.fits") as data:
    gaia = pd.DataFrame(data[1].data, dtype="float64")

gaia_mc = pd.merge(mc, gaia, on="kepid", how="left")
print(len(gaia_mc), "stars")

# S/N cuts
sn = gaia_mc.parallax.values/gaia_mc.parallax_error.values

m = (sn > 10)
m &= (gaia_mc.parallax.values > 0) * np.isfinite(gaia_mc.parallax.values)
m &= gaia_mc.astrometric_excess_noise.values < 5
print(len(gaia_mc.iloc[m]), "stars after S/N cuts")

# Jason's wide binary cuts
# m &= gaia_mc.astrometric_excess_noise.values > 0
# m &= gaia_mc.astrometric_excess_noise_sig.values > 6

# Jason's short-period binary cuts
# m &= radial_velocity_error < 4
# print(len(gaia_mc.iloc[m]), "stars after Jason's binary cuts")
# assert 0

gaia_mc = gaia_mc.iloc[m]

print("Loading Dustmaps")
bayestar = BayestarQuery(max_samples=2, version='bayestar2019')

print("Calculating Ebv")
coords = SkyCoord(gaia_mc.ra.values*units.deg, gaia_mc.dec.values*units.deg,
                  distance=gaia_mc.r_est.values*units.pc)

ebv, flags = bayestar(coords, mode='percentile', pct=[16., 50., 84.],
                      return_flags=True)

# Calculate Av
Av_bayestar = 2.742 * ebv
print(np.shape(Av_bayestar), "shape")
Av = Av_bayestar[:, 1]
Av_errm = Av - Av_bayestar[:, 0]
Av_errp = Av_bayestar[:, 2] - Av
Av_std = .5*(Av_errm + Av_errp)

# Catch places where the extinction uncertainty is zero and default to an
# uncertainty of .05
m = Av_std == 0
Av_std[m] = .05

gaia_mc["ebv"] = ebv[:, 1]  # The median ebv value.
gaia_mc["Av"] = Av
gaia_mc["Av_errp"] = Av_errp
gaia_mc["Av_errm"] = Av_errm
gaia_mc["Av_std"] = Av_std

# Calculate dereddened photometry
AG, Abp, Arp = getDust(gaia_mc.phot_g_mean_mag.values,
                       gaia_mc.phot_bp_mean_mag.values,
                       gaia_mc.phot_rp_mean_mag.values, gaia_mc.ebv.values)

gaia_mc["bp_dered"] = gaia_mc.phot_bp_mean_mag.values - Abp
gaia_mc["rp_dered"] = gaia_mc.phot_rp_mean_mag.values - Arp
gaia_mc["bprp_dered"] = gaia_mc["bp_dered"] - gaia_mc["rp_dered"]
gaia_mc["G_dered"] = gaia_mc.phot_g_mean_mag.values - AG

# Calculate Absolute magntitude
def mM(m, D):
    return 5 - 5*np.log10(D) + m

abs_G = mM(gaia_mc.G_dered.values, gaia_mc.r_est)
gaia_mc["abs_G"] = abs_G

# Remove NaNs
m2 = np.isfinite(gaia_mc.abs_G.values)
gaia_mc = gaia_mc.iloc[m2]

# Remove binaries
x = gaia_mc.bp_dered - gaia_mc.rp_dered
y = gaia_mc.abs_G

AT = np.vstack((x**6, x**5, x**4, x**3, x**2, x, np.ones_like(x)))
ATA = np.dot(AT, AT.T)
w = np.linalg.solve(ATA, np.dot(AT, y))

minb, maxb, extra = 0, 2.2, .27
xs = np.linspace(minb, maxb, 1000)
subcut = 4.

m = (minb < x) * (x < maxb)
m &= (y < np.polyval(w, x) - extra) + (subcut > y)
flag = np.zeros(len(gaia_mc))
flag[~m] = np.ones(len(flag[~m]))
gaia_mc["flag"] = flag

test = gaia_mc.iloc[gaia_mc.flag.values == 1]
plt.plot(gaia_mc.bp_dered - gaia_mc.rp_dered, gaia_mc.abs_G, ".", alpha=.1)
plt.plot(test.bp_dered - test.rp_dered, test.abs_G, ".", alpha=.1)
plt.ylim(10, 1)
plt.savefig("test")

# Calculate photometric Teff
teffs = bprp_to_teff(gaia_mc.bp_dered - gaia_mc.rp_dered)
gaia_mc["color_teffs"] = teffs

print("Calculating gyro ages")
logages = []
for i, p in enumerate(gaia_mc.Prot.values):
    logages.append(age_model(np.log10(p), gaia_mc.phot_bp_mean_mag.values[i] -
                             gaia_mc.phot_rp_mean_mag.values[i]))
gaia_mc["log_age"] = np.array(logages)
gaia_mc["age"] = (10**np.array(logages))*1e-9

plt.figure(figsize=(16, 9), dpi=200)
singles = gaia_mc.flag.values == 1
plt.scatter(gaia_mc.bprp_dered.values[singles], gaia_mc.abs_G.values[singles],
            c=gaia_mc.age.values[singles], vmin=0, vmax=5, s=50, alpha=.2,
            cmap="viridis", rasterized=True, edgecolor="none")
plt.xlabel("$\mathrm{G_{BP}-G_{RP}~[dex]}$")
plt.ylabel("$\mathrm{G~[dex]}$")
plt.colorbar(label="$\mathrm{Gyrochronal~age~[Gyr]}$")
plt.ylim(11, 5.5)
plt.xlim(.8, 2.7);
plt.savefig("age_gradient.pdf")

print("Calculating vb")
pmb_samples, vb_samples = calc_vb(gaia_mc)
pmb, vb = np.median(pmb_samples, axis=1), np.median(vb_samples, axis=1)
pmb_err, vb_err = np.std(pmb_samples, axis=1), np.std(vb_samples, axis=1)
vb_errp = np.percentile(vb_samples, 84, axis=1) - vb
vb_errm = vb - np.percentile(vb_samples, 16, axis=1)
gaia_mc["vb"] = vb
gaia_mc["vb_err"] = vb_err

# print("Calculating vl")
# vl_samples = calc_vl(gaia_mc)
# vl, vl_err = np.median(vl_samples, axis=1), np.std(vl_samples, axis=1)
# vl_errp = np.percentile(vl_samples, 84, axis=1) - vl
# vl_errm = vl - np.percentile(vl_samples, 16, axis=1)
# gaia_mc["vl"] = vl
# gaia_mc["vl_err"] = vl_err

# Calculate b
icrs = ICRS(ra=gaia_mc.ra.values*u.degree,
            dec=gaia_mc.dec.values*u.degree)
lb = icrs.transform_to(Galactic)
b = lb.b*u.degree
l = lb.l*u.degree
gaia_mc["b"] = b.value
gaia_mc["l"] = l.value

print("Calculating VZ")
mrv = gaia_mc.radial_velocity.values != 0.00
vz, vz_err = calc_vz(gaia_mc)
vz[~mrv] = np.ones(len(vz[~mrv]))*np.nan
vz_err[~mrv] = np.ones(len(vz_err[~mrv]))*np.nan
gaia_mc["vz"] = vz
gaia_mc["vz_err"] = vz_err

# Calculate v_ra and v_dec
d = gaia_mc.r_est.values*u.pc

vra = (gaia_mc.pmra.values*u.mas/u.yr * d).to(u.km/u.s,
                                              u.dimensionless_angles())
vdec = (gaia_mc.pmdec.values*u.mas/u.yr * d).to(u.km/u.s,
                                                u.dimensionless_angles())

c = coord.SkyCoord(ra=gaia_mc.ra.values*u.deg, dec=gaia_mc.dec.values*u.deg,
                   distance=d, pm_ra_cosdec=gaia_mc.pmra.values*u.mas/u.yr,
                   pm_dec=gaia_mc.pmdec.values*u.mas/u.yr)
gal = c.galactic
v_b = (gal.pm_b * gal.distance).to(u.km/u.s, u.dimensionless_angles())
gaia_mc["v_ra"] = vra.value
gaia_mc["v_dec"] = vdec.value
gaia_mc["v_b"] = v_b

print("Saving file")
gaia_mc.to_csv("gaia_mc5.csv")
