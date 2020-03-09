import kinematics as k
import numpy as np


def test_construct_and_sample_from_cov():
    N = 2
    ra_err, dec_err, plx_err = [np.ones(N)*.1 for i in range(3)]
    pmra_err, pmdec_err = [np.ones(N)*.1 for i in range(2)]
    ra_dec_corr, ra_plx_corr = np.ones(N)*.2, np.ones(N)*.3
    ra_pmra_corr, ra_pmdec_corr = np.ones(N)*.4, np.ones(N)*.5
    dec_plx_corr, dec_pmra_corr = np.ones(N)*.6, np.ones(N)*.7
    dec_pmdec_corr = np.ones(N)*.8
    plx_pmra_corr, plx_pmdec_corr = np.ones(N)*.9, np.ones(N)*.15
    pmra_pmdec_corr = np.ones(N)*.25

    cov_list = [ra_err, dec_err, plx_err, pmra_err, pmdec_err, ra_dec_corr,
                ra_plx_corr, ra_pmra_corr, ra_pmdec_corr, dec_plx_corr,
                dec_pmra_corr, dec_pmdec_corr, plx_pmra_corr, plx_pmdec_corr,
                pmra_pmdec_corr]
    cov1 = k.construct_cov(cov_list, 5)
    assert np.shape(cov1) == (5, 5, N)

    # cov = np.zeros((5, 5, len(ra_err)))
    # cov[0, 0, :] = ra_err**2
    # cov[0, 1, :] = ra_dec_corr
    # cov[0, 2, :] = ra_plx_corr
    # cov[0, 3, :] = ra_pmra_corr
    # cov[0, 4, :] = ra_pmdec_corr

    # cov[1, 0, :] = ra_dec_corr
    # cov[1, 1, :] = dec_err**2
    # cov[1, 2, :] = dec_plx_corr
    # cov[1, 3, :] = dec_pmra_corr
    # cov[1, 4, :] = dec_pmdec_corr

    # cov[2, 0, :] = ra_plx_corr
    # cov[2, 1, :] = dec_plx_corr
    # cov[2, 2, :] = plx_err**2
    # cov[2, 3, :] = plx_pmra_corr
    # cov[2, 4, :] = plx_pmdec_corr

    # cov[3, 0, :] = ra_pmra_corr
    # cov[3, 1, :] = dec_pmra_corr
    # cov[3, 2, :] = plx_pmra_corr
    # cov[3, 3, :] = pmra_err**2
    # cov[3, 4, :] = pmra_pmdec_corr

    # cov[4, 0, :] = ra_pmdec_corr
    # cov[4, 1, :] = dec_pmdec_corr
    # cov[4, 2, :] = plx_pmdec_corr
    # cov[4, 3, :] = pmra_pmdec_corr
    # cov[4, 4, :] = pmdec_err**2

    # for i in range(5):
    #     for j in range(5):
    #         for n in range(N):
    #             assert cov1[i, j, n] == cov[i, j, n]

    ra, dec, plx = np.ones(N)*10, np.ones(N)*20, np.ones(N)*30
    pmra, pmdec  = np.ones(N)*2, np.ones(N)*3

    means = [ra, dec, plx, pmra, pmdec]
    samples = k.sample_from_cov(means, cov_list, 100)
    print(np.shape(samples))

if __name__ == "__main__":
    test_construct_and_sample_from_cov()
