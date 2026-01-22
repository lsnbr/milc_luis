from pathlib import Path
import numpy as np
import gvar as gv
import lsqfit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flowing import flowtime_to_radius, radius_to_flowtime
from measurements import *
from statana import *
from ana_tail import make_prior_constr, expx_single_mats, plot_flowtime_and_tau_fits, p_mats




nt = 16
ns = 64
flowtimes = np.array([ 0.02572923590301565, 0.05303278154520656, 0.08103495224784385, 0.110208782326772,  0.141058391298962,  0.1736893835495303,
                       0.2086835114617738,  0.2454549361253429,  0.2838117228448263,  0.3249152519239425, 0.3705101434001126, 0.4179449298122797,
                       0.4696233239687699,  0.5279559816139976,  0.5890993819917063,  0.6570215111572449, 0.734680305158274,  0.8257738792221512,
                       0.9353328035912798,  1.06959076807306,    1.233820281085228,   1.435936459574629 ], dtype=float)#, 1.68034447439404, 1.972161672512587 ]







def main2():

    # get correlators  G(w_n, r) / T^7
    print('loading correlator data...')
    ense_all = get_data_mats_unbinned()
    print(ense_all.shape, ense_all.dtype)

    # get distances
    dist = radial_separations(ns)


    # bin with constant r bins
    bin_size = [0.25, 0.5][1]
    bins = find_distance_bins(dist, bin_size)
    dist_binned, = bin_averages(bins, dist)


    # flowtime
    iflow = 10



    ###################################################################
    ########################  prepare data  ###########################
    ###################################################################
    
    ense_m0 = ense_all[:, iflow, 0, :]
    ense_m1 = ense_all[:, iflow, 1, :]
    ense_m2 = ense_all[:, iflow, 2, :]

    ense_m0_b, ense_m1_b, ense_m2_b = bin_averages(bins, ense_m0, ense_m1, ense_m2)

    data_m0 = gv.dataset.avg_data(ense_m0_b)
    data_m1 = gv.dataset.avg_data(ense_m1_b)
    data_m2 = gv.dataset.avg_data(ense_m2_b)



    # compute integrand for n=0,1,2 for each ensemble
    ense_m0_int = build_integrand(dist, ense_all[:, iflow, 0, :], 0)
    ense_m1_int = build_integrand(dist, ense_all[:, iflow, 1, :], 1)
    ense_m2_int = build_integrand(dist, ense_all[:, iflow, 2, :], 2)

    # do binning
    ense_m0_int_b, ense_m1_int_b, ense_m2_int_b = bin_averages(bins, ense_m0_int, ense_m1_int, ense_m2_int)

    # ensemble average
    data_m0_int = gv.dataset.avg_data(ense_m0_int_b)
    data_m1_int = gv.dataset.avg_data(ense_m1_int_b)
    data_m2_int = gv.dataset.avg_data(ense_m2_int_b)

    # do subtractions
    data_sub_int = data_m0_int + (1/3) * data_m1_int + (-4/3) * data_m2_int    



    ###################################################################
    ########################  plot raw data  ##########################
    ###################################################################

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
    axes = np.reshape(axes, shape=(nrows, ncols))

    # left and right limits for different bin_sizes
    il0, ir0 = {0.5 : (0, 21), 0.25 : (0,  38)}[bin_size]
    il1, ir1 = {0.5 : (4, 18), 0.25 : (6,  32)}[bin_size]
    il2, ir2 = {0.5 : (0, 22), 0.25 : (0,  40)}[bin_size]
    il3, ir3 = {0.5 : (12, 25), 0.25 : (21, 53)}[bin_size]

    # plot integrands of individual matsubara frequencies
    plot_dist_many(dist_binned, [data_m0_int, data_m1_int, data_m2_int], il0, ir0, ['n=0', 'n=1', 'n=2'], axes[0,1])
    axes[0,1].set_ylabel('integrand / T^5')

    plot_dist_many(dist_binned, [data_m0_int, data_m1_int, data_m2_int], il1, ir1, ['n=0', 'n=1', 'n=2'], axes[1,1])
    axes[1,1].set_ylabel('integrand / T^5')

    # plot subtracted modes
    plot_dist_many(dist_binned, [data_sub_int], il0, ir0, ['sub012'], axes[0,2])
    axes[0,2].set_ylabel('integrand / T^5')

    plot_dist_many(dist_binned, [data_sub_int], il1, ir1, ['sub012'], axes[1,2])
    axes[1,2].set_ylabel('integrand / T^5')

    # plot pure matsubara modes
    plot_dist_many(dist_binned, [data_m0, data_m1, data_m2], il2, ir2, ['n=0', 'n=1', 'n=2'], axes[0,0])
    axes[0,0].set_ylabel('G(w_n, r) / T^7')

    plot_dist_many(dist_binned, [data_m0, data_m1, data_m2], il3, ir3, ['n=0', 'n=1', 'n=2'], axes[1,0])
    axes[1,0].set_ylabel('G(w_n, r) / T^7')

    # finalize and save plots
    fig.tight_layout()
    fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'gridmats.png', dpi=400)



    ###################################################################
    #################  fit and plot pure mats data  ###################
    ###################################################################
    print('do fits...')

    mats_flowtimes = [
        ( 0, [10, 15] ),
        ( 1, [10, 15] ),
        ( 2, [10, 15] ),
    ]

    labels = {}
    for mats, iflows in mats_flowtimes:
        for iflow in iflows:
            rf = flowtime_to_radius(flowtimes[iflow], nt) * nt
            labels[iflow, mats] = f'rf={rf:.2f}a, n={mats}'

    dist_fit, data_fit, ense_fit, r_lims_fit = bin_cut_avg_data(dist, ense_all, {idx : bins for idx in labels.keys()}, sn_cut_left=10)

    fit0 = fit_flowtime_and_mats_tails_with_prior(dist_fit, data_fit, excited_max=0, mats_max=2)
    print(fit0)

    bins2 = bin_through_simultaneous_fit(dist, ense_all, labels, fit0, reltol=0.025, max_bin_size=10)
    dist_fit, data_fit, ense_fit, r_lims_fit = bin_cut_avg_data(dist, ense_all, bins2, sn_cut_left=10)

    fit = fit_flowtime_and_mats_tails_with_prior(dist_fit, data_fit, excited_max=0, mats_max=2)
    print(fit)


    # plot fits
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
    axes = np.reshape(axes, shape=(nrows, ncols))

    plot_flowtime_and_tau_fits(dist, labels, r_lims_fit, fit, synchro=True, axes=axes)

    fig.tight_layout()
    fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'gridmats_fits.png', dpi=400)



    ###################################################################
    ##################  fit and plot integrands  ######################
    ###################################################################
    print('do fits (integrand)...')

    ense_int = ense_all[:, :, :3, :].copy()
    for mats in range(3):
        for iflow, _ in enumerate(flowtimes):
            ense_int[:, iflow, mats, :] = build_integrand(dist, ense_int[:, iflow, mats, :], mats)

    rlims_int = { 0 : (8, 20), 1 : (7, 17), 2 : (6, 13) }
    r_cuts0_left  = { (iflow, mats) : rlims_int[mats][0] for iflow, mats in labels.keys() }
    r_cuts0_right = { (iflow, mats) : rlims_int[mats][1] for iflow, mats in labels.keys() }
    # dist_fit, data_fit, ense_fit, r_lims_fit = bin_cut_avg_data(dist, ense_int, {idx : bins for idx in labels.keys()}, sn_cut_left=15, err_max_right=0.02)
    dist_fit, data_fit, ense_fit, r_lims_fit = bin_cut_avg_data(dist, ense_int, {idx : bins for idx in labels.keys()}, r_cuts0_left=r_cuts0_left, r_cuts0_right=r_cuts0_right)

    fit0 = fit_flowtime_and_mats_tails_int_with_prior(dist_fit, data_fit, excited_max=0, mats_max=2)
    print(fit0)


    # plot fits
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
    axes = np.reshape(axes, shape=(nrows, ncols))

    plot_flowtime_and_tau_fits(dist, labels, r_lims_fit, fit0, synchro=True, axes=axes)

    fig.tight_layout()
    fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'gridmats_int_fits.png', dpi=400)











def fit_flowtime_and_mats_tails_with_prior(dist : dict[Any, np.ndarray], data : dict[Any, np.ndarray], excited_max : int, mats_max : int, p0 : dict|None = None, corr : bool = True) -> lsqfit.nonlinear_fit:
    '''fit with priors'''

    iflows = sorted(iflow for iflow,_ in data.keys())
    prior  = make_prior_constr(excited_max, mats_max, iflows)

    def fitfcn(x, p):
        y = {}
        for idx in x.keys():
            iflow, mats = idx
            y[idx] = expx_single_mats(x[idx], p, iflow, mats, excited_max)
        return y
    
    return (
        lsqfit.nonlinear_fit( data =(dist, data), fcn=fitfcn, prior=prior, p0=p0 )
        if corr else
        lsqfit.nonlinear_fit( udata=(dist, data), fcn=fitfcn, prior=prior, p0=p0 )
    )






def fit_flowtime_and_mats_tails_int_with_prior(dist : dict[Any, np.ndarray], data : dict[Any, np.ndarray], excited_max : int, mats_max : int, p0 : dict|None = None, corr : bool = True) -> lsqfit.nonlinear_fit:
    '''fit with priors'''

    iflows = sorted(iflow for iflow,_ in data.keys())
    prior  = make_prior_constr(excited_max, mats_max, iflows)

    def fitfcn(x, p):
        y = {}
        for idx in x.keys():
            iflow, mats = idx
            y[idx] = build_integrand(
                x[idx],
                expx_single_mats(x[idx], p, iflow, mats, excited_max),
                mats
            )
        return y
    
    return (
        lsqfit.nonlinear_fit( data =(dist, data), fcn=fitfcn, prior=prior, p0=p0 )
        if corr else
        lsqfit.nonlinear_fit( udata=(dist, data), fcn=fitfcn, prior=prior, p0=p0 )
    )






def plot_dist_many(dist : np.ndarray, datas : list[np.ndarray], il : int, ir : int, labels : list[str], axes : plt.Axes) -> None:
    '''plot, including data up to signa; to noise sn_max'''

    for data, label in zip(datas, labels):
        plot_dist(
            dist = dist[il:ir],
            data = data[il:ir],
            axes = axes,
            label = label
        )




def build_integrand(dist : np.ndarray, data : np.ndarray, mats : int) -> np.ndarray:
    '''computes integrand of H_E in units of T^5'''

    if mats == 0:
        res = data * dist**2 / nt**2
    
    else:
        w = p_mats(mats)
        res = data * dist * np.sinh(w*dist/nt) / w / nt

    return res







def get_mats012_subtraction(ense : np.ndarray) -> np.ndarray:
    '''computes m0 + 1/3 m1 - 4/3 m2, which removes 1/w^2 and 1/w^4 contributions'''

    return ense[:,:,0,:] + (1/3) * ense[:,:,1,:] + (-4/3) * ense[:,:,2,:]

































if __name__ == '__main__':

    main2()
