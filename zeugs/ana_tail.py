from pathlib import Path
import dataclasses
from itertools import product
import pickle
from typing import Callable, Any
import numpy as np
import gvar as gv
import lsqfit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flowing import flowtime_to_radius, radius_to_flowtime
from measurements import *
from statana import *



from ensemble_data import data_16_x_64_1p5Tc

nt = data_16_x_64_1p5Tc['nt']
ns = data_16_x_64_1p5Tc['ns']
flowtimes = data_16_x_64_1p5Tc['flowtimes']



def main1():

    # TODO
    # bootstrap error analysis
    # more thought about data/fit contribution to r-sum


    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(14, 20))


    # getting data
    ense_raw = get_data_unbinned()
    dist_raw = radial_separations(ns)

    bin_size = 0.9
    ense = get_data_binned(bin_size)
    dist = bin_distances(ns, bin_size)

    tau = 5
    iflowtimes = [6,7,8,9]

    dataset = ense[:, 6:10, tau, :].copy()
    data = gv.dataset.avg_data(dataset)


    # flowtime window
    vis_flowtime_window(tau, axes[0,1])


    # r correlations
    ai = 2
    vis_correlations_in_r(dist, data[ai], axes[ai,1], fig)
    axes[2,1].set_title(f'Correlations in r (r_F={flowtime_to_radius(flowtimes[iflowtimes[ai]], nt)*nt:.2f}a, tau={tau})')


    # multiple flowtimes combined fit
    sn_cut = 10

    do_svdcut = False
    data_cut = do_svd_cut(dataset, axes[1,1])
    if do_svdcut: data = data_cut

    fit, ir_cuts = combined_fit(dist, data, sn_cut, 1)
    print(fit)
    plot_fitc(dist, data, ir_cuts, fit, tau, iflowtimes, axes[:,0].flatten())


    # do sum over r
    rsums = []
    for iifl, ifl in enumerate(iflowtimes):
        data_raw = gv.dataset.avg_data(ense_raw[:, ifl, tau, :].copy())
        ddist_raw = radial_multiplicities_r(ns)
        res = sum_over_r(dist_raw, data_raw, ddist_raw, fit.p['a'][iifl], fit.p['m'], dist[ir_cuts[iifl]], axes[iifl,1] if iifl==3 else None)
        if iifl==3: axes[iifl,1].set_title(f'partial sums up to r (r_F={flowtime_to_radius(flowtimes[ifl], nt)*nt:.2f}a, tau={tau})')
        rsums.append(res)
    

    # plot G_F(tau)
    # plot_over_flowtime(tau, iflowtimes, rsums, axes[-1,1])


    # test binning dependent on function
    _ = bin_data_through_fcn(dist_raw, ense_raw[:, iflowtimes[-1], tau, :].copy(), lambda r: r**(-6), 0.001, axes[-1, 0])


    # finalize figure
    plt.tight_layout()
    plt.savefig(Path.cwd() / 'zeugs' / 'plots' / 'grid.png', dpi=600)

def main2():

    # getting distances and configuration data
    dist = radial_separations(ns)
    ense_all = get_data_unbinned()

    # tau and flowtime indices
    tau, i_fts = [ (5, [6,7,8,9]),
                   (6, [9, 10, 11, 12]), ][0]

    # ense and labels
    ense = ense_all[:, i_fts, tau:tau+1, :]
    labels = {}
    for idx in np.ndindex(ense.shape[1:-1]):
        labels[idx] = f'tf={flowtimes[idx[0]]:.2f}a^2, tau={idx[1]}a'

    # run stuff
    print('starting...')
    # result = do_tail_fit_and_sum(dist, ense, labels, 10)
    results = do_tailfit_bootstrap(dist, ense, labels, 10)
    res0 = do_tail_fit_and_sum(dist, ense, labels, 10)
    print()
    print('r_cuts_sn      =', res0.r_cuts_sn)
    print('r_cuts_sum     =', res0.r_cuts_sum)
    print('number of bins =', len(res0.bins.flatten()[0]))
    print('total sums     =', res0.rsums)
    print()
    print(res0.bins)
    print()
    print(res0.fit)
    print()

    # plot stuff
    nrows, ncols = len(i_fts), 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
    plot_tail_fits(dist, ense, res0, labels, axes[:, 0].flatten())
    plot_tailfit_bootstrap(results, labels, axes[:, 1].flatten())

    # save results
    # with open('./zeugs/data/tailfits.pkl', 'wb') as f:
    #     pickle.dump([dataclasses.replace(res, fit=None) for res in results], f)

    # finalize figure
    plt.tight_layout()
    plt.savefig(Path.cwd() / 'zeugs' / 'plots' / 'grid2.png', dpi=600)





def main_windows():

    nrows = 1
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
    axes = np.reshape(axes, shape=(nrows, ncols))

    tau = 7
    # vis_flowtime_window(tau, axes[0,0])

    vis_flowtime_windows(axes[0,0])

    plt.tight_layout()
    plt.savefig(Path.cwd() / 'zeugs' / 'plots' / 'flowtime_window.png', dpi=400)






def main_simfit():

    # only narrow flowtime windows
    windows_narrow = [
        ( 5, [6,7,8,9] ),
        ( 6, [9,10,11,12] ),
        ( 7, [11,12,13,14,15] ),
    ]
    # many more flowtimes
    windows_broad = [
        ( 5, list(range(6, 14)) ),
        ( 6, list(range(8, 15)) ),
        ( 7, list(range(10, 16)) ),
    ]
    # include lots of data
    windows_range = [
        ( 0, [10,15] ),
        ( 1, [10,15] ),
        ( 2, [10,15] ),
        ( 3, [10,15] ),
        ( 4, [10,15] ),
        ( 5, [10,15] ),
        ( 6, [10,15] ),
        ( 7, [   15] ),
        ( 8, [   15] ),
    ]
    # lots of data and full windows for high tau
    windows_range2 = [
        ( 0, [10,14] ),
        ( 1, [10,14] ),
        ( 2, [10,14] ),
        ( 3, [10,14] ),
        ( 4, [10,14] ),
        ( 5, [6,8,10] ),
        ( 6, [8,10,12] ),
        ( 7, [11,13,15] ),
        ( 8, [13,15,17] )
    ]
    flowtime_windows = windows_range2

    # labels for each flowtime-tau combination
    labels = {}
    for tau, iflows in flowtime_windows:
        for iflow in iflows:
            rf = flowtime_to_radius(flowtimes[iflow], nt) * nt
            labels[iflow, tau] = f'rf={rf:.2f}a, tau={tau}a'


    # some parameters
    reltol       = 0.025
    sn_cut_fit   = 10
    sn_cut_sum   = 2
    max_bin_size = 10
    ex_max_seq   = [(0,0), (0,1), (0,2)]#, (0,3), (0,4)]



    # preparing unbinned data
    dist     = radial_separations(ns)
    ense_all = get_data_unbinned()



    # the whole routine
    def fit_then_sum_procedure(ense : np.ndarray) -> Any:
        fit, ense_binned, r_lims = iterative_fit(
            dist         = dist,
            ense         = ense,
            labels       = labels,
            reltol       = reltol,
            sn_cut       = sn_cut_fit,
            max_bin_size = max_bin_size,
            ex_mats_seq  = ex_max_seq
        )
        tsums, psums, r_cuts_sum = sums_over_r_linear(dist, ense, labels, r_lims, fit, sn_cut_sum)
        return fit, ense_binned, r_lims, tsums, psums, r_cuts_sum





    ###############################################################################
    #####################  fit and sum for bootstrap samples  #####################
    ###############################################################################
    if 0:
        print('starting...\n')


        @dataclasses.dataclass
        class FitAndSumData:
            '''collection of data resulting from fitting and summing'''

            chi2 : float
            dof  : int
            Q    : float

            p     : gv.BufferDict
            psums : dict[Any, np.ndarray]
            tsums : dict[Any, float]

            ense_binned : dict[Any, np.ndarray]
            r_lims      : dict[Any, tuple[float, float]]
            r_cuts_sum  : dict[Any, float]



        bootstrap_results = []
        n_bootstrap       = 10

        for i_bs, ense_bs in enumerate(gv.dataset.bootstrap_iter(ense_all, n_bootstrap)):
            print(f'\n\n\nStarting bootstraps sample {i_bs+1}/{n_bootstrap}...\n')
            fit, ense_binned, r_lims, tsums, psums, r_cuts_sum = fit_then_sum_procedure(ense_bs)
            bootstrap_results.append(FitAndSumData(
                chi2 = fit.chi2, dof = fit.dof, Q = fit.Q,
                p = fit.p, psums = psums, tsums = tsums,
                ense_binned = ense_binned, r_lims = r_lims, r_cuts_sum = r_cuts_sum
            ))







    ###############################################################################
    ################  fit and sum for original dataset (ense_all)  ################
    ###############################################################################
    if 0:
        print('starting...\n')

        fit, ense_binned, r_lims, tsums, psums, r_cuts_sum = fit_then_sum_procedure(ense_all)


        # do all the plotting of tails
        synchro = True
        nrows = len(set().union(*(set(iflows) for _,iflows in flowtime_windows))) if synchro else max(len(iflows) for _,iflows in flowtime_windows)
        ncols = len(flowtime_windows)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
        axes = np.reshape(axes, shape=(nrows, ncols))
        
        plot_flowtime_and_tau_fits(dist, labels, r_lims, fit, synchro, axes, max_r=40, r_cuts_sum=r_cuts_sum)

        fig.tight_layout()
        fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'simplot_prior.png', dpi=400)


        # other plots
        nrows2, ncols2 = 2, 1
        fig2, axes2 = plt.subplots(nrows=nrows2, ncols=ncols2, figsize=(7*ncols2, 4*nrows2))
        axes2 = np.reshape(axes2, shape=(nrows2, ncols2))

        plot_sums_over_r({(iflow, tau) : tsum for (iflow, tau), tsum in tsums.items() if tau in (5,6,7,8)}, axes2[0, 0], noerr=True)

        plot_corr_eigenvals(ense_binned, axes2[1, 0])

        fig2.tight_layout()
        fig2.savefig(Path.cwd() / 'zeugs' / 'plots' / 'other_plots.png', dpi=400)








main = main_simfit







###############################################################################
#########################  high level routines  ###############################
###############################################################################



def iterative_fit( dist   : np.ndarray, ense   : np.ndarray, labels       : dict[Any, str],
                   reltol : float,      sn_cut : float,      max_bin_size : float|None,
                   ex_mats_seq : list[tuple[int, int]]
                 ) -> tuple[lsqfit.nonlinear_fit, dict, dict]:
    '''...'''

    # preliminary constant size bins
    bins_c = find_distance_bins(dist, 1)
    bins0 = {idx : bins_c for idx in labels.keys()}

    # initial parameters initially determined solely through prior
    p0 = None

    # iteratively fit with increasing maximal excited states and matsubara modes
    for i, (ex_max, mats_max) in enumerate(ex_mats_seq, start=1):

        # bin data using previously determined bins
        dist_binned, data_binned, _, _ = bin_cut_avg_data(dist, ense, bins0, sn_cut_left=sn_cut)

        # do initial fit
        fit0 = fit_flowtime_and_tau_tails_with_prior(dist_binned, data_binned, ex_max, mats_max, p0=p0)

        # use fit to create more accurate bins, and bin data
        bins = bin_through_simultaneous_fit(dist, ense, labels, fit0, reltol, max_bin_size)
        dist_binned, data_binned, ense_binned, r_lims = bin_cut_avg_data(dist, ense, bins, sn_cut_left=sn_cut)

        # do the actual fit
        fit = fit_flowtime_and_tau_tails_with_prior(dist_binned, data_binned, ex_max, mats_max, p0=p0)

        # print fit results
        banner = f'XXXXXXXXXXXXXXXX  fit {i}: {ex_max=} and {mats_max=}  XXXXXXXXXXXXXXXX'
        print('X' * len(banner))
        print(banner)
        print('X' * len(banner))
        print(fit)

        # set initial parameters and bins for next iteration
        if fit.Q > 0.05:
            p0    = fit.pmean
            # bins0 = bins

    print()
    return fit, ense_binned, r_lims






###############################################################################
########################  simultaneous fitting  ###############################
###############################################################################




def fit_flowtime_and_tau_tails_with_prior(dist : dict[Any, np.ndarray], data : dict[Any, np.ndarray], excited_max : int, mats_max : int, p0 : dict|None = None, corr : bool = True) -> lsqfit.nonlinear_fit:
    '''fit with priors'''

    iflows = sorted(iflow for iflow,_ in data.keys())
    prior  = make_prior_constr(excited_max, mats_max, iflows)

    def fitfcn(x, p):
        y = {}
        for idx in x.keys():
            iflow, tau = idx
            y[idx] = expx_single_tau(x[idx], p, iflow, tau, excited_max, mats_max)
        return y
    
    return (
        lsqfit.nonlinear_fit( data =(dist, data), fcn=fitfcn, prior=prior, p0=p0 )
        if corr else
        lsqfit.nonlinear_fit( udata=(dist, data), fcn=fitfcn, prior=prior, p0=p0 )
    )







# ground state and first 2 excited 0-+ glueball state masses (in units of T) (in continuum R^3 SU(3))
masses_0p = [6.24, 8.8, 11.04]

# matsubara frequencies (in units of T)
p_mats = lambda mats : mats * 2 * np.pi

# rough estimate of higher matsubara mass
m_mats = lambda m, mats: np.sqrt( m**2 + p_mats(mats)**2 )




def make_prior(excited_max : int, mats_max : int, iflowtimes : list[int]) -> gv.BufferDict:
    '''constructs prior'''

    if excited_max > 2:
        raise Exception('Not yet implemented for excited_max > 2.')

    # now build prior
    prior = gv.BufferDict()

    for n_ex, m_0p in enumerate(masses_0p[:excited_max+1]):
        for mats in range(mats_max+1):

            # mass (flowtime idependent)
            m_guess = m_mats(m_0p, mats)
            prior[f'm_{n_ex}_{mats}'] = gv.gvar(m_guess, m_guess * 0.95)

            # amplitudes (flowtime dependent)
            for ift in iflowtimes:
                prior[f'a_{ift}_{n_ex}_{mats}'] = gv.gvar(-1, 100)

    return prior





def make_prior_mats_constr(mats : int, excited_max : int, iflowtimes : list[int]) -> gv.BufferDict:
    '''constructs prior for single matsubara modes (constraining masses). todo: add e0 < e1 < ... constraint on excited energies'''

    if excited_max > 2:
        raise Exception('Not yet implemented for excited_max > 2.')
    
    # add distribution to reparametrize masses to enforce constraint on higher matsubara mode masses
    if not gv.BufferDict.has_distribution(f'f_p{mats}'):
        gv.BufferDict.add_distribution(
            f'f_p{mats}',
            lambda x: p_mats(mats) + gv.exp(x)
        )

    # gv.BufferDict instead of python dict for custom distribution support
    prior = gv.BufferDict()

    # prior for lowest mass
    m_guess = m_mats(masses_0p[0], mats)
    prior[f'f_p{mats}(m_0_{mats})'] = gv.log(gv.gvar(m_guess, m_guess*1) - p_mats(mats))

    # priors for higher masses
    for n_ex in range(1, excited_max+1):
        dm_guess = m_mats(masses_0p[n_ex], mats) - m_mats(masses_0p[n_ex-1], mats)
        prior[f'log(dm_{n_ex}_{mats})'] = gv.log(gv.gvar(dm_guess, dm_guess*1))
        # prior[f'u(dm_{n_ex}_{mats})'] = gv.BufferDict.uniform('u', 0, 5)

    # priors for all the amplitudes
    for n_ex in range(excited_max+1):
        for ift in iflowtimes:
            prior[f'log(a_{ift}_{n_ex}_{mats})'] = gv.log(gv.gvar(1, 50))

    return prior





def make_prior_many_mats(excited_max : int, mats_list : list[int], iflowtimes : list[int]) -> gv.BufferDict:
    '''...'''

    prior = gv.BufferDict()

    for mats in mats_list:
        prior.update(make_prior_mats_constr(mats, excited_max, iflowtimes))

    return prior




def make_prior_constr(excited_max : int, mats_max : int, iflowtimes : list[int]) -> gv.BufferDict:
    '''construct prior, incorporating constraints on masses'''

    prior = gv.BufferDict()

    for mats in range(mats_max+1):
        prior.update(make_prior_mats_constr(mats, excited_max, iflowtimes))

    return prior







def expx_single_tau(x : float, p : dict, iflow : int, tau : int, ex_max : int, mats_max : int) -> float:
    '''fit function with multiple excited states and matsubara modes'''

    return sum(
        2 * gv.cos(p_mats(mats) * tau/nt) * expx_single_mats(x, p, iflow, mats, ex_max)
        for mats in range(mats_max + 1)
    )




def expx_single_mats(x : float, p : dict, iflow : int, mats : int, ex_max : int) -> float:
    '''fit function for a single matsubara mode and possibly multiple excited states'''

    masses = [p[f'm_0_{mats}']]
    for n_ex in range(1, ex_max+1):
        masses.append(masses[-1] + p[f'dm_{n_ex}_{mats}'])

    return sum(
        expx(
            x / nt,
            - (nt/ns)**2 * masses[n_ex] * p[f'a_{iflow}_{n_ex}_{mats}'],
            masses[n_ex]
        )
        for n_ex in range(ex_max + 1)
    )







def plot_flowtime_and_tau_fits( dist : np.ndarray, labels : dict[Any, str], r_lims : dict[Any, tuple[float, float]], fit : lsqfit.nonlinear_fit, synchro : bool, axes : np.ndarray,
                                min_r : float|None = None, max_r : float|None = None, r_cuts_sum : dict[Any, float]|None = None ) -> None:
    '''plot all tails with data and fit, optionally synchro flowtimes across columns'''

    # map (iflow, tau) onto indices of grid of plots
    axes_idxs = {}
    for i_tau, tau in enumerate(sorted({tau for _,tau in labels.keys()})):
        for i_iflow, iflow in enumerate(sorted({iflow for iflow,tau0 in labels.keys() if synchro or tau0==tau})):
            axes_idxs[iflow, tau] = (i_iflow, i_tau)

    # plot original and fitted data
    for idx in labels.keys():
        iflow, tau = idx

        r_left, r_right = r_lims[idx]
        if r_right > dist[-1]:
            r_right *= 0.75 / 1.1
        if max_r is not None:
            r_right = max_r
        if min_r is not None:
            r_left = min_r
        ir_left, ir_right = index_from_distance(dist, r_left), index_from_distance(dist, r_right)

        y_fit = fit.fcn({idx : dist[ir_left:ir_right]}, fit.p)[idx]

        plot_dist(fit.x[idx], fit.y[idx], axes[axes_idxs[idx]])
        plot_fitfcn(dist[ir_left:ir_right], y_fit, axes[axes_idxs[idx]])

        if r_cuts_sum is not None:
            axes[axes_idxs[idx]].axvline(x=r_cuts_sum[idx], color='black', alpha=0.5)

        # y_min = min(min(gv.mean(y_fit)), min(gv.mean(fit.y[idx])))
        # y_max = max(max(gv.mean(y_fit)), max(gv.mean(fit.y[idx])))
        y_min = min(gv.mean(y_fit))
        y_max = max(gv.mean(y_fit))
        y_range = y_max - y_min
        axes[axes_idxs[idx]].set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        axes[axes_idxs[idx]].set_xlim(0 if min_r is None else min_r, r_right)
        axes[axes_idxs[idx]].set_title(labels[idx])

    





###############################################################################
###########################  summing over r  ##################################
###############################################################################


def sums_over_r_linear(dist : np.ndarray, ense : np.ndarray, labels : dict[Any, str], r_lims : dict[Any, tuple[float, float]], fit : lsqfit.nonlinear_fit, sn_cut_right : float) -> tuple[dict, dict, dict]:
    '''Computes partial sums, where first data, then linear interpolation of data and fit, and then fit is used.
    Returns total sums, partial sums, r_cuts.'''

    r_cuts_right = {}
    partial_sums = {}
    total_sums   = {}

    for idx in labels.keys():

        # fit function for this idx
        fitfcn = lambda x: fit.fcn({idx : x}, fit.pmean)[idx]

        # find ir_cut_left, until which only measured data is used
        r_cut_left  = r_lims[idx][0]
        ir_cut_left = index_from_distance(dist, r_cut_left)

        # find ir_cut_right, from where on only fit data is used
        sfit_ndata        = gv.gvar(fitfcn(fit.x[idx]), gv.sdev(fit.y[idx]))
        ir_cut_right_bin  = signal_to_noise_cut(sfit_ndata, sn_cut_right)
        r_cut_right       = fit.x[idx][ir_cut_right_bin]
        r_cuts_right[idx] = r_cut_right
        ir_cut_right      = index_from_distance(dist, r_cut_right)

        if ir_cut_right <= ir_cut_left:
            raise Exception(f'Bad r_cuts for {idx=}: {r_cut_left=} and {r_cut_right=}.')

        total_sums[idx], partial_sums[idx] = sum_lin(ense[:, *idx, :], fitfcn, r_cut_left, r_cut_right, pbins=0.5)

    return total_sums, partial_sums, r_cuts_right





def sum_lin(ense : np.ndarray, fcn : Callable, r_left : float, r_right : float, pbinsize : float) -> tuple[float, np.ndarray]:
    '''Computes sum of data in three parts s1, s2 and s3: s1 is only data, s2 is linear interpolation of data and fcn, and s3 is only fcn.
    Returns: total sum, partial sums at regular intervals.'''

    # prepare dist and its multiplicities
    dist    = radial_separations(ns)
    dr_list = radial_multiplicities_r(ns)

    # indicies of left and right r cutoffs
    ir_left  = index_from_distance(dist, r_left)
    ir_right = index_from_distance(dist, r_right)

    # segments of measured data
    data_left = ense[:, :ir_left].mean(axis=0)
    data_mid  = ense[:, ir_left:ir_right].mean(axis=0)

    # segments of fitted data
    fit_mid   = fcn(dist[ir_left:ir_right])
    fit_right = fcn(dist[ir_right:])

    # combine data and fit
    ratio = (dist[ir_left:ir_right] - r_left) / (r_right - r_left)
    data_mixed = np.concatenate((
        data_left,
        (1-ratio) * data_mid + ratio * fit_mid,
        fit_right
    ))

    # multiply by radial multiplicities, then sum, then convert from T^n to T^(n-3) units
    partial_sums_fine = np.cumsum(data_mixed * dr_list, axis=0) / nt**3

    # partial sums at regular intervals
    cbins    = find_distance_bins(dist, pbinsize)
    cbin_ils = [il for il,_ in cbins]
    partial_sums = partial_sums_fine[cbin_ils]

    return partial_sums_fine[-1], partial_sums






def sums_over_r_hardcut(dist : np.ndarray, ense : np.ndarray, labels : dict[Any, str], fit : lsqfit.nonlinear_fit, sn_cut : float) -> tuple[dict, dict, dict]:
    '''Computes partial sums, where fit function is used instead of data starting at ir_cut. Returns total sums, partial sums, r_cuts.'''

    dr_list  = radial_multiplicities_r(ns)

    cbins    = find_distance_bins(dist, 1)
    cbin_ils = [il for il,_ in cbins]

    r_cuts = {}
    psums  = {}
    tsums  = {}

    for idx in labels.keys():
        iflow, tau = idx

        # fit function for this idx
        fitfcn = lambda x: fit.fcn({idx : x}, fit.pmean)[idx]

        # find ir_cut, from where on s(fit)/n(data) < sn_cut
        sfit_ndata  = gv.gvar(fitfcn(fit.x[idx]), gv.sdev(fit.y[idx]))
        ir_cut_bin  = signal_to_noise_cut(sfit_ndata, sn_cut)
        r_cuts[idx] = fit.x[idx][ir_cut_bin]
        ir_cut      = index_from_distance(dist, r_cuts[idx])

        # construct data out of original data and fit data
        data_mixed = np.concatenate((
            ense[:, iflow, tau, :ir_cut].mean(axis=0),
            fitfcn(dist[ir_cut:])
        ))

        # multiply by radial multiplicities, then sum, then convert from T^8 to T^5 units
        psums_fine = np.cumsum(data_mixed * dr_list, axis=0) / nt**3
        
        # take only partial sum after every r=a step, including last
        psums[idx] = psums_fine[cbin_ils]

        # total sum
        tsums[idx] = psums_fine[-1]

    return tsums, psums, r_cuts





def plot_sums_over_r(tsums : dict, axes : plt.Axes, noerr : bool = True) -> None:
    '''Plot all the G_t(tau) values for all t and tau.'''

    tau_curves = {}
    for (iflow, tau), tsum in tsums.items():
        if tau in tau_curves: tau_curves[tau].append((iflow, tsum))
        else:                 tau_curves[tau] = [(iflow, tsum)]

    for tau, curve in tau_curves.items():
        if tau == 0: continue
        curve_sorted = sorted(curve, key=lambda v: v[0])
        iflow_list = [iflow for iflow,_ in curve_sorted]
        tsum_list  = [tsum  for _,tsum  in curve_sorted]

        axes.errorbar(
            x    = 8 * flowtimes[iflow_list] / tau**2,
            y    = tsum_list if noerr else gv.mean(tsum_list),
            yerr = None if noerr else gv.sdev(tsum_list),
            marker = 'o',
            label  = f'tau = {tau}a = {tau/nt:.2f}/T',
        )

    axes.axvline(x=(1/4)**2, color='red',    label='8t/tau^2 = (1/4)^2', alpha=0.75)
    axes.axvline(x=(1/3)**2, color='purple', label='8t/tau^2 = (1/3)^2', alpha=0.75)

    iflow_max = max(8 * flowtimes[iflow] / tau**2 for iflow,tau in tsums.keys() if tau != 0)
    tsum_max  = min(tsums.values())
    axes.set_xlim(0, iflow_max * 1.1)
    axes.set_ylim(tsum_max * 1.2, 0)

    axes.set_xlabel('8 t / tau^2')
    axes.set_ylabel('G_t(tau) / T^5')

    axes.legend()
    









###############################################################################
##############################  old stuff  ####################################
###############################################################################





def expx_mats(x : float, p : dict, iflow : int, tau : int, mats_max : int) -> float:
    '''Fit function incorporating multiple matsubara modes but only lowest energy for each.'''

    omega = lambda mats: 2 * np.pi * mats / nt

    return sum( expx(x, p['a', iflow, mats], p['m', mats]) * gv.cos(omega(mats) * tau)
                for mats in range(mats_max+1) )






def fit_flowtime_and_tau_tails(dist : dict[Any, np.ndarray], data : dict[Any, np.ndarray], mats_max : int) -> lsqfit.nonlinear_fit:
    '''Keys are (iflow, tau). Fit is to all tails simultaneous. Include matsubara modes up to mats_max. Only lowest mass for each matsubara mode.'''

    iflows = { iflow for iflow, tau in data.keys() }
    taus   = { tau   for iflow, tau in data.keys() }

    omega = lambda mats: 2 * np.pi * mats / nt

    p0 = {
        **{ ('a', iflow, mats) : -1.
            for iflow, mats in product(iflows, range(mats_max+1)) },     # one prefactor for each flowtime and matsubara mode
        **{ ('m', mats) : 0.5 + omega(mats)
            for mats in range(mats_max+1) }     # one mass for each matsubara mode
    }

    def fcn(x, p):
        y = {}
        for idx in x.keys():
            iflow, tau = idx
            y[idx] = expx_mats(x[idx], p, iflow, tau, mats_max)
        return y
    
    return lsqfit.nonlinear_fit(
        data = (dist, data),
        fcn  = fcn,
        p0   = p0,
        debug = True
    )







@dataclasses.dataclass
class TailFitData:

    bins       : np.ndarray             # bins for each r-series
    r_cuts_sn  : np.ndarray             # float for each r-series, corresponding to minimum r used in fit    

    fit        : lsqfit.nonlinear_fit   # fit object of combined fit

    r_cuts_sum : np.ndarray             # float for each r-series, corresponding to r from when on fit instead of data is used in sum
    psums      : np.ndarray             # partial sums (array of float) for each r-series (only for r integer)
    rsums      : float                  # G_F(tau) = sum over r  for each r-series






def do_tailfit_bootstrap(dist : np.ndarray, ense : np.ndarray, labels : np.ndarray, sn_cut : float) -> None:
    '''...'''

    results = []
    n_bs = 10

    for i, ense_bs in enumerate(gv.dataset.bootstrap_iter(ense, n_bs)):
        result_bs = do_tail_fit_and_sum(dist, ense_bs, labels, sn_cut)
        results.append(result_bs)
        print(f'\rDone {i+1}/{n_bs} bootstraps...', end='', flush=True)

    print()
    return results




def do_tail_fit_combined(dist : dict[Any, np.ndarray], data : dict[Any, np.ndarray], labels : dict[Any, str]) -> lsqfit.nonlinear_fit:
    '''...'''

    # some constants
    n_exp  = 1          # number of exponentials in fit
    a0, m0 = -1., 1.    # initial values for a and m in fit

    # the fit function
    def f(p):
        y = {}
        for idx in labels.keys():
            y[idx] = expx_fcn(dist[idx], p['a', idx], p['m'])
        return y
    
    # do the fit (debug = False faster..)
    fit = lsqfit.nonlinear_fit(
        data = data,
        fcn  = f,
        p0 = { **{('a', idx) : np.full(n_exp, a0) for idx in labels.keys()},
               'm' : np.linspace(m0, m0 * n_exp, n_exp)},     # one m for all r-series for now
        debug = True
    )

    return fit




def do_tail_fit_and_sum(dist : np.ndarray, ense : np.ndarray, labels : np.ndarray, sn_cut : float) -> TailFitData:
    '''dist:   list of r-separations
       ense:   data ensemble, ense.shape = (config, ..., distance)
       labels: labels for r-series, labels.shape = ense.shape[1:-1]'''
    
    # some constants
    reltol = 0.05       # bin_error / data_error <= reltol
    
    # variable size bins
    bins = bin_multiple_series_through_fit(dist, ense, reltol)

    # get binned stuff, starting at sn_cut
    dist_binned, data_binned, r_cuts_sn, _ = bin_cut_avg_data(dist, ense, bins, sn_cut)

    # do the fit
    fit = do_tail_fit_combined(dist_binned, data_binned, labels)

    # determine points from where on in the sum fit data is used instead of real data
    ir_cuts_sum = {}
    r_cuts_sum  = {}
    for idx in labels.keys():
        ir_cut = index_from_distance(dist, r_cuts_sn[idx])    # for now
        ir_cuts_sum[idx] = ir_cut
        r_cuts_sum[idx]  = dist[ir_cut]

    # do sums (partial and total)
    data_mean = ense.mean(axis=0)
    for idx, ir_cut in ir_cuts_sum.items():
        data_mean[idx][ir_cut:] = expx_fcn(dist[ir_cut:], fit.pmean['a', idx], fit.pmean['m'])
    data_mean *= radial_multiplicities_r(ns)
    partial_sums = np.cumsum(data_mean, axis=-1) / nt   # divide by nt to go from T^6 to T^5 units
    total_sums   = partial_sums[..., -1].copy()

    # preparing the return value
    return TailFitData(
        bins,
        r_cuts_sn,
        fit,
        r_cuts_sum,
        partial_sums[..., [il for il, ir in find_distance_bins(dist, 1)]],
        total_sums
    )




def plot_tailfit_bootstrap(results : list[TailFitData], labels : np.ndarray, axes : np.ndarray) -> None:
    '''plot bs timelines'''

    x = list(range(len(results)))

    axes[0].scatter(
        x = x,
        y = [res.fit.Q for res in results],
        s = 5,
        label = 'Q'
    )
    axes[0].set_yscale('log')
    axes[1].scatter(
        x = x,
        y = [res.fit.chi2 / res.fit.dof for res in results],
        s = 5,
        label = 'chi2/dof'
    )
    # axes[2].scatter(
    #     x = x,
    #     y = [len(res.bins) for res in results],
    #     s = 5,
    #     label = '#bins'
    # )
    axes[2].errorbar(
        x    = x,
        y    = [res.fit.p['m'][0].mean for res in results],
        yerr = [res.fit.p['m'][0].sdev for res in results],
        marker = 'o',
        markersize = 5,
        linestyle = 'none',
        label = 'fitparam m'
    )
    for idx in labels.keys():
        axes[3].scatter(
            x = x,
            y = [res.r_cuts_sn[idx] for res in results],
            s = 5,
            label = f'ir_cut ({labels[idx]})'
        )

    for i in range(4):
        axes[i].set_xlabel('bootstrap sample')
        axes[i].legend()




def plot_tail_fits(dist : np.ndarray, ense : np.ndarray, result : TailFitData, labels : np.ndarray, axes : np.ndarray) -> None:
    '''plot tail fits'''

    r_mid = dist[index_from_distance(dist, dist[-1]/2)]

    for i, idx in enumerate(labels.keys()):

        # prepare data
        dist_b, ense_b = bin_averages(result.bins[idx], dist, ense[:, *idx, :])
        data_b = gv.dataset.avg_data(ense_b)

        # plot the fit
        ir_cut   = index_from_distance(dist, result.r_cuts_sn[idx])
        ir_cut_b = index_from_distance(dist_b, result.r_cuts_sn[idx])
        plot_dist(
            dist_b[ir_cut_b:],
            data_b[ir_cut_b:],
            axes[i]
        )
        plot_fitfcn(
            dist[ir_cut:],
            expx_fcn(dist[ir_cut:], result.fit.p['a', idx], result.fit.p['m']),
            axes[i]
        )
        axes[i].set_xlim(0, r_mid)
        axes[i].set_title(labels[idx])




def plot_tail_fit_and_sum(dist : np.ndarray, ense : np.ndarray, result : TailFitData, labels : np.ndarray, axes : np.ndarray) -> None:
    '''plot stuff'''

    r_mid = dist[index_from_distance(dist, dist[-1]/2)]

    for i, idx in enumerate(labels.keys()):

        # prepare data
        dist_b, ense_b = bin_averages(result.bins[idx], dist, ense[:, *idx, :])
        data_b = gv.dataset.avg_data(ense_b)

        # plot the fit
        ir_cut   = index_from_distance(dist, result.r_cuts_sn[idx])
        ir_cut_b = index_from_distance(dist_b, result.r_cuts_sn[idx])
        plot_dist(
            dist_b[ir_cut_b:],
            data_b[ir_cut_b:],
            axes[i, 0]
        )
        plot_fitfcn(
            dist[ir_cut:],
            expx_fcn(dist[ir_cut:], result.fit.p['a'][idx], result.fit.p['m']),
            axes[i, 0]
        )
        axes[i, 0].set_xlim(0, r_mid)
        axes[i, 0].set_title(labels[idx])

        # plot the partial sum
        r_ints = np.arange(0, result.psums.shape[-1], 1)
        axes[i, 1].scatter(
            x = r_ints,
            y = result.psums[idx],
            s = 5
        )
        axes[i, 1].set_xlabel('r / a')
        axes[i, 1].set_ylabel('partial sum / T^5') 
        axes[i, 1].set_title(labels[idx])       











def bin_data_through_fcn(dist : np.ndarray, ense : np.ndarray, fcn : Callable[[float], float], reltol : float, axes : plt.Axes|None = None) -> Bins:
    '''Bin data according to function.'''

    bins = bin_in_r_through_fcn_and_data(dist, ense, fcn, reltol)

    if axes is not None:
        cbin_size = 0.5
        sn_cut = 8

        dist_cbinned, ense_cbinned = bin_averages(find_distance_bins(dist, cbin_size), dist, ense)
        data_cbinned = gv.dataset.avg_data(ense_cbinned)
        ir_cut_c = signal_to_noise_cut(data_cbinned, sn_cut)
        plot_dist(dist_cbinned[ir_cut_c:], data_cbinned[ir_cut_c:], axes, alpha=0.25, color='blue', label='const bins')

        dist_vbinned, ense_vbinned = bin_averages(bins, dist, ense)
        data_vbinned = gv.dataset.avg_data(ense_vbinned)
        ir_cut_v = index_from_distance(dist_vbinned, dist_cbinned[ir_cut_c])
        plot_dist(dist_vbinned[ir_cut_v:], data_vbinned[ir_cut_v:], axes, color='purple', label='var bins')

        axes.set_title(f'Constant bins (Δr={cbin_size}a) and variable sized bins ({len(bins)} bins).')

    return bins


    







def vis_fcn_binning(dist : np.ndarray, data : np.ndarray, fcn : Callable[[float], float], tol : float, axes : plt.Axes) -> list[tuple[int, int]]:
    '''...'''

    bins = bin_in_r_through_fcn(dist, fcn, tol)
    plot_dist(dist, data, axes)
    for il, ir in bins:
        axes.axvline(x=dist[il], color='orange', alpha=0.75)
    return bins





def vis_correlations_in_r(dist : np.ndarray, data : np.ndarray, axes : plt.Axes, fig : plt.Figure) -> None:
    '''visualize correlations in r'''

    img = axes.imshow(gv.evalcorr(data), vmin=0, vmax=1)

    tick_positions = np.arange(0, len(dist), len(dist)//5)
    tick_labels    = [f'{dist[i]:.2f}' for i in tick_positions]

    axes.set_xticks(tick_positions, tick_labels, rotation=45)
    axes.set_yticks(tick_positions, tick_labels)

    axes.set_title(f'radius correlations')
    fig.colorbar(img, ax=axes, label='Correlation')






def plot_over_flowtime(tau : int, iflowtimes : np.ndarray, corrs : np.ndarray, axes : plt.Axes) -> None:
    '''...'''

    axes.errorbar(
        x    = [flowtimes[i] for i in iflowtimes],
        y    = gv.mean(corrs),
        yerr = gv.sdev(corrs),
        marker     = 'o',
        markersize = 2,
        linestyle  = 'none',
        linewidth  = 1,
        label      = f'tau = {tau}'
    )
    axes.set_xlim(0, 0.5)
    axes.set_ylim(-0.2, 0)
    axes.set_xlabel('flowtime / a^2')
    axes.set_ylabel('G_F(tau) / T^5')
    axes.set_title(f'G_F(tau, r) summed over r, as a function of flowtime at fixed tau')
    axes.legend()





def vis_flowtime_window(tau : int, axes : plt.Axes) -> None:
    '''...'''

    flowtimes_norm = flowtime_to_radius(flowtimes, nt) * nt / tau
    for ft in flowtimes_norm:
        axes.axvline(x=ft, color='blue', linestyle='--')
    axes.axvline(x=1/4, color='red', label='1/4', alpha=0.75)
    axes.axvline(x=1/3, color='red', label='1/3', alpha=0.75)
    axes.set_xlabel(f'r_F / a / tau  where  tau={tau}')
    axes.set_title('Blue: all flowtimes, Red: flowtime window (3/2 < r_F/a < tau/3)')
    axes.legend()




def vis_flowtime_windows(axes : plt.Axes) -> None:
    '''show all flowtimes plus min and max'''

    flowtimes_rf = [flowtime_to_radius(ft, nt) * nt for ft in flowtimes]

    for rf in flowtimes_rf:
        axes.axvline(x=rf, color='blue', linestyle='--')

    for tau in range(8+1):
        axes.axvline(x=tau/3, color='red', alpha=0.75)

    axes.axvline(x=1, color='violet')

    axes.set_xlabel(f'flowtime_radius / a')
    axes.set_title('violet: r=1, red: r=tau/2 for tau=0,...,8')
    # axes.legend()





def sum_over_r(dist : np.ndarray, data : np.ndarray, ddist : np.ndarray, a : list[gv.GVar], m : list[gv.GVar], r_cut : float, axes : plt.Axes|None) -> None:
    '''do partial sums'''

    ir_cut = min(i for i,r in enumerate(dist) if r >= r_cut)

    partial_sums = np.empty_like(data)
    for i, (r, v, dr) in enumerate(zip(dist, data, ddist)):
        prev = 0 if i==0 else partial_sums[i-1]
        if i < ir_cut:
            partial_sums[i] = prev + v * dr
        else:
            partial_sums[i] = prev + expx_fcn(r, a, m) * dr
    partial_sums /= nt

    if axes is not None:
        dist_sliced, psum_sliced = slice_in_r(dist, partial_sums, 1)
        axes.errorbar(
            x          = dist_sliced,
            y          = gv.mean(psum_sliced),
            yerr       = gv.sdev(psum_sliced),
            marker     = 'o',
            markersize = 2,
            linestyle  = 'none',
            linewidth  = 1,
            label      = 'partial sums'
        )
        axes.axvline(x=dist[ir_cut], alpha=0.5, color='orange', label='sn_cut')

        axes.set_xlabel('r / a')
        axes.set_ylabel('psum / T^5')
        axes.legend()

    return partial_sums[-1]
    







def get_data_binned(bin_size : float) -> np.ndarray:
    '''Bin dataset in r.'''

    skip_configs = 55
    return bin_by_distance_ensemble(
        distances = radial_separations(ns),
        ensemble  = np.load(Path.cwd() / 'zeugs' / 'data' / 'data0.npy')[skip_configs::1],
        bin_size  = bin_size
    )




def do_svd_cut(dataset : np.ndarray, axes : plt.Axes|None = None) -> np.ndarray:
    '''Make svd analysis, do svdcut, plot svd analysis.'''

    svd = gv.dataset.svd_diagnosis(dataset.reshape(dataset.shape[0], -1), nbstrap=100)
    data_cut = gv.svd(svd.avgdata, svdcut=svd.svdcut).reshape(*dataset.shape[1:])
    print('svdcut =', svd.svdcut)
    if axes is not None:
        svd.plot_ratio(plot = GVarAxesAdapter(axes))
        axes.set_title('SVD analysis of covariance matrix eigenvalues')
    return data_cut








def single_fit(dist : np.ndarray, data : np.ndarray, n_exp : int = 1) -> lsqfit.nonlinear_fit:
    '''...'''

    f = lambda x, p: expx_fcn(x, p['a'], p['m'])
    fit = lsqfit.nonlinear_fit(
        data = (dist, data),
        fcn  = f,
        p0   = {'a' : np.full(n_exp, -1.), 'm' : np.linspace(1, n_exp, n_exp)},
        debug = True
    )
    return fit




def combined_fit(dist : np.ndarray, data : np.ndarray, sn_cut : float, n_exp : int = 1) -> tuple[lsqfit.nonlinear_fit, list[int]]:
    '''...'''
    
    # determine ir_cuts
    ir_cuts = [signal_to_noise_cut(data_i, sn_cut) for data_i in data]
    if None in ir_cuts: raise Exception(f'Got None in ir_cuts: {ir_cuts}.')

    # prepare data (dict due to potentially unequal ir_cuts)
    data_dict = {}
    for i, (data_i, ir_cut) in enumerate(zip(data, ir_cuts)):
        data_dict[i] = data_i[ir_cut:]

    # fit function
    def f(p):
        y = {}
        for i, (a_vals, ir_cut) in enumerate(zip(p['a'], ir_cuts)):
            y[i] = expx_fcn(dist[ir_cut:], a_vals, p['m'])
        return y

    # do fit
    fit = lsqfit.nonlinear_fit(
        data = data_dict,
        fcn  = f,
        p0   = {'a' : np.full((data.shape[0], n_exp), -1.), 'm' : np.linspace(1, n_exp, n_exp)},
        debug = True
    )

    return fit, ir_cuts














def plot_fit(dist : np.ndarray, data : np.ndarray, i_cut : int, fit : lsqfit.nonlinear_fit, axes : plt.Axes) -> None:
    '''...'''

    il  = max(i_cut-3, 0)
    ir = -1

    # the data
    axes.errorbar(
        x          = dist[il:ir],
        y          = gv.mean(data[il:ir]),
        yerr       = gv.sdev(data[il:ir]),
        marker     = 'o',
        markersize = 2,
        linestyle  = 'none',
        linewidth  = 1,
        label      = 'data'
    )

    # the fit
    data_fit = fit.fcn(dist[il:ir], fit.p)
    axes.fill_between(
        x  = dist[il:ir],
        y1 = gv.mean(data_fit) - gv.sdev(data_fit),
        y2 = gv.mean(data_fit) + gv.sdev(data_fit),
        color = 'red',
        alpha = 0.75,
        label = 'fit',
    )

    # vertical line at ir_cut
    axes.axvline(x=dist[i_cut], alpha=0.75, color='orange', label='s/n cut')

    # other stuff
    axes.set_xlabel('r / a')
    axes.set_ylabel('G / T^6')
    axes.legend()




def plot_fitc(dist : np.ndarray, data : np.ndarray, i_cuts : list[int], fit : lsqfit.nonlinear_fit, tau : int, ifts : list[int], axes : list[plt.Axes]) -> None:
    '''....'''

    il  = max(min(i_cuts), 0)
    ir = -1

    for i, (ax, ift) in enumerate(zip(axes, ifts)):
        ax.errorbar(
            x          = dist[il:ir],
            y          = gv.mean(data[i, il:ir]),
            yerr       = gv.sdev(data[i, il:ir]),
            marker     = 'o',
            markersize = 2,
            linestyle  = 'none',
            linewidth  = 1,
            label      = 'data'
        )
        data_fit = expx_fcn(dist[il:ir], fit.p['a'][i], fit.p['m'])
        ax.fill_between(
            x  = dist[il:ir],
            y1 = gv.mean(data_fit) - gv.sdev(data_fit),
            y2 = gv.mean(data_fit) + gv.sdev(data_fit),
            color = 'red',
            alpha = 0.75,
            label = 'fit',
        )
        ax.axvline(x=dist[i_cuts[i]], alpha=0.75, color='orange', label='s/n cut')
        ax.set_xlabel('r / a')
        ax.set_ylabel('G / T^6')
        ax.set_ylim(-0.0005, 0.0002)
        ax.set_title(f'Flowtime t_F={flowtimes[ift]:.2f}a^2 (r_F={flowtime_to_radius(flowtimes[ift], nt)*nt:.2f}a)  and  tau={tau}')
        ax.legend()


    





































if __name__ == '__main__':

    main()