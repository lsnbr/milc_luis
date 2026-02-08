from pathlib import Path
import dataclasses
import numpy as np
import gvar as gv
import lsqfit
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flowing import flowtime_to_radius, radius_to_flowtime
from measurements import *
from statana import *
from ana_tail import make_prior_constr, expx_single_mats, plot_flowtime_and_tau_fits, p_mats, make_prior_many_mats, sum_lin




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
    iflow_o = 12
    print(f't_f = {flowtimes[iflow_o]} a^2, r_f = {flowtime_to_radius(flowtimes[iflow_o], nt) * nt} a')



    ###################################################################
    ########################  prepare data  ###########################
    ###################################################################
    
    ense_m0 = ense_all[:, iflow_o, 0, :]
    ense_m1 = ense_all[:, iflow_o, 1, :]
    ense_m2 = ense_all[:, iflow_o, 2, :]

    ense_m0_b, ense_m1_b, ense_m2_b = bin_averages(bins, ense_m0, ense_m1, ense_m2)

    data_m0 = gv.dataset.avg_data(ense_m0_b)
    data_m1 = gv.dataset.avg_data(ense_m1_b)
    data_m2 = gv.dataset.avg_data(ense_m2_b)



    # compute integrand for n=0,1,2 for each ensemble
    ense_m0_int = build_integrand(dist, ense_all[:, iflow_o, 0, :], 0)
    ense_m1_int = build_integrand(dist, ense_all[:, iflow_o, 1, :], 1)
    ense_m2_int = build_integrand(dist, ense_all[:, iflow_o, 2, :], 2)

    # unbinned subtracted
    ense_sub_unbinned = mats012_subtraction(ense_m0_int, ense_m1_int, ense_m2_int)

    # do binning
    ense_m0_int_b, ense_m1_int_b, ense_m2_int_b = bin_averages(bins, ense_m0_int, ense_m1_int, ense_m2_int)

    # ensemble average
    data_m0_int = gv.dataset.avg_data(ense_m0_int_b)
    data_m1_int = gv.dataset.avg_data(ense_m1_int_b)
    data_m2_int = gv.dataset.avg_data(ense_m2_int_b)

    # do subtractions
    data_sub_int = mats012_subtraction(data_m0_int, data_m1_int, data_m2_int)



    ###################################################################
    ########################  plot raw data  ##########################
    ###################################################################

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
    axes = np.reshape(axes, shape=(nrows, ncols))

    # left and right limits for different bin_sizes
    il0, ir0 = {0.5 : (0, 21), 0.25 : (0,  38)}[bin_size]
    il1, ir1 = {0.5 : (4, 21), 0.25 : (6,  32)}[bin_size]
    il2, ir2 = {0.5 : (0, 22), 0.25 : (0,  40)}[bin_size]
    il3, ir3 = {0.5 : (12, 25), 0.25 : (21, 53)}[bin_size]

    # plot integrands of individual matsubara frequencies
    plot_dist_many(dist_binned, [data_m0_int, data_m1_int, data_m2_int], il0, ir0, ['n=0', 'n=1', 'n=2'], axes[0,1])
    axes[0,1].set_ylabel('G * sinh/r / T^7')

    plot_dist_many(dist_binned, [data_m0_int, data_m1_int, data_m2_int], il1, ir1, ['n=0', 'n=1', 'n=2'], axes[1,1])
    axes[1,1].set_ylabel('G * sinh/r / T^7')

    # plot subtracted modes
    plot_dist_many(dist_binned, [data_sub_int], il0, ir0, ['sub012'], axes[0,2])
    axes[0,2].set_ylabel('G * sinh/r / T^7')

    plot_dist_many(dist_binned, [data_sub_int], il1, ir1, ['sub012'], axes[1,2])
    axes[1,2].set_ylabel('G * sinh/r / T^7')

    # plot pure matsubara modes
    plot_dist_many(dist_binned, [data_m0, data_m1, data_m2], il2, ir2, ['n=0', 'n=1', 'n=2'], axes[0,0])
    axes[0,0].set_ylabel('G(w_n, r) / T^7')

    plot_dist_many(dist_binned, [data_m0, data_m1, data_m2], il3, ir3, ['n=0', 'n=1', 'n=2'], axes[1,0])
    axes[1,0].set_ylabel('G(w_n, r) / T^7')


    # test correlations
    ense_m012 = ense_all[:, iflow_o, 0:3, :]
    ense_m012_b, = bin_averages(bins, ense_m012)
    data_m012 = gv.dataset.avg_data(ense_m012_b)
    corr_m012 = gv.evalcorr(data_m012)
    
    # axes[2,0].imshow(corr_m012[0,:, 0,:], vmin=0, vmax=1)
    # axes[2,0].set_title('n=0 n=0 correlations')
    # axes[2,1].imshow(corr_m012[1,:, 1,:], vmin=0, vmax=1)
    # axes[2,1].set_title('n=1 n=1 correlations')
    # axes[2,2].imshow(corr_m012[2,:, 2,:], vmin=0, vmax=1)
    # axes[2,2].set_title('n=2 n=2 correlations')

    axes[2,0].imshow(corr_m012[0,:, 1,:], vmin=0, vmax=1)
    axes[2,0].set_title('n=0 n=1 correlations')
    axes[2,1].imshow(corr_m012[1,:, 2,:], vmin=0, vmax=1)
    axes[2,1].set_title('n=1 n=2 correlations')
    axes[2,2].imshow(corr_m012[0,:, 2,:], vmin=0, vmax=1)
    axes[2,2].set_title('n=0 n=2 correlations')


    # finalize and save plots
    fig.tight_layout()
    fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'gridmats.png', dpi=400)





    ###################################################################
    ################  understand single mats fits  ####################
    ###################################################################
    if 1:
        print('sigle mats fit analysis...')



        @dataclasses.dataclass
        class FitStuff:
            '''all kind of stuff for and from fit, for viewing and plotting results'''

            labels : dict[Any, str]
            dist   : dict[Any, np.ndarray]
            data   : dict[Any, np.ndarray]
            ense   : dict[Any, np.ndarray]
            rlims  : dict[Any, tuple[float, float]]
            fit    : lsqfit.nonlinear_fit



        # constant bins used in 'const' and 'var' modes
        cbin_size = 0.2
        cbins     = find_distance_bins(dist, cbin_size)

        def do_fits_for_many_r_min_lefts(r_min_left_list : Iterable[float], iflows : Iterable[int], mats : int, mode : str) -> list[FitStuff]:
            '''one fit for each r_min_left'''

            fitstuff_list = []
            labels = {(iflow, mats) : f'rf={flowtime_to_radius(flowtimes[iflow],nt)*nt:.2f}a, n={mats}' for iflow in iflows}

            for r_min_left in r_min_left_list:
                print(f'\n{r_min_left = :.2f}\n')

                if mode == 'const':
                    dist_fit, data_fit, ense_fit, rlims_fit = bin_cut_avg_data(dist, ense_all, {(iflow, mats) : cbins for iflow in iflows}, r_min_left=r_min_left, r_max_right=30)

                if mode == 'var':
                    dist_fit0, data_fit0, _, _ = bin_cut_avg_data(dist, ense_all, {(iflow, mats) : cbins for iflow in iflows}, r_min_left=r_min_left, r_max_right=None)
                    fit0 = fit_flowtime_and_mats_tails_with_prior(dist_fit0, data_fit0, excited_max=0, mats_list=[mats], corr=False)
                    vbins = bin_through_simultaneous_fit(dist, ense_all, labels, fit0, reltol=0.01, max_bin_size=5)
                    dist_fit, data_fit, ense_fit, rlims_fit = bin_cut_avg_data(dist, ense_all, vbins, r_min_left=r_min_left, r_max_right=None)

                fit = fit_flowtime_and_mats_tails_with_prior(dist_fit, data_fit, excited_max=0, mats_list=[mats])
                fitstuff_list.append(FitStuff(labels, dist_fit, data_fit, ense_fit, rlims_fit, fit))
                print(fit)
                
            return fitstuff_list



        def plot_fitp_and_Q(r_list : Iterable[float], fitstuff_list : Iterable[FitStuff], pname : str, ax : plt.Axes, labelx : str|None = None, **plt_kwargs : Any) -> None:
            '''plots fit paramater and Q in same axes'''

            ax.errorbar(
                x    = r_list,
                y    = [fitstuff.fit.p[pname].mean for fitstuff in fitstuff_list],
                yerr = [fitstuff.fit.p[pname].sdev for fitstuff in fitstuff_list],
                marker    = 'o',
                linestyle = 'none',
                label     = 'm/T' + ('' if labelx is None else f' ({labelx})'),
                **plt_kwargs
            )
            ax.set_xlabel('min r/a used in fit')
            ax.set_ylabel('m/T')

            ax_twin = ax.twinx()
            ax_twin.plot(
                r_list,
                [fitstuff.fit.Q for fitstuff in fitstuff_list],
                marker    = 's',
                linestyle = '-',
                alpha     = 0.5,
                label     = 'Q' + ('' if labelx is None else f' ({labelx})'),
                **plt_kwargs
            )
            ax_twin.set_ylabel('Q')
            ax_twin.set_ylim(0, 1)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best')



        # set up plots
        nrows, ncols = 3, 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
        axes = np.reshape(axes, shape=(nrows, ncols))


        # some parameters for fits
        iflows = [6, 8, 10, 12, 14]
        r_min_left_list_mats = (
            np.arange(5, 15+1e-6, 1/3),
            np.arange(5, 12+1e-6, 1/3),
            np.arange(5, 10+1e-6, 1/3),
        )


        # comparison of fits to individual flowtimes
        if 0:

            for mats in (0, 1, 2):
                r_min_left_list = r_min_left_list_mats[mats]

                cmap = plt.colormaps['plasma']
                colors = cmap(np.linspace(0, 1, len(iflows)))

                for iflow, color in zip(iflows, colors):
                    fitstuff_list = do_fits_for_many_r_min_lefts(r_min_left_list, [iflow], mats, 'var')
                    plot_fitp_and_Q(r_min_left_list, fitstuff_list, f'm_0_{mats}', axes[mats,1], labelx=f'{iflow}', color=color)


        # fits to multiple flotimes combined
        if 1:

            fitstuff_list_mats = []
            for mats in (0, 1, 2):
                r_min_left_list = r_min_left_list_mats[mats]

                fitstuff_list = do_fits_for_many_r_min_lefts(r_min_left_list, iflows, mats, 'var')
                fitstuff_list_mats.append(fitstuff_list)
                print(', '.join(str(fitstuff.fit.p[f'm_0_{mats}']) for fitstuff in fitstuff_list))
                plot_fitp_and_Q(r_min_left_list, fitstuff_list, f'm_0_{mats}', axes[mats,0])


        fig.tight_layout()
        fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'mats_fit_ana.png', dpi=400)




        # now extract first good fit for each mats
        # we choose the one where  Q >= 0.75 Q_max  for the first time
        fitstuff_mats = []
        rleft_mats    = []
        for fitstuff_list, rleft_list in zip(fitstuff_list_mats, r_min_left_list_mats):
            Q_max = max(fitstuff.fit.Q for fitstuff in fitstuff_list)
            for fitstuff, rleft in zip(fitstuff_list, rleft_list):
                if fitstuff.fit.Q >= 0.75 * Q_max:
                    fitstuff_mats.append(fitstuff)
                    rleft_mats.append(rleft)
                    break
        print(rleft_mats)



        # construct functions from fits
        fcn_mats   = lambda ifl, mats, x: fitstuff_mats[mats].fit.fcn({(ifl,mats) : x}, fitstuff_mats[mats].fit.p)[ifl,mats]
        fcn_sinh   = lambda ifl, mats, x: build_integrand(x, fcn_mats(ifl, mats, x), mats)
        fcn_sub01  = lambda ifl,       x: mats01_subtraction(*(fcn_sinh(ifl, mats, x) for mats in (0,1)))
        fcn_sub012 = lambda ifl,       x: mats012_subtraction(*(fcn_sinh(ifl, mats, x) for mats in (0,1,2)))



        # plot pure mats fits
        nrows, ncols = len(iflows)+1, 3
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
        axes = np.reshape(axes, shape=(nrows, ncols))

        for mats in (0,1,2):
            plot_flowtime_and_tau_fits(dist, fitstuff_mats[mats].labels, fitstuff_mats[mats].rlims, fitstuff_mats[mats].fit, synchro=True, axes=axes[:,mats:mats+1], max_r=30)
            plot_corr_eigenvals(fitstuff_mats[mats].ense, axes[-1, mats])

        fig.tight_layout()
        fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'mats_single_fits.png', dpi=400)



        # plot integrands using fits
        ...





        # plot subtractions of 01 and 012
        # and their summations

        nrows, ncols = len(iflows), 4
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
        axes = np.reshape(axes, shape=(nrows, ncols))

        cbin_size = 0.25
        cbins     = find_distance_bins(dist, cbin_size)
        dist_b,   = bin_averages(cbins, dist)

        rleft_sub01  = max(rleft_mats[0:1+1])
        rleft_sub012 = max(rleft_mats[0:2+1])
        ileft_sub01  = index_from_distance(dist_b, rleft_sub01)
        ileft_sub012 = index_from_distance(dist_b, rleft_sub012)

        for iiflow, iflow in enumerate(iflows):

            # prepare data
            ense_sub01  = mats01_subtraction(*(build_integrand(dist, ense_all[:, iflow, mats, :], mats) for mats in (0,1)))
            ense_sub012 = mats012_subtraction(*(build_integrand(dist, ense_all[:, iflow, mats, :], mats) for mats in (0,1,2)))
            ense_sub01_b, ense_sub012_b = bin_averages(cbins, ense_sub01, ense_sub012)
            data_sub01_b  = gv.dataset.avg_data(ense_sub01_b)
            data_sub012_b = gv.dataset.avg_data(ense_sub012_b)

            # plot subtractions plus fit
            plot_dist(dist_b, data_sub01_b, axes[iiflow, 0], label='01sub')
            plot_fitfcn(dist_b[ileft_sub01:], fcn_sub01(iflow, dist_b[ileft_sub01:]), axes[iiflow, 0])
            axes[iiflow, 0].set_xlim(3, 14)
            axes[iiflow, 0].set_ylim(-0.5, 1)
            plot_dist(dist_b, data_sub012_b, axes[iiflow, 2], label='012sub')
            plot_fitfcn(dist_b[ileft_sub012:], fcn_sub012(iflow, dist_b[ileft_sub012:]), axes[iiflow, 2])
            axes[iiflow, 2].set_xlim(3, 14)
            axes[iiflow, 2].set_ylim(-1.5, 3)

            # find right r for use in summing (such that sn is 2 times worse than at rleft)
            _, rright_sub01  = find_rright_where_sn_worse_than_rleft(dist, ense_sub01, (lambda x: gv.mean(fcn_sub01(iflow, np.array([x]))[0])), rleft_sub01, 0.2)
            _, rright_sub012 = find_rright_where_sn_worse_than_rleft(dist, ense_sub012, (lambda x: gv.mean(fcn_sub012(iflow, np.array([x]))[0])), rleft_sub012, 0.2)
            # rright_sub01  = rleft_sub01 + 2
            # rright_sub012 = rleft_sub012 + 1

            # do sums
            pbinsize = 0.5
            tsum_sub01, psums_sub01 = sum_lin(ense_sub01, (lambda x: gv.mean(fcn_sub01(iflow, x))), rleft_sub01, rright_sub01, pbinsize)
            print(f'total sum (sub01) = {tsum_sub01:.2f} T^4')
            tsum_sub012, psums_sub012 = sum_lin(ense_sub012, (lambda x: gv.mean(fcn_sub012(iflow, x))), rleft_sub012, rright_sub012, pbinsize)
            print(f'total sum (sub012) = {tsum_sub012:.2f} T^4')

            # plot partial sums
            dist_psums = bin_distances(ns, pbinsize)     
            plot_dist(dist_psums, psums_sub01, axes[iiflow, 1], markersize=4, linestyle='--')
            axes[iiflow, 1].set_ylabel('partial sums (sub01) / T^4')
            axes[iiflow, 1].axvline(x=rleft_sub01,  color='black', alpha=0.5)
            axes[iiflow, 1].axvline(x=rright_sub01, color='black', alpha=0.5)
            axes[iiflow, 1].set_xlim(0, 25)
            plot_dist(dist_psums, psums_sub012, axes[iiflow, 3], markersize=4, linestyle='--')
            axes[iiflow, 3].set_ylabel('partial sums (sub012) / T^4')
            axes[iiflow, 3].axvline(x=rleft_sub012,  color='black', alpha=0.5)
            axes[iiflow, 3].axvline(x=rright_sub012, color='black', alpha=0.5)
            axes[iiflow, 3].set_xlim(0, 25)



        fig.tight_layout()
        fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'mats_sub_fits_sums.png', dpi=400)





    ###################################################################
    ####################  fit single mats mode  #######################
    ###################################################################
    if 0:
        print('do individual mats fits...')

        iflows_fit   = [8,10,12]
        r_min_left   = 6
        excited_max  = 2
        reltol       = 0.025
        max_bin_size = 5

        labels_mats      = []
        dist_fits_mats   = []
        data_fits_mats   = []
        ense_fits_mats   = []
        r_lims_fits_mats = []
        fits_mats        = []

        for mats in (0,1,2):
            print(f'Fit for mats = {mats}:\n')

            labels = {}
            for ifl in iflows_fit:
                rf = flowtime_to_radius(flowtimes[ifl], nt) * nt
                labels[ifl, mats] = f'rf={rf:.2f}a, n={mats}'
            labels_mats.append(labels)

            dist_fit0, data_fit0, ense_fit0, r_lims_fit0 = bin_cut_avg_data(dist, ense_all, {idx : bins for idx in labels.keys()}, r_min_left=r_min_left)
            fit0 = fit_flowtime_and_mats_tails_with_prior(dist_fit0, data_fit0, excited_max=excited_max, mats_list=[mats])
            print(fit0)

            bins_custom = bin_through_simultaneous_fit(dist, ense_all, labels, fit0, reltol=reltol, max_bin_size=max_bin_size)
            dist_fit, data_fit, ense_fit, r_lims_fit = bin_cut_avg_data(dist, ense_all, bins_custom, r_min_left=r_min_left)
            fit = fit_flowtime_and_mats_tails_with_prior(dist_fit, data_fit, excited_max=excited_max, mats_list=[mats])
            print(fit)

            dist_fits_mats.append(dist_fit),
            data_fits_mats.append(data_fit)
            ense_fits_mats.append(ense_fit)
            r_lims_fits_mats.append(r_lims_fit)
            fits_mats.append(fit)


        # plotting
        nrows, ncols = len(iflows_fit)+1, 3
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
        axes = np.reshape(axes, shape=(nrows, ncols))

        for mats in (0,1,2):
            plot_flowtime_and_tau_fits(dist, labels_mats[mats], r_lims_fits_mats[mats], fits_mats[mats], synchro=True, axes=axes[:,mats:mats+1], max_r=30)
            plot_corr_eigenvals(ense_fits_mats[mats], axes[-1, mats])

        fig.tight_layout()
        fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'gridmats_single_fits.png', dpi=400)





    ###################################################################
    #################  fit and plot pure mats data  ###################
    ###################################################################
    if 0:
        print('do fits...')

        mats_flowtimes = [
            ( 0, [iflow_o, ] ),
            ( 1, [iflow_o, ] ),
            ( 2, [iflow_o, ] ),
        ]

        labels = {}
        for mats, iflows in mats_flowtimes:
            for iflow in iflows:
                rf = flowtime_to_radius(flowtimes[iflow], nt) * nt
                labels[iflow, mats] = f'rf={rf:.2f}a, n={mats}'

        r_min_left  = 6
        excited_max = 2

        dist_fit, data_fit, ense_fit, r_lims_fit = bin_cut_avg_data(dist, ense_all, {idx : bins for idx in labels.keys()}, r_min_left=r_min_left)
        # dist_fit, data_fit, ense_fit, r_lims_fit = bin_cut_avg_data(dist, ense_all, {idx : bins for idx in labels.keys()}, sn_cut_left=10)

        fit0 = fit_flowtime_and_mats_tails_with_prior(dist_fit, data_fit, excited_max=excited_max, mats_list=[0,1,2])
        print(fit0)

        bins2 = bin_through_simultaneous_fit(dist, ense_all, labels, fit0, reltol=0.025, max_bin_size=5)
        dist_fit, data_fit, ense_fit, r_lims_fit = bin_cut_avg_data(dist, ense_all, bins2, r_min_left=r_min_left)
        # dist_fit, data_fit, ense_fit, r_lims_fit = bin_cut_avg_data(dist, ense_all, bins2, sn_cut_left=10)

        fit = fit_flowtime_and_mats_tails_with_prior(dist_fit, data_fit, excited_max=excited_max, mats_list=[0,1,2])
        print(fit)


        # plot fits
        nrows, ncols = 3, 3
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
        axes = np.reshape(axes, shape=(nrows, ncols))

        plot_flowtime_and_tau_fits(dist, labels, r_lims_fit, fit, synchro=True, axes=axes, max_r=20)

        plot_corr_eigenvals(ense_fit, axes[2, 0])

        fig.tight_layout()
        fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'gridmats_fits.png', dpi=400)




    ###################################################################
    ###################  use fit for integrands  ######################
    ###################################################################
    if 0:
        print('use fit for integrands...')

        fcn_mats = lambda ifl, mats, x: fits_mats[mats].fcn({(ifl,mats) : x}, fits_mats[mats].p)[ifl,mats]
        fcn_sinh = lambda ifl, mats, x: build_integrand(x, fcn_mats(ifl, mats, x), mats)
        fcn_sub  = lambda ifl, x: mats012_subtraction(*(fcn_sinh(ifl, mats, x) for mats in (0,1,2)))
        

        nrows, ncols = len(iflows_fit), 5
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
        axes = np.reshape(axes, shape=(nrows, ncols))


        for iiflow_o, iflow_o in enumerate(iflows_fit):

            for mats in (0,1,2):
                i_right = [35, 26, 13][mats]
                ylim    = [(-0.42,0.02), (-0.52, 0.05), (-0.9, 0.2)][mats]

                dist = dist_fits_mats[mats][iflow_o, mats][0:i_right]
                data = data_fits_mats[mats][iflow_o, mats][0:i_right]
                plot_dist(
                    dist,
                    build_integrand(dist, data, mats),
                    axes[iiflow_o, mats],
                    label=f'n={mats} sinh'
                )
                plot_fitfcn(
                    dist,
                    fcn_sinh(iflow_o, mats, dist),
                    axes[iiflow_o, mats]
                )
                axes[iiflow_o, mats].set_ylim(*ylim)


            # do subtraction
            dist = dist_binned[5:25]
            data = data_sub_int[5:25]
            plot_dist(
                dist,
                data,
                axes[iiflow_o, 3],
                label='012sub data'
            )
            i_left = index_from_distance(dist, r_min_left)
            plot_fitfcn(
                dist[i_left:],
                fcn_sub(iflow_o, dist[i_left:]),
                axes[iiflow_o, 3]
            )
            axes[iiflow_o, 3].set_ylim(-1.5, 3)




    ###################################################################
    ############  do sum of (subtracted )integrand(s)  ################
    ###################################################################
    if 0:
        print('do summing...')


        for iiflow_o, iflow_o in enumerate(iflows_fit):
            print(f'iflow_o = {iflow_o}:')

            r_right_sum = dist[-1]
            for mats in (0,1,2):
                sfit_ndata = gv.gvar(
                    gv.mean(fcn_sinh(iflow_o, mats, fits_mats[mats].x[iflow_o, mats])),
                    gv.sdev(build_integrand(fits_mats[mats].x[iflow_o, mats], fits_mats[mats].y[iflow_o, mats], mats))
                )
                ir_right_sum_new = signal_to_noise_cut(sfit_ndata, sn_cut=2)
                r_right_sum_new  = fits_mats[mats].x[iflow_o, mats][ir_right_sum_new]
                print(f'n={mats}: r={r_right_sum_new} (ir={ir_right_sum_new})')
                if r_right_sum_new < r_right_sum:
                    r_right_sum = r_right_sum_new


            pbinsize = 0.5
            total_sum, partial_sums = sum_lin(
                ense     = ense_sub_unbinned,
                fcn      = lambda x: gv.mean(fcn_sub(iflow_o, x)),
                r_left   = r_min_left,
                r_right  = r_right_sum,
                pbinsize = pbinsize
            )
            dist_partial_sums = bin_distances(ns, pbinsize)
            print(f'total sum = {total_sum:.2f} T^5')


            i_left  = 0
            i_right = index_from_distance(dist_partial_sums, r_right_sum*3)
            plot_dist(
                dist = dist_partial_sums[i_left:i_right],
                data = partial_sums[i_left:i_right],
                axes = axes[iiflow_o, 4],
                markersize = 4,
                linestyle = '--',
            )
            axes[iiflow_o, 4].set_ylabel('partial sums / T^5')
            axes[iiflow_o, 4].axvline(x=r_min_left,  color='black', alpha=0.5)
            axes[iiflow_o, 4].axvline(x=r_right_sum, color='black', alpha=0.5)



        fig.tight_layout()
        fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'mats_fit_to_int.png', dpi=400)

        





    ###################################################################
    ##################  fit and plot integrands  ######################
    ###################################################################
    if 0:
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









def find_rright_where_sn_worse_than_rleft(dist : np.ndarray, ense : np.ndarray, fcn : Callable[[float], float], rleft : float, fac : float) -> tuple[int, float]:
    '''Find r, such that sn(r) <= fac * sn(rleft).
    Uses s/n where signal is from fcn and noise is from ense.'''

    ileft    = index_from_distance(dist, rleft)
    ense_sem = ense.std(axis=0, ddof=1) / np.sqrt(ense.shape[0])
    sn_left  = fcn(dist[ileft]) / ense_sem[ileft]

    for iright in range(ileft+1, len(dist)):
        sn_right = fcn(dist[iright]) / ense_sem[iright]
        if sn_right <= fac * sn_left:
            break

    return iright, dist[iright]










def fit_flowtime_and_mats_tails_with_prior(dist : dict[Any, np.ndarray], data : dict[Any, np.ndarray], excited_max : int, mats_list : list[int], p0 : dict|None = None, corr : bool = True) -> lsqfit.nonlinear_fit:
    '''fit with priors'''

    iflows = sorted(iflow for iflow,_ in data.keys())
    prior  = make_prior_many_mats(excited_max, mats_list, iflows)

    def fitfcn(x, p):
        y = {}
        for idx in x.keys():
            iflow, mats = idx
            y[idx] = expx_single_mats(x[idx], p, iflow, mats, excited_max)
        return y
    
    return (
        lsqfit.nonlinear_fit( data =(dist, data), fcn=fitfcn, prior=prior, p0=p0, debug=True )
        if corr else
        lsqfit.nonlinear_fit( udata=(dist, data), fcn=fitfcn, prior=prior, p0=p0, debug=True )
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
    '''computes integrand of H_E in units of T^7'''

    if mats == 0:
        res = data
    
    else:
        w = p_mats(mats)
        fac = np.empty_like(dist)
        if dist[0] == 0:
            fac[0] = 1
            fac[1:] = gv.sinh(w*dist[1:]/nt) / (w*dist[1:]/nt)
        else:
            fac = gv.sinh(w*dist/nt) / (w*dist/nt)
        res = data * fac

    return res







def get_mats012_subtraction(ense : np.ndarray) -> np.ndarray:
    '''computes m0 + 1/3 m1 - 4/3 m2, which removes 1/w^2 and 1/w^4 contributions'''

    return ense[:,:,0,:] + (1/3) * ense[:,:,1,:] + (-4/3) * ense[:,:,2,:]




def mats012_subtraction(d0 : float, d1 : float, d2 : float) -> float:
    '''subtraction'''

    return d0 + (1/3) * d1 + (-4/3) * d2


def mats01_subtraction(d0 : float, d1 : float) -> float:
    '''subtraction'''

    return d0 - d1

































if __name__ == '__main__':

    main2()
