from pathlib import Path
import pickle
import numpy as np
import gvar as gv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })

from flowing import flowtime_to_radius
from measurements import *
from statana import *
from ana_mats import FitStuff, FitSubSum, build_integrand, mats_subtraction, plot_dist_many



from ensemble_data import data_16_x_64_1p5Tc as data_ense

nt = data_ense['nt']
ns = data_ense['ns']
flowtimes = data_ense['flowtimes']







def main_visdata():

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

    # do binning
    ense_m0_int_b, ense_m1_int_b, ense_m2_int_b = bin_averages(bins, ense_m0_int, ense_m1_int, ense_m2_int)

    # ensemble average
    data_m0_int = gv.dataset.avg_data(ense_m0_int_b)
    data_m1_int = gv.dataset.avg_data(ense_m1_int_b)
    data_m2_int = gv.dataset.avg_data(ense_m2_int_b)

    # do subtractions
    data_sub_int = mats_subtraction((0,1,2), [data_m0_int, data_m1_int, data_m2_int])



    ###################################################################
    ########################  plot raw data  ##########################
    ###################################################################

    nrows, ncols = 4, 3
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
    # ense_m012_b, = bin_averages(bins, ense_m012)
    # data_m012 = gv.dataset.avg_data(ense_m012_b)
    data_m012 = gv.dataset.avg_data(ense_m012)
    corr_m012 = gv.evalcorr(data_m012)
    

    plot_distance_correlations(corr_m012[0,:, 0,:], dist, 0, 20, axes[2,0])
    axes[2,0].set_title('distance correlations (Matsubara mode 0)')
    plot_distance_correlations(corr_m012[1,:, 1,:], dist, 0, 20, axes[2,1])
    axes[2,1].set_title('distance correlations (Matsubara mode 1)')
    plot_distance_correlations(corr_m012[2,:, 2,:], dist, 0, 20, axes[2,2])
    axes[2,2].set_title('distance correlations (Matsubara mode 2)')

    plot_distance_correlations(corr_m012[0,:, 1,:], dist, 0, 20, axes[3,0])
    axes[3,0].set_title('distance correlations (Matsubara modes 0 and 1)')
    plot_distance_correlations(corr_m012[1,:, 2,:], dist, 0, 20, axes[3,1])
    axes[3,1].set_title('distance correlations (Matsubara modes 1 and 2)')
    plot_distance_correlations(corr_m012[0,:, 2,:], dist, 0, 20, axes[3,2])
    axes[3,2].set_title('distance correlations (Matsubara modes 0 and 2)')


    # finalize and save plots
    fig.tight_layout()
    fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'gridmats.png', dpi=400)







def main2():

    fss = FitSubSum(bs=False)



    nrows1, ncols1 = 3, 2
    fig1, axes1 = plt.subplots(nrows=nrows1, ncols=ncols1, figsize=(7*ncols1, 4*nrows1))
    axes1 = np.reshape(axes1, shape=(nrows1, ncols1))

    nrows2, ncols2 = len(fss.iflows), 4
    fig2, axes2 = plt.subplots(nrows=nrows2, ncols=ncols2, figsize=(7*ncols2, 4*nrows2))
    axes2 = np.reshape(axes2, shape=(nrows2, ncols2))



    if 0:
        # fss.plot_iflow_comparison_of_rleft_fits(0, axes=axes1[:,1])
        fss.do_fits_for_all_mats_onlygs('var', axes=axes1[:,0])
    if 1:
        # r_min_left = [9+1/3, 7+2/3, 7]
        r_min_left = [10, 7+2/3, 6+3/3]
        fss.do_fits_for_all_mats_manym(ex_max=0, r_min_left=r_min_left, printfits=True)

    fss.plot_mats_fits(path = Path.cwd() / 'zeugs' / 'plots' / 'mats_single_fits.png')

    fss.plot_effective_mass_curves(path = Path.cwd() / 'zeugs' / 'plots' / 'mats_eff_mass.png')

    fss.plot_subtraction_fits((0,1),   axes=axes2[:,0])
    fss.plot_subtraction_fits((0,1,2), axes=axes2[:,2])

    fss.do_sums_for_sub((0,1))
    fss.do_sums_for_sub((0,1,2))

    fss.plot_partial_sums((0,1),   axes=axes2[:,1])
    fss.plot_partial_sums((0,1,2), axes=axes2[:,3])



    fig1.tight_layout()
    fig1.savefig(Path.cwd() / 'zeugs' / 'plots' / 'mats_fit_ana.png', dpi=400)

    fig2.tight_layout()
    fig2.savefig(Path.cwd() / 'zeugs' / 'plots' / 'mats_sub_fits_sums.png', dpi=400)




def main_bs():

    n_bs        = 200
    do_comp     = False
    pickle_path = Path.cwd() / 'zeugs' / 'data' / 'sums_bs.pkl'

    iflows = [8,10,12,14]

    if do_comp:

        tsums_bs = []
        psums_bs = []

        for i_bs in range(n_bs):
            print(f'bootstrap sample {i_bs+1}/{n_bs}...')

            fss = FitSubSum(bs=True)
            fss.do_fits_for_all_mats_manym(ex_max=0, r_min_left=[9+1/3, 7+2/3, 7])
            fss.do_sums_for_sub((0,1))

            tsums_bs.append(fss.tsums_sub)
            psums_bs.append(fss.psums_sub)

        
        with open(pickle_path, 'wb') as f:
            pickle.dump((tsums_bs, psums_bs), f)


    else:

        with open(pickle_path, 'rb') as f:
            tsums_bs, psums_bs = pickle.load(f)


    print(len(tsums_bs), len(psums_bs))

    tsums_bsense = { iflow : np.array([tsum[(0,1), iflow] for tsum in tsums_bs])
                     for iflow in iflows }
    psums_bsense = { iflow : np.array([psum[(0,1), iflow] for psum in psums_bs])
                     for iflow in iflows }
    
    tsums_data = gv.dataset.avg_data(tsums_bsense, bstrap=True)
    print(tsums_data)
    psums_data = gv.dataset.avg_data(psums_bsense, bstrap=True)

    
    nrows, ncols = 4, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
    axes = np.reshape(axes, shape=(nrows, ncols))

    distb = bin_distances(ns, 0.5)

    for iiflow, iflow in enumerate(iflows):
        axes[iiflow,0].hist(tsums_bsense[iflow], bins=10)
        axes[iiflow,0].set_title(f'flowtime = {flowtimes[iflow]:.2} a^2')
        plot_dist(distb, psums_data[iflow], axes[iiflow,1])
        axes[iiflow,1].set_xlim(0, 30)

    # plot final tsums
    x_list = flowtimes[iflows]
    y_list = [tsums_data[iflow] for iflow in iflows]
    axes[0,2].errorbar(
        x=x_list, y=gv.mean(y_list), yerr=gv.sdev(y_list),
        marker='o', linestyle='none'    
    )
    axes[0,2].set_xlabel('flowtime t / a^2')
    axes[0,2].set_ylabel('sum over r')
    axes[0,2].set_xlim(0, max(x_list) * 1.1)
    axes[0,2].set_ylim(min(y.mean - y.sdev for y in y_list) * 1.1, 0)

    # correlation between tsums (and plot tsums with error)
    tsums_corr = gv.evalcorr([tsums_data[iflow] for iflow in iflows])
    im = axes[1,2].imshow(tsums_corr, vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=axes[1,2])
    axes[1,2].set_title('correlation of sums in flowtime')

    fig.tight_layout()
    fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'mats_bootstrap.png', dpi=400)







def mass_rmin():

    fss = FitSubSum(bs=False)
    fss.iflows = [8,10,12]
    print([f'{flowtimes[ifl]:.2f}' for ifl in fss.iflows])
    fss.r_min_left_list_mats = tuple(np.arange(5, 12+1e-6, 1/3) for _ in range(3))


    nrows1, ncols1 = 3, 2
    fig1, axes1 = plt.subplots(nrows=nrows1, ncols=ncols1, figsize=(7*ncols1, 4*nrows1))
    axes1 = np.reshape(axes1, shape=(nrows1, ncols1))


    fss.do_fits_for_many_r_min_lefts_for_all_mats('var', 0, axes1[:,0])

    r_min_left = [10, 7+2/3, 7]
    fss.do_fits_for_all_mats_manym(0, r_min_left, printfits=True)

    # fss.plot_mats_fits(path = Path.cwd() / 'zeugs' / 'plots' / 'mats_single_fits.png')

    fss.do_sums_for_sub((0,1))
    fss.do_sums_for_sub((0,1,2))


    # fss.plot_effective_mass_curves(path = Path.cwd() / 'zeugs' / 'plots' / 'mats_eff_mass.png')


    for mats in (0,1,2):
        axes1[mats,0].set_xlim([(4.9,12.1),(4.9,10.1),(4.9,7.4)][mats])

    fig1.tight_layout()
    fig1.savefig(Path.cwd() / 'zeugs' / 'plots' / 'mats_rmin_tests_gs.png', dpi=400)



def mass_rmin_bs():

    n_bs        = 100
    do_comp     = False
    pickle_path = Path('/home/ln29bamu/code/milc_luis') / 'zeugs' / 'data' / 'bs_rmin_fits.npy'

    iflows = [8,10,12,14]
    r_mins = np.arange(5, 12+1e-6, 1/3)



    if do_comp:
        # bs, mats, rmin
        fit_masses = np.zeros(shape=(n_bs, 3, len(r_mins)), dtype=float)

        for i_bs in range(n_bs):
            print(f'bootstrap sample {i_bs+1}/{n_bs}...')

            fss = FitSubSum(bs=True)
            fss.iflows = iflows
            fss.r_min_left_list_mats = tuple(r_mins for _ in range(3))

            res = fss.do_fits_for_many_r_min_lefts_for_all_mats('var', 0)

            for mats in (0,1,2):
                for irmin in range(len(r_mins)):
                    fit_masses[i_bs, mats, irmin] = res[mats][irmin].fit.pmean[f'm_0_{mats}']

            np.save(pickle_path, fit_masses)


    else:
        # fit_masses = np.zeros(shape=(102, 3, len(r_mins)), dtype=float)
        # il = 0
        # for s, length in zip('0-20 0-26 0-26b 0-26c'.split(), (21, 27, 27, 27)):
        #     new_arr = np.load(Path('/home/ln29bamu/code/milc_luis') / 'zeugs' / 'data' / f'bs_rmin_fits{s}.npy')[:length]
        #     fit_masses[il:il+length] = new_arr
        #     il += length
        # np.save(pickle_path, fit_masses)

        fit_masses = np.load(pickle_path)



    print(fit_masses.shape, fit_masses.dtype)
    masses_data = gv.dataset.avg_data(fit_masses, bstrap=True)
    print(masses_data.shape, masses_data.dtype)


    nrows, ncols = 1, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
    axes = np.reshape(axes, shape=(nrows, ncols))


    for mats in (0,1,2):
        axes[0,mats].errorbar(
            x    = r_mins,
            y    = gv.mean(masses_data[mats, :]),
            yerr = gv.sdev(masses_data[mats, :]),
            marker    = 'o',
            linestyle = 'none' 
        )
        axes[0,mats].set_xlim([(4.9,12.1),(4.9,10.1),(4.9,7.4)][mats])
        axes[0,mats].set_xlabel('r_min / a')
        axes[0,mats].set_ylabel(f'm_0 T (Matsubara mode {mats})')


    fig.tight_layout()
    fig.savefig(Path('/home/ln29bamu/code/milc_luis') / 'zeugs' / 'plots' / 'bootstrap_rmin_fits.png', dpi=400)





# fits for subcorr-data-and-fits in section 5.6 (Subtracted correlators from data and fit)
def fits_gs():

    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 5*nrows))
    axes = np.reshape(axes, shape=(nrows, ncols))


    # ex_max=0 fit
    fss0 = FitSubSum(bs=False)
    fss0.iflows = [8,10,12]
    fss0.do_fits_for_all_mats_manym(ex_max=0, r_min_left=[10, 7.7, 6.7], printfits=False)
    fss0.plot_subtraction_fits((0,1), path=Path.cwd() / 'zeugs' / 'plots' / 'mats_sub_fits0.png', cbin_size=0.25, xlim=(3,20), ylim=(-0.2, 0.5))


    # ex_max=1 fit
    fss1 = FitSubSum(bs=False)
    fss1.iflows = [8,10,12] 
    fss1.do_fits_for_all_mats_manym(ex_max=1, r_min_left=[6.7, 6.7, 6.7], printfits=False)
    fss1.plot_subtraction_fits((0,1), path=Path.cwd() / 'zeugs' / 'plots' / 'mats_sub_fits1.png', cbin_size=0.25, xlim=(3,20), ylim=(-0.2, 0.5))



    # combined plot for sub=(0,1)
    iflow = 12
    sub   = (0,1)
    ax01  = axes[0,0]

    cbin_size = 0.25
    cbins     = find_distance_bins(fss0.dist, cbin_size)
    dist_b,   = bin_averages(cbins, fss0.dist)

    ense_sub    = mats_subtraction(sub, [build_integrand(fss0.dist, fss0.ense_all[:, iflow, mats, :], mats) for mats in sub])
    ense_sub_b, = bin_averages(cbins, ense_sub)
    data_sub_b  = gv.dataset.avg_data(ense_sub_b)

    plot_dist(dist_b, data_sub_b, ax01, color='blue', alpha=0.66, label='data', markersize=3)

    for i, fss in zip((1,0), (fss1, fss0)):
        rleft = max(fss.fitstuff_mats[mats].rlims[iflow,mats][0] for mats in sub)
        ileft = index_from_distance(dist_b, rleft)
        color = ['red', 'green'][i]
        label = ['fit (one mass)', 'fit (two masses)'][i]
        alpha = [0.7, 0.7][i]
        plot_fitfcn(dist_b[ileft:], fss.fitfcn_sub(sub, iflow, dist_b[ileft:]), ax01, color=color, label=label, alpha=alpha)

    ax01.set_xlim((3,20))
    ax01.set_ylim((-0.12, 0.35))
    ax01.set_ylabel('G_sub^(01) / T^7')



    # combined plot for sub=(0,1,2)
    iflow = 12
    sub   = (0,1,2)
    ax012 = axes[0,1]

    cbin_size = 0.25
    cbins     = find_distance_bins(fss0.dist, cbin_size)
    dist_b,   = bin_averages(cbins, fss0.dist)

    ense_sub    = mats_subtraction(sub, [build_integrand(fss0.dist, fss0.ense_all[:, iflow, mats, :], mats) for mats in sub])
    ense_sub_b, = bin_averages(cbins, ense_sub)
    data_sub_b  = gv.dataset.avg_data(ense_sub_b)

    plot_dist(dist_b, data_sub_b, ax012, color='blue', alpha=0.66, label='data', markersize=3)

    for i, fss in zip((1,0), (fss1, fss0)):
        rleft = max(fss.fitstuff_mats[mats].rlims[iflow,mats][0] for mats in sub)
        ileft = index_from_distance(dist_b, rleft)
        color = ['red', 'green'][i]
        label = ['fit (one mass)', 'fit (two masses)'][i]
        alpha = [0.7, 0.7][i]
        plot_fitfcn(dist_b[ileft:], fss.fitfcn_sub(sub, iflow, dist_b[ileft:]), ax012, color=color, label=label, alpha=alpha)

    ax012.set_xlim((3,14))
    ax012.set_ylim((-1, 3))
    ax012.set_ylabel('G_sub^(012) / T^7')



    fig.tight_layout()
    fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'two-fits-on-subcorr.png', dpi=400)





def fits_gs_bs():

    n_bs        = 200
    do_comp     = True
    pickle_path = Path('/home/ln29bamu/code/milc_luis') / 'zeugs' / 'data' / 'bs_fits_gs.pkl'

    iflows = [8,10,12,14]



    if do_comp:

        # bs, mats -> pmean
        fit_pmean_bs = np.zeros(shape=(n_bs, 3), dtype=object)
        # bs -> (sub -> sums)
        tsums_bs = []
        psums_bs = []

        for i_bs in range(n_bs):
            print(f'bootstrap sample {i_bs+1}/{n_bs}...')

            fss = FitSubSum(bs=True)
            fss.iflows = iflows

            fss.do_fits_for_all_mats_manym(ex_max=0, r_min_left=[10, 7.7, 7], printfits=False)

            for mats in (0,1,2):
                fit_pmean_bs[i_bs, mats] = fss.fitstuff_mats[mats].fit.pmean

            fss.do_sums_for_sub((0,1))
            fss.do_sums_for_sub((0,1,2))

            tsums_bs.append(fss.tsums_sub)
            psums_bs.append(fss.psums_sub)

            with pickle_path.open('wb') as f:
                pickle.dump((fit_pmean_bs[:i_bs+1], tsums_bs, psums_bs), f)


    else:
        with open(pickle_path, 'rb') as f:
            fit_pmean_bs, tsums_bs, psums_bs = pickle.load(f)



    # fit parameters
    ####################################

    print(fit_pmean_bs.shape, fit_pmean_bs.dtype)
    masses_ense = np.array(
        [ [fit_pmean_bs[i_bs, mats][f'm_0_{mats}'] for mats in (0,1,2)]
          for i_bs in range(n_bs) ],
        dtype=float
    )
    print(masses_ense.shape, masses_ense.dtype)
    masses_data = gv.dataset.avg_data(masses_ense, bstrap=True)
    print(masses_data.shape, masses_data.dtype)
    print(masses_data)

    print()

    a_ense = np.array(
        [ [[fit_pmean_bs[i_bs, mats][f'a_{ifl}_0_{mats}'] for ifl in iflows] for mats in (0,1,2)]
          for i_bs in range(n_bs) ],
          dtype=float
    )
    print(a_ense.shape, a_ense.dtype)
    a_data = gv.dataset.avg_data(a_ense, bstrap=True)
    print(a_data.shape, a_data.dtype)
    print(a_data)

    print()



    # total and partial sums
    ###############################################

    sub = (0,1)

    print(len(tsums_bs), len(psums_bs))

    tsums_bsense = { iflow : np.array([tsum[sub, iflow] for tsum in tsums_bs])
                     for iflow in iflows }
    psums_bsense = { iflow : np.array([psum[sub, iflow] for psum in psums_bs])
                     for iflow in iflows }
    
    tsums_data = gv.dataset.avg_data(tsums_bsense, bstrap=True)
    print(tsums_data)
    psums_data = gv.dataset.avg_data(psums_bsense, bstrap=True)


    # plotting stuff
    #######################

    nrows, ncols = 4, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
    axes = np.reshape(axes, shape=(nrows, ncols))

    distb = bin_distances(ns, 0.5)

    for iiflow, iflow in enumerate(iflows):
        axes[iiflow,0].hist(tsums_bsense[iflow], bins=10)
        axes[iiflow,0].set_title(f'flowtime = {flowtimes[iflow]:.2} a^2')
        plot_dist(distb, psums_data[iflow], axes[iiflow,1])
        axes[iiflow,1].set_xlim(0, 30)

    # plot final tsums
    x_list = flowtimes[iflows]
    y_list = [tsums_data[iflow] for iflow in iflows]
    axes[0,2].errorbar(
        x=x_list, y=gv.mean(y_list), yerr=gv.sdev(y_list),
        marker='o', linestyle='none'    
    )
    axes[0,2].set_xlabel('flowtime t / a^2')
    axes[0,2].set_ylabel('sum over r')
    axes[0,2].set_xlim(0, max(x_list) * 1.1)
    axes[0,2].set_ylim(min(y.mean - y.sdev for y in y_list) * 1.1, 0)

    # correlation between tsums (and plot tsums with error)
    tsums_corr = gv.evalcorr([tsums_data[iflow] for iflow in iflows])
    im = axes[1,2].imshow(tsums_corr, vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=axes[1,2])
    axes[1,2].set_title('correlation of sums in flowtime')

    fig.tight_layout()
    fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'mats_bootstrap.png', dpi=400)






def mass_rmin_manym():

    fss = FitSubSum(bs=False)

    # decent Q's
    if 1:
        ex_max = 1
        fss.iflows = [8,10,12]
        r_min_left = [6+2/3, 6+2/3, 6+2/3]
    # 2 excited states (shouldnt change much)
    if 0:
        ...
    # push rmin lower
    if 0:
        ...
    # add iflow 14
    if 0:
        ...



    if 1:

        nrows1, ncols1 = 1, 3
        fig1, axes1 = plt.subplots(nrows=nrows1, ncols=ncols1, figsize=(7*ncols1, 4*nrows1))
        axes1 = np.reshape(axes1, shape=(nrows1, ncols1))

        r_mins = np.arange(3, 10+1e-6, 1/3)

        fss.r_min_left_list_mats = tuple(r_mins for _ in range(3))
        fss.do_fits_for_many_r_min_lefts_for_all_mats('var', ex_max, axes1[0,:].flatten())

        for mats in (0,1,2):
            axes1[0,mats].set_xlim([(2.9,10.1),(2.9,9.1),(2.9,7.4)][mats])

        fig1.tight_layout()
        fig1.savefig(Path.cwd() / 'zeugs' / 'plots' / 'mats_rmin_tests.png', dpi=400)



    # r_min_left = [7, 6+2/3, 6+2/3]

    fss.do_fits_for_all_mats_manym(ex_max, r_min_left, printfits=True)

    fss.plot_mats_fits(path = Path.cwd() / 'zeugs' / 'plots' / 'mats_single_fits.png')

    fss.do_sums_for_sub((0,1))
    fss.do_sums_for_sub((0,1,2))

    # fss.plot_effective_mass_curves(path = Path.cwd() / 'zeugs' / 'plots' / 'mats_eff_mass.png')



    nrows2, ncols2 = len(fss.iflows), 4
    fig2, axes2 = plt.subplots(nrows=nrows2, ncols=ncols2, figsize=(7*ncols2, 4*nrows2))
    axes2 = np.reshape(axes2, shape=(nrows2, ncols2))
    
    fss.plot_subtraction_fits((0,1),   axes=axes2[:,0])
    fss.plot_subtraction_fits((0,1,2), axes=axes2[:,2])

    fss.plot_partial_sums((0,1),   axes=axes2[:,1])
    fss.plot_partial_sums((0,1,2), axes=axes2[:,3])

    fig2.tight_layout()
    fig2.savefig(Path.cwd() / 'zeugs' / 'plots' / 'mats_sub_fits_sums.png', dpi=400)



def mass_rmin_manym_bs():
    
    n_bs        = 200
    do_comp     = False
    pickle_path = Path('/home/ln29bamu/code/milc_luis') / 'zeugs' / 'data' / 'bs_rmin_fits_ex1.npy'

    # decent Q's
    if 1:
        ex_max = 1
        iflows = [8,10,12]
        r_min_left = [6+2/3, 6+2/3, 6+2/3]

    r_mins = np.arange(3, 10+1e-6, 1/3)



    if do_comp:
        # bs, ex, mats, rmin
        fit_masses = np.zeros(shape=(n_bs, ex_max+1, 3, len(r_mins)), dtype=float)

        for i_bs in range(n_bs):
            print(f'bootstrap sample {i_bs+1}/{n_bs}...')

            fss = FitSubSum(bs=True)
            fss.iflows = iflows
            fss.r_min_left_list_mats = tuple(r_mins for _ in range(3))

            res = fss.do_fits_for_many_r_min_lefts_for_all_mats('var', ex_max)

            for ex in range(ex_max+1):
                for mats in (0,1,2):
                    for irmin in range(len(r_mins)):
                        fit_masses[i_bs, ex, mats, irmin] = res[mats][irmin].fit.pmean[('m' if ex==0 else 'dm') + f'_{ex}_{mats}']

            np.save(pickle_path, fit_masses[:i_bs+1])



    else:
        fit_masses = np.load(pickle_path)



    print(fit_masses.shape, fit_masses.dtype)
    masses_data = gv.dataset.avg_data(fit_masses, bstrap=True)
    print(masses_data.shape, masses_data.dtype)



    nrows, ncols = 1, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
    axes = np.reshape(axes, shape=(nrows, ncols))


    for mats in (0,1,2):
        for ex in range(ex_max+1):
            axes[0,mats].errorbar(
                x    = r_mins,
                y    = gv.mean(masses_data[ex, mats, :]),
                yerr = gv.sdev(masses_data[ex, mats, :]),
                marker    = 'o',
                linestyle = '--' ,
                label = ('m' if ex==0 else 'dm') + f'_{ex}_{mats}'
            )
        axes[0,mats].set_xlim([(2.9,10.1),(2.9,9.1),(2.9,7.4)][mats])
        axes[0,mats].set_xlabel('r_min / a')
        axes[0,mats].set_ylabel(f'(d)m / T')
        axes[0,mats].legend()



    # do Q values
    if 1:

        fss = FitSubSum(bs=False)
        fss.iflows = iflows
        fss.r_min_left_list_mats = tuple(r_mins for _ in range(3))

        res = fss.do_fits_for_many_r_min_lefts_for_all_mats('var', ex_max)

        for mats in (0,1,2):
            ax = axes[0,mats]
            ax_twin = ax.twinx()
            ax_twin.plot(
                r_mins,
                [fitstuff.fit.Q for fitstuff in res[mats]],
                marker    = 's',
                linestyle = '-',
                alpha     = 0.5,
                label     = 'Q',
            )
            ax_twin.set_ylabel('Q')
            ax_twin.set_ylim(0, 1)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best')


    fig.tight_layout()
    fig.savefig(Path('/home/ln29bamu/code/milc_luis') / 'zeugs' / 'plots' / 'bootstrap_rmin_fits_ex1.png', dpi=400)







def fits_manym():

    fss = FitSubSum(bs=False)

    # decent Q's
    if 1:
        ex_max = 1
        fss.iflows = [8,10,12]
        r_min_left = [6+2/3, 6+2/3, 6+2/3]
    # add iflow 14
    if 0:
        ex_max = 1
        fss.iflows = [8,10,12,14]
        r_min_left = [6+2/3, 6+2/3, 6+2/3]
    # 2 excited states (shouldnt change much)
    if 0:
        ex_max = 2
        fss.iflows = [8,10,12]
        r_min_left = [6+2/3, 6+2/3, 6+2/3]
    # push rmin lower
    if 0:
        ex_max = 1
        fss.iflows = [8,10,12]
        r_min_left = [6+1/3, 6+1/3, 6+1/3]



    fss.do_fits_for_all_mats_manym(ex_max, r_min_left, printfits=True)

    fss.plot_mats_fits(path = Path.cwd() / 'zeugs' / 'plots' / 'mats_single_fits.png')

    fss.do_sums_for_sub((0,1))
    fss.do_sums_for_sub((0,1,2))

    fss.plot_effective_mass_curves(path = Path.cwd() / 'zeugs' / 'plots' / 'mats_eff_mass.png')



    nrows2, ncols2 = len(fss.iflows), 4
    fig2, axes2 = plt.subplots(nrows=nrows2, ncols=ncols2, figsize=(7*ncols2, 4*nrows2))
    axes2 = np.reshape(axes2, shape=(nrows2, ncols2))
    
    fss.plot_subtraction_fits((0,1),   axes=axes2[:,0])
    fss.plot_subtraction_fits((0,1,2), axes=axes2[:,2])

    fss.plot_partial_sums((0,1),   axes=axes2[:,1])
    fss.plot_partial_sums((0,1,2), axes=axes2[:,3])

    fig2.tight_layout()
    fig2.savefig(Path.cwd() / 'zeugs' / 'plots' / 'mats_sub_fits_sums.png', dpi=400)





def fits_manym_bs():

    n_bs        = 100
    do_comp     = False
    pickle_path = Path('/home/ln29bamu/code/milc_luis') / 'zeugs' / 'data' / 'bs_fits_ex.pkl'


    restart = False
    # bs_data[name] = bs -> ( (sub,ifl) -> tsum,  (sub,ifl) -> psums,
    #                         (sub,ifl) -> rleft, (sub,ifl) -> rright,
    #                         mats -> pmean )

    if (not do_comp) or (not restart and pickle_path.exists()):
        with open(pickle_path, 'rb') as f:
            bs_data = pickle.load(f)
    else:
        bs_data = {}


    # decent Q's
    if 0:
        name       = 'flow3'
        ex_max     = 1
        iflows     = [8,10,12]
        r_min_left = [6+2/3, 6+2/3, 6+2/3]
    # add iflow 14
    if 0:
        name       = 'flow4'
        ex_max     = 1
        iflows     = [8,10,12,14]
        r_min_left = [6+2/3, 6+2/3, 6+2/3]
    # add iflow 14 and 16
    if 0:
        name       = 'flow5'
        ex_max     = 1
        iflows     = [8,10,12,14,16]
        r_min_left = [6+2/3, 6+2/3, 6+2/3]
    # 2 excited states
    if 0:
        name       = 'ex2'
        ex_max     = 2
        iflows     = [8,10,12,14]
        r_min_left = [6+2/3, 6+2/3, 6+2/3]
    # lower rmin
    if 0:
        name       = 'rmin6p3'
        ex_max     = 1
        iflows     = [8,10,12,14]
        r_min_left = [6+1/3, 6+1/3, 6+1/3]
    # even lower rmin
    if 0:
        name       = 'rmin6p0'
        ex_max     = 1
        iflows     = [8,10,12,14]
        r_min_left = [6, 6, 6]
    # only ground state
    if 1:
        name       = 'gs'
        ex_max     = 0
        iflows     = [8,10,12,14]
        r_min_left = [10, 7.7, 6.7]



    if do_comp:

        if name not in bs_data:
            bs_data[name] = []


        for i_bs in range(n_bs):
            print(f'bootstrap sample {i_bs+1}/{n_bs}...')

            fss = FitSubSum(bs=True)
            fss.iflows = iflows

            fss.do_fits_for_all_mats_manym(ex_max, r_min_left, printfits=False)

            pmeans = [fss.fitstuff_mats[mats].fit.pmean for mats in (0,1,2)]

            fss.do_sums_for_sub((0,1))
            fss.do_sums_for_sub((0,1,2))

            bs_data[name].append((fss.tsums_sub, fss.psums_sub, fss.rleft_sub, fss.rright_sub, pmeans))

            with pickle_path.open('wb') as f:
                pickle.dump(bs_data, f)


    print()
    print([(nm, len(bs)) for nm, bs in bs_data.items()])
    print()



    # fit parameters
    ####################################
    if 0:

        for nm in bs_data.keys():

            ex_maxes = {'flow3' : 1, 'flow4' : 1, 'flow5' : 1, 'rmin6p3' : 1, 'rmin6p0' : 1, 'ex2' : 2, 'gs' : 0}

            print()
            print(nm)
            masses_ense = np.array(
                [ [ [pmeans[mats][('m' if ex==0 else 'dm') + f'_{ex}_{mats}'] for ex in range(ex_maxes[nm]+1)]
                    for mats in (0,1,2) ]
                for _,_,_,_,pmeans in bs_data[nm] ],
                dtype=float
            )
            print(masses_ense.shape, masses_ense.dtype)
            masses_data = gv.dataset.avg_data(masses_ense, bstrap=True)
            print(masses_data.shape, masses_data.dtype)
            print(masses_data)

        print()

        # a_ense = np.array(
        #     [ [[fit_pmean_bs[i_bs, mats][f'a_{ifl}_0_{mats}'] for ifl in iflows] for mats in (0,1,2)]
        #       for i_bs in range(n_bs) ],
        #       dtype=float
        # )
        # print(a_ense.shape, a_ense.dtype)
        # a_data = gv.dataset.avg_data(a_ense, bstrap=True)
        # print(a_data.shape, a_data.dtype)
        # print(a_data)

        # print()



    # total and partial sums
    ###############################################
    if 0:

        nrows, ncols = len(iflows), 3*2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
        axes = np.reshape(axes, shape=(nrows, ncols))



        for isub, sub in enumerate([(0,1), (0,1,2)]):
            print('isub,sub =', isub, sub)

            tsums_bsense = { iflow : np.array([tsum[sub, iflow] for tsum,_,_,_,_ in bs_data[name]])
                            for iflow in iflows }
            psums_bsense = { iflow : np.array([psum[sub, iflow] for _,psum,_,_,_ in bs_data[name]])
                            for iflow in iflows }
            
            tsums_data = gv.dataset.avg_data(tsums_bsense, bstrap=True)
            print(tsums_data)
            psums_data = gv.dataset.avg_data(psums_bsense, bstrap=True)


            # plotting stuff
            #######################

            distb = bin_distances(ns, 0.5)

            for iiflow, iflow in enumerate(iflows):
                axes[iiflow,3*isub].hist(tsums_bsense[iflow], bins=20)
                axes[iiflow,3*isub].set_title(f'flowtime = {flowtimes[iflow]:.2} a^2')

                plot_dist(distb, psums_data[iflow], axes[iiflow,3*isub+1])
                axes[iiflow,3*isub+1].set_xlim(0, 17)

                rleft = gv.dataset.avg_data(np.array([ll[sub, iflow] for _,_,ll,_,_ in bs_data[name]]))
                axes[iiflow,3*isub+1].axvline(rleft.mean, linewidth=2, alpha=0.8, color='grey')
                axes[iiflow,3*isub+1].axvspan(rleft.mean - rleft.sdev, rleft.mean + rleft.sdev, alpha=0.3, color='grey')

                rright = gv.dataset.avg_data(np.array([rr[sub, iflow] for _,_,_,rr,_ in bs_data[name]]))
                axes[iiflow,3*isub+1].axvline(rright.mean, linewidth=2, alpha=0.8, color='grey')
                axes[iiflow,3*isub+1].axvspan(rright.mean - rright.sdev, rright.mean + rright.sdev, alpha=0.5, color='grey')


            # plot final tsums
            x_list = flowtimes[iflows]
            y_list = [tsums_data[iflow] for iflow in iflows]
            axes[0,3*isub+2].errorbar(
                x=x_list, y=gv.mean(y_list), yerr=gv.sdev(y_list),
                marker='o', linestyle='none'    
            )
            axes[0,3*isub+2].set_xlabel('flowtime t / a^2')
            axes[0,3*isub+2].set_ylabel('sum over r')
            axes[0,3*isub+2].set_xlim(0, max(x_list) * 1.1)
            axes[0,3*isub+2].set_ylim(min(y.mean - y.sdev for y in y_list) * 1.1, 0)

            # correlation between tsums (and plot tsums with error)
            tsums_corr = gv.evalcorr([tsums_data[iflow] for iflow in iflows])
            im = axes[1,3*isub+2].imshow(tsums_corr, vmin=0, vmax=1)
            cbar = plt.colorbar(im, ax=axes[1,3*isub+2])
            axes[1,3*isub+2].set_title('correlation of sums in flowtime')



        fig.tight_layout()
        fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'mats_bootstrap_ex1.png', dpi=400)



    # figure comparing gs and ex1 fits
    ###############################################
    if 0:

        nrows, ncols = 1, 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 5*nrows))
        axes = np.reshape(axes, shape=(nrows, ncols))

        names = ['gs', 'flow4']
        subs  = [(0,1), (0,1,2)]


        for isub, sub in enumerate(subs):

            for iname, name in enumerate(names):

                iflows = [[8,10,12,14], [8,10,12,14]][iname]
                x_list = [flt + iname*0.01 for flt in flowtimes[iflows]]
                
                tsums_ense = { iflow : np.array([tsum[sub, iflow] for tsum,_,_,_,_ in bs_data[name]])
                            for iflow in iflows }
                tsums_data = gv.dataset.avg_data(tsums_ense, bstrap=True)
                y_list = [tsums_data[ifl] for ifl in iflows]

                label = ['fit (one mass)', 'fit (two masses)'][iname]
                axes[0,isub].errorbar(
                    x=x_list, y=gv.mean(y_list), yerr=gv.sdev(y_list),
                    marker='o', linestyle='none', label=label,
                )

            axes[0,isub].set_xlabel('flowtime t / a^2')
            axes[0,isub].set_ylabel(f'H{sub} / T^4')
            axes[0,isub].set_xlim(0, max(x_list) * 1.1)
            axes[0,isub].set_ylim(min(y.mean - y.sdev for y in y_list) * 1.1, 0)
            axes[0,isub].legend()


        fig.tight_layout()
        fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'subcorr-comp-gs-ex1.png', dpi=400)



    # figure comparing gs and ex1 fits
    ###############################################
    if 1:

        nrows, ncols = 1, 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 5*nrows))
        axes = np.reshape(axes, shape=(nrows, ncols))

        names = ['flow3', 'flow4', 'flow5', 'ex2', 'rmin6p3']
        subs  = [(0,1), (0,1,2)]

        cmap = matplotlib.colormaps['plasma']
        flow_colors = [cmap(v) for v in np.linspace(0, 0.4, 3)]
        extra_colors = ['mediumseagreen', 'darkturquoise']
        all_colors = flow_colors + extra_colors



        for isub, sub in enumerate(subs):

            for iname, name in enumerate(names):

                iflows = [[8,10,12], [8,10,12,14], [8,10,12,14,16], [8,10,12,14], [8,10,12,14]][iname]
                x_list = [flt + iname*0.01 for flt in flowtimes[iflows]]
                
                tsums_ense = { iflow : np.array([tsum[sub, iflow] for tsum,_,_,_,_ in bs_data[name]])
                            for iflow in iflows }
                tsums_data = gv.dataset.avg_data(tsums_ense, bstrap=True)
                y_list = [tsums_data[ifl] for ifl in iflows]

                axes[0,isub].errorbar(
                    x=x_list, y=gv.mean(y_list), yerr=gv.sdev(y_list),
                    marker    = 'o',
                    linestyle = 'none',
                    color     = all_colors[iname],
                    label     = ['fit (3 flowtimes)', 'fit (4 flowtimes)', 'fit (5 flowtimes)', 'fit (3 masses)', 'fit (r0 = 6.3)'][iname],
                )

            axes[0,isub].set_xlabel('flowtime t / a^2')
            axes[0,isub].set_ylabel(f'H{sub} / T^4')
            axes[0,isub].set_xlim(0, flowtimes[16] * 1.1)
            axes[0,isub].set_ylim(min(y.mean - y.sdev for y in y_list) * 1.1, 0)
            
            handles, labels = axes[0,isub].get_legend_handles_labels()

            flow_handles  = handles[:3]
            flow_labels   = labels[:3]
            extra_handles = handles[3:]
            extra_labels  = labels[3:]

            separator = mlines.Line2D([], [], linestyle='none', label='')

            axes[0,isub].legend(
                flow_handles + [separator] + extra_handles,
                flow_labels  + ['']        + extra_labels,
                frameon=True
            )


        fig.tight_layout()
        fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'subcorr-comp-various.png', dpi=400)















main = fits_gs




if __name__ == '__main__':

    main()