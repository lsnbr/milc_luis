from pathlib import Path
import pickle
import numpy as np
import gvar as gv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator

import scienceplots
plt.style.use('science')
plt.rcParams.update({
    'font.size' : 10,
    'text.usetex' : True
})

from measurements import *
from statana import *
from ana_mats import FitStuff, FitSubSum, build_integrand, mats_subtraction, plot_dist_many



from ensemble_data import data_16_x_64_1p5Tc as data_ense

nt = data_ense['nt']
ns = data_ense['ns']
flowtimes = data_ense['flowtimes']




latex_width_pts = 469.75502
wlatex = 6.5243753





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
    axes[0,1].set_ylabel(r'$G \cdot \sinh / r / T^7$')

    plot_dist_many(dist_binned, [data_m0_int, data_m1_int, data_m2_int], il1, ir1, ['n=0', 'n=1', 'n=2'], axes[1,1])
    axes[1,1].set_ylabel(r'$G \cdot \sinh / r / T^7$')

    # plot subtracted modes
    plot_dist_many(dist_binned, [data_sub_int], il0, ir0, ['sub012'], axes[0,2])
    axes[0,2].set_ylabel(r'$G \cdot \sinh / r / T^7$')

    plot_dist_many(dist_binned, [data_sub_int], il1, ir1, ['sub012'], axes[1,2])
    axes[1,2].set_ylabel(r'$G \cdot \sinh / r / T^7$')

    # plot pure matsubara modes
    plot_dist_many(dist_binned, [data_m0, data_m1, data_m2], il2, ir2, ['n=0', 'n=1', 'n=2'], axes[0,0])
    axes[0,0].set_ylabel(r'G$(w_n, r) / T^7$')

    plot_dist_many(dist_binned, [data_m0, data_m1, data_m2], il3, ir3, ['n=0', 'n=1', 'n=2'], axes[1,0])
    axes[1,0].set_ylabel(r'G$(w_n, r) / T^7$')


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




def vis_binning():

    # get correlators  G(w_n, r) / T^7
    print('loading correlator data...')
    ense_all = get_data_mats_unbinned()
    print(ense_all.shape, ense_all.dtype)

    # get distances
    dist = radial_separations(ns)

    # bin with constant r bins
    bin_size = 0.001
    bins = find_distance_bins(dist, bin_size)
    dist_binned, = bin_averages(bins, dist)

    # flowtime
    iflow_o = 12
    mats = 1
    print(f't_f = {flowtimes[iflow_o]} a^2, r_f = {flowtime_to_radius(flowtimes[iflow_o], nt) * nt} a')


    fig, axes = plt.subplots(1,2, figsize=(5*2, 4*1))


    ense_m1    = ense_all[:, iflow_o, mats, :]
    ense_m1_b, = bin_averages(bins, ense_m1)
    data_m1    = gv.dataset.avg_data(ense_m1_b)

    plot_dist(dist_binned, data_m1, axes[0])
    axes[0].set_ylabel('$G/T^7$')
    axes[0].set_xlim(5.5, 25)
    axes[0].set_ylim(-0.1, 0.02)



    fss = FitSubSum(bs=False)
    fss.iflows[mats] = [8,10,12,14,16,18]
    ex_max = 1

    rmin_per_iflow_mats_exmax : list[list[dict[int,float]]] = [
        [   # ex_max = 0
            {8:10.5, 10:11, 12:11.5, 14:12, 16:12.5, 18:13, 20:14, 21:15},  # mats = 0
            {8:7.5, 10:7.5, 12:8, 14:8.5, 16:9, 18:9.5, 20:10, 21:10.5},    # mats = 1
            {8:6+1/3, 10:6+1/3, 12:6+2/3, 14:6+2/3},                        # mats = 2
        ],
        # [   # ex_max = 1
        #     {8:9, 10:9, 12:10.5, 14:11, 16:11.5, 18:11.5, 20:12, 21:13},    # mats = 0
        #     {8:6.5, 10:7, 12:7.5, 14:8, 16:9, 18:9.5, 20:10, 21:11},        # mats = 1
        #     {8:6+1/3, 10:6+1/3, 12:6+2/3, 14:6+2/3},                        # mats = 2
        # ],
        [   # ex_max = 1
            {8:6, 10:6.5, 12:6.5, 14:7, 16:7.5, 18:8, 20:9, 21:10},    # mats = 0
            {8:6.5, 10:6.5, 12:7, 14:7.5, 16:8, 18:8.5, 20:9, 21:10},        # mats = 1
            {8:6+1/3, 10:6+1/3, 12:6+2/3, 14:6+2/3},                        # mats = 2
        ]
    ]

    rmin_per_iflow = rmin_per_iflow_mats_exmax[ex_max][mats]
    fss.do_fit_one_mat_diff_rmin(mats, rmin_per_iflow, ex_max, printfits=True)
    fss.do_sums_for_mats(mats_vals=(1,), printsums=False)

    cbin_size = 0.25
    cbins = find_distance_bins(fss.dist, cbin_size)
    dist_b, = bin_averages(cbins, fss.dist)

    r_left, r_right = fss.fitstuff_mats[mats].rlims[iflow_o,mats]
    ir_left, ir_right = index_from_distance(fss.dist, r_left), index_from_distance(fss.dist, r_right)

    plot_dist(fss.fitstuff_mats[mats].fit.x[iflow_o,mats], fss.fitstuff_mats[mats].fit.y[iflow_o,mats], axes[1], label=f'data', alpha=0.7)
    axes[1].set_ylabel('$G/T^7$')
    axes[1].set_xlim(5.5, 25)
    axes[1].set_ylim(-0.08, 0.02)


    # finalize and save plots
    fig.tight_layout()
    fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'abc_bins.png', dpi=400)   




# some analysis of simultaneous fits for different flowtime and max_ex combinations
def mass_rmin():



    rmin_per_iflow_mats_exmax : list[list[dict[int,float]]] = [
        [   # ex_max = 0
            {8:10.5, 10:11, 12:11.5, 14:12, 16:12.5, 18:13, 20:14, 21:15},  # mats = 0
            {8:7.5, 10:7.5, 12:8, 14:8.5, 16:9, 18:9.5, 20:10, 21:10.5},    # mats = 1
            {8:6+1/3, 10:6+1/3, 12:6+2/3, 14:6+2/3},                        # mats = 2
        ],
        # [   # ex_max = 1
        #     {8:9, 10:9, 12:10.5, 14:11, 16:11.5, 18:11.5, 20:12, 21:13},    # mats = 0
        #     {8:6.5, 10:7, 12:7.5, 14:8, 16:9, 18:9.5, 20:10, 21:11},        # mats = 1
        #     {8:6+1/3, 10:6+1/3, 12:6+2/3, 14:6+2/3},                        # mats = 2
        # ],
        [   # ex_max = 1
            {8:6, 10:6.5, 12:6.5, 14:7, 16:7.5, 18:8, 20:9, 21:10},    # mats = 0
            {8:6.5, 10:6.5, 12:7, 14:7.5, 16:8, 18:8.5, 20:9, 21:10},        # mats = 1
            {8:6+1/3, 10:6+1/3, 12:6+2/3, 14:6+2/3},                        # mats = 2
        ]
    ]



    # check fit masses and Q
    if 0:

        for mats in (0,1,2):
            for ex_max in (0,1):
           
                print(f'\n-----  {mats=}  -----  {ex_max=}  -----')
                iflows = [8,10,12,14,16,18,20,21] if mats in (0,1) else [8,10,12,14]

                for i in range(1, len(iflows)+1):
                    fss = FitSubSum(printinit=False)
                    fss.iflows[mats] = iflows[:i]
                    fs = fss.do_fit_one_mat_diff_rmin(mats, rmin_per_iflow_mats_exmax[ex_max][mats], ex_max)
                    m_name = f'm_0_{mats}'
                    print(f'{m_name} = {fs.fit.p[m_name]},  ', end='')
                    for ex in range(1, ex_max+1):
                        dm_name = f'dm_{ex}_{mats}'
                        print(f'{dm_name} = {fs.fit.p[dm_name]},  ', end='')
                    print(f'Q = {fs.fit.Q:.4f},  ', end='')
                    print(f'iflows = {fss.iflows[mats]}')


    # use ex_max=1 for  r_0 from ex_max=0
    if 0:

        ex_max_rmin = 1
        ex_max_fit  = 1

        for mats in (0,1,2):
            print(f'\n-----  {mats=}  -----')
            iflows = [8,10,12,14,16,18,20,21] if mats in (0,1) else [8,10,12,14]

            for i in range(1, len(iflows)+1):
                fss = FitSubSum(printinit=False)
                fss.iflows[mats] = iflows[:i]
                fs = fss.do_fit_one_mat_diff_rmin(mats, rmin_per_iflow_mats_exmax[ex_max_rmin][mats], ex_max_fit)
                m_name = f'm_0_{mats}'
                print(f'{m_name} = {fs.fit.p[m_name]},  ', end='')
                for ex in range(1, ex_max_fit+1):
                    dm_name = f'dm_{ex}_{mats}'
                    print(f'{dm_name} = {fs.fit.p[dm_name]},  ', end='')
                print(f'Q = {fs.fit.Q:.4f},  ', end='')
                print(f'iflows = {fss.iflows[mats]}')





# fits for subcorr-data-and-fits in section 5.6 (Subtracted correlators from data and fit)
def fits_gs():

    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(wlatex, wlatex/ncols*0.8))
    axes = np.reshape(axes, shape=(nrows, ncols))


    # ex_max=0 fit
    fss0 = FitSubSum(bs=False)
    fss0.iflows = [8,10,12]
    fss0.do_fits_for_all_mats_manym(ex_max=0, r_min_left=[10, 7.7, 6.7], printfits=False)
    # fss0.plot_subtraction_fits((0,1), path=Path.cwd() / 'zeugs' / 'plots' / 'mats_sub_fits0.png', cbin_size=0.25, xlim=(3,20), ylim=(-0.2, 0.5))


    # ex_max=1 fit
    fss1 = FitSubSum(bs=False)
    fss1.iflows = [8,10,12] 
    fss1.do_fits_for_all_mats_manym(ex_max=1, r_min_left=[6.7, 6.7, 6.7], printfits=False)
    # fss1.plot_subtraction_fits((0,1), path=Path.cwd() / 'zeugs' / 'plots' / 'mats_sub_fits1.png', cbin_size=0.25, xlim=(3,20), ylim=(-0.2, 0.5))



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

    plot_dist(dist_b, data_sub_b, ax01, color='blue', alpha=0.5, label='data', markersize=3)

    for i, fss in zip((1,0), (fss1, fss0)):
        rleft = max(fss.fitstuff_mats[mats].rlims[iflow,mats][0] for mats in sub)
        ileft = index_from_distance(dist_b, rleft)
        color = ['red', 'green'][i]
        label = ['fit (one mass)', 'fit (two masses)'][i]
        alpha = [0.8, 1][i]
        plot_fitfcn(dist_b[ileft:], fss.fitfcn_sub(sub, iflow, dist_b[ileft:]), ax01, color=color, label=label, alpha=alpha)

    ax01.set_xlim((3,17))
    ax01.set_ylim((-0.12, 0.39))
    # ax01.set_ylabel(r'$f^{(0,1)}(r) / T^7$')



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

    plot_dist(dist_b, data_sub_b, ax012, color='blue', alpha=0.5, label='data', markersize=3)

    for i, fss in zip((1,0), (fss1, fss0)):
        rleft = max(fss.fitstuff_mats[mats].rlims[iflow,mats][0] for mats in sub)
        ileft = index_from_distance(dist_b, rleft)
        color = ['red', 'green'][i]
        label = ['fit (one mass)', 'fit (two masses)'][i]
        alpha = [0.8, 1][i]
        plot_fitfcn(dist_b[ileft:], fss.fitfcn_sub(sub, iflow, dist_b[ileft:]), ax012, color=color, label=label, alpha=alpha)

    ax012.set_xlim((3,12))
    ax012.set_ylim((-0.14, 0.5))
    # ax012.set_ylabel(r'$f^{(0,1,2)}(r) / T^7$')



    for iax, ax in enumerate(axes.flat):
        ax.tick_params(direction="in", top=True, right=True)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.set_xlim((3,15))
        ax.set_ylabel(None)
        ax.text(
            0.05, 0.95,
            r'$f^{(0,1)}(r) / T^7$' if iax==0 else r'$f^{(0,1,2)}(r) / T^7$',
            transform=ax.transAxes,
            ha="left",
            va="top"
        )
        handles, labels = ax.get_legend_handles_labels()
        order = [2, 1, 0]
        ax.legend(
            [handles[i] for i in order], [labels[i] for i in order],
            loc='upper right',
            frameon=True, facecolor="white", edgecolor="none", framealpha=0.95,
        )



    fig.tight_layout(w_pad=0.5)
    fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'two-fits-on-subcorr.pdf', dpi=400)




# bootstrap analysis of partial and total sums (of subtracted H)
def fits_gs_bs():

    n_bs        = 200
    do_comp     = False
    pickle_path = Path.cwd() / 'zeugs' / 'data' / 'bs_fits_gs.pkl'

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
    print()

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
        axes[iiflow,0].set_title(rf'$t_\mathrm{{f}} = {flowtimes[iflow]:.2} a^2$')
        plot_dist(distb, psums_data[iflow], axes[iiflow,1])
        axes[iiflow,1].set_xlim(0, 30)

    # plot final tsums
    x_list = flowtimes[iflows]
    y_list = [tsums_data[iflow] for iflow in iflows]
    axes[0,2].errorbar(
        x=x_list, y=gv.mean(y_list), yerr=gv.sdev(y_list),
        marker='o', linestyle='none'    
    )
    axes[0,2].set_xlabel(rf'$t_\mathrm{{f}} / a^2$')
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














main = fits_gs_bs




if __name__ == '__main__':

    main()