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
})

from flowing import flowtime_to_radius
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





# for the plot showing correlatios in r. between matsubara modes, and in t
def vis_correlations():

    ense_all = get_data_mats_unbinned()  # shape: (nconf, nflow, nmats, ndist)
    dist = radial_separations(ns)
    nflow = ense_all.shape[1]


    # correlations in r
    iflow_r = 16
    flow_r  = flowtimes[iflow_r]

    ense_modes = ense_all[:, iflow_r, 0:2, :]          
    data_modes = gv.dataset.avg_data(ense_modes)                 
    corr_modes = gv.evalcorr(data_modes) 

    rmin = 0
    rmax = 20
    il = index_from_distance(dist, rmin)
    ir = index_from_distance(dist, rmax)                       

    corr_r_m     = corr_modes[1, il:ir, 1, il:ir]      
    corr_r_cross = corr_modes[0, il:ir, 1, il:ir]

    n_r = corr_r_m.shape[0]


    # correlation in t
    mats_flow = 1
    r_flow    = 10
    ir_flow   = index_from_distance(dist, r_flow)

    data_flow = gv.dataset.avg_data(ense_all[:, :, mats_flow, ir_flow])
    corr_flow = gv.evalcorr(data_flow)                                    


    # --- plot 1x3 with single colorbar ---
    fig, axes = plt.subplots(
        1, 3,
        figsize=(wlatex, 0.38 * wlatex),
        constrained_layout=True
    )

    # ----- LEFT: r-corr mode mats_for_r -----
    im0 = axes[0].imshow(
        corr_r_m,
        origin="upper",
        cmap="coolwarm",
        vmin=0, vmax=1,
        interpolation="nearest",
        aspect="equal"
    )

    n_ticks = 5
    ticks = np.linspace(0, n_r - 1, n_ticks)
    labels = np.linspace(rmin, rmax, n_ticks)

    axes[0].set_xticks(ticks)
    axes[0].set_yticks(ticks)
    axes[0].set_xticklabels([f"{x:.0f}" for x in labels])
    axes[0].set_yticklabels([f"{x:.0f}" for x in labels])

    axes[0].set_xlabel(None) 
    axes[0].set_ylabel(None) 
    axes[0].text(
        0.05, 0.05, r"$r/a$",
        transform=axes[0].transAxes,
        ha="left", va="bottom",
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=1, pad=1.5),
    )
    # axes[0].set_xlabel(r"$r / a$")
    # axes[0].set_ylabel(r"$r / a$")
    axes[0].set_title(f"mode 1, $t={flow_r:.2f} a^2$")

    # ----- MIDDLE: cross r-corr -----
    im1 = axes[1].imshow(
        corr_r_cross,
        origin="upper",
        cmap="coolwarm",
        vmin=0, vmax=1,
        interpolation="nearest",
        aspect="equal"
    )

    axes[1].set_xticks(ticks)
    axes[1].set_yticks(ticks)
    axes[1].set_xticklabels([f"{x:.0f}" for x in labels])
    axes[1].set_yticklabels([f"{x:.0f}" for x in labels])

    axes[1].set_xlabel(None) 
    axes[1].set_ylabel(None) 
    axes[1].text(
        0.05, 0.05, r"$r/a$",
        transform=axes[1].transAxes,
        ha="left", va="bottom",
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=1, pad=1.5),
    )
    # axes[1].set_xlabel(r"$r / a$")
    # axes[1].set_ylabel(r"$r / a$")
    axes[1].set_title(f"mode 0 vs 1, $t={flow_r:.2f} a^2$")

    # ----- RIGHT: flowtime corr -----
    im2 = axes[2].imshow(
        corr_flow,
        origin="upper",
        cmap="coolwarm",
        vmin=0, vmax=1,
        interpolation="nearest",
        aspect="equal"
    )

    nflow = len(flowtimes)
    n_ticks_t = 4
    tick_idx = np.linspace(0, nflow - 1, n_ticks_t).round().astype(int)

    axes[2].set_xticks(tick_idx)
    axes[2].set_yticks(tick_idx)
    axes[2].set_xticklabels([f"{flowtimes[i]:.2f}" for i in tick_idx])
    axes[2].set_yticklabels([f"{flowtimes[i]:.2f}" for i in tick_idx])

    axes[2].set_xlabel(None) 
    axes[2].set_ylabel(None) 
    axes[2].text(
        0.05, 0.05, r"$t/a^2$",
        transform=axes[2].transAxes,
        ha="left", va="bottom",
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=1, pad=1.5),
    )
    # axes[2].set_xlabel(r"t / $a^2$")
    # axes[2].set_ylabel(r"flowtime / $a^2$")
    axes[2].set_title(f"mode ${mats_flow}$, $r={r_flow}$")

    # single colorbar
    cbar = fig.colorbar(
        im2,
        ax=axes[2],     # attach to the rightmost axis only
        fraction=0.046, # controls colorbar height relative to axis
        pad=0.08
    )
    cbar.set_label("correlation")

    outpath = Path.cwd() / 'zeugs' / 'plots' / 'corr_1x3.pdf'
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)









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






# new plots to different r_0
# nnnnneeeeewwwwww
def mass_rmin():


    # plots for different r_0
    if 0:

        fss = FitSubSum(bs=False)
        fss.iflows = [8,10,12,14,16,18,20,21]
        print([f'{flowtimes[ifl]:.2f}' for ifl in fss.iflows])

        nrows1, ncols1 = len(fss.iflows), 3
        fig1, axes1 = plt.subplots(nrows=nrows1, ncols=ncols1, figsize=(9*ncols1, 4*nrows1))
        axes1 = np.reshape(axes1, shape=(nrows1, ncols1))


        ex_max = 1

        mats = 2

        if ex_max == 0:
            fss.r_min_left_list_mats = [
                tuple(np.arange(5, 18+1e-6, 1/2) for _ in range(3)),
                tuple(np.arange(5, 13+1e-6, 1/2) for _ in range(3)),
                tuple(np.arange(5, 11+1e-6, 1/3) for _ in range(3)),
            ][mats]
        elif ex_max == 1:
            fss.r_min_left_list_mats = [
                tuple(np.arange(5, 16+1e-6, 1/2) for _ in range(3)),
                tuple(np.arange(5, 13+1e-6, 1/2) for _ in range(3)),
                tuple(np.arange(4, 10+1e-6, 1/3) for _ in range(3)),
            ][mats]
        else:
            raise Exception('efjibwiofubwi')

        allfits = fss.plot_iflow_comparison_of_rleft_fits_single_mats(mats=mats, ex_max=ex_max, axes=axes1[:,0])



        for iifl, ifl in enumerate(fss.iflows):

            if ex_max == 0:
                rmin1, rmin2 = {
                    0 : {8:(10.5,13), 10:(11,14.5), 12:(11.5,16), 14:(12,16.5), 16:(12.5,17), 18:(13,18), 20:(14,18), 21:(14.5,18)},
                    1 : {8:(7,10), 10:(7.5,10.5), 12:(8,11), 14:(8,11.5), 16:(9,12), 18:(9,12), 20:(10,13), 21:(10.5,13)},
                    2 : {8:(5+2/3,7), 10:(6,7), 12:(6+1/3,7+2/3), 14:(6+2/3,8+1/3), 16:(7,8+1/3), 18:(7+2/3,9), 20:(8,9+1/3), 21:(9,10+1/3)},
                }[mats][ifl]
            elif ex_max == 1:
                rmin1, rmin2 = {
                    0 : {8:(6.5,9.5), 10:(7,8.5), 12:(7.5,11.5), 14:(9,11), 16:(9.5,12), 18:(9,11.5), 20:(11.5,15), 21:(10,12)},
                    1 : {8:(5.5,6.5), 10:(6.5,8), 12:(7,9.5), 14:(7.5,9), 16:(7,8.5), 18:(8,10), 20:(9,11), 21:(10,12)},
                    2 : {8:(5+1/3,6+2/3), 10:(5+2/3,6+2/3), 12:(6,6+1/3), 14:(6+2/3,7+1/3), 16:(6+2/3,7+2/3), 18:(6,7+2/3), 20:(8,9), 21:(8,9+2/3)},
                }[mats][ifl]
            else:
                raise Exception('wefiwuhefuwihf')
            irmin1, irmin2 = (np.argwhere(np.abs(fss.r_min_left_list_mats[mats] - rmin) < 1e-6)[0][0] for rmin in (rmin1, rmin2))
 
            for i, (irmin, rmin) in enumerate(zip((irmin1, irmin2), (rmin1, rmin2))):
                ax = axes1[iifl, 1+i]
                r_left, r_right = allfits[iifl][irmin].rlims[ifl,mats]
                ir_left, ir_right = index_from_distance(fss.dist, r_left), index_from_distance(fss.dist, r_right)
                plot_dist(allfits[iifl][irmin].fit.x[ifl,mats], allfits[iifl][irmin].fit.y[ifl,mats], ax, label=f'data', alpha=0.7)
                x_fit = fss.dist[ir_left:ir_right]
                y_fit = allfits[iifl][irmin].fit.fcn({(ifl,mats) : fss.dist[ir_left:ir_right]}, allfits[iifl][irmin].fit.p)[ifl,mats]
                plot_fitfcn(x_fit, y_fit, ax, label=rf'fit $(r_0={rmin:.1f})$')

                ax.set_xlabel(r'$r / a$')
                ax.set_xlim([(6,35), (5,25), (5,18)][mats])
                yliml, ylimr = min(gv.mean(y_fit)), max(gv.mean(y_fit))
                ax.set_ylim(yliml - (ylimr-yliml)*0.1, ylimr + (ylimr-yliml)*0.1)
                ax.set_ylabel(r'$G / T^7$')
                ax.legend(loc='lower right')



        fig1.tight_layout()
        fnameabc = 'gs' if ex_max==0 else f'ex{ex_max}'
        fig1.savefig(Path.cwd() / 'zeugs' / 'plots' / f'mats_rmin_tests_{fnameabc}_mats{mats}.png', dpi=400)






    rmin_per_iflow_mats_exmax : list[list[dict[int,float]]] = [
        [   # ex_max = 0
            {8:10.5, 10:11, 12:11.5, 14:12, 16:12.5, 18:13, 20:14, 21:15},  # mats = 0
            {8:7.5, 10:7.5, 12:8, 14:8.5, 16:9, 18:9.5, 20:10, 21:10.5},    # mats = 1
            {8:6+1/3, 10:6+1/3, 12:6+2/3, 14:6+2/3},                        # mats = 2
        ],
        [   # ex_max = 1
            {8:9, 10:9, 12:10.5, 14:11, 16:11.5, 18:11.5, 20:12, 21:13},    # mats = 0
            {8:6.5, 10:7, 12:7.5, 14:8, 16:9, 18:9.5, 20:10, 21:11},        # mats = 1
            {8:6+1/3, 10:6+1/3, 12:6+2/3, 14:6+2/3},                        # mats = 2
        ]
    ]



    # do simultaneous plot
    if 0:

        ex_max = 1

        fss = FitSubSum(bs=False)

        fss.iflows[0] = [8,10,12,14,16,18]
        fss.iflows[1] = [8,10,12,14,16,18]
        fss.iflows[2] = [8,10,12,14]

        for mats in (0,1,2):

            rmin_per_iflow = rmin_per_iflow_mats_exmax[ex_max][mats]

            fss.do_fit_one_mat_diff_rmin(mats, rmin_per_iflow, ex_max=ex_max, printfits=True)


        fss.plot_mats_fits(path = Path.cwd() / 'zeugs' / 'plots' / 'mats_fits.png')

        # fss.do_sums_for_sub(sub=(0,1))
        # fss.do_sums_for_sub(sub=(0,1,2))
        # fss.plot_subtraction_fits(sub=(0,1,2), path = Path.cwd() / 'zeugs' / 'plots' / 'mats_sub012.png')

        fss.do_sums_for_mats(mats_vals=(0,1,2), printsums=True)
        fss.do_sums_for_sub_from_mats(sub=(0,1))
        fss.do_sums_for_sub_from_mats(sub=(0,1,2))

        fss.plot_mats_integrand_fits(path = Path.cwd() / 'zeugs' / 'plots' / 'mats_sinh_fits.png')




    # check fit masses and Q
    if 1:

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







# new subs sums
# nnnnneeeeewwwwww
def sub_sums_bs():

    n_bs        = 50
    do_comp     = True
    pickle_path = Path.cwd() / 'zeugs' / 'data' / 'bs_subsums_many.pkl'


    restart = False
    # bs_data[name][bs] = (ifl -> tsum01, ifl -> tsum012, mats -> pmean, (sums_mats, rleft, rright))

    if (not do_comp) or (not restart and pickle_path.exists()):
        with open(pickle_path, 'rb') as f:
            bs_data = pickle.load(f)
    else:
        bs_data = {}



    name = 'gs21'


    if name == 'gs18': name = 'gs'

    if name[:2] == 'gs':
        ex_max = [0,0,0]
        rmin_per_iflow = [
            {8:10.5, 10:11, 12:11.5, 14:12, 16:12.5, 18:13, 20:14, 21:15},
            {8:7.5, 10:7.5, 12:8, 14:8.5, 16:9, 18:9.5, 20:10, 21:10.5},
            {8:6+1/3, 10:6+1/3, 12:6+2/3, 14:6+2/3},
        ]
    if name == 'ex110':
        ex_max = [1,1,0]
        rmin_per_iflow = [
            {8:9, 10:9, 12:10.5, 14:11, 16:11.5, 18:11.5, 20:12, 21:13},
            {8:6.5, 10:7, 12:7.5, 14:8, 16:9, 18:9.5, 20:10, 21:11},
            {8:6+1/3, 10:6+1/3, 12:6+2/3, 14:6+2/3},
        ]

    if name == 'ex1':
        ex_max = [1,1,1]
        rmin_per_iflow = [
            {8:9, 10:9, 12:10.5, 14:11, 16:11.5, 18:11.5, 20:12, 21:13},
            {8:6.5, 10:7, 12:7.5, 14:8, 16:9, 18:9.5, 20:10, 21:11},
            {8:6+1/3, 10:6+1/3, 12:6+2/3, 14:6+2/3},
        ]

    def iflows_mats_name(nm : str) -> dict[int, list[int]]:
        if nm[:2] == 'gs':
            ifl_max = int(nm[2:]) if len(nm)>2 else 18
            return {
                0 : [ifl for ifl in [8,10,12,14,16,18,20,21] if ifl <= ifl_max],
                1 : [ifl for ifl in [8,10,12,14,16,18,20,21] if ifl <= ifl_max],
                2 : [ifl for ifl in [8,10,12,14] if ifl <= ifl_max]
            }
        if nm == 'ex110':
            return {
                0 : [8,10,12,14,16,18],
                1 : [8,10,12,14,16,18],
                2 : [8,10,12,14]
            }
        if nm == 'ex1':
            return {
                0 : [8,10,12,14,16,18],
                1 : [8,10,12,14,16,18],
                2 : [8,10,12,14]
            }
        raise Exception(f'bad name: {nm}')

    iflows_mats = iflows_mats_name(name)

 

    if do_comp:

        if name not in bs_data:
            bs_data[name] = []

        for i_bs in range(n_bs):
            print(f'bootstrap sample {i_bs+1}/{n_bs}...')

            fss = FitSubSum(bs=True)
            fss.iflows = iflows_mats

            for mats in (0,1,2):
                fss.do_fit_one_mat_diff_rmin(mats, rmin_per_iflow[mats], ex_max=ex_max[mats], printfits=False)

            pmeans = [fss.fitstuff_mats[mats].fit.pmean for mats in (0,1,2)]

            fss.do_sums_for_mats(mats_vals=(0,1,2), printsums=False)
            tsums01  = fss.do_sums_for_sub_from_mats((0,1))
            tsums012 = fss.do_sums_for_sub_from_mats((0,1,2))

            bs_data[name].append((tsums01, tsums012, pmeans, (fss.tsums_sinh, fss.rleft_sinh, fss.rright_sinh)))

            with pickle_path.open('wb') as f:
                pickle.dump(bs_data, f)

    print()
    print(*((name, len(bsd)) for name,bsd in bs_data.items()))
    print()



    # fit parameters
    ####################################
    if 1:

        masses_ense = np.array(
            [ [ [pmeans[mats][('m' if ex==0 else 'dm') + f'_{ex}_{mats}'] for ex in range(min(ex_max)+1)]
                for mats in (0,1,2) ]
            for _,_,pmeans,_ in bs_data[name] ],
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



    # plot tsums
    ###############################################
    if 1:

        nrows, ncols = 1, 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 5*nrows))
        axes = np.reshape(axes, shape=(nrows, ncols))

        subs  = [(0,1), (0,1,2)]
        names = 'gs12 gs14 gs16 gs ex110 ex1'.split()
        names = 'ex1 ex110 gs gs20 gs21'.split()


        for isub, sub in enumerate(subs):

            for iname, name in enumerate(names):

                iflows = sorted(set.intersection(*(set(iflows_mats_name(name)[mats]) for mats in sub)))
                x_list = [flt + iname*0.01 for flt in flowtimes[iflows]]
                
                tsums_ense = { iflow : np.array([(tsums01 if sub==(0,1) else tsums012)[iflow] for tsums01,tsums012,_,_ in bs_data[name]])
                            for iflow in iflows }
                tsums_data = gv.dataset.avg_data(tsums_ense, bstrap=True)
                y_list = [tsums_data[ifl] for ifl in iflows]

                axes[0,isub].errorbar(
                    x=x_list, y=gv.mean(y_list), yerr=gv.sdev(y_list),
                    marker='o', linestyle='none', label=name,
                )

            axes[0,isub].set_xlabel(r'flowtime $t / a^2$')
            axes[0,isub].set_ylabel(rf'$H_E^{{{sub}}} / T^4$')
            axes[0,isub].set_xlim(0, max(x_list) * 1.1)
            axes[0,isub].set_ylim(min(y.mean - y.sdev for y in y_list) * 1.1, 0)
            axes[0,isub].legend()


        fig.tight_layout()
        fig.savefig(Path.cwd() / 'zeugs' / 'plots' / f'mats_sums_many.png', dpi=400)






# creation of plot of  m over r_0  fit for one-mass fit
def mass_rmin_bs():

    n_bs        = 100
    do_comp     = False
    pickle_path = Path.cwd() / 'zeugs' / 'data' / 'bs_rmin_fits.npy'

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
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(wlatex, wlatex/ncols*0.99))
    axes = np.reshape(axes, shape=(nrows, ncols))


    for mats in (0,1,2):
        ax = axes[0,mats]
        ax.errorbar(
            x    = r_mins,
            y    = gv.mean(masses_data[mats, :]),
            yerr = gv.sdev(masses_data[mats, :]),
            marker     = 'o',
            markersize = 4,
            linestyle  = 'none' 
        )
        ax.set_xlim([(4.9,12.1),(4.9,10.1),(4.9,7.4)][mats])
        ax.set_ylim([(4.5,13.9),(8,21),(12,26)][mats])
        ax.set_xlabel(r'$r_0 / a$')
        ax.set_ylabel(None)
        ax.text(
            0.05, 0.95,
            rf'$m_{mats} / T$',# + f' (mode {mats})',
            transform=ax.transAxes,
            ha="left",
            va="top"
        )


    fig.tight_layout(w_pad=0.5)
    fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'bootstrap_rmin_fits.pdf', dpi=400)







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





# creation of figures for one- and two-mass fits and their plots (including data and fit for mats=0,1,2)
def mass_rmin_manym():

    # decent Q's
    if 0:
        name       = 'flow3'
        ex_max     = 1
        iflows     = [8,10,12]
        r_min_left = [6+2/3, 6+2/3, 6+2/3]
    # add iflow 14
    # used for Fig of fits for two-mass fit
    if 0:
        name       = 'flow4'
        ex_max     = 1
        iflows     = [8,10,12,14]
        r_min_left = [6+2/3, 6+2/3, 6+2/3]
    # only ground state
    # used for Fig of fits for one-mass fit
    if 1:
        name       = 'gs'
        ex_max     = 0
        iflows     = [8,10,12,14]
        r_min_left = [10, 7.7, 6.7]
        

    fss = FitSubSum(bs=False)
    fss.iflows = iflows



    if 0:

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



    

    fss.do_fits_for_all_mats_manym(ex_max, r_min_left, printfits=False)


    nrows, ncols = 1, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(wlatex, wlatex/ncols*0.99))
    axes = np.reshape(axes, shape=(nrows, ncols))


    ifl = 12
    for mats in (0,1,2):
        ax = axes[0,mats]

        r_left, r_right = fss.fitstuff_mats[mats].rlims[ifl,mats]
        ir_left, ir_right = index_from_distance(fss.dist, r_left), index_from_distance(fss.dist, r_right)

        plot_dist(fss.fitstuff_mats[mats].fit.x[ifl,mats], fss.fitstuff_mats[mats].fit.y[ifl,mats], ax, label=f'data (n={mats})', alpha=0.7)
        plot_fitfcn(fss.dist[ir_left:ir_right], fss.fitfcn_mats(ifl, mats, fss.dist[ir_left:ir_right]), ax, label=f'fit (n={mats})')

        ax.set_xlabel(r'$r / a$')
        ax.set_xlim({
            'gs'    : [(9,23), (7,17), (6.2,11)],
            'flow4' : [(5,19), (5,16), (6,12)],
        }[name][mats])

        ax.set_ylim({
            'gs'    : [(-0.026,0.0025), (-0.038,0.0025), (-0.025,0.0025)],
            'flow4' : [(-0.25,0.02), (-0.11, 0.01), (-0.025, 0.002)],
        }[name][mats])
        ax.set_ylabel(None)
        if mats==0:
            ax.text(0.05, 0.95, r'$G / T^7$', transform=ax.transAxes, ha="left", va="top")

        # ax.tick_params(axis="y", direction="in", pad=-18-11, right=True, labelright=False)

        ax.legend(loc='lower right')



    fname = {'gs' : 'gs', 'flow4' : 'ex1'}[name]
    fig.tight_layout(w_pad=0.1)
    fig.savefig(Path.cwd() / 'zeugs' / 'plots' / f'mats_fits_{fname}.pdf', dpi=400)





    if 0:

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




# creation of plot of  m and dm and Q over r_0  for two-mass fit
def mass_rmin_manym_bs():
    
    n_bs        = 200
    do_comp     = False
    pickle_path = Path.cwd() / 'zeugs' / 'data' / 'bs_rmin_fits_ex1.npy'

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
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(wlatex, wlatex/ncols*0.99), sharey=True)
    axes = np.reshape(axes, shape=(nrows, ncols))


    for mats in (0,1,2):
        for ex in range(ex_max+1):
            axes[0,mats].errorbar(
                x    = r_mins,
                y    = gv.mean(masses_data[ex, mats, :]),
                yerr = gv.sdev(masses_data[ex, mats, :]),
                marker    = 'o',
                markersize= 3,
                linestyle = '--' ,
                label = rf'$m_{mats}$' if ex==0 else rf'$\Delta m_{mats}$'
            )
        axes[0,mats].set_xlim([(2.9,10.1),(2.9,9.1),(2.9,7.4)][mats])
        axes[0,mats].set_xlabel('r_min / a')
        axes[0,mats].set_ylabel(f'(d)m / T')
        axes[0,mats].legend()




    # do Q values
    if 1:

        do_comp     = False
        pickle_path = Path.cwd() / 'zeugs' / 'data' / 'bs_rmin_fits_ex1_Q.pkl'

        if do_comp:
            fss = FitSubSum(bs=False)
            fss.iflows = iflows
            fss.r_min_left_list_mats = tuple(r_mins for _ in range(3))
            res = fss.do_fits_for_many_r_min_lefts_for_all_mats('var', ex_max)
            Q_vals = [[fitstuff.fit.Q for fitstuff in res[mats]] for mats in (0,1,2)]
            with pickle_path.open('wb') as f:
                pickle.dump(Q_vals, f)
        else:
            with open(pickle_path, 'rb') as f:
                Q_vals = pickle.load(f)

            

        for mats in (0,1,2):
            ax = axes[0,mats]
            ax_twin = ax.twinx()
            ax_twin.plot(
                r_mins,
                Q_vals[mats],
                marker     = 's',
                markersize = 3,
                linestyle  = '--',
                alpha      = 0.7,
                label      = r'$Q$',
                color      = 'plum',
            )
            ax_twin.set_ylim(0, 1)

            # show the Q axis only on the rightmost subplot
            if mats < 2:
                ax_twin.set_ylabel(None)
                ax_twin.tick_params(right=False, labelright=False)
                ax_twin.spines["right"].set_visible(False)
            else:
                ax_twin.set_ylabel(None)
                ax_twin.tick_params(right=True, labelright=True)
                ax_twin.spines["right"].set_visible(True)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax.legend(
                lines1+lines2, labels1+labels2,
                loc=['upper center', 'upper center', 'lower left'][mats],
                labelspacing=0.25,
                bbox_to_anchor=[(0.50, 1.00), (0.62, 1.00), (0.10, 0.0)][mats],
            )



    for mats in (0,1,2):
        ax = axes[0,mats]
        ax.set_ylabel(None)
        if mats==0:
            ax.text(
                0.05, 0.95,
                r'$m / T$',# + f' (mode {mats})',
                transform=ax.transAxes,
                ha="left",
                va="top"
            )
        if mats==2:
            ax.text(
                0.95, 0.95,
                r'$Q$',
                transform=ax.transAxes,
                ha="right",
                va="top"
            )
        ax.set_xlabel(r'$r_0 / a$')
        ax.set_ylim((0,33))
        # ax.set_title(rf'Matsubara mode {mats}')



    fig.tight_layout(w_pad=0.25)
    fig.savefig(Path.cwd() / 'zeugs' / 'plots' / 'bootstrap_rmin_fits_ex1.pdf', dpi=400)








def fits_manym():

    fss = FitSubSum(bs=False)

    # decent Q's
    if 0:
        ex_max = 1
        fss.iflows = [8,10,12]
        r_min_left = [6+2/3, 6+2/3, 6+2/3]
    # add iflow 14
    if 1:
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
    
    fss.plot_subtraction_fits((0,1),   axes=axes2[:,0], ylim=(-0.2, 0.6))
    fss.plot_subtraction_fits((0,1,2), axes=axes2[:,2], ylim=(-0.3, 0.6))

    fss.plot_partial_sums((0,1),   axes=axes2[:,1])
    fss.plot_partial_sums((0,1,2), axes=axes2[:,3])

    fig2.tight_layout()
    fig2.savefig(Path.cwd() / 'zeugs' / 'plots' / 'mats_sub_fits_sums.png', dpi=400)





def fits_manym_bs():

    n_bs        = 50
    do_comp     = True
    pickle_path = Path.cwd() / 'zeugs' / 'data' / 'bs_fits_ex.pkl'


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
    if 1:
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
    if 0:
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
    if 1:

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
    if 1:

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
                axes[iiflow,3*isub].set_title(rf'flowtime $= {flowtimes[iflow]:.2} a^2$')

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
            axes[0,3*isub+2].set_xlabel(r'flowtime $t / a^2$')
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
    if 0:

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















main = mass_rmin




if __name__ == '__main__':

    main()