import numpy as np
from import_and_plotting_tech import *


from some_intermediate_results import rmin_per_iflow_mats_exmax





nrows, ncols = 2, 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(wlatex, wlatex/ncols*nrows))
axes = np.reshape(axes, shape=(nrows, ncols))




# gs (ex_max=0) fit
if 1:
    ex_max = 0
    iflows_for_mats = [(11,18), (11,18), (11,14)]
    xlim_mats = [(8,30), (6,20), (5,15)]
    ylim_iifl_mats = [[(-0.014,0.002),(-0.0055,0.0005)], [(-0.0325, 0.005),(-0.0085,0.001)], [(-0.0325,0.005),(-0.025,0.0025)]]

# ex1 (ex_max=1) fit
if 0:
    ex_max = 1
    iflows_for_mats = [(8,18), (8,18), (8,14)]
    xlim_mats = [(4.5,20), (4.5,18), (4.5,15)]
    ylim_iifl_mats = [[(-0.38,0.02),(-0.08,0.005)], [(-0.12,0.01),(-0.018,0.0018)], [(-0.029,0.0025),(-0.027,0.002)]]




fss = FitSubSum(bs=False)
fss.iflows[0] = [11,12,14,16,18]
fss.iflows[1] = [11,12,14,16,18]
fss.iflows[2] = [11,12,14]




for mats in (0,1,2):
    fss.do_fit_one_mat_diff_rmin(mats, rmin_per_iflow_mats_exmax[ex_max][mats], ex_max=ex_max, printfits=True)

    for iifl, ifl in enumerate(iflows_for_mats[mats]):
        ax = axes[iifl, mats]

        r_left, r_right = fss.fitstuff_mats[mats].rlims[ifl,mats]
        ir_left, ir_right = index_from_distance(fss.dist, r_left), index_from_distance(fss.dist, r_right)

        plot_dist(fss.fitstuff_mats[mats].fit.x[ifl,mats], fss.fitstuff_mats[mats].fit.y[ifl,mats], ax, label=f'data', alpha=0.7)
        plot_fitfcn(fss.dist[ir_left:ir_right], fss.fitfcn_mats(ifl, mats, fss.dist[ir_left:ir_right]), ax, label=f'fit')

        ax.set_xlabel(r'$r / a$')
        ax.set_xlim(xlim_mats[mats])

        ax.set_ylim(ylim_iifl_mats[mats][iifl])
        ax.set_ylabel(None)
        ax.text(0.04, 0.96, r'$G / T^7$', transform=ax.transAxes, ha="left", va="top")

        # ax.tick_params(axis="y", direction="in", pad=-18-11, right=True, labelright=False)

        ax.legend(loc='lower right')
        ax.set_title(f'$n={mats}$, $t={flowtimes[ifl]:.2f}a^2$')




fig.tight_layout(w_pad=0.5)
fig.savefig(Path.cwd() / 'zeugs' / 'plots' / f'fit_data_plot_ex{ex_max}.pdf', dpi=400)