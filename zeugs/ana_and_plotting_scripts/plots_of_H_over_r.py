import numpy as np
from import_and_plotting_tech import *



from ensemble_data import data_16_x_64_1p5Tc as data_ense
nt = data_ense['nt']
ns = data_ense['ns']
flowtimes = data_ense['flowtimes']


from some_intermediate_results import rmin_per_iflow_mats_exmax






ex_max = 1
matss  = (1,2)
iflows = (8,14)

fss = FitSubSum(bs=False)
fss.iflows[1] = [8,10,12,14,16,18]
fss.iflows[2] = [8,10,12,14]



for mats in matss:
    rmin_per_iflow = rmin_per_iflow_mats_exmax[ex_max][mats]
    fss.do_fit_one_mat_diff_rmin(mats, rmin_per_iflow, ex_max, printfits=True)
fss.do_sums_for_mats(mats_vals=(1,2), printsums=False)



# --- 2x2 plot for selected mats and iflows ---
nrows, ncols = len(iflows), len(matss)
fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols,
    figsize=(wlatex, 0.8 * wlatex),
    constrained_layout=True
)
axes = np.reshape(axes, (nrows, ncols))

cbin_size = 0.25
cbins = find_distance_bins(fss.dist, cbin_size)
dist_b, = bin_averages(cbins, fss.dist)

for j, mats in enumerate(matss):
    for i, iflow in enumerate(iflows):
        ax = axes[i, j]

        # fit range
        r_left, r_right = fss.fitstuff_mats[mats].rlims[iflow, mats]
        ir_left = index_from_distance(fss.dist, r_left)
        ir_right = index_from_distance(fss.dist, r_right)

        # data integrand (binned)
        ense_int = build_integrand(fss.dist, fss.ense_all[:, iflow, mats, :], mats)
        ense_int_b, = bin_averages(cbins, ense_int)
        data_int_b = gv.dataset.avg_data(ense_int_b)

        # plot data + fit
        plot_dist(dist_b, data_int_b, ax, label='data', alpha=0.7)
        xfit = fss.dist[ir_left:ir_right]
        yfit = fss.fitfcn_sinh(iflow, mats, xfit)
        plot_fitfcn(xfit, yfit, ax, label='fit')

        # optional rleft/rright markers if present
        if (iflow, mats) in fss.rleft_sinh:
            ax.axvline(x=fss.rleft_sinh[iflow, mats], color='black', alpha=0.7)
        if (iflow, mats) in fss.rright_sinh:
            ax.axvline(x=fss.rright_sinh[iflow, mats], color='black', alpha=0.7)

        # cosmetics
        ax.set_xlim([(5.5,20), (5.5,15), (5.5,15)][mats])
        if i == 1: ax.set_xlabel('$r/a$')
        else:      ax.set_xlabel(None)
        if j == 0: ax.set_ylabel('$h_{E,t}(\\omega_n, r) / T^7$')
        else:      ax.set_ylabel(None)

        ax.set_ylim([(-0.02, 0.002), (-0.22, 0.05), (-0.95, 0.25)][mats])

        tf = flowtimes[iflow]
        rf = flowtime_to_radius(tf, nt) * nt
        # ax.set_title(rf'$n={mats}$, $t_f={tf:.2f}a^2$ ($r_f={rf:.2f}a$)')
        ax.set_title(rf'$n={mats}$, $t={tf:.2f}a^2$')

        ax.legend(
            loc='lower right',
            frameon=True,
            facecolor='white',
            edgecolor='none',
            framealpha=0.7,
            # fontsize=8,
            # labelspacing=0.2,
            # borderpad=0.25,
            # handletextpad=0.4,
            # handlelength=1.2,
        )

out = Path.cwd() / 'zeugs' / 'plots' / 'mats_sinh_fits_2x2.pdf'
fig.savefig(out, bbox_inches='tight', dpi=400)
plt.close(fig)
print(f"Saved: {out}")