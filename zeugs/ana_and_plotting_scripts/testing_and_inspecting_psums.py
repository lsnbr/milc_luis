from import_and_plotting_tech import *

from some_intermediate_results import rmin_per_iflow_mats_exmax




ex_max = 0

fss = FitSubSum(bs=False, psums_binsize=0.2)
fss.iflows[0] = [11,12,14,16,18]
fss.iflows[1] = [11,12,14,16,18]
fss.iflows[2] = [11,12,14]

for mats in (0,1,2):
    rmin_per_iflow = rmin_per_iflow_mats_exmax[ex_max][mats]
    fss.do_fit_one_mat_diff_rmin(mats, rmin_per_iflow, ex_max=ex_max, printfits=True)
    fss.do_sums_for_mats(mats_vals=mats, printsums=True)

fss.do_sums_for_sub_from_mats(sub=(0,1))
fss.do_sums_for_sub_from_mats(sub=(0,1,2))




fig, axs = plt.subplots(2, 2, figsize=(2*wlatex, 2*hlatex), constrained_layout=True)
fls = lambda ifl: f'{flowtimes[ifl]:.2f} a^2'




for mats in (0,1):
    iflow = 18
    color = ['blue', 'red'][mats]

    ax = axs[0,0]
    plot_dist(fss.dist_psums, fss.psums_sinh[iflow, mats], ax, markersize=3, linestyle=':', label=f'$n={mats} (t={fls(iflow)})$', color=color)
    ax.set_ylabel(f'partial sums' + r'$ / T^4$')
    ax.axvline(x=fss.rleft_sinh[ iflow, mats], color=color, alpha=0.5)
    ax.axvline(x=fss.rright_sinh[iflow, mats], color=color, alpha=0.5)
    ax.set_xlim(0, 15)
    ax.set_xlabel(None)

    ax = axs[1,0]
    ense_int    = build_integrand(fss.dist, fss.ense_all[:, iflow, mats, :], mats)
    ense_int_b, = bin_averages(find_distance_bins(fss.dist, fss.psums_binsize), ense_int)
    data_int_b  = gv.dataset.avg_data(ense_int_b)
    plot_dist(fss.dist_psums, data_int_b * fss.dist_psums**2 / nt**2, ax, label=f'$n={mats} (t={fls(iflow)})$', color=color, alpha=0.7, linestyle=':')
    ax.axvline(x=fss.rleft_sinh[ iflow, mats], color=color, alpha=0.5)
    ax.axvline(x=fss.rright_sinh[iflow, mats], color=color, alpha=0.5)
    ax.set_ylabel(r'$h_{E,t}(\omega_n, r) \cdot r^2 / T^5$')
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.13, 0.23)




for mats in (0,1,2):
    ax = axs[0,1]
    iflow = 14
    color = ['blue', 'red', 'green'][mats]

    plot_dist(fss.dist_psums, fss.psums_sinh[iflow, mats], ax, markersize=3, linestyle=':', label=f'$n={mats} (t={fls(iflow)})$', color=color)
    ax.axvline(x=fss.rleft_sinh[ iflow, mats], color=color, alpha=0.5)
    ax.axvline(x=fss.rright_sinh[iflow, mats], color=color, alpha=0.5)
    ax.set_xlim(0, 15)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    ax = axs[1,1]
    ense_int    = build_integrand(fss.dist, fss.ense_all[:, iflow, mats, :], mats)
    ense_int_b, = bin_averages(find_distance_bins(fss.dist, fss.psums_binsize), ense_int)
    data_int_b  = gv.dataset.avg_data(ense_int_b)
    plot_dist(fss.dist_psums, data_int_b * fss.dist_psums**2 / nt**2, ax, label=f'$n={mats} (t={fls(iflow)})$', color=color, alpha=0.7, linestyle=':')
    ax.axvline(x=fss.rleft_sinh[ iflow, mats], color=color, alpha=0.5)
    ax.axvline(x=fss.rright_sinh[iflow, mats], color=color, alpha=0.5)
    ax.set_ylabel(None)
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.3, 0.6)

if 0:   # plot mats=3 data (bottom right)
    ense_int    = build_integrand(fss.dist, fss.ense_all[:, 14, 3, :], 3)
    ense_int_b, = bin_averages(find_distance_bins(fss.dist, fss.psums_binsize), ense_int)
    data_int_b  = gv.dataset.avg_data(ense_int_b)
    plot_dist(fss.dist_psums, data_int_b * fss.dist_psums**2 / nt**2, axs[1,1], label=f'$n=3 (t={fls(14)})$', color='orange', alpha=0.7, linestyle=':')




out = Path.cwd() / 'zeugs' / 'plots' / 'test_psums.pdf'
fig.savefig(out, bbox_inches='tight', dpi=400)
plt.close(fig)
print(f"Saved: {out}")


