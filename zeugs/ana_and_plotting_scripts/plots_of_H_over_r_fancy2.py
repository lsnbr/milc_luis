from import_and_plotting_tech import *
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


from ensemble_data import data_16_x_64_1p5Tc as data_ense
nt = data_ense['nt']
ns = data_ense['ns']
flowtimes = data_ense['flowtimes']


from some_intermediate_results import rmin_per_iflow_mats_exmax




ex_max = 0
mats   = 0
iflow  = 18

fss = FitSubSum(bs=False)
fss.iflows[0] = [12,14,16,18]
fss.iflows[1] = [8,10,12,14,16,18]
fss.iflows[2] = [8,10,12,14]

rmin_per_iflow = rmin_per_iflow_mats_exmax[ex_max][mats]
fss.do_fit_one_mat_diff_rmin(mats, rmin_per_iflow, ex_max, printfits=True)
fss.do_sums_for_mats(mats_vals=mats, printsums=True)



fig, ax = plt.subplots(figsize=(wlatex, hlatex), constrained_layout=True)

cbin_size = 0.25
cbins = find_distance_bins(fss.dist, cbin_size)
dist_b, = bin_averages(cbins, fss.dist)


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
# if (iflow, mats) in fss.rleft_sinh:
#     ax.axvline(x=fss.rleft_sinh[iflow, mats], color='black', alpha=0.7)
# if (iflow, mats) in fss.rright_sinh:
#     ax.axvline(x=fss.rright_sinh[iflow, mats], color='black', alpha=0.7)

ax.set_xlim(-0.5, 25)
# ax.set_ylim(-0.3, 0.6)



# --- inset zoom ---
axins = ax.inset_axes([0.3, 0.5, 0.5, 0.45])  # [x0, y0, width, height] in axes fraction coords

# redraw the same data on the inset
plot_dist(dist_b, data_int_b, axins, label=None, alpha=0.7)
plot_fitfcn(xfit, yfit, axins, label=None)
axins.set_xlabel('')
axins.set_ylabel('')

# vertical lines
axins.axvline(x=fss.rleft_sinh[iflow, mats], color='orange', alpha=0.7)
axins.axvline(x=fss.rright_sinh[iflow, mats], color='orange', alpha=0.7)

# define the zoomed region
axins.set_xlim(12, 21)
axins.set_ylim(-0.007, 0.0005)
axins.tick_params(axis='both', which='both', labelsize=6)

# draw the box on the main plot and connecting lines to the inset
mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5", lw=0.8)



ax.set_xlabel(r'$r/a$')
ax.set_ylabel(r'$h_{E,t}(\omega_n, r) / T^7$')
ax.legend(
    loc='lower right',
    frameon=True,
    facecolor='white',
    edgecolor='none',
    framealpha=0.7,
)

out = Path.cwd() / 'zeugs' / 'plots' / 'test_H_plot2.pdf'
fig.savefig(out, bbox_inches='tight', dpi=400)
plt.close(fig)
print(f"Saved: {out}")