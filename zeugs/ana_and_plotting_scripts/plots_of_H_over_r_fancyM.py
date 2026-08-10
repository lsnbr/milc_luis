from import_and_plotting_tech import *
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


from ensemble_data import data_16_x_64_1p5Tc as data_ense
nt = data_ense['nt']
ns = data_ense['ns']
flowtimes = data_ense['flowtimes']


from some_intermediate_results import rmin_per_iflow_mats_exmax




ex_max = 0
mats   = 1
iflow  = 18

iflowB   = 14
do_flowB = True

fss = FitSubSum(bs=False)
fss.iflows[0] = [12,14,16,18]
fss.iflows[1] = [12,14,16,18]
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
if do_flowB:
    r_leftB, r_rightB = fss.fitstuff_mats[mats].rlims[iflowB, mats]
    ir_leftB = index_from_distance(fss.dist, r_leftB)
    ir_rightB = index_from_distance(fss.dist, r_rightB)

# data integrand (binned)
ense_int = build_integrand(fss.dist, fss.ense_all[:, iflow, mats, :], mats)
ense_int_b, = bin_averages(cbins, ense_int)
data_int_b = gv.dataset.avg_data(ense_int_b)
if do_flowB:
    ense_intB = build_integrand(fss.dist, fss.ense_all[:, iflowB, mats, :], mats)
    ense_int_bB, = bin_averages(cbins, ense_intB)
    data_int_bB = gv.dataset.avg_data(ense_int_bB)

# plot data + fit
if do_flowB:
    plot_dist(dist_b, data_int_bB * dist_b**2 / nt**2, ax, label='dataB', alpha=0.7)
    xfitB = fss.dist[ir_leftB:ir_rightB]
    yfitB = fss.fitfcn_sinh(iflowB, mats, xfitB)
plot_dist(dist_b, data_int_b * dist_b**2 / nt**2, ax, label='data', alpha=0.7)
xfit = fss.dist[ir_left:ir_right]
yfit = fss.fitfcn_sinh(iflow, mats, xfit)
# plot_fitfcn(xfit, yfit * xfit**2 / nt**2, ax, label='fit')

# optional rleft/rright markers if present
# if (iflow, mats) in fss.rleft_sinh:
#     ax.axvline(x=fss.rleft_sinh[iflow, mats], color='black', alpha=0.7)
# if (iflow, mats) in fss.rright_sinh:
#     ax.axvline(x=fss.rright_sinh[iflow, mats], color='black', alpha=0.7)

ax.set_xlim(-0.5, 25)
ax.set_ylim(-0.25, 0.55)



# --- inset zoom ---
axins = ax.inset_axes([0.3, 0.45, 0.5, 0.5])  # [x0, y0, width, height] in axes fraction coords

# redraw the same data on the inset
if do_flowB:
    plot_dist(dist_b+0.05, data_int_bB * dist_b**2 / nt**2, axins, label=None, alpha=0.7)
    plot_fitfcn(xfitB, yfitB * xfitB**2 / nt**2, axins, color='pink', alpha=0.99, label=None)
plot_dist(dist_b, data_int_b * dist_b**2 / nt**2, axins, label=None, alpha=0.7)
plot_fitfcn(xfit, yfit * xfit**2 / nt**2, axins, label=None)
axins.set_xlabel('')
axins.set_ylabel('')

# vertical lines
if do_flowB:
    axins.axvline(x=fss.rleft_sinh[iflowB, mats], color='pink', alpha=0.99)
    axins.axvline(x=fss.rright_sinh[iflowB, mats], color='pink', alpha=0.99)
axins.axvline(x=fss.rleft_sinh[iflow, mats], color='red', alpha=0.5)
axins.axvline(x=fss.rright_sinh[iflow, mats], color='red', alpha=0.5)

# define the zoomed region
axins.set_xlim(7.5, 14)
axins.set_ylim(-0.03, 0.005)
axins.tick_params(axis='both', which='both', labelsize=6)

# draw the box on the main plot and connecting lines to the inset
mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5", lw=0.8)



ax.set_xlabel(r'$r/a$')
ax.set_ylabel(r'$h_{E,t}(\omega_n, r) \cdot r^2 / T^5$')
ax.legend(
    loc='lower right',
    frameon=True,
    facecolor='white',
    edgecolor='none',
    framealpha=0.7,
)
ax.set_title(r'$n=1 \qquad t_\mathrm{f} = 0.59, 0.94 a^2$')

out = Path.cwd() / 'zeugs' / 'plots' / 'test_H_plotM.pdf'
fig.savefig(out, bbox_inches='tight', dpi=400)
plt.close(fig)
print(f"Saved: {out}")