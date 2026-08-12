from import_and_plotting_tech import *
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from ensemble_data import data_16_x_64_1p5Tc as data_ense
nt = data_ense['nt']
ns = data_ense['ns']
flowtimes = data_ense['flowtimes']

from some_intermediate_results import rmin_per_iflow_mats_exmax




# which correlators to plot
ex_max      = 0
mats_list   = [1,2]
iflows_mats = {1:[14,18], 2:[10,14]}

# prepare figures
fig, axs = plt.subplots(1, len(mats_list), figsize=(wlatex, 0.8*hlatex), constrained_layout=True)

# prepare fitting and summing (summing since there r1 is computed)
fss = FitSubSum(bs=False)
fss.iflows[0] = [12,14,16,18]
fss.iflows[1] = [12,14,16,18]
fss.iflows[2] = [8,10,12,14]

# constant sized bins
cbin_size = 0.25
cbins = find_distance_bins(fss.dist, cbin_size)
dist_b, = bin_averages(cbins, fss.dist)



# do one plot for each mats
for j, mats in enumerate(mats_list):
    # the pair of flowtimes to plot
    iflows  = iflows_mats[mats]

    # perform fit and sum for this matsubara mode
    rmin_per_iflow = rmin_per_iflow_mats_exmax[ex_max][mats]
    fss.do_fit_one_mat_diff_rmin(mats, rmin_per_iflow, ex_max, printfits=True)
    fss.do_sums_for_mats(mats_vals=mats, printsums=True)

    # normal and zoomin axes
    ax = axs[j]
    axins = ax.inset_axes([0.45, 0.5, 0.5, 0.45])  # [x0, y0, width, height] in axes fraction coords
    fit_handles = []  # proxy legend entries for fit plots



    # loop over all flowtimes we want to plot
    for k, iflow in enumerate(iflows):

        # some parameters
        flow       = flowtimes[iflow]     # flowtime in a^2 units
        color_data = ['blue', 'green'][k]
        color_fit  = ['orange', 'red'][k]
        r_offset   = 0.04 * k

        # fit range
        r_left, r_right = fss.fitstuff_mats[mats].rlims[iflow, mats]
        ir_left  = index_from_distance(fss.dist, r_left)
        ir_right = index_from_distance(fss.dist, r_right)

        # data integrand (binned)
        ense_int    = build_integrand(fss.dist, fss.ense_all[:, iflow, mats, :], mats)
        ense_int_b, = bin_averages(cbins, ense_int)
        data_int_b  = gv.dataset.avg_data(ense_int_b)

        # plot data and fit
        plot_dist(dist_b+r_offset, data_int_b * dist_b**2 / nt**2, ax, label=rf'data ($t_\mathrm{{f}}={flow:.2f}a^2$)', color=color_data, alpha=0.7)
        xfit = fss.dist[ir_left:ir_right]
        yfit = fss.fitfcn_sinh(iflow, mats, xfit)
        plot_fitfcn(xfit, yfit * xfit**2 / nt**2, ax, label=None, color=color_fit)

        # plot again on zoomed in view
        plot_dist(dist_b+r_offset, data_int_b * dist_b**2 / nt**2, axins, label=None, color=color_data, alpha=0.7)
        fit_line = plot_fitfcn(xfit, yfit * xfit**2 / nt**2, axins, label=None, color=color_fit)
        fit_handles.append(mpatches.Patch(color=color_fit, alpha=0.5, label=rf'fit ($t_\mathrm{{f}}={flow:.2f}a^2$)'))

        # vertical lines at r0 and r1
        axins.axvline(x=fss.rleft_sinh[iflow, mats],  color=color_fit, alpha=0.5)
        axins.axvline(x=fss.rright_sinh[iflow, mats], color=color_fit, alpha=0.5)



    # define outside plot limits
    ax.set_xlim({1:(-0.45,18),   2:(-0.45,11) }[mats])
    ax.set_ylim({1:(-0.35,0.65),  2:(-0.9,1.45)}[mats])

    # pull y-tick numbers to the inside of the panel for outside plots
    ax.tick_params(axis='y', direction='in', pad=-5)
    plt.setp(ax.get_yticklabels(), ha='left')

    # define the zoomed region
    axins.set_xlim({1:(7.5,14),       2:(6,9)        }[mats])
    axins.set_ylim({1:(-0.027,0.005), 2:(-0.11,0.017)}[mats])
    axins.tick_params(axis='both', which='both', labelsize=6)

    # draw the box on the main plot and connecting lines to the zoomin
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5", lw=0.8)

    # no axis labels for zoomin plot
    axins.set_xlabel('')
    axins.set_ylabel('')

    # same xaxis labels and shared yaxis label
    ax.set_xlabel(r'$r/a$')
    if j==0: ax.set_ylabel(r'$h_{E,t}(\omega_n, r) \cdot r^2 / T^5$')
    else:    ax.set_ylabel(None)

    # combine legends of data and fitfcn plots
    handles, labels = ax.get_legend_handles_labels()
    handles = fit_handles + handles
    labels  = [h.get_label() for h in fit_handles] + labels

    # build the legend
    ax.legend(
        handles=handles,
        labels=labels,
        loc='lower right',
        bbox_to_anchor=(0.9, 0.0),   # shift legend left slightly
        frameon=True,
        facecolor='white',
        edgecolor='none',
        framealpha=0.7,
        # the following are for compactifying the legend
        fontsize=8,
        handlelength=1.0,
        handletextpad=0.5,
        labelspacing=0.3,
        borderpad=0.3,
        borderaxespad=0.3,
    )

    # building the title
    flows_a2 = [f'{flowtimes[iflow]:.2f} a^2' for iflow in iflows]
    ax.set_title(rf'$n={mats} \qquad t_\mathrm{{f}} = {", ".join(flows_a2)}$')




# saving the figure
out = Path.cwd() / 'zeugs' / 'plots' / 'H_plot_lots.pdf'
fig.savefig(out, bbox_inches='tight', dpi=400)
plt.close(fig)
print(f"Saved: {out}")