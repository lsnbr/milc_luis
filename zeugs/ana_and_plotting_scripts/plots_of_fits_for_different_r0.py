from import_and_plotting_tech import *






# plots for gs case
if 1:
    ex_max = 0
    iflows_for_mats = [(10,18), (10,16), (10,14)]
    rminlist_for_mats = [np.arange(7, 18+1e-6, 1), np.arange(6, 11+1e-6, 1/2), np.arange(5, 8+1e-6, 1/3)]
# plots for ex1 case
if 0:
    ex_max = 1
    iflows_for_mats = [(12,18), (12,18), (10,14)]
    rminlist_for_mats = [np.arange(5, 10.5+1e-6, 1/2), np.arange(5, 10.5+1e-6, 1/2), np.arange(5, 7+2/3+1e-6, 1/3)]


nrows, ncols = 2, 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(wlatex, wlatex/ncols*nrows*1.2))
axes = np.reshape(axes, shape=(nrows, ncols))


for mats in (0,1,2):
    for iifl, ifl in enumerate(iflows_for_mats[mats]):
        ax = axes[iifl,mats]

        fss = FitSubSum(bs=False)
        rmins = rminlist_for_mats[mats]
        allfits = fss.do_fits_for_many_r_min_lefts(rmins, [ifl], mats, 'var', ex_max)
        masses_data = [fs.fit.p[f'm_0_{mats}'] for fs in allfits]

        if ex_max > 0:
            dmasses_data = [fs.fit.p[f'dm_1_{mats}'] for fs in allfits]

        Qs = [fs.fit.Q for fs in allfits]

        ax.errorbar(
            x    = rmins,
            y    = gv.mean(masses_data),
            yerr = gv.sdev(masses_data),
            marker     = 'o',
            markersize = 4,
            linestyle  = '--',
            label      = f'$m_{mats}$' 
        )

        if ex_max > 0:
            ax.errorbar(
                x    = rmins,
                y    = gv.mean(dmasses_data),
                yerr = gv.sdev(dmasses_data),
                marker     = 'o',
                markersize = 4,
                linestyle  = '--', 
                label      = f'$\\Delta m_{mats}$' 
            )

        ax.set_xlim(min(rmins)-0.1, max(rmins)+0.1)
        # ax.set_ylim([(4.5,13.9),(8,21),(12,26)][mats])
        ax.set_ylim([(0,30), (0,30)][ex_max])
        if iifl == nrows - 1:
            ax.set_xlabel(r"$r_0/a$")
        else:
            ax.set_xlabel(None)
        ax.set_ylabel(None)

        axQ = ax.twinx()
        axQ.plot(
            rmins,
            Qs,
            marker='s',
            markersize=3,
            linestyle='--',
            label = 'Q',
            alpha = 0.7,
            color = 'plum',
        )
        axQ.set_ylim(0, 1)
        axQ.set_ylabel(None)

        leftmost  = (mats == 0)
        rightmost = (mats == ncols - 1)
        if not leftmost:
            ax.set_ylabel(None)
            ax.tick_params(axis='y', which='both', labelleft=False)

        if not rightmost:
            axQ.set_ylabel(None)
            axQ.tick_params(axis='y', which='both', labelright=False)

        ax.text(
            0.04, 0.96,
            '$m / T$',# + f' (mode {mats})',
            transform=ax.transAxes,
            ha="left",
            va="top"
        )
        ax.text(
            0.96, 0.96,
            '$Q$',
            transform=ax.transAxes,
            ha="right",
            va="top"
        )

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axQ.get_legend_handles_labels()
        ax.legend(
            h1 + h2,
            l1 + l2,
            loc='upper center',
            frameon=True,
            facecolor='white',
            edgecolor='none',
            framealpha=0.8,

            fontsize=8,          # smaller text
            handlelength=1.2,    # shorter line samples
            handletextpad=0.4,   # space between marker and text
            labelspacing=0.3,    # vertical spacing between entries
            borderpad=0.3,       # padding inside legend box
            borderaxespad=0.3,   # distance to axes
            columnspacing=0.6
        )

        ax.set_title(f'$n={mats}$, $t={flowtimes[ifl]:.2f}a^2$')


fig.tight_layout(w_pad=0.5)
fig.savefig(Path.cwd() / 'zeugs' / 'plots' / f'rmin_fits_{"gs" if ex_max==0 else "ex1"}_pretty.pdf', dpi=400)