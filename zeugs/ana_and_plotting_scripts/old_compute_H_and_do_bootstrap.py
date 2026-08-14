import pickle
from import_and_plotting_tech import *






n_bs        = 5
do_comp     = False
pickle_path = Path.cwd() / 'zeugs' / 'data' / 'bs_subsums_many.pkl'


restart = False

# bs_data[name][bs] = (ifl -> tsum01, ifl -> tsum012, mats -> pmean, (sums_mats, rleft, rright))

if (not do_comp) or (not restart and pickle_path.exists()):
    with open(pickle_path, 'rb') as f:
        bs_data = pickle.load(f)
else:
    bs_data = {}



from some_intermediate_results import rmin_per_iflow_mats_exmax


name = 'gs-18'


if name == 'gs-18':
    ex_max = [0,0,0]
    iflows_mats = {0:[8,10,12,14,16,18], 1:[8,10,12,14,16,18], 2:[8,10,12,14]}
    rmin_per_iflow = rmin_per_iflow_mats_exmax[0]

if name == 'ex110-18':
    ex_max = [1,1,0]
    iflows_mats = {0:[8,10,12,14,16,18], 1:[8,10,12,14,16,18], 2:[8,10,12,14]}
    rmin_per_iflow = [rmin_per_iflow_mats_exmax[ex][mats] for mats,ex in enumerate(ex_max)]

if name == 'ex111-18':
    ex_max = [1,1,1]
    iflows_mats = {0:[8,10,12,14,16,18], 1:[8,10,12,14,16,18], 2:[8,10,12,14]}
    rmin_per_iflow = rmin_per_iflow_mats_exmax[1]

if name == 'gs-14':
    ex_max = [0,0,0]
    iflows_mats = {0:[8,10,12,14], 1:[8,10,12,14], 2:[8,10,12,14]}
    rmin_per_iflow = rmin_per_iflow_mats_exmax[0]
if name == 'gs-16':
    ex_max = [0,0,0]
    iflows_mats = {0:[8,10,12,14,16], 1:[8,10,12,14,16], 2:[8,10,12,14]}
    rmin_per_iflow = rmin_per_iflow_mats_exmax[0]
if name == 'gs-20':
    ex_max = [0,0,0]
    iflows_mats = {0:[8,10,12,14,16,18,20], 1:[8,10,12,14,16,18,20], 2:[8,10,12,14]}
    rmin_per_iflow = rmin_per_iflow_mats_exmax[0]
if name == 'gs-21':
    ex_max = [0,0,0]
    iflows_mats = {0:[8,10,12,14,16,18,20,21], 1:[8,10,12,14,16,18,20,21], 2:[8,10,12,14]}
    rmin_per_iflow = rmin_per_iflow_mats_exmax[0]

if name == 'ex111-14':
    ex_max = [1,1,1]
    iflows_mats = {0:[8,10,12,14], 1:[8,10,12,14], 2:[8,10,12,14]}
    rmin_per_iflow = rmin_per_iflow_mats_exmax[1]
if name == 'ex111-16':
    ex_max = [1,1,1]
    iflows_mats = {0:[8,10,12,14,16], 1:[8,10,12,14,16], 2:[8,10,12,14]}
    rmin_per_iflow = rmin_per_iflow_mats_exmax[1]
if name == 'ex111-20':
    ex_max = [1,1,1]
    iflows_mats = {0:[8,10,12,14,16,18,20], 1:[8,10,12,14,16,18,20], 2:[8,10,12,14]}
    rmin_per_iflow = rmin_per_iflow_mats_exmax[1]
if name == 'ex111-21':
    ex_max = [1,1,1]
    iflows_mats = {0:[8,10,12,14,16,18,20,21], 1:[8,10,12,14,16,18,20,21], 2:[8,10,12,14]}
    rmin_per_iflow = rmin_per_iflow_mats_exmax[1]

def iflows_mats_name(name : str) -> dict[int, list[int]]:
    iflowmax = int(name.split('-')[-1])
    ifls01 = [ifl for ifl in [8,10,12,14,16,18,20,21] if ifl <= iflowmax]
    return {0:ifls01, 1:ifls01, 2:[8,10,12,14]}

assert iflows_mats == iflows_mats_name(name)




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



# print fit parameters
####################################
if 0:

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
if 0:

    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 5*nrows))
    axes = np.reshape(axes, shape=(nrows, ncols))

    subs  = [(0,1), (0,1,2)]
    names = 'gs-18 ex110-18 ex111-18'.split()
    names = 'gs-14 gs-16 gs-18 gs-20 gs-21'.split()
    # names = 'ex111-14 ex111-16 ex111-18 ex111-20 ex111-21'.split()


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




# pretty plot for one-mass and two-mass comp (01 and 012)
####################################################################
if 0:

    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(wlatex, wlatex*0.5))
    axes = np.reshape(axes, shape=(nrows, ncols))

    names_left  = 'gs-18 ex111-18'.split()
    names_right = 'gs-18 ex110-18 ex111-18'.split()
    subs = [(0,1), (0,1,2)]


    for isub, sub in enumerate(subs):
        for iname, name in enumerate(names_left if isub==0 else names_right):
            ax = axes[0,isub]

            iflows = sorted(set.intersection(*(set(iflows_mats_name(name)[mats]) for mats in sub)))
            x_list = [flt + iname*0.005 for flt in flowtimes[iflows]]
            
            tsums_ense = { iflow : np.array([(tsums01 if sub==(0,1) else tsums012)[iflow] for tsums01,tsums012,_,_ in bs_data[name]])
                            for iflow in iflows }
            tsums_data = gv.dataset.avg_data(tsums_ense, bstrap=True)
            y_list = [tsums_data[ifl] for ifl in iflows]

            ax.errorbar(
                x=x_list, y=gv.mean(y_list), yerr=gv.sdev(y_list),
                marker='o', alpha=(1 if iname in (0,2) else 0.85), linestyle='none', label=(['one-mass fit', 'two-mass fit'][iname] if isub==0 else ['one-mass fit', 'mixed fit', 'two-mass fit'][iname]),
            )

        ax.set_xlabel(r'flowtime $t / a^2$')

        ax.set_ylabel(None)
        ax.text(0.05, 0.95, rf'$H_{{E,t}}^{{{sub}}} / T^4$', transform=ax.transAxes, ha="left", va="top")
    
        ax.set_xlim(0, max(x_list) * 1.1)

        if isub==0: ax.set_ylim(min(y.mean - y.sdev for y in y_list) * 1.1, 0)
        if isub==1: ax.set_ylim(-0.038, 0.012)

        ax.legend(
            loc='upper right',
            frameon=True,
            facecolor='white',
            edgecolor='none',
            framealpha=0.6
        )


    fig.tight_layout()
    fig.savefig(Path.cwd() / 'zeugs' / 'plots' / f'sub_corrs_01012.pdf', dpi=400)





# pretty plot for different included iflows
####################################################################
if 0:

    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(wlatex, wlatex*0.5))
    axes = np.reshape(axes, shape=(nrows, ncols))

    names_left  = 'gs-14 gs-16 gs-18 gs-20 gs-21'.split()
    names_right = 'ex111-14 ex111-16 ex111-18 ex111-20 ex111-21'.split()
    subs = [(0,1), (0,1)]


    for isub, sub in enumerate(subs):
        ax = axes[0,isub]

        for iname, name in enumerate(names_left if isub==0 else names_right):

            iflows = sorted(set.intersection(*(set(iflows_mats_name(name)[mats]) for mats in sub)))
            x_list = [flt + iname*0.005 for flt in flowtimes[iflows]]
            
            tsums_ense = { iflow : np.array([(tsums01 if sub==(0,1) else tsums012)[iflow] for tsums01,tsums012,_,_ in bs_data[name]])
                            for iflow in iflows }
            tsums_data = gv.dataset.avg_data(tsums_ense, bstrap=True)
            y_list = [tsums_data[ifl] for ifl in iflows]

            ax.errorbar(
                x=x_list, y=gv.mean(y_list), yerr=gv.sdev(y_list),
                marker='o', alpha=0.85, markersize=5, linestyle='none', label=f'$t_\\text{{max}} = {flowtimes[int(name.split("-")[1])]:.2f}a^2$',
            )

        ax.set_xlabel(r'flowtime $t / a^2$')

        ax.set_ylabel(None)
        ax.text(0.05, 0.95, rf'$H_{{E, t}}^{{{sub}}} / T^4$', transform=ax.transAxes, ha="left", va="top")
    
        ax.set_xlim(0, max(x_list) * 1.1)

        ax.set_ylim(-0.052, -0.01)

        ax.legend(
            loc='upper right',
            frameon=True,
            facecolor='white',
            edgecolor='none',
            framealpha=0.6,

            fontsize=8,          # smaller text
            handlelength=1.2,    # shorter line sample
            handletextpad=0.4,   # space between marker and text
            labelspacing=0.3,    # vertical spacing between entries
            borderpad=0.3,       # padding inside box
            borderaxespad=0.3    # distance from axes
        )

        ax.set_title(f'{"one" if isub==0 else "two"}-mass fits')


    fig.tight_layout()
    fig.savefig(Path.cwd() / 'zeugs' / 'plots' / f'sub_corrs_viflows.pdf', dpi=400)




# t ------> 0  extrapolation
####################################################################
if 0:

    iflows = [8,10,12,14,16,18]

    nrows, ncols = 1, 1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(0.75*wlatex, 0.75*wlatex*0.6))


    plot_corrs = True
    if plot_corrs:
        fig2, axes2 = plt.subplots(1, 2, figsize=(wlatex, 0.55 * wlatex), constrained_layout=True)
        ims = []
        flow_labels = [f"{flowtimes[ifl]:.2f}" for ifl in iflows]

    

    for iname, name in enumerate('gs-18 ex111-18'.split()):
        fit_name = 'one-mass fit' if iname==0 else 'two-mass fit'
        color = ['blue', 'green'][iname]

        tsums_ense = { iflow : np.array([tsums01[iflow] for tsums01,_,_,_ in bs_data[name]])
                        for iflow in iflows }
        tsums_data = gv.dataset.avg_data(tsums_ense, bstrap=True)

        y6 = [tsums_data[ifl] for ifl in iflows]
        x6 = np.array([flowtimes[ifl] for ifl in iflows], dtype=float)

        x_offset = iname*0.005
        x6s = np.asarray(x6, float) + x_offset  # offset for visual distinction

        # fit only first three points
        x3 = x6[:3]
        y3 = y6[:3]

        # linear model
        def fcn(x, p):
            return p['a'] + p['b'] * x
        
        fit = lsqfit.nonlinear_fit(
            data=(x3, y3),
            p0={'a':0, 'b':0},
            fcn=fcn,
            # linear=['a', 'b'],
        )

        # extrapolated value at t=0
        y0 = fit.p['a']
        print(f'{y0=}, {fit.Q=:.2f}')

        # line from the 3rd point down to 0
        x_line = np.linspace(0.0, x3[2], 200)
        y_line = fcn(x_line, fit.pmean)

        ax.errorbar(
            x6s,
            gv.mean(y6),
            yerr=gv.sdev(y6),
            marker='o',
            linestyle='none',
            color=color,
            alpha=0.9,
            markersize=5,
            # label=f'data ({fit_name})',
            label = fit_name
        )   
        ax.plot(
            x_line + x_offset,
            y_line,
            linestyle='--',
            color=color,
            alpha=0.9,
            # label=f'fit ({fit_name})',
        )
        ax.errorbar(
            [0.0 + x_offset],
            [gv.mean(y0)],
            yerr=[gv.sdev(y0)],
            marker='s',
            linestyle='none',
            color=color,
            alpha=0.9,
            # label=f'$t=0$ extrapolation ({fit_name})',
        )
        ax.set_xlabel('$t/a^2$')
        ax.set_ylabel(None)
        ax.text(0.05, 0.95, rf'$H_{{E,t}}^{{(0,1)}} / T^4$', transform=ax.transAxes, ha="left", va="top")
        ymin, _ = ax.get_ylim()
        ax.set_ylim(ymin, 0.0)
        ax.legend(
            loc='upper right',
            frameon=True,
            facecolor='white',
            edgecolor='none',
            framealpha=0.7,
            # fontsize=8,
            # labelspacing=0.3,
            # borderpad=0.3,
            # handletextpad=0.4,
            # handlelength=1.2,
        )


        if plot_corrs:
            ax2 = axes2[iname]
            corr = gv.evalcorr(y6)  # (6,6) correlation matrix

            im = ax2.imshow(
                corr,
                origin="upper",
                cmap="coolwarm",
                vmin=0, vmax=1,
                interpolation="nearest",
                aspect="equal",
            )
            ims.append(im)

            ax2.set_title(f'Correlations of $H_{{E,t}}^{{(0,1)}}$ in $t$\n({fit_name})')

            ax2.set_xticks(range(len(iflows)))
            ax2.set_yticks(range(len(iflows)))
            ax2.set_xticklabels(flow_labels, rotation=45, ha="right")
            ax2.set_yticklabels(flow_labels)

            ax2.set_xlabel(r"$t/a^2$")
            ax2.set_ylabel(r"$t/a^2$")


    out = Path.cwd() / "zeugs" / "plots" / "tsums_linear_extrap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


    if plot_corrs:
        cbar = fig2.colorbar(ims[-1], ax=axes2[-1], pad=0.08, fraction=0.046)
        cbar.set_label("correlation")

        out = Path.cwd() / "zeugs" / "plots" / "tsums_corr_1x2.pdf"
        fig2.savefig(out, bbox_inches="tight")
        plt.close(fig2)