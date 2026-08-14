from import_and_plotting_tech import *





# load H data
pickle_path = Path.cwd() / 'zeugs' / 'data' / 'av_H.pkl'
with open(pickle_path, 'rb') as f:
    _, H_data = gv.load(f)




sub_list = [(0,1), (0,1,2)]
colors   = ['blue', 'green']
exclude_idx = {(0,1):[1,3], (0,1,2):[]}

fig, ax = plt.subplots(figsize=(wlatex, 1.2*hlatex), constrained_layout=True)



for j, sub in enumerate(sub_list):

    color     = colors[j]
    sub_str   = ','.join(map(str, sub))
    x_offset  = j * 0.005
    excl      = exclude_idx[sub]

    # build x-data (flowtime/a^2) and y-data (H_sub/T^4)
    x, y = [], []
    for name, value in H_data.items():
        _, sm, iflow = name.split('_')
        if sm != str(sub): continue
        x.append(flowtimes[int(iflow)])
        y.append(value)
    order = np.argsort(x)
    x = np.array(x)[order]
    y = np.array(y)[order]

    # mask for points used in the fit
    mask = np.ones(len(x), dtype=bool)
    mask[excl] = False

    x_fit = x[mask]
    y_fit = y[mask]

    # linear model, fit to all available points
    fitfcns = { 'const' : (lambda x, p: x*0 + p['a']), 'linear' : (lambda x, p: p['a'] + p['b'] * x) }
    p0s     = { 'const' : {'a':0},                     'linear' : {'a':0, 'b':0}                     }
    model = 'const'
    fitfcn, p0 = fitfcns[model], p0s[model]

    fit = lsqfit.nonlinear_fit(
        data = (x_fit, y_fit),
        p0   = p0,
        fcn  = fitfcn,
    )

    # extrapolated value at t=0
    y0 = fit.p['a']
    print(f'\n{sub=}, {y0=}, {fit.Q=:.2f}, excluded={excl}')

    # fit line from t=0 to the last data point
    x_line = np.linspace(0.0, x_fit[-1], 200)
    y_line = fitfcn(x_line, fit.pmean)

    # plot included data points (full color)
    ax.errorbar(
        x    = x_fit + x_offset,
        y    = gv.mean(y_fit),
        yerr = gv.sdev(y_fit),
        marker     = 'o',
        linestyle  = 'none',
        color      = color,
        alpha      = 0.9,
        markersize = 5,
        label      = rf'data $H_{{E,t}}^{{({sub_str})}}$',
    )

    # plot excluded data points (paler)
    if excl:
        x_excl = x[~mask]
        y_excl = y[~mask]
        ax.errorbar(
            x    = x_excl + x_offset,
            y    = gv.mean(y_excl),
            yerr = gv.sdev(y_excl),
            marker     = 'o',
            linestyle  = 'none',
            color      = color,
            alpha      = 0.3,
            markersize = 5,
            label      = None,
        )

    # plot fit line
    ax.plot(
        x_line + x_offset,
        y_line,
        linestyle = '--',
        color     = color,
        alpha     = 0.9,
        label     = rf'{model} fit $({sub_str})$',
    )

    # plot t=0 extrapolation
    ax.errorbar(
        x    = [0.0 + x_offset],
        y    = [gv.mean(y0)],
        yerr = [gv.sdev(y0)],
        marker    = 's',
        linestyle = 'none',
        color     = color,
        alpha     = 0.9,
        label     = rf'$t=0$ extrap. $({sub_str})$',
    )



    # print correlation matrix for this sub
    corr = gv.evalcorr(y_fit)
    flow_labels = [f'{xi:.2f}' for xi in x_fit]

    print(f'\ncorrelations for sub={sub}:')
    col_width = 8
    print(' ' * col_width + ''.join(lbl.ljust(col_width) for lbl in flow_labels))
    for lbl, row in zip(flow_labels, corr):
        print(lbl.ljust(col_width) + ''.join(f'{val:.3f}'.ljust(col_width) for val in row))



# axis labels and annotation
ax.set_xlabel('$t/a^2$')
ax.set_ylabel(r'$H_{E,t} / T^4$')

ymin, _ = ax.get_ylim()
ax.set_ylim(ymin, 0.01)

ax.legend(
    loc        = 'upper right',
    frameon    = True,
    facecolor  = 'white',
    edgecolor  = 'none',
    framealpha = 0.7,
    fontsize      = 10,
    handlelength  = 1.2,
    handletextpad = 0.4,
    labelspacing  = 0.3,
    borderpad     = 0.3,
)

# saving the figure
out = Path.cwd() / "zeugs" / "plots" / "zero_tf_extrapolation.pdf"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

