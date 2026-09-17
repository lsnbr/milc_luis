from import_and_plotting_tech import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch




# load H data
pickle_path = Path.cwd() / 'zeugs' / 'data' / 'av_H.pkl'
with open(pickle_path, 'rb') as f:
    _, H_data = gv.load(f)




sub_list    = [(0,1), (0,1,2)]
models      = ['const', 'linear']
exclude_idx = {(0,1):[], (0,1,2):[]}
colors_data = {(0,1) : 'dodgerblue', (0,1,2) : 'firebrick'}
colors_fit  = {(0,1)   : {'const' : 'deepskyblue', 'linear' : 'darkturquoise'}, 
               (0,1,2) : {'const' : 'orangered', 'linear' : 'mediumvioletred'}}

sc = 0.8
fig, ax = plt.subplots(figsize=(sc*wlatex, sc*1.1*hlatex), constrained_layout=True)



for j, sub in enumerate(sub_list):
    sub_str    = ','.join(map(str, sub))
    excl       = exclude_idx[sub]
    x_offset_j = j * 0.007
    color_data = colors_data[sub]

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


    for i, model in enumerate(models):
        color_fit  = colors_fit[sub][model]
        x_offset_i = -3*0.007/2 + (2*j+i) * 0.007

        # linear model, fit to all available points
        fitfcns = { 'const' : (lambda x, p: x*0 + p['a']), 'linear' : (lambda x, p: p['a'] + p['b'] * x) }
        p0s     = { 'const' : {'a':0},                     'linear' : {'a':0, 'b':0}                     }
        fitfcn, p0 = fitfcns[model], p0s[model]

        fit = lsqfit.nonlinear_fit(
            data = (x_fit, y_fit),
            p0   = p0,
            fcn  = fitfcn,
        )

        # extrapolated value at t=0
        y0 = fit.p['a']
        print(f'\n{sub=}, {y0=}, chi2/dof{fit.chi2/fit.dof:.2f}, {fit.Q=:.2f}, {model=}, excluded={excl}')

        # fit line from t=0 to the last data point
        x_line = np.linspace(0.0, x_fit[-1], 200)
        y_line = fitfcn(x_line, fit.pmean)

        # plot fit line (no legend entry)
        ax.plot(
            x_line + x_offset_i,
            y_line,
            linestyle = '--' if model=='const' else ':',
            color     = color_fit,
            alpha     = 0.99,
        )

        # annotate the line: 'linear' above, 'const' below
        x_txt = 0.25                      # where along the line to place the text
        y_txt = fitfcn(x_txt, fit.pmean)
        dy    = {'linear':4, 'const':-4}[model]      # offset in points
        ax.annotate(
            'constant fit' if model=='const' else 'linear fit',
            xy         = (x_txt + x_offset_i, y_txt),
            xytext     = (0, dy),
            textcoords = 'offset points',
            ha         = 'center',
            va         = 'bottom' if model == 'linear' else 'top',
            color      = color_fit,
            fontsize   = 9,
            alpha      = 0.99
        )

        # plot t=0 extrapolation (no legend entry)
        ax.errorbar(
            x    = [0.0 + x_offset_i],
            y    = [gv.mean(y0)],
            yerr = [gv.sdev(y0)],
            marker    = 's',
            linestyle = 'none',
            color     = color_fit,
            alpha     = 0.99,
        )


    # plot included data points (full color)
    ax.errorbar(
        x    = x_fit + x_offset_j,
        y    = gv.mean(y_fit),
        yerr = gv.sdev(y_fit),
        marker     = 'o',
        linestyle  = 'none',
        color      = color_data,
        alpha      = 0.9,
        markersize = 5,
        label      = rf'data $H_{{E,t}}^{{({sub_str})}}$',
    )

    # plot excluded data points (paler)
    if excl:
        x_excl = x[~mask]
        y_excl = y[~mask]
        ax.errorbar(
            x    = x_excl + x_offset_j,
            y    = gv.mean(y_excl),
            yerr = gv.sdev(y_excl),
            marker     = 'o',
            linestyle  = 'none',
            color      = color_data,
            alpha      = 0.3,
            markersize = 5,
            label      = None,
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
ax.set_xlabel('$t_\\mathrm{f}/a^2$')
ax.set_ylabel(r'$H_{E,t_\mathrm{f}} / T^4$')

ymin, _ = ax.get_ylim()
ax.set_ylim(-0.044, 0.0075)

legend_handles = [
    Patch(facecolor=colors_data[sub],
          label=rf'$H_{{E,t_\mathrm{{f}}}}^{{({",".join(map(str, sub))})}}$')
    for sub in sub_list[::-1]
]

ax.legend(
    handles    = legend_handles,
    loc        = 'upper right',
    frameon    = True,
    facecolor  = 'white',
    edgecolor  = 'none',
    framealpha = 0.7,
)

# saving the figure
out = Path.cwd() / "zeugs" / "plots" / "zero_tf_extrapolation.pdf"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

