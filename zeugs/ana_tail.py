from pathlib import Path
import numpy as np
import gvar as gv
import lsqfit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from measurements import *
from statana import *





nt = 16
ns = 64
flowtimes = np.array([ 0.02572923590301565, 0.05303278154520656, 0.08103495224784385, 0.110208782326772,  0.141058391298962,  0.1736893835495303,
                       0.2086835114617738,  0.2454549361253429,  0.2838117228448263,  0.3249152519239425, 0.3705101434001126, 0.4179449298122797,
                       0.4696233239687699,  0.5279559816139976,  0.5890993819917063,  0.6570215111572449, 0.734680305158274,  0.8257738792221512,
                       0.9353328035912798,  1.06959076807306,    1.233820281085228,   1.435936459574629 ], dtype=float)#, 1.68034447439404, 1.972161672512587 ]



def main():

    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(15, 24))


    # getting data
    bin_size = 0.5
    ense = get_data_binned(bin_size)
    dist = bin_distances(ns, bin_size)

    iflow = 12
    tau = 5

    data = gv.dataset.avg_data(ense[:, iflow, tau, :].copy())


    # plot of correlator for all r
    plot_dist(dist, data, axes[0,0])


    # signal to noise threshold
    sn_cut = 10
    ir_cut = signal_to_noise_cut(data, sn_cut)
    print(f'{sn_cut=}: i={ir_cut}, r={dist[ir_cut]}')


    # do single fit
    fit = single_fit(dist[ir_cut:], data[ir_cut:])
    print(fit)
    plot_fit(dist, data, ir_cut, fit, axes[1,0])


    # r correlations
    img = axes[2,0].imshow(gv.evalcorr(data), vmin=0, vmax=1)
    fig.colorbar(img, ax=axes[2,0], label='Correlation')


    # multiple flowtimes combined fit
    datac = gv.dataset.avg_data(ense[:, 10:17, tau, :].copy())
    fitc, ir_cuts = combined_fit(dist, datac, sn_cut)
    print(fitc)
    plot_fitc(dist, datac, ir_cuts, fitc, axes[:,1].flatten())


    # finalize figure
    plt.tight_layout()
    plt.savefig(Path.cwd() / 'zeugs' / 'plots' / 'grid.png', dpi=600)









def get_data_binned(bin_size : float) -> np.ndarray:
    skip_configs = 55
    return bin_by_distance_ensemble(
        distances = radial_separations(ns),
        ensemble  = np.load(Path.cwd() / 'zeugs' / 'data' / 'data0.npy')[skip_configs:],
        bin_size  = bin_size
    )




def single_fit(dist : np.ndarray, data : np.ndarray) -> lsqfit.nonlinear_fit:
    '''...'''

    f = lambda x, p: p['a'] * gv.exp(- p['m'] * x) / x
    fit = lsqfit.nonlinear_fit(
        data = (dist, data),
        fcn  = f,
        p0   = {'a' : -1, 'm' : 1},
        debug = True
    )
    return fit




def combined_fit(dist : np.ndarray, data : np.ndarray, sn_cut : float) -> tuple[lsqfit.nonlinear_fit, list[int]]:
    '''...'''

    # fit function
    def f(x, p):
        a_vals, m = p['a'], p['m']
        y = {}
        for i, a in enumerate(a_vals):
            y[i] = a * gv.exp(-m * x) / x
        return y
    
    # determine ir_cuts
    ir_cuts = [signal_to_noise_cut(data_i, sn_cut) for data_i in data]
    if None in ir_cuts: raise Exception(f'Got None in ir_cuts: {ir_cuts}.')

    # prepare data (dict due to potentially unequal ir_cuts)
    data_dict = {}
    for i, (data_i, ir_cut) in enumerate(zip(data, ir_cuts)):
        data_dict[i] = data_i[ir_cut:]

    # do fit
    fit = lsqfit.nonlinear_fit(
        data = (dist, data_dict),
        fcn  = f,
        p0   = {'a' : np.full(data.shape[0], -1, float), 'm' : 1},
        debug = True
    )

    return fit, ir_cuts




def plot_dist(dist : np.ndarray, data : np.ndarray, axes : plt.Axes) -> None:
    '''plot G over r'''

    axes.errorbar(
        x          = dist,
        y          = gv.mean(data),
        yerr       = gv.sdev(data),
        marker     = 'o',
        markersize = 2,
        linestyle  = 'none',
        linewidth  = 1,
        label      = 'data'
    )
    axes.set_xlabel('r / a')
    axes.set_ylabel('G / T^6')
    axes.legend()




def plot_fit(dist : np.ndarray, data : np.ndarray, i_cut : int, fit : lsqfit.nonlinear_fit, axes : plt.Axes) -> None:
    '''...'''

    il  = max(i_cut-3, 0)
    ir = -1

    # the data
    axes.errorbar(
        x          = dist[il:ir],
        y          = gv.mean(data[il:ir]),
        yerr       = gv.sdev(data[il:ir]),
        marker     = 'o',
        markersize = 2,
        linestyle  = 'none',
        linewidth  = 1,
        label      = 'data'
    )

    # the fit
    axes.plot(
        dist[il:ir],
        fit.fcn(dist[il:ir], fit.pmean),
        label = 'fit'
    )

    # vertical line at ir_cut
    axes.axvline(x=dist[i_cut], alpha=0.5, color='orange', label='sn_cut')

    # other stuff
    axes.set_xlabel('r / a')
    axes.set_ylabel('G / T^6')
    axes.legend()




def plot_fitc(dist : np.ndarray, data : np.ndarray, i_cuts : list[int], fit : lsqfit.nonlinear_fit, axes : list[plt.Axes]) -> None:
    '''....'''

    il  = max(min(i_cuts)-3, 0)
    ir = -1

    for i, ax in enumerate(axes):
        ax.errorbar(
            x          = dist[il:ir],
            y          = gv.mean(data[i, il:ir]),
            yerr       = gv.sdev(data[i, il:ir]),
            marker     = 'o',
            markersize = 2,
            linestyle  = 'none',
            linewidth  = 1,
            label      = f'data {i}'
        )
        ax.plot(
            dist[il:ir],
            fit.pmean['a'][i] * gv.exp(- fit.pmean['m'] * dist[il:ir]) / dist[il:ir],
            # fit.fcn(dist[il:ir], fit.pmean)[i],
            label = f'fit {i}'
        )
        ax.axvline(x=dist[i_cuts[i]], alpha=0.5, color='orange', label='sn_cut')
        ax.set_ylabel('G / T^6')
        ax.legend()

    axes[-1].set_xlabel('r / a')

    





def signal_to_noise_cut(data : np.ndarray, sn_cut : float) -> int|None:
    '''Find index of r, such that s/n < sn_cut for all r' >= r. (None if no such index exists)'''

    sn_data = np.abs(gv.mean(data) / gv.sdev(data))
    i_cand = None
    for ir, sn in enumerate(sn_data):
        if   i_cand is None     and sn <  sn_cut: i_cand = ir
        elif i_cand is not None and sn >= sn_cut: i_cand = None
    return i_cand








# print('loading data...')
# code_path = Path.cwd()
# skip_configs = 50
# corrs_ensemble = np.load(code_path / 'zeugs' / 'data' / 'data0.npy')[skip_configs:]
# print(corrs_ensemble.shape, corrs_ensemble.dtype)

# print('bin in r...')


# print('doing stuff')
# d = gvar.dataset.avg_data(corrs_ensemble[:, :, 5, 10])
# img = axes[0,1].imshow(gvar.evalcorr(d), vmin=0, vmax=1)
# fig.colorbar(img, ax=axes[0,1], label='Correlation')


# print('big ensemble')
# db = gvar.dataset.avg_data(corrs_ensemble[:, :, :, :10].copy())
# print(db.shape)
# print(gvar.evalcorr(db).shape)

# img = axes[1,1].imshow(gvar.evalcorr(db)[:,5,:,5], vmin=0, vmax=1)
# fig.colorbar(img, ax=axes[1,1], label='Correlation')







# TODO
#
# bin in r, incorporating covariances
#
# plot to exponential, checking chi2 and Q





















if __name__ == '__main__':

    main()