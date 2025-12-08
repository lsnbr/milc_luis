from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Any
import numpy as np
import gvar as gv
import lsqfit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flowing import flowtime_to_radius, radius_to_flowtime
from measurements import *
from statana import *





nt = 16
ns = 64
flowtimes = np.array([ 0.02572923590301565, 0.05303278154520656, 0.08103495224784385, 0.110208782326772,  0.141058391298962,  0.1736893835495303,
                       0.2086835114617738,  0.2454549361253429,  0.2838117228448263,  0.3249152519239425, 0.3705101434001126, 0.4179449298122797,
                       0.4696233239687699,  0.5279559816139976,  0.5890993819917063,  0.6570215111572449, 0.734680305158274,  0.8257738792221512,
                       0.9353328035912798,  1.06959076807306,    1.233820281085228,   1.435936459574629 ], dtype=float)#, 1.68034447439404, 1.972161672512587 ]



def main1():

    # TODO
    # bootstrap error analysis
    # more thought about data/fit contribution to r-sum


    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(14, 20))


    # getting data
    ense_raw = get_data_unbinned()
    dist_raw = radial_separations(ns)

    bin_size = 0.9
    ense = get_data_binned(bin_size)
    dist = bin_distances(ns, bin_size)

    tau = 5
    iflowtimes = [6,7,8,9]

    dataset = ense[:, 6:10, tau, :].copy()
    data = gv.dataset.avg_data(dataset)


    # flowtime window
    vis_flowtime_window(tau, axes[0,1])


    # r correlations
    ai = 2
    vis_correlations_in_r(dist, data[ai], axes[ai,1], fig)
    axes[2,1].set_title(f'Correlations in r (r_F={flowtime_to_radius(flowtimes[iflowtimes[ai]], nt)*nt:.2f}a, tau={tau})')


    # multiple flowtimes combined fit
    sn_cut = 10

    do_svdcut = False
    data_cut = do_svd_cut(dataset, axes[1,1])
    if do_svdcut: data = data_cut

    fit, ir_cuts = combined_fit(dist, data, sn_cut, 1)
    print(fit)
    plot_fitc(dist, data, ir_cuts, fit, tau, iflowtimes, axes[:,0].flatten())


    # do sum over r
    rsums = []
    for iifl, ifl in enumerate(iflowtimes):
        data_raw = gv.dataset.avg_data(ense_raw[:, ifl, tau, :].copy())
        ddist_raw = radial_multiplicities_r(ns)
        res = sum_over_r(dist_raw, data_raw, ddist_raw, fit.p['a'][iifl], fit.p['m'], dist[ir_cuts[iifl]], axes[iifl,1] if iifl==3 else None)
        if iifl==3: axes[iifl,1].set_title(f'partial sums up to r (r_F={flowtime_to_radius(flowtimes[ifl], nt)*nt:.2f}a, tau={tau})')
        rsums.append(res)
    

    # plot G_F(tau)
    # plot_over_flowtime(tau, iflowtimes, rsums, axes[-1,1])


    # test binning dependent on function
    _ = bin_data_through_fcn(dist_raw, ense_raw[:, iflowtimes[-1], tau, :].copy(), lambda r: r**(-6), 0.001, axes[-1, 0])


    # finalize figure
    plt.tight_layout()
    plt.savefig(Path.cwd() / 'zeugs' / 'plots' / 'grid.png', dpi=600)






def main2():

    # getting distances and configuration data
    dist = radial_separations(ns)
    ense_all = get_data_unbinned()

    # tau and flowtime indices
    tau   = 5
    i_fts = [6,7,8,9]

    # ense and labels
    ense = ense_all[:, i_fts, tau, :]
    labels = np.array([f'tf={flowtimes[i]:.2f}a^2' for i in i_fts])

    # run stuff
    print('starting...')
    result = do_tail_fit_and_sum(dist, ense, labels, 10)
    print('r_cuts_sn      =', result.r_cuts_sn)
    print('r_cuts_sum     =', result.r_cuts_sum)
    print('number of bins =', len(result.bins))
    print('total sums     =', result.rsums)
    print(result.fit)

    # plot stuff
    nrows, ncols = len(i_fts), 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
    plot_tail_fit_and_sum(dist, ense, result, labels, axes)

    # finalize figure
    plt.tight_layout()
    plt.savefig(Path.cwd() / 'zeugs' / 'plots' / 'grid2.png', dpi=600)




main = main2






@dataclass
class TailFitData:

    bins       : Bins                   # bins determined through comparing to preliminary fit
    r_cuts_sn  : np.ndarray             # float for each r-series, corresponding to minimum r used in fit    

    fit        : lsqfit.nonlinear_fit   # fit object of combined fit

    r_cuts_sum : np.ndarray             # float for each r-series, corresponding to r from when on fit instead of data is used in sum
    psums      : np.ndarray             # partial sums (array of float) for each r-series (only for r integer)
    rsums      : float                  # G_F(tau) = sum over r  for each r-series




def do_tail_fit_and_sum(dist : np.ndarray, ense : np.ndarray, labels : np.ndarray, sn_cut : float) -> TailFitData:
    '''dist:   list of r-separations
       ense:   data ensemble, ense.shape = (config, ..., distance)
       labels: labels for r-series, labels.shape = ense.shape[1:-1]'''
    
    # some constants
    reltol = 0.01       # bin_error / data_error <= reltol
    n_exp  = 1          # number of exponentials in fit
    a0, m0 = -1., 1.    # initial values for a and m in fit
    
    # evaluate bins for all r-series, then choose the finest
    bins = None
    for idx in np.ndindex(labels.shape):
        new_bins = bin_data_through_fcn(dist, ense[:, *idx, :], lambda r: r**(-6), reltol)
        if bins is None or len(new_bins) > len(bins):
            bins = new_bins

    # bin dist and ense according to bins and compute gvar array of ense (could do after ir_cuts_sn to reduce size of cov)
    dist_binned, ense_binned = bin_averages(bins, dist, ense)
    data_binned = gv.dataset.avg_data(ense_binned)

    # find ir_cut (and corresponding distance r_cut) for all r-series
    ir_cuts_sn = np.empty(shape=labels.shape, dtype=int)
    r_cuts_sn  = np.empty(shape=ir_cuts_sn.shape, dtype=float)
    for idx in np.ndindex(labels.shape):
        ir_cut = signal_to_noise_cut(data_binned[idx], sn_cut)
        if ir_cut is None:
            raise Exception(f'Found no ir_cut for label={labels[idx]}.')
        ir_cuts_sn[idx] = ir_cut
        r_cuts_sn[idx]  = dist_binned[ir_cut]

    # prepare data for fit (dict due to unequal ir_cuts_sn)
    data_dict = {}
    for idx, ir_cut in np.ndenumerate(ir_cuts_sn):
        data_dict[idx] = data_binned[*idx, ir_cut:]

    # the fit function
    def f(p):
        y = {}
        for idx in np.ndindex(labels.shape):
            y[idx] = expx_fcn(dist_binned[ir_cuts_sn[idx]:], p['a'][idx], p['m'])
        return y

    # do the fit
    fit = lsqfit.nonlinear_fit(
        data = data_dict,
        fcn  = f,
        p0   = { 'a' : np.full(shape = (*labels.shape, n_exp), fill_value = a0),
                 'm' : np.linspace(m0, m0 * n_exp, n_exp)},
        debug = True
    )

    # determine points from where on in the sum fit data is used instead of real data
    ir_cuts_sum = np.empty(shape=labels.shape, dtype=int)
    r_cuts_sum  = np.empty(shape=ir_cuts_sum.shape, dtype=float)
    for idx in np.ndindex(ir_cuts_sum.shape):
        ir_cut = index_from_distance(dist, r_cuts_sn[idx])    # for now
        ir_cuts_sum[idx] = ir_cut
        r_cuts_sum[idx]  = dist[ir_cut]

    # do sums (partial and total)
    data_mean = ense.mean(axis=0)
    for idx, ir_cut in np.ndenumerate(ir_cuts_sum):
        data_mean[idx][ir_cut:] = expx_fcn(dist[ir_cut:], fit.pmean['a'][idx], fit.pmean['m'])
    data_mean *= radial_multiplicities_r(ns)
    partial_sums = np.cumsum(data_mean, axis=-1) / nt   # divide by nt to go from T^6 to T^5 units
    total_sums   = partial_sums[..., -1].copy()

    # preparing the return value
    return TailFitData(
        bins,
        r_cuts_sn,
        fit,
        r_cuts_sum,
        partial_sums[..., [il for il, ir in find_distance_bins(dist, 1)]],
        total_sums
    )




def plot_tail_fit_and_sum(dist : np.ndarray, ense : np.ndarray, result : TailFitData, labels : np.ndarray, axes : np.ndarray) -> None:
    '''plot stuff'''

    # prepare data
    dist_b, ense_b = bin_averages(result.bins, dist, ense)
    data_b = gv.dataset.avg_data(ense_b)

    for i, idx in enumerate(np.ndindex(labels.shape)):

        # plot the fit
        ir_cut   = index_from_distance(dist, result.r_cuts_sn[idx])
        ir_cut_b = index_from_distance(dist_b, result.r_cuts_sn[idx])
        plot_dist   ( dist_b[ir_cut_b:],
                      data_b[*idx, ir_cut_b:],
                      axes[i, 0] )
        plot_fitfcn ( dist[ir_cut:],
                      expx_fcn(dist[ir_cut:], result.fit.p['a'][idx], result.fit.p['m']),
                      axes[i, 0] )
        axes[1, 0].set_title(labels[idx])

        # plot the partial sum
        ...





def bin_data_through_fcn(dist : np.ndarray, ense : np.ndarray, fcn : Callable[[float], float], reltol : float, axes : plt.Axes|None = None) -> Bins:
    '''Bin data according to function.'''

    bins = bin_in_r_through_fcn_and_data(dist, ense, fcn, reltol)

    if axes is not None:
        cbin_size = 0.5
        sn_cut = 8

        dist_cbinned, ense_cbinned = bin_averages(find_distance_bins(dist, cbin_size), dist, ense)
        data_cbinned = gv.dataset.avg_data(ense_cbinned)
        ir_cut_c = signal_to_noise_cut(data_cbinned, sn_cut)
        plot_dist(dist_cbinned[ir_cut_c:], data_cbinned[ir_cut_c:], axes, alpha=0.25, color='blue', label='const bins')

        dist_vbinned, ense_vbinned = bin_averages(bins, dist, ense)
        data_vbinned = gv.dataset.avg_data(ense_vbinned)
        ir_cut_v = index_from_distance(dist_vbinned, dist_cbinned[ir_cut_c])
        plot_dist(dist_vbinned[ir_cut_v:], data_vbinned[ir_cut_v:], axes, color='purple', label='var bins')

        axes.set_title(f'Constant bins (Δr={cbin_size}a) and variable sized bins ({len(bins)} bins).')

    return bins




def bin_data_through_fit(dist : np.ndarray, ense : np.ndarray, axes : list[plt.Axes]|None = None) -> Bins:
    '''Bin data with constant bin size, then do fit, then do new bins based such that bin_error is neglectable compared to data error.
    If visualize, axes should be length >= 2.'''

    # TODO
    # if performance is a problem, maybe ignore correlation in preliminary fit

    cbin_size = 1       # bin size for preliminary fit
    sn_cut    = 10      # s/n cut for (constant bin size) binned data in preliminary fit
    reltol    = 0.01    # binning_error / data_error < reltol

    cbins = find_distance_bins(dist, cbin_size)
    dist_cbinned, ense_cbinned = bin_averages(cbins, dist, ense)
    data_cbinned = gv.dataset.avg_data(ense_cbinned)

    ir_cut = signal_to_noise_cut(data_cbinned, sn_cut)
    f = lambda x, p: expx_fcn(x, [p['a']], [p['m']])
    fit = lsqfit.nonlinear_fit(
        data = (dist_cbinned[ir_cut:], data_cbinned[ir_cut:]),
        fcn  = f,
        p0   = {'a' : -1, 'm' : 1}
    )

    vbins = bin_in_r_through_fcn_and_data(dist, ense, lambda r: f(r, fit.pmean), reltol)

    if axes is not None:
        plot_dist(dist_cbinned[ir_cut:], data_cbinned[ir_cut:], axes[0])
        plot_fitfcn(dist_cbinned[ir_cut:], f(dist_cbinned[ir_cut:], fit.p), axes[0])
        for il,_ in vbins:
            axes[0].axvline(x=dist[il], color='orange', alpha=0.75)
        axes[0].legend()
        axes[0].set_title(f'Data and fit is based on constant Δr={cbin_size} bins, vertical lines are bins based on fit.')
        axes[1].axis('off')
        axes[1].text(0, 1, f'{fit}\n\nnumber of bins = {len(vbins)}', family="monospace", va="top")

    return vbins
    



def vis_fcn_binning(dist : np.ndarray, data : np.ndarray, fcn : Callable[[float], float], tol : float, axes : plt.Axes) -> list[tuple[int, int]]:
    '''...'''

    bins = bin_in_r_through_fcn(dist, fcn, tol)
    plot_dist(dist, data, axes)
    for il, ir in bins:
        axes.axvline(x=dist[il], color='orange', alpha=0.75)
    return bins




def vis_correlations_in_r(dist : np.ndarray, data : np.ndarray, axes : plt.Axes, fig : plt.Figure) -> None:
    '''visualize correlations in r'''

    img = axes.imshow(gv.evalcorr(data), vmin=0, vmax=1)

    tick_positions = np.arange(0, len(dist), len(dist)//5)
    tick_labels    = [f'{dist[i]:.2f}' for i in tick_positions]

    axes.set_xticks(tick_positions, tick_labels, rotation=45)
    axes.set_yticks(tick_positions, tick_labels)

    axes.set_title(f'radius correlations')
    fig.colorbar(img, ax=axes, label='Correlation')




def plot_over_flowtime(tau : int, iflowtimes : np.ndarray, corrs : np.ndarray, axes : plt.Axes) -> None:
    '''...'''

    axes.errorbar(
        x    = [flowtimes[i] for i in iflowtimes],
        y    = gv.mean(corrs),
        yerr = gv.sdev(corrs),
        marker     = 'o',
        markersize = 2,
        linestyle  = 'none',
        linewidth  = 1,
        label      = f'tau = {tau}'
    )
    axes.set_xlim(0, 0.5)
    axes.set_ylim(-0.2, 0)
    axes.set_xlabel('flowtime / a^2')
    axes.set_ylabel('G_F(tau) / T^5')
    axes.set_title(f'G_F(tau, r) summed over r, as a function of flowtime at fixed tau')
    axes.legend()




def vis_flowtime_window(tau : int, axes : plt.Axes) -> None:
    '''...'''

    flowtimes_norm = flowtime_to_radius(flowtimes, nt) * nt / tau
    for ft in flowtimes_norm:
        axes.axvline(x=ft, color='blue', linestyle='--')
    axes.axvline(x=1/4, color='red', label='1/4', alpha=0.75)
    axes.axvline(x=1/3, color='red', label='1/3', alpha=0.75)
    axes.set_xlabel(f'r_F / a / tau  where  tau={tau}')
    axes.set_title('Blue: all flowtimes, Red: flowtime window (3/2 < r_F/a < tau/3)')
    axes.legend()




def sum_over_r(dist : np.ndarray, data : np.ndarray, ddist : np.ndarray, a : list[gv.GVar], m : list[gv.GVar], r_cut : float, axes : plt.Axes|None) -> None:
    '''do partial sums'''

    ir_cut = min(i for i,r in enumerate(dist) if r >= r_cut)

    partial_sums = np.empty_like(data)
    for i, (r, v, dr) in enumerate(zip(dist, data, ddist)):
        prev = 0 if i==0 else partial_sums[i-1]
        if i < ir_cut:
            partial_sums[i] = prev + v * dr
        else:
            partial_sums[i] = prev + expx_fcn(r, a, m) * dr
    partial_sums /= nt

    if axes is not None:
        dist_sliced, psum_sliced = slice_in_r(dist, partial_sums, 1)
        axes.errorbar(
            x          = dist_sliced,
            y          = gv.mean(psum_sliced),
            yerr       = gv.sdev(psum_sliced),
            marker     = 'o',
            markersize = 2,
            linestyle  = 'none',
            linewidth  = 1,
            label      = 'partial sums'
        )
        axes.axvline(x=dist[ir_cut], alpha=0.5, color='orange', label='sn_cut')

        axes.set_xlabel('r / a')
        axes.set_ylabel('psum / T^5')
        axes.legend()

    return partial_sums[-1]
    




def get_data_unbinned() -> np.ndarray:
    '''Get full dataset.'''

    skip_configs = 55
    return np.load(Path.cwd() / 'zeugs' / 'data' / 'data0.npy')[skip_configs:]



def get_data_binned(bin_size : float) -> np.ndarray:
    '''Bin dataset in r.'''

    skip_configs = 55
    return bin_by_distance_ensemble(
        distances = radial_separations(ns),
        ensemble  = np.load(Path.cwd() / 'zeugs' / 'data' / 'data0.npy')[skip_configs::1],
        bin_size  = bin_size
    )




def do_svd_cut(dataset : np.ndarray, axes : plt.Axes|None = None) -> np.ndarray:
    '''Make svd analysis, do svdcut, plot svd analysis.'''

    svd = gv.dataset.svd_diagnosis(dataset.reshape(dataset.shape[0], -1), nbstrap=100)
    data_cut = gv.svd(svd.avgdata, svdcut=svd.svdcut).reshape(*dataset.shape[1:])
    print('svdcut =', svd.svdcut)
    if axes is not None:
        svd.plot_ratio(plot = GVarAxesAdapter(axes))
        axes.set_title('SVD analysis of covariance matrix eigenvalues')
    return data_cut




def expx_fcn(x : float, a_vals : list[float], m_vals : list[float]) -> float:
    '''Sum of len(a_vals) = len(m_vals) exponentials.'''

    return sum( a * gv.exp(- m * x) / x
                for a, m in zip(a_vals, m_vals, strict=True) )



def single_fit(dist : np.ndarray, data : np.ndarray, n_exp : int = 1) -> lsqfit.nonlinear_fit:
    '''...'''

    f = lambda x, p: expx_fcn(x, p['a'], p['m'])
    fit = lsqfit.nonlinear_fit(
        data = (dist, data),
        fcn  = f,
        p0   = {'a' : np.full(n_exp, -1.), 'm' : np.linspace(1, n_exp, n_exp)},
        debug = True
    )
    return fit




def combined_fit(dist : np.ndarray, data : np.ndarray, sn_cut : float, n_exp : int = 1) -> tuple[lsqfit.nonlinear_fit, list[int]]:
    '''...'''
    
    # determine ir_cuts
    ir_cuts = [signal_to_noise_cut(data_i, sn_cut) for data_i in data]
    if None in ir_cuts: raise Exception(f'Got None in ir_cuts: {ir_cuts}.')

    # prepare data (dict due to potentially unequal ir_cuts)
    data_dict = {}
    for i, (data_i, ir_cut) in enumerate(zip(data, ir_cuts)):
        data_dict[i] = data_i[ir_cut:]

    # fit function
    def f(p):
        y = {}
        for i, (a_vals, ir_cut) in enumerate(zip(p['a'], ir_cuts)):
            y[i] = expx_fcn(dist[ir_cut:], a_vals, p['m'])
        return y

    # do fit
    fit = lsqfit.nonlinear_fit(
        data = data_dict,
        fcn  = f,
        p0   = {'a' : np.full((data.shape[0], n_exp), -1.), 'm' : np.linspace(1, n_exp, n_exp)},
        debug = True
    )

    return fit, ir_cuts




def plot_dist(dist : np.ndarray, data : np.ndarray, axes : plt.Axes, **plt_args : Any) -> None:
    '''plot G over r'''

    if plt_args is None: plt_args = {}
    plt_default = { 'marker'     : 'o',
                    'markersize' : 2,
                    'linestyle'  : 'none',
                    'linewidth'  : 1,
                    'label'      : 'data' }
    for kw, val in plt_default.items():
        if kw not in plt_args: plt_args[kw] = val

    axes.errorbar(
        x          = dist,
        y          = gv.mean(data),
        yerr       = gv.sdev(data),
        **plt_args
    )
    axes.set_xlabel('r / a')
    axes.set_ylabel('G / T^6')
    axes.legend()



def plot_fitfcn(dist : np.ndarray, data : np.ndarray, axes : plt.Axes) -> None:
    '''plot function with error bands'''

    axes.fill_between(
        x  = dist,
        y1 = gv.mean(data) - gv.sdev(data),
        y2 = gv.mean(data) + gv.sdev(data),
        color = 'red',
        alpha = 0.5,
        label = 'fit',
    )




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
    data_fit = fit.fcn(dist[il:ir], fit.p)
    axes.fill_between(
        x  = dist[il:ir],
        y1 = gv.mean(data_fit) - gv.sdev(data_fit),
        y2 = gv.mean(data_fit) + gv.sdev(data_fit),
        color = 'red',
        alpha = 0.75,
        label = 'fit',
    )

    # vertical line at ir_cut
    axes.axvline(x=dist[i_cut], alpha=0.75, color='orange', label='s/n cut')

    # other stuff
    axes.set_xlabel('r / a')
    axes.set_ylabel('G / T^6')
    axes.legend()




def plot_fitc(dist : np.ndarray, data : np.ndarray, i_cuts : list[int], fit : lsqfit.nonlinear_fit, tau : int, ifts : list[int], axes : list[plt.Axes]) -> None:
    '''....'''

    il  = max(min(i_cuts), 0)
    ir = -1

    for i, (ax, ift) in enumerate(zip(axes, ifts)):
        ax.errorbar(
            x          = dist[il:ir],
            y          = gv.mean(data[i, il:ir]),
            yerr       = gv.sdev(data[i, il:ir]),
            marker     = 'o',
            markersize = 2,
            linestyle  = 'none',
            linewidth  = 1,
            label      = 'data'
        )
        data_fit = expx_fcn(dist[il:ir], fit.p['a'][i], fit.p['m'])
        ax.fill_between(
            x  = dist[il:ir],
            y1 = gv.mean(data_fit) - gv.sdev(data_fit),
            y2 = gv.mean(data_fit) + gv.sdev(data_fit),
            color = 'red',
            alpha = 0.75,
            label = 'fit',
        )
        ax.axvline(x=dist[i_cuts[i]], alpha=0.75, color='orange', label='s/n cut')
        ax.set_xlabel('r / a')
        ax.set_ylabel('G / T^6')
        ax.set_ylim(-0.0005, 0.0002)
        ax.set_title(f'Flowtime t_F={flowtimes[ift]:.2f}a^2 (r_F={flowtime_to_radius(flowtimes[ift], nt)*nt:.2f}a)  and  tau={tau}')
        ax.legend()


    



def signal_to_noise_cut(data : np.ndarray, sn_cut : float) -> int|None:
    '''Find index of r, such that s/n < sn_cut for all r' >= r. (None if no such index exists)'''

    sn_data = np.abs(gv.mean(data) / gv.sdev(data))
    i_cand = None
    for ir, sn in enumerate(sn_data):
        if   i_cand is None     and sn <  sn_cut: i_cand = ir
        elif i_cand is not None and sn >= sn_cut: i_cand = None
    return i_cand



def index_from_distance(dist : np.ndarray, r_cut : float) -> int:
    '''Smallest i such that  dist[j] > r_cut  for all  j > i. 0 if none exists.'''

    for i, r in enumerate(dist):
        if r > r_cut: return max(0, i-1)

    return i




class GVarAxesAdapter:
    '''Adapter to make plt.Axes behave like gvar's expected plot object.'''

    def __init__(self, ax):
        self._ax = ax

    def __getattr__(self, name):
        if name in ('xlabel', 'ylabel', 'title', 'xscale'):
            return getattr(self._ax, f'set_{name}')
        return getattr(self._ax, name)



























if __name__ == '__main__':

    main()