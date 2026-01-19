from pathlib import Path
from typing import Any
import numpy as np
import gvar as gv
import lsqfit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from measurements import *






###############################################################################
##########################  basic data stuff  #################################
###############################################################################


def get_data_unbinned() -> np.ndarray:
    '''Get full (time, distance) dataset.'''

    skip_configs = 55
    return np.load(Path.cwd() / 'zeugs' / 'data' / 'data0.npy')[skip_configs:]




def get_data_mats_unbinned() -> np.ndarray:
    '''Get full matsubara (frequency, distance) dataset.'''

    skip_configs = 55
    return np.load(Path.cwd() / 'zeugs' / 'data' / 'data0m.npy')[skip_configs:]




def index_from_distance(dist : np.ndarray, r_cut : float) -> int:
    '''Smallest i such that  dist[j] > r_cut  for all  j > i. 0 if none exists.'''

    for i, r in enumerate(dist):
        if r > r_cut: return max(0, i-1)

    return i




def signal_to_noise_cut(data : np.ndarray, sn_cut : float) -> int|None:
    '''Find index of r, such that s/n < sn_cut for all r' >= r. (None if no such index exists)'''

    sn_data = np.abs(gv.mean(data) / gv.sdev(data))
    i_cand = None
    for ir, sn in enumerate(sn_data):
        if   i_cand is None     and sn <  sn_cut: i_cand = ir
        elif i_cand is not None and sn >= sn_cut: i_cand = None
    return i_cand




def expx_fcn(x : float, a_vals : list[float], m_vals : list[float]) -> float:
    '''Sum of len(a_vals) = len(m_vals) exponentials.'''

    return sum( a * np.exp(- m * x) / x
                for a, m in zip(a_vals, m_vals, strict=True) )



def expx(x : float, a : float, m : float) -> float:
    '''Simple a exp(-m x) / x'''

    return a * gv.exp(- m * x) / x







###############################################################################
###############################  binning  #####################################
###############################################################################


def bin_data_through_fit(dist : np.ndarray, ense : np.ndarray, reltol : float, max_bin_size : float|None = None, axes : list[plt.Axes]|None = None) -> tuple[Bins, lsqfit.nonlinear_fit]:
    '''1. Bin data with constant bin size.
    2. Do fit (ignoring correlations for performance since only rough shape of fit is needed) starting at data with s/n < sn_cut (after binning).
    3. Find new bins such that bin_error is neglectable compared to data error.
    4. If visualize, axes should be length 2.'''

    cbin_size = 1       # bin size for preliminary fit
    sn_cut    = 10      # s/n cut for (constant bin size) binned data in preliminary fit

    cbins = find_distance_bins(dist, cbin_size)
    dist_cbinned, ense_cbinned = bin_averages(cbins, dist, ense)
    data_cbinned = gv.dataset.avg_data(ense_cbinned)

    ir_cut = signal_to_noise_cut(data_cbinned, sn_cut)
    f = lambda x, p: expx_fcn(x, [p['a']], [p['m']])
    fit = lsqfit.nonlinear_fit(
        udata = (dist_cbinned[ir_cut:], data_cbinned[ir_cut:]),
        fcn   = f,
        p0    = {'a' : -1, 'm' : 1}
    )

    vbins = bin_in_r_through_fcn_and_data(dist, ense, lambda r: f(r, fit.pmean), reltol, max_bin_size)

    # optional plotting
    if axes is not None:
        plot_dist(dist_cbinned[ir_cut:], data_cbinned[ir_cut:], axes[0])
        plot_fitfcn(dist_cbinned[ir_cut:], f(dist_cbinned[ir_cut:], fit.p), axes[0])
        for il,_ in vbins:
            axes[0].axvline(x=dist[il], color='orange', alpha=0.75)
        axes[0].legend()
        axes[0].set_title(f'Data and fit is based on constant Δr={cbin_size} bins, vertical lines are bins based on fit.')
        axes[1].axis('off')
        axes[1].text(0, 1, f'{fit}\n\nnumber of bins = {len(vbins)}', family="monospace", va="top")

    return vbins, fit






def bin_multiple_series_through_fit(dist : np.ndarray, ense : np.ndarray, labels : dict[Any, str], reltol : float, max_bin_size : float|None = None) -> dict[Any, Bins]:
    '''For all r-series (ense.shape[1:-1]) do binning by fit.'''

    bins : dict[Any, Bins] = {}

    for idx in labels.keys():
        bins[idx], fit_for_binning = bin_data_through_fit(dist, ense[:, *idx, :], reltol, max_bin_size)

        if fit_for_binning.pmean['a'] > 0 or fit_for_binning.pmean['m'] < 0:
            raise Exception(f'Bad preliminary fit for {idx=}:\n{fit_for_binning}')
        
    return bins





def bin_through_simultaneous_fit(dist : np.ndarray, ense : np.ndarray, labels : dict[Any, str], fit : lsqfit.nonlinear_fit, reltol : float, max_bin_size : float|None = None) -> dict[Any, Bins]:
    '''...'''

    bins = {}
    for idx in labels.keys():
        bins[idx] = bin_in_r_through_fcn_and_data(
            dist   = dist,
            ense   = ense[:, *idx, :],
            fcn    = lambda r: fit.fcn({idx : r}, fit.pmean)[idx],
            reltol = reltol,
            max_bin_size = max_bin_size
        )

    return bins





def bin_cut_avg_data(
        dist : np.ndarray, ense : np.ndarray, bins : dict[Any, Bins],
        sn_cut : float|None = None, r_min : float|None = None, r_cuts0 : dict[Any, float]|None = None
    ) -> tuple[dict[Any, np.ndarray], dict[Any, np.ndarray], dict[Any, np.ndarray], dict[Any, float]]:
    '''1. Bin each r-series individually (no cross-correlation computations).
    1. Find ir_cut for each r-series based on sn_cut.
    3. Bin and avg data starting at ir_cut, computing all correlations, returning dicts of dist, data, ense and r_cuts.'''

    # bin dist and ense
    dist_binned : dict[Any, np.ndarray] = {}
    ense_binned : dict[Any, np.ndarray] = {}
    for idx in bins.keys():
        dist_binned[idx], ense_binned[idx] = bin_averages(bins[idx], dist, ense[:, *idx, :])

    # determine ir_cuts based on sn_cut
    ir_cuts : dict[Any, int]   = {}
    r_cuts  : dict[Any, float] = {}

    for idx in bins.keys():

        if sn_cut is not None and r_min is None and r_cuts0 is None:
            ir_cut = signal_to_noise_cut(gv.dataset.avg_data(ense_binned[idx]), sn_cut)

        elif sn_cut is None and r_min is not None and r_cuts0 is None:
            ir_cut = index_from_distance(dist_binned[idx], r_min)

        elif sn_cut is None and r_min is None and r_cuts0 is not None:
            ir_cut = index_from_distance(dist_binned[idx], r_cuts0[idx])

        else:
            raise Exception(f'Exactly one must be None: {sn_cut=}, {r_min=}, {r_cuts0=}.')

        if ir_cut is None:
            raise Exception(f'Found no ir_cut for {idx=}.')
        
        ir_cuts[idx] = ir_cut
        r_cuts[idx]  = dist_binned[idx][ir_cut]

    # build dist and ense starting from ir_cuts, then average
    dist_binned_cut : dict[Any, np.ndarray] = {}
    ense_binned_cut : dict[Any, np.ndarray] = {}

    for idx in bins.keys():
        ir_cut = ir_cuts[idx]
        dist_binned_cut[idx], ense_binned_cut[idx] = dist_binned[idx][ir_cut:], ense_binned[idx][:, ir_cut:]

    data_binned_cut : dict[Any, np.ndarray] = gv.dataset.avg_data(ense_binned_cut)

    return dist_binned_cut, data_binned_cut, ense_binned_cut, r_cuts
    

    







###############################################################################
##############################  plotting  #####################################
###############################################################################


def plot_dist(dist : np.ndarray, data : np.ndarray, axes : plt.Axes, **plt_args : Any) -> None:
    '''Plot data over distance r.'''

    if plt_args is None: plt_args = {}
    plt_default = { 'marker'     : 'o',
                    'markersize' : 2,
                    'linestyle'  : 'none',
                    'linewidth'  : 1,
                    'label'      : 'data' }
    for kw, val in plt_default.items():
        if kw not in plt_args: plt_args[kw] = val

    axes.errorbar(
        x    = dist,
        y    = gv.mean(data),
        yerr = gv.sdev(data),
        **plt_args
    )
    axes.set_xlabel('r / a')
    axes.set_ylabel('G / T^8')
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
    axes.legend()




def plot_corr_eigenvals(dataset : dict|np.ndarray, axes : plt.Axes) -> None:
    '''Make svd analysis, do svdcut, plot svd analysis.'''

    svd = gv.dataset.svd_diagnosis(dataset, nbstrap=100)
    svd.plot_ratio(plot = GVarAxesAdapter(axes))
    axes.set_title('SVD analysis of covariance matrix eigenvalues')




class GVarAxesAdapter:
    '''Adapter to make plt.Axes behave like gvar's expected plot object.'''

    def __init__(self, ax):
        self._ax = ax

    def __getattr__(self, name):
        if name in ('xlabel', 'ylabel', 'title', 'xscale'):
            return getattr(self._ax, f'set_{name}')
        return getattr(self._ax, name)