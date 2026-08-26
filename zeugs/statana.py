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




def index_max_error(data : np.ndarray, err_max : float) -> int:
    '''finds the index i, such that data[j].sdev > err_max for all j >= i.'''

    for i, d in enumerate(data[::-1]):
        if d.sdev <= err_max:
            return len(data) - i
    return 0




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


def bin_averages(bins : Bins, *arrays : np.ndarray) -> tuple[np.ndarray]:
    '''Bin values and compute its averages. If array has D>1, the last dimension is binned.'''

    arrays_binned = []

    for arr in arrays:
        new_dtype = float if np.issubdtype(arr.dtype, np.integer) else arr.dtype
        arr_binned = np.empty(shape=(*arr.shape[:-1], len(bins)), dtype=new_dtype)

        for ibin, (il, ir) in enumerate(bins):
            arr_binned[..., ibin] = arr[..., il:ir].mean(axis=-1)

        arrays_binned.append(arr_binned)

    return tuple(arrays_binned)






def find_distance_bins(distances : np.ndarray, bin_size : float) -> Bins:
    '''Finds indx pairs (il, ir) such that distances[il:ir] contain distances in a range of bin_size.
    Assumes distances is non-empty and monotonically rising.'''

    bins = []

    # current bin (index and distance of left side)
    il = 0
    rl = distances[0]

    for i, r in enumerate(distances):
        if r < rl + bin_size:
            continue

        bins.append((il, i))
        while r >= rl + bin_size:
            rl += bin_size
        il = i

    return bins + [(il, len(distances))]





def bin_distances(ns : int, bin_size : float) -> np.ndarray:
    '''Average distances of each bin.'''

    distances = radial_separations(ns)
    return bin_averages(
        find_distance_bins(distances, bin_size),
        distances
    )[0]





def bin_in_r_through_fcn_and_data(dist : np.ndarray, ense : np.ndarray, fcn : Callable[[float], float], reltol : float, max_bin_size : float|None = None) -> Bins:
    '''Finds bins such that  bin_error / data_error <= reltol.
    Here, bin_error = |(mean of values at bin-points) - (val at mean-point of bin)|.'''

    cov = np.cov(ense, rowvar=False) / ense.shape[0]      # covariances of the means

    bins    = []
    i_left  = 0
    i_start = 1
    cov_sum = 0

    # prevent potential div by 0
    if dist[0] == 0:
        bins.append((0,1))
        i_left = 1
        i_start += 1

    for i in range(i_start, len(dist)+1):

        rbin      = dist[i_left:i].mean()
        bin_vals  = fcn(dist[i_left:i])
        bin_error = abs(bin_vals.mean() - fcn(rbin))
        # bin_error  = np.max(bin_vals) - np.min(bin_vals)          # alternative, more conservative definition of bin_error

        cov_sum += cov[i_left:i, i-1].sum() + cov[i-1, i_left:i-1].sum()
        if cov_sum < 0: raise Exception(f'negative covsum! {cov_sum=}, i={i-1}, r={dist[i-1]}')
        data_error = np.sqrt(max(0,cov_sum)) / (i - i_left)

        if (max_bin_size is not None and dist[i-1] - dist[i_left] > max_bin_size) or (bin_error / data_error > reltol):
            bins.append((i_left, i-1))
            i_left = i-1
            cov_sum = 0

    bins.append((i_left, i))
    return bins





def bin_by_distance(distances : np.ndarray, values : np.ndarray, cov : np.ndarray|None, bin_size : float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Bin-averages values in bins of r-extent bin_size, computing standart deviations using the covariance matrix cov.
    Returns [bin-averages of distances], [bin-averages of values], [stderr of bin-averages]. The last on only if cov is not None.'''

    bins = find_distance_bins(distances, bin_size)

    r_bins = np.empty(shape=(len(bins),), dtype=float)
    v_bins = np.empty(shape=(len(bins),), dtype=float)
    e_bins = np.empty(shape=(len(bins),), dtype=float)

    for i, (il, ir) in enumerate(bins):
        r_bins[i] = np.mean(distances[il:ir])
        v_bins[i] = np.mean(values[il:ir])
        if cov is not None:
            n   = ir - il
            var = cov[il:ir, il:ir].sum() / (n*n)
            e_bins[i] = np.sqrt(var)

    return r_bins, v_bins, e_bins





def bin_through_simultaneous_fit(dist : np.ndarray, ense : np.ndarray, labels : dict[Any, str], fit : lsqfit.nonlinear_fit, reltol : float, max_bin_size : float|None = None) -> dict[Any, Bins]:
    '''...'''

    bins = {}
    for idx in labels.keys():
        bins[idx] = bin_in_r_through_fcn_and_data(
            dist         = dist,
            ense         = ense[:, *idx, :],
            fcn          = lambda r: fit.fcn({idx : r}, fit.pmean)[idx],
            reltol       = reltol,
            max_bin_size = max_bin_size
        )

    return bins





def bin_cut_avg_data(
        dist : np.ndarray, ense : np.ndarray, bins : dict[Any, Bins|None],
        sn_cut_left : float|None = None, r_min_left : float|None = None, r_cuts0_left : dict[Any, float]|None = None,
        err_max_right : float|None = None, r_cuts0_right : dict[Any, float]|None = None, r_max_right : float|None = None
    ) -> tuple[dict[Any, np.ndarray], dict[Any, np.ndarray], dict[Any, np.ndarray], dict[Any, tuple[float, float]]]:
    '''
    What it does:
        1. Bins each series according to its bins
        2. Determines left and right limits of distances to be included in final data
        3. Averages data over all samples, computing all correlations

    What it returns:
        - dict of binned distances (dict of arrays of floats)
        - dict of binned data (dict of arrays of gvars)
        - dict of binned ensemble data (dict of 2d arrays of floats)
        - dict of left and right most distances (dict of (float, float))
    '''

    # bin dist and ense
    dist_binned : dict[Any, np.ndarray] = {}
    ense_binned : dict[Any, np.ndarray] = {}
    for idx in bins.keys():
        if bins[idx] is None: dist_binned[idx], ense_binned[idx] = dist, ense[:, *idx, :]
        else:                 dist_binned[idx], ense_binned[idx] = bin_averages(bins[idx], dist, ense[:, *idx, :])


    # determine range of r values included in final data
    ir_lims : dict[Any, tuple[int, int]]     = {}
    r_lims  : dict[Any, tuple[float, float]] = {}

    for idx in bins.keys():
        data = gv.dataset.avg_data(ense_binned[idx])


        # determining left most r-value for each series
        if sum(x is not None for x in (sn_cut_left, r_min_left, r_cuts0_left)) != 1:
            raise Exception(f'Exactly one xyz_left argument must be not None.')

        if sn_cut_left is not None:
            ir_left = signal_to_noise_cut(data, sn_cut_left)
            if ir_left is None:
                raise Exception(f'Found no ir_cut_left for {idx=}.')

        elif r_min_left is not None:
            ir_left = index_from_distance(dist_binned[idx], r_min_left)

        elif r_cuts0_left is not None:
            ir_left = index_from_distance(dist_binned[idx], r_cuts0_left[idx])
        
        r_left = dist_binned[idx][ir_left]
  

        # determining right most r-value for each series
        if sum(x is not None for x in (err_max_right, r_cuts0_right, r_max_right)) > 1:
            raise Exception(f'At most one xyz_right argument must be not None.')

        if err_max_right is not None:
            ir_right = index_max_error(data, err_max_right)

        elif r_cuts0_right is not None:
            ir_right = index_from_distance(dist_binned[idx], r_cuts0_right[idx])

        elif r_max_right is not None:
            ir_right = index_from_distance(dist_binned[idx], r_max_right)

        else:
            ir_right = len(data)

        r_right = dist_binned[idx][ir_right] if ir_right < len(data) else 1.1 * dist_binned[idx][-1]


        ir_lims[idx] = (ir_left, ir_right)
        r_lims[idx]  = (r_left,  r_right)


    # build dist and ense starting in determined range of distances, then compute sample average
    dist_binned_cut : dict[Any, np.ndarray] = {}
    ense_binned_cut : dict[Any, np.ndarray] = {}

    for idx in bins.keys():
        ir_left, ir_right = ir_lims[idx]
        dist_binned_cut[idx], ense_binned_cut[idx] = dist_binned[idx][ir_left:ir_right], ense_binned[idx][:, ir_left:ir_right]

    data_binned_cut : dict[Any, np.ndarray] = gv.dataset.avg_data(ense_binned_cut)


    return dist_binned_cut, data_binned_cut, ense_binned_cut, r_lims
    

    







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
    axes.set_xlabel(r'$r / a$')
    axes.set_ylabel(r'$G / T^8$')
    if axes.get_legend_handles_labels()[1]: axes.legend()




def plot_fitfcn(dist : np.ndarray, data : np.ndarray, axes : plt.Axes, **plt_args : Any) -> None:
    '''plot function with error bands'''

    if plt_args is None: plt_args = {}
    plt_default = { 'color' : 'red',
                    'alpha' : 0.5,
                    'label' : 'fit' }
    for kw, val in plt_default.items():
        if kw not in plt_args: plt_args[kw] = val

    axes.fill_between(
        x  = dist,
        y1 = gv.mean(data) - gv.sdev(data),
        y2 = gv.mean(data) + gv.sdev(data),
        **plt_args
    )
    if axes.get_legend_handles_labels()[1]: axes.legend()




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
    



def plot_distance_correlations(corrs : np.ndarray, dist : np.ndarray, rmin : float, rmax : float, axes : plt.Axes) -> None:
    '''...'''

    il = index_from_distance(dist, rmin)
    ir = index_from_distance(dist, rmax)

    data = corrs[il:ir, il:ir]
    n_r = data.shape[0]

    axes.imshow(data, cmap='coolwarm', vmin=0, vmax=1)

    n_ticks = 6
    ticks = np.linspace(0, n_r - 1, n_ticks)
    axes.set_xticks(ticks)
    axes.set_yticks(ticks)

    labels = np.linspace(rmin, rmax, n_ticks)
    axes.set_xticklabels([f'{x:.0f}' for x in labels])
    axes.set_yticklabels([f'{x:.0f}' for x in labels])

    axes.set_xlabel('r / a')
    axes.set_ylabel('r / a')
    axes.set_title('radius correlations')






