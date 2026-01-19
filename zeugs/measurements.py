from dataclasses import dataclass
import re
from pathlib import Path
from itertools import product
from typing import Callable
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




@dataclass
class Measurements:

    rtime_q    : float  # time in seconds for q=FF* computation
    rtime_fft  : float  # time in seconds for correlator computation
    rtime_flow : float  # time in seconds for flowstep
    
    flow_time  : float

    clover_t   : float
    clover_s   : float
    iclover_t  : float
    iclover_s  : float

    plaq_t     : float
    plaq_s     : float
    rect_t     : float
    rect_s     : float

    charge     : float
    charge_is  : float  # improved fs only for spatial part
    charge_it  : float  # improved fs only for temporal part
    icharge    : float  # improved fs for everything

    q_corrs    : list[list[float]] | None  # first index is time, second is spatial distance squared: q_corrs[t][s^2]


FlowMeasurements = list[Measurements]

Bins = list[tuple[int, int]]





def parse_flow_output(output : str, flowtimes : list[str]|None = None) -> FlowMeasurements:
    '''parsing output of wilson_flow program'''

    flow_measurements : FlowMeasurements = []

    for section in re.split(r'\s*\n\n\s*', output):
        if not section.startswith('Time to complete flowstep'): continue

        lines = re.split(r'\n', section)

        rtime_flow, rtime_q, rtime_fft = [ float(re.search(r'= (.+) seconds', lines[i])[1])
                                           for i in (0,1,4) ]
        
        obs = map(float, lines[2].removeprefix('GFLOW: ').split())

        corrs = [ [float(c) for c in tslice.split()]
                  for tslice in lines[3].removeprefix('q-corrs ').split(',') ]
        
        flow_measurements.append(Measurements(rtime_q, rtime_fft, rtime_flow, *obs, corrs))

    # fix flowtimes
    if flowtimes is not None:
        if len(flowtimes) != len(flow_measurements): raise Exception('wrong number of flowtimes :(')
        for flowtime, flowmeas in zip(flowtimes, flow_measurements):
            flowmeas.flow_time = flowtime

    return flow_measurements





def extract_flowtimes(flow_measurements : FlowMeasurements) -> list[float]:
    '''Extract flowtimes from list of measurements.'''

    return [meas.flow_time for meas in flow_measurements]







def radial_multiplicities_r2(ns : int) -> np.ndarray:
    '''result[s^2] is number of cells in a ns^3 lattice with distance s^2.'''

    s2_max = 3 * (ns//2)**2
    result = [0 for _ in range(s2_max+1)]

    for x,y,z in product(range(ns), repeat=3):
        x,y,z = (min(a, ns-a) for a in (x,y,z))
        s2 = x**2 + y**2 + z**2
        result[s2] += 1

    return np.array(result, dtype=int)



def radial_multiplicities_r(ns : int) -> np.ndarray:
    '''result[r] is number of cells in a ns^3 lattice with distance r^2. Only distances with dr>0 are present.'''

    result = [dr for dr in radial_multiplicities_r2(ns) if dr > 0]
    return np.array(result, dtype=int)




def radial_separations(ns : int) -> np.ndarray:
    '''list of all reachable distances r on the lattice.'''

    result = [r2**.5 for r2, count in enumerate(radial_multiplicities_r2(ns)) if count > 0]
    return np.array(result, dtype=float)




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




def bin_in_r_through_fcn(dist : np.ndarray, fcn : Callable[[float], float], tol : float) -> Bins:
    '''dist is 1d array of float, fcn takes a float to a float, tol is a fraction in [0,1].
    Here, relerr is average value inside bin divided by value of middle of bin. (DIFFERENT to function incorporating data)'''

    bins    = []
    i_left  = 0
    i_start = 1

    if dist[0] == 0:
        bins.append((0,1))
        i_left = 1
        i_start += 1

    for i in range(i_start, len(dist)+1):
        rbin   = dist[i_left:i].mean()
        Gbin   = fcn(dist[i_left:i]).mean()
        relerr = abs((Gbin - fcn(rbin)) / fcn(rbin))
        if relerr > tol:
            bins.append((i_left, i-1))
            i_left = i-1

    bins.append((i_left, i))
    return bins




def bin_in_r_through_fcn_and_data(dist : np.ndarray, ense : np.ndarray, fcn : Callable[[float], float], reltol : float, max_bin_size : float|None = None) -> Bins:
    '''Finds bins such that  bin_error / data_error <= reltol.
    Here, bin_error = |(mean of values at bin-points) - (val at mean-point of bin)|.'''

    cov = np.cov(ense, rowvar=False) / ense.shape[0]      # covariances of the means

    bins    = []
    i_left  = 0
    i_start = 1
    cov_sum = 0

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
        data_error = np.sqrt(cov_sum) / (i - i_left)

        if (max_bin_size is not None and dist[i-1] - dist[i_left] > max_bin_size) or (bin_error / data_error > reltol):
            bins.append((i_left, i-1))
            i_left = i-1
            cov_sum = 0

    bins.append((i_left, i))
    return bins





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




def bin_distances(ns : int, bin_size : float) -> np.ndarray:
    '''Average distances of each bin.'''

    distances = radial_separations(ns)
    return bin_averages(
        find_distance_bins(distances, bin_size),
        distances
    )[0]



def radial_multiplcities_bins(ns : int, bin_size : float) -> np.ndarray:
    '''Radial multiplicities summed for each bin.'''

    dr_list = radial_multiplicities_r(ns)
    bins    = find_distance_bins(radial_separations(ns), bin_size)
    result  = np.empty(shape=len(bins), dtype=int)
    for i, (il, ir) in enumerate(bins):
        result[i] = dr_list[il:ir].sum()
    return result



def slice_in_r(r_list : np.ndarray, v_list : np.ndarray, step : float) -> tuple[np.ndarray, np.ndarray]:
    '''Slice v_list, but in terms of r instead of indices.'''

    bins = find_distance_bins(r_list, step)
    r_sliced = np.empty(shape=len(bins), dtype=r_list.dtype)
    v_sliced = np.empty(shape=len(bins), dtype=v_list.dtype)

    for i, (il, ir) in enumerate(bins):
        r_sliced[i] = r_list[il]
        v_sliced[i] = v_list[il]

    return r_sliced, v_sliced





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




def bin_by_distance_ensemble(distances : np.ndarray, ensemble : np.ndarray, bin_size : float) -> np.ndarray:
    '''For ensemble.shape=(..., distances), bins along the last axis.'''

    bins = find_distance_bins(distances, bin_size)

    ensemble_binned = np.empty(
        shape = (*ensemble.shape[:-1], len(bins)),
        dtype = float
    )

    for ibin, (il, ir) in enumerate(bins):
        ensemble_binned[..., ibin] = np.mean(ensemble[..., il:ir], axis=-1)

    return ensemble_binned







def make_distance_corr_arrays(radial_corrs : list, normalize : bool, ds_list : list[float]|None = None) -> list[float]:
    '''Takes a list where the index corresponds to s^2.
    Removes all unreachable s^2 and returns [G(s)] ([G(s) / ds] if normalize is true).'''

    s2_max = len(radial_corrs) - 1
    ns = round(2 * (s2_max/3)**.5)

    if ds_list is None:
        ds_list = radial_multiplicities_r2(ns)
    G_list = []

    for s2, ds in enumerate(ds_list):
        if ds == 0: continue
        G_list.append(radial_corrs[s2] / (ds if normalize else 1))
            
    return G_list






