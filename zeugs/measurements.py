from dataclasses import dataclass
import re
from pathlib import Path
from itertools import product
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







def radial_multiplicities(ns : int) -> list[int]:
    '''result[s^2] is number of cells in a ns^3 lattice with distance s^2.'''

    s2_max = 3 * (ns//2)**2
    result = [0 for _ in range(s2_max+1)]

    for x,y,z in product(range(ns), repeat=3):
        x,y,z = (min(a, ns-a) for a in (x,y,z))
        s2 = x**2 + y**2 + z**2
        result[s2] += 1

    return result




def radial_separations(ns : int) -> list[float]:
    '''list of all reachable distances r on the lattice.'''

    return [r2**.5 for r2, count in enumerate(radial_multiplicities(ns)) if count > 0]




def find_distance_bins(distances : float, bin_size : float) -> list[tuple[int, int]]:
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




def bin_by_distance2(distances : np.ndarray, values : np.ndarray, cov : np.ndarray|None, bin_size : float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    return (r_bins, v_bins) if cov is None else (r_bins, v_bins, e_bins)






def make_distance_corr_arrays(radial_corrs : list, normalize : bool, ds_list : list[float]|None = None) -> list[float]:
    '''Takes a list where the index corresponds to s^2.
    Removes all unreachable s^2 and returns [G(s)] ([G(s) / ds] if normalize is true).'''

    s2_max = len(radial_corrs) - 1
    ns = round(2 * (s2_max/3)**.5)

    if ds_list is None:
        ds_list = radial_multiplicities(ns)
    G_list = []

    for s2, ds in enumerate(ds_list):
        if ds == 0: continue
        G_list.append(radial_corrs[s2] / (ds if normalize else 1))
            
    return G_list






