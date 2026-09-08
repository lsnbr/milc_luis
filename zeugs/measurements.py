from dataclasses import dataclass
import re
from itertools import product
import numpy as np





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




def flowtime_to_radius(t : float, Nt : int) -> float:
    '''Given flowtime in units of a^2, and number of time divisions, compute flow radius in units of beta.'''
    return (8 * t)**.5 / Nt


def radius_to_flowtime(r : float, Nt : int) -> float:
    '''Given flow radius in units of beta, and number of time divisions, compute flowtime in units of a^2.'''
    return (r * Nt)**2 / 8


def flow_params(n_steps : int, Nt : int, r_max : float = 0.25) -> tuple[float, float]:
    '''Given amnount of steps ([0,1,2] has two steps), number of time divisions and maximal flow radius in units of beta,
    computes (stoptime, stepsize) parameters in lattice units.'''

    stoptime = radius_to_flowtime(r_max, Nt)
    stepsize = stoptime / n_steps
    return stoptime, stepsize





def radial_multiplicities_r2(ns : int) -> np.ndarray:
    '''result[s^2] is number of cells in a ns^3 lattice with distance s.'''

    s2_max = 3 * (ns//2)**2
    result = [0 for _ in range(s2_max+1)]

    for x,y,z in product(range(ns), repeat=3):
        x,y,z = (min(a, ns-a) for a in (x,y,z))
        s2 = x**2 + y**2 + z**2
        result[s2] += 1

    return np.array(result, dtype=int)



def radial_multiplicities_r(ns : int) -> np.ndarray:
    '''result[ir] is number of cells in a ns^3 lattice with distance r. Only distances with dr>0 are present.'''

    result = [dr for dr in radial_multiplicities_r2(ns) if dr > 0]
    return np.array(result, dtype=int)




def radial_separations(ns : int) -> np.ndarray:
    '''list of all reachable distances r on the lattice.'''

    result = [r2**.5 for r2, count in enumerate(radial_multiplicities_r2(ns)) if count > 0]
    return np.array(result, dtype=float)









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






