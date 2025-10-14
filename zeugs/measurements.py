from dataclasses import dataclass
import re
from pathlib import Path
from itertools import product
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from statana import stat_error




@dataclass(frozen=True)
class Measurements:
    
    flow_time : float

    clover_t  : float
    clover_s  : float
    iclover_t : float
    iclover_s : float

    plaq_t    : float
    plaq_s    : float
    rect_t    : float
    rect_s    : float

    charge    : float
    charge_is : float   # improved fs only for spatial part
    charge_it : float   # improved fs only for temporal part
    icharge   : float   # improved fs for everything

    q_corrs   : list[list[float]]   # first index is time, second is spatial distance squared: q_corrs[t][s^2]


FlowMeasurements = list[Measurements]



def parse_flow_output(output : str) -> FlowMeasurements:
    '''parsing output of wilson_flow program'''

    flow_measurements = []

    for section in re.split(r'\s*\n\n\s*', output):
        if not section.startswith('GFLOW'): continue
        
        obs, corrs = re.fullmatch(r'GFLOW: (.+)\nq-corrs (.+)', section).groups()
        obs   = [ float(o) for o in obs.split() ]
        corrs = [ [float(o) for o in slc.split()] 
                  for slc in corrs.split(',')     ]
        
        flow_measurements.append(Measurements(*obs, corrs))

    return flow_measurements






def radial_multiplicities(ns : int) -> list[int]:
    '''result[s^2] is number of cells in a ns^3 lattice with distance s^2.'''

    s2_max = 3 * (ns//2)**2
    result = [0 for _ in range(s2_max+1)]

    for x,y,z in product(range(ns), repeat=3):
        x,y,z = (min(a, ns-a) for a in (x,y,z))
        s2 = x**2 + y**2 + z**2
        result[s2] += 1

    return result




def make_distance_corr_arrays(radial_corrs : list, normalize : bool) -> tuple[list, list]:
    '''Takes a list where the index corresponds to s^2.
    Removes all unreachable s^2 and returns [s] and [G(s)] ([G(s) / ds] if normalize is true).'''

    s2_max = len(radial_corrs) - 1
    ns = round(2 * (s2_max/3)**.5)

    ds_list = radial_multiplicities(ns)
    s_list = []
    G_list = []

    for s2, ds in enumerate(ds_list):
        if ds == 0: continue
        s_list.append(s2**.5)
        G_list.append(radial_corrs[s2] / (ds if normalize else 1))
            
    return s_list, G_list




def get_correlators(ensemble : list[FlowMeasurements], t : int, tf : float) -> tuple[list, list, list]:
    '''For a fixed pair (t, tf), extract the data series G(t,s) for each configuration.
    Then for each data point, compute its mean and standard deviation.
    Returns [s values], [G(t,s) values], [G(t,s) errors].'''
    
    if len(ensemble) == 0:
        raise Exception('Empty ensemble :(')

    # for each s^2 store the value of each configuration in an array
    data_all_s2 = [ [] for _ in ensemble[0][0].q_corrs[0] ]

    # the index of the Measurement with flowtime tf in a FlowMeasurement
    i_tf = min( range(len(ensemble[0])),
                key=lambda i: abs(ensemble[0][i].flow_time - tf) )

    # fill data_all_s2
    for flow_meas in ensemble:
        for s2, v in enumerate(flow_meas[i_tf].q_corrs[t]):
            data_all_s2[s2].append(v)

    # divide by radial multiplicity factors and remove unreachable s^2
    s_list, data_all_s = make_distance_corr_arrays(list(map(np.array, data_all_s2)), normalize=True)

    return s_list, \
           [data.mean() for data in data_all_s], \
           [np.std(data) for data in data_all_s]

        









if __name__ == '__main__':



    # some test
    if 0:

        with open(Path('outputs') / 'flow_test.txt', 'r', encoding='utf-8') as f:
            flow_output = f.read()

        flow_meas = parse_flow_output(flow_output)

        print(len(flow_meas[2].q_corrs[0]))
    



    # ensemble flow
    if 1:

        from statana import search_uncorr

        data : list[FlowMeasurements] = []

        for i in range(100):
            with open(Path('outputs') / f'flow_out_{i:05}.txt', 'r', encoding='utf-8') as f:
                flow_output = f.read()
            data.append(parse_flow_output(flow_output))

        print()


        # iq_ftmax = np.array([ d[-1].charge for d in data ])
        # plt.scatter(range(len(iq_ftmax)), iq_ftmax)
        # plt.ylim(-1.5, 1.5)
        # plt.xlabel('config')
        # plt.ylabel('icharge')
        # plt.savefig('plot_charge.png')
        # search_uncorr(iq_ftmax)


        print(', '.join(f'{meas.flow_time:.2f}' for meas in data[0]))

        t=4
        tf=0.43
        s_vals, corr_vals, corr_errors = get_correlators(data, t=t, tf=tf)

        s2_max = round(s_vals[-1]**2)
        ns = round(2 * (s2_max/3)**.5)
        ds = radial_multiplicities(ns)
        print(f'{s2_max = }, {ns = },\n{ds = }\n')
        
        # print(f'corrs = {[v for v,e in correlators]}\n')

        plt.errorbar( s_vals ,
                      corr_vals,
                      yerr=corr_errors,
                      elinewidth=1,
                      marker='o',
                      markersize=3 )
        plt.xlabel('spatial distance')
        plt.ylabel('corr')
        plt.title(f'tau={t}, tf={tf}')
        plt.savefig('plot_corr.png')


    






