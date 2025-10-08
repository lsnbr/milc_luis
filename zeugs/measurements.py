from dataclasses import dataclass
import re
from pathlib import Path
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




def get_correlators(ensemble : list[FlowMeasurements], t : int, tf : float, bin_size : int = 1) -> list[tuple[float, float]]:
    
    if len(ensemble) == 0:
        raise Exception('Empty ensemble :(')

    # for each s^2 store the value of each configuration in an array
    data_all_s2 = [ [] for _ in ensemble[0][0].q_corrs[0] ]

    # the index of the Measurement with flowtime tf in a FlowMeasurement
    i_tf = min( range(len(ensemble[0])),
                key=lambda i: abs(ensemble[0][i].flow_time - tf) )

    for flow_meas in ensemble:
        for i, v in enumerate(flow_meas[i_tf].q_corrs[t]):
            data_all_s2[i].append(v)

    return [ (data.mean(), stat_error(data, bin_size))
             for data in map(np.array, data_all_s2) ]
            

        









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

        correlators = get_correlators(data, t=1, tf=0.36)
        
        plt.plot( [s2**.5 for s2 in range(len(correlators))] ,
                  [val for val, err in correlators],
                  marker='o' )
        plt.xlabel('spatial distance')
        plt.ylabel('corr')
        plt.savefig('plot_corr.png')


    






