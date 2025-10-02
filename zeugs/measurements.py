from dataclasses import dataclass
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt




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




def parse_flow_output(output : str) -> list[Measurements]:
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




def get_correlators(t, tf):
    ...












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

        data : list[list[Measurements]] = []

        for i in range(100):
            with open(Path('outputs') / f'flow_out_{i:05}.txt', 'r', encoding='utf-8') as f:
                flow_output = f.read()
            data.append(parse_flow_output(flow_output))

        print()


        # iq_ftmax = [ d[-1].charge for d in data ]
        # plt.scatter(range(100), iq_ftmax)
        # plt.ylim(-1.5, 1.5)
        # plt.xlabel('config')
        # plt.ylabel('icharge')
        # plt.show()


        vals = np.array([ d[0].charge for d in data ])
        # plt.scatter(range(len(vals)), plaq_t_vals)
        # plt.show()
        search_uncorr(vals)

    






