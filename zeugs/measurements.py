from dataclasses import dataclass
import re
from pathlib import Path




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




def parse_flow_output(output : str) -> ...:
    '''pasing output of wilson_flow program'''

    flow_measurements = []

    for section in re.split(r'\s*\n\n\s*', output):
        if not section.startswith('GFLOW'): continue
        
        obs, corrs = re.fullmatch(r'GFLOW: (.+)\nq-corrs (.+)', section).groups()
        obs   = [ float(o) for o in obs.split() ]
        corrs = [ [float(o) for o in slc.split()] 
                  for slc in corrs.split(',')     ]
        
        flow_measurements.append(Measurements(*obs, corrs))

    return flow_measurements












if __name__ == '__main__':


    with open(Path('outputs') / 'flow_test.txt', 'r', encoding='utf-8') as f:
        flow_output = f.read()

    flow_meas = parse_flow_output(flow_output)

    print(len(flow_meas[2].q_corrs[0]))

    






