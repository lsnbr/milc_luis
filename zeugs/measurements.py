import re



def parse_flow_output(output : str) -> ...:
    '''pasing output of wilson_flow program'''

    for section in re.split(r'\s*\n\n\s*', output):
        if not section.startswith('GFLOW'): continue

        obs, corrs = re.fullmatch(r'GFLOW: (.+)\nq-corrs (.+)', section)[1:]
        obs   = [ float(o) for o in obs.split() ]
        corrs = [ [float(o) for o in slc.split()] 
                  for slc in corrs.split(',')     ]
        




