import subprocess
from pathlib import Path




def do_warmups(sweeps : int, ns : int, nt : int, beta : float, out : Path) -> str:
    ...

    input = \
        f'''
        prompt 0
        nx {ns}
        ny {ns}
        nz {ns}
        nt {nt}
        iseed 2314

        warms {sweeps}
        trajecs 0
        traj_between_meas 1
        beta {beta}
        steps_per_trajectory 0
        qhb_steps 1
        fresh
        no_gauge_fix
        forget
        '''
    
    