import subprocess
from pathlib import Path
import re
import math
import numpy as np

from typing import List


ncores = 4




def run_command(exe_path : Path, input : str, ncores : int) -> str:
    '''runs cmd with mpi, returning stdout'''

    cmd = [
        "mpirun",
        "-np", str(ncores),
        str(exe_path)
    ]

    proc = subprocess.run(
        cmd,                        # the full command line as a list
        input=input,                # send this string to the program's stdin
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,    # Use this instead of 'text=True' for Python 3.6
        check=False                 # we'll handle non-zero exit codes ourselves
    )

    out = proc.stdout

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command {cmd!r} failed with exit code {proc.returncode}.\n"
            f"Output was:\n{out}"
        )

    return out





def gen_input_initial(ns : int, nt : int, iseed : int|None = None) -> str:
    '''Generated initial part of input file with nx=ny=nz=ns.'''

    return f'''
           prompt 0
           nx {ns}
           ny {ns}
           nz {ns}
           nt {nt}
           {f'iseed {iseed}' if iseed is not None else ''}
           '''





def do_warmups(sweeps : int, beta : float, input_initial : str, lat_out : Path) -> str:
    '''Do sweeps amount of heat bath sweeps.
    input_initial: prompt, nx, ny, nz, nt.'''

    exe_path = Path("../pure_gauge/su3_ora")
    if not exe_path.exists():
        raise FileNotFoundError(f"Could not find ORA executable at {exe_path}")
    
    input_initial += \
        f'''
        warms {sweeps}
        trajecs 0
        traj_between_meas 1
        beta {beta}
        steps_per_trajectory 0
        qhb_steps 1
        fresh
        no_gauge_fix
        save_serial {lat_out}
        '''
    
    return run_command(exe_path, input_initial, ncores)







def flow_in_steps(flow_times : List[float], lat_initial : Path, input_initial : str) -> str:
    '''Flows lat_initial to flow times (rkmk3).
    input_initial: prompt, nx, ny, nz, nt.'''

    exe_path = Path("../wilson_flow/region_flow_rkmk3")
    if not exe_path.exists():
        raise FileNotFoundError(f"Could not find gradient flow executable at {exe_path}")

    if sorted(flow_times) != list(flow_times):
        raise Exception('List flow_times not sorted in ascending order.')
    
    tf_current = 0
    for i, tf in enumerate(flow_times):
        tf_step = tf - tf_current
        input_initial += \
            f'''
            {f'reload_serial {lat_initial}' if i==0 else 'continue'}
            zeuthen
            exp_order 8
            stepsize {tf_step if tf_step>0 else 111}
            stoptime {tf_step}
            forget
            '''
        tf_current = tf

    return run_command(exe_path, input_initial, ncores)






def flow_rkmk3(stoptime : float, stepsize : float, lat_initial : Path, input_initial : str) -> str:
    '''Flows lat_initial with given stoptime and stepsize using zeuthen flow with rkmk3 integrator.
    input_initial: prompt, nx, ny, nz, nt.'''

    exe_path = Path("../wilson_flow/region_flow_rkmk3")
    if not exe_path.exists():
        raise FileNotFoundError(f"Could not find gradient flow executable at {exe_path}")
    
    input_flow = \
        f'''
        reload_serial {lat_initial}
        zeuthen
        exp_order 8
        stepsize {stepsize}
        stoptime {stoptime}
        forget
        '''
    
    return run_command(exe_path, input_initial + input_flow, ncores)










if __name__ == '__main__':


    # do warmups
    if 1:

        ns = 16
        nt = 8
        
        input_initial = gen_input_initial(ns, nt, iseed=2314)

        out = do_warmups(
            sweeps        = 10,
            beta          = 6.237,
            input_initial = input_initial,
            lat_out       = Path('thermalized_configs') / 'ns16_nt8_T1p3_10hb.lat'
        )

        print(out)



    # do flow
    if 0:

        ns = 16
        nt = 8

        input_initial = \
            f'''
            prompt 0
            nx {ns}
            ny {ns}
            nz {ns}
            nt {nt}
            '''
        
        out = flow_in_steps(
            flow_times    = np.linspace(0, 2, 21),
            lat_initial   = Path('thermalized_configs') / 'ns16_nt8_T1p3.lat',
            input_initial = input_initial
        )

        print(out)



    if 0:

        ns = 16
        nt = 8

        input_initial = \
            f'''
            prompt 0
            nx {ns}
            ny {ns}
            nz {ns}
            nt {nt}
            '''
        
        out = flow_rkmk3(
            stoptime      = 1,
            stepsize      = 0.1,
            lat_initial   = Path('thermalized_configs') / 'ns16_nt8_T1p3.lat',
            input_initial = input_initial
        )

        print(out)
