import subprocess
from pathlib import Path
import re
import math
import numpy as np






def gen_pure_gauge_ensemble_hmc(input : str, ncores : int = 4) -> str:
    '''Runs ../pure_gauge/su3_hmc with mpirun -np ncores.'''

    exe_path = Path("../pure_gauge/su3_hmc")
    if not exe_path.exists():
        raise FileNotFoundError(f"Could not find HMC executable at {exe_path}")
    
    cmd = [
        "mpirun",
        "-np", str(ncores),
        exe_path
    ]
    
    proc = subprocess.run(
        cmd,                # the full command line as a list
        input=input,        # send this string to the program's stdin
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False         # we'll handle non-zero exit codes ourselves
    )

    out = proc.stdout

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command {cmd!r} failed with exit code {proc.returncode}.\n"
            f"Output was:\n{out}"
        )

    return out




# generating pure gauge ensemble
if 0:

    n_configs = 100
    out_file = lambda i: Path('gauge_configs100') / f'pg_{i:05d}_t=0.lat'

    beta = 5.938    # same dimensions roughly as other configurations in 2020 topcharge paper

    input_initial = \
        '''
        prompt 0
        nx 16
        ny 16
        nz 16
        nt 4
        iseed 8352
        '''
    
    input_set = lambda start_lat, i_config: \
        f'''
        warms 0
        trajecs 1
        traj_between_meas 1
        beta {beta}
        microcanonical_time_step .01
        steps_per_trajectory 4
        {start_lat}
        save_ascii {out_file(i_config)}
        '''
    
    input_complete = input_initial + \
                     input_set('fresh', 0) + \
                     ''.join(input_set('continue', i) for i in range(1, n_configs))
    

    #print(input_complete)
    
    out = gen_pure_gauge_ensemble_hmc(input_complete)

    print(out)










def gen_pure_gauge_ensemble_ora(input : str, ncores : int = 4) -> str:
    '''Runs ../pure_gauge/su3_ora with mpirun -np ncores.'''

    exe_path = Path("../pure_gauge/su3_ora")
    if not exe_path.exists():
        raise FileNotFoundError(f"Could not find ORA executable at {exe_path}")
    
    cmd = [
        "mpirun",
        "-np", str(ncores),
        exe_path
    ]
    
    proc = subprocess.run(
        cmd,                # the full command line as a list
        input=input,        # send this string to the program's stdin
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False         # we'll handle non-zero exit codes ourselves
    )

    out = proc.stdout

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command {cmd!r} failed with exit code {proc.returncode}.\n"
            f"Output was:\n{out}"
        )

    return out



if 0:

    nxyz = 24
    nt = 12

    warms = 1000
    trajecs = 100
    beta = 6.5305    # T / Tc = 1.3
    steps_per_trajectory = 4    # number of overrelaxation sweeps
    qhb_steps = 1   # number of heatbath steps
    out_file = lambda i: Path('gauge_configs') / f'pg_{i:05d}_t=0.lat'

    input_initial = \
        f'''
        prompt 0
        nx {nxyz}
        ny {nxyz}
        nz {nxyz}
        nt {nt}
        iseed 2314
        '''
    
    input_warms = \
        f'''
        warms {warms}
        trajecs 0
        traj_between_meas 1
        beta {beta}
        steps_per_trajectory {steps_per_trajectory}
        qhb_steps {qhb_steps}
        fresh
        no_gauge_fix
        forget
        '''
    
    input_config = lambda i_config: \
        f'''
        warms 0
        trajecs 1
        traj_between_meas 1
        beta {beta}
        steps_per_trajectory {steps_per_trajectory}
        qhb_steps {qhb_steps}
        continue
        no_gauge_fix
        save_ascii {out_file(i_config)}
        '''
    
    input_complete = input_initial + \
                     input_warms + \
                     ''.join(input_config(i) for i in range(trajecs))
    
    out = gen_pure_gauge_ensemble_ora(input_complete)
    print(out)










def filename_change_flowtime(fp : Path, t : float) -> Path:
    return fp.with_name(
        re.sub(r't=.+', rf't={t}.lat', fp.name)
    )



def flow_ensemble(dir : Path, flow_time : float, nsaves : int = 1, ncores : int = 4) -> str:
    '''Flow every config of the ensemble (all .lat files in dir) to all flow times.
    Saves nsaves times in equal intervals, and also once at t=0.'''

    exe_path = Path("../wilson_flow/wilson_flow_rkmk3")
    if not exe_path.exists():
        raise FileNotFoundError(f"Could not find HMC executable at {exe_path}")

    ensemble_t0 = sorted(p for p in dir.iterdir() if p.is_file() and p.suffix == '.lat')

    min_nsteps     = 16
    steps_per_save = math.ceil(min_nsteps / nsaves)
    flow_per_save  = flow_time / nsaves
    stepsize       = flow_per_save / steps_per_save

    input_initial = \
        '''
        prompt 0
        nx 16
        ny 16
        nz 16
        nt 4
        '''
    
    input_new_config = lambda lattice_in: \
        f'''
        reload_ascii {lattice_in}
        zeuthen
        exp_order 8
        stepsize {stepsize}
        stoptime 0
        save_ascii {filename_change_flowtime(lattice_in, 0)}
        '''
    
    input_continue_config = lambda lattice_out: \
        f'''
        continue
        zeuthen
        exp_order 8
        stepsize {stepsize}
        stoptime {flow_per_save}
        save_ascii {lattice_out}
        '''
            
    input_complete = input_initial
    for f in ensemble_t0:
        input_complete += input_new_config(f)
        input_complete += ''.join( input_continue_config(filename_change_flowtime(f, flow_per_save * isave))
                                   for isave in range(1, nsaves+1) )
    
    cmd = [
        "mpirun",
        "-np", str(ncores),
        str(exe_path)
    ]

    proc = subprocess.run(
        cmd,                        # the full command line as a list
        input=input_complete,       # send this string to the program's stdin
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False                 # we'll handle non-zero exit codes ourselves
    )

    out = proc.stdout

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command {cmd!r} failed with exit code {proc.returncode}.\n"
            f"Output was:\n{out}"
        )

    return out





# flowing t=0 ensemble to some flow time
if 0:
    
    # roughly r_F * T = 0.15 (cf 2020 paper)
    out = flow_ensemble(
        dir       = Path('gauge_configs'),
        flow_time = 0.72,
        nsaves    = 3,
        ncores    = 4
    )    

    print(out)


