from pathlib import Path
from typing import List

from various_goods import run_command_stream







def gen_input_initial(ns : int, nt : int, iseed : int|None = None) -> str:
    '''Generated initial part of input file with nx=ny=nz=ns.'''

    return \
        f'''
        prompt 0
        nx {ns}
        ny {ns}
        nz {ns}
        nt {nt}
        {f'iseed {iseed}' if iseed is not None else ''}
        '''





def do_warmups(sweeps : int, beta : float, input_initial : str, lat_out : Path, run_cmd : list[str]) -> str:
    '''Do sweeps amount of heat bath sweeps.
    input_initial: prompt, nx, ny, nz, nt.'''
    
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
        save_dir /
        '''
    
    return run_command_stream(run_cmd, Path("../pure_gauge/su3_ora"), input_initial)





def gen_configs_ora( configs : int, every_nth : int, beta : float,
                     lat_initial : Path|str|None, save_dir : Path, input_initial : str, run_cmd : list[str] ) -> str:
    '''Takes a start config from which it generates new ones, saving every measurement interval.'''

    if   lat_initial is None:           lat_initial = 'fresh'
    elif isinstance(lat_initial, Path): lat_initial = f'reload_serial {lat_initial}'

    input_gen = \
        f'''
        warms 0
        trajecs {configs}
        traj_between_meas {every_nth}
        beta {beta}
        steps_per_trajectory 4
        qhb_steps 1
        {lat_initial}
        no_gauge_fix
        forget
        save_dir {save_dir}
        '''
    
    return run_command_stream(run_cmd, Path("../pure_gauge/su3_ora"), input_initial + input_gen)








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





def flow_in_steps(flow_times : List[float], lat_initial : Path, input_initial : str, flow : str, run_cmd : list[str]) -> str:
    '''Flows lat_initial to flow times (rkmk3).
    input_initial: prompt, nx, ny, nz, nt.'''

    if sorted(flow_times) != list(flow_times):
        raise Exception('List flow_times not sorted in ascending order.')
    
    tf_current = 0
    for i, tf in enumerate(flow_times):
        tf_step = tf - tf_current
        input_initial += \
            f'''
            {f'reload_serial {lat_initial}' if i==0 else 'continue'}
            {flow}
            exp_order 8
            stepsize {tf_step if tf_step>0 else 11111}
            stoptime {tf_step}
            forget
            '''
        tf_current = tf

    return run_command_stream(run_cmd, Path("../wilson_flow/region_flow_rkmk3"), input_initial)





def flow_rkmk3(stoptime : float, stepsize : float, lat_initial : Path, input_initial : str, flow : str, run_cmd : list[str]) -> str:
    '''Flows lat_initial with given stoptime and stepsize using gradient flow with rkmk3 integrator.
    input_initial: prompt, nx, ny, nz, nt.'''
    
    input_flow = \
        f'''
        reload_serial {lat_initial}
        {flow}
        exp_order 8
        stepsize {stepsize}
        stoptime {stoptime}
        forget
        '''
    
    return run_command_stream(run_cmd, Path("../wilson_flow/region_flow_rkmk3"), input_initial + input_flow)




def flow_adpt(stoptime : float, stepsize : float, local_tol : float, lat_initial : Path, input_initial : str, flow : str, run_cmd : list[str]) -> str:
    '''Flows lat_initial with given stoptime and local_tol (and initial stepsize) using gradient flow with rkmk3 adaptive stepsize.
    input_initial: prompt, nx, ny, nz, nt.'''
    
    input_flow = \
        f'''
        reload_serial {lat_initial}
        {flow}
        exp_order 8
        stepsize {stepsize}
        local_tol {local_tol}
        stoptime {stoptime}
        forget
        '''
    
    # print('XXXXXXXXX')
    # print()
    # print()
    # print(repr(input_initial + input_flow))
    # print()
    # print()
    # print('XXXXXXXXX')
    
    return run_command_stream(run_cmd, Path("../wilson_flow/region_flow_adpt"), input_initial + input_flow)










