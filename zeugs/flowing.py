from pathlib import Path
import numpy as np
from typing import List

from various_goods import run_command_stream


ncores = 8





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





def do_warmups(sweeps : int, beta : float, input_initial : str, lat_out : Path) -> str:
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
        '''
    
    return run_command_stream(Path("../pure_gauge/su3_ora"), input_initial, ncores)





def gen_configs_ora(configs : int, skip : int, beta : float, lat_initial : Path, input_initial : str) -> str:
    '''Takes a start config from which it generates new ones, saving every measurement interval.'''

    input_gen = \
        f'''
        warms 0
        trajecs {configs}
        traj_between_meas {skip}
        beta {beta}
        steps_per_trajectory 4
        qhb_steps 1
        reload_serial {lat_initial}
        no_gauge_fix
        forget
        '''
    
    return run_command_stream(Path("../pure_gauge/su3_ora"), input_initial + input_gen, ncores)





def flow_in_steps(flow_times : List[float], lat_initial : Path, input_initial : str) -> str:
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
            zeuthen
            exp_order 8
            stepsize {tf_step if tf_step>0 else 111}
            stoptime {tf_step}
            forget
            '''
        tf_current = tf

    return run_command_stream(Path("../wilson_flow/region_flow_rkmk3"), input_initial, ncores)





def flow_rkmk3(stoptime : float, stepsize : float, lat_initial : Path, input_initial : str) -> str:
    '''Flows lat_initial with given stoptime and stepsize using zeuthen flow with rkmk3 integrator.
    input_initial: prompt, nx, ny, nz, nt.'''
    
    input_flow = \
        f'''
        reload_serial {lat_initial}
        zeuthen
        exp_order 8
        stepsize {stepsize}
        stoptime {stoptime}
        forget
        '''
    
    return run_command_stream(Path("../wilson_flow/region_flow_rkmk3"), input_initial + input_flow, ncores)





def flow_params(n_steps : int, Nt : int, r_max : float = 0.25) -> tuple[float, float]:
    '''Given amnount of steps ([0,1,2] has two steps), number of time divisions and maximal flow radius,
    computes (stoptime, stepsize) parameters in lattice units.'''

    stoptime = (r_max * Nt)**2 / 8
    stepsize = stoptime / n_steps

    return stoptime, stepsize









if __name__ == '__main__':


    ns = 16
    nt = 16

    beta = (
            6.237,  # nt = 8,  T/Tc = 1.3
            6.531,  # nt = 12, T/Tc = 1.3
            6.754,  # nt = 16, T/Tc = 1.3
            6.623,  # nt = 16, T/Tc = 1.1
            5.826,  # nt = 4,  T/Tc = 1.3
            6.868,  # nt = 16, T/Tc = 1.5
            6.640,  # nt = 12, T/Tc = 1.5
            6.337,  # nt = 8,  T/Tc = 1.5
        )[2]
    
    iseed = 2314



    # do warmups
    if 0:

        sweeps = 1000
        
        input_initial = gen_input_initial(ns, nt, iseed=iseed)

        out = do_warmups(
            sweeps        = sweeps,
            beta          = beta,
            input_initial = input_initial,
            lat_out       = Path('thermalized_configs') / f'ns{ns}_nt{nt}_T1p3_{sweeps}hb.lat'
        )

        print(out)



    # generate pg configs
    if 0:

        input_initial = gen_input_initial(ns, nt, iseed=iseed)
    
        out = gen_configs_ora(
            configs       = 10000,
            skip          = 100,
            beta          = beta,
            lat_initial   = Path('thermalized_configs') / f'ns{ns}_nt{nt}_T1p3_1000hb.lat',
            input_initial = input_initial
        )

        (Path('outputs') / 'ora_test.txt').write_text(out)



    # do flow with prescribed itermediate flowtimes
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



    # do flow with prescibed stoptime and stepsize
    if 0:

        input_initial = gen_input_initial(ns, nt)
        
        out = flow_rkmk3(
            stoptime      = 1,
            stepsize      = 0.1,
            lat_initial   = Path('thermalized_configs') / f'ns{ns}_nt{nt}_T1p3_1000hb.lat',
            input_initial = input_initial
        )

        print(out)

        (Path('outputs') / 'flow_test.txt').write_text(out)



    # flow for ensemble
    if 1:

        input_initial = gen_input_initial(ns, nt)

        stoptime, stepsize = flow_params(10, nt, r_max=0.15)

        for i_config in range(100):
            print( '#################################################')
            print(f'###############  config {i_config:05}  ##################')
            print( '#################################################')

            out = flow_rkmk3(
                stoptime      = stoptime,
                stepsize      = stepsize,
                lat_initial   = Path('gauge_configs') / f'pg_{i_config:05}.lat',
                input_initial = input_initial
            )

            (Path('outputs') / f'flow_out_{i_config:05}.txt').write_text(out)


