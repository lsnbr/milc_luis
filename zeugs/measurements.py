from dataclasses import dataclass
import re
from pathlib import Path
from itertools import product
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




@dataclass(frozen=True)
class Measurements:

    rtime_q    : float  # time in seconds for q=FF* computation
    rtime_fft  : float  # time in seconds for correlator computation
    rtime_flow : float  # time in seconds for flowstep
    
    flow_time  : float

    clover_t   : float
    clover_s   : float
    iclover_t  : float
    iclover_s  : float

    plaq_t     : float
    plaq_s     : float
    rect_t     : float
    rect_s     : float

    charge     : float
    charge_is  : float  # improved fs only for spatial part
    charge_it  : float  # improved fs only for temporal part
    icharge    : float  # improved fs for everything

    q_corrs    : list[list[float]]  # first index is time, second is spatial distance squared: q_corrs[t][s^2]


FlowMeasurements = list[Measurements]




def parse_flow_output(output : str) -> FlowMeasurements:
    '''parsing output of wilson_flow program'''

    flow_measurements = []

    for section in re.split(r'\s*\n\n\s*', output):
        if not section.startswith('Time to complete flowstep'): continue

        lines = re.split(r'\n', section)

        rtime_flow, rtime_q, rtime_fft = [ float(re.search(r'= (.+) seconds', lines[i])[1])
                                           for i in (0,1,4) ]
        
        obs = map(float, lines[2].removeprefix('GFLOW: ').split())

        corrs = [ [float(c) for c in tslice.split()]
                  for tslice in lines[3].removeprefix('q-corrs ').split(',') ]
        
        flow_measurements.append(Measurements(rtime_q, rtime_fft, rtime_flow, *obs, corrs))

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



    # test runtimes of flow and measurements
    if 0:

        with open(Path('outputs') / 'flow_test.txt', 'r', encoding='utf-8') as f:
            flow_output = f.read()

        flow_meas = parse_flow_output(flow_output)

        times_q = []
        times_fft = []
        times_flow = []

        for meas in flow_meas:
            times_q.append(meas.rtime_q)
            times_fft.append(meas.rtime_fft)
            times_flow.append(meas.rtime_flow)

        for name, lst in zip('flow,q   ,fft '.split(','), (times_flow, times_q, times_fft)):
            avg = sum(lst) / len(lst)
            err = ( sum((t - avg)**2 for t in lst) / (len(lst) - 1) )**.5
            print(f'time for {name} = {avg:.2f} ± {err:.2f} seconds')

        print('\n', times_fft)




    # test flow numerical errors as function of stepsize
    if 1:

        from flowing import flow_rkmk3, flow_params, gen_input_initial, radius_to_flowtime

        nt = 16
        ns = 64

        rf_max = 0.125  # in units of beta

        max_log_steps = 5
        file = lambda ln: Path('outputs') / 'flow_err_test' / f'flow_ln={ln}.txt'


        # compute flows for various stepsizes
        for ln in range(max_log_steps + 1):
            if file(ln).exists(): continue

            stoptime, stepsize = flow_params(2**ln, nt, rf_max)
            out = flow_rkmk3(
                stoptime      = stoptime,
                stepsize      = stepsize,
                lat_initial   = Path('gauge_configs') / 'pg_00009.lat',
                input_initial = gen_input_initial(ns, nt),
                flow          = 'zeuthen'
            )
            file(ln).write_text(out)


        # extract final flowtime measurement for each stepsize ln
        finalmeas_for_ln = [parse_flow_output(file(ln).read_text())[-1] for ln in range(max_log_steps + 1)]
        number_t_r2 = len(finalmeas_for_ln[0].q_corrs) * len(finalmeas_for_ln[0].q_corrs[0])

        assert number_t_r2 == (nt//2) * (3*(ns//2)**2 + 1)
        for ln in range(max_log_steps+1):
            assert finalmeas_for_ln[ln].flow_time == radius_to_flowtime(rf_max, nt)

        
        # extract G(s) data series for each ln and each tau
        corrsdist_ln_tau = []
        for meas in finalmeas_for_ln:
            corrsdist_ln_tau.append([ make_distance_corr_arrays(corrs_r2, False)[1]
                                      for corrs_r2 in meas.q_corrs ])


        # compute average difference of correlators between two neighbooring stepsizes
        number_t_r = len(corrsdist_ln_tau[0]) * len(corrsdist_ln_tau[0][0])
        print(len(corrsdist_ln_tau[0]), len(corrsdist_ln_tau[0][0]))
        diff_for_ln    = []    # ln's elements is diff between ln and ln+1
        differr_for_ln = []   

        for ln in range(max_log_steps):
            diff = 0
            for tslice0, tslice1 in zip(corrsdist_ln_tau[ln], corrsdist_ln_tau[ln+1]):
                for corr0, corr1 in zip(tslice0, tslice1):
                    diff += abs(corr0 - corr1)
            diff_for_ln.append(diff / number_t_r)

            differr = 0
            for tslice0, tslice1 in zip(corrsdist_ln_tau[ln], corrsdist_ln_tau[ln+1]):
                for corr0, corr1 in zip(tslice0, tslice1):
                    differr += (abs(corr0 - corr1) - diff_for_ln[ln])**2
            differr_for_ln.append(np.sqrt(differr / (number_t_r-1)))


        # plot the stuff
        plt.errorbar(
            x      = [2**ln for ln in range(max_log_steps)],
            y      = diff_for_ln,
            yerr   = differr_for_ln,
            marker = 'o',
        )
        # plt.ylim(0, diff_for_ln[1] * 1.5)
        plt.yscale('log')
        plt.xlabel('Number n of flow-steps')
        plt.ylabel('Average absolute error between n and 2n flow-steps')
        plt.savefig('plot_flow_wilson_error.png')





    # ensemble flow
    if 0:

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


    






