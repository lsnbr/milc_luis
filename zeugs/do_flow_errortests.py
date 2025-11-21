import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt # type: ignore

from flowing import flow_rkmk3, flow_adpt, flow_params, gen_input_initial, radius_to_flowtime, flowtime_to_radius
from measurements import parse_flow_output, make_distance_corr_arrays, radial_separations, radial_multiplicities, bin_by_distance, extract_flowtimes






############################################################
##################  general parameters  ####################
############################################################


SLURM = False

ncores  = int(os.environ['SLURM_NTASKS']) if SLURM else 8
run_cmd = ['srun' if SLURM else 'mpirun', '-n', str(ncores)]




nt = 16
ns = 64

flow_type = 'zeuthen'


configs = [11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000][:-2]

scratch_path = Path('/work/scratch/ln29bamu')
lattice_initial = lambda config: scratch_path / 'gauge_configs' / 'init' / f'pg_000{config:05}.lat'




rf_max = 0.25  # in units of beta

ADAPTIVE = True

# local_tol's if adaptive else number of steps (last one is reference)
flow_precisions = (
    [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-6]
    if ADAPTIVE else
    [2, 4, 8, 16, 32]
)

# location of flow output files
code_path = Path('/home/ln29bamu') / 'code' / 'milc_luis'
flowfiles_path = code_path / 'zeugs' / 'outputs' / 'flow_err_test' / 'zeuthen' / '16x64_0.25b'
file = lambda config, prec: flowfiles_path / f'config{config:05}-{"flow_tol" if ADAPTIVE else "flow_n"}={prec:e}.txt'






############################################################
###################  extracting data  ######################
############################################################


# compute flows for various stepsizes, skip if already done
print('generate data..')
for config in configs:
    for prec in flow_precisions:
        if file(config, prec).exists(): continue

        if ADAPTIVE:
            out = flow_adpt(
                stoptime    = radius_to_flowtime(rf_max, nt), stepsize      = radius_to_flowtime(1/nt, nt),
                local_tol   = prec,
                lat_initial = lattice_initial(config),        input_initial = gen_input_initial(ns, nt),
                flow        = flow_type,                      run_cmd       = run_cmd
            )

        else:
            stoptime, stepsize = flow_params(prec, nt, rf_max)
            out = flow_rkmk3(
                stoptime    = stoptime,                stepsize      = stepsize,
                lat_initial = lattice_initial(config), input_initial = gen_input_initial(ns, nt),
                flow        = flow_type,               run_cmd       = run_cmd
            )

        file(config, prec).write_text(out)



# extract flow measurement data from text files
print('extract data..')
flowmeas_config_prec = [ [parse_flow_output(file(config, prec).read_text()) for prec in flow_precisions]
                         for config in configs ]

# extract final flowtime measurement for each config and precision (local_tol or number of steps)
finalmeas_config_prec = [ [flowmeas[-1] for flowmeas in flowmeas_prec]
                          for flowmeas_prec in flowmeas_config_prec ]

# some checks
assert len(finalmeas_config_prec[0][0].q_corrs) == (nt//2)
assert len(finalmeas_config_prec[0][0].q_corrs[0]) == (3*(ns//2)**2 + 1)
for iprec, prec in enumerate(flow_precisions):
    assert finalmeas_config_prec[0][iprec].flow_time == radius_to_flowtime(rf_max, nt)



# extract G(r) data series for each ln and each tau
dr_vals = radial_multiplicities(ns)
corrs_config_prec_tau_r = [ [ [ make_distance_corr_arrays(corrs_r2, False, dr_vals)
                                for corrs_r2 in finalmeas.q_corrs ]
                              for finalmeas in finalmeas_prec ]
                            for finalmeas_prec in finalmeas_config_prec ]

corrs_config_prec_tau_r = np.array(corrs_config_prec_tau_r)

corrsavg_prec_tau_r = corrs_config_prec_tau_r.mean(axis=0)





############################################################
################  differences for each r  ##################
############################################################
print('compute differences..')


tau = 6      # plot for some fixed tau
rbs = 2      # r-binsize

RELATIVE = True


# list of possible radial separations (every nth)
r_vals = radial_separations(ns)


ref = corrsavg_prec_tau_r[-1]
diff = corrsavg_prec_tau_r[:-1] - ref

if RELATIVE: diff = np.abs(diff / ref)
else:        diff = np.abs(diff)



for iprec, prec in enumerate(flow_precisions[:-1]):
    label = f'local_tol={prec:.2e}' if ADAPTIVE else f'nsteps={prec}'
    plt.plot(*bin_by_distance(r_vals, diff[iprec, tau], rbs), label=label, marker='o', ms=3, alpha=0.75)

plt.yscale('log')
plt.xlabel('separation r (binned)')
plt.ylabel(f'Diff({tau=}, r; ln)')
plt.title('Diff(tau, r; ln) = C(tau, r; ln) - C(tau, r; ref)')
plt.legend()
plt.savefig( Path('/home/ln29bamu/code/milc_luis/zeugs/plots') / f"flow_errors_binned_{'adpt' if ADAPTIVE else 'fixed'}_{'rel' if RELATIVE else 'abs'}.png" )
plt.close()





############################################################
################    adaptive flowtimes    ##################
############################################################



# extract flowtimes generated by adaptive stepsize algorithm
flowtimes_prec = [ max( [extract_flowtimes(flowmeas_config_prec[iconfig][iprec]) for iconfig in range(len(configs))], 
                        key=len )
                   for iprec in range(len(flow_precisions)) ]

for iprec, flowtimes in enumerate(flowtimes_prec[:-1]):
    print(f'{flow_precisions[iprec]:.2e}', flowtimes)
    plt.scatter(
        x      = flowtimes[:],
        # x      = [flowtime_to_radius(ft, nt) for ft in flowtimes],    # flowradius instead of flowtime
        y      = [iprec] * len(flowtimes[:]),
        label  = f'local_tol={flow_precisions[iprec]:.2e}',
        marker = '|',
    )

# plt.axvline(x = 1/nt, color='grey', label='1a')
plt.axvline(x = radius_to_flowtime(1/nt, nt), color='grey', label='r=a')

for tau in range(nt//2):
    plt.axvline(x = radius_to_flowtime(tau/(2*nt), nt), color='red', alpha=0.4)#, label=f'rmax({tau=})')

# plt.xlabel('flowtime radius in units of beta')
plt.xlabel('flowtime in units of a^2')
plt.xlim(0, max(flowtimes_prec[-1]))
plt.title('flowtimes from adaptive algorithm')
plt.legend(loc='lower right')
plt.savefig( Path('/home/ln29bamu/code/milc_luis/zeugs/plots') / 'flowtimes_adaptive.png' )





