import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from flowing import flow_rkmk3, flow_adpt, flow_params, gen_input_initial, radius_to_flowtime
from measurements import parse_flow_output, make_distance_corr_arrays, radial_separations, radial_multiplicities, bin_by_distance






############################################################
##################  general parameters  ####################
############################################################


SLURM = False

ncores  = int(os.environ['SLURM_NTASKS']) if SLURM else 4
run_cmd = ['srun' if SLURM else 'mpirun', '-n', str(ncores)]




nt = 16
ns = 16

flow_type = 'zeuthen'

lattice_initial = Path('gauge_configs') / 'pg_00000100.lat'




rf_max = 0.125  # in units of beta

ADAPTIVE = True

# local_tol's if adaptive else number of steps
flow_precisions = (
    [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    if ADAPTIVE else
    [2, 4, 8, 16, 32]
)

# location of flow output files
file = lambda prec: Path('outputs') / 'flow_err_test' / 'zeuthen' / (f'flow_tol={prec:e}.txt' if ADAPTIVE else f'flow_n={prec}.txt')






############################################################
###################  extracting data  ######################
############################################################


# compute flows for various stepsizes, skip if already done
for prec in flow_precisions:
    if file(prec).exists(): continue

    if ADAPTIVE:
        out = flow_adpt(
            stoptime    = radius_to_flowtime(rf_max, nt), stepsize      = radius_to_flowtime(1/nt, nt),
            local_tol   = prec,
            lat_initial = lattice_initial,                input_initial = gen_input_initial(ns, nt),
            flow        = flow_type,                      run_cmd       = run_cmd
        )

    else:
        stoptime, stepsize = flow_params(prec, nt, rf_max)
        out = flow_rkmk3(
            stoptime    = stoptime,        stepsize      = stepsize,
            lat_initial = lattice_initial, input_initial = gen_input_initial(ns, nt),
            flow        = flow_type,       run_cmd       = run_cmd
        )

    file(prec).write_text(out)



# extract final flowtime measurement for each precision (local_tol or number of steps)
finalmeas_for_prec = [parse_flow_output(file(prec).read_text())[-1] for prec in flow_precisions]

# some checks
assert len(finalmeas_for_prec[0].q_corrs) == (nt//2)
assert len(finalmeas_for_prec[0].q_corrs[0]) == (3*(ns//2)**2 + 1)
for iprec, prec in enumerate(flow_precisions):
    assert finalmeas_for_prec[iprec].flow_time == radius_to_flowtime(rf_max, nt)



# extract G(r) data series for each ln and each tau
corrs_prec_tau_r = []
for meas in finalmeas_for_prec:
    corrs_prec_tau_r.append([ make_distance_corr_arrays(corrs_r2, False)[1]     # calling make_dist... repeatedly could be made much faster
                              for corrs_r2 in meas.q_corrs ])






############################################################
################  differences for each r  ##################
############################################################


tau = 3      # plot for some fixed tau
rbs = 1      # r-binsize

RELATIVE = True


# list of possible radial separations (every nth)
r_vals  = radial_separations(ns)
dr_vals = radial_multiplicities(ns)



# prec is difference between prec and prec+1
diff_prec_tau_r = []

for iprec, prec in enumerate(flow_precisions[:-1]):
    diff_tau_r = []

    for corrs_r0, corrs_r1 in zip(corrs_prec_tau_r[iprec], corrs_prec_tau_r[iprec+1]):
        diff_tau_r.append([ abs( (corr1-corr0) / (corr1 if RELATIVE else 1) )
                            for corr0, corr1 in zip(corrs_r0, corrs_r1)       ])
        
    diff_prec_tau_r.append(diff_tau_r)



for prec, diff_tau_r in zip(flow_precisions[:-1], diff_prec_tau_r, strict=True):
    label = f'local_tol={prec:.2e}' if ADAPTIVE else f'nsteps={prec}'
    plt.plot(*bin_by_distance(r_vals, diff_tau_r[tau], rbs), label=label, marker='o', ms=3, alpha=0.75)
    # plt.scatter(*bin_by_distance(r_vals, diff_tau_r[tau], rbs), label=label', s=4)

plt.yscale('log')
plt.xlabel('separation r (binned)')
plt.ylabel(f'Diff({tau=}, r; ln)')
plt.title('Diff(tau, r; ln) = C(tau, r; ln+1) - C(tau, r; ln)')
plt.legend()
plt.savefig( Path('plots') / f'flow_errors_binned_{'adpt' if ADAPTIVE else 'fixed'}_{'rel' if RELATIVE else 'abs'}.png' )









