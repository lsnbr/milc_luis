import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from flowing import flow_rkmk3, flow_params, gen_input_initial, radius_to_flowtime
from measurements import parse_flow_output, make_distance_corr_arrays, radial_separations





cluster = False

ncores  = int(os.environ['SLURM_NTASKS']) if cluster else 4
run_cmd = 'srun' if cluster else 'mpirun'






nt = 16
ns = 64

flow_type = 'zeuthen'

lattice_initial = Path('gauge_configs') / 'pg_00009.lat'




rf_max = 0.125  # in units of beta

# stepsizes = (2^0, 2^1, 2^2, ..., 2^max_log_steps)
max_log_steps = 5

# location of flow output files
file = lambda ln: Path('outputs') / 'flow_err_test' / f'flow_ln={ln}.txt'





# compute flows for various stepsizes, skip if already done
for ln in range(max_log_steps + 1):
    if file(ln).exists(): continue

    stoptime, stepsize = flow_params(2**ln, nt, rf_max)
    out = flow_rkmk3(
        stoptime      = stoptime,
        stepsize      = stepsize,
        lat_initial   = lattice_initial,
        input_initial = gen_input_initial(ns, nt),
        flow          = flow_type,
        ncores        = ncores,
        run_cmd       = run_cmd
    )
    file(ln).write_text(out)





# extract final flowtime measurement for each stepsize ln
finalmeas_for_ln = [parse_flow_output(file(ln).read_text())[-1] for ln in range(max_log_steps + 1)]

# some checks
assert len(finalmeas_for_ln[0].q_corrs) == (nt//2)
assert len(finalmeas_for_ln[0].q_corrs[0]) == (3*(ns//2)**2 + 1)
for ln in range(max_log_steps+1):
    assert finalmeas_for_ln[ln].flow_time == radius_to_flowtime(rf_max, nt)




# extract G(s) data series for each ln and each tau
corrsdist_ln_tau = []
for meas in finalmeas_for_ln:
    corrsdist_ln_tau.append([ make_distance_corr_arrays(corrs_r2, False)[1]     # calling make_dist... repeatedly could be made much faster
                                for corrs_r2 in meas.q_corrs ])



# ln's elements is diff between ln and ln+1
diff_for_ln    = []    
differr_for_ln = []   


# compute average difference of correlators C(t,r) between two different stepsizes
for ln in range(max_log_steps):
    sq_diff_vals = []

    for tslice0, tslice1 in zip(corrsdist_ln_tau[ln], corrsdist_ln_tau[ln+1]):
        for corr0, corr1 in zip(tslice0, tslice1):
            mid = (corr0 + corr1) / 2
            sq_diff_vals.append( (corr0 - corr1)**2 )

    mean_sq_diff      = sum(sq_diff_vals[:100]) / len(sq_diff_vals[:100])
    root_mean_sq_diff = np.sqrt(mean_sq_diff)
    diff_for_ln.append(root_mean_sq_diff)

    # do variance...
    ...



# plot the errors
if 1:
    plt.errorbar(
        x      = [2**ln for ln in range(max_log_steps)],
        y      = diff_for_ln,
        # yerr   = differr_for_ln,
        marker = 'o',
    )
    # plt.ylim(0, diff_for_ln[1] * 1.5)
    plt.yscale('log')
    plt.xlabel('Number n of flow-steps')
    plt.ylabel('Average absolute error between n and 2n flow-steps')
    plt.savefig('plots/plot_flow_wilson_error.png')


# plot some correlators
if 0:
    tau = 3
    ln0, ln1 = max_log_steps-1, max_log_steps
    r_vals = radial_separations(ns)[:100]
    plt.scatter(r_vals, corrsdist_ln_tau[ln0][tau][:100], s=0.3, label=f'{tau=}, ln={ln0}')
    plt.scatter(r_vals, corrsdist_ln_tau[ln1][tau][:100], s=0.3, label=f'{tau=}, ln={ln1}')
    plt.xlabel('radial separation r')
    plt.ylabel('C(tau,r)')
    plt.legend()
    plt.savefig('plots/plot_some_corrs_stepsizes.png')