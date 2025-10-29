import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from flowing import flow_rkmk3, flow_params, gen_input_initial, radius_to_flowtime
from measurements import parse_flow_output, make_distance_corr_arrays, radial_separations, radial_multiplicities






############################################################
##################  general parameters  ####################
############################################################


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
file = lambda ln: Path('outputs') / 'flow_err_test' / 'zeuthen' / f'flow_ln={ln}.txt'






############################################################
###################  extracting data  ######################
############################################################


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



# extract G(r) data series for each ln and each tau
corrs_ln_tau_r = []
for meas in finalmeas_for_ln:
    corrs_ln_tau_r.append([ make_distance_corr_arrays(corrs_r2, False)[1]     # calling make_dist... repeatedly could be made much faster
                            for corrs_r2 in meas.q_corrs ])






############################################################
#################  average differences  ####################
############################################################


relative = False        # relative or absolute differences
n_used_rvals = 100      # use only the first 100 values of r


# ln's elements is diff between ln and ln+1
diff_for_ln    = []    
differr_for_ln = []   


# compute average difference of correlators C(t,r) between two different stepsizes
for ln in range(max_log_steps):
    sq_diff_vals = []

    for tslice0, tslice1 in zip(corrs_ln_tau_r[ln], corrs_ln_tau_r[ln+1]):
        for corr0, corr1 in zip(tslice0, tslice1):

            mid  = (corr0 + corr1) / 2
            diff = corr1 - corr0
            sq_diff_vals.append( (diff/mid if relative else diff)**2 )

    mean_sq_diff      = sum(sq_diff_vals[:n_used_rvals]) / len(sq_diff_vals[:n_used_rvals])
    root_mean_sq_diff = np.sqrt(mean_sq_diff)
    diff_for_ln.append(root_mean_sq_diff)

    # do variance...
    ...






############################################################
################  differences for each r  ##################
############################################################


tau = 2     # plot for some fixed tau
nth = 10     # only take every nth r (starts with first) 


# list of possible radial separations (every nth)
r_vals  = radial_separations(ns)
dr_vals = radial_multiplicities(ns)

# ln is difference between ln and ln+1
diff_ln_tau_r = []

for ln in range(max_log_steps):
    diff_ln_tau_r.append([ [abs(corr1 - corr0) for corr0, corr1 in zip(tslice0, tslice1)] 
                           for tslice0, tslice1 in zip(corrs_ln_tau_r[ln], corrs_ln_tau_r[ln+1]) ])



for ln, diff_tau_r in enumerate(diff_ln_tau_r):
    plt.plot(r_vals[::nth], diff_tau_r[tau][::nth], label=f'{ln=}')
    # plt.scatter(x=r_vals[::nth], y=diff_tau_r[tau][::nth], label=f'{ln=}')

plt.yscale('log')
plt.xlabel('separation r')
plt.ylabel(f'Diff({tau=}, r; ln)')
plt.title('Diff(tau, r; ln) = C(tau, r; ln+1) - C(tau, r; ln)')
plt.legend()
plt.savefig( Path('plots') / 'flow_errors_r.png' )












# plot the errors
if 0:
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
    plt.scatter(r_vals, corrs_ln_tau_r[ln0][tau][:100], s=0.3, label=f'{tau=}, ln={ln0}')
    plt.scatter(r_vals, corrs_ln_tau_r[ln1][tau][:100], s=0.3, label=f'{tau=}, ln={ln1}')
    plt.xlabel('radial separation r')
    plt.ylabel('C(tau,r)')
    plt.legend()
    plt.savefig('plots/plot_some_corrs_stepsizes.png')