import os
from pathlib import Path

from flowing import gen_input_initial, radius_to_flowtime, flow_adpt





slurm = False

ncores  = int(os.environ['SLURM_NTASKS']) if slurm else 4
run_cmd = ['srun' if slurm else 'mpirun', '-n', str(ncores)]




ns = 16
nt = 16

local_tol = 1e-2
stoptime = radius_to_flowtime(0.125, nt)
stepsize = radius_to_flowtime(1/nt, nt)

flow_type = 'zeuthen'

# scratch_path = Path('/work/scratch/ln29bamu')
# lattice_initial = scratch_path / ...

lattice_initial = Path('gauge_configs') / 'pg_00000100.lat'






out = flow_adpt(
    stoptime      = stoptime,
    stepsize      = stepsize,
    local_tol     = local_tol,
    lat_initial   = lattice_initial,
    input_initial = gen_input_initial(ns, nt),
    flow          = flow_type,
    run_cmd       = run_cmd
)





( Path('outputs') / 'flow_test.txt' ).write_text(out)




