import os
from pathlib import Path

from flowing import flow_rkmk3, gen_input_initial






ns = 64
nt = 16

stoptimes_stepsizes = [
    (1, 0.2),
]

flow_type = 'zeuthen'

lattice_i = 100
lattice_initial = Path('/work/scratch/ln29bamu') / 'gauge_configs' / f'pg_{lattice_i:08}.lat'

ncores = int(os.environ['SLURM_NTASKS'])






# only works for one segment as of now
out = flow_rkmk3(
    stoptime      = stoptimes_stepsizes[0][0],
    stepsize      = stoptimes_stepsizes[0][1],
    lat_initial   = lattice_initial,
    input_initial = gen_input_initial(ns, nt),
    flow          = flow_type,
    ncores        = ncores,
    run_cmd       = 'srun'
)




