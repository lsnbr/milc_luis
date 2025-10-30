import os
import sys
from pathlib import Path

from flowing import gen_input_initial, gen_configs_ora








ns = 64
nt = 16

T_Tc = 1.5      # just as info
beta = 6.868    # depends on nt and T_Tc


number_of_saved_configs = 2
save_every_nth_config   = 10




# different for each execution of this file
ncores, branch = map(int, sys.argv[1:])

iseed = 2314 + 1_000_000*branch

scratch_path    = Path('/work/scratch/ln29bamu')
lattice_initial = scratch_path / 'gauge_configs' / f'branch{branch}' / 'pg_00006000.lat'
save_dir        = scratch_path / 'gauge_configs' / f'branch{branch}c'






run_cmd = [
    'srun',
    '--exclusive',
    '-n',
    str(ncores),
]


# outputs is livestreamed to stdout AND saved in the end into out
out = gen_configs_ora(

    configs       = number_of_saved_configs * save_every_nth_config,
    every_nth     = save_every_nth_config,

    beta          = beta,

    lat_initial   = lattice_initial,
    save_dir      = save_dir,

    input_initial = gen_input_initial(ns, nt, iseed),

    run_cmd       = run_cmd

)






# redirect stdout in sbatch file instead
# (Path('outputs') / 'ora_test.txt').write_text(out)


