import os
from pathlib import Path

from flowing import gen_input_initial, gen_configs_ora





# location of lattice configurations is specified in pure_gauge/control.c
# location of stdout, stderr is specified in sbatch script





ns = 64
nt = 16

T_Tc = 1.5      # just as info
beta = 6.868    # depends on nt and T_Tc

iseed = 2314


number_of_saved_configs = 200
save_every_nth_config   = 100

lattice_initial = None

save_dir = Path('/work/scratch/ln29bamu') / 'gauge_configs' / 'init'


ncores = int(os.environ['SLURM_NTASKS'])





run_cmd = [
    'srun',
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


