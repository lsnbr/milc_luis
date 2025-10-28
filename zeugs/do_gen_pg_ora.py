import os
from pathlib import Path

from flowing import gen_input_initial, gen_configs_ora







ns = 64
nt = 16

T_Tc = 1.5      # just as info
beta = 6.868    # depends on nt and T_Tc

iseed = 2314


number_of_saved_configs = 10
save_every_nth_config   = 10

lattice_initial = None


ncores = int(os.environ['SLURM_NTASKS'])







out = gen_configs_ora(

    configs       = number_of_saved_configs * save_every_nth_config,
    skip          = save_every_nth_config,

    beta          = beta,
    lat_initial   = lattice_initial,
    input_initial = gen_input_initial(ns, nt, iseed),

    ncores        = ncores,
    run_cmd       = 'srun'

)






# redirect stdout in sbatch file instead
# (Path('outputs') / 'ora_test.txt').write_text(out)


