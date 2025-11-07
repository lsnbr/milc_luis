import os
from pathlib import Path

from flowing import gen_input_initial, gen_configs_ora





slurm = True

ncores  = int(os.environ['SLURM_NTASKS']) if slurm else 4
run_cmd = ['srun' if slurm else 'mpirun', '-n', str(ncores)]





ns = 64
nt = 16

T_Tc = 1.5      # just as info
beta = 6.868    # depends on nt and T_Tc


number_of_saved_configs = 200
save_every_nth_config   = 100




# continue from last config of some branch

branch = 0

iseed = 2314 + 1_000_000*branch

scratch_path    = Path('/work/scratch/ln29bamu')
lattice_initial = scratch_path / 'gauge_configs' / f'branch{branch}cc' / 'pg_00022200.lat'
save_dir        = scratch_path / 'gauge_configs' / f'branch{branch}ccc'

# lattice_initial = None
# save_dir = Path('/home/luis/codeundso/milc_luis/zeugs') / 'gauge_configs'








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






# saves output in file
# (Path('outputs') / 'ora_test.txt').write_text(out)


