import os
from pathlib import Path

from flowing import gen_input_initial, gen_configs_ora





slurm = True

ncores  = int(os.environ['SLURM_NTASKS']) if slurm else 4
run_cmd = ['srun' if slurm else 'mpirun', '-n', str(ncores)]





ns = 36
nt = 16

T_Tc = 1.5      # just as info
beta = 6.868    # depends on nt and T_Tc


number_of_saved_configs = 5000
save_every_nth_config   = 25






branch = 0

iseed = 2314 + 1_000_000*branch + 1_000_000_000*(0 if ns==64 else ns)      # third part with ns was not used for ns=64

scratch_path    = Path('/work/scratch/ln29bamu')

# lattice_initial = None
lattice_initial = scratch_path / 'gauge_configs_36' / f'init' / 'pg_00025000.lat'       # continue from last config of some branch

save_dir        = scratch_path / 'gauge_configs_36' / f'branch{branch}'










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


