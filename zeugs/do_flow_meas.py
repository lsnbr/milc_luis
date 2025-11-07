import os
import sys
from pathlib import Path

from flowing import flow_in_steps, gen_input_initial





slurm = True 

ncores  = int(os.environ['SLURM_NTASKS']) if slurm else 8
run_cmd = ['srun' if slurm else 'mpirun', '-n', str(ncores)]





ns = 64
nt = 16


local_tol = 3.00e-04    # flowtimes based on this (not used here)

flowtimes = [ 0.02572923590301565, 0.05303278154520656, 0.08103495224784385, 0.110208782326772, 0.141058391298962, 0.1736893835495303,
              0.2086835114617738, 0.2454549361253429, 0.2838117228448263, 0.3249152519239425, 0.3705101434001126, 0.4179449298122797,
              0.4696233239687699, 0.5279559816139976, 0.5890993819917063, 0.6570215111572449, 0.734680305158274, 0.8257738792221512,
              0.9353328035912798, 1.06959076807306, 1.233820281085228, 1.435936459574629 ]#, 1.68034447439404, 1.972161672512587 ]

flow_type = 'zeuthen'




scratch_path = Path('/work/scratch/ln29bamu')

gauge_folder = sys.argv[1]

lattice_initial  = lambda config: scratch_path / 'gauge_configs' / gauge_folder / f'pg_{config:08}.lat'
flow_output_file = lambda config: scratch_path / 'gauge_configs' / gauge_folder / f'pg_{config:08}.flow'


last_config = int(sys.argv[2])
configs = list(range(100, last_config + 1, 100))





for config in configs:

    out = flow_in_steps(
        flow_times    = flowtimes,
        lat_initial   = lattice_initial(config),
        input_initial = gen_input_initial(ns, nt),
        flow          = flow_type,
        run_cmd       = run_cmd
    )

    flow_output_file(config).write_text(out)



