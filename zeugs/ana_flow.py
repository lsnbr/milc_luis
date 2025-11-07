from pathlib import Path

from measurements import parse_flow_output





ns = 64
nt = 16

local_tol = 3.00e-04    # flowtimes based on this

flowtimes = [ 0.02572923590301565, 0.05303278154520656, 0.08103495224784385, 0.110208782326772, 0.141058391298962, 0.1736893835495303,
              0.2086835114617738, 0.2454549361253429, 0.2838117228448263, 0.3249152519239425, 0.3705101434001126, 0.4179449298122797,
              0.4696233239687699, 0.5279559816139976, 0.5890993819917063, 0.6570215111572449, 0.734680305158274, 0.8257738792221512,
              0.9353328035912798, 1.06959076807306, 1.233820281085228, 1.435936459574629 ]#, 1.68034447439404, 1.972161672512587 ]

flow_type = 'zeuthen'




scratch_path = Path('/work/scratch/ln29bamu')

flow_output_file = lambda config: scratch_path / 'gauge_configs' / 'branch0c' / f'pg_{config:08}.flow'






flowmeas = parse_flow_output(flow_output_file(100).read_text(), flowtimes)

print('flowtimes =', [f'{m.flow_time:.3f}' for m in flowmeas])
print('rtime_flow =', sum(m.rtime_flow for m in flowmeas))
print('rtime_q    =', sum(m.rtime_q for m in flowmeas))
print('rtime_fft  =', sum(m.rtime_fft for m in flowmeas))
