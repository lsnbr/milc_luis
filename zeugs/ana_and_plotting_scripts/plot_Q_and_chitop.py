from import_and_plotting_tech import *
import pickle
  
from ensemble_data import data_16_x_64_1p5Tc as data_ense
nt = data_ense['nt']
ns = data_ense['ns']
flowtimes = data_ense['flowtimes']
 
 
 
 
# flowtime for Q plot
iflow = -1
 
# config skipped for thermalizing
skip_configs = 55
 
# load data
data_path = Path.cwd() / 'zeugs' / 'data'
flowmeas_config : list[FlowMeasurements] = pickle.load((data_path / 'data0a.pkl').open('rb'))[skip_configs:]
n_configs = len(flowmeas_config)
print('number of configs =', n_configs)
 
 
 
# Q for all configs and flowtimes
icharge_config_flow = np.array(
    [ [meas.icharge for meas in flowmeas]
      for flowmeas in flowmeas_config ],
    dtype=float
)
 
# topological charge at the chosen flowtime
Q = icharge_config_flow[:, iflow]
 
# chi_top as function of flowtime (in units of T^4)
top_sus     = np.mean(icharge_config_flow**2, axis=0) / (ns/nt)**3
top_sus_std = np.std(icharge_config_flow**2, axis=0, ddof=1) / np.sqrt(n_configs) / (ns/nt)**3
 
top_sus_4     = top_sus ** (1/4)
top_sus_4_std = top_sus_4 * top_sus_std / (4 * top_sus)
 
 
 
# prepare figure
sc = 0.9
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(sc*wlatex, sc*0.42*wlatex), constrained_layout=True)
 
 
 
# left panel: topological charge for each configuration
ax0.scatter(range(n_configs), Q, s=1)
 
ax0.set_xlabel('config')
ax0.set_ylabel(r'$Q$')
ax0.set_title(
    rf'$t_\mathrm{{f}} = {flowtimes[iflow]:.2f} a^2$'
    rf' $(\sqrt{{8 t_\mathrm{{f}}}} = {flowtime_to_radius(flowtimes[iflow], nt):.2f} \beta)$'
)
 
 
 
# right panel: chi^(1/4) T as function of flowtime
ax1.errorbar(
    flowtimes,
    top_sus_4,
    yerr=top_sus_4_std,
    marker='o',
    markersize=4,
    ls='--',
)
 
ax1.set_xlabel(r'$t_\mathrm{f} / a^2$')
ax1.set_ylabel(r'$\chi_\mathrm{top}^{1/4}\,T$')
ax1.set_ylim(0, 1)
 
 
 
 
# saving the figure
out = Path.cwd() / 'zeugs' / 'plots' / 'Q_and_chi_1x2.pdf'
fig.savefig(out, bbox_inches='tight', dpi=400)
plt.close(fig)
print(f"Saved: {out}")