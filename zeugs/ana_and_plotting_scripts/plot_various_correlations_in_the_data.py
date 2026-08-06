from import_and_plotting_tech import *





ense_all = get_data_mats_unbinned()  # shape: (nconf, nflow, nmats, ndist)
dist = radial_separations(ns)
nflow = ense_all.shape[1]


# correlations in r
iflow_r = 16
flow_r  = flowtimes[iflow_r]

ense_modes = ense_all[:, iflow_r, 0:2, :]          
data_modes = gv.dataset.avg_data(ense_modes)                 
corr_modes = gv.evalcorr(data_modes) 

rmin = 0
rmax = 20
il = index_from_distance(dist, rmin)
ir = index_from_distance(dist, rmax)                       

corr_r_m     = corr_modes[1, il:ir, 1, il:ir]      
corr_r_cross = corr_modes[0, il:ir, 1, il:ir]

n_r = corr_r_m.shape[0]


# correlation in t
mats_flow = 1
r_flow    = 10
ir_flow   = index_from_distance(dist, r_flow)

data_flow = gv.dataset.avg_data(ense_all[:, :, mats_flow, ir_flow])
corr_flow = gv.evalcorr(data_flow)                                    


# --- plot 1x3 with single colorbar ---
fig, axes = plt.subplots(
    1, 3,
    figsize=(wlatex, 0.38 * wlatex),
    constrained_layout=True
)

# ----- LEFT: r-corr mode mats_for_r -----
im0 = axes[0].imshow(
    corr_r_m,
    origin="upper",
    cmap="coolwarm",
    vmin=0, vmax=1,
    interpolation="nearest",
    aspect="equal"
)

n_ticks = 5
ticks = np.linspace(0, n_r - 1, n_ticks)
labels = np.linspace(rmin, rmax, n_ticks)

axes[0].set_xticks(ticks)
axes[0].set_yticks(ticks)
axes[0].set_xticklabels([f"{x:.0f}" for x in labels])
axes[0].set_yticklabels([f"{x:.0f}" for x in labels])

axes[0].set_xlabel(None) 
axes[0].set_ylabel(None) 
axes[0].text(
    0.05, 0.05, r"$r/a$",
    transform=axes[0].transAxes,
    ha="left", va="bottom",
    color="black",
    bbox=dict(facecolor="white", edgecolor="none", alpha=1, pad=1.5),
)
# axes[0].set_xlabel(r"$r / a$")
# axes[0].set_ylabel(r"$r / a$")
axes[0].set_title(f"mode 1, $t_\\mathrm{{f}}={flow_r:.2f} a^2$")

# ----- MIDDLE: cross r-corr -----
im1 = axes[1].imshow(
    corr_r_cross,
    origin="upper",
    cmap="coolwarm",
    vmin=0, vmax=1,
    interpolation="nearest",
    aspect="equal"
)

axes[1].set_xticks(ticks)
axes[1].set_yticks(ticks)
axes[1].set_xticklabels([f"{x:.0f}" for x in labels])
axes[1].set_yticklabels([f"{x:.0f}" for x in labels])

axes[1].set_xlabel(None) 
axes[1].set_ylabel(None) 
axes[1].text(
    0.05, 0.05, r"$r/a$",
    transform=axes[1].transAxes,
    ha="left", va="bottom",
    color="black",
    bbox=dict(facecolor="white", edgecolor="none", alpha=1, pad=1.5),
)
# axes[1].set_xlabel(r"$r / a$")
# axes[1].set_ylabel(r"$r / a$")
axes[1].set_title(f"mode 0 vs 1, $t_\\mathrm{{f}}={flow_r:.2f} a^2$")

# ----- RIGHT: flowtime corr -----
im2 = axes[2].imshow(
    corr_flow,
    origin="upper",
    cmap="coolwarm",
    vmin=0, vmax=1,
    interpolation="nearest",
    aspect="equal"
)

nflow = len(flowtimes)
n_ticks_t = 4
tick_idx = np.linspace(0, nflow - 1, n_ticks_t).round().astype(int)

axes[2].set_xticks(tick_idx)
axes[2].set_yticks(tick_idx)
axes[2].set_xticklabels([f"{flowtimes[i]:.2f}" for i in tick_idx])
axes[2].set_yticklabels([f"{flowtimes[i]:.2f}" for i in tick_idx])

axes[2].set_xlabel(None) 
axes[2].set_ylabel(None) 
axes[2].text(
    0.05, 0.05, r"$t_\mathrm{f}/a^2$",
    transform=axes[2].transAxes,
    ha="left", va="bottom",
    color="black",
    bbox=dict(facecolor="white", edgecolor="none", alpha=1, pad=1.5),
)
# axes[2].set_xlabel(r"t / $a^2$")
# axes[2].set_ylabel(r"flowtime / $a^2$")
axes[2].set_title(f"mode ${mats_flow}$, $r={r_flow} a$")

# single colorbar
cbar = fig.colorbar(
    im2,
    ax=axes[2],     # attach to the rightmost axis only
    fraction=0.046, # controls colorbar height relative to axis
    pad=0.08
)
cbar.set_label("correlation")

outpath = Path.cwd() / 'zeugs' / 'plots' / 'corr_1x3.pdf'
fig.savefig(outpath, bbox_inches="tight")
plt.close(fig)