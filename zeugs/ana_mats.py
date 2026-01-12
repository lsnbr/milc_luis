from pathlib import Path
import numpy as np
import gvar as gv
import lsqfit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flowing import flowtime_to_radius, radius_to_flowtime
from measurements import *
from statana import *




nt = 16
ns = 64
flowtimes = np.array([ 0.02572923590301565, 0.05303278154520656, 0.08103495224784385, 0.110208782326772,  0.141058391298962,  0.1736893835495303,
                       0.2086835114617738,  0.2454549361253429,  0.2838117228448263,  0.3249152519239425, 0.3705101434001126, 0.4179449298122797,
                       0.4696233239687699,  0.5279559816139976,  0.5890993819917063,  0.6570215111572449, 0.734680305158274,  0.8257738792221512,
                       0.9353328035912798,  1.06959076807306,    1.233820281085228,   1.435936459574629 ], dtype=float)#, 1.68034447439404, 1.972161672512587 ]





def main():

    # plot stuff
    nrows, ncols = 4, 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
    axes = np.reshape(axes, shape=(nrows, ncols))

    # fourier_and_save()

    # preparing data
    print('loading matsu data...')
    ense_all = get_data_mats_unbinned()
    print(ense_all.shape, ense_all.dtype)
    dist = radial_separations(ns)

    # bin with constant r bins
    bins = find_distance_bins(dist, 0.5)
    dist_binned, ense_all_binned = bin_averages(bins, dist, ense_all)
    print(ense_all_binned.shape, ense_all_binned.dtype)

    # matsubara modes data
    iflow = 10

    data_m0 = gv.dataset.avg_data(ense_all_binned[:, iflow, 0, :])
    data_m1 = gv.dataset.avg_data(ense_all_binned[:, iflow, 1, :])
    data_m2 = gv.dataset.avg_data(ense_all_binned[:, iflow, 2, :])


    # modes times r sinh(w_n r)
    data_m0b = data_m0 * dist_binned**2
    data_m1b = data_m1 * dist_binned * np.sinh(mats_freq(1) * dist_binned) / mats_freq(1)
    data_m2b = data_m2 * dist_binned * np.sinh(mats_freq(2) * dist_binned) / mats_freq(2)


    # subtracted
    data_sub = data_m0b + (1/3) * data_m1b + (-4/3) * data_m2b

    plot_dist(dist_binned[:50], data_sub[:50], axes[3,0])
    axes[3,0].set_ylim(-20, 20)


    plot_dist(dist_binned[:50], data_m0b[:50], axes[0,0], label='mats n=0')
    plot_dist(dist_binned[:50], data_m1b[:50], axes[0,0], label='mats n=1')
    plot_dist(dist_binned[:50], data_m2b[:50], axes[0,0], label='mats n=2')
    axes[0,0].set_ylim(-44, 44)

    # smaller 
    plot_dist(dist_binned[:50], data_m0b[:50], axes[1,0], label='mats n=0')
    plot_dist(dist_binned[:50], data_m1b[:50], axes[1,0], label='mats n=1')
    plot_dist(dist_binned[:50], data_m2b[:50], axes[1,0], label='mats n=2')
    axes[1,0].set_ylim(-4, 0.7)

    # even smaller 
    plot_dist(dist_binned[:50], data_m0b[:50], axes[2,0], label='mats n=0')
    plot_dist(dist_binned[:50], data_m1b[:50], axes[2,0], label='mats n=1')
    plot_dist(dist_binned[:50], data_m2b[:50], axes[2,0], label='mats n=2')
    axes[2,0].set_ylim(-1, 0.2)

    # finalize figure
    plt.tight_layout()
    plt.savefig(Path.cwd() / 'zeugs' / 'plots' / 'gridmats.png', dpi=400)







def mats_freq(mats : int) -> float:
    '''matsubara frequency in units of (1/a)'''

    return mats * 2*np.pi / nt 





def get_mats012_subtraction(ense : np.ndarray) -> np.ndarray:
    '''computes m0 + 1/3 m1 - 4/3 m2, which removes 1/w^2 and 1/w^4 contributions'''

    return ense[:,:,0,:] + (1/3) * ense[:,:,1,:] + (-4/3) * ense[:,:,2,:]














def fourier_ense(ense : np.ndarray) -> np.ndarray:
    '''do discrete fft over euclidean time (tau).'''

    # add redundent (because symmetric around beta/2) time separations
    nt_half = ense.shape[-2] - 1
    ense_redu = np.concatenate(
        [ ense, ense[:, :, nt_half-1:0:-1, :] ],
        axis = -2
    )

    # ense.shape = (config, flowtime, tau, r)
    ense_mats = np.fft.rfft(ense_redu, axis=-2)

    # due to the symmetry around beta/2 the fourier transform is real
    return ense_mats.real




def fourier_and_save() -> None:
    '''do fourier in tau, then save'''

    ense = np.load(Path.cwd() / 'zeugs' / 'data' / 'data0.npy')
    ense_mats = fourier_ense(ense)
    np.save(Path.cwd() / 'zeugs' / 'data' / 'data0m.npy', ense_mats)

































if __name__ == '__main__':

    main()
