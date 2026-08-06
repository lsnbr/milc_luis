from typing import Callable, Any
import numpy as np
import gvar as gv
import lsqfit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from measurements import *
from statana import *



from ensemble_data import data_16_x_64_1p5Tc

nt = data_16_x_64_1p5Tc['nt']
ns = data_16_x_64_1p5Tc['ns']
flowtimes = data_16_x_64_1p5Tc['flowtimes']









###############################################################################
########################  priors and parameters  ##############################
###############################################################################



# ground state and first 2 excited 0-+ glueball state masses (in units of T) (in continuum R^3 SU(3))
masses_0p = [6.24, 8.8, 11.04]

# matsubara frequencies (in units of T)
p_mats = lambda mats : mats * 2 * np.pi

# rough estimate of higher matsubara mass
m_mats = lambda m, mats: np.sqrt( m**2 + p_mats(mats)**2 )





def make_prior_mats_constr(mats : int, excited_max : int, iflowtimes : list[int]) -> gv.BufferDict:
    '''constructs prior for single matsubara modes (constraining masses). todo: add e0 < e1 < ... constraint on excited energies'''

    if excited_max > 2:
        raise Exception('Not yet implemented for excited_max > 2.')
    
    # add distribution to reparametrize masses to enforce constraint on higher matsubara mode masses
    if not gv.BufferDict.has_distribution(f'f_p{mats}'):
        gv.BufferDict.add_distribution(
            f'f_p{mats}',
            lambda x: p_mats(mats) + gv.exp(x)
        )

    # gv.BufferDict instead of python dict for custom distribution support
    prior = gv.BufferDict()

    # prior for lowest mass
    m_guess = m_mats(masses_0p[0], mats)
    prior[f'f_p{mats}(m_0_{mats})'] = gv.log(gv.gvar(m_guess, m_guess*1) - p_mats(mats))

    # priors for higher masses
    for n_ex in range(1, excited_max+1):
        dm_guess = m_mats(masses_0p[n_ex], mats) - m_mats(masses_0p[n_ex-1], mats)
        prior[f'log(dm_{n_ex}_{mats})'] = gv.log(gv.gvar(dm_guess, dm_guess*1))
        # prior[f'u(dm_{n_ex}_{mats})'] = gv.BufferDict.uniform('u', 0, 5)

    # priors for all the amplitudes
    for n_ex in range(excited_max+1):
        for ift in iflowtimes:
            prior[f'log(a_{ift}_{n_ex}_{mats})'] = gv.log(gv.gvar(1, 50))

    return prior




def make_prior_mats_uniform(mats : int, ex_max : int, iflows : list[int]) -> gv.BufferDict:
    '''Construct prior for single Matsubara mode using only uniform distributions.'''

    prior = gv.BufferDict()
    for ex in range(ex_max+1):

        if ex == 0:
            lbound = p_mats(mats)
            prior[f'u_0_{mats}(m_0_{mats})'] = gv.BufferDict.uniform(f'u_0_{mats}', lbound, lbound+20)
        elif ex > 0:
            prior[f'u_{ex}_{mats}(dm_{ex}_{mats})'] = gv.BufferDict.uniform(f'u_{ex}_{mats}', 0, 20)

        for iflow in iflows:
            prior[f'u_{iflow}_{ex}_{mats}(a_{iflow}_{ex}_{mats})'] = gv.BufferDict.uniform(f'u_{iflow}_{ex}_{mats}', 0, 100)

    return prior




def make_prior_uniform_many_mats(excited_max : int, mats_list : list[int], iflowtimes : list[int]) -> gv.BufferDict:
    '''...'''

    prior = gv.BufferDict()

    for mats in mats_list:
        prior.update(make_prior_mats_uniform(mats, excited_max, iflowtimes))

    return prior




def make_prior_many_mats(excited_max : int, mats_list : list[int], iflowtimes : list[int]) -> gv.BufferDict:
    '''...'''

    prior = gv.BufferDict()

    for mats in mats_list:
        prior.update(make_prior_mats_constr(mats, excited_max, iflowtimes))

    return prior




def make_prior_constr(excited_max : int, mats_max : int, iflowtimes : list[int]) -> gv.BufferDict:
    '''construct prior, incorporating constraints on masses'''

    prior = gv.BufferDict()

    for mats in range(mats_max+1):
        prior.update(make_prior_mats_constr(mats, excited_max, iflowtimes))

    return prior




def make_p0_mats(ex_max : int, mats : int, iflows : list[int]) -> dict[Any, float]:
    '''Construct starting guesses for single matsubara modes and multiple excited states.'''

    p0 = {}
    for ex in range(ex_max+1):

        if ex == 0:
            p0[f'm_0_{mats}'] = m_mats(masses_0p[0], mats)
        elif ex > 0:
            p0[f'dm_{ex}_{mats}'] = m_mats(masses_0p[ex], mats) - m_mats(masses_0p[ex-1], mats)

        for iflow in iflows:
            p0[f'a_{iflow}_{ex}_{mats}'] = 1

    return p0



def make_p0_many_mats(ex_max : int, mats_list : list[int], iflows : list[int]) -> dict[Any, float]:
    '''Construct starting guesses for multiple matsubara modes.'''

    p0 = {}
    for mats in mats_list:
        p0.update(make_p0_mats(ex_max, mats, iflows))
    return p0





def expx_single_mats(x : float, p : dict, iflow : int, mats : int, ex_max : int) -> float:
    '''fit function for a single matsubara mode and possibly multiple excited states'''

    masses = [p[f'm_0_{mats}']]
    for n_ex in range(1, ex_max+1):
        masses.append(masses[-1] + p[f'dm_{n_ex}_{mats}'])

    return sum(
        expx(
            x / nt,
            # - (nt/ns)**2 * masses[n_ex] * p[f'a_{iflow}_{n_ex}_{mats}'],
            - masses[n_ex] * p[f'a_{iflow}_{n_ex}_{mats}'],
            masses[n_ex]
        )
        for n_ex in range(ex_max + 1)
    )







###############################################################################
############################  some plotting  ##################################
###############################################################################


def plot_flowtime_and_tau_fits( dist : np.ndarray, labels : dict[Any, str], r_lims : dict[Any, tuple[float, float]], fit : lsqfit.nonlinear_fit, synchro : bool, axes : np.ndarray,
                                min_r : float|None = None, max_r : float|None = None, r_cuts_sum : dict[Any, float]|None = None ) -> None:
    '''plot all tails with data and fit, optionally synchro flowtimes across columns'''

    # map (iflow, tau) onto indices of grid of plots
    axes_idxs = {}
    for i_tau, tau in enumerate(sorted({tau for _,tau in labels.keys()})):
        for i_iflow, iflow in enumerate(sorted({iflow for iflow,tau0 in labels.keys() if synchro or tau0==tau})):
            axes_idxs[iflow, tau] = (i_iflow, i_tau)

    # plot original and fitted data
    for idx in labels.keys():
        iflow, tau = idx

        r_left, r_right = r_lims[idx]
        if r_right > dist[-1]:
            r_right *= 0.75 / 1.1
        if max_r is not None:
            r_right = max_r
        if min_r is not None:
            r_left = min_r
        ir_left, ir_right = index_from_distance(dist, r_left), index_from_distance(dist, r_right)

        y_fit = fit.fcn({idx : dist[ir_left:ir_right]}, fit.p)[idx]

        plot_dist(fit.x[idx], fit.y[idx], axes[axes_idxs[idx]])
        plot_fitfcn(dist[ir_left:ir_right], y_fit, axes[axes_idxs[idx]])

        if r_cuts_sum is not None:
            axes[axes_idxs[idx]].axvline(x=r_cuts_sum[idx], color='black', alpha=0.5)

        # y_min = min(min(gv.mean(y_fit)), min(gv.mean(fit.y[idx])))
        # y_max = max(max(gv.mean(y_fit)), max(gv.mean(fit.y[idx])))
        y_min = min(gv.mean(y_fit))
        y_max = max(gv.mean(y_fit))
        y_range = y_max - y_min
        axes[axes_idxs[idx]].set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        axes[axes_idxs[idx]].set_xlim(0 if min_r is None else min_r, r_right)
        axes[axes_idxs[idx]].set_title(rf'{labels[idx]}')

    





###############################################################################
###########################  summing over r  ##################################
###############################################################################


def sum_lin(ense : np.ndarray, fcn : Callable, r_left : float, r_right : float, pbinsize : float) -> tuple[float, np.ndarray]:
    '''Computes sum of data in three parts s1, s2 and s3: s1 is only data, s2 is linear interpolation of data and fcn, and s3 is only fcn.
    Returns: total sum, partial sums at regular intervals.'''

    # prepare dist and its multiplicities
    dist    = radial_separations(ns)
    dr_list = radial_multiplicities_r(ns)

    # indicies of left and right r cutoffs
    ir_left  = index_from_distance(dist, r_left)
    ir_right = index_from_distance(dist, r_right)

    # segments of measured data
    data_left = ense[:, :ir_left].mean(axis=0)
    data_mid  = ense[:, ir_left:ir_right].mean(axis=0)

    # segments of fitted data
    fit_mid   = fcn(dist[ir_left:ir_right])
    fit_right = fcn(dist[ir_right:])

    # combine data and fit
    ratio = (dist[ir_left:ir_right] - r_left) / (r_right - r_left)
    data_mixed = np.concatenate((
        data_left,
        (1-ratio) * data_mid + ratio * fit_mid,
        fit_right
    ))

    # multiply by radial multiplicities, then sum, then convert from T^n to T^(n-3) units
    partial_sums_fine = np.cumsum(data_mixed * dr_list, axis=0) / nt**3

    # partial sums at regular intervals
    cbins    = find_distance_bins(dist, pbinsize)
    cbin_ils = [il for il,_ in cbins]
    partial_sums = partial_sums_fine[cbin_ils]

    return partial_sums_fine[-1], partial_sums


    








if __name__ == '__main__':

    ...,...,...