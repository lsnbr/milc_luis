from pathlib import Path
import dataclasses
import numpy as np
import gvar as gv
import lsqfit
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from measurements import *
from statana import *
from fitsumtech import *




from ensemble_data import data_16_x_64_1p5Tc

nt = data_16_x_64_1p5Tc['nt']
ns = data_16_x_64_1p5Tc['ns']
flowtimes = data_16_x_64_1p5Tc['flowtimes']







@dataclasses.dataclass
class FitStuff:
    '''all kind of stuff for and from fit, for viewing and plotting results'''

    labels : dict[Any, str]
    dist   : dict[Any, np.ndarray]
    data   : dict[Any, np.ndarray]
    ense   : dict[Any, np.ndarray]
    rlims  : dict[Any, tuple[float, float]]
    fit    : lsqfit.nonlinear_fit




class FitSubSum:

    def __init__(self, bs : bool = False, printinit : bool = True, **kwargs : Any) -> None:
        '''Setting up ense and dist, and other variables.'''

        # get correlators  G(w_n, r) / T^7
        if printinit: print('loading correlator data...')
        ense_all = get_data_mats_unbinned()
        if bs: self.ense_all = next(gv.dataset.bootstrap_iter(ense_all))
        else:  self.ense_all = ense_all
        if printinit: print(self.ense_all.shape, self.ense_all.dtype)

        # get distances
        self.dist = radial_separations(ns)

        # some variables for fits
        self.iflows : dict[int, list[int]] = {}
        self.r_min_left_list_mats = (
            np.arange(5, 15+1e-6, 1/3),
            np.arange(5, 12+1e-6, 1/3),
            np.arange(5, 10+1e-6, 1/3),
        )
        self.cbin_size = 0.2
        self.cbins     = find_distance_bins(self.dist, self.cbin_size)

        # some parameters for variable sized bins
        self.reltol       = 0.01
        self.max_bin_size = 5

        # place where fits per mats are stored
        self.fitstuff_mats : dict[int, FitStuff] = {}

        # some variables for sums
        self.tsums_sub = {}
        
        self.tsums_sinh = {}
        self.psums_sinh = {}
        self.rleft_sinh  = {}
        self.rright_sinh = {}

        self.psums_binsize = 0.25 if 'psums_binsize' not in kwargs else kwargs['psums_binsize']
        self.dist_psums    = bin_distances(ns, self.psums_binsize)
        



    def get_labels_mats(self, mats : int) -> dict[Any, str]:
        '''generate labels for mats and self.iflows'''

        labels = {}
        for iflow in self.iflows[mats]:
            tf = flowtimes[iflow]
            rf = flowtime_to_radius(tf, nt) * nt
            labels[iflow, mats] = rf'$t_f={tf:.2f}a^2, r_f={rf:.2f}a, n={mats}$'
        return labels
    


    def get_common_iflows(self, mats_vals : Iterable[int]|None = None) -> list[int]:
        '''iflows that are common in all mats'''

        if mats_vals is None: mats_vals = (0,1,2)
        return sorted(set.intersection(*(set(self.iflows[mats]) for mats in mats_vals)))
    

    def get_all_iflows(self, mats_vals : Iterable[int]|None = None) -> list[int]:
        '''get union of iflows of all mats'''

        if mats_vals is None: mats_vals = (0,1,2)
        return sorted(set.union(*(set(self.iflows[mats]) for mats in mats_vals)))




    def fitfcn_mats(self, ifl : int, mats : int, x : np.ndarray) -> np.ndarray:
        return self.fitstuff_mats[mats].fit.fcn({(ifl,mats) : x}, self.fitstuff_mats[mats].fit.p)[ifl,mats]
    
    def fitfcn_sinh(self, ifl : int, mats : int, x : np.ndarray) -> np.ndarray:
        return build_integrand(x, self.fitfcn_mats(ifl, mats, x), mats)
    
    def fitfcn_sub(self, sub : Iterable[int], ifl : int, x : np.ndarray) -> np.ndarray:
        return mats_subtraction(sub, [self.fitfcn_sinh(ifl, mats, x) for mats in sub])
    
    




    ##################################################################
    ###########################  fitting  ############################
    ##################################################################


    def do_fit_one_mat_diff_rmin(self, mats : int, rmin_dict : dict[int, float], ex_max : int, printfits : bool = False) -> FitStuff:
        '''do fit for one mats, but use different rmin for each flowtime'''

        labels = self.get_labels_mats(mats)

        # preliminary fit with constant size bins
        dist_fit0, data_fit0, _, _ = bin_cut_avg_data(
            self.dist, self.ense_all,
            {idx : self.cbins for idx in labels.keys()},
            r_cuts0_left = {(ifl,mats) : rmin_dict[ifl] for ifl in self.iflows[mats]}
        )
        fit0 = fit_flowtime_and_mats_tails_with_prior(
            dist_fit0, data_fit0,
            excited_max=ex_max, mats=mats, corr=False
        )
        if printfits: print(fit0)

        # actual fit with variable sized bins based on previous fit
        vbins = bin_through_simultaneous_fit(
            self.dist, self.ense_all, labels, fit0,
            reltol=self.reltol, max_bin_size=self.max_bin_size
        )
        dist_fit, data_fit, ense_fit, rlims_fit = bin_cut_avg_data(
            self.dist, self.ense_all, vbins,
            r_cuts0_left = {(ifl,mats) : rmin_dict[ifl] for ifl in self.iflows[mats]}
        )
        fit = fit_flowtime_and_mats_tails_with_prior(
            dist_fit, data_fit,
            excited_max=ex_max, mats=mats
        )
        if printfits: print(fit)

        self.fitstuff_mats[mats] = FitStuff(labels, dist_fit, data_fit, ense_fit, rlims_fit, fit)
        return self.fitstuff_mats[mats]

    



    def do_fits_for_many_r_min_lefts(self, r_min_left_list : Iterable[float], iflows : Iterable[int], mats : int, mode : str, ex_max : int, printfit : bool = False) -> list[FitStuff]:
        '''one fit for each r_min_left'''

        fitstuff_list = []
        labels = {(iflow, mats) : f'rf={flowtime_to_radius(flowtimes[iflow],nt)*nt:.2f}a, n={mats}' for iflow in iflows}

        for r_min_left in r_min_left_list:
            print(f'{r_min_left = :.2f}')

            if mode == 'const':
                dist_fit, data_fit, ense_fit, rlims_fit = bin_cut_avg_data(
                    self.dist, self.ense_all,
                    {(iflow, mats) : self.cbins for iflow in iflows},
                    r_min_left=r_min_left, r_max_right=30
                )

            if mode == 'var':
                dist_fit0, data_fit0, _, _ = bin_cut_avg_data(
                    self.dist, self.ense_all,
                    {(iflow, mats) : self.cbins for iflow in iflows},
                    r_min_left=r_min_left, r_max_right=None
                )
                fit0 = fit_flowtime_and_mats_tails_with_prior(
                    dist_fit0, data_fit0,
                    excited_max=ex_max, mats=mats, corr=False
                )
                vbins = bin_through_simultaneous_fit(
                    self.dist, self.ense_all, labels, fit0,
                    reltol=self.reltol, max_bin_size=self.max_bin_size
                )
                dist_fit, data_fit, ense_fit, rlims_fit = bin_cut_avg_data(
                    self.dist, self.ense_all, vbins,
                    r_min_left=r_min_left, r_max_right=None
                )

            fit = fit_flowtime_and_mats_tails_with_prior(
                dist_fit, data_fit,
                excited_max=ex_max, mats=mats
            )
            fitstuff_list.append(FitStuff(labels, dist_fit, data_fit, ense_fit, rlims_fit, fit))
            if printfit: print(fit)
            
        return fitstuff_list
    




    ##################################################################
    ###########################  summing  ############################
    ##################################################################


    def do_sums_for_sub_from_mats(self, sub : Iterable[int], printsums : bool = True) -> dict[int, float]:
        '''use sum over mats sinh for sub sums'''

        res = {}

        for iflow in self.get_common_iflows(sub):
            tsum = mats_subtraction(sub, [self.tsums_sinh[iflow, mats] for mats in sub])
            res[iflow] = tsum
            self.tsums_sub[sub, iflow] = tsum
            if printsums: print(f'sum {sub} = {tsum:.3f} T^4')

        return res



    def do_sums_for_mats(self, mats_vals : int|list[int], printsums : bool = False, **kwargs : Any) -> None:
        '''do sum for each mats sinh integrand individually'''

        cbin_size = kwargs['cbin_size'] if 'cbin_size' in kwargs else 0.25
        cbins     = find_distance_bins(self.dist, cbin_size)
        dist_b,   = bin_averages(cbins, self.dist)

        if isinstance(mats_vals, int): mats_vals = [mats_vals]
        for mats in mats_vals:
            for iflow in self.iflows[mats]:

                ense_int = build_integrand(self.dist, self.ense_all[:, iflow, mats, :], mats)
                ense_int_b, = bin_averages(cbins, ense_int)

                rleft     = self.fitstuff_mats[mats].rlims[iflow,mats][0]
                _, rright = find_rright_where_sn_worse_than_rleft(dist_b, ense_int_b, (lambda x: gv.mean(self.fitfcn_sinh(iflow, mats, np.array([x]))[0])), rleft, 0.2)
                # _, rright = find_rright_where_sn_worse_than_rleft(self.dist, ense_int, (lambda x: gv.mean(self.fitfcn_sinh(iflow, mats, np.array([x]))[0])), rleft, 0.2)
                self.rleft_sinh[iflow, mats]  = rleft
                self.rright_sinh[iflow, mats] = rright
                if printsums: print(f'{rleft=:.2f}, {rright=:.2f}')

                tsum, psum = sum_lin(ense_int, (lambda x: gv.mean(self.fitfcn_sinh(iflow, mats, x))), rleft, rright, self.psums_binsize)
                self.tsums_sinh[iflow, mats] = tsum
                self.psums_sinh[iflow, mats] = psum
                if printsums: print(f'total sum ({mats=}) ({iflow=}) = {tsum:.5f} T^4')





    ##################################################################
    #######################  plot final fits  ########################
    ##################################################################


    def plot_mats_fits(self, axes : np.ndarray|None = None, path : Path|None = None) -> None:
        '''plots fit to matsubara modes'''

        if path is not None:
            nrows, ncols = len(self.get_all_iflows())+1, 3
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
            axes = np.reshape(axes, shape=(nrows, ncols))

        for mats in (0,1,2):
            if mats not in self.fitstuff_mats: continue
            plot_flowtime_and_tau_fits(self.dist, self.fitstuff_mats[mats].labels, self.fitstuff_mats[mats].rlims, self.fitstuff_mats[mats].fit, synchro=True, axes=axes[:,mats:mats+1], max_r=30)
            plot_corr_eigenvals(self.fitstuff_mats[mats].ense, axes[-1, mats])

        if path is not None:
            fig.tight_layout()
            fig.savefig(path, dpi=400)




    def plot_mats_integrand_fits(self, axes : np.ndarray|None = None, path : Path|None = None, **kwargs : Any) -> None:
        '''plots fits to integrands of matsubara modes'''

        iflows_all = self.get_all_iflows()

        if path is not None:
            nrows, ncols = len(iflows_all), 3
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
            axes = np.reshape(axes, shape=(nrows, ncols))

        cbin_size = kwargs['cbin_size'] if 'cbin_size' in kwargs else 0.25
        cbins     = find_distance_bins(self.dist, cbin_size)
        dist_b,   = bin_averages(cbins, self.dist)


        for mats in (0,1,2):
            for iflow in self.iflows[mats]:
                ax = axes[min(i for i,ifl in enumerate(iflows_all) if ifl==iflow), mats]

                r_left, r_right = self.fitstuff_mats[mats].rlims[iflow,mats]
                ir_left, ir_right = index_from_distance(self.dist, r_left), index_from_distance(self.dist, r_right)

                ense_int    = build_integrand(self.dist, self.ense_all[:, iflow, mats, :], mats)
                ense_int_b, = bin_averages(cbins, ense_int)
                data_int_b  = gv.dataset.avg_data(ense_int_b)
                
                plot_dist(dist_b, data_int_b, ax, label=f'data', alpha=0.7)
                plot_fitfcn(self.dist[ir_left:ir_right], self.fitfcn_sinh(iflow, mats, self.dist[ir_left:ir_right]), ax, label=f'fit')

                if (iflow,mats) in self.rleft_sinh:
                    ax.axvline(x=self.rleft_sinh[iflow,mats], color='black', alpha=0.7)
                if (iflow,mats) in self.rright_sinh:
                    ax.axvline(x=self.rright_sinh[iflow,mats], color='brown', alpha=0.7)

                ax.set_xlabel(r'$r / a$')
                ax.set_xlim(5, 20)
                ax.set_ylabel(r'$G sinh / T^7$')
                ax.set_ylim([(-0.02,0.002), (-0.2,0.05), (-1,0.1)][mats])

                ax.legend(loc='lower right')

                tf = flowtimes[iflow]
                rf = flowtime_to_radius(tf, nt) * nt
                ax.set_title(rf'$n = {mats}, t_f = {tf:.2f} a^2 (r_f = {rf:.2f} a)$')


        if path is not None:
            fig.tight_layout()
            fig.savefig(path, dpi=400)





    ##################################################################
    #########################  other plots  ##########################
    ##################################################################


    def plot_iflow_comparison_of_rleft_fits_single_mats(self, mats : int, ex_max : int, axes : np.ndarray|None = None, path : Path|None = None) -> list[list[FitStuff]]:
        '''returns: fitstuff2dlist[iiflow][irmin]'''

        if path is not None:
            nrows, ncols = len(self.iflows), 1
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))

        res = []

        r_min_left_list = self.r_min_left_list_mats[mats]

        for iiflow, iflow in enumerate(self.iflows):
            fitstuff_list = self.do_fits_for_many_r_min_lefts(r_min_left_list, [iflow], mats, 'var', ex_max)
            pnames = [f'm_0_{mats}'] + [f'dm_{ex}_{mats}' for ex in range(1, ex_max+1)]
            plot_fitp_and_Q(r_min_left_list, fitstuff_list, pnames, axes[iiflow], labelx=f'{iflow}, {flowtimes[iflow]:.2f}')
            res.append(fitstuff_list)

        if path is not None:
            fig.tight_layout()
            fig.savefig(path, dpi=400)

        return res
    



    def plot_effective_mass_curves(self, iflows : int|list[int]|None = None, mats : int|list[int]|None = None, axes : np.ndarray|None = None, path : Path|None = None) -> None:
        '''plots effective mass curves'''

        if iflows is None:            iflows = self.iflows
        elif isinstance(iflows, int): iflows = [iflows]
        if mats is None:              mats = [0,1,2]
        elif isinstance(mats, int):   mats = [mats]

        if path is not None:
            nrows, ncols = len(iflows), len(mats)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
            axes = np.reshape(axes, shape=(nrows, ncols))

        cbinsize = 1
        cbins = find_distance_bins(self.dist, cbinsize)
        
        for iiflow, iflow in enumerate(iflows):
            for imat, mat in enumerate(mats):

                dist, ense = bin_averages(cbins, self.dist, self.ense_all[:, iflow, mat, :])
                data = gv.dataset.avg_data(ense)
                meff = [gv.log(d0/d1) for d0, d1 in zip(data[:-1], data[1:])]

                axes[iiflow, imat].errorbar(
                    x=dist[:-1]+cbinsize/2, y=gv.mean(meff), yerr=gv.sdev(meff),
                    marker='o', linestyle='--',
                )
                axes[iiflow, imat].set_xlim(0, 25)
                axes[iiflow, imat].set_ylim(-1, 2)
                axes[iiflow, imat].set_xlabel(r'$r / a$')
                axes[iiflow, imat].set_ylabel(r'$m_\text{eff}$')
                axes[iiflow, imat].set_title(rf'$t={flowtimes[iflow]:.2f}a^2 \, \text{{and}} \, n={mat}$')

        if path is not None:
            fig.tight_layout()
            fig.savefig(path, dpi=400)







def plot_fitp_and_Q(r_list : Iterable[float], fitstuff_list : Iterable[FitStuff], pnames : list[str], ax : plt.Axes, labelx : str|None = None, **plt_kwargs : Any) -> None:
    '''plots fit paramater and Q in same axes'''

    for pname in pnames:
        ax.errorbar(
            x    = r_list,
            y    = [fitstuff.fit.p[pname].mean for fitstuff in fitstuff_list],
            yerr = [fitstuff.fit.p[pname].sdev for fitstuff in fitstuff_list],
            marker    = 'o',
            linestyle = '--',
            # label     = 'm/T' + ('' if labelx is None else f' ({labelx})'),
            label     = pname,
            **plt_kwargs
        )
        ax.set_xlabel('r_0 / a')
        ax.set_ylabel('m / T')

    ax_twin = ax.twinx()
    ax_twin.plot(
        r_list,
        [fitstuff.fit.Q for fitstuff in fitstuff_list],
        marker    = 's',
        linestyle = '-',
        alpha     = 0.5,
        label     = 'Q' + ('' if labelx is None else f' ({labelx})'),
        **plt_kwargs
    )
    ax_twin.set_ylabel('Q')
    ax_twin.set_ylim(0, 1)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')







##################################################################
########  general integrand and subtraction constructions  #######
##################################################################


def build_integrand(dist : np.ndarray, data : np.ndarray, mats : int) -> np.ndarray:
    '''computes integrand of H_E in units of T^7'''

    if mats == 0:
        res = data
    
    else:
        w = p_mats(mats)
        fac = np.empty_like(dist)
        if dist[0] == 0:
            fac[0] = 1
            fac[1:] = gv.sinh(w*dist[1:]/nt) / (w*dist[1:]/nt)
        else:
            fac = gv.sinh(w*dist/nt) / (w*dist/nt)
        res = data * fac

    return res




def mats_subtraction(sub : tuple[int], datas : Iterable[np.ndarray]) -> np.ndarray:
    '''do subtraction depending on sub'''

    if sub == (0,1):   return datas[0] - datas[1]
    if sub == (0,1,2): return datas[0] + (-4/3) * datas[1] + (1/3) * datas[2]

    raise Exception(f'Subtraction for {sub=} is not implemented.')



