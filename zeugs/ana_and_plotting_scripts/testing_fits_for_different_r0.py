from import_and_plotting_tech import *







fss = FitSubSum(bs=False)
fss.iflows = [8,10,11,12,14,16,18,20,21]
print([f'{flowtimes[ifl]:.2f}' for ifl in fss.iflows])

nrows1, ncols1 = len(fss.iflows), 3
fig1, axes1 = plt.subplots(nrows=nrows1, ncols=ncols1, figsize=(9*ncols1, 4*nrows1))
axes1 = np.reshape(axes1, shape=(nrows1, ncols1))


ex_max = 1

mats = 2

if ex_max == 0:
    fss.r_min_left_list_mats = [
        tuple(np.arange(5, 18+1e-6, 1/2) for _ in range(3)),
        tuple(np.arange(5, 13+1e-6, 1/2) for _ in range(3)),
        tuple(np.arange(5, 11+1e-6, 1/3) for _ in range(3)),
    ][mats]
elif ex_max == 1:
    fss.r_min_left_list_mats = [
        tuple(np.arange(5, 16+1e-6, 1/2) for _ in range(3)),
        tuple(np.arange(5, 13+1e-6, 1/2) for _ in range(3)),
        tuple(np.arange(4, 10+1e-6, 1/3) for _ in range(3)),
    ][mats]
else:
    raise Exception('efjibwiofubwi')

allfits = fss.plot_iflow_comparison_of_rleft_fits_single_mats(mats=mats, ex_max=ex_max, axes=axes1[:,0])







for iifl, ifl in enumerate(fss.iflows):

    if ex_max == 0:
        rmin1, rmin2 = {
            0 : {8:(10.5,13), 10:(11,14.5), 11:(11,15), 12:(11.5,16), 14:(12,16.5), 16:(12.5,17), 18:(13,18), 20:(14,18), 21:(14.5,18)},
            1 : {8:(7,10), 10:(7.5,10.5), 11:(8,11), 12:(8,11), 14:(8,11.5), 16:(9,12), 18:(9,12), 20:(10,13), 21:(10.5,13)},
            2 : {8:(5+2/3,7), 10:(6,7), 11:(6,7+1/3), 12:(6+1/3,7+2/3), 14:(6+2/3,8+1/3), 16:(7,8+1/3), 18:(7+2/3,9), 20:(8,9+1/3), 21:(9,10+1/3)},
        }[mats][ifl]
    elif ex_max == 1:
        rmin1, rmin2 = {
            0 : {8:(6,9.5), 10:(6.5,8.5), 11:(6.5,11), 12:(6.5,11.5), 14:(7,11), 16:(7.5,12), 18:(8,11.5), 20:(9,15), 21:(10,12)},
            1 : {8:(5.5,6.5), 10:(6.5,8), 11:(6.5,9), 12:(7,9.5), 14:(7.5,9), 16:(7,8.5), 18:(8,10), 20:(9,11), 21:(10,12)},
            2 : {8:(5+1/3,6+2/3), 10:(5+2/3,6+2/3), 11:(5+2/3,6+2/3), 12:(6,6+1/3), 14:(6+2/3,7+1/3), 16:(6+2/3,7+2/3), 18:(6,7+2/3), 20:(8,9), 21:(8,9+2/3)},
        }[mats][ifl]
    else:
        raise Exception('wefiwuhefuwihf')
    irmin1, irmin2 = (np.argwhere(np.abs(fss.r_min_left_list_mats[mats] - rmin) < 1e-6)[0][0] for rmin in (rmin1, rmin2))

    for i, (irmin, rmin) in enumerate(zip((irmin1, irmin2), (rmin1, rmin2))):
        ax = axes1[iifl, 1+i]
        r_left, r_right = allfits[iifl][irmin].rlims[ifl,mats]
        ir_left, ir_right = index_from_distance(fss.dist, r_left), index_from_distance(fss.dist, r_right)
        plot_dist(allfits[iifl][irmin].fit.x[ifl,mats], allfits[iifl][irmin].fit.y[ifl,mats], ax, label=f'data', alpha=0.7)
        x_fit = fss.dist[ir_left:ir_right]
        y_fit = allfits[iifl][irmin].fit.fcn({(ifl,mats) : fss.dist[ir_left:ir_right]}, allfits[iifl][irmin].fit.p)[ifl,mats]
        plot_fitfcn(x_fit, y_fit, ax, label=rf'fit $(r_0={rmin:.1f})$')

        ax.set_xlabel(r'$r / a$')
        ax.set_xlim([(6,35), (5,25), (5,18)][mats])
        yliml, ylimr = min(gv.mean(y_fit)), max(gv.mean(y_fit))
        ax.set_ylim(yliml - (ylimr-yliml)*0.1, ylimr + (ylimr-yliml)*0.1)
        ax.set_ylabel(r'$G / T^7$')
        ax.legend(loc='lower right')



fig1.tight_layout()
fnameabc = 'gs' if ex_max==0 else f'ex{ex_max}'
fig1.savefig(Path.cwd() / 'zeugs' / 'plots' / f'mats_rmin_tests_{fnameabc}_mats{mats}.png', dpi=400)