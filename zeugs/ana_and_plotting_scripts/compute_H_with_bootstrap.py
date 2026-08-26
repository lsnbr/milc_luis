from import_and_plotting_tech import *
import pickle





# preparing parameters used in fits and sums
from some_intermediate_results import rmin_per_iflow_mats_exmax
mats_list = [0, 1, 2]
ex_max_mats = {0:0, 1:0, 2:0}
iflows_mats = {
    0 : [11,12,14,16,18],
    1 : [11,12,14,16,18],
    2 : [11,12,14]
}
sub_list = [(0,1), (0,1,2)]
psums_binsize = 0.2




def do_computation(bs : bool, printstuff : bool = False) -> ResultData:
    '''do full fit and sums'''
    
    fss = FitSubSum(bs=bs, psums_binsize=psums_binsize, printinit=printstuff)
    fss.iflows = iflows_mats

    for mats in mats_list:
        ex_max = ex_max_mats[mats]
        rmin_per_iflow = rmin_per_iflow_mats_exmax[ex_max][mats]
        fss.do_fit_one_mat_diff_rmin(mats, rmin_per_iflow, ex_max, printfits=printstuff)

    fss.do_sums_for_mats(mats_list, printsums=printstuff)
    for sub in sub_list:
        fss.do_sums_for_sub_from_mats(sub, printsums=printstuff)

    return ResultData(
        pmean      = {mats : fss.fitstuff_mats[mats].fit.pmean for mats in mats_list},
        rleft      = fss.rleft_sinh,
        rright     = fss.rright_sinh,
        tsums_mats = fss.tsums_sinh,
        tsums_subs = fss.tsums_sub,
        psums_mats = fss.psums_sinh
    )

# do_computation(bs=False, printstuff=True)






pickle_path = Path.cwd() / 'zeugs' / 'data' / 'bs_H.pkl'
do_comp = False      # compute new bootstrap samples if True
restart = False     # delete old bootstrap samples if True else append to them
n_bs    = 5        # number of new bootstrap samples to compute if do_comp==True

if (not do_comp) or (not restart and pickle_path.exists()):
    with open(pickle_path, 'rb') as f:
        bs_data : list[ResultData] = pickle.load(f)
else:
    bs_data : list[ResultData] = []

if do_comp:
    for i_bs in range(n_bs):
        print(f'bootstrap sample {i_bs+1}/{n_bs}...', end=('\r' if i_bs<n_bs-1 else '\n'))
        bs_data.append(do_computation(bs=True))
        with pickle_path.open('wb') as f:
            pickle.dump(bs_data, f)

print('Number of bootstrap samples =', len(bs_data), end='\n\n\n')






# compute fitting params and H's with errors from bootstrap ensemble? (or load data)
do_comp = True
pickle_path = Path.cwd() / 'zeugs' / 'data' / 'av_H.pkl'


if do_comp:
    # fitting paramters from bootstrap ensemble (only for ex_max=0 right now)
    p_ense = {}
    for mats in mats_list:
        p_ense[f'm_{mats}'] = [rd.pmean[mats][f'm_0_{mats}'] for rd in bs_data]
        for iflow in iflows_mats[mats]:
            p_ense[f'a_{mats}_{iflow}'] = [rd.pmean[mats][f'a_{iflow}_0_{mats}'] for rd in bs_data]
    p_data = gv.dataset.avg_data(p_ense, bstrap=True)

    # H_n and H_sub from bootstrap ensemble
    H_ense = {}
    for mats in mats_list:
        for iflow in iflows_mats[mats]:
            H_ense[f'H_{mats}_{iflow}'] = [rd.tsums_mats[iflow,mats] for rd in bs_data]
    for sub in sub_list:
        for iflow in set.intersection(*(set(iflows_mats[mats]) for mats in sub)):
            H_ense[f'H_{sub}_{iflow}'] = [rd.tsums_subs[sub,iflow] for rd in bs_data]
    H_data = gv.dataset.avg_data(H_ense, bstrap=True)

    # save data
    with pickle_path.open('wb') as f:
        gv.dump((p_data, H_data), f)

else:
    with open(pickle_path, 'rb') as f:
        p_data, H_data = gv.load(f)





# print fitting parameters
for mats in mats_list:
    print(f'n={mats}:  m=' + str(p_data[f'm_{mats}']), end='  ')
    print(' '.join(f'a({flowtimes[iflow]:.2f})={p_data[f"a_{mats}_{iflow}"]}' for iflow in iflows_mats[mats]))
print('\n')

# print H_n and H_sub
col0_width, col_width = 8, 14
all_iflows = sorted(set.union(*(set(iflows_mats[mats]) for mats in mats_list)))
print(' ' * col0_width + ''.join(f'{flowtimes[iflow]:.2f} a^2'.ljust(col_width) for iflow in all_iflows))
print('mats:')
for mats in mats_list:
    row = f' n={mats}'.ljust(col0_width)
    for iflow in all_iflows:
        key = f'H_{mats}_{iflow}'
        cell = str(H_data[key]) if key in H_data else ''
        row += f'{cell:<{col_width}}'
    print(row)
print('subs:')
for sub in sub_list:
    row = (' '+ ''.join(map(str, sub))).ljust(col0_width)
    for iflow in all_iflows:
        key = f'H_{sub}_{iflow}'
        cell = str(H_data[key]) if key in H_data else ''
        row += f'{cell:<{col_width}}'
    print(row)

    




