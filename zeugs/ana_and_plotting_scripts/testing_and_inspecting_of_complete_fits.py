from import_and_plotting_tech import *


from some_intermediate_results import rmin_per_iflow_mats_exmax





ex_max = 0

fss = FitSubSum(bs=False)
fss.iflows[0] = [11,12,14,16,18] #+ [20,21]
fss.iflows[1] = [11,12,14,16,18] #+ [20,21]
fss.iflows[2] = [11,12,14]


for mats in (0,1,2):
    rmin_per_iflow = rmin_per_iflow_mats_exmax[ex_max][mats]
    fss.do_fit_one_mat_diff_rmin(mats, rmin_per_iflow, ex_max=ex_max, printfits=True)



fss.plot_mats_fits(path = Path.cwd() / 'zeugs' / 'plots' / 'mats_fitsN.png')



# fss.do_sums_for_sub(sub=(0,1))
# fss.do_sums_for_sub(sub=(0,1,2))
# fss.plot_subtraction_fits(sub=(0,1,2), path = Path.cwd() / 'zeugs' / 'plots' / 'mats_sub012.png')



fss.do_sums_for_mats(mats_vals=(0,1,2), printsums=True)
fss.do_sums_for_sub_from_mats(sub=(0,1))
fss.do_sums_for_sub_from_mats(sub=(0,1,2))

fss.plot_mats_integrand_fits(path = Path.cwd() / 'zeugs' / 'plots' / 'mats_sinh_fitsN.png')




