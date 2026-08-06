from import_and_plotting_tech import *



rmin_per_iflow_mats_exmax : list[list[dict[int,float]]] = [
    [   # ex_max = 0
        {8:10.5, 10:11, 12:11.5, 14:12, 16:12.5, 18:13, 20:14, 21:15},  # mats = 0
        {8:7.5, 10:7.5, 12:8, 14:8.5, 16:9, 18:9.5, 20:10, 21:10.5},    # mats = 1
        {8:6+1/3, 10:6+1/3, 12:6+2/3, 14:6+2/3},                        # mats = 2
    ],
    [   # ex_max = 1
        {8:6, 10:6.5, 12:6.5, 14:7, 16:7.5, 18:8, 20:9, 21:10},         # mats = 0
        {8:6.5, 10:6.5, 12:7, 14:7.5, 16:8, 18:8.5, 20:9, 21:10},       # mats = 1
        {8:6+1/3, 10:6+1/3, 12:6+2/3, 14:6+2/3},                        # mats = 2
    ]
]





ex_max = 1

fss = FitSubSum(bs=False)
fss.iflows[0] = [8,10,12,14,16,18] #+ [20,21]
fss.iflows[1] = [8,10,12,14,16,18] #+ [20,21]
fss.iflows[2] = [8,10,12,14]


for mats in (0,1,2):

    rmin_per_iflow = rmin_per_iflow_mats_exmax[ex_max][mats]

    fss.do_fit_one_mat_diff_rmin(mats, rmin_per_iflow, ex_max=ex_max, printfits=True)



fss.plot_mats_fits(path = Path.cwd() / 'zeugs' / 'plots' / 'mats_fits.png')



# fss.do_sums_for_sub(sub=(0,1))
# fss.do_sums_for_sub(sub=(0,1,2))
# fss.plot_subtraction_fits(sub=(0,1,2), path = Path.cwd() / 'zeugs' / 'plots' / 'mats_sub012.png')



fss.do_sums_for_mats(mats_vals=(0,1,2), printsums=True)
fss.do_sums_for_sub_from_mats(sub=(0,1))
fss.do_sums_for_sub_from_mats(sub=(0,1,2))

fss.plot_mats_integrand_fits(path = Path.cwd() / 'zeugs' / 'plots' / 'mats_sinh_fits.png')