




# Leftmost values of r to be used in fits
# for each flowtime, matsubara mode and number of masses used.

rmin_per_iflow_mats_exmax : list[list[dict[int,float]]] = [
    [   # ex_max = 0
        {8:10.5, 10:11, 12:11.5, 14:12, 16:12.5, 18:13, 20:14, 21:15},  # mats = 0
        {8:7.5, 10:7.5, 12:8, 14:8.5, 16:9, 18:9.5, 20:10, 21:10.5},    # mats = 1
        {8:6+1/3, 10:6+1/3, 12:6+2/3, 14:6+2/3},                        # mats = 2
    ],
    [   # ex_max = 1
        {8:6, 10:6.5, 12:6.5, 14:7, 16:7.5, 18:8, 20:9, 21:10},    # mats = 0
        {8:6.5, 10:6.5, 12:7, 14:7.5, 16:8, 18:8.5, 20:9, 21:10},  # mats = 1
        {8:6+1/3, 10:6+1/3, 12:6+2/3, 14:6+2/3},                   # mats = 2
    ]
]