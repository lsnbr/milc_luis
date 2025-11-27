

from measurements import *





nt = 16
ns = 64
Vs = ns**3

r_list   = radial_separations(ns)
dr2_list = radial_multiplicities(ns)
dr_list  = make_distance_corr_arrays(dr2_list, False)

assert Vs == sum(dr2_list)
assert Vs == sum(dr_list)
assert all(x == 1 for x in make_distance_corr_arrays(dr2_list, True))

print(dr_list)