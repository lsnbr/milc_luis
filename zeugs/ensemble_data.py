import numpy as np




data_16_x_64_1p5Tc = {

    # lattice dimensions
    'nt' : 16,
    'ns' : 64,

    # temperature chosen first, then beta determined using functions in 'scale_setting.py'
    'T_Tc' : 1.5,
    'beta' : 6.868,


    # random seeds used in generating configs, iseed_init for initial stream and then multiple branches (init = branch 0)
    'iseed_init' : 2314,
    'iseed_branch' : (lambda branch : 2314 + 1_000_000*branch),

    # cold start
    'lattice_initial' : 'fresh',

    # overrelaxation and quasi-heatbath sweeps per overall sweep
    'or_sweeps' : 4,
    'qhb_sweeps' : 1,


    # flowtimes determined using adaptive rkmk3 with given local_tol,
    # chosen such that relative errors due to numerical error of integrating gradient flow no more than 1e-3
    'flowtimes' : np.array([ 0.02572923590301565, 0.05303278154520656, 0.08103495224784385, 0.110208782326772,  0.141058391298962,  0.1736893835495303,
                             0.2086835114617738,  0.2454549361253429,  0.2838117228448263,  0.3249152519239425, 0.3705101434001126, 0.4179449298122797,
                             0.4696233239687699,  0.5279559816139976,  0.5890993819917063,  0.6570215111572449, 0.734680305158274,  0.8257738792221512,
                             0.9353328035912798,  1.06959076807306,    1.233820281085228,   1.435936459574629 ], dtype=float),
    'flow_type' : 'zeuthen',
    'local_tol' : 3.00e-04,

    # gradient flow integrator details
    'integrator' : 'rkmk3',
    'exp_order' : 8,

}






data_16_x_96_0p99Tc = {

    # lattice dimensions
    'nt' : 16,
    'ns' : 96,

    # temperature chosen first, then beta determined using functions in 'scale_setting.py'
    'T_Tc' : 0.99,
    'beta' : 6.542,


    # random seeds used in generating configs, iseed_init for initial stream and then multiple branches (init = branch 0)
    'iseed_init' : 3141,
    'iseed_branch' : (lambda branch : 3141 + 1_000_000*branch),

    # hot start (so we are in the low temperature phase)
    'lattice_initial' : 'warm',

    # overrelaxation and quasi-heatbath sweeps per overall sweep
    'or_sweeps' : 4,
    'qhb_sweeps' : 1,


    # flowtimes determined using adaptive rkmk3 with given local_tol, chosen such that relative errors due to numerical error of integrating gradient flow no more than 1e-3
    'flowtimes' : ...,
    'flow_type' : 'zeuthen',
    'local_tol' : ...,

    # gradient flow integrator details
    'integrator' : 'rkmk3',
    'exp_order' : 8,

}






data_16_x_64_15Tc = {

    # lattice dimensions
    'nt' : 16,
    'ns' : ...,

    # temperature chosen first, then beta determined using functions in 'scale_setting.py'
    'T_Tc' : 15,
    'beta' : ...,


    # random seeds used in generating configs, iseed_init for initial stream and then multiple branches (init = branch 0)
    'iseed_init' : 1412,
    'iseed_branch' : (lambda branch : 1412 + 1_000_000*branch),

    # cold start
    'lattice_initial' : 'fresh',

    # overrelaxation and quasi-heatbath sweeps per overall sweep
    'or_sweeps' : 4,
    'qhb_sweeps' : 1,


    # flowtimes determined using adaptive rkmk3 with given local_tol, chosen such that relative errors due to numerical error of integrating gradient flow no more than 1e-3
    'flowtimes' : ...,
    'flow_type' : 'zeuthen',
    'local_tol' : ...,

    # gradient flow integrator details
    'integrator' : 'rkmk3',
    'exp_order' : 8,

}