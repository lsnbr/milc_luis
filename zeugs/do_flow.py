







# do flow with prescibed stoptime and stepsize
if 0:

    input_initial = gen_input_initial(ns, nt)
    # lat_initial = Path('thermalized_configs') / f'ns{ns}_nt{nt}_T1p3_1000hb.lat'
    
    out = flow_rkmk3(
        stoptime      = 0.5,
        stepsize      = 0.5,
        lat_initial   = Path('gauge_configs') / f'pg_00009.lat',
        input_initial = input_initial,
        flow          = 'wilson'
    )

    (Path('outputs') / 'flow_test.txt').write_text(out)



# flow for ensemble
if 0:

    input_initial = gen_input_initial(ns, nt)

    stoptime, stepsize = flow_params(10, nt, r_max=0.15)

    for i_config in range(100):
        print( '#################################################')
        print(f'###############  config {i_config:05}  ##################')
        print( '#################################################')

        out = flow_rkmk3(
            stoptime      = stoptime,
            stepsize      = stepsize,
            lat_initial   = Path('gauge_configs') / f'pg_{i_config:05}.lat',
            input_initial = input_initial
        )

        (Path('outputs') / f'flow_out_{i_config:05}.txt').write_text(out)

