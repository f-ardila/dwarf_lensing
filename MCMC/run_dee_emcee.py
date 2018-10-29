from functions import  *
import sys

if __name__ == '__main__':

    time1 = time.time()

    # if sys.argv[1] == 'test':
    #     config_initial = parse_config('/Users/fardila/Documents/GitHub/dwarf_lensing/MCMC/mcmc_test_config.yaml')

    config_initial = parse_config('/Users/fardila/Documents/GitHub/dwarf_lensing/MCMC/mcmc_default_config.yaml')

    config, cosmos_data, sim_data = initial_model(config_initial)

    emcee_fit(config, cosmos_data, sim_data)

    run_time = time.time() - time1
    print('Total time: {0} seconds; {1} minutes; {2} hours'.format(run_time, run_time/60., run_time/3600. ))


# global variables not defined in functions.py
