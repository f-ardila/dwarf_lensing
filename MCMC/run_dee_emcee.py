from functions import  *
import argparse
# from memory_profiler import mprof

#argument for testing
parser = argparse.ArgumentParser()
parser.add_argument("--test", action='store_true', help="Test config file")
args = parser.parse_args()

if __name__ == '__main__':

    time1 = time.time()

    #test
    if args.test:
        config_initial = parse_config('/Users/fardila/Documents/GitHub/dwarf_lensing/MCMC/mcmc_test_config.yaml')
        print('TEST')

    else:
        config_initial = parse_config('/Users/fardila/Documents/GitHub/dwarf_lensing/MCMC/mcmc_default_config.yaml')

    config, cosmos_data, sim_data = initial_model(config_initial)

    emcee_fit(config, cosmos_data, sim_data)

    run_time = time.time() - time1
    print('Total time: {0} seconds; {1} minutes; {2} hours'.format(run_time, run_time/60., run_time/3600. ))
