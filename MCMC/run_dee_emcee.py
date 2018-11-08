from functions import  *
import argparse
# from memory_profiler import mprof

#argument for testing
parser = argparse.ArgumentParser()
parser.add_argument("--test", action='store_true', help="Test config file")
parser.add_argument("--GM", action='store_true', help="Check if running on Graymalkin")
args = parser.parse_args()

if __name__ == '__main__':

    time1 = time.time()

    #check directory is dwarf_lensing
    assert os.path.basename(os.getcwd()) == 'dwarf_lensing', 'Need to run in `dwarf_lensing` directory!'

    #test
    if args.test:
        print('TEST')
        config_file = 'MCMC/mcmc_config_test.yaml'

    else:
        config_file = 'MCMC/mcmc_config_3.yaml'

    print(config_file)
    config_initial = parse_config(config_file)

    #check if running on graymalkin
    if args.GM:
        print('RUNNING ON GRAYMALKIN')
        config_initial = GM_data_location(config_initial)
    else:
        print('NOT RUNNING ON GRAYMALKIN')

    config, cosmos_data, sim_data = initial_model(config_initial)

    emcee_fit(config, cosmos_data, sim_data)

    run_time = time.time() - time1
    print('Total time: {0} seconds; {1} minutes; {2} hours'.format(run_time, run_time/60., run_time/3600. ))
