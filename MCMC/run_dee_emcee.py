from functions import  *
import argparse
# from memory_profiler import mprof


#argument for testing
parser = argparse.ArgumentParser()
# parser.add_argument("--test", action='store_true', help="Test config file")
parser.add_argument("-config", type=str, help="which config file to use")
parser.add_argument("--GM", action='store_true', help="Check if running on Graymalkin")
parser.add_argument("--smf_only", action='store_true', help="only use SMF likelihood")
parser.add_argument("--ds_only", action='store_true', help="only use DS likelihood")
args = parser.parse_args()

################################################################################
#Classes
################################################################################
class Unbuffered(object):     #Unbuffered class to output stdout. https://stackoverflow.com/questions/107705/disable-output-buffering. (Other suggestions didn't work)
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)
################################################################################

if __name__ == '__main__':

    time1 = time.time()
    #log stdout and stderr in oputput file
    sys.stdout = open('MCMC/logs/run_{0}.log'.format(args.config), 'w')
    sys.stderr = open('MCMC/logs/run_{0}.log'.format(args.config), 'w')
    sys.stdout = Unbuffered(sys.stdout)
    sys.stderr = Unbuffered(sys.stderr)

    sys.stderr = sys.stdout
    # sys.stderr = Unbuffered(sys.stderr)

    #check directory is dwarf_lensing
    assert os.path.basename(os.getcwd()) == 'dwarf_lensing', 'Need to run in `dwarf_lensing` directory!'

    #test
    if args.config in ['Test', 'test', 'TEST']:
        print('TEST')

    #config file to use
    config_file = 'MCMC/mcmc_config_{0}.yaml'.format(args.config)
    print(config_file)
    config_initial = parse_config(config_file)

    #check if running on graymalkin
    if args.GM:
        print('RUNNING ON GRAYMALKIN')
        config_initial = GM_data_location(config_initial)
    else:
        print('NOT RUNNING ON GRAYMALKIN')

    config, cosmos_data, sim_data = initial_model(config_initial)

    emcee_fit(config, cosmos_data, sim_data, smf_only=args.smf_only, ds_only=args.ds_only)

    run_time = time.time() - time1
    print('Total time: {0} seconds; {1} minutes; {2} hours'.format(run_time, run_time/60., run_time/3600. ))
