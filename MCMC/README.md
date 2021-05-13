(Based off of Song's MCMC code from ASAP. That is why it might seem a bit more complicated than it needs to be.)

# Directories
- `config`: 
- `logs`: 
- `notebooks`: 
- `outfiles`: 
- `plots`: 

# Files
- `functions.py`: all functions used in MCMC located here.
- `run_dee_emcee.py`: script for runing MCMC (using emcee package). Config file passed as argument. Other arguments allow fitting to only SMF or only Delta Sigma. Originally written to run on Graymalkin machine.
  - example: `python MCMC/run_dee_emcee.py -config 1`
