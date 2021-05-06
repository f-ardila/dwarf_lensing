# Measuring weak lensing in dwarf galaxies
Project goal: 

## Directories
- `MCMC`: most of the work is done here. This is where MCMCs were run to fit model parameters to data.
- `examples`: notebooks trying different specific issues.
- `mock_dwarfs`: catalogs of mock dwarf galaxies built by sampling existing COSMOS dwarfs to match distribution.
- `Data`: too large for GitHub. Contains data used. presently in Lux.

### Notes
- mostly run on graymalking and saved to my computer (Chia): run with printing in realtime (unbuffered) piping to log file and allow for disconnecting from graymalkin:
  - `nohup python -u MCMC/run_dee_emcee.py -config 4 --GM > MCMC/logs/run4.log &`
