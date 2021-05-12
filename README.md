# Measuring weak lensing in dwarf galaxies
Project goal: 

## Directories
- `MCMC`: most of the work is done here. This is where MCMCs were run to fit model parameters to data.
- `notebooks`: notebooks for initial tests
- `mock_dwarfs`: catalogs of mock dwarf galaxies built by sampling existing COSMOS dwarfs to match distribution.
- `useful_halotools_scripts`: python scripts for adding local particle and halo catalogs to halotools cache.
- `Data`: too large for GitHub. data in lux: `/data/groups/leauthaud/fardila/dwarf_lensing`

## Files
- `dwarf_lensing.ipynb` : first step in the project and provides a general overview of the data, the goal, and the methods.
- `bplanck_dwarfs_matched_distribution.ipynb` : we use the subsamples of bplanck galaxies with mass distributions matched to COSMOS to measure lensing.
- `variable_scatter_demo.ipynb` : demo from Andrew Hearin showing how to use halotools to build SHMR mdoel with variable scatter
- `halotools_vs_chris.ipynb` : comparing the use of halotools vs. Chris Bradshaw's code for creating a stellar mass sample using SHMR parameters. Chris' code is faster because it's more specialized.
- `histogram_with_Asher_ddd.ipynb` : results from Asher's run of his Dark matter Deficient Doppelgangers (DDDs).

### Notes
- mostly run on graymalking and saved to my computer (Chia): run with printing in realtime (unbuffered) piping to log file and allow for disconnecting from graymalkin:
  - `nohup python -u MCMC/run_dee_emcee.py -config 4 --GM > MCMC/logs/run4.log &`
